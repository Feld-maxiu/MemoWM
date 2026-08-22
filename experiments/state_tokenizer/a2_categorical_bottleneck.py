"""A2 grouped categorical (PQ-8) bottleneck experiment on real Static PCA states.

A1 sends a continuous ``r_t`` from encoder to decoder; A2 discretizes it and
changes nothing else. The encoder/decoder are inherited from a formal A1
checkpoint, the codebook is initialised by per-subspace K-means on the *train*
split only, and the softmax temperature is calibrated so the posterior is neither
saturated nor uniform.

Deliberately absent: prior, KL, commitment loss, codebook loss, usage balancing,
entropy regularisation, Gumbel noise, task loss, transition, action, history,
time. The only objective is masked reconstruction MSE.

Why there is no long small-sample overfit stage here: a 4096-bit code dwarfs the
11 bits needed to index 2,000 training states, so train-side reconstruction can
be solved by memorising *which* state it is. Capacity conclusions come from the
held-out split only.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from residualmem.encoders.normalization import GroupChannelNormalizer
from residualmem.world_model.categorical_bottleneck import (
    CODEBOOK,
    EMBEDDING_TABLE,
    FORMAT_VERSION,
    GROUP_NAMES,
    PROTOCOL,
    SELECTOR_BIAS,
    SELECTOR_MIXER,
    SELECTOR_WEIGHT,
    CategoricalBottleneckConfig,
    attention_diagnostics,
    calibrate_temperature,
    code_health,
    code_histogram,
    decode,
    decoder_attention,
    embed_codes,
    encode,
    expected_param_shapes,
    initialize_params,
    learned_params_from_pq,
    masked_group_mse,
    masked_mse,
    masked_weighted_mse,
    metrics_from_sse,
    parameter_count,
    quantize,
    reconstruct,
)
from residualmem.world_model.continuous_bottleneck import (
    PROTOCOL as A1_PROTOCOL,
    expected_param_shapes as a1_expected_param_shapes,
)

from .a1_continuous_bottleneck import (
    FeatureStore,
    SplitData,
    _host_group_sse,
    _numpy,
    build_split,
    save_checkpoint,
    select_split_rows,
    select_task_balanced_rows,
)
from .common import iter_jsonl, sha256_file, write_json


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def filter_split(split: SplitData, keep: np.ndarray, name: str) -> SplitData:
    index = np.flatnonzero(np.asarray(keep, bool))
    return SplitData(
        name=name,
        rows=[split.rows[i] for i in index],
        global_indices=[split.global_indices[i] for i in index],
        tasks=[split.tasks[i] for i in index],
        x_t=split.x_t[index],
        valid=split.valid[index],
        xbar_device=split.xbar_device[jnp.asarray(index)],
        valid_device=split.valid_device[jnp.asarray(index)],
    )


def nearest_train_mse(target: SplitData, train: SplitData, chunk: int = 64) -> np.ndarray:
    """Per-state normalized MSE to the closest train state.

    The split is episode-aligned, but the dataset itself contains repeated states
    (identical opening pages across episodes of the same task), so a slice of
    validation has near-exact twins in train. A 4096-bit code can win those by
    memorisation, hence the de-duplicated secondary metric.
    """
    flat_train = train.xbar_device.reshape(len(train), -1)
    train_sq = jnp.sum(jnp.square(flat_train), -1)[None, :]
    dimensions = flat_train.shape[1]
    out = []
    for start in range(0, len(target), chunk):
        block = target.xbar_device[start:start + chunk].reshape(-1, dimensions)
        block_sq = jnp.sum(jnp.square(block), -1)[:, None]
        distance = block_sq + train_sq - 2.0 * block @ flat_train.T
        out.append(np.asarray(jnp.min(jnp.maximum(distance, 0.0), -1) / dimensions))
    return np.concatenate(out) if out else np.zeros((0,), np.float32)


# --------------------------------------------------------------------------- #
# codebook initialisation
# --------------------------------------------------------------------------- #
def _pairwise_sq(points, centers):
    return jnp.maximum(
        jnp.sum(jnp.square(points), -1)[..., :, None]
        + jnp.sum(jnp.square(centers), -1)[..., None, :]
        - 2.0 * jnp.einsum("...nd,...cd->...nc", points, centers),
        0.0,
    )


def _occupancy_stats(occupancy, *, num_clusters, problems, samples, iterations):
    occupancy = np.asarray(occupancy, np.int64)
    return {
        "clusters": int(num_clusters),
        "problems": int(problems),
        "samples": int(samples),
        "iterations": int(iterations),
        "empty_clusters": int((occupancy == 0).sum()),
        "singleton_clusters": int((occupancy == 1).sum()),
        "occupancy_min": int(occupancy.min()),
        "occupancy_p5": float(np.percentile(occupancy, 5)),
        "occupancy_median": float(np.median(occupancy)),
        "occupancy_p95": float(np.percentile(occupancy, 95)),
        "occupancy_max": int(occupancy.max()),
    }


@jax.jit
def _lloyd_step(block_points, block_centers):
    """One independent Lloyd update for every problem in a device block."""
    num_clusters = block_centers.shape[1]
    distance = _pairwise_sq(block_points, block_centers)
    assignment = jnp.argmin(distance, -1)
    onehot = jax.nn.one_hot(assignment, num_clusters, dtype=jnp.float32)
    counts = jnp.sum(onehot, 1)
    totals = jnp.einsum("bnc,bnd->bcd", onehot, block_points)
    updated = totals / jnp.maximum(counts, 1.0)[..., None]
    updated = jnp.where(counts[..., None] > 0, updated, block_centers)
    # Deterministic empty-cluster repair: hand the j-th empty cluster the
    # j-th worst-fitting point.
    residual = jnp.take_along_axis(distance, assignment[..., None], -1)[..., 0]
    order = jnp.argsort(-residual, axis=-1)
    empty = counts == 0
    rank = jnp.clip(jnp.cumsum(empty, -1) - 1, 0, block_points.shape[1] - 1)
    donor = jnp.take_along_axis(order, rank, -1)
    rescued = jnp.take_along_axis(
        block_points, donor[..., None].repeat(block_points.shape[-1], -1), 1
    )
    updated = jnp.where(empty[..., None], rescued, updated)
    return updated, counts


def kmeans_codebook(
    points,
    num_clusters,
    seed,
    iterations,
    chunk=64,
    *,
    problem_offset=0,
    return_occupancy=False,
):
    """Batched K-means++ over ``(problems, samples, dim)`` with empty repair.

    One independent problem per ``(latent token, subspace)`` pair. Average
    occupancy being low does not by itself imply empty clusters -- K-means++ plus
    repair usually keeps every centroid populated -- so occupancy is measured and
    reported rather than assumed.
    """
    points = jnp.asarray(points, jnp.float32)
    problems, samples, _ = points.shape
    if iterations <= 0:
        raise ValueError("K-means iterations must be positive")
    if num_clusters > samples:
        raise ValueError(
            f"K-means clusters ({num_clusters}) exceed samples ({samples})"
        )

    # A key is derived from the global problem id, not the current block shape.
    # Consequently a memory-only change to the problem batch preserves exactly
    # the same K-means++ initialisation for every (token, subspace) problem.
    base_key = jax.random.PRNGKey(seed)
    problem_ids = jnp.arange(
        problem_offset, problem_offset + problems, dtype=jnp.uint32
    )
    problem_keys = jax.vmap(lambda problem: jax.random.fold_in(base_key, problem))(
        problem_ids
    )

    # ---- K-means++ seeding ----
    first_keys = jax.vmap(lambda key: jax.random.fold_in(key, 0))(problem_keys)
    first = jax.vmap(lambda key: jax.random.randint(key, (), 0, samples))(first_keys)
    centers = [jnp.take_along_axis(points, first[:, None, None], 1)[:, 0]]
    closest = _pairwise_sq(points, centers[0][:, None, :])[..., 0]
    for cluster_index in range(1, num_clusters):
        logits = jnp.log(jnp.maximum(closest, 1e-30))
        choice_keys = jax.vmap(
            lambda key: jax.random.fold_in(key, cluster_index)
        )(problem_keys)
        picked = jax.vmap(
            lambda key, problem_logits: jax.random.categorical(
                key, problem_logits, axis=-1
            )
        )(choice_keys, logits)
        chosen = jnp.take_along_axis(points, picked[:, None, None], 1)[:, 0]
        centers.append(chosen)
        closest = jnp.minimum(closest, _pairwise_sq(points, chosen[:, None, :])[..., 0])
    centers = jnp.stack(centers, 1)  # (problems, clusters, dim)

    counts = None
    for _ in range(iterations):
        blocks, block_counts = [], []
        for start in range(0, problems, chunk):
            stop = min(start + chunk, problems)
            updated, count = _lloyd_step(points[start:stop], centers[start:stop])
            blocks.append(updated)
            block_counts.append(count)
        centers = jnp.concatenate(blocks, 0)
        counts = jnp.concatenate(block_counts, 0)

    occupancy = np.asarray(counts, np.int64)
    stats = _occupancy_stats(
        occupancy, num_clusters=num_clusters, problems=problems,
        samples=samples, iterations=iterations,
    )
    if return_occupancy:
        return centers, stats, occupancy
    return centers, stats


def collect_latents(params, split: SplitData, config: CategoricalBottleneckConfig,
                    batch_size: int) -> np.ndarray:
    """Encode every state once into a bounded GPU batch and a host array.

    Keeping the complete latent tensor on the accelerator costs 9.18 GB for the
    70,018-state v9 train split. More importantly, the old subsequent transpose
    required another contiguous 9.18 GB allocation. The host result remains the
    same full train tensor; this is memory scheduling, not state subsampling.
    """
    continuous = config.continuous
    latents = np.empty(
        (len(split), config.num_e_tokens, config.e_dim), dtype=np.float32
    )
    report_batches = max(1, math.ceil(max(len(split), 1) / batch_size / 10))
    for batch_index, start in enumerate(range(0, len(split), batch_size)):
        stop = min(start + batch_size, len(split))
        encoded = encode(
            params, split.xbar_device[start:stop],
            split.valid_device[start:stop], continuous,
        )
        latents[start:stop] = np.asarray(encoded, np.float32)
        if (batch_index + 1) % report_batches == 0 or stop == len(split):
            print(json.dumps({
                "event": "a2_collect_latents",
                "states": stop,
                "total_states": len(split),
            }, sort_keys=True), flush=True)
    return latents


def fit_codebook(params, train: SplitData, config: CategoricalBottleneckConfig,
                 *, seed: int, iterations: int, batch_size: int,
                 problem_batch: int = 64):
    """Per-subspace K-means centroids fitted on the train split only."""
    if train.name != "train":
        raise ValueError("codebook centroids may only be fitted on the train split")
    if problem_batch <= 0:
        raise ValueError("K-means problem batch must be positive")
    latents = collect_latents(params, train, config, batch_size)  # (n, N, e_dim)
    tokens, subspaces = config.num_e_tokens, config.num_subspaces
    latent_subspaces = latents.reshape(
        len(train), tokens, subspaces, config.subspace_dim
    )
    problems = tokens * subspaces
    centers_host = np.empty(
        (problems, config.num_categories, config.subspace_dim), np.float32
    )
    occupancy = np.empty((problems, config.num_categories), np.int64)
    for start in range(0, problems, problem_batch):
        stop = min(start + problem_batch, problems)
        problem_ids = np.arange(start, stop)
        token_ids = problem_ids // subspaces
        subspace_ids = problem_ids % subspaces
        # Only this device block is materialised in problem-major order. The
        # complete latent tensor stays host-backed and is never transposed whole.
        block_points = np.ascontiguousarray(
            np.transpose(
                latent_subspaces[:, token_ids, subspace_ids, :], (1, 0, 2)
            )
        )
        block_centers, _, block_occupancy = kmeans_codebook(
            block_points, config.num_categories, seed, iterations,
            chunk=problem_batch, problem_offset=start, return_occupancy=True,
        )
        centers_host[start:stop] = np.asarray(block_centers, np.float32)
        occupancy[start:stop] = block_occupancy
        print(json.dumps({
            "event": "a2_kmeans_block",
            "problems": stop,
            "total_problems": problems,
            "train_states": len(train),
        }, sort_keys=True), flush=True)

    stats = _occupancy_stats(
        occupancy, num_clusters=config.num_categories, problems=problems,
        samples=len(train), iterations=iterations,
    )
    stats.update({
        "problem_batch": int(problem_batch),
        "problem_rng": "per_problem_fold_in_v1",
        "state_sampling": "none_full_train_split",
        "latent_storage": "float32_host",
    })
    centers = jnp.asarray(centers_host)
    codebook = centers.reshape(tokens, subspaces, config.num_categories, config.subspace_dim)
    return codebook, stats, latents


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_split(
    params,
    split: SplitData,
    normalizer: GroupChannelNormalizer,
    config: CategoricalBottleneckConfig,
    *,
    batch_size: int,
    diagnostic_states: int,
    forward,
    soft_forward=None,
) -> dict:
    """Authoritative metrics use the hard path; the soft path is diagnostic only."""
    totals: dict[str, float] = defaultdict(float)
    task_sse: dict[str, float] = defaultdict(float)
    task_zero: dict[str, float] = defaultdict(float)
    task_scalars: dict[str, float] = defaultdict(float)
    histogram = np.zeros(
        (config.num_e_tokens, config.num_subspaces, config.num_categories), np.int64
    )
    raw_sse = 0.0
    raw_count = 0
    invalid_nonzero = 0
    soft_sse = 0.0
    soft_scalars = 0.0
    margin_sum = 0.0
    margin_count = 0
    quant_sse = 0.0
    latent_square = 0.0
    latent_count = 0
    max_prob_sum = 0.0
    count = len(split)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        index = jnp.arange(start, stop)
        prediction, codes, margin, latent, quantized, max_prob = forward(
            params, split.xbar_device, split.valid_device, index
        )
        prediction_np = np.asarray(prediction, np.float32)
        target = np.asarray(split.xbar_device[start:stop], np.float32)
        valid = split.valid[start:stop]

        batch_totals, error, zero = _host_group_sse(prediction_np, target, valid, config)
        for name, value in batch_totals.items():
            totals[name] += value
        histogram += code_histogram(codes, config)
        margin_sum += float(np.asarray(margin, np.float64).sum())
        margin_count += int(np.asarray(margin).size)
        max_prob_sum += float(np.asarray(max_prob, np.float64).sum())

        # Geometric mismatch between encoder output and quantization prototypes.
        # This is a training-health indicator, not an information-loss fraction:
        # after joint training r has no fixed meaning (the encoder is free to
        # rescale or rotate its latent basis). It is only defined under PQ, where
        # the prototype compared against is also what the decoder receives.
        latent_np = np.asarray(latent, np.float64)
        quant_sse += float(np.square(latent_np - np.asarray(quantized, np.float64)).sum())
        latent_square += float(np.square(latent_np).sum())
        latent_count += int(latent_np.size)

        recovered = normalizer.denormalize(prediction_np, valid)
        difference = recovered[valid] - split.x_t[start:stop][valid]
        raw_sse += float(np.sum(difference.astype(np.float64) ** 2))
        raw_count += int(difference.size)
        invalid_nonzero += int(np.count_nonzero(prediction_np[~valid]))

        if soft_forward is not None:
            soft_prediction = np.asarray(
                soft_forward(params, split.xbar_device, split.valid_device, index),
                np.float32,
            )
            soft_totals, _, _ = _host_group_sse(soft_prediction, target, valid, config)
            soft_sse += soft_totals["all/sse"]
            soft_scalars += soft_totals["all/valid_scalars"]

        per_state = error.sum(axis=(1, 2))
        per_state_zero = zero.sum(axis=(1, 2))
        per_state_scalars = valid.sum(axis=1) * config.input_dim
        for offset, task in enumerate(split.tasks[start:stop]):
            task_sse[task] += float(per_state[offset])
            task_zero[task] += float(per_state_zero[offset])
            task_scalars[task] += float(per_state_scalars[offset])

    metrics = metrics_from_sse(totals)
    metrics["denormalized/rmse"] = math.sqrt(raw_sse / max(raw_count, 1))
    metrics["invalid_output_nonzero"] = float(invalid_nonzero)
    metrics["states"] = float(count)
    metrics["code/top1_top2_margin_mean"] = margin_sum / max(margin_count, 1)
    # Logit-scale drift: the learned modes have no temperature holding the
    # softmax in range, so a saturating selector would silently starve the
    # straight-through gradient. Tracked every evaluation.
    metrics["code/max_prob_mean"] = max_prob_sum / max(margin_count, 1)
    if not config.is_learned:
        metrics["quant/relative_error"] = quant_sse / max(latent_square, 1e-30)
        metrics["quant/codeword_rms"] = float(
            jnp.sqrt(jnp.mean(jnp.square(params[CODEBOOK])))
        )
    metrics["quant/latent_rms"] = math.sqrt(latent_square / max(latent_count, 1))
    metrics.update(code_health(histogram, config))
    if soft_forward is not None:
        metrics["soft/mse"] = soft_sse / max(soft_scalars, 1.0)
        metrics["hard_soft_ratio"] = metrics["all/mse"] / max(metrics["soft/mse"], 1e-30)

    per_task = {}
    for task in sorted(task_sse):
        scalars = max(task_scalars[task], 1.0)
        mse = task_sse[task] / scalars
        zero = task_zero[task] / scalars
        per_task[task] = {"mse": mse, "r2": 1.0 - mse / max(zero, 1e-12)}
    metrics["task_macro_mse"] = float(np.mean([v["mse"] for v in per_task.values()]))
    metrics["task_macro_r2"] = float(np.mean([v["r2"] for v in per_task.values()]))

    diagnostic_count = min(diagnostic_states, count)
    _, encoder_weights = encode(
        params, split.xbar_device[:diagnostic_count],
        split.valid_device[:diagnostic_count], config.continuous, return_attention=True,
    )
    metrics.update(_numpy(attention_diagnostics(
        encoder_weights, decoder_attention(params, config.continuous),
        split.valid_device[:diagnostic_count], config.continuous,
    )))
    metrics["diagnostic_states"] = float(diagnostic_count)
    return {"metrics": metrics, "per_task": per_task, "histogram": histogram}


def make_forward(config: CategoricalBottleneckConfig):
    @jax.jit
    def forward(params, xbar, valid, index):
        latent = encode(params, xbar[index], valid[index], config.continuous)
        quantized, codes, diagnostics = quantize(
            params, latent, config, return_diagnostics=True
        )
        prediction = decode(params, quantized, valid[index], config.continuous)
        return (prediction, codes, diagnostics["top1_top2_margin"], latent, quantized,
                diagnostics["max_prob"])

    @jax.jit
    def soft_forward(params, xbar, valid, index):
        return reconstruct(params, xbar[index], valid[index], config, soft=True)

    return forward, soft_forward


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def _config_dict(config: CategoricalBottleneckConfig) -> dict:
    output = dataclasses.asdict(config)
    output["group_sizes"] = list(output["group_sizes"])
    return output


def _stored_config(metadata: dict) -> dict:
    """Config from a checkpoint, defaulting pre-``assignment`` runs to PQ.

    The twelve completed PQ runs were written before the field existed; without
    this they would all fail to load.
    """
    stored = dict(metadata.get("config", {}))
    stored.setdefault("assignment", "pq")
    return stored


def load_checkpoint(path: str | Path, config: CategoricalBottleneckConfig) -> dict:
    with np.load(path, allow_pickle=False) as checkpoint:
        if "metadata" not in checkpoint.files:
            raise ValueError(f"{path} has no metadata; not an A2 checkpoint")
        metadata = json.loads(str(np.asarray(checkpoint["metadata"]).item()))
        if metadata.get("protocol") != PROTOCOL:
            raise ValueError(
                f"checkpoint protocol {metadata.get('protocol')!r} != {PROTOCOL!r}"
            )
        if int(metadata.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError("unsupported A2 checkpoint format version")
        stored = _stored_config(metadata)
        if stored != _config_dict(config):
            raise ValueError(f"checkpoint config {stored} != requested {_config_dict(config)}")
        params = {
            name.replace("__", "/"): jnp.asarray(checkpoint[name])
            for name in checkpoint.files if name != "metadata"
        }
    expected = expected_param_shapes(config)
    if set(params) != set(expected):
        missing = sorted(set(expected) - set(params))
        extra = sorted(set(params) - set(expected))
        raise ValueError(f"checkpoint parameter mismatch; missing={missing} extra={extra}")
    for name, shape in expected.items():
        if tuple(params[name].shape) != shape:
            raise ValueError(f"{name} has shape {params[name].shape}, expected {shape}")
    return params


def read_checkpoint_config(path: str | Path) -> dict:
    """Config dict stored inside an A2 checkpoint (for resuming)."""
    with np.load(path, allow_pickle=False) as checkpoint:
        if "metadata" not in checkpoint.files:
            raise ValueError(f"{path} has no metadata; not an A2 checkpoint")
        metadata = json.loads(str(np.asarray(checkpoint["metadata"]).item()))
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError(
            f"checkpoint protocol {metadata.get('protocol')!r} != {PROTOCOL!r}"
        )
    return _stored_config(metadata)


def load_pq_for_equivalence(path: str | Path, config: CategoricalBottleneckConfig):
    """Load a trained PQ checkpoint and convert it to the learned parameterisation.

    Returns parameters whose step-0 behaviour reproduces the PQ model exactly:
    same codes, same decoder input. Training then starts from that model's
    quality rather than from scratch.
    """
    stored = read_checkpoint_config(path)
    if stored.get("assignment", "pq") != "pq":
        raise ValueError("--init-pq-checkpoint expects a PQ checkpoint")
    pq_config = CategoricalBottleneckConfig(
        **{**stored, "group_sizes": tuple(stored["group_sizes"])}
    )
    pq_params = load_checkpoint(path, pq_config)
    return learned_params_from_pq(pq_params, pq_config, config), pq_config


def _a1_checkpoint_metadata(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as checkpoint:
        if "metadata" not in checkpoint.files:
            raise ValueError(f"{path} has no metadata; not an A1 checkpoint")
        return json.loads(str(np.asarray(checkpoint["metadata"]).item()))


def _assert_a1_coordinates(path: str | Path, expected_pca_sha256: str | None) -> None:
    """Reject an A1 checkpoint fitted under different PCA axes.

    This is the one mismatch the geometry checks cannot see. A1 and A2 both
    operate on 64x512, so an A1 fitted under different axes passes every shape
    assertion and warm-starts happily onto coordinates its weights were never
    trained for. Nothing raises; A2 simply starts from a mis-rotated encoder and
    converges somewhere quietly worse. The two coordinate trees under
    ``outputs/state_tokenizer/`` carry identical file names, so this is a live
    hazard rather than a hypothetical one -- as of 2026-08-22 the checked-in A1
    is bound to the invalidated first-N PCA while the official normalization is
    the task-balanced one.
    """
    if not expected_pca_sha256:
        return
    found = _a1_checkpoint_metadata(path).get("pca_sha256")
    if found != expected_pca_sha256:
        raise ValueError(
            f"A1 checkpoint {path} was fitted under PCA {str(found)[:16]}... but "
            f"this run normalizes with {expected_pca_sha256[:16]}...; retrain A1 "
            "on these coordinates before warm-starting A2"
        )


def load_a1_warm_start(
    path: str | Path,
    config: CategoricalBottleneckConfig,
    expected_pca_sha256: str | None = None,
) -> dict:
    """Inherit encoder/decoder from a formal A1 checkpoint; codebook stays fresh."""
    _assert_a1_coordinates(path, expected_pca_sha256)
    with np.load(path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(np.asarray(checkpoint["metadata"]).item()))
        if metadata.get("protocol") != A1_PROTOCOL:
            raise ValueError(
                f"warm start expects {A1_PROTOCOL!r}, got {metadata.get('protocol')!r}"
            )
        shared = {
            name.replace("__", "/"): jnp.asarray(checkpoint[name])
            for name in checkpoint.files if name != "metadata"
        }
    expected = a1_expected_param_shapes(config.continuous)
    if set(shared) != set(expected):
        raise ValueError("A1 checkpoint does not match the shared A2 geometry")
    for name, shape in expected.items():
        if tuple(shared[name].shape) != shape:
            raise ValueError(f"{name} has shape {shared[name].shape}, expected {shape}")
    return shared


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def _parse_group_weights(args: argparse.Namespace):
    """``--group-weights`` as a tuple in ``GROUP_NAMES`` order, or ``None``.

    Rejected alongside ``--detail-weight``: the two express the same intent
    through different arithmetic, and silently combining them would make the
    effective detail multiplier depend on which flag the reader looked at.
    """
    raw = getattr(args, "group_weights", None)
    if not raw:
        return None
    if args.detail_weight:
        raise ValueError("--group-weights and --detail-weight are mutually exclusive")
    values = [float(piece) for piece in raw.replace(",", " ").split()]
    if len(values) != len(GROUP_NAMES):
        raise ValueError(
            f"--group-weights needs {len(GROUP_NAMES)} values in order "
            f"{GROUP_NAMES}, got {len(values)}"
        )
    if any(value < 0 for value in values):
        raise ValueError("--group-weights must be non-negative")
    return tuple(values)


def run(args: argparse.Namespace) -> dict:
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    group_weights = _parse_group_weights(args)
    records = list(iter_jsonl(args.records))
    normalizer = GroupChannelNormalizer.from_npz(args.normalization)
    if (normalizer.num_tokens, normalizer.token_dim) != (64, 512):
        raise ValueError("A2 requires a 64x512 normalizer")

    # Before the split build, which reads tens of thousands of states: warm start
    # happens long after that, and a coordinate mismatch discovered there costs
    # the whole load for nothing.
    if args.init_a1_checkpoint:
        _assert_a1_coordinates(args.init_a1_checkpoint, normalizer.pca_sha256)

    devices = jax.devices(args.platform)
    if not 0 <= args.device_index < len(devices):
        raise ValueError(f"device-index {args.device_index} unavailable for {args.platform}")
    device = devices[args.device_index]
    store = FeatureStore(args.features)

    if args.overfit_states:
        train_rows = select_task_balanced_rows(records, args.overfit_states, args.seed)
        mode = "overfit"
    else:
        train_rows = select_split_rows(records, "train")
        mode = "split"
    train = build_split("train", train_rows, records, store, normalizer, device)
    selection_name = "train" if mode == "overfit" else "validation"
    selection = train if mode == "overfit" else build_split(
        "validation", select_split_rows(records, "validation"), records, store,
        normalizer, device,
    )

    base_config = CategoricalBottleneckConfig(
        num_e_tokens=args.num_e_tokens, e_dim=args.e_dim,
        num_heads=args.num_heads, ffn_hidden=args.ffn_hidden,
        num_subspaces=args.num_subspaces, num_categories=args.num_categories,
        temperature=args.temperature, assignment=args.assignment,
    )

    sources = [args.init_a1_checkpoint, args.init_checkpoint, args.init_pq_checkpoint]
    if sum(1 for value in sources if value) > 1:
        raise ValueError(
            "pass at most one of --init-a1-checkpoint / --init-checkpoint / "
            "--init-pq-checkpoint"
        )
    if base_config.is_learned and args.init_a1_checkpoint:
        raise ValueError(
            "a learned assignment has no codebook to fit; start it from a trained "
            "PQ model with --init-pq-checkpoint so step 0 matches that model"
        )

    # ---- parameters: A2 resume, PQ-equivalence init, or A1 + K-means ----
    warm_start = None
    equivalence = None
    resumed = bool(args.init_checkpoint)
    if resumed:
        stored = read_checkpoint_config(args.init_checkpoint)
        geometry = {k: v for k, v in stored.items() if k != "temperature"}
        current = {k: v for k, v in _config_dict(base_config).items() if k != "temperature"}
        if geometry != current:
            raise ValueError(f"resume geometry {geometry} != requested {current}")
        resume_config = dataclasses.replace(
            base_config, temperature=float(stored["temperature"])
        )
        params = load_checkpoint(args.init_checkpoint, resume_config)
        params = jax.device_put(params, device)
        # Resuming must not refit the codebook or recalibrate: both would discard
        # what training has already learned.
        kmeans_stats = None
        calibration = None
        temperature = resume_config.temperature
        latents = None
    elif args.init_pq_checkpoint:
        params, pq_config = load_pq_for_equivalence(args.init_pq_checkpoint, base_config)
        params = jax.device_put(params, device)
        kmeans_stats = None
        calibration = None
        # Temperature is folded into the selector weights and bias at this point;
        # the learned modes carry no separate temperature thereafter.
        temperature = args.temperature
        latents = None
        equivalence = {
            "source": str(Path(args.init_pq_checkpoint).resolve()),
            "source_temperature": pq_config.temperature,
            "note": "W = 2B/tau, b = -||B||^2/tau, E = B; step 0 reproduces the PQ "
                    "model's codes and decoder input exactly",
        }
    else:
        params = initialize_params(base_config, args.seed)
        if args.init_a1_checkpoint:
            warm_start = load_a1_warm_start(
                args.init_a1_checkpoint, base_config, normalizer.pca_sha256
            )
            params.update(warm_start)
        params = jax.device_put(params, device)

        kmeans_stats = None
        if args.kmeans_iterations > 0:
            codebook, kmeans_stats, latents = fit_codebook(
                params, train, base_config, seed=args.seed,
                iterations=args.kmeans_iterations, batch_size=args.init_batch_size,
                problem_batch=getattr(args, "kmeans_problem_batch", 64),
            )
            params[CODEBOOK] = codebook
        else:
            latents = collect_latents(params, train, base_config, args.init_batch_size)

        calibration = None
        if args.calibrate_temperature:
            sample = latents[: args.calibration_states]
            temperature, calibration = calibrate_temperature(
                params, sample, base_config, target=args.target_max_prob,
            )
            calibration["states"] = int(sample.shape[0])
        else:
            temperature = args.temperature
    config = dataclasses.replace(base_config, temperature=temperature)
    del latents

    forward, soft_forward = make_forward(config)

    def loss_fn(current, xbar_batch, valid_batch):
        prediction = reconstruct(current, xbar_batch, valid_batch, config)
        if group_weights is not None:
            # Shares the natural loss's global denominator, so all-ones is
            # exactly masked_mse and each weight is a per-element multiplier.
            return masked_weighted_mse(
                prediction, xbar_batch, valid_batch, config, group_weights
            )
        loss = masked_mse(prediction, xbar_batch, valid_batch)
        if args.detail_weight:
            # Counted a second time, over the same global denominator, so
            # --detail-weight 1 is exactly a 2x per-element weight on detail.
            loss = loss + args.detail_weight * masked_group_mse(
                prediction, xbar_batch, valid_batch, config, "detail"
            )
        return loss

    # ---- optimizer: quantizer adapts fast, inherited backbone slowly ----
    # Freezing is expressed as a third parameter group with a zero transform, so
    # the diagnostic runs share exactly one training loop with the real ones.
    # "quantizer" covers whatever the active assignment owns: the PQ codebook, or
    # the learned selector plus its embedding table.
    QUANTIZER_KEYS = (CODEBOOK, SELECTOR_WEIGHT, SELECTOR_BIAS, SELECTOR_MIXER,
                      EMBEDDING_TABLE)

    def label_of(name: str) -> str:
        if name in QUANTIZER_KEYS:
            return "frozen" if args.freeze_codebook else "quantizer"
        if name.startswith("encoder/"):
            return "frozen" if args.freeze_encoder else "backbone"
        return "backbone"

    labels = {name: label_of(name) for name in params}
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.clip_norm),
        optax.multi_transform(
            {
                "quantizer": optax.adamw(args.codebook_learning_rate,
                                         weight_decay=args.weight_decay),
                "backbone": optax.adamw(args.backbone_learning_rate,
                                        weight_decay=args.weight_decay),
                "frozen": optax.set_to_zero(),
            },
            labels,
        ),
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def update(current, state, xbar_batch, valid_batch):
        """The batch arrives as an argument, not through the closure.

        Gathering inside the jit would make the whole train split an operand of
        the compiled graph. At v6's 7,013 states that is 0.92 GB and compiles; at
        v7's 70,023 it is 9.18 GB and XLA segfaults during compilation -- with no
        Python-level error, so the run dies silently with an empty log.
        """
        loss, grads = jax.value_and_grad(loss_fn)(current, xbar_batch, valid_batch)
        updates, state = optimizer.update(grads, state, current)
        return optax.apply_updates(current, updates), state, loss, grads


    def take_batch(index):
        """Gather outside the jit so only the batch is compiled into the graph."""
        return train.xbar_device[index], train.valid_device[index]

    batch_size = len(train) if mode == "overfit" else args.batch_size
    rng = np.random.default_rng(args.seed)
    # One batch is always built: with --max-steps 0 it still feeds the initial
    # gradient diagnostic, which is the whole point of an evaluation-only run.
    schedule_steps = max(args.max_steps, 1)
    if mode == "overfit":
        order = np.tile(np.arange(len(train)), (schedule_steps, 1))
    else:
        pool: list[int] = []
        while len(pool) < schedule_steps * batch_size:
            pool.extend(rng.permutation(len(train)).tolist())
        order = np.asarray(pool[: schedule_steps * batch_size]).reshape(
            schedule_steps, batch_size
        )

    def evaluate(current, split, *, soft=False):
        return evaluate_split(
            current, split, normalizer, config,
            batch_size=args.eval_batch_size,
            diagnostic_states=args.diagnostic_states,
            forward=forward, soft_forward=soft_forward if soft else None,
        )

    initial = evaluate(params, selection)["metrics"]
    _, initial_grads = jax.value_and_grad(loss_fn)(
        params, train.xbar_device[jnp.asarray(order[0])],
        train.valid_device[jnp.asarray(order[0])],
    )
    quantizer_keys = [name for name in QUANTIZER_KEYS if name in params]
    gradient_norms = {
        name: float(jnp.sqrt(jnp.sum(jnp.square(initial_grads[name]))))
        for name in quantizer_keys + ["encoder/e_queries", "encoder/input_positions",
                                      "decoder/e_addresses", "decoder/output_queries"]
    }
    encoder_before = {
        name: value for name, value in params.items() if name.startswith("encoder/")
    }
    quantizer_before = {name: params[name] for name in quantizer_keys}

    history: list[dict] = []
    best = {"metric": math.inf, "step": 0}
    best_params = params
    step = 0
    stop_reason = "budget_exhausted" if args.max_steps else "evaluation_only"
    evals_without_improvement = 0
    for _ in range(args.max_steps):
        index = jnp.asarray(order[step])
        xbar_batch, valid_batch = take_batch(index)
        params, opt_state, loss, _ = update(params, opt_state, xbar_batch, valid_batch)
        step += 1
        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = evaluate(params, selection)["metrics"]
            history.append({
                "step": step,
                "train_batch_loss": float(loss),
                f"{selection_name}/mse": metrics["all/mse"],
                f"{selection_name}/r2": metrics["all/r2"],
                "code/perplexity_median": metrics["code/perplexity_median"],
            })
            print(json.dumps({
                "event": "a2_eval",
                "step": step,
                "total_steps": args.max_steps,
                "train_batch_loss": float(loss),
                f"{selection_name}/mse": metrics["all/mse"],
                f"{selection_name}/r2": metrics["all/r2"],
                "code/perplexity_median": metrics["code/perplexity_median"],
            }, sort_keys=True), flush=True)
            if metrics["all/mse"] < best["metric"]:
                best = {"metric": metrics["all/mse"], "step": step}
                best_params = params
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
            if 0 < args.patience_evals <= evals_without_improvement:
                stop_reason = "validation_patience"
                break

    behaviour = {
        "gradients_finite": all(
            bool(jnp.all(jnp.isfinite(value))) for value in initial_grads.values()
        ),
        "latent_gradient_nonzero": gradient_norms["encoder/e_queries"] > 0.0,
        "codebook_gradient_nonzero": all(
            gradient_norms[name] > 0.0 for name in quantizer_keys
        ),
    }
    if step > 0:
        # A frozen group must not move; a trainable one must. Asserting both
        # directions keeps the diagnostic runs from silently training something
        # they were supposed to hold fixed.
        encoder_moved = any(
            not bool(jnp.array_equal(params[name], before))
            for name, before in encoder_before.items()
        )
        codebook_moved = any(
            not bool(jnp.array_equal(params[name], before))
            for name, before in quantizer_before.items()
        )
        if args.freeze_encoder:
            behaviour["encoder_stayed_frozen"] = not encoder_moved
        else:
            behaviour["encoder_params_moved"] = encoder_moved
        if args.freeze_codebook:
            behaviour["codebook_stayed_frozen"] = not codebook_moved
        else:
            behaviour["codebook_moved"] = codebook_moved
        behaviour["training_improved"] = best["metric"] < initial["all/mse"]
    else:
        best = {"metric": initial["all/mse"], "step": 0}
    if calibration is not None:
        behaviour["temperature_calibrated"] = bool(calibration["converged"])

    params = best_params
    splits = {"train": evaluate(params, train, soft=True)}
    if mode != "overfit":
        splits[selection_name] = evaluate(params, selection, soft=True)
        distance = nearest_train_mse(selection, train)
        keep = distance >= args.dedup_threshold
        splits["validation_dedup"] = evaluate(
            params, filter_split(selection, keep, "validation_dedup"), soft=True
        )
        dedup_info = {
            "threshold": args.dedup_threshold,
            "removed": int((~keep).sum()),
            "kept": int(keep.sum()),
            "nearest_train_mse_median": float(np.median(distance)),
        }
    else:
        splits[selection_name] = splits["train"]
        dedup_info = None
    if args.evaluate_test:
        test = build_split("test", select_split_rows(records, "test"), records, store,
                           normalizer, device)
        splits["test"] = evaluate(params, test, soft=True)

    selection_metrics = splits[selection_name]["metrics"]
    gates = {
        "metrics_finite": all(
            math.isfinite(v) for v in selection_metrics.values() if isinstance(v, float)
        ),
        "encoder_invalid_attention_zero":
            selection_metrics["encoder/invalid_weight_sum"] == 0.0,
        "encoder_row_sum_exact": selection_metrics["encoder/row_sum_max_error"] < 1e-6,
        "decoder_row_sum_exact": selection_metrics["decoder/row_sum_max_error"] < 1e-6,
        "invalid_output_zero": selection_metrics["invalid_output_nonzero"] == 0.0,
        **behaviour,
    }
    if args.gate_profile == "discrete":
        gates["all_r2"] = selection_metrics["all/r2"] >= args.r2_gate
        for group in GROUP_NAMES:
            gates[f"group_{group}"] = selection_metrics[f"{group}/r2"] >= args.group_r2_gate
    gates["passed"] = all(gates.values())

    baseline = None
    if args.a1_reference:
        reference = json.loads(Path(args.a1_reference).read_text())
        a1_metrics = reference["splits"].get(selection_name, {})
        if a1_metrics:
            baseline = {
                "a1_result": str(Path(args.a1_reference).resolve()),
                "a1_selection_mse": a1_metrics["all/mse"],
                "delta_disc": selection_metrics["all/mse"] - a1_metrics["all/mse"],
                "rho_disc": selection_metrics["all/mse"] / max(a1_metrics["all/mse"], 1e-30),
                "rho_disc_per_group": {
                    group: selection_metrics[f"{group}/mse"]
                    / max(a1_metrics[f"{group}/mse"], 1e-30)
                    for group in GROUP_NAMES
                },
                "a1_denormalized_rmse": a1_metrics.get("denormalized/rmse"),
            }

    config_dict = _config_dict(config)
    result = {
        "protocol": PROTOCOL,
        "format_version": FORMAT_VERSION,
        "label": args.label,
        "mode": mode,
        "seed": args.seed,
        "device": str(device),
        "numerics": {
            "matmul_precision": args.matmul_precision,
            "activations": "float32",
            "metric_accumulation": "float64_host",
            "eval_batch_size": args.eval_batch_size,
            "protocol": "worklog_section_0",
        },
        "config": config_dict,
        "wiring": {
            "quantizer": "product_quantization",
            "assignment": config.assignment,
            "codebook_role": (
                "prototype_and_embedding" if not config.is_learned
                else "embedding_only (selector is separate)"
            ),
            "logits": (
                "negative_squared_distance_over_temperature" if not config.is_learned
                else f"linear_classifier_over_{'full_token' if config.assignment == 'learned_full' else 'own_slice'}"
            ),
            "estimator": "straight_through_argmax",
            "forward": "hard_gather_only",
            "rng": "none",
            "soft_path": "diagnostic_only",
            "losses": (
                ["masked_reconstruction_mse"] if not args.detail_weight else
                ["masked_reconstruction_mse",
                 f"detail_group_mse_global_denominator x {args.detail_weight}"]
            ) if not group_weights else
            ["per_group_weighted_mse_global_denominator "
             + " ".join(f"{n}={w:g}" for n, w in zip(GROUP_NAMES, group_weights))],
            "detail_weight": args.detail_weight,
            "group_weights": (
                dict(zip(GROUP_NAMES, group_weights)) if group_weights else None
            ),
        },
        "capacity": {
            "codes_per_state": config.codes_per_state,
            "categories": config.num_categories,
            "code_bits_per_state": config.code_bits,
            "mask_bits_per_state": config.num_input_slots,
            "total_bits_per_state": config.code_bits + config.num_input_slots,
            "latent_tensor_bits": config.latent_tensor_bits,
            "tensor_reduction": config.tensor_reduction,
            "parameters": parameter_count(params),
            "note": "nominal size reduction against the raw FP32 latent tensor; NOT a "
                    "measured bitrate (nothing is entropy coded). The 64 mask bits are "
                    "external side information.",
        },
        "data": {
            "records": str(Path(args.records).resolve()),
            "records_sha256": sha256_file(args.records),
            "feature_root": str(Path(args.features).resolve()),
            "normalization_artifact": str(Path(args.normalization).resolve()),
            "normalization_hash": normalizer.hash_bytes.hex(),
            "pca_sha256": normalizer.pca_sha256,
            "train_states": len(train),
            "selection_split": selection_name,
            "selection_states": len(selection),
            "evaluate_test": bool(args.evaluate_test),
            "dedup": dedup_info,
            "scope": "same 12 tasks, new episodes of the same templates; not "
                     "cross-task generalization",
        },
        "init": {
            "a1_checkpoint": (
                str(Path(args.init_a1_checkpoint).resolve())
                if args.init_a1_checkpoint else None
            ),
            "resumed_from": (
                str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None
            ),
            "pq_equivalence": equivalence,
            "warm_started_keys": sorted(warm_start) if warm_start else [],
            "kmeans": kmeans_stats,
            "temperature_calibration": calibration,
            "temperature": temperature,
            "initial_gradient_norms": gradient_norms,
            "initial_selection_mse": initial["all/mse"],
        },
        "optimizer": {
            "codebook_learning_rate": args.codebook_learning_rate,
            "backbone_learning_rate": args.backbone_learning_rate,
            "freeze_encoder": bool(args.freeze_encoder),
            "freeze_codebook": bool(args.freeze_codebook),
            "weight_decay": args.weight_decay,
            "clip_norm": args.clip_norm,
            "batch_size": batch_size,
            "max_steps": args.max_steps,
            "steps_ran": step,
            "eval_every": args.eval_every,
            "patience_evals": args.patience_evals,
            "stop_reason": stop_reason,
        },
        "best_selection_mse": best["metric"],
        "best_selection_step": best["step"],
        "history": history,
        "splits": {name: value["metrics"] for name, value in splits.items()},
        "per_task": {name: value["per_task"] for name, value in splits.items()},
        "baseline": baseline,
        "gate_profile": args.gate_profile,
        "gates": gates,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    save_checkpoint(output.with_suffix(".npz"), params, {
        "protocol": PROTOCOL,
        "format_version": FORMAT_VERSION,
        "config": config_dict,
        "wiring": result["wiring"],
        "normalization_hash": result["data"]["normalization_hash"],
        "pca_sha256": result["data"]["pca_sha256"],
        "records_sha256": result["data"]["records_sha256"],
        "note": "parameters only; optimizer moments are not stored",
    })
    if args.codes_npz:
        codes = []
        for start in range(0, len(selection), args.eval_batch_size):
            index = jnp.arange(start, min(start + args.eval_batch_size, len(selection)))
            codes.append(np.asarray(
                forward(params, selection.xbar_device, selection.valid_device, index)[1],
                np.int32,
            ))
        np.savez_compressed(
            args.codes_npz,
            codes=np.concatenate(codes, 0),
            global_indices=np.asarray(selection.global_indices, np.int64),
            valid=np.asarray(selection.valid, np.bool_),
            split=np.asarray(selection.name),
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="a2")
    parser.add_argument("--init-a1-checkpoint")
    parser.add_argument("--init-checkpoint",
                        help="resume from an A2 checkpoint; keeps the trained codebook "
                             "and stored temperature (no refit, no recalibration)")
    parser.add_argument("--init-pq-checkpoint",
                        help="start a learned assignment from a trained PQ checkpoint "
                             "via the exact-equivalence transform, so step 0 reproduces "
                             "that model's codes and reconstruction")
    parser.add_argument("--assignment",
                        choices=("pq", "learned_slice", "learned_mix", "learned_full"),
                        default="pq",
                        help="pq: nearest prototype (geometry-driven). learned_*: a free "
                             "linear classifier decoupled from the decoder embedding, "
                             "reading its own slice, a shared learned rotation of the "
                             "token (learned_mix), or the whole token per category")
    parser.add_argument("--a1-reference", help="A1 result JSON for delta/rho reporting")
    parser.add_argument("--num-e-tokens", type=int, default=64)
    parser.add_argument("--e-dim", type=int, default=512)
    parser.add_argument("--num-subspaces", type=int, default=8)
    parser.add_argument("--num-categories", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-hidden", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--calibrate-temperature", action="store_true", default=True)
    parser.add_argument("--no-calibrate-temperature", dest="calibrate_temperature",
                        action="store_false")
    parser.add_argument("--target-max-prob", type=float, default=0.8)
    parser.add_argument("--calibration-states", type=int, default=256)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--kmeans-problem-batch", type=int, default=64,
                        help="number of independent (token, subspace) K-means "
                             "problems resident on the accelerator at once; all "
                             "train states still participate")
    parser.add_argument("--overfit-states", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=70000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=250)
    parser.add_argument("--init-batch-size", type=int, default=250,
                        help="batching used to collect latents for K-means and "
                             "temperature calibration; kept separate from "
                             "--eval-batch-size so initialisation does not silently "
                             "depend on how evaluation is chunked")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience-evals", type=int, default=20)
    parser.add_argument("--codebook-learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="hold the A1 encoder fixed; with --freeze-codebook the "
                             "codes become fixed, making the run a pure capacity probe")
    parser.add_argument("--freeze-codebook", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--group-weights",
                        help="per-group weights in GROUP_NAMES order "
                             "(image,detail,context,prompt), e.g. '1,2,0.5,0.5'. "
                             "Shares the natural loss's global denominator, so "
                             "'1,1,1,1' is exactly the plain MSE and each weight "
                             "is a per-element multiplier. Excludes --detail-weight")
    parser.add_argument("--detail-weight", type=float, default=0.0,
                        help="extra weight on the DOM-detail group, added over the "
                             "global valid-element denominator, so 1.0 means detail "
                             "counts twice (2x per element). 0 reproduces the plain "
                             "natural MSE bit for bit")
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--diagnostic-states", type=int, default=128)
    parser.add_argument("--dedup-threshold", type=float, default=0.01)
    parser.add_argument("--codes-npz")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--gate-profile", choices=("report_only", "discrete"),
                        default="report_only")
    parser.add_argument("--r2-gate", type=float, default=0.90)
    parser.add_argument("--group-r2-gate", type=float, default=0.80)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--matmul-precision",
                        choices=("highest", "float32", "tensorfloat32", "bfloat16"),
                        default="highest")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    selection = result["data"]["selection_split"]
    summary = {
        "output": str(Path(args.output).resolve()),
        "label": result["label"],
        "mode": result["mode"],
        "assignment": result["config"]["assignment"],
        "code_bits_per_state": result["capacity"]["code_bits_per_state"],
        "tensor_reduction": result["capacity"]["tensor_reduction"],
        "parameters": result["capacity"]["parameters"],
        "temperature": result["init"]["temperature"],
        "steps_ran": result["optimizer"]["steps_ran"],
        "stop_reason": result["optimizer"]["stop_reason"],
        f"{selection}/mse": result["splits"][selection]["all/mse"],
        f"{selection}/r2": result["splits"][selection]["all/r2"],
        "code/perplexity_median": result["splits"][selection]["code/perplexity_median"],
        "code/max_prob_mean": result["splits"][selection]["code/max_prob_mean"],
        "gates": result["gates"],
    }
    if result["baseline"]:
        summary["rho_disc"] = result["baseline"]["rho_disc"]
    if "validation_dedup" in result["splits"]:
        summary["dedup/r2"] = result["splits"]["validation_dedup"]["all/r2"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
