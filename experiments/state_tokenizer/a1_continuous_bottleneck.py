"""A1 continuous latent bottleneck experiment on real Static PCA states.

Adds exactly one element on top of A0: a deterministic continuous bottleneck
``xbar_t -> e_t (N x d_e) -> xbar_hat_t``. No categorical latent, no prior, no
KL, no transition, no action, no temporal sequence -- every row is an
independent state.

Two run modes:

``--overfit-states K``
    Full-batch memorisation of K task-balanced train states. This is the A1-Wide
    structural control (run it first at ``N=64, d_e=512``) and the small-sample
    capacity curve.

split mode (default)
    2,000 train / 500 validation with deterministic mini-batches, periodic
    validation and best-validation parameter selection. The 1,000-state test
    split is only touched with an explicit ``--evaluate-test``.

The A0 runner, model and checkpoints are untouched; this file mirrors A0's
verified data semantics rather than importing its CLI or training loop.
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
import ml_dtypes
import numpy as np
import optax

from residualmem.encoders.normalization import GroupChannelNormalizer
from residualmem.world_model.continuous_bottleneck import (
    GROUP_NAMES,
    PROTOCOL,
    ContinuousBottleneckConfig,
    attention_diagnostics,
    decoder_attention,
    encode,
    expected_param_shapes,
    initialize_params,
    masked_mse,
    metrics_from_sse,
    parameter_count,
    reconstruct,
)

from .common import iter_jsonl, sha256_file, write_json
from .slot_layout import KEY64_LAYOUT

FORMAT_VERSION = 1
# A0 slot-key reference on the same 64 train states (STATE_TOKENIZER_WORKLOG.md).
A0_SLOTKEY_MSE = 8.27e-5
A0_SLOTKEY_R2 = 0.999918
A0_SLOTKEY_RAW_RMSE = 0.00751


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def select_split_rows(records: list[dict], split: str) -> list[int]:
    rows = [row for row, record in enumerate(records) if record["split"] == split]
    if not rows:
        raise ValueError(f"split {split!r} is empty")
    return rows


def select_task_balanced_rows(
    records: list[dict], num_states: int, seed: int, split: str = "train"
) -> list[int]:
    """Deterministic task round-robin selection (same rule as the A0 runner)."""
    if num_states < 1:
        raise ValueError("num-states must be positive")
    rng = np.random.default_rng(seed)
    by_task = defaultdict(list)
    for row, record in enumerate(records):
        if record["split"] == split:
            by_task[record["task"]].append(row)
    for rows in by_task.values():
        rng.shuffle(rows)
    selected: list[int] = []
    tasks = sorted(by_task)
    cursor = {task: 0 for task in tasks}
    while len(selected) < num_states:
        progressed = False
        for task in tasks:
            if cursor[task] < len(by_task[task]):
                selected.append(by_task[task][cursor[task]])
                cursor[task] += 1
                progressed = True
                if len(selected) == num_states:
                    break
        if not progressed:
            raise ValueError(f"requested {num_states} states but {split} split is exhausted")
    return selected


class FeatureStore:
    """Memory-mapped Static PCA shards; BF16 bits -> FP32 exactly as in A0."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.locations: dict[int, tuple[Path, int]] = {}
        workers = sorted(self.root.glob("worker*"))
        if not workers:
            raise ValueError(f"no feature shards under {self.root}")
        for worker in workers:
            done = np.load(worker / "done.npy", mmap_mode="r")
            pca_done = np.load(worker / "key64-static-pca-done.npy", mmap_mode="r")
            if not bool(np.asarray(done).all()) or not bool(np.asarray(pca_done).all()):
                raise ValueError(f"incomplete final feature shard: {worker}")
            for local, index in enumerate(np.load(worker / "record_indices.npy")):
                self.locations[int(index)] = (worker, local)
        self._arrays: dict[Path, tuple[np.ndarray, np.ndarray]] = {}

    def _shard(self, worker: Path):
        if worker not in self._arrays:
            self._arrays[worker] = (
                np.load(worker / "key64-static-pca-bf16.npy", mmap_mode="r"),
                np.load(worker / "key64-static-valid.npy", mmap_mode="r"),
            )
        return self._arrays[worker]

    def load(self, global_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        missing = [index for index in global_indices if index not in self.locations]
        if missing:
            raise KeyError(f"feature store misses global indices: {missing[:20]}")
        x_rows, masks = [], []
        for index in global_indices:
            worker, local = self.locations[index]
            bits, valid = self._shard(worker)
            x_rows.append(
                np.asarray(bits[local], np.uint16)
                .copy()
                .view(ml_dtypes.bfloat16)
                .astype(np.float32)
            )
            masks.append(np.asarray(valid[local], np.bool_).copy())
        return np.stack(x_rows), np.stack(masks)


@dataclasses.dataclass
class SplitData:
    name: str
    rows: list[int]
    global_indices: list[int]
    tasks: list[str]
    x_t: np.ndarray
    valid: np.ndarray
    xbar_device: jax.Array
    valid_device: jax.Array

    def __len__(self) -> int:
        return len(self.rows)


def build_split(
    name: str,
    rows: list[int],
    records: list[dict],
    store: FeatureStore,
    normalizer: GroupChannelNormalizer,
    device,
) -> SplitData:
    global_indices = [int(records[row]["global_index"]) for row in rows]
    x_t, valid = store.load(global_indices)
    # xbar_t is generated online in FP32 and never persisted.
    xbar = normalizer.normalize(x_t, valid)
    if not np.isfinite(xbar).all() or np.count_nonzero(xbar[~valid]):
        raise ValueError(f"{name}: normalization produced invalid or nonzero padding")
    return SplitData(
        name=name,
        rows=rows,
        global_indices=global_indices,
        tasks=[str(records[row]["task"]) for row in rows],
        x_t=x_t,
        valid=valid,
        xbar_device=jax.device_put(jnp.asarray(xbar), device),
        valid_device=jax.device_put(jnp.asarray(valid), device),
    )


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def _numpy(values: dict) -> dict[str, float]:
    return {name: float(np.asarray(value)) for name, value in values.items()}


def _host_group_sse(prediction, target, valid, config: ContinuousBottleneckConfig):
    """Per-batch SSE accumulated on the host in FP64 (numerics protocol §0).

    The FP32 device-side reduction in ``group_sse`` is accurate enough for a
    training loss, but reported errors sit at ~1e-5 where FP32 accumulation and
    batch-splitting order are worth ~1% -- the same order as the capacity
    differences A1 needs to resolve.
    """
    error = np.square(prediction.astype(np.float64) - target.astype(np.float64))
    zero = np.square(target.astype(np.float64))
    mask = valid[..., None]
    totals = {}

    def accumulate(name, sliced_error, sliced_zero, sliced_valid):
        totals[f"{name}/sse"] = float(sliced_error.sum())
        totals[f"{name}/zero_sse"] = float(sliced_zero.sum())
        count = float(sliced_valid.sum())
        totals[f"{name}/valid_slots"] = count
        totals[f"{name}/valid_scalars"] = count * prediction.shape[-1]

    accumulate("all", error * mask, zero * mask, valid)
    start = 0
    for name, size in zip(GROUP_NAMES, config.group_sizes):
        stop = start + size
        accumulate(
            name,
            error[..., start:stop, :] * mask[..., start:stop, :],
            zero[..., start:stop, :] * mask[..., start:stop, :],
            valid[..., start:stop],
        )
        start = stop
    return totals, error * mask, zero * mask


def evaluate_split(
    params,
    split: SplitData,
    normalizer: GroupChannelNormalizer,
    config: ContinuousBottleneckConfig,
    *,
    batch_size: int,
    diagnostic_states: int,
    forward,
) -> dict:
    """Aggregate SSE across batches, then divide once (never average MSEs)."""
    totals: dict[str, float] = defaultdict(float)
    task_sse: dict[str, float] = defaultdict(float)
    task_zero: dict[str, float] = defaultdict(float)
    task_scalars: dict[str, float] = defaultdict(float)
    raw_sse = 0.0
    raw_count = 0
    invalid_nonzero = 0
    e_sum = 0.0
    e_square = 0.0
    e_count = 0
    e_max = 0.0
    count = len(split)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        index = jnp.arange(start, stop)
        prediction, e_t = forward(params, split.xbar_device, split.valid_device, index)
        prediction_np = np.asarray(prediction, np.float32)
        target = np.asarray(split.xbar_device[start:stop], np.float32)
        valid = split.valid[start:stop]

        batch_totals, error, zero = _host_group_sse(prediction_np, target, valid, config)
        for name, value in batch_totals.items():
            totals[name] += value

        recovered = normalizer.denormalize(prediction_np, valid)
        difference = recovered[valid] - split.x_t[start:stop][valid]
        raw_sse += float(np.sum(difference.astype(np.float64) ** 2))
        raw_count += int(difference.size)
        invalid_nonzero += int(np.count_nonzero(prediction_np[~valid]))

        e_np = np.asarray(e_t, np.float32)
        e_sum += float(e_np.astype(np.float64).sum())
        e_square += float(np.square(e_np.astype(np.float64)).sum())
        e_count += int(e_np.size)
        e_max = max(e_max, float(np.abs(e_np).max()))

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
    mean = e_sum / max(e_count, 1)
    metrics["e_t/mean"] = mean
    metrics["e_t/rms"] = math.sqrt(e_square / max(e_count, 1))
    metrics["e_t/std"] = math.sqrt(max(e_square / max(e_count, 1) - mean * mean, 0.0))
    metrics["e_t/max_abs"] = e_max
    metrics["states"] = float(count)

    per_task = {}
    for task in sorted(task_sse):
        scalars = max(task_scalars[task], 1.0)
        mse = task_sse[task] / scalars
        zero = task_zero[task] / scalars
        per_task[task] = {
            "mse": mse,
            "r2": 1.0 - mse / max(zero, 1e-12),
        }
    metrics["task_macro_mse"] = float(np.mean([v["mse"] for v in per_task.values()]))
    metrics["task_macro_r2"] = float(np.mean([v["r2"] for v in per_task.values()]))

    diagnostic_count = min(diagnostic_states, count)
    _, encoder_weights = encode(
        params,
        split.xbar_device[:diagnostic_count],
        split.valid_device[:diagnostic_count],
        config,
        return_attention=True,
    )
    decoder_weights = decoder_attention(params, config)
    metrics.update(
        _numpy(
            attention_diagnostics(
                encoder_weights,
                decoder_weights,
                split.valid_device[:diagnostic_count],
                config,
            )
        )
    )
    metrics["diagnostic_states"] = float(diagnostic_count)
    return {"metrics": metrics, "per_task": per_task}


def make_forward(config: ContinuousBottleneckConfig):
    @jax.jit
    def forward(params, xbar, valid, index):
        return reconstruct(
            params, xbar[index], valid[index], config, return_latent=True
        )

    return forward


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def save_checkpoint(path: Path, params, metadata: dict) -> None:
    payload = {name.replace("/", "__"): np.asarray(value) for name, value in params.items()}
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)


def load_checkpoint(path: str | Path, config: ContinuousBottleneckConfig) -> dict:
    """Strict load: A0 checkpoints and mismatched A1 capacities are rejected."""
    with np.load(path, allow_pickle=False) as checkpoint:
        if "metadata" not in checkpoint.files:
            raise ValueError(f"{path} has no metadata; not an A1 checkpoint")
        metadata = json.loads(str(np.asarray(checkpoint["metadata"]).item()))
        if metadata.get("protocol") != PROTOCOL:
            raise ValueError(
                f"checkpoint protocol {metadata.get('protocol')!r} != {PROTOCOL!r}"
            )
        if int(metadata.get("format_version", -1)) != FORMAT_VERSION:
            raise ValueError("unsupported A1 checkpoint format version")
        stored = metadata.get("config", {})
        current = dataclasses.asdict(config)
        current["group_sizes"] = list(current["group_sizes"])
        stored = {key: (list(value) if isinstance(value, list) else value)
                  for key, value in stored.items()}
        if stored != current:
            raise ValueError(f"checkpoint config {stored} != requested {current}")
        params = {
            name.replace("__", "/"): jnp.asarray(checkpoint[name])
            for name in checkpoint.files
            if name != "metadata"
        }
    expected = expected_param_shapes(config)
    if set(params) != set(expected):
        missing = sorted(set(expected) - set(params))
        extra = sorted(set(params) - set(expected))
        raise ValueError(f"checkpoint parameter mismatch; missing={missing} extra={extra}")
    for name, shape in expected.items():
        if tuple(params[name].shape) != shape:
            raise ValueError(
                f"checkpoint parameter {name} has shape {params[name].shape}, expected {shape}"
            )
    return params


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def integrity_gates(final_metrics: dict, initial_selection_mse: float, best_metric: float) -> dict:
    finite = all(
        math.isfinite(value)
        for value in final_metrics.values()
        if isinstance(value, float)
    )
    return {
        "metrics_finite": bool(
            finite and math.isfinite(initial_selection_mse) and math.isfinite(best_metric)
        ),
        "encoder_invalid_attention_zero": final_metrics["encoder/invalid_weight_sum"] == 0.0,
        "encoder_row_sum_exact": final_metrics["encoder/row_sum_max_error"] < 1e-6,
        "decoder_row_sum_exact": final_metrics["decoder/row_sum_max_error"] < 1e-6,
        "invalid_output_zero": final_metrics["invalid_output_nonzero"] == 0.0,
        "training_improved": best_metric < initial_selection_mse,
    }


def quality_gates(args, selection_metrics: dict, train_metrics: dict) -> dict:
    """Gate set chosen by ``--gate-profile``; thresholds are frozen up front."""
    if args.gate_profile == "structural":
        # A1-Wide: no scalar compression, so A0's near-lossless bar applies, plus
        # a pre-registered tolerance relative to the A0 slot-key baseline.
        groups = {
            f"group_{group}": (
                selection_metrics[f"{group}/mse"] < args.group_mse_gate
                and selection_metrics[f"{group}/r2"] > args.group_r2_gate
            )
            for group in GROUP_NAMES
        }
        return {
            "all_mse": selection_metrics["all/mse"] < args.total_mse_gate,
            "all_r2": selection_metrics["all/r2"] > args.total_r2_gate,
            "denormalized_rmse": selection_metrics["denormalized/rmse"] < args.raw_rmse_gate,
            "baseline_ratio": (
                selection_metrics["all/mse"]
                <= args.structural_mse_ratio * args.baseline_mse
            ),
            **groups,
        }
    if args.gate_profile == "compressed":
        groups = {
            f"group_{group}": selection_metrics[f"{group}/r2"] >= args.compressed_group_r2_gate
            for group in GROUP_NAMES
        }
        return {
            "all_r2": selection_metrics["all/r2"] >= args.compressed_r2_gate,
            "generalization_gap": (
                train_metrics["all/r2"] - selection_metrics["all/r2"]
            ) <= args.max_r2_gap,
            **groups,
        }
    return {}  # report_only: capacity/distortion facts, no pass/fail verdict


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> dict:
    if len(args.stage_steps) != len(args.stage_learning_rates):
        raise ValueError("stage-steps and stage-learning-rates must have equal length")
    # Numerics protocol (STATE_TOKENIZER_WORKLOG.md §0): TF32 costs A1 ~20% of its
    # reconstruction error and makes results batch-order sensitive at the ~1%
    # level, which is the same order as the capacity effects under test.
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    records = list(iter_jsonl(args.records))
    normalizer = GroupChannelNormalizer.from_npz(args.normalization)
    if (normalizer.num_tokens, normalizer.token_dim) != (64, 512):
        raise ValueError("A1 requires a 64x512 normalizer")

    config = ContinuousBottleneckConfig(
        num_input_slots=64,
        input_dim=512,
        num_e_tokens=args.num_e_tokens,
        e_dim=args.e_dim,
        num_heads=args.num_heads,
        ffn_hidden=args.ffn_hidden,
        # Derived: a literal here silently mis-slices every group once the layout
        # changes, and the loss is computed per group.
        group_sizes=tuple(KEY64_LAYOUT),
    )
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
    selection = (
        train
        if mode == "overfit"
        else build_split(
            "validation", select_split_rows(records, "validation"), records, store,
            normalizer, device,
        )
    )

    params = (
        load_checkpoint(args.init_checkpoint, config)
        if args.init_checkpoint
        else initialize_params(config, args.seed)
    )
    params = jax.device_put(params, device)
    forward = make_forward(config)

    def loss_fn(current, xbar_batch, valid_batch):
        prediction = reconstruct(current, xbar_batch, valid_batch, config)
        return masked_mse(prediction, xbar_batch, valid_batch)

    @jax.jit
    def loss_and_grad(current, xbar_batch, valid_batch):
        """The batch arrives as an argument, not through the closure.

        Gathering inside the jit would make the whole train split an operand of
        the compiled graph. At v6's 7,013 states that is 0.92 GB and compiles; at
        70,018 it is 9.18 GB and XLA segfaults during compilation -- with no
        Python-level error, so the run dies with a log holding only a fault
        handler dump. This is the same defect that was fixed in a2 earlier.
        """
        return jax.value_and_grad(loss_fn)(current, xbar_batch, valid_batch)

    def take_batch(index):
        """Gather outside the jit so only the batch is compiled into the graph."""
        return train.xbar_device[index], train.valid_device[index]

    batch_size = len(train) if mode == "overfit" else args.batch_size
    total_steps = int(sum(args.stage_steps))
    rng = np.random.default_rng(args.seed)
    if mode == "overfit":
        order = np.tile(np.arange(len(train)), (total_steps, 1))
    else:
        pool: list[int] = []
        while len(pool) < total_steps * batch_size:
            pool.extend(rng.permutation(len(train)).tolist())
        order = np.asarray(pool[: total_steps * batch_size]).reshape(total_steps, batch_size)

    initial_loss, initial_grads = loss_and_grad(params, *take_batch(jnp.asarray(order[0])))
    initial_loss = float(initial_loss)
    gradient_norms = {
        name: float(jnp.sqrt(jnp.sum(jnp.square(initial_grads[name]))))
        for name in (
            "encoder/e_queries",
            "encoder/input_positions",
            "decoder/e_addresses",
            "decoder/output_queries",
        )
    }
    finite_gradients = all(
        bool(jnp.all(jnp.isfinite(value))) for value in initial_grads.values()
    )
    initial_selection_mse = evaluate_split(
        params, selection, normalizer, config,
        batch_size=args.eval_batch_size,
        diagnostic_states=args.diagnostic_states,
        forward=forward,
    )["metrics"]["all/mse"]

    history: list[dict] = []
    best = {"loss": math.inf, "metric": math.inf, "step": 0}
    best_params = params
    step = 0
    stopped_early = False
    stop_reason = "budget_exhausted"
    evals_without_improvement = 0
    for stage, (stage_steps, learning_rate) in enumerate(
        zip(args.stage_steps, args.stage_learning_rates)
    ):
        # Fresh optimizer per stage: matches A0's repeated-invocation semantics,
        # where each stage started from a clean AdamW state.
        optimizer = optax.chain(
            optax.clip_by_global_norm(args.clip_norm),
            optax.adamw(learning_rate, weight_decay=args.weight_decay),
        )
        opt_state = optimizer.init(params)

        @jax.jit
        def update(current, state, xbar_batch, valid_batch):
            """Batch by argument, not by closure -- see loss_and_grad above."""
            loss, grads = jax.value_and_grad(loss_fn)(current, xbar_batch, valid_batch)
            updates, state = optimizer.update(grads, state, current)
            return optax.apply_updates(current, updates), state, loss

        for _ in range(int(stage_steps)):
            index = jnp.asarray(order[step])
            params, opt_state, loss = update(params, opt_state, *take_batch(index))
            loss = float(loss)
            step += 1
            best["loss"] = min(best["loss"], loss)
            if step % args.eval_every == 0 or step == total_steps:
                selection_metrics = evaluate_split(
                    params, selection, normalizer, config,
                    batch_size=args.eval_batch_size,
                    diagnostic_states=args.diagnostic_states,
                    forward=forward,
                )["metrics"]
                history.append({
                    "step": step,
                    "stage": stage,
                    "learning_rate": learning_rate,
                    "train_batch_loss": loss,
                    f"{selection_name}/mse": selection_metrics["all/mse"],
                    f"{selection_name}/r2": selection_metrics["all/r2"],
                })
                print(json.dumps({
                    "event": "a1_eval",
                    "step": step,
                    "total_steps": total_steps,
                    "stage": stage,
                    "learning_rate": learning_rate,
                    "train_batch_loss": loss,
                    f"{selection_name}/mse": selection_metrics["all/mse"],
                    f"{selection_name}/r2": selection_metrics["all/r2"],
                }, sort_keys=True), flush=True)
                if selection_metrics["all/mse"] < best["metric"]:
                    best = {
                        "loss": best["loss"],
                        "metric": selection_metrics["all/mse"],
                        "step": step,
                    }
                    best_params = params
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1
                # Patience is opt-in so existing fixed-budget runs are unchanged.
                # A1 and A2 must share one stopping rule, otherwise the
                # discretization ratio compares two differently-converged models.
                if 0 < args.patience_evals <= evals_without_improvement:
                    stopped_early = True
                    stop_reason = "validation_patience"
                    break
            if args.early_stop_mse > 0 and loss <= args.early_stop_mse:
                stopped_early = True
                stop_reason = "train_loss_threshold"
                break
        if stopped_early:
            break

    params = best_params
    train_result = evaluate_split(
        params, train, normalizer, config,
        batch_size=args.eval_batch_size,
        diagnostic_states=args.diagnostic_states,
        forward=forward,
    )
    if mode == "overfit":
        selection_result = train_result
    else:
        selection_result = evaluate_split(
            params, selection, normalizer, config,
            batch_size=args.eval_batch_size,
            diagnostic_states=args.diagnostic_states,
            forward=forward,
        )
    splits = {"train": train_result, selection_name: selection_result}
    if args.evaluate_test:
        test = build_split(
            "test", select_split_rows(records, "test"), records, store, normalizer, device
        )
        splits["test"] = evaluate_split(
            params, test, normalizer, config,
            batch_size=args.eval_batch_size,
            diagnostic_states=args.diagnostic_states,
            forward=forward,
        )

    selection_metrics = selection_result["metrics"]
    gates = integrity_gates(selection_metrics, initial_selection_mse, best["metric"])
    gates["initial_gradients_finite"] = finite_gradients
    quality = quality_gates(args, selection_metrics, train_result["metrics"])
    gates.update(quality)
    gates["passed"] = all(gates.values())

    config_dict = dataclasses.asdict(config)
    config_dict["group_sizes"] = list(config_dict["group_sizes"])
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
            "encoder_key": "layer_norm(xbar_safe + input_position)",
            "encoder_value": "xbar_safe",
            "decoder_key": "latent_index_address",
            "decoder_value": "e_t",
            "decoder_attention": "content_independent",
            "latent": "continuous_deterministic",
            "pre_norm": True,
            "final_layer_norm": False,
            "mask_use": ["encoder_key_masking", "output_zeroing", "loss_exclusion"],
        },
        "capacity": {
            "input_scalars": config.input_scalars,
            "e_scalars": config.e_scalars,
            "scalar_ratio": config.scalar_ratio,
            "compression_factor": 1.0 / config.scalar_ratio,
            "parameters": parameter_count(params),
            "note": "scalar-dimensional only; e_t is unquantized FP32 and the 64 "
                    "mask bits are external metadata",
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
            "train_global_indices": train.global_indices if mode == "overfit" else None,
            "train_rows": train.rows if mode == "overfit" else None,
        },
        "optimizer": {
            "stage_steps": [int(value) for value in args.stage_steps],
            "stage_learning_rates": [float(value) for value in args.stage_learning_rates],
            "weight_decay": args.weight_decay,
            "clip_norm": args.clip_norm,
            "batch_size": batch_size,
            "steps_ran": step,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "patience_evals": args.patience_evals,
            "optimizer_state_reset_per_stage": True,
        },
        "init_checkpoint": (
            str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None
        ),
        "initial_loss": initial_loss,
        "initial_selection_mse": initial_selection_mse,
        "initial_gradient_norms": gradient_norms,
        "best_train_batch_loss": best["loss"],
        "best_selection_mse": best["metric"],
        "best_selection_step": best["step"],
        "history": history,
        "splits": {name: value["metrics"] for name, value in splits.items()},
        "per_task": {name: value["per_task"] for name, value in splits.items()},
        "baseline": {
            "a0_slotkey_mse": A0_SLOTKEY_MSE,
            "a0_slotkey_r2": A0_SLOTKEY_R2,
            "a0_slotkey_denormalized_rmse": A0_SLOTKEY_RAW_RMSE,
            "selection_mse_over_baseline": selection_metrics["all/mse"] / A0_SLOTKEY_MSE,
        },
        "gate_profile": args.gate_profile,
        "gates": gates,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    save_checkpoint(
        output.with_suffix(".npz"),
        params,
        {
            "protocol": PROTOCOL,
            "format_version": FORMAT_VERSION,
            "config": config_dict,
            "wiring": result["wiring"],
            "normalization_hash": result["data"]["normalization_hash"],
            "pca_sha256": result["data"]["pca_sha256"],
            "records_sha256": result["data"]["records_sha256"],
            "note": "parameters only; optimizer moments are not stored, so "
                    "--init-checkpoint restarts AdamW from zero state",
        },
    )
    if args.diagnostics_npz:
        count = min(args.diagnostic_states, len(selection))
        _, encoder_weights = encode(
            params, selection.xbar_device[:count], selection.valid_device[:count],
            config, return_attention=True,
        )
        np.savez_compressed(
            args.diagnostics_npz,
            encoder_attention=np.asarray(jnp.mean(encoder_weights, axis=-3), np.float32),
            decoder_attention=np.asarray(
                jnp.mean(decoder_attention(params, config), axis=0), np.float32
            ),
            global_indices=np.asarray(selection.global_indices[:count], np.int64),
            valid=np.asarray(selection.valid[:count], np.bool_),
            split=np.asarray(selection.name),
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="a1")
    parser.add_argument("--num-e-tokens", type=int, default=16)
    parser.add_argument("--e-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-hidden", type=int, default=1024)
    parser.add_argument("--overfit-states", type=int, default=0,
                        help="full-batch overfit on K task-balanced train states")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage-steps", type=int, nargs="+", default=[2000, 3000, 5000])
    parser.add_argument("--stage-learning-rates", type=float, nargs="+",
                        default=[1e-3, 3e-4, 1e-4])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--diagnostic-states", type=int, default=128)
    parser.add_argument("--diagnostics-npz")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--early-stop-mse", type=float, default=0.0)
    parser.add_argument("--patience-evals", type=int, default=0,
                        help="stop after this many consecutive evaluations without "
                             "selection-split improvement (0 disables). A1 and A2 must "
                             "use the same value for their comparison to be fair.")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="only after capacity selection and gate thresholds are frozen")
    parser.add_argument("--gate-profile",
                        choices=("structural", "report_only", "compressed"),
                        default="report_only")
    parser.add_argument("--baseline-mse", type=float, default=A0_SLOTKEY_MSE)
    parser.add_argument("--structural-mse-ratio", type=float, default=3.0)
    parser.add_argument("--total-mse-gate", type=float, default=1e-4)
    parser.add_argument("--total-r2-gate", type=float, default=0.999)
    parser.add_argument("--group-mse-gate", type=float, default=1e-3)
    parser.add_argument("--group-r2-gate", type=float, default=0.99)
    parser.add_argument("--raw-rmse-gate", type=float, default=1e-2)
    parser.add_argument("--compressed-r2-gate", type=float, default=0.90)
    parser.add_argument("--compressed-group-r2-gate", type=float, default=0.80)
    parser.add_argument("--max-r2-gap", type=float, default=0.10)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--matmul-precision",
                        choices=("highest", "float32", "tensorfloat32", "bfloat16"),
                        default="highest",
                        help="numerics protocol default is 'highest'; anything else "
                             "must be labelled a throughput run and never compared "
                             "side by side with protocol results")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    selection = result["data"]["selection_split"]
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "label": result["label"],
        "mode": result["mode"],
        "num_e_tokens": result["config"]["num_e_tokens"],
        "e_dim": result["config"]["e_dim"],
        "compression_factor": result["capacity"]["compression_factor"],
        "steps_ran": result["optimizer"]["steps_ran"],
        "initial_loss": result["initial_loss"],
        f"{selection}/mse": result["splits"][selection]["all/mse"],
        f"{selection}/r2": result["splits"][selection]["all/r2"],
        "mse_over_a0_slotkey": result["baseline"]["selection_mse_over_baseline"],
        "gate_profile": result["gate_profile"],
        "gates": result["gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
