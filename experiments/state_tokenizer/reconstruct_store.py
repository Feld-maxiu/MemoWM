"""Materialise a model's reconstructions as a drop-in feature store.

The probes read raw Static PCA states through ``FixedRepresentationStore``, which
looks up ``key64-static-pca-bf16.npy`` by ``global_index``. Writing
reconstructions in that exact layout means the probe side needs no changes at
all: point it at this directory instead of ``static_features``.

Two things are easy to get wrong and are asserted here rather than trusted:

* the probes consume the *raw* PCA state ``x_t``, so the normalized
  reconstruction has to be denormalized before it is written;
* a store that silently disagrees with the run it came from would corrupt every
  downstream comparison, so the validation R^2 is recomputed from what was
  actually written and checked against the training record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from residualmem.encoders.normalization import GroupChannelNormalizer
from residualmem.world_model import categorical_bottleneck as Q
from residualmem.world_model import continuous_bottleneck as C

from .a1_continuous_bottleneck import FeatureStore, select_split_rows
from .a1_continuous_bottleneck import load_checkpoint as load_a1_checkpoint
from .a2_categorical_bottleneck import load_checkpoint as load_a2_checkpoint
from .a2_categorical_bottleneck import read_checkpoint_config
from .common import iter_jsonl, write_json


def _load_model(path: Path):
    """Return ``(params, config, kind)`` for either an A1 or an A2 checkpoint."""
    with np.load(path, allow_pickle=False) as checkpoint:
        if "metadata" not in checkpoint.files:
            raise ValueError(f"{path} has no metadata")
        protocol = json.loads(
            str(np.asarray(checkpoint["metadata"]).item())
        ).get("protocol")
    if protocol == C.PROTOCOL:
        stored = json.loads(
            str(np.asarray(np.load(path, allow_pickle=False)["metadata"]).item())
        )["config"]
        config = C.ContinuousBottleneckConfig(
            **{**stored, "group_sizes": tuple(stored["group_sizes"])}
        )
        return load_a1_checkpoint(path, config), config, "a1"
    if protocol == Q.PROTOCOL:
        stored = read_checkpoint_config(path)
        config = Q.CategoricalBottleneckConfig(
            **{**stored, "group_sizes": tuple(stored["group_sizes"])}
        )
        return load_a2_checkpoint(path, config), config, "a2"
    raise ValueError(f"unknown checkpoint protocol: {protocol!r}")


def _reconstruct(params, xbar, valid, config, kind):
    if kind == "a1":
        return C.reconstruct(params, xbar, valid, config)
    return Q.reconstruct(params, xbar, valid, config)


def _copy_positions(store: FeatureStore, indices: list[int]) -> np.ndarray:
    """Slot geometry for the given global indices, taken from the source store.

    ``FixedRepresentationStore`` reads ``key64-static-positions.npy`` alongside
    the tokens for this representation. Positions describe the layout, not the
    content, so they are copied rather than recomputed.
    """
    cache: dict[Path, np.ndarray] = {}
    rows = []
    for index in indices:
        worker, local = store.locations[index]
        if worker not in cache:
            cache[worker] = np.load(worker / "key64-static-positions.npy", mmap_mode="r")
        rows.append(np.asarray(cache[worker][local], np.float32))
    return np.stack(rows)


def run(args: argparse.Namespace) -> dict:
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    records = list(iter_jsonl(args.records))
    normalizer = GroupChannelNormalizer.from_npz(args.normalization)
    store = FeatureStore(args.features)
    device = jax.devices(args.platform)[args.device_index]
    params, config, kind = _load_model(Path(args.checkpoint))
    params = jax.device_put(params, device)

    output = Path(args.output)
    rows = list(range(len(records)))
    shards = max(1, args.shards)
    written = 0
    for shard in range(shards):
        subset = rows[shard::shards]
        if not subset:
            continue
        indices = [int(records[row]["global_index"]) for row in subset]
        x_t, valid = store.load(indices)
        xbar = normalizer.normalize(x_t, valid)
        pieces = []
        for start in range(0, len(subset), args.batch_size):
            block = slice(start, start + args.batch_size)
            prediction = _reconstruct(
                params, jnp.asarray(xbar[block]), jnp.asarray(valid[block]), config, kind
            )
            pieces.append(np.asarray(prediction, np.float32))
        prediction = np.concatenate(pieces, 0)
        # Probes consume raw PCA states; padding stays exactly zero either way.
        recovered = normalizer.denormalize(prediction, valid)
        if np.count_nonzero(recovered[~valid]):
            raise ValueError("denormalized reconstruction leaked into padding")

        worker = output / f"worker{shard:02d}"
        worker.mkdir(parents=True, exist_ok=True)
        np.save(worker / "key64-static-pca-bf16.npy",
                recovered.astype(ml_dtypes.bfloat16).view(np.uint16))
        np.save(worker / "key64-static-valid.npy", valid)
        np.save(worker / "record_indices.npy", np.asarray(indices, np.int64))
        np.save(worker / "subset_rows.npy", np.asarray(subset, np.int64))
        np.save(worker / "done.npy", np.ones((len(subset),), bool))
        np.save(worker / "key64-static-pca-done.npy", np.ones((len(subset),), bool))
        # Slot geometry is a property of the layout, not of the reconstruction;
        # FixedRepresentationStore reads it alongside the tokens, so it is copied
        # across per global index rather than regenerated.
        np.save(worker / "key64-static-positions.npy",
                _copy_positions(store, indices))
        written += len(subset)

    # Recompute the headline metric from the model itself and compare against the
    # training record: a store that disagrees with its source is worse than no
    # store at all, because nothing downstream would notice.
    #
    # Evaluated in batches with the SSE accumulated in FP64 on the host. Doing it
    # in one shot allocates a tensor proportional to the whole split, which is
    # fine for 2,008 validation states and asks for 42 GB at 20,002 -- the check
    # would then die exactly on the datasets where it matters most.
    module = C if kind == "a1" else Q
    rows = select_split_rows(records, "validation")
    indices = [int(records[row]["global_index"]) for row in rows]
    totals: dict[str, float] = {}
    for start in range(0, len(indices), args.batch_size):
        block = indices[start:start + args.batch_size]
        block_x, block_valid = store.load(block)
        block_xbar = jnp.asarray(normalizer.normalize(block_x, block_valid))
        block_mask = jnp.asarray(block_valid)
        block_prediction = _reconstruct(
            params, block_xbar, block_mask, config, kind
        )
        for name, value in module.group_sse(
            block_prediction, block_xbar, block_mask, config
        ).items():
            totals[name] = totals.get(name, 0.0) + float(np.asarray(value, np.float64))
    metrics = module.metrics_from_sse(totals)
    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "kind": kind,
        "assignment": getattr(config, "assignment", None),
        "code_bits_per_state": getattr(config, "code_bits", None),
        "states_written": written,
        "shards": shards,
        "validation_r2_recomputed": metrics["all/r2"],
        "validation_mse_recomputed": metrics["all/mse"],
        "normalization_hash": normalizer.hash_bytes.hex(),
        "pca_sha256": normalizer.pca_sha256,
        "note": "raw PCA space (denormalized); drop-in for FixedRepresentationStore",
    }
    if args.expected_r2 is not None:
        drift = abs(metrics["all/r2"] - args.expected_r2)
        report["expected_r2"] = args.expected_r2
        report["r2_drift"] = drift
        if drift > args.r2_tolerance:
            raise ValueError(
                f"recomputed validation R2 {metrics['all/r2']:.6f} differs from the "
                f"recorded {args.expected_r2:.6f} by {drift:.2e}"
            )
    write_json(output / "summary.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--expected-r2", type=float,
                        help="validation R2 from the training record; the store is "
                             "rejected if what was written disagrees")
    parser.add_argument("--r2-tolerance", type=float, default=1e-5)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--matmul-precision", default="highest")
    return parser


def main() -> None:
    print(json.dumps(run(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
