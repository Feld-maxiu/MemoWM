"""Does the frozen tokenizer still hold on a differently-collected dataset?

v7 was collected with half its transitions drawn uniformly over the page's legal
actions, so it visits states the v6 encoder never saw -- and that encoder is
frozen by design (PCA, normalisation, A1 backbone and codebook are all reused
verbatim). If it no longer covers the new distribution, a world model trained on
these codes would be learning an already-distorted state, and nothing downstream
would reveal it.

This is the gate: reconstruction R^2 on the new data against the number the
frozen encoder was accepted at, plus codebook occupancy. A codebook that was
overfitted to the old distribution shows up as a large share of codewords that
the new states never select.

Evaluation follows the worklog's numerics protocol: matmul precision ``highest``,
FP32 activations, and SSE accumulated in FP64 on the host so the result does not
depend on how the split was batched.
"""
from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from residualmem.encoders.normalization import GroupChannelNormalizer
from residualmem.world_model import categorical_bottleneck as Q

from .a1_continuous_bottleneck import FeatureStore, select_split_rows
from .a2_categorical_bottleneck import load_checkpoint, read_checkpoint_config
from .common import iter_jsonl, write_json
from .slot_layout import KEY64_LAYOUT


def run(args: argparse.Namespace) -> dict:
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    records = list(iter_jsonl(args.records))
    rows = select_split_rows(records, args.split)
    indices = [int(records[row]["global_index"]) for row in rows]

    store = FeatureStore(args.features)
    normalizer = GroupChannelNormalizer.from_npz(args.normalization)
    # The checkpoint already records its own layout; overriding it with a literal
    # is what made this gate slice group metrics over the wrong slot ranges. The
    # dead (32, 12, 16, 4) predates the prompt band being recycled into detail,
    # so image/detail/context/prompt were reported over shifted spans while
    # all/* stayed correct -- the same silent failure that once had
    # fit_normalization compute prompt statistics for a group of width 0.
    config = Q.CategoricalBottleneckConfig(
        **{**read_checkpoint_config(args.checkpoint), "group_sizes": KEY64_LAYOUT}
    )
    device = jax.devices(args.platform)[args.device_index]
    params = jax.device_put(load_checkpoint(args.checkpoint, config), device)

    totals: dict[str, float] = {}
    histogram = np.zeros(
        (config.num_e_tokens, config.num_subspaces, config.num_categories), np.int64
    )
    for start in range(0, len(indices), args.batch_size):
        block = indices[start:start + args.batch_size]
        x_t, valid = store.load(block)
        xbar = jnp.asarray(normalizer.normalize(x_t, valid))
        mask = jnp.asarray(valid)
        prediction, codes = Q.reconstruct(
            params, xbar, mask, config, return_codes=True
        )
        # FP64 on the host: the batching must not move the metric
        for name, value in Q.group_sse(prediction, xbar, mask, config).items():
            totals[name] = totals.get(name, 0.0) + float(np.asarray(value, np.float64))
        histogram += Q.code_histogram(np.asarray(codes), config)

    metrics = Q.metrics_from_sse(totals)
    health = Q.code_health(histogram, config)
    used = int((histogram > 0).sum())
    slots = int(histogram.size)

    report = {
        "split": args.split,
        "states": len(indices),
        "checkpoint": args.checkpoint,
        "normalization": args.normalization,
        "features": args.features,
        "metrics": {k: v for k, v in metrics.items() if k.endswith("/r2")},
        "mse": {k: v for k, v in metrics.items() if k.endswith("/mse")},
        "code_health": health,
        "codebook_entries_used": used,
        "codebook_entries_total": slots,
        "codebook_unused_fraction": 1.0 - used / slots,
        "numerics": {
            "matmul_precision": args.matmul_precision,
            "metric_accumulation": "float64_host",
            "eval_batch_size": args.batch_size,
        },
    }
    if args.reference_r2 is not None:
        drop = args.reference_r2 - metrics["all/r2"]
        report["reference_r2"] = args.reference_r2
        report["r2_drop"] = drop
        report["verdict"] = (
            "hold" if drop < 0.02 else
            "hold_with_shift" if drop <= 0.05 else "stop"
        )
    write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation",
                        choices=("train", "validation", "test"))
    parser.add_argument("--reference-r2", type=float,
                        help="the R2 the frozen encoder was accepted at")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--matmul-precision", default="highest")
    report = run(parser.parse_args())
    print(json.dumps({
        "states": report["states"],
        "all_r2": report["metrics"]["all/r2"],
        "r2_drop": report.get("r2_drop"),
        "verdict": report.get("verdict"),
        "codebook_unused_fraction": report["codebook_unused_fraction"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
