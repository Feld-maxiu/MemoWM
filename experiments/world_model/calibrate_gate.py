"""Post-hoc copy-gate calibration (experiment, 方案0).

Fits a single global bias delta added to every axis's copy_logit
(implemented as +delta on params["copy_head"]["b"]) on a TRAIN split,
then reports the calibrated NLL on the matching validation split.

The knob answers: how many bits does the gate's *global* open rate buy,
as opposed to per-axis conditioning?  Ceiling measured here bounds what
any gate-conditioning fix (aux head, delta softmax) can recover on top
of the uncalibrated checkpoint.

Usage:
  python -m experiments.world_model.calibrate_gate \
    --checkpoint outputs/.../best.pkl \
    --config configs/world_model/webworld_v2_c64.yaml \
    --train-cache outputs/wm_train/ama-v1/cache-h4 \
    --val-cache outputs/wm_train/ama-v1/cache-h4 \
    --grid -6:6:0.25 --fit-rows 2000 \
    --variant full
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np

from .config import load_config
from .cache import FrozenCache
from .train import evaluate_model
from .schema import validate_variant


def _parse_grid(spec: str) -> list[float]:
    lo, hi, step = (float(x) for x in spec.split(":"))
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


def _set_bias(params, delta: float):
    adjusted = {
        k: (dict(v) if isinstance(v, dict) else v) for k, v in params.items()
    }
    adjusted["copy_head/b"] = (
        np.asarray(adjusted["copy_head/b"], np.float32) + delta
    )
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-cache", required=True,
                        help="cache providing the TRAIN split used to fit delta")
    parser.add_argument("--val-cache", required=True,
                        help="cache providing the VALIDATION split to report")
    parser.add_argument("--grid", default="-6:6:0.25")
    parser.add_argument("--fit-rows", type=int, default=2000,
                        help="train-split transitions used to fit delta")
    parser.add_argument("--variant", default="full", choices=(
        "t_only", "no_action", "structural_action", "no_history", "full",
        "state_only", "struct_no_history", "semantic_action",
    ))
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=1)
    args = parser.parse_args()
    validate_variant(args.variant)

    import jax

    platform = jax.local_devices()[0].platform
    devices = jax.local_devices()
    device = devices[min(args.device_index, len(devices) - 1)]

    train_cache = FrozenCache(args.train_cache)
    config = load_config(args.config, num_tasks=len(train_cache.task_names))
    checkpoint = pickle.load(open(args.checkpoint, "rb"))
    params = checkpoint["params"]

    val_cache = FrozenCache(args.val_cache)
    train_cache.max_history = config.model.max_history
    val_cache.max_history = config.model.max_history
    train_rows_all = train_cache.indices_for_split("train")
    rng = np.random.default_rng(0)
    fit_rows = rng.permutation(train_rows_all)[: args.fit_rows]
    val_rows = val_cache.indices_for_split("validation")

    grid = _parse_grid(args.grid)
    print(f"fitting delta on {len(fit_rows)} train transitions "
          f"({args.train_cache}); grid {len(grid)} points", flush=True)

    scores = []
    for delta in grid:
        moved = jax.device_put(_set_bias(params, delta), device)
        result, _ = evaluate_model(
            train_cache, fit_rows, moved, args.variant, config,
            keep_per_transition=False,
        )
        scores.append((result["total_bits_per_transition"], delta))
        print(f"  delta {delta:+.2f} -> train bits {scores[-1][0]:.2f}",
              flush=True)
    best_bits, best_delta = min(scores)
    print(f"best delta {best_delta:+.2f} (train bits {best_bits:.2f})",
          flush=True)

    calibrated = jax.device_put(_set_bias(params, best_delta), device)
    val_result, _ = evaluate_model(
        val_cache, val_rows, calibrated, args.variant, config,
        keep_per_transition=False,
    )
    base_result, _ = evaluate_model(
        val_cache, val_rows, jax.device_put(params, device),
        args.variant, config, keep_per_transition=False,
    )
    print("\n=== validation ===")
    print(f"uncalibrated      : {base_result['total_bits_per_transition']:.2f}")
    print(f"calibrated (d={best_delta:+.2f}): "
          f"{val_result['total_bits_per_transition']:.2f}")
    print("(reference baselines are history-free and unchanged by delta)")


if __name__ == "__main__":
    main()
