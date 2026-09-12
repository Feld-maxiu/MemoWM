"""Post-hoc per-action-type copy-gate calibration (action-bias probe).

Extends 方案0's global-delta calibration to one delta per action type:
bias[a] added to every axis's copy_logit for transitions whose *last*
action has type a.  Fitted post-hoc on an AMA split, this answers the
decisive question the trained auxbias run could not: does an
action-conditioned gate prior buy any bits at all on top of the plain
v2fix checkpoint, when the prior is fitted on the *evaluation domain*
instead of WebWorld (whose per-action keep rates differ from AMA's)?

The total bits decompose exactly across transitions, so fitting uses
uniform-delta passes over mixed-type rows (per-type curves from
per-transition bits), and verification re-evaluates each type's rows
with its own fitted delta.

Usage:
  python -m experiments.world_model.calibrate_gate_action \
    --checkpoint outputs/.../full_seed0_c64_v2fix/best.pkl \
    --config configs/world_model/webworld_v2_c64.yaml \
    --cache outputs/wm_train/ama-v1/cache-h4 \
    --grid -3:3:0.25 --fit-rows 4000 --min-type-count 30
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from .config import load_config
from .cache import FrozenCache
from .schema import validate_variant
from .schema_web import ACTION_TYPE_IDS
from .train import evaluate_model


def _parse_grid(spec: str) -> list[float]:
    lo, hi, step = (float(x) for x in spec.split(":"))
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


def _with_action_bias(params, bias: np.ndarray):
    adjusted = {
        k: (dict(v) if isinstance(v, dict) else v) for k, v in params.items()
    }
    adjusted["copy_action_bias"] = jnp.asarray(bias, jnp.float32)
    return adjusted


def _last_action_types(cache: FrozenCache, rows: np.ndarray) -> np.ndarray:
    types = np.full(len(rows), ACTION_TYPE_IDS["PAD"], np.int32)
    for start in range(0, len(rows), 512):
        chunk = rows[start:start + 512]
        batch = cache.batch(chunk)
        types[start:start + len(chunk)] = np.asarray(
            batch["action_types"], np.int32
        )[:, -1]
    return types


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True,
                        help="AMA cache; deltas fitted on its train split, "
                             "reported on its validation split")
    parser.add_argument("--grid", default="-3:3:0.25")
    parser.add_argument("--fit-rows", type=int, default=4000)
    parser.add_argument("--min-type-count", type=int, default=30,
                        help="types with fewer fit transitions keep delta 0")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--device-index", type=int, default=5)
    args = parser.parse_args()
    validate_variant(args.variant)

    device = jax.local_devices()[min(args.device_index, len(jax.local_devices()) - 1)]
    cache = FrozenCache(args.cache)
    config = load_config(args.config, num_tasks=len(cache.task_names))
    cache.max_history = config.model.max_history
    config = dataclasses.replace(
        config,
        model=dataclasses.replace(config.model, use_action_gate_bias=True),
    )
    checkpoint = pickle.load(open(args.checkpoint, "rb"))
    params = checkpoint["params"]
    assert "copy_head/w" in params, "checkpoint has no copy gate"
    assert "persistence_head/w" not in params, "expected the plain v2fix ckpt"
    num_action_types = config.model.num_action_types

    train_rows = cache.indices_for_split("train")
    val_rows = cache.indices_for_split("validation")
    rng = np.random.default_rng(0)
    fit_rows = rng.permutation(train_rows)[: args.fit_rows]
    fit_types = _last_action_types(cache, fit_rows)
    val_types = _last_action_types(cache, val_rows)
    pad_id = ACTION_TYPE_IDS["PAD"]
    print(f"fit rows {len(fit_rows)}, val rows {len(val_rows)}; "
          f"fit last-action PAD count {(fit_types == pad_id).sum()}",
          flush=True)

    grid = _parse_grid(args.grid)
    # Per-type bits curve: one uniform-delta pass over the mixed fit rows.
    curves = np.full((num_action_types, len(grid)), np.nan, np.float64)
    counts = np.zeros(num_action_types, np.int64)
    for t in fit_types:
        if 0 <= t < num_action_types:
            counts[t] += 1
    for gi, delta in enumerate(grid):
        moved = jax.device_put(
            _with_action_bias(params, np.full(num_action_types, delta)), device
        )
        result, parts = evaluate_model(
            cache, fit_rows, moved, args.variant, config,
            keep_per_transition=True,
        )
        per_bits = parts["total_bits"]
        for a in range(num_action_types):
            mask = fit_types == a
            if mask.any():
                curves[a, gi] = per_bits[mask].mean()
        print(f"  delta {delta:+.2f} -> fit bits {result['total_bits_per_transition']:.2f}",
              flush=True)

    fitted = np.zeros(num_action_types, np.float32)
    print("\nper-type fitted delta:")
    for a in range(num_action_types):
        if counts[a] == 0:
            continue
        name = next(k for k, v in ACTION_TYPE_IDS.items() if v == a)
        if counts[a] < args.min_type_count:
            print(f"  {name:>12} n={counts[a]:>5}  skipped (below --min-type-count)")
            continue
        gi = int(np.nanargmin(curves[a]))
        fitted[a] = grid[gi]
        zero_gi = grid.index(0.0)
        print(f"  {name:>12} n={counts[a]:>5}  best {grid[gi]:+.2f} "
              f"(bits {curves[a, gi]:.1f} vs d=0 {curves[a, zero_gi]:.1f})")

    def eval_with_per_type_bias(rows, types, use_fitted: bool) -> float:
        # Bits decompose per transition, so evaluate each type's rows with
        # its own delta and recombine by count.
        total = 0.0
        count = 0
        for a in np.unique(types):
            if a == pad_id or a >= num_action_types:
                continue
            subset = rows[types == a]
            bias_a = np.zeros(num_action_types, np.float32)
            if use_fitted:
                bias_a[a] = fitted[a]
            moved = jax.device_put(_with_action_bias(params, bias_a), device)
            result, _ = evaluate_model(
                cache, subset, moved, args.variant, config,
                keep_per_transition=False,
            )
            total += result["total_bits_per_transition"] * result["transitions"]
            count += result["transitions"]
        return total / max(count, 1)

    base = eval_with_per_type_bias(val_rows, val_types, use_fitted=False)
    calibrated = eval_with_per_type_bias(val_rows, val_types, use_fitted=True)
    print("\n=== AMA validation ===")
    print(f"uncalibrated (per-type eval): {base:.2f} bits/transition")
    print(f"per-action calibrated       : {calibrated:.2f} bits/transition")
    print(f"v2fix single-pass best      : 4733.74 bits/transition")


if __name__ == "__main__":
    main()
