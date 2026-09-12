"""Where do we lose vs the copy/source baselines, and the cheapest fix.

Decomposes the world model's per-axis code bits on the AMA validation
split into kept (target == source) and changed axes, then re-scores the
same axes under post-hoc mixtures of the model with train-fitted
baseline distributions:

    P' = (1 - w) * P_model + w * P_base     (base = marginal / copy / source)

Protocol: baseline tables are built on train[:8000], the mixing weight w
is fitted on train[8000:] (out-of-table rows), and reported on the
untouched validation split.  No retraining anywhere.

Usage:
  python -m experiments.world_model.diagnose_change_path \
    --checkpoint outputs/.../full_seed0_c64_v2fix/best.pkl \
    --bias-json  outputs/.../full_seed0_c64_v2fix/calibrated_action_bias.json \
    --cache outputs/wm_train/ama-v1/cache-h4 \
    --config configs/world_model/webworld_v2_c64.yaml --device-index 0
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from .config import load_config
from .cache import FrozenCache
from .model import loss_and_metrics
from .schema_web import ACTION_TYPE_IDS
from .train import _device_batch, model_batch_keys


def model_bits_per_axis(cache, rows, params, variant, config,
                        batch_size=32):
    keys = model_batch_keys(config.model)
    width = config.model.num_latent_tokens * config.model.num_subspaces
    out = np.zeros((len(rows), width), np.float64)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        host = cache.batch(chunk)
        batch = _device_batch(host, keys)
        _, (_, rates, _, _) = loss_and_metrics(
            params, batch, variant, config.model)
        matrix = jax.device_get(rates["code_matrix"])
        out[start:start + len(chunk)] = matrix.reshape(len(chunk), -1)
    return out


def source_and_target(cache, rows):
    host = cache.batch(rows)
    source = np.asarray(host["history_codes"], np.int32)[:, -1]
    target = np.asarray(host["target_codes"], np.int32)
    n = source.shape[0]
    return source.reshape(n, -1), target.reshape(n, -1)


class BaselineTables:
    """baselines.py distributions: marginal / copy / source (per axis)."""

    def __init__(self, fit_source, fit_target, num_categories):
        alpha = 0.5
        self.alpha = alpha
        self.C = num_categories
        positions = fit_source.shape[1]
        self.positions = positions
        self.n_fit = len(fit_source)
        offsets = np.arange(positions, dtype=np.int64) * num_categories
        self.offsets = offsets
        self.marginal_counts = np.bincount(
            (fit_target.astype(np.int64) + offsets).ravel(),
            minlength=positions * num_categories,
        ).reshape(positions, num_categories)
        fit_same = fit_source == fit_target
        keep_count = fit_same.sum(axis=0)
        self.change_count = self.n_fit - keep_count
        changed_counts = np.bincount(
            ((fit_target.astype(np.int64) + offsets))[~fit_same].ravel(),
            minlength=positions * num_categories,
        ).reshape(positions, num_categories)
        self.p_keep = (keep_count + alpha) / (self.n_fit + 2 * alpha)
        self.changed_counts = changed_counts

        pair_stride = num_categories * num_categories
        self.pair_keys, self.pair_counts = np.unique(
            ((np.arange(positions, dtype=np.int64) * pair_stride)[None, :]
             + (fit_source.astype(np.int64) * num_categories
                + fit_target)).ravel(),
            return_counts=True)
        self.src_keys, self.src_counts = np.unique(
            (offsets[:, None] + fit_source.T).T.ravel(), return_counts=True)

    @staticmethod
    def _lookup(keys, counts, wanted):
        idx = np.searchsorted(keys, wanted.ravel())
        idx = np.minimum(idx, len(keys) - 1)
        hit = keys[idx] == wanted.ravel()
        return np.where(hit, counts[idx], 0).reshape(wanted.shape)

    def probs(self, eval_source, eval_target):
        pos_idx = np.arange(self.positions)[None, :]
        p_marginal = (
            self.marginal_counts[pos_idx, eval_target] + self.alpha
        ) / (self.n_fit + self.C * self.alpha)
        eval_same = eval_source == eval_target
        dest_num = self.changed_counts[pos_idx, eval_target] + self.alpha
        dest_den = (self.change_count[None, :] + self.C * self.alpha
                    - (self.changed_counts[pos_idx, eval_source] + self.alpha))
        p_copy = np.where(eval_same, self.p_keep[None, :],
                          (1 - self.p_keep[None, :]) * dest_num / dest_den)
        pair_stride = self.C * self.C
        eval_pair = ((np.arange(self.positions, dtype=np.int64) * pair_stride)
                     [None, :]
                     + (eval_source.astype(np.int64) * self.C
                        + eval_target))
        eval_src = self.offsets[None, :] + eval_source
        pair_n = self._lookup(self.pair_keys, self.pair_counts, eval_pair)
        source_n = self._lookup(self.src_keys, self.src_counts, eval_src)
        p_source = (pair_n + 32.0 * p_copy) / (source_n + 32.0)
        return p_marginal, p_copy, p_source


def bits_of(p):
    return -np.log2(p).sum(axis=1)


def mix_bits(p_model, p_base, w):
    return -np.log2((1 - w) * p_model + w * p_base).sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bias-json")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--table-rows", type=int, default=8000)
    parser.add_argument("--fit-rows", type=int, default=4765)
    parser.add_argument("--variant", default="full")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    device = jax.local_devices()[min(
        args.device_index, len(jax.local_devices()) - 1)]
    cache = FrozenCache(args.cache)
    config = load_config(args.config, num_tasks=len(cache.task_names))
    cache.max_history = config.model.max_history

    checkpoint = pickle.load(open(args.checkpoint, "rb"))
    params = checkpoint["params"]
    if args.bias_json:
        bias = json.load(open(args.bias_json))["fitted_delta"]
        name_to_id = dict(ACTION_TYPE_IDS)
        vec = np.zeros(len(name_to_id), np.float32)
        for name, delta in bias.items():
            vec[name_to_id[name]] = delta
        params = dict(params)
        params["copy_action_bias"] = jnp.asarray(vec)
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, use_action_gate_bias=True),
        )
    params = jax.device_put(params, device)

    val_rows = cache.indices_for_split("validation")
    train_rows = cache.indices_for_split("train")
    rng = np.random.default_rng(0)
    perm = rng.permutation(train_rows)
    table_rows, fit_rows = perm[: args.table_rows], perm[args.table_rows:
                                                         args.table_rows
                                                         + args.fit_rows]
    print(f"table rows {len(table_rows)}, w-fit rows {len(fit_rows)}, "
          f"val rows {len(val_rows)}", flush=True)

    # ---- model per-axis bits ----
    p_model_val = np.exp2(-model_bits_per_axis(
        cache, val_rows, params, args.variant, config))
    p_model_fit = np.exp2(-model_bits_per_axis(
        cache, fit_rows, params, args.variant, config))
    print(f"model bits/transition: w-fit {bits_of(p_model_fit).mean():.2f} "
          f"val {bits_of(p_model_val).mean():.2f}", flush=True)

    # ---- baseline tables + probabilities ----
    ts, tt = source_and_target(cache, table_rows)
    tables = BaselineTables(ts, tt, config.model.num_categories)
    fs, ft = source_and_target(cache, fit_rows)
    vs, vt = source_and_target(cache, val_rows)
    fit_m, fit_c, fit_s = tables.probs(fs, ft)
    val_m, val_c, val_s = tables.probs(vs, vt)
    print(f"sanity on val (table={len(table_rows)} rows): "
          f"marginal {bits_of(val_m).mean():.2f} "
          f"copy {bits_of(val_c).mean():.2f} "
          f"source {bits_of(val_s).mean():.2f} "
          f"(official full-train: 4577.85 / 4064.33 / 3820.64)", flush=True)

    # ---- kept vs changed decomposition of the model on val ----
    same = vs == vt
    print("\nper-axis bits on val:")
    print(f"  kept axes   ({same.mean():.1%}): ours {(-np.log2(p_model_val[same]).mean()):.2f}"
          f"  copy {-np.log2(val_c[same]).mean():.2f}"
          f"  source {-np.log2(val_s[same]).mean():.2f}")
    print(f"  changed axes({(~same).mean():.1%}): ours {(-np.log2(p_model_val[~same]).mean()):.2f}"
          f"  copy {-np.log2(val_c[~same]).mean():.2f}"
          f"  source {-np.log2(val_s[~same]).mean():.2f}"
          f"  uniform 6.00")

    # ---- fit w on out-of-table train rows, report on val ----
    grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    print("\nw sweep on w-fit rows:")
    best = {}
    for name, base in (("marginal", fit_m), ("copy", fit_c), ("source", fit_s)):
        curves = np.asarray([mix_bits(p_model_fit, base, w).mean() for w in grid])
        w_best = grid[int(np.argmin(curves))]
        best[name] = w_best
        print(f"  {name:>8}: best w={w_best:.2f}  bits {curves.min():.2f}  "
              f"(w=0: {curves[0]:.2f}, w=1: {curves[-1]:.2f})")

    print("\n=== AMA validation (weights fitted out-of-table, no retraining) ===")
    print(f"model alone                                   : "
          f"{bits_of(p_model_val).mean():.2f}")
    for name, base in (("marginal", val_m), ("copy", val_c), ("source", val_s)):
        bits = mix_bits(p_model_val, base, best[name])
        print(f"(1-w)*model + w*{name:<8} (w={best[name]:.2f})          : "
              f"{bits.mean():.2f}")


if __name__ == "__main__":
    main()
