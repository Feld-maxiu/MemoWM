"""Offline probe: is per-axis persistence predictable from cheap features?

Mutual-information counting over the webworld-v2 cache:
  * per axis (i,g): I(source_code ; persist) in bits
  * global: I(action_type ; per-axis persist)
If these are ~0, per-axis gate conditioning has no signal to learn and the
copy-baseline gap is unreachable from these features.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/home/cbd/project/residual-mem")
from experiments.world_model.cache import FrozenCache  # noqa: E402


def main() -> None:
    cache = FrozenCache(
        "/home/cbd/project/residual-mem/outputs/wm_train/webworld-v2/cache-h8"
    )
    rows = cache.indices_for_split("train")
    rows = rows[np.random.default_rng(0).permutation(len(rows))][:8000]
    batch = cache.batch(rows)
    source = batch["history_codes"][:, -1].astype(np.int64)  # (B, 32, 32)
    target = batch["target_codes"].astype(np.int64)  # (B, 32, 32)
    persist = (source == target).astype(np.int64)  # (B, 32, 32)
    action = batch["action_types"][:, -1].astype(np.int64)  # (B,)
    print(f"rows {len(rows)}  persist rate {persist.mean():.4f}")

    # --- per-axis I(source_code ; persist) via counting ---
    mi = np.zeros((32, 32))
    p_persist = persist.mean()
    sampled = 0
    for i in range(32):
        for g in range(32):
            code = source[:, i, g]
            keep = persist[:, i, g]
            counts = np.zeros((64, 2), np.int64)
            for c in (0, 1):
                np.add.at(counts[:, c], code, keep == c)
            total = counts.sum()
            if total < 500:
                continue
            sampled += 1
            joint = counts / total
            p_code = counts.sum(1) / total
            p_keep_marg = counts.sum(0) / total
            denom = p_code[:, None] * p_keep_marg[None, :]
            mask = joint > 0
            mi[i, g] = np.sum(
                joint[mask] * np.log2(joint[mask] / denom[mask])
            )
    flat = mi.ravel()
    print(
        f"I(code;persist) per axis: mean {flat.mean():.5f} bits, "
        f"p50 {np.percentile(flat, 50):.5f}, p90 {np.percentile(flat, 90):.5f}, "
        f"max {flat.max():.5f} (axes with >=500 samples: {sampled}/1024)"
    )

    # --- global I(action_type ; per-axis persist) ---
    act_counts = np.zeros((17, 2), np.int64)
    for a in range(17):
        mask = action == a
        if mask.sum() == 0:
            continue
        act_counts[a, 1] = persist[mask].sum()
        act_counts[a, 0] = persist[mask].size - persist[mask].sum()
    total = act_counts.sum()
    mi_action = 0.0
    joint_a = act_counts / total
    p_act = act_counts.sum(1) / total
    p_keep_marg = act_counts.sum(0) / total
    denom_a = p_act[:, None] * p_keep_marg[None, :]
    mask_a = joint_a > 0
    mi_action = float(
        np.sum(joint_a[mask_a] * np.log2(joint_a[mask_a] / denom_a[mask_a]))
    )
    rates = act_counts[:, 1] / np.maximum(act_counts.sum(1), 1)
    shown = [(a, f"{rates[a]:.3f}") for a in range(17) if act_counts[a].sum() >= 100]
    print(f"I(action;persist) global: {mi_action:.5f} bits; per-action persist rates: {shown}")


if __name__ == "__main__":
    main()
