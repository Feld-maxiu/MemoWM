"""Fit the frozen group x channel normalization for Static PCA states.

Statistics come from valid slots of the train split only, matching the artifact
the world-model runners consume. Padding stays exactly zero after normalization
because the runners re-apply the mask.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np

from .common import iter_jsonl, write_json
from .slot_layout import GROUP_NAMES as _LAYOUT_GROUPS, KEY64_LAYOUT

GROUP_NAMES = ("image", "detail", "context", "prompt")
# Derived, never a literal. These statistics are per group x channel, so a layout
# that disagrees with the store computes each group's mean and sigma over the
# wrong slots -- and it does so silently. A hardcoded (32,12,16,4) survived the
# prompt band being recycled into detail and reported 4890 valid "prompt" slots
# for a representation that has none.
LAYOUT = tuple(KEY64_LAYOUT)


def run(args: argparse.Namespace) -> dict:
    if sum(LAYOUT) != 64 or len(LAYOUT) != len(GROUP_NAMES):
        raise ValueError(f"layout {LAYOUT} does not describe 64 slots in {len(GROUP_NAMES)} groups")
    records = [row for row in iter_jsonl(args.records) if row["split"] == "train"]
    wanted = {int(row["global_index"]) for row in records}
    group_index = np.repeat(np.arange(len(LAYOUT)), LAYOUT)

    total = np.zeros((len(LAYOUT), args.token_dim), np.float64)
    square = np.zeros((len(LAYOUT), args.token_dim), np.float64)
    counts = np.zeros((len(LAYOUT),), np.int64)
    for shard in sorted(Path(args.features).glob("worker*")):
        indices = np.load(shard / "record_indices.npy")
        keep = np.array([int(i) in wanted for i in indices], bool)
        if not keep.any():
            continue
        bits = np.load(shard / "key64-static-pca-bf16.npy", mmap_mode="r")
        valid = np.load(shard / "key64-static-valid.npy", mmap_mode="r")
        rows = np.flatnonzero(keep)
        for start in range(0, len(rows), args.batch_states):
            block = rows[start:start + args.batch_states]
            values = (
                np.asarray(bits[block], np.uint16).view(ml_dtypes.bfloat16).astype(np.float64)
            )
            mask = np.asarray(valid[block], bool)
            for group in range(len(LAYOUT)):
                slots = group_index == group
                selected = values[:, slots][mask[:, slots]]
                if selected.size:
                    total[group] += selected.sum(0)
                    square[group] += np.square(selected).sum(0)
                    counts[group] += selected.shape[0]
    # A zero-width group is a deliberate layout choice -- the prompt band is 0
    # once recycled into detail -- so it is allowed to have no slots. A group that
    # *has* slots but no valid ones is a real fault: every state masked it out.
    widths = np.asarray(LAYOUT, np.int64)
    starved = (counts == 0) & (widths > 0)
    if starved.any():
        raise ValueError(
            f"groups {[GROUP_NAMES[i] for i in np.flatnonzero(starved)]} have slots "
            f"but no valid train data: {counts.tolist()}"
        )

    safe = np.maximum(counts, 1)[:, None]
    mean = total / safe
    variance = np.maximum(square / safe - np.square(mean), 0.0)
    std = np.sqrt(variance)
    scale = np.maximum(std, args.sigma_min)

    pca_sha256 = hashlib.sha256(Path(args.pca).read_bytes()).hexdigest()
    output = Path(args.output)
    np.savez(
        output,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        scale=scale.astype(np.float32),
        counts=counts,
        group_names=np.asarray(GROUP_NAMES),
        layout=np.asarray(LAYOUT, np.int32),
        sigma_min=np.asarray(args.sigma_min, np.float32),
        pca_sha256=np.asarray(pca_sha256),
        protocol=np.asarray("group_channel_train_v1"),
    )
    report = {
        "protocol": "group_channel_train_v1",
        "records": str(Path(args.records).resolve()),
        "features": str(Path(args.features).resolve()),
        "pca": str(Path(args.pca).resolve()),
        "pca_sha256": pca_sha256,
        "train_states": len(records),
        "valid_slots_per_group": {
            name: int(value) for name, value in zip(GROUP_NAMES, counts)
        },
        "sigma_min": args.sigma_min,
        "scale_min": float(scale.min()),
        "scale_max": float(scale.max()),
        "scale_ratio": float(scale.max() / scale.min()),
        "clamped_channels": int((std < args.sigma_min).sum()),
    }
    write_json(output.with_suffix(".summary.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--token-dim", type=int, default=512)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--batch-states", type=int, default=64)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
