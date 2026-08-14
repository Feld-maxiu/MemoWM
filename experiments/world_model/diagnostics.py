"""Descriptive M1 slices for a paired WM/baseline validation artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.common import sha256_file, write_json

from .cache import CACHE_FILES, FrozenCache
from .schema import POLICY_NAMES


def _load(path: str | Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    return {name: archive[name] for name in archive.files}


def _slice(baseline: dict, model: dict, selected: np.ndarray) -> dict:
    count = int(np.sum(selected))
    if not count:
        raise ValueError("diagnostic slice is empty")
    model_bits = float(np.mean(model["total_bits"][selected], dtype=np.float64))
    copy_bits = float(
        np.mean(baseline["copy_total_bits"][selected], dtype=np.float64)
    )
    source_bits = float(
        np.mean(baseline["source_total_bits"][selected], dtype=np.float64)
    )
    return {
        "transitions": count,
        "model_bits_per_transition": model_bits,
        "copy_bits_per_transition": copy_bits,
        "source_bits_per_transition": source_bits,
        "gain_vs_copy_bits_per_transition": copy_bits - model_bits,
        "gain_vs_source_bits_per_transition": source_bits - model_bits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache = FrozenCache(args.cache)
    baseline = _load(args.baseline)
    model = _load(args.model)
    for name in ("transition_indices", "task_ids", "episode_ids", "policies"):
        if name not in model or not np.array_equal(baseline[name], model[name]):
            raise ValueError(f"model/baseline are not paired on {name}")
    rows = model["transition_indices"]
    validation = set(cache.indices_for_split("validation").tolist())
    if any(int(row) not in validation for row in rows):
        raise ValueError("M1 diagnostic input is not validation-only")
    task_ids = model["task_ids"]
    policies = model["policies"]
    steps = model["steps"]
    report = {
        "protocol": "v8_wm_m1_descriptive_diagnostics_v1",
        "scope": "single-seed descriptive validation slices; no confidence interval",
        "overall": _slice(baseline, model, np.ones((len(rows),), np.bool_)),
        "by_task": {
            cache.task_names[int(task)]: _slice(baseline, model, task_ids == task)
            for task in sorted(np.unique(task_ids).tolist())
        },
        "by_policy": {
            POLICY_NAMES[int(policy)]: _slice(baseline, model, policies == policy)
            for policy in sorted(np.unique(policies).tolist())
        },
        "by_source_step": {
            str(int(step)): _slice(baseline, model, steps == step)
            for step in sorted(np.unique(steps).tolist())
        },
        "artifacts": {
            "cache_manifest_sha256": sha256_file(
                Path(args.cache) / CACHE_FILES["manifest"]
            ),
            "baseline": str(Path(args.baseline).resolve()),
            "baseline_sha256": sha256_file(args.baseline),
            "model": str(Path(args.model).resolve()),
            "model_sha256": sha256_file(args.model),
        },
        "interpretation_guard": (
            "These slices localize the failed M1 point gate. They do not replace "
            "the preregistered three-seed episode bootstrap and cannot support C1/C2."
        ),
    }
    write_json(args.output, report)
    print(json.dumps(report["overall"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
