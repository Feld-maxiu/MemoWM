"""Freeze keep/reject decision for the optional three-seed predicted-prefix curriculum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.common import write_json

from .statistics import paired_episode_bootstrap


def _load_rollout(path):
    archive = np.load(path, allow_pickle=False)
    return {name: archive[name] for name in archive.files}


def _cumulative_at_horizon(value, horizon):
    running = {}
    selected = []
    cumulative = []
    order = np.argsort(value["horizons"], kind="stable")
    for index in order:
        episode = int(value["episode_ids"][index])
        running[episode] = running.get(episode, 0.0) + float(value["total_bits"][index])
        if int(value["horizons"][index]) == horizon:
            selected.append(index)
            cumulative.append(running[episode])
    selected = np.asarray(selected, np.int64)
    return {
        "transition_indices": value["transition_indices"][selected],
        "task_ids": value["task_ids"][selected],
        "episode_ids": value["episode_ids"][selected],
        "cumulative_bits": np.asarray(cumulative, np.float64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-rollout", nargs=2, action="append", metavar=("SEED", "NPZ"), required=True)
    parser.add_argument("--candidate-rollout", nargs=2, action="append", metavar=("SEED", "NPZ"), required=True)
    parser.add_argument("--curriculum-json", nargs=2, action="append", metavar=("SEED", "JSON"), required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def keyed(entries, loader):
        result = {int(seed): loader(path) for seed, path in entries}
        if set(result) != {0, 1, 2}:
            raise ValueError(f"expected seeds 0/1/2, got {sorted(result)}")
        return result

    base = keyed(args.base_rollout, _load_rollout)
    candidate = keyed(args.candidate_rollout, _load_rollout)
    curricula = keyed(
        args.curriculum_json,
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
    )
    gains = {}
    for horizon in (3, 7):
        seed_gains = []
        reference = None
        for seed in (0, 1, 2):
            left = _cumulative_at_horizon(base[seed], horizon)
            right = _cumulative_at_horizon(candidate[seed], horizon)
            for name in ("transition_indices", "task_ids", "episode_ids"):
                if not np.array_equal(left[name], right[name]):
                    raise ValueError(f"seed {seed} horizon {horizon} not paired on {name}")
            if reference is None:
                reference = left
            elif not np.array_equal(
                reference["transition_indices"], left["transition_indices"]
            ):
                raise ValueError(f"seed {seed} horizon {horizon} rows differ")
            seed_gains.append(left["cumulative_bits"] - right["cumulative_bits"])
        gains[str(horizon)] = paired_episode_bootstrap(
            np.stack(seed_gains), reference["task_ids"], reference["episode_ids"],
            replicates=args.replicates, seed=args.seed,
        )
    degradation = {
        str(seed): float(curricula[seed]["one_step_relative_degradation"])
        for seed in (0, 1, 2)
    }
    horizon_pass = {
        horizon: gains[horizon]["micro"]["ci95"][0] > 0 for horizon in ("3", "7")
    }
    one_step_pass = max(degradation.values()) <= 0.01
    keep = all(horizon_pass.values()) and one_step_pass
    report = {
        "protocol": "v8_wm_curriculum_decision_v1",
        "cumulative_gain": gains,
        "horizon_gate": horizon_pass,
        "one_step_relative_degradation_by_seed": degradation,
        "one_step_gate_max_1pct": one_step_pass,
        "decision": "kept" if keep else "rejected",
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
