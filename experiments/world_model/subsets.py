"""Build deterministic, task-stratified, episode-complete nested train subsets."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.common import sha256_file, write_json

from .cache import CACHE_FILES, FrozenCache
from .schema import SPLIT_IDS


def _rank(name: str, salt: str) -> bytes:
    return hashlib.sha256((salt + "\0" + name).encode("utf-8")).digest()


def build_subsets(
    cache: FrozenCache,
    targets: list[int],
    *,
    salt: str = "residualmem-v8-wm-subsets-v1",
) -> tuple[dict[str, np.ndarray], dict]:
    train_rows = np.flatnonzero(
        cache.transitions["split_ids"] == SPLIT_IDS["train"]
    )
    full_count = len(train_rows)
    requested = sorted(set(int(value) for value in targets) | {full_count})
    if not requested or requested[0] < 1 or requested[-1] > full_count:
        raise ValueError(f"subset targets must lie in 1..{full_count}: {requested}")

    episode_rows: dict[int, list[int]] = defaultdict(list)
    episode_task: dict[int, int] = {}
    for row in train_rows:
        episode = int(cache.transitions["episode_ids"][row])
        task = int(cache.transitions["task_ids"][row])
        if episode in episode_task and episode_task[episode] != task:
            raise ValueError(f"episode {episode} spans tasks")
        episode_task[episode] = task
        episode_rows[episode].append(int(row))

    by_task: dict[int, list[int]] = defaultdict(list)
    for episode, task in episode_task.items():
        by_task[task].append(episode)
    for task, episodes in by_task.items():
        episodes.sort(key=lambda ep: _rank(cache.episode_names[ep], salt))

    task_totals = {
        task: sum(len(episode_rows[episode]) for episode in episodes)
        for task, episodes in by_task.items()
    }
    arrays: dict[str, np.ndarray] = {}
    summaries = {}
    previous: set[int] = set()
    for target in requested:
        if target == full_count:
            selected = set(int(row) for row in train_rows)
        else:
            selected: set[int] = set()
            for task in sorted(by_task):
                quota = int(round(target * task_totals[task] / full_count))
                accumulated = 0
                for episode in by_task[task]:
                    if accumulated >= quota:
                        break
                    selected.update(episode_rows[episode])
                    accumulated += len(episode_rows[episode])
        if not previous.issubset(selected):
            raise AssertionError("subset construction is not nested")
        previous = selected
        name = f"train_{target}"
        rows = np.asarray(sorted(selected), np.int64)
        arrays[name] = rows
        per_task = {
            cache.task_names[task]: int(np.sum(cache.transitions["task_ids"][rows] == task))
            for task in sorted(by_task)
        }
        summaries[name] = {
            "requested_transitions": target,
            "actual_transitions": len(rows),
            "episodes": int(len(set(cache.transitions["episode_ids"][rows].tolist()))),
            "per_task": per_task,
        }
    return arrays, {
        "protocol": "v8_episode_complete_nested_subsets_v1",
        "salt": salt,
        "full_train_transitions": full_count,
        "subsets": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--targets", nargs="+", type=int, default=[10_000, 30_000])
    parser.add_argument("--salt", default="residualmem-v8-wm-subsets-v1")
    args = parser.parse_args()

    cache = FrozenCache(args.cache)
    arrays, manifest = build_subsets(cache, args.targets, salt=args.salt)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "subsets.npz"
    temporary = archive.with_name(archive.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, archive)
    manifest.update({
        "cache_manifest": str((Path(args.cache) / CACHE_FILES["manifest"]).resolve()),
        "cache_manifest_sha256": sha256_file(Path(args.cache) / CACHE_FILES["manifest"]),
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
    })
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest["subsets"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
