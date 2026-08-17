"""Build the 7:2:1 task-stratified split and the incremental extraction manifest.

Two constraints drive the design.

**The frozen PCA/normalizer must not leak.** They were fitted on the original
train split, so every state that was train under the 6:2:2 split has to stay
train under 7:2:1. The stratification therefore starts from the old train set and
only ever promotes held-out episodes into train, never the reverse.

**Bucketing on ``episode_index`` cannot be used.** Collection cycled tasks with
period 4, so each task's episodes land exclusively on even or exclusively on odd
indices; any split taking an odd number of ``% 10`` buckets loses half the tasks
(a single-bucket test split covers 6 of 12). Splitting per task instead gives
every split all 12 tasks at the exact target ratio.

Emits:

* ``full-721.jsonl`` -- all states, new ``split``, final ``global_index``
  (already-extracted states keep the index their features are stored under).
* ``extend-<n>.jsonl`` -- only the not-yet-extracted states, re-indexed from 0 so
  the extraction chain can run self-contained; ``final_global_index`` carries the
  value the shards are remapped to afterwards.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from .common import iter_jsonl, write_json
from .fixed_prompt import OBSERVATION_PROMPT, build_fixed_prompt_records

SPLITS = ("train", "validation", "test")


def stratified_split(records: list[dict], ratios: dict[str, float]) -> dict[str, str]:
    """Episode -> split, preserving old-train membership and per-task ratios."""
    episodes: dict[str, dict] = collections.OrderedDict()
    for record in records:
        key = record["episode_id"]
        if key not in episodes:
            episodes[key] = {
                "task": record["task"],
                "old": record["split"],
                "order": (record["task"], int(record["episode_index"]), key),
                "states": 0,
            }
        episodes[key]["states"] += 1

    total = len(records)
    targets = {name: round(total * ratios[name]) for name in SPLITS}
    assignment = {key: "train" for key, value in episodes.items() if value["old"] == "train"}
    already = sum(value["states"] for key, value in episodes.items() if key in assignment)
    shortfall = targets["train"] - already
    if shortfall < 0:
        raise ValueError(
            f"old train already holds {already} states, above the {targets['train']} target; "
            "shrinking train would move fitted states into held-out data"
        )

    pool: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for key, value in sorted(episodes.items(), key=lambda item: item[1]["order"]):
        if value["old"] != "train":
            pool[value["task"]].append((key, value["states"]))
    available = sum(states for entries in pool.values() for _, states in entries)
    promote = shortfall / max(available, 1)

    held_out_ratio = ratios["validation"] / max(ratios["validation"] + ratios["test"], 1e-12)
    for entries in pool.values():
        quota = round(sum(states for _, states in entries) * promote)
        taken = index = 0
        while index < len(entries) and taken < quota:
            assignment[entries[index][0]] = "train"
            taken += entries[index][1]
            index += 1
        remainder = entries[index:]
        validation_quota = round(sum(states for _, states in remainder) * held_out_ratio)
        taken = index = 0
        while index < len(remainder) and taken < validation_quota:
            assignment[remainder[index][0]] = "validation"
            taken += remainder[index][1]
            index += 1
        for key, _ in remainder[index:]:
            assignment[key] = "test"
    return assignment


def summarize(records: list[dict], assignment: dict[str, str]) -> dict:
    counts = collections.Counter()
    tasks = collections.defaultdict(set)
    per_task = collections.defaultdict(collections.Counter)
    for record in records:
        split = assignment[record["episode_id"]]
        counts[split] += 1
        tasks[split].add(record["task"])
        per_task[record["task"]][split] += 1
    total = len(records)
    leaked = [
        record for record in records
        if record["split"] == "train" and assignment[record["episode_id"]] != "train"
    ]
    spanning = collections.defaultdict(set)
    for record in records:
        spanning[record["episode_id"]].add(assignment[record["episode_id"]])
    return {
        "total": total,
        "counts": {name: counts[name] for name in SPLITS},
        "fractions": {name: counts[name] / total for name in SPLITS},
        "tasks_per_split": {name: len(tasks[name]) for name in SPLITS},
        "old_train_leaked_to_held_out": len(leaked),
        "episodes_spanning_splits": sum(1 for value in spanning.values() if len(value) > 1),
        "per_task_fractions": {
            task: {name: per_task[task][name] / sum(per_task[task].values()) for name in SPLITS}
            for task in sorted(per_task)
        },
    }


def run(args: argparse.Namespace) -> dict:
    # The static key64 protocol is leakage-controlled: every state is encoded
    # under one task-independent observation prompt, so the manifest must carry
    # that instruction, not the per-task one from collection. extract_fixed_prompt
    # rejects a manifest with more than one distinct instruction.
    records = build_fixed_prompt_records(list(iter_jsonl(args.records)))
    # No ``--extracted`` means a fresh line with nothing encoded yet: every state
    # is pending and the index-preservation check below is vacuously satisfied.
    # v7 needed it because its features already existed under fixed indices.
    extracted = (
        {row["state_id"]: int(row["global_index"]) for row in iter_jsonl(args.extracted)}
        if args.extracted else {}
    )
    ratios = {"train": args.train_ratio, "validation": args.validation_ratio}
    ratios["test"] = 1.0 - ratios["train"] - ratios["validation"]
    if min(ratios.values()) <= 0:
        raise ValueError(f"ratios must be positive: {ratios}")

    assignment = stratified_split(records, ratios)
    report = summarize(records, assignment)
    if report["old_train_leaked_to_held_out"]:
        raise ValueError("split would move PCA-fitted train states into held-out data")
    if report["episodes_spanning_splits"]:
        raise ValueError("episodes must not span splits")

    # ``global_index`` is the row position in the merged manifest, which is what
    # the existing shards are keyed by. Pending states therefore take the free
    # positions rather than a fresh range: the index space stays 0..N-1 and no
    # already-extracted feature has to move.
    positions = {record["state_id"]: index for index, record in enumerate(records)}
    mismatched = [
        state for state, index in extracted.items() if positions.get(state) != index
    ]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} extracted states have a global_index that is not their "
            "merged-manifest position; the incremental index scheme does not apply"
        )

    full, pending = [], []
    for record in records:
        row = dict(record)
        row["split"] = assignment[record["episode_id"]]
        state = record["state_id"]
        row["global_index"] = positions[state]
        row["extracted"] = state in extracted
        if not row["extracted"]:
            local = dict(row)
            local["final_global_index"] = row["global_index"]
            local["global_index"] = len(pending)  # self-contained 0..n-1 for extraction
            pending.append(local)
        full.append(row)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    full_path = output / "full-721.jsonl"
    full_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in full), encoding="utf-8"
    )
    pending_path = output / f"extend-{len(pending)}.jsonl"
    pending_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in pending), encoding="utf-8"
    )
    report.update({
        "protocol": "split_721_task_stratified_v1",
        "instruction_protocol": "fixed_task_independent_observation_v1",
        "observation_prompt": OBSERVATION_PROMPT,
        "unique_instructions": len({row["instruction"] for row in full}),
        "source_records": str(Path(args.records).resolve()),
        "extracted_records": str(Path(args.extracted).resolve()) if args.extracted else None,
        "already_extracted": len(extracted),
        "pending_extraction": len(pending),
        "full_manifest": str(full_path.resolve()),
        "pending_manifest": str(pending_path.resolve()),
        "index_remap": {
            "scheme": "merged_manifest_position",
            "local_start": 0,
            "local_count": len(pending),
        },
    })
    write_json(output / "split-721.summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="merged collection records")
    parser.add_argument("--extracted",
                        help="records whose features already exist (keeps their "
                             "global_index). Omit for a fresh line with nothing "
                             "encoded yet, as in v8")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({
        "counts": report["counts"],
        "fractions": {k: round(v, 4) for k, v in report["fractions"].items()},
        "tasks_per_split": report["tasks_per_split"],
        "old_train_leaked_to_held_out": report["old_train_leaked_to_held_out"],
        "episodes_spanning_splits": report["episodes_spanning_splits"],
        "already_extracted": report["already_extracted"],
        "pending_extraction": report["pending_extraction"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
