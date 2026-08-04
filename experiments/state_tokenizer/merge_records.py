"""Merge collector shards into one stable manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import iter_jsonl, sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.input_dir)
    records = []
    for path in sorted(root.glob("records-worker*.jsonl")):
        records.extend(iter_jsonl(path))
    records.sort(key=lambda item: item["state_id"])
    seen = set()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    split_counts = {"train": 0, "validation": 0, "test": 0}
    task_counts: dict[str, int] = {}
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            state_id = record["state_id"]
            if state_id in seen:
                raise ValueError(f"duplicate state_id: {state_id}")
            seen.add(state_id)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            split_counts[record["split"]] += 1
            task_counts[record["task"]] = task_counts.get(record["task"], 0) + 1
    write_json(output.with_suffix(".summary.json"), {
        "records": len(records),
        "split_counts": split_counts,
        "task_counts": task_counts,
        "sha256": sha256_file(output),
    })


if __name__ == "__main__":
    main()
