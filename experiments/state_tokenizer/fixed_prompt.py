"""Build the leakage-controlled subset with one task-independent instruction."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .common import iter_jsonl, sha256_file, write_json


OBSERVATION_PROMPT = (
    "请忠实表示当前页面状态，保留可见文本、输入值、控件类型、"
    "选中/聚焦/启用状态及空间关系。"
)


def build_fixed_prompt_records(records: list[dict]) -> list[dict]:
    output = []
    for raw in records:
        record = copy.deepcopy(raw)
        record["instruction"] = OBSERVATION_PROMPT
        record["instruction_protocol"] = "fixed_task_independent_observation_v1"
        output.append(record)
    return output


def has_random_five_digit_value(record: dict) -> bool:
    digits = record.get("v2", {}).get("random_digits", [])
    return len(digits) == 5 and all(isinstance(value, int) and 0 <= value <= 9 for value in digits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--random-value-only", action="store_true")
    args = parser.parse_args()
    source_path = Path(args.records)
    source = list(iter_jsonl(source_path))
    records = build_fixed_prompt_records(source)
    if args.random_value_only:
        records = [record for record in records if has_random_five_digit_value(record)]
        if not records:
            raise ValueError("random-value-only subset is empty")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    source_summary = json.loads(source_path.with_suffix(".summary.json").read_text())
    summary = dict(source_summary)
    summary.update({
        "source_subset": str(source_path.resolve()),
        "source_subset_sha256": sha256_file(source_path),
        "subset_sha256": sha256_file(output),
        "instruction_protocol": "fixed_task_independent_observation_v1",
        "observation_prompt": OBSERVATION_PROMPT,
        "unique_instructions": len({record["instruction"] for record in records}),
        "random_value_only": args.random_value_only,
        "records": len(records),
        "split_counts": {
            split: sum(record["split"] == split for record in records)
            for split in ("train", "validation", "test")
        },
        "unique_episodes": len({record["episode_id"] for record in records}),
        "task_counts": {
            task: sum(record["task"] == task for record in records)
            for task in sorted({record["task"] for record in records})
        },
    })
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
