"""Merge sharded local LongMemEval Web Small QA results."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def shared_generation_config(rows: list[dict]):
    configs = {json.dumps(row.get("generation"), sort_keys=True) for row in rows}
    if len(configs) > 1:
        raise ValueError("refusing to merge different or missing reader generation configurations")
    return json.loads(next(iter(configs))) if configs else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--anchor", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    expected = [
        row for row in _jsonl(args.data_root / "questions.jsonl")
        if row.get("domain") == "web"
    ]
    order = {str(row["id"]): position for position, row in enumerate(expected)}
    rows = [row for path in args.input for row in _jsonl(path)]
    generation = shared_generation_config(rows)
    memories = {json.dumps(row.get('memory_config'), sort_keys=True) for row in rows}
    if len(memories) > 1:
        raise ValueError('refusing to merge different memory configurations')
    memory = json.loads(next(iter(memories))) if memories else None
    counts = collections.Counter(str(row.get("question_id")) for row in rows)
    duplicates = sorted(qid for qid, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate question ids: {duplicates[:5]}")
    unexpected = sorted(set(counts) - set(order))
    missing = sorted(set(order) - set(counts))
    if unexpected or missing:
        raise ValueError(
            f"question coverage mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected"
        )
    rows.sort(key=lambda row: order[str(row["question_id"])])
    scored = [row for row in rows if isinstance(row.get("score"), bool)]
    by_type: dict[str, dict[str, float | int | None]] = {}
    for question_type in sorted({str(row.get("question_type")) for row in rows}):
        subset = [row for row in scored if str(row.get("question_type")) == question_type]
        by_type[question_type] = {
            "questions": sum(str(row.get("question_type")) == question_type for row in rows),
            "scored_questions": len(subset),
            "accuracy": float(np.mean([row["score"] for row in subset])) if subset else None,
        }
    summary = {
        "protocol": "longmemeval-local-run-v2" if generation is not None else "longmemeval-local-run-v1",
        "domain": "web",
        "tier": "small",
        "questions": len(rows),
        "scored_questions": len(scored),
        "score_errors": len(rows) - len(scored),
        "accuracy": float(np.mean([row["score"] for row in scored])) if scored else None,
        "anchor_enabled": bool(args.anchor),
        "cache": str(args.cache.resolve()),
        "inputs": [str(path.resolve()) for path in args.input],
        "output": str(args.output.resolve()),
        "by_question_type": by_type,
    }
    if generation is not None:
        summary["generation"] = generation
    if memory is not None:
        summary['memory'] = memory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
