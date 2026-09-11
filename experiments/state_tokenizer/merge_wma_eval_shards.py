"""Merge disjoint official WMA eval shards and recompute aggregate metrics."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _read(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _sample_number(row: dict) -> tuple[int, str]:
    sample_id = str(row.get("sample_id", ""))
    try:
        return int(sample_id.rsplit("_", 1)[1]), sample_id
    except (IndexError, ValueError):
        return 1 << 30, sample_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wma-root", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--baseline", default="ResidualMem-Instruct-Xbar-Input-RAG"
    )
    parser.add_argument("--expected-samples", type=int, default=27)
    parser.add_argument("--expected-qa", type=int, default=1459)
    args = parser.parse_args()

    wma_root = Path(args.wma_root).resolve()
    if str(wma_root) not in sys.path:
        sys.path.insert(0, str(wma_root))
    from eval_framework.evaluators.aggregate import aggregate_metrics

    session_rows: list[dict] = []
    qa_rows: list[dict] = []
    timings: list[dict] = []
    seen_samples: set[str] = set()
    for raw in args.input:
        root = Path(raw).resolve()
        sessions = _read(root / "session_records.jsonl")
        qa = _read(root / "qa_records.jsonl")
        shard_samples = {str(row["sample_id"]) for row in [*sessions, *qa]}
        overlap = seen_samples & shard_samples
        if overlap:
            raise ValueError(f"shards overlap on samples: {sorted(overlap)}")
        seen_samples |= shard_samples
        session_rows.extend(sessions)
        qa_rows.extend(qa)
        timings.append(json.loads((root / "aggregate_metrics.json").read_text())["timing"])

    if len(seen_samples) != args.expected_samples:
        raise ValueError(f"merged {len(seen_samples)} samples, expected {args.expected_samples}")
    if len(qa_rows) != args.expected_qa:
        raise ValueError(f"merged {len(qa_rows)} QA, expected {args.expected_qa}")
    for kind, rows in (("session", session_rows), ("QA", qa_rows)):
        failures = [row for row in rows if not isinstance(row.get("eval"), dict) or "error" in row["eval"]]
        if failures:
            raise ValueError(f"{len(failures)} {kind} rows have missing/failed evaluations")

    session_rows.sort(key=_sample_number)
    qa_rows.sort(key=_sample_number)
    aggregate = aggregate_metrics(
        args.baseline,
        session_evaluations=[row["eval"] for row in session_rows],
        qa_evaluations=[row["eval"] for row in qa_rows],
    )
    aggregate["timing"] = {
        "pipeline_seconds_sum": round(sum(float(t["pipeline_seconds"]) for t in timings), 2),
        "eval_seconds_sum": round(sum(float(t["eval_seconds"]) for t in timings), 2),
        "parallel_wall_seconds": round(max(float(t["total_seconds"]) for t in timings), 2),
        "merged_shards": len(timings),
    }
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "session_records.jsonl", session_rows)
    _write_jsonl(output / "qa_records.jsonl", qa_rows)
    temporary = output / f"aggregate_metrics.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, output / "aggregate_metrics.json")
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
