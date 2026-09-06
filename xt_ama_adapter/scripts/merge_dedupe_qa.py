"""Merge generated QA shards and remove exact normalized pair duplicates."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from xt_ama_adapter.humantrajs import normalized_qa_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duplicates", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--audits", type=Path, nargs="*",
                        help="optional verifier audit shards for rejection statistics")
    parser.add_argument("--strip-instruction", action="store_true",
                        help="remove task instruction from the final training artifact")
    args = parser.parse_args()

    rows = []
    for shard_index, path in enumerate(args.inputs):
        with path.open(encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                if line.strip():
                    row = json.loads(line)
                    row["_merge_order"] = (shard_index, line_index)
                    rows.append(row)
    # If an identical QA appears across splits, retain train before validation
    # before test. This prevents a generic repeated question leaking into eval.
    priority = {"train": 0, "validation": 1, "test": 2}
    rows.sort(key=lambda row: (priority.get(row.get("split"), 9), row["_merge_order"]))
    kept, duplicates, seen = [], [], {}
    for row in rows:
        key = normalized_qa_key(row.get("question", ""), row.get("answer", ""))
        row.pop("_merge_order", None)
        if key in seen:
            duplicates.append({
                "reason": "duplicate_normalized_qa_pair",
                "trajectory_id": row.get("trajectory_id"), "step_idx": row.get("step_idx"),
                "split": row.get("split"), "question": row.get("question"), "answer": row.get("answer"),
                "retained_trajectory_id": seen[key].get("trajectory_id"),
                "retained_step_idx": seen[key].get("step_idx"),
                "retained_split": seen[key].get("split"),
            })
        else:
            if args.strip_instruction:
                row.pop("instruction", None)
                row["instruction_excluded_from_training"] = True
            seen[key] = row
            kept.append(row)

    for path, values in ((args.output, kept), (args.duplicates, duplicates)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    split_trajectories = {}
    image_splits = {}
    for row in kept:
        split_trajectories.setdefault(row.get("split", "unknown"), set()).add(
            row.get("trajectory_id"))
        image_splits.setdefault(row.get("image_sha256"), set()).add(row.get("split"))
    report = {"input_rows": len(rows), "kept_rows": len(kept),
              "duplicate_rows": len(duplicates),
              "kept_by_split": dict(Counter(row.get("split", "unknown") for row in kept)),
              "duplicates_by_split": dict(Counter(row.get("split", "unknown") for row in duplicates)),
              "unique_trajectories_by_split": {
                  split: len(values) for split, values in split_trajectories.items()},
              "cross_split_image_hashes_in_kept": sum(len(splits) > 1
                                                      for splits in image_splits.values())}
    if args.audits:
        audit_rows = []
        for path in args.audits:
            with path.open(encoding="utf-8") as handle:
                audit_rows.extend(json.loads(line) for line in handle if line.strip())
        report["verifier_audit"] = {
            "processed": len(audit_rows),
            "accepted": sum(bool(row.get("accepted")) for row in audit_rows),
            "rejected": sum(not row.get("accepted") for row in audit_rows),
            "acceptance_rate": (sum(bool(row.get("accepted")) for row in audit_rows)
                                / len(audit_rows) if audit_rows else 0),
            "rejection_reasons": dict(Counter(
                row.get("rejection_reason", "unknown") for row in audit_rows
                if not row.get("accepted"))),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
