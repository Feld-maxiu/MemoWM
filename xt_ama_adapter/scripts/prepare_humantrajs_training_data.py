"""Build an aligned, leakage-safe HumanTrajs manifest and QA audit sample."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from xt_ama_adapter.humantrajs import align_trajectories, file_sha256, read_jsonl


def inspect_image(path_text: str) -> dict:
    path = Path(path_text)
    if not path.is_file():
        return {"exists": False, "sha256": None, "low_information": True, "reason": "missing"}
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(path) as image:
            sample = image.convert("RGB")
            sample.thumbnail((64, 64))
            extrema = sample.getextrema()
            ranges = [high - low for low, high in extrema]
            low_information = max(ranges) <= 2
            return {
                "exists": True,
                "sha256": file_sha256(path),
                "low_information": low_information,
                "size_bytes": path.stat().st_size,
                "width": image.width,
                "height": image.height,
                "channel_ranges": ranges,
            }
    except (OSError, UnidentifiedImageError) as exc:
        return {"exists": True, "sha256": file_sha256(path), "low_information": True,
                "reason": f"unreadable:{type(exc).__name__}", "size_bytes": path.stat().st_size}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rejections", type=Path, required=True)
    parser.add_argument("--alignment-audit", type=Path, required=True)
    parser.add_argument("--audit-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()

    source = read_jsonl(args.input)
    prepared, rejected = align_trajectories(
        source, inspect_image, args.train_fraction, args.validation_fraction
    )
    write_jsonl(args.output, prepared)
    write_jsonl(args.rejections, rejected)

    rng = random.Random(args.seed)
    audit_pool = [row for row in prepared if row["transition_eligible"]]
    audit = rng.sample(audit_pool, min(args.audit_size, len(audit_pool)))
    audit = [{
        "trajectory_id": row["trajectory_id"],
        "step_idx": row["step_idx"],
        "action": row["action"],
        "observation": row["observation"],
        "before_image_path": row["before_image_path"],
        "post_action_image_path": row["image_path"],
        "review_action_effect_visible": None,
        "review_notes": None,
    } for row in audit]
    write_jsonl(args.alignment_audit, audit)

    trajectory_splits: dict[str, str] = {}
    split_steps = Counter()
    action_counts = Counter()
    duplicate_rows = 0
    for row in prepared:
        trajectory_splits[row["trajectory_id"]] = row["split"]
        split_steps[row["split"]] += 1
        action_counts[row["action"].get("action_name", "unknown")] += 1
        duplicate_rows += len(row["equivalent_step_ids"]) > 1
    split_trajectories = Counter(trajectory_splits.values())
    rejection_counts = Counter(row["reason"] for row in rejected)
    report = {
        "protocol": "humantrajs-post-action-v1",
        "source": str(args.input),
        "source_rows": len(source),
        "source_trajectories": len({str(row["trajectory_id"]) for row in source}),
        "kept_rows": len(prepared),
        "rejected_rows": len(rejected),
        "keep_rate": round(len(prepared) / len(source), 6) if source else 0,
        "alignment": "action_i_to_screenshot_i_plus_1",
        "instruction_used_for_qa": False,
        "split_steps": dict(sorted(split_steps.items())),
        "split_trajectories": dict(sorted(split_trajectories.items())),
        "action_counts": dict(action_counts.most_common()),
        "rejection_counts": dict(rejection_counts.most_common()),
        "rows_in_duplicate_state_groups": duplicate_rows,
        "transition_eligible_rows": sum(bool(row["transition_eligible"]) for row in prepared),
        "alignment_audit_rows": len(audit),
        "alignment_audit_status": "pending_manual_review",
        "qa_generation_status": "not_started",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
