"""Select a formally gated SyQA Q-Former candidate before Bridge training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-retrieval-z", type=float, default=1.96)
    args = parser.parse_args()

    audited = []
    for directory in sorted(path for path in args.candidates_dir.iterdir() if path.is_dir()):
        required = {
            "frozen": directory / "frozen.json",
            "qa": directory / "qa-controls.json",
            "p4": directory / "p4.json",
            "retrieval": directory / "head-recall.json",
            "artifact": directory / "frozen.pt",
            "head": directory / "head.pt",
            "cache": directory / "cache.npz",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            continue
        frozen = _read(required["frozen"])
        qa = _read(required["qa"])
        p4 = _read(required["p4"])
        retrieval = _read(required["retrieval"])
        within = retrieval["within_sample"]
        observed = float(within["recall_at_1"])
        chance = float(within["chance_recall_at_1"])
        # Use the number of independent groups rather than rows in this normal
        # approximation.  It is deliberately conservative for repeated states
        # within a canonical-URL group.
        groups = max(int(retrieval["samples"]), 1)
        standard_error = math.sqrt(max(chance * (1.0 - chance), 1e-12) / groups)
        z_score = (observed - chance) / standard_error
        retrieval_passed = observed > chance and z_score >= args.min_retrieval_z
        record = {
            "name": directory.name,
            "formal_freeze_accepted": frozen.get("accepted") is True,
            "matched_validation_answer_ce": float(
                qa["matched_validation_answer_ce"]
            ),
            "p4_gap": float(p4["gap"]),
            "p4_matched_beats_mismatched": float(
                p4["matched_beats_mismatched"]
            ),
            "retrieval_recall_at_1": observed,
            "retrieval_chance_recall_at_1": chance,
            "retrieval_independent_groups": groups,
            "retrieval_z_conservative": z_score,
            "retrieval_gate_passed": retrieval_passed,
            "artifact": str(required["artifact"].resolve()),
            "head": str(required["head"].resolve()),
            "cache": str(required["cache"].resolve()),
        }
        record["accepted"] = record["formal_freeze_accepted"] and retrieval_passed
        audited.append(record)

    if not audited:
        raise RuntimeError("no Q-Former candidate completed all audits")
    # User-authorized continuation policy: gates remain reported but do not
    # filter candidates.  Matched QA validation CE is the primary selection
    # metric; P4 and retrieval break exact CE ties.  Bridge results never feed
    # back into Q-Former selection.
    selected = min(audited, key=lambda row: (
        row["matched_validation_answer_ce"],
        -row["p4_gap"],
        -row["retrieval_recall_at_1"],
        row["name"],
    ))
    report = {
        "protocol": "syqa_qformer_bridge_candidate_selection_v1",
        "selection_policy": (
            "user_override_gate_filter_then_min_matched_validation_answer_ce_"
            "with_p4_gap_and_retrieval_r1_tiebreakers"
        ),
        "selected_requires_gate_pass": False,
        "selected_formal_and_retrieval_gates_passed": bool(selected["accepted"]),
        "candidates": audited,
        "selected": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
