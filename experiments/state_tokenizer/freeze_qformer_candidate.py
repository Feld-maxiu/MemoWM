"""Validate and freeze one K32/QK-norm Q-Former candidate artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from residualmem.latent.qformer import QFORMER_PROTOCOL


SUPPORTED_PAIRS_PROTOCOLS = {
    "humantrajs_qformer_qa_v1",
    "molmoweb_text_qformer_qa_v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--p4-report", required=True)
    parser.add_argument("--qa-controls")
    parser.add_argument("--exploratory", action="store_true",
                        help="materialize a clearly marked artifact despite a failed "
                             "formal gate; never sets accepted=true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-p4-observations", type=int, default=48)
    parser.add_argument("--min-matched-beats-mismatched", type=float, default=0.90)
    parser.add_argument("--min-qa-observations", type=int, default=96)
    parser.add_argument("--max-skipped-steps", type=int, default=0)
    args = parser.parse_args()
    if not args.exploratory and not args.qa_controls:
        parser.error("formal freeze requires --qa-controls")

    source = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("frozen candidate output already exists; refusing overwrite")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("protocol") != QFORMER_PROTOCOL:
        raise ValueError("source checkpoint protocol mismatch")
    metadata = dict(payload.get("metadata") or {})
    required_metadata = {
        "queries": 32,
        "layer": 16,
        "qk_norm": True,
        "self_attention": False,
    }
    for name, expected in required_metadata.items():
        if metadata.get(name) != expected:
            raise ValueError(f"checkpoint {name}={metadata.get(name)!r}; expected {expected!r}")
    if metadata.get("pairs_protocol") not in SUPPORTED_PAIRS_PROTOCOLS:
        raise ValueError(
            f"checkpoint pairs_protocol={metadata.get('pairs_protocol')!r}; "
            f"expected one of {sorted(SUPPORTED_PAIRS_PROTOCOLS)}"
        )
    source_hash = _sha256(source)

    training_path = Path(args.training_report).resolve()
    p4_path = Path(args.p4_report).resolve()
    qa_path = Path(args.qa_controls).resolve() if args.qa_controls else None
    training, p4 = map(_read_json, (training_path, p4_path))
    qa = _read_json(qa_path) if qa_path else None
    if training.get("qk_norm") is not True or training.get("self_attention") is not False:
        raise ValueError("training report architecture does not match K32/QK-norm/no-self-attn")
    if training.get("queries") != 32 or training.get("pairs_protocol") != metadata["pairs_protocol"]:
        raise ValueError("training report query count or pairs protocol mismatch")
    if training.get("drop_microbatches") is not False:
        raise ValueError("formal run unexpectedly enabled micro-batch dropping")
    if int(training.get("skipped_steps", -1)) > args.max_skipped_steps:
        raise ValueError("formal run exceeded the registered skipped-step allowance")
    if int(training.get("dropped_microbatches", -1)) != 0:
        raise ValueError("formal run dropped micro-batches")
    objective = str(training.get("objective", ""))
    for term in ("answer_ce", "distill_kl[observation]", "sem[same-session]",
                 "observation_kl"):
        if term not in objective:
            raise ValueError(f"training report objective lacks {term}")

    if Path(p4.get("checkpoint", "")).resolve() != source:
        raise ValueError("P4 report was not computed from this source checkpoint")
    if int(p4.get("observations", 0)) < args.min_p4_observations:
        raise ValueError("P4 report has too few observations")
    if float(p4.get("gap", 0.0)) <= 0:
        raise ValueError("P4 mismatched-minus-matched gap is not positive")
    p4_gate_passed = (
        float(p4.get("matched_beats_mismatched", 0.0))
        >= args.min_matched_beats_mismatched
    )
    if not args.exploratory and not p4_gate_passed:
        raise ValueError("P4 matched-beats-mismatched is below 0.90")

    matched = shuffled = zero = None
    qa_gate_passed = False
    if qa is not None:
        if qa.get("checkpoint_sha256") != source_hash:
            raise ValueError("QA-control report checkpoint hash mismatch")
        if int(qa.get("observations", 0)) < args.min_qa_observations:
            raise ValueError("QA-control report has too few observations")
        matched = float(qa["matched_validation_answer_ce"])
        shuffled = float(qa["shuffled_validation_answer_ce"])
        zero = float(qa["zero_validation_answer_ce"])
        qa_gate_passed = (
            matched < shuffled and matched < zero
            and qa.get("write_side_question_independent") is True
        )
        if not args.exploratory and not qa_gate_passed:
            raise ValueError("QA controls did not pass matched-vs-shuffled/zero gates")
    formal_gate_passed = p4_gate_passed and qa_gate_passed

    state = payload.get("state_dict") or {}
    queries = state.get("qformer.queries")
    input_projection = state.get("qformer.input_projection.weight")
    output_projection = state.get("qformer.output_projection.weight")
    if queries is None or tuple(queries.shape) != (32, 1024):
        raise ValueError("source state does not encode 32 x 1024 learned queries")
    if input_projection is None or tuple(input_projection.shape) != (1024, 4096):
        raise ValueError("source QFormer input projection shape mismatch")
    if output_projection is None or tuple(output_projection.shape) != (512, 1024):
        raise ValueError("source QFormer output projection shape mismatch")
    blocks = sorted({key.split(".")[2] for key in state if key.startswith("qformer.blocks.")})
    if blocks != ["0", "1", "2", "3"]:
        raise ValueError(f"source QFormer block set is {blocks}")

    selected_step = int(metadata.get("best_step", -1))
    history_row = next((row for row in training.get("history", [])
                        if int(row.get("step", -2)) == selected_step), None)
    if history_row is None:
        raise ValueError("selected checkpoint step is absent from training history")
    canonical = {
        **metadata,
        "objective": objective,
        "qformer_hidden": 1024,
        "qformer_heads": 8,
        "qformer_layers": 4,
        "input_dimension": 4096,
        "output_dimension": 512,
        "modalities": 4,
        "connector_slots": 32,
        "frozen_candidate": True,
        "formal_gate_passed": formal_gate_passed,
        "exploratory_only": bool(args.exploratory),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_hash,
        "training_report_sha256": _sha256(training_path),
        "p4_report_sha256": _sha256(p4_path),
        "qa_controls_sha256": (_sha256(qa_path) if qa_path else None),
        "external_p4": {
            "observations": int(p4["observations"]),
            "gap": float(p4["gap"]),
            "matched_beats_mismatched": float(p4["matched_beats_mismatched"]),
        },
        "qa_controls": ({
            "observations": int(qa["observations"]),
            "matched_ce": matched,
            "shuffled_ce": shuffled,
            "zero_ce": zero,
            "question_only_ce": float(qa["question_only_validation_answer_ce"]),
        } if qa is not None else None),
        # Per the experiment manual this monitor is reported, not a hard gate.
        "collapse_monitor": history_row.get("monitors"),
        "collapse_regressed": bool(history_row.get("collapse_regressed")),
        "collapse_gate_policy": "reported_not_hard_gate_per_QFORMER_experiment_manual",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**payload, "metadata": canonical}, output)
    frozen_hash = _sha256(output)
    report = {
        "protocol": "frozen_qformer_candidate_v1",
        "accepted": formal_gate_passed,
        "exploratory_only": bool(args.exploratory),
        "artifact": str(output),
        "artifact_sha256": frozen_hash,
        "source_sha256": source_hash,
        "selected_by": metadata.get("selected_by"),
        "selected_step": selected_step,
        "validation_answer_ce": metadata.get("validation_answer_ce"),
        "p4": canonical["external_p4"],
        "qa_controls": canonical["qa_controls"],
        "collapse_regressed": canonical["collapse_regressed"],
        "collapse_gate_policy": canonical["collapse_gate_policy"],
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True),
                                           encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
