"""Keep screenshot QA only when its answer is supported by pseudo web text."""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def evidence_is_present(evidence: str, observation: str) -> bool:
    evidence = normalize_text(evidence).strip(" .,:;!?\"'`[]()")
    return len(evidence) >= 2 and evidence in normalize_text(observation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="screenshot-verified QA JSONL")
    parser.add_argument("--manifest", type=Path, required=True, help="materialized pseudo-web manifest")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-score", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    observations: dict[tuple[str, int], dict] = {}
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                observations[(row["trajectory_id"], int(row["step_idx"]))] = row

    with args.input.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = rows[args.start:args.end or None]

    schema = {
        "type": "object",
        "properties": {
            "supported": {"type": "boolean"},
            "answer_entailment": {"type": "boolean"},
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "evidence_quote": {"type": "string", "maxLength": 400},
            "reason": {"type": "string", "maxLength": 500},
        },
        "required": ["supported", "answer_entailment", "score", "evidence_quote", "reason"],
        "additionalProperties": False,
    }
    llm = LLM(
        model=str(args.model), dtype="auto", max_model_len=8192,
        max_num_seqs=args.batch_size,
        language_model_only=True,
        structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True},
        enforce_eager=True,
    )
    sampling = SamplingParams(
        temperature=0, max_tokens=384,
        structured_outputs=StructuredOutputsParams(json=schema),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.output.open("w", encoding="utf-8") as output, \
            args.audit.open("w", encoding="utf-8") as audit:
        for base in range(0, len(rows), args.batch_size):
            batch = rows[base:base + args.batch_size]
            pending: list[tuple[dict, dict]] = []
            for row in batch:
                source = observations.get((row.get("trajectory_id"), int(row.get("step_idx", -1))))
                fixed_reason = None
                if source is None:
                    fixed_reason = "missing_pseudo_observation"
                elif source.get("cross_split_image_overlap") and row.get("split") != "train":
                    fixed_reason = "cross_split_image_overlap_in_eval"
                if fixed_reason:
                    audit.write(json.dumps({
                        "trajectory_id": row.get("trajectory_id"), "step_idx": row.get("step_idx"),
                        "split": row.get("split"), "accepted": False,
                        "rejection_reason": fixed_reason, "verdict": None,
                    }, ensure_ascii=False) + "\n")
                else:
                    pending.append((row, source))

            requests = []
            for row, source in pending:
                prompt = (
                    "<|im_start|>user\n/no_think\n"
                    "Judge the proposed answer using ONLY the web observation below. Do not use outside "
                    "knowledge or infer omitted visual content. `evidence_quote` must be a short exact "
                    "contiguous quote copied from the observation. Reject ambiguous questions and answers "
                    "that are plausible but not explicitly supported. Score direction is mandatory: "
                    "5 means the answer is fully and explicitly supported; 4 means clearly supported "
                    "with only harmless paraphrasing; 1-3 means reject. If supported=true and "
                    "answer_entailment=true, use score 4 or 5, never 1.\n\n"
                    f"WEB OBSERVATION:\n{source['text_observation']}\n\n"
                    f"QUESTION: {row.get('question', '')}\n"
                    f"PROPOSED ANSWER: {row.get('answer', '')}\n"
                    "Return the required JSON only.<|im_end|>\n<|im_start|>assistant\n"
                )
                requests.append(prompt)

            results = llm.generate(requests, sampling, use_tqdm=False) if requests else []
            for (row, source), result in zip(pending, results):
                raw = result.outputs[0].text
                try:
                    verdict = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                except (ValueError, json.JSONDecodeError):
                    verdict = None
                quote_present = bool(verdict and evidence_is_present(
                    verdict.get("evidence_quote", ""), source["text_observation"]))
                accepted = bool(
                    verdict and verdict.get("supported") and verdict.get("answer_entailment")
                    and int(verdict.get("score", 0)) >= args.min_score and quote_present
                )
                reason = None
                if verdict is None:
                    reason = "invalid_verifier_json"
                elif not quote_present:
                    reason = "evidence_quote_not_in_observation"
                elif not accepted:
                    reason = "not_supported_by_pseudo_observation"
                if accepted:
                    record = dict(row)
                    record["qa_input_protocol"] = "vl-pseudo-web-observation-v1"
                    record["text_observation"] = source["text_observation"]
                    record["pseudo_text_verification"] = verdict
                    record["verification_status"] = "pseudo_web_text_verified"
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1
                audit.write(json.dumps({
                    "trajectory_id": row.get("trajectory_id"), "step_idx": row.get("step_idx"),
                    "split": row.get("split"), "accepted": accepted,
                    "rejection_reason": reason, "evidence_quote_present": quote_present,
                    "verdict": verdict, "model_output": raw,
                }, ensure_ascii=False) + "\n")
            output.flush()
            audit.flush()
            print(f"processed={min(base + args.batch_size, len(rows))} kept={kept}", flush=True)
    print(json.dumps({"processed": len(rows), "kept": kept, "output": str(args.output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
