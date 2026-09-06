"""Independently verify QA using only its declared gold screenshots."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from xt_ama_adapter.humantrajs import normalized_qa_key


def evidence_paths(row: dict) -> list[str]:
    mode = row.get("qa_mode")
    if mode == "local":
        return [row["image_path"]]
    if mode == "transition":
        return [row["before_image_path"], row["image_path"]]
    if mode == "multistep":
        return [frame["image_path"] for frame in row.get("evidence_frames", [])]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="verified, kept QA")
    parser.add_argument("--audit", type=Path, required=True, help="all verifier decisions")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-score", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()
    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    with args.input.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = rows[args.start:args.end or None]
    max_images = max((len(evidence_paths(row)) for row in rows), default=1)
    schema = {"type": "object", "properties": {
        "keep": {"type": "boolean"}, "answer_visible": {"type": "boolean"},
        "gold_steps_sufficient": {"type": "boolean"},
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "evidence": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["keep", "answer_visible", "gold_steps_sufficient", "score", "evidence", "reason"],
        "additionalProperties": False}
    llm = LLM(model=str(args.model), dtype="auto", max_model_len=16384,
              max_num_seqs=args.batch_size, limit_mm_per_prompt={"image": max_images},
              mm_processor_kwargs={"max_pixels": 1280 * 1024},
              structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True},
              enforce_eager=True)
    sampling = SamplingParams(temperature=0, max_tokens=384,
        structured_outputs=StructuredOutputsParams(json=schema))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    seen_qa: set[tuple[str, str]] = set()
    with args.output.open("w", encoding="utf-8") as output, \
            args.audit.open("w", encoding="utf-8") as audit:
        for base in range(0, len(rows), args.batch_size):
            batch = rows[base:base + args.batch_size]
            requests = []
            for row in batch:
                paths = evidence_paths(row)
                images = [Image.open(path).convert("RGB") for path in paths]
                vision = "\n".join(f"Evidence image {i + 1}:<|vision_start|><|image_pad|><|vision_end|>"
                                   for i in range(len(images)))
                prompt = ("<|im_start|>user\n" + vision +
                    "\n/no_think\nAudit the proposed QA using ONLY the evidence images. Reject if the answer "
                    "is not visibly supported, if required evidence is missing, or if the question is ambiguous.\n"
                    f"Question: {row.get('question')}\nProposed answer: {row.get('answer')}"
                    '\nReturn exactly: {"keep":true,"answer_visible":true,"gold_steps_sufficient":true,'
                    '"score":5,"evidence":"...","reason":"..."}'
                    "<|im_end|>\n<|im_start|>assistant\n")
                requests.append({"prompt": prompt,
                    "multi_modal_data": {"image": images[0] if len(images) == 1 else images}})
            results = llm.generate(requests, sampling, use_tqdm=False)
            for row, result in zip(batch, results):
                raw = result.outputs[0].text
                try:
                    verdict = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                except (ValueError, json.JSONDecodeError):
                    verdict = None
                accepted = bool(verdict and verdict.get("keep") and verdict.get("answer_visible") and
                                verdict.get("gold_steps_sufficient") and
                                int(verdict.get("score", 0)) >= args.min_score)
                rejection_reason = None
                qa_key = normalized_qa_key(row.get("question", ""), row.get("answer", ""))
                if accepted and qa_key in seen_qa:
                    accepted = False
                    rejection_reason = "duplicate_verified_qa"
                if accepted:
                    seen_qa.add(qa_key)
                    record = dict(row); record["verification_status"] = "verified"
                    record["verifier"] = verdict
                    output.write(json.dumps(record, ensure_ascii=False) + "\n"); kept += 1
                audit.write(json.dumps({"trajectory_id": row.get("trajectory_id"),
                    "step_idx": row.get("step_idx"), "qa_mode": row.get("qa_mode"),
                    "accepted": accepted, "rejection_reason": rejection_reason,
                    "verdict": verdict, "model_output": raw},
                    ensure_ascii=False) + "\n")
            output.flush(); audit.flush()
            print(f"processed={min(base + args.batch_size, len(rows))} kept={kept}", flush=True)
    print(json.dumps({"processed": len(rows), "kept": kept, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
