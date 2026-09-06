"""Generate leakage-safe local-state or transition QA from HumanTrajs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

# This host has a CUDA runtime but no nvcc toolchain. FlashInfer's sampler
# attempts a first-run JIT build; vLLM's native sampler needs no compiler.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

LOCAL_PROMPT = (
    "/no_think\nCreate ONE short QA pair whose answer is directly visible in the screenshot. "
    "Do not infer the user's goal or facts outside the image. Avoid generic questions. "
    "Return exactly this JSON object with all three fields: "
    '{"question":"...","answer":"...","evidence":"visible words or object supporting the answer"}.'
)
TRANSITION_PROMPT = (
    "/no_think\nImage 1 is before an action and Image 2 is after it. Create ONE short QA pair "
    "about a concrete, visible change caused by the action. Do not infer later outcomes. "
    "Return exactly this JSON object with all three fields: "
    '{"question":"...","answer":"...","evidence":"visible change supporting the answer"}.'
)


def parse_json(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    required = ("question", "answer", "evidence")
    return value if all(isinstance(value.get(k), str) and value[k].strip() for k in required) else None


def compact_context(row: dict) -> str:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else {}
    return (
        "Executed action: " + json.dumps(action, ensure_ascii=False) + "\n"
        "Current metadata: " + json.dumps(observation, ensure_ascii=False) + "\n"
        "Metadata is context only; the answer must be visually grounded."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="humantrajs-post-action-v1 manifest")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--mode", choices=("local", "transition"), default="local")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()

    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    with args.input.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [r for r in rows if args.split == "all" or r.get("split") == args.split]
    if args.mode == "transition":
        rows = [r for r in rows if r.get("transition_eligible") and
                r.get("previous_memory_step_idx") is not None]
    rows = rows[args.start:args.end or None]

    image_limit = 1 if args.mode == "local" else 2
    llm = LLM(model=str(args.model), dtype="auto", max_model_len=16384,
              max_num_seqs=args.batch_size, limit_mm_per_prompt={"image": image_limit},
              mm_processor_kwargs={"max_pixels": 1280 * 1024},
              structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True},
              enforce_eager=True)
    schema = {"type": "object", "properties": {
        "question": {"type": "string"}, "answer": {"type": "string"},
        "evidence": {"type": "string"}},
        "required": ["question", "answer", "evidence"], "additionalProperties": False}
    sampling = SamplingParams(temperature=0, max_tokens=args.max_new_tokens,
        structured_outputs=StructuredOutputsParams(json=schema))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.output.open("w", encoding="utf-8") as output, \
            args.audit.open("w", encoding="utf-8") as audit:
        for base in range(0, len(rows), args.batch_size):
            batch = rows[base:base + args.batch_size]
            requests = []
            for row in batch:
                if args.mode == "local":
                    images = [Image.open(row["image_path"]).convert("RGB")]
                    vision, prompt = "<|vision_start|><|image_pad|><|vision_end|>", LOCAL_PROMPT
                else:
                    images = [Image.open(row["before_image_path"]).convert("RGB"),
                              Image.open(row["image_path"]).convert("RGB")]
                    vision = ("Image 1:<|vision_start|><|image_pad|><|vision_end|>\n"
                              "Image 2:<|vision_start|><|image_pad|><|vision_end|>")
                    prompt = TRANSITION_PROMPT
                requests.append({"prompt": "<|im_start|>user\n" + vision + "\n" + prompt + "\n" +
                                 compact_context(row) + "<|im_end|>\n<|im_start|>assistant\n",
                                 "multi_modal_data": {"image": images[0] if len(images) == 1 else images}})
            results = llm.generate(requests, sampling, use_tqdm=False)
            for row, result in zip(batch, results):
                raw = result.outputs[0].text
                qa = parse_json(raw)
                audit_record = {"trajectory_id": row.get("trajectory_id"),
                    "step_idx": row.get("step_idx"), "mode": args.mode,
                    "parsed": bool(qa), "model_output": raw}
                if qa:
                    record = dict(row)
                    record.update(qa)
                    record.update({"qa_mode": args.mode,
                        "qa_source": "qwen3.5-9b-generated-unverified",
                        "verification_status": "pending"})
                    if args.mode == "local":
                        groups = [row.get("equivalent_step_ids") or [row["step_idx"]]]
                    else:
                        groups = [[row["previous_memory_step_idx"]],
                                  row.get("equivalent_step_ids") or [row["step_idx"]]]
                    record["gold_step_groups"] = groups
                    record["gold_step_ids"] = sorted({x for group in groups for x in group})
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    audit_record.update({k: qa[k] for k in ("question", "answer", "evidence")})
                    kept += 1
                audit.write(json.dumps(audit_record, ensure_ascii=False) + "\n")
            output.flush(); audit.flush()
            print(f"processed={min(base + args.batch_size, len(rows))} generated={kept}", flush=True)
    print(json.dumps({"processed": len(rows), "generated": kept, "mode": args.mode,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
