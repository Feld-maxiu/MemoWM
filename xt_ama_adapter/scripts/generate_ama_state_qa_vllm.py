"""Generate leakage-safe per-state QA pairs from AMA text observations.

Port of ``generate_step_qa_vllm.py`` (humantrajs screenshot protocol) to the
AMA domain: states are pure text (terminal / tool outputs, no screenshots),
so the "answerable from this frame" protocol is grounded in the observation
text instead of an image.

Fit distribution  = states with ``split == train``   (utility |U| fitting)
Eval distribution = states with ``split == validation`` (lambda curve quality)

The two batches are generated independently by the same prompt/sampler, so
they are same-distribution but disjoint. Official AMA-Bench test questions
are never touched (kept for the official end-to-end evaluation).
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

# This host has a CUDA runtime but no nvcc toolchain; keep vLLM off
# FlashInfer's first-run JIT sampler build (same as the humantrajs script).
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

LOCAL_PROMPT = (
    "/no_think\nBelow is ONE observation captured by a coding/terminal agent. "
    "Create 3 short QA pairs (2 to 4 of them) whose answers are DIRECTLY "
    "visible in this observation text.\n"
    "Rules:\n"
    "- Each answer must be literally present in, or a trivial reformatting of, "
    "spans of the observation text. Quote exact values.\n"
    "- Do NOT ask yes/no questions. Do NOT ask about the user's goal, the "
    "agent's next action, or facts requiring world knowledge.\n"
    "- Avoid generic questions (e.g. \"what is this text about\"). Ask about "
    "concrete values, names, paths, statuses, or counts shown in the text.\n"
    'Return exactly this JSON object: {"qa_pairs": [{"question": "...", '
    '"answer": "...", "evidence": "..."}]}'
)

SCHEMA = {
    "type": "object",
    "properties": {
        "qa_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["question", "answer", "evidence"],
                "additionalProperties": False,
            },
            "minItems": 2,
            "maxItems": 4,
        }
    },
    "required": ["qa_pairs"],
    "additionalProperties": False,
}

MIN_TEXT_CHARS = 120
MAX_TEXT_CHARS = 8_000  # p95 of the corpus is ~6k; long tails are truncated


def parse_payload(text: str) -> list[dict] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    pairs = value.get("qa_pairs") if isinstance(value, dict) else None
    if not isinstance(pairs, list):
        return None
    clean = []
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        fields = [pair.get(k) for k in ("question", "answer", "evidence")]
        if not all(isinstance(f, str) and f.strip() for f in fields):
            continue
        answer = fields[1].strip()
        lowered = answer.lower().strip(" .!")
        if lowered in {"yes", "no", "true", "false", "none", "null", "unknown"}:
            continue  # yes/no / negative answers are banned by the protocol
        clean.append({
            "question": fields[0].strip(),
            "answer": answer,
            "evidence": fields[2].strip(),
        })
    return clean or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path(
        "outputs/wm_train/ama-v1/states.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("/data1/models/Qwen3.5-9B"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--num-states", type=int, default=0,
                        help="sub-sample size; 0 = use every state in the split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if r.get("split") == args.split
            and len(r.get("text", "")) >= MIN_TEXT_CHARS]
    if args.num_states and args.num_states < len(rows):
        rows = sorted(random.Random(args.seed).sample(rows, args.num_states),
                      key=lambda r: r["state_id"])
    rows = rows[args.start:args.end or None]
    print(f"states selected for split={args.split}: {len(rows)}", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(model=str(args.model), dtype="auto",
              max_model_len=args.max_model_len,
              max_num_seqs=args.batch_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              structured_outputs_config={"backend": "xgrammar",
                                         "disable_any_whitespace": True},
              enforce_eager=True)
    sampling = SamplingParams(temperature=0, max_tokens=768,
                              structured_outputs=StructuredOutputsParams(json=SCHEMA))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    kept_states = 0
    kept_pairs = 0
    with args.output.open("w", encoding="utf-8") as output, \
            args.audit.open("w", encoding="utf-8") as audit:
        for base in range(0, len(rows), args.batch_size):
            batch = rows[base:base + args.batch_size]
            requests = [{
                "prompt": "<|im_start|>user\n" + LOCAL_PROMPT + "\n\nObservation:\n"
                          + row["text"][:MAX_TEXT_CHARS] + "<|im_end|>\n"
                          "<|im_start|>assistant\n",
            } for row in batch]
            results = llm.generate(requests, sampling, use_tqdm=False)
            for row, result in zip(batch, results):
                raw = result.outputs[0].text
                pairs = parse_payload(raw)
                audit.write(json.dumps({
                    "state_id": row["state_id"], "split": row["split"],
                    "parsed": pairs is not None,
                    "n_pairs": 0 if pairs is None else len(pairs),
                    "text_chars": len(row["text"]),
                    "model_output": raw,
                }, ensure_ascii=False) + "\n")
                if pairs:
                    kept_states += 1
                    for pair in pairs:
                        output.write(json.dumps({
                            "state_id": row["state_id"],
                            "split": row["split"],
                            **pair,
                        }, ensure_ascii=False) + "\n")
                        kept_pairs += 1
            output.flush()
            audit.flush()
            print(f"processed={min(base + args.batch_size, len(rows))} "
                  f"states_with_qa={kept_states} pairs={kept_pairs}", flush=True)
    print(json.dumps({"processed": len(rows), "states_with_qa": kept_states,
                      "pairs": kept_pairs, "split": args.split,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
