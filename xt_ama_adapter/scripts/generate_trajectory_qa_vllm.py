"""Generate multi-step QA from cleaned HumanTrajs windows without task leakage."""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()

    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    grouped = defaultdict(list)
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("split") == args.split:
                    grouped[row["trajectory_id"]].append(row)
    windows = []
    for trajectory_id, rows in grouped.items():
        rows.sort(key=lambda row: row["step_idx"])
        for offset in range(0, len(rows) - 1, args.stride):
            window, hashes = [], set()
            for row in rows[offset:offset + args.window_size]:
                if row["image_sha256"] not in hashes:
                    window.append(row); hashes.add(row["image_sha256"])
            if len(window) >= 2:
                windows.append((trajectory_id, window))
    windows = windows[args.start:args.end or None]

    schema = {"type": "object", "properties": {
        "question": {"type": "string"}, "answer": {"type": "string"},
        "evidence": {"type": "string"},
        "evidence_step_ids": {"type": "array", "items": {"type": "integer"},
                              "minItems": 2, "uniqueItems": True}},
        "required": ["question", "answer", "evidence", "evidence_step_ids"],
        "additionalProperties": False}
    llm = LLM(model=str(args.model), dtype="auto", max_model_len=16384,
              max_num_seqs=args.batch_size,
              limit_mm_per_prompt={"image": args.window_size},
              mm_processor_kwargs={"max_pixels": 1280 * 1024},
              structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True},
              enforce_eager=True)
    sampling = SamplingParams(temperature=0, max_tokens=512,
        structured_outputs=StructuredOutputsParams(json=schema))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    with args.output.open("w", encoding="utf-8") as output, \
            args.audit.open("w", encoding="utf-8") as audit:
        for base in range(0, len(windows), args.batch_size):
            batch = windows[base:base + args.batch_size]
            requests = []
            for _, window in batch:
                images, chunks = [], []
                for row in window:
                    images.append(Image.open(row["image_path"]).convert("RGB"))
                    chunks.append(f"Memory step {row['step_idx']}:<|vision_start|><|image_pad|><|vision_end|>\n"
                                  f"Action: {json.dumps(row['action'], ensure_ascii=False)}")
                prompt = ("<|im_start|>user\n" + "\n".join(chunks) +
                    "\n/no_think\nCreate ONE short temporal or causal QA that genuinely requires at least two "
                    "of these memory steps. Use only visible evidence. Return JSON and list every required "
                    "memory step ID in evidence_step_ids. Return exactly: "
                    '{"question":"...","answer":"...","evidence":"...","evidence_step_ids":[1,2]}'
                    "<|im_end|>\n<|im_start|>assistant\n")
                requests.append({"prompt": prompt, "multi_modal_data": {"image": images}})
            results = llm.generate(requests, sampling, use_tqdm=False)
            for (trajectory_id, window), result in zip(batch, results):
                raw = result.outputs[0].text
                try:
                    qa = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
                except (ValueError, json.JSONDecodeError):
                    qa = None
                allowed = {row["step_idx"] for row in window}
                ids = qa.get("evidence_step_ids", []) if isinstance(qa, dict) else []
                valid = (isinstance(qa, dict) and all(isinstance(qa.get(k), str) and qa[k].strip()
                         for k in ("question", "answer", "evidence")) and len(set(ids)) >= 2 and set(ids) <= allowed)
                if valid:
                    by_id = {row["step_idx"]: row for row in window}
                    groups = [by_id[idx].get("equivalent_step_ids") or [idx] for idx in ids]
                    record = {"protocol": "humantrajs-multistep-qa-v1",
                        "trajectory_id": trajectory_id, "split": args.split,
                        "question": qa["question"], "answer": qa["answer"],
                        "evidence": qa["evidence"], "gold_step_groups": groups,
                        "gold_step_ids": sorted({x for group in groups for x in group}),
                        "evidence_frames": [{"step_idx": row["step_idx"],
                            "image_path": row["image_path"]} for row in window if row["step_idx"] in ids],
                        "qa_mode": "multistep", "qa_source": "qwen3.5-9b-generated-unverified",
                        "verification_status": "pending"}
                    output.write(json.dumps(record, ensure_ascii=False) + "\n"); generated += 1
                audit.write(json.dumps({"trajectory_id": trajectory_id,
                    "candidate_step_ids": sorted(allowed), "valid": bool(valid), "model_output": raw},
                    ensure_ascii=False) + "\n")
            output.flush(); audit.flush()
            print(f"processed={min(base + args.batch_size, len(windows))} generated={generated}", flush=True)
    print(json.dumps({"windows": len(windows), "generated": generated,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
