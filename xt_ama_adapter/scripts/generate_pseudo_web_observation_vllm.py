"""Transcribe unique HumanTrajs screenshots into visible pseudo web observations."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

PROMPT = """/no_think
Transcribe this webpage screenshot into a compact structured observation for a web agent.
Use ONLY visually supported content. Do not infer hidden DOM nodes, off-screen content, the
user's task, or what happens later. Omit uncertain text. Keep at most 20 important visible
text snippets and 15 visible interactive elements. Never repeat a phrase or list item.
Element roles should be simple ARIA-like
roles such as link, button, textbox, checkbox, combobox, tab, menuitem, heading, image, text,
or other. Return exactly:
{"page_title":"...","visible_text":["..."],"interactive_elements":[{"role":"button","name":"...","value":"","state":""}],"page_summary":"..."}
"""


def parse_record(text: str) -> dict | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(value.get("page_title"), str) or not isinstance(value.get("page_summary"), str):
        return None
    if not isinstance(value.get("visible_text"), list) or not all(
            isinstance(item, str) for item in value["visible_text"]):
        return None
    if not isinstance(value.get("interactive_elements"), list):
        return None
    for element in value["interactive_elements"]:
        if not isinstance(element, dict) or not all(
                isinstance(element.get(key), str) for key in ("role", "name", "value", "state")):
            return None
    value["visible_text"] = [item.strip()[:300] for item in value["visible_text"][:20] if item.strip()]
    value["interactive_elements"] = [{key: element[key].strip()[:300]
        for key in ("role", "name", "value", "state")}
        for element in value["interactive_elements"][:15]]
    value["page_title"] = value["page_title"].strip()[:300]
    value["page_summary"] = value["page_summary"].strip()[:600]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--retry-audit", type=Path, nargs="*",
                        help="generate only hashes with parsed=false in these audit JSONL files")
    parser.add_argument("--exclude-observations", type=Path, nargs="*",
                        help="skip hashes already present in successful observation JSONL shards")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=0)
    args = parser.parse_args()

    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    unique = {}
    with args.input.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            unique.setdefault(row["image_sha256"], row)
    rows = list(unique.values())
    if args.retry_audit:
        failed = set()
        for path in args.retry_audit:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        item = json.loads(line)
                        if not item.get("parsed"):
                            failed.add(item["image_sha256"])
        rows = [row for row in rows if row["image_sha256"] in failed]
    if args.exclude_observations:
        completed = set()
        for path in args.exclude_observations:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        completed.add(json.loads(line)["image_sha256"])
        rows = [row for row in rows if row["image_sha256"] not in completed]
    rows = rows[args.start:args.end or None]

    schema = {"type": "object", "properties": {
        "page_title": {"type": "string", "maxLength": 300},
        "visible_text": {"type": "array", "items": {"type": "string", "maxLength": 300},
                         "maxItems": 20},
        "interactive_elements": {"type": "array", "maxItems": 15, "items": {
            "type": "object", "properties": {
                "role": {"type": "string", "maxLength": 30},
                "name": {"type": "string", "maxLength": 300},
                "value": {"type": "string", "maxLength": 300},
                "state": {"type": "string", "maxLength": 100}},
            "required": ["role", "name", "value", "state"], "additionalProperties": False}},
        "page_summary": {"type": "string", "maxLength": 600}},
        "required": ["page_title", "visible_text", "interactive_elements", "page_summary"],
        "additionalProperties": False}
    llm = LLM(model=str(args.model), dtype="auto", max_model_len=16384,
              max_num_seqs=args.batch_size, limit_mm_per_prompt={"image": 1},
              mm_processor_kwargs={"max_pixels": 1280 * 1024},
              structured_outputs_config={"backend": "xgrammar", "disable_any_whitespace": True},
              enforce_eager=True)
    sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens,
        structured_outputs=StructuredOutputsParams(json=schema))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    generated = 0
    with args.output.open("w", encoding="utf-8") as output, \
            args.audit.open("w", encoding="utf-8") as audit:
        for base in range(0, len(rows), args.batch_size):
            batch = rows[base:base + args.batch_size]
            requests = []
            for row in batch:
                meta = row.get("observation") or {}
                context = (f"Recorded URL: {meta.get('url', '')}\n"
                           f"Recorded page title: {(meta.get('open_pages_titles') or [''])[0]}\n")
                image = Image.open(row["image_path"]).convert("RGB")
                requests.append({"prompt": "<|im_start|>user\n<|vision_start|><|image_pad|>"
                    "<|vision_end|>\n" + PROMPT + context + "<|im_end|>\n<|im_start|>assistant\n",
                    "multi_modal_data": {"image": image}})
            results = llm.generate(requests, sampling, use_tqdm=False)
            for row, result in zip(batch, results):
                raw = result.outputs[0].text
                observation = parse_record(raw)
                audit_row = {"image_sha256": row["image_sha256"], "image_path": row["image_path"],
                             "parsed": bool(observation), "model_output": raw}
                if observation:
                    output.write(json.dumps({"protocol": "vl-pseudo-web-observation-v1",
                        "image_sha256": row["image_sha256"], "image_path": row["image_path"],
                        "source_url": (row.get("observation") or {}).get("url"),
                        "pseudo_observation": observation,
                        "generator": "Qwen3.5-9B-vLLM", "verification_status": "pending"},
                        ensure_ascii=False) + "\n")
                    generated += 1
                audit.write(json.dumps(audit_row, ensure_ascii=False) + "\n")
            output.flush(); audit.flush()
            print(f"processed={min(base + args.batch_size, len(rows))} generated={generated}", flush=True)
    print(json.dumps({"processed": len(rows), "generated": generated,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
