from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


PROTOCOL = "residualmem_molmoweb_captions_v1"

CAPTION_PROMPT = (
    "Describe this browser screenshot in about 60 words, as a neutral "
    "third-person description of what is visible on the screen.\n"
    "Name the concrete on-screen text: headings, button labels, menu items, "
    "field labels and values, and any prominent notice.\n"
    "Describe only what is visible. Do not guess the user's goal, do not say "
    "what should be clicked next, and do not mention any task or intent.\n"
    "Write flowing prose in 2-3 sentences. Do not use bullet points or "
    "headings, and stop when you reach about 60 words."
)



def _jsonl_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield line


def _load(model_path: str, device: torch.device):
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation="sdpa",
    )
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model


def _batch_inputs(processor, images: list[Image.Image], device):
    texts = []
    for _ in images:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": CAPTION_PROMPT}]}]
        texts.append(processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        ))
    inputs = processor(
        text=texts, images=images, return_tensors="pt", padding=True,
    )
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}


def _trim_to_sentence(text: str) -> str:
    stripped = text.strip()
    if not stripped or stripped[-1] in ".!?\"":
        return stripped
    cut = max(stripped.rfind(mark) for mark in (". ", "! ", "? ", ".\n"))
    return stripped[: cut + 1].strip() if cut > 0 else stripped


def run(
    records: list[dict], images_root: Path, output: Path, *,
    model_path: str, device: str, batch_size: int, max_new_tokens: int,
) -> None:
    target = torch.device(device)
    processor, model = _load(model_path, target)
    processor.tokenizer.padding_side = "left"

    done: set[str] = set()
    if output.exists():
        for line in _jsonl_lines(output):
            try:
                done.add(json.loads(line)["state_id"])
            except (json.JSONDecodeError, KeyError):
                break
    pending = [r for r in records if r["state_id"] not in done]
    logging.info("%d records, %d already captioned, %d to do",
                 len(records), len(done), len(pending))

    started = time.time()
    written = 0
    with output.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            images = []
            keep = []
            for record in chunk:
                path = images_root / record["screenshot"]
                try:
                    images.append(Image.open(path).convert("RGB"))
                    keep.append(record)
                except (OSError, ValueError):
                    logging.warning("unreadable screenshot %s", path)
            if not images:
                continue

            inputs = _batch_inputs(processor, images, target)
            with torch.no_grad():
                generated = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                )
            trimmed = generated[:, inputs["input_ids"].shape[1]:]
            captions = processor.batch_decode(trimmed, skip_special_tokens=True)

            for record, caption in zip(keep, captions):
                handle.write(json.dumps({
                    "state_id": record["state_id"],
                    "caption": _trim_to_sentence(caption),
                }, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()

            if (start // batch_size) % 20 == 0:
                rate = written / max(time.time() - started, 1e-9)
                remaining = (len(pending) - written) / max(rate, 1e-9)
                logging.info("%d/%d  %.2f states/s  eta %.1f min",
                             written, len(pending), rate, remaining / 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True, action="append")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--episode-limit", type=int, default=None,
                        help="cap on whole trajectories, so transitions stay intact")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    records: list[dict] = []
    for path in args.records:
        for line in _jsonl_lines(path):
            if line.strip():
                records.append(json.loads(line))

    if args.episode_limit is not None:
        keep, seen = [], set()
        for record in records:
            if record["episode_id"] not in seen:
                if len(seen) >= args.episode_limit:
                    continue
                seen.add(record["episode_id"])
            keep.append(record)
        records = keep

    if args.shard_count > 1:
        episodes = sorted({r["episode_id"] for r in records})
        mine = {e for i, e in enumerate(episodes) if i % args.shard_count == args.shard_index}
        records = [r for r in records if r["episode_id"] in mine]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(records, args.images, args.output,
        model_path=args.model, device=args.device,
        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
