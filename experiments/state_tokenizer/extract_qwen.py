"""Extract Qwen3.5 layer-16 states and stream fixed pooled representations.

Each worker owns one GPU and a deterministic strided subset of the manifest.
Only Y64, Y32, and the fixed H probe summaries are written.  Full H_t never
leaves the worker and is never stored on disk.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from .common import MODALITIES, SLOT_LAYOUTS, iter_jsonl, write_json


DOM_START = "<STATE_DOM_7F3A>"
DOM_END = "<STATE_DOM_END_2D8C>"
INSTRUCTION_START = "<STATE_INSTRUCTION_9C2D>"
INSTRUCTION_END = "<STATE_INSTRUCTION_END_4B1E>"


class _StopAtLayer(RuntimeError):
    pass


def _find_subsequence(values: Sequence[int], query: Sequence[int], name: str) -> int:
    if not query:
        raise ValueError(f"empty marker tokenization for {name}")
    hits = []
    limit = len(values) - len(query) + 1
    for index in range(max(0, limit)):
        if list(values[index:index + len(query)]) == list(query):
            hits.append(index)
    if len(hits) != 1:
        raise ValueError(f"expected one {name} marker, found {hits}")
    return hits[0]


def _marker_ids(processor, marker: str) -> list[int]:
    return list(processor.tokenizer.encode(marker, add_special_tokens=False))


def _prompt(dom: str, instruction: str) -> str:
    return (
        f"{DOM_START}\n{dom}\n{DOM_END}\n"
        f"{INSTRUCTION_START}\n{instruction}\n{INSTRUCTION_END}"
    )


PROMPT_MODES = ("base", "instruct")


def _input_text(
    processor, dom: str, instruction: str, prompt_mode: str = "base",
    enable_thinking: bool | None = None,
) -> str:
    """Serialize the frozen observation prompt for Base or Instruct Qwen.

    Marker-delimited DOM/instruction spans are deliberately identical between
    modes.  Only the model-native conversation wrapper differs, which keeps
    modality indexing exact while allowing v8 Base artifacts and v9 Instruct
    artifacts to coexist.

    ``enable_thinking`` defaults to ``None``, meaning the argument is not passed
    to the template at all -- byte-identical to what every existing caller has
    always produced.  It exists for callers that *generate* from this prompt:
    the template at ``models/Qwen3.5-9B/chat_template.jinja:148-153`` closes the
    prompt with ``<think>\\n`` unless told otherwise, and greedy decoding from
    there spends its whole budget on a reasoning preamble
    (see ``connector_fidelity.py:201``, ``instruct_bridge.py:361``).  Extraction
    never generates, so extraction never needs it.
    """
    if prompt_mode == "base":
        if enable_thinking is not None:
            raise ValueError("enable_thinking only applies to the instruct chat template")
        return (
            f"{processor.vision_start_token}{processor.image_token}"
            f"{processor.vision_end_token}\n{_prompt(dom, instruction)}"
        )
    if prompt_mode != "instruct":
        raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}, got {prompt_mode!r}")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": _prompt(dom, instruction)},
        ],
    }]
    extra = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **extra
    )


def _process_once(
    processor, image: Image.Image, dom: str, instruction: str,
    prompt_mode: str = "base", enable_thinking: bool | None = None,
):
    # Qwen3.5-9B-Base intentionally ships without a chat template.  Supply the
    # native multimodal sentinel sequence directly; Qwen3VLProcessor expands
    # the single image token to the exact number required by image_grid_thw.
    text = _input_text(processor, dom, instruction, prompt_mode, enable_thinking)
    return processor(
        images=[image],
        text=[text],
        return_tensors="pt",
        return_mm_token_type_ids=True,
    )


def prepare_inputs(
    processor, image: Image.Image, dom: str, instruction: str, max_length: int,
    prompt_mode: str = "base", enable_thinking: bool | None = None,
):
    inputs = _process_once(processor, image, dom, instruction, prompt_mode, enable_thinking)
    full_length = int(inputs["input_ids"].shape[1])
    if full_length <= max_length:
        return inputs, False, len(dom), full_length

    low, high = 0, len(dom)
    best = None
    while low <= high:
        middle = (low + high) // 2
        candidate = _process_once(
            processor, image, dom[:middle], instruction, prompt_mode, enable_thinking
        )
        length = int(candidate["input_ids"].shape[1])
        if length <= max_length:
            best = (candidate, middle, length)
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise ValueError(
            f"image + instruction alone exceed max_length={max_length} for instruction={instruction!r}"
        )
    return best[0], True, best[1], full_length


def modality_indices(processor, model, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    ids = input_ids[0].tolist()
    image_id = int(model.config.image_token_id)
    image = [index for index, value in enumerate(ids) if value == image_id]
    if not image:
        raise ValueError("processor emitted no image tokens")

    marker_ids = {
        "dom_start": _marker_ids(processor, DOM_START),
        "dom_end": _marker_ids(processor, DOM_END),
        "instruction_start": _marker_ids(processor, INSTRUCTION_START),
        "instruction_end": _marker_ids(processor, INSTRUCTION_END),
    }
    positions = {
        name: _find_subsequence(ids, values, name)
        for name, values in marker_ids.items()
    }
    dom_start = positions["dom_start"] + len(marker_ids["dom_start"])
    dom_end = positions["dom_end"]
    instruction_start = positions["instruction_start"] + len(marker_ids["instruction_start"])
    instruction_end = positions["instruction_end"]
    if not (dom_start < dom_end <= instruction_start < instruction_end):
        raise ValueError(f"invalid marker order: {positions}")
    device = input_ids.device
    return (
        torch.tensor(image, dtype=torch.long, device=device),
        torch.arange(dom_start, dom_end, dtype=torch.long, device=device),
        torch.arange(instruction_start, instruction_end, dtype=torch.long, device=device),
    )


def aligned_dom_token_offsets(
    processor,
    model,
    text: str,
    dom: str,
    processed_ids: torch.Tensor,
    processed_dom_indices: torch.Tensor,
) -> list[tuple[int, int]]:
    """Map processed DOM hidden positions to exact offsets in the raw DOM string."""
    if text.count(dom) != 1:
        raise ValueError("expected compact DOM to occur exactly once in Qwen input text")
    dom_start = text.index(dom)
    standalone = processor.tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    standalone_dom = modality_indices(processor, model, standalone["input_ids"])[1]
    processed_values = processed_ids[0].index_select(
        0, processed_dom_indices.to(processed_ids.device)
    ).cpu()
    standalone_values = standalone["input_ids"][0].index_select(0, standalone_dom).cpu()
    if not torch.equal(processed_values, standalone_values):
        raise ValueError("standalone and multimodal processor DOM token IDs differ")
    offsets = standalone["offset_mapping"][0].index_select(0, standalone_dom)
    return [
        (int(start) - dom_start, int(stop) - dom_start)
        for start, stop in offsets.tolist()
    ]


def adaptive_pool_torch(values: torch.Tensor, slots: int) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"expected (tokens, width), got {tuple(values.shape)}")
    count, width = values.shape
    if count == 0:
        return torch.zeros((slots, width), dtype=torch.float32, device=values.device)
    chunks = []
    for index in range(slots):
        start = math.floor(index * count / slots)
        stop = math.ceil((index + 1) * count / slots)
        stop = max(start + 1, min(stop, count))
        chunks.append(values[start:stop].float().mean(dim=0))
    return torch.stack(chunks, dim=0)


def pooled_and_stats(hidden: torch.Tensor, indices: tuple[torch.Tensor, ...]):
    modalities = [hidden[0].index_select(0, index) for index in indices]
    if any(values.shape[0] == 0 for values in modalities):
        raise ValueError(f"empty modality: {[int(x.shape[0]) for x in modalities]}")
    pooled = {}
    for total_slots, layout in SLOT_LAYOUTS.items():
        pooled[total_slots] = torch.cat([
            adaptive_pool_torch(values, slots)
            for values, slots in zip(modalities, layout)
        ], dim=0)
    stats = torch.stack([
        item
        for values in modalities
        for item in (values.float().mean(dim=0), values.float().amax(dim=0))
    ], dim=0)
    lengths = np.asarray([values.shape[0] for values in modalities], np.int32)
    return pooled[64], pooled[32], stats, lengths


def bf16_bits(values: torch.Tensor) -> np.ndarray:
    return values.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()


def _open_array(path: Path, dtype, shape, resume: bool):
    if resume and path.exists():
        array = np.load(path, mmap_mode="r+")
        if tuple(array.shape) != tuple(shape) or array.dtype != np.dtype(dtype):
            raise ValueError(f"resume array mismatch at {path}: {array.shape}/{array.dtype}")
        return array
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _load_model(model_path: str, device: torch.device, use_kernels: bool,
                dtype: torch.dtype = torch.bfloat16):
    """``dtype`` exists to test one hypothesis and should stay bfloat16 otherwise.

    Every divergence in the joint trainer has been a non-finite *gradient* at an
    unpredictable step, and after the loss functions were proved bounded by
    construction the only shared path left is the backward through the frozen
    9B itself. Running it in fp32 doubles the memory and settles that: if the
    skips vanish it is numerical, if they persist the hypothesis is dead.
    """
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    kwargs = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "attn_implementation": "sdpa",
    }
    if use_kernels:
        kwargs["use_kernels"] = True
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_path, **kwargs)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model


def extract(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real Qwen extraction")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    records = list(iter_jsonl(args.records))
    worker_indices = np.arange(args.rank, len(records), args.world_size, dtype=np.int64)
    output = Path(args.output) / f"worker{args.rank:02d}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "record_indices.npy", worker_indices)
    count = len(worker_indices)
    arrays = {
        "y64": _open_array(output / "y64-bf16.npy", np.uint16, (count, 64, 4096), args.resume),
        "y32": _open_array(output / "y32-bf16.npy", np.uint16, (count, 32, 4096), args.resume),
        "h_stats": _open_array(output / "h-stats-fp16.npy", np.float16, (count, 6, 4096), args.resume),
        "lengths": _open_array(output / "modality-lengths.npy", np.int32, (count, 3), args.resume),
        "sequence": _open_array(output / "sequence-lengths.npy", np.int32, (count, 2), args.resume),
        "truncated": _open_array(output / "truncated.npy", np.bool_, (count,), args.resume),
        "done": _open_array(output / "done.npy", np.bool_, (count,), args.resume),
    }
    if not args.resume:
        arrays["done"][:] = False
        arrays["done"].flush()

    started = time.time()
    processor, model = _load_model(args.model, device, args.use_kernels)
    layers = model.model.language_model.layers
    if not 1 <= args.layer <= len(layers):
        raise ValueError(f"layer must be in [1,{len(layers)}]")
    capture: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        capture["hidden"] = output[0] if isinstance(output, tuple) else output
        if args.early_stop:
            raise _StopAtLayer

    handle = layers[args.layer - 1].register_forward_hook(hook)
    completed = int(np.asarray(arrays["done"]).sum())
    try:
        for local_index, record_index in enumerate(worker_indices):
            if bool(arrays["done"][local_index]):
                continue
            record = records[int(record_index)]
            image_path = Path(args.records).resolve().parent / record["screenshot"]
            with Image.open(image_path) as image_handle:
                image = image_handle.convert("RGB")
                inputs, truncated, kept_chars, original_length = prepare_inputs(
                    processor, image, record["dom"], record["instruction"],
                    args.max_length, args.prompt_mode,
                )
            inputs = inputs.to(device)
            indices = modality_indices(processor, model, inputs["input_ids"])
            capture.clear()
            try:
                with torch.inference_mode():
                    model.model(**inputs, use_cache=False, output_hidden_states=False)
            except _StopAtLayer:
                pass
            hidden = capture.get("hidden")
            if hidden is None:
                raise RuntimeError(f"layer hook did not fire for {record['state_id']}")
            y64, y32, h_stats, lengths = pooled_and_stats(hidden, indices)
            arrays["y64"][local_index] = bf16_bits(y64)
            arrays["y32"][local_index] = bf16_bits(y32)
            arrays["h_stats"][local_index] = h_stats.to(torch.float16).cpu().numpy()
            arrays["lengths"][local_index] = lengths
            arrays["sequence"][local_index] = (int(inputs["input_ids"].shape[1]), original_length)
            arrays["truncated"][local_index] = truncated
            arrays["done"][local_index] = True
            completed += 1
            del inputs, hidden, y64, y32, h_stats
            if completed % args.flush_every == 0 or completed == count:
                for array in arrays.values():
                    array.flush()
                elapsed = time.time() - started
                logging.info(
                    "rank=%d completed=%d/%d rate=%.3f states/s kept_dom_chars=%d",
                    args.rank, completed, count, completed / max(elapsed, 1e-6), kept_chars,
                )
    finally:
        handle.remove()
    elapsed = time.time() - started
    summary = {
        "rank": args.rank,
        "world_size": args.world_size,
        "records": count,
        "completed": completed,
        "elapsed_seconds": elapsed,
        "states_per_second": completed / max(elapsed, 1e-6),
        "device": str(device),
        "model": str(Path(args.model).resolve()),
        "layer": args.layer,
        "max_length": args.max_length,
        "use_kernels": args.use_kernels,
        "early_stop": args.early_stop,
        "prompt_mode": args.prompt_mode,
        "truncated_count": int(np.asarray(arrays["truncated"]).sum()),
        "slot_layouts": {str(key): list(value) for key, value in SLOT_LAYOUTS.items()},
        "modalities": list(MODALITIES),
    }
    write_json(output / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-kernels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-mode", choices=PROMPT_MODES, default="base")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error("rank must be in [0, world-size)")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    print(json.dumps(extract(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
