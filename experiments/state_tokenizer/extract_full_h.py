"""Extract temporary ragged Full-H tokens for the v2 reference subset."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import iter_jsonl, write_json
from .extract_qwen import (
    _StopAtLayer,
    _load_model,
    bf16_bits,
    modality_indices,
    prepare_inputs,
)
from .ragged_store import global_modality_lengths


def _open_array(path: Path, dtype, shape, resume: bool):
    if resume and path.exists():
        array = np.load(path, mmap_mode="r+")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"resume mismatch for {path}: {array.shape}/{array.dtype}")
        return array
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def extract(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    subset = list(iter_jsonl(args.records))
    if args.limit is not None:
        subset = subset[:args.limit]
    subset_rows = np.arange(args.rank, len(subset), args.world_size, dtype=np.int64)
    global_indices = np.asarray([subset[row]["global_index"] for row in subset_rows], np.int64)
    source_count = sum(1 for _ in iter_jsonl(args.source_records))
    lengths_global = global_modality_lengths(args.features, source_count)
    modality_lengths = lengths_global[global_indices]
    token_lengths = modality_lengths.sum(axis=1, dtype=np.int64)
    offsets = np.concatenate((np.zeros((1,), np.int64), np.cumsum(token_lengths, dtype=np.int64)))

    output = Path(args.output) / f"worker{args.rank:02d}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "subset_rows.npy", subset_rows)
    np.save(output / "record_indices.npy", global_indices)
    np.save(output / "offsets.npy", offsets)
    np.save(output / "modality-lengths.npy", modality_lengths)
    modality_ids = _open_array(
        output / "modality-ids.npy", np.uint8, (int(offsets[-1]),), args.resume
    )
    for local, values in enumerate(modality_lengths):
        cursor = int(offsets[local])
        for modality, count in enumerate(values):
            modality_ids[cursor:cursor + int(count)] = modality
            cursor += int(count)
    modality_ids.flush()
    tokens = _open_array(
        output / "tokens-bf16.npy", np.uint16, (int(offsets[-1]), 4096), args.resume
    )
    done = _open_array(output / "done.npy", np.bool_, (len(subset_rows),), args.resume)
    if not args.resume:
        done[:] = False
        done.flush()

    processor, model = _load_model(args.model, device, args.use_kernels)
    layers = model.model.language_model.layers
    if not 1 <= args.layer <= len(layers):
        raise ValueError(f"layer must be in [1,{len(layers)}]")
    capture: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output_value):
        capture["hidden"] = output_value[0] if isinstance(output_value, tuple) else output_value
        if args.early_stop:
            raise _StopAtLayer

    handle = layers[args.layer - 1].register_forward_hook(hook)
    started = time.time()
    completed = int(np.asarray(done).sum())
    try:
        for local, subset_row in enumerate(subset_rows):
            if bool(done[local]):
                continue
            record = subset[int(subset_row)]
            image_path = Path(args.records).resolve().parent / record["screenshot"]
            with Image.open(image_path) as image_handle:
                inputs, truncated, _, _ = prepare_inputs(
                    processor, image_handle.convert("RGB"), record["dom"],
                    record["instruction"], args.max_length, args.prompt_mode,
                )
            if truncated:
                raise ValueError(f"v2 Full-H record was truncated: {record['state_id']}")
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
            parts = [hidden[0].index_select(0, index) for index in indices]
            actual = np.asarray([len(part) for part in parts], np.int32)
            if not np.array_equal(actual, modality_lengths[local]):
                raise ValueError(
                    f"modality length mismatch for {record['state_id']}: "
                    f"{actual.tolist()} != {modality_lengths[local].tolist()}"
                )
            flat = torch.cat(parts, dim=0)
            start, stop = int(offsets[local]), int(offsets[local + 1])
            tokens[start:stop] = bf16_bits(flat)
            done[local] = True
            completed += 1
            del inputs, hidden, flat, parts
            if completed % args.flush_every == 0 or completed == len(subset_rows):
                tokens.flush()
                done.flush()
                elapsed = time.time() - started
                logging.info(
                    "rank=%d completed=%d/%d rate=%.3f states/s",
                    args.rank, completed, len(subset_rows), completed / max(elapsed, 1e-6),
                )
    finally:
        handle.remove()
    elapsed = time.time() - started
    summary = {
        "rank": args.rank,
        "world_size": args.world_size,
        "records": len(subset_rows),
        "completed": completed,
        "tokens": int(offsets[-1]),
        "logical_bf16_gib": float(int(offsets[-1]) * 4096 * 2 / 1024**3),
        "elapsed_seconds": elapsed,
        "states_per_second": completed / max(elapsed, 1e-6),
        "layer": args.layer,
        "model": str(Path(args.model).resolve()),
        "subset_manifest": str(Path(args.records).resolve()),
        "early_stop": args.early_stop,
        "use_kernels": args.use_kernels,
        "prompt_mode": args.prompt_mode,
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-kernels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt-mode", choices=("base", "instruct"), default="base")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error("rank must be in [0, world-size)")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    print(json.dumps(extract(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
