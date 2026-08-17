"""Extract fixed-prompt hidden states, optionally keeping only prompt tokens."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import SLOT_LAYOUTS, iter_jsonl, write_json
from .extract_full_h import _open_array
from .extract_qwen import (
    _StopAtLayer,
    _input_text,
    _load_model,
    adaptive_pool_torch,
    aligned_dom_token_offsets,
    bf16_bits,
    modality_indices,
    prepare_inputs,
)
from .key_pooling import DETAIL_SLOTS, KEY64_LAYOUT, build_key64
from .ragged_store import global_modality_lengths


def _prepare(
    processor, model, record: dict, records_path: Path, max_length: int,
    *, need_dom_offsets: bool = True,
):
    image_path = records_path.resolve().parent / record["screenshot"]
    with Image.open(image_path) as handle:
        inputs, truncated, _, _ = prepare_inputs(
            processor, handle.convert("RGB"), record["dom"], record["instruction"], max_length
        )
    if truncated:
        raise ValueError(f"fixed-prompt record was truncated: {record['state_id']}")
    indices = modality_indices(processor, model, inputs["input_ids"])
    dom_offsets = None
    if need_dom_offsets:
        text = _input_text(processor, record["dom"], record["instruction"])
        dom_offsets = aligned_dom_token_offsets(
            processor, model, text, record["dom"], inputs["input_ids"], indices[1]
        )
    return inputs, indices, dom_offsets


def cache_modality_lengths(
    expected_lengths: np.ndarray, instruction_only_cache: bool,
) -> np.ndarray:
    cached = np.asarray(expected_lengths, np.int32).copy()
    if instruction_only_cache:
        cached[:, :2] = 0
    return cached


def extract(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    records_path = Path(args.records)
    subset = list(iter_jsonl(records_path))
    if args.limit is not None:
        subset = subset[:args.limit]
    instructions = {record["instruction"] for record in subset}
    if len(instructions) != 1:
        raise ValueError(f"expected one fixed instruction, found {len(instructions)}")
    fixed_instruction = next(iter(instructions))
    subset_rows = np.arange(args.rank, len(subset), args.world_size, dtype=np.int64)
    if not len(subset_rows):
        raise ValueError("worker received no records")
    global_indices = np.asarray([subset[row]["global_index"] for row in subset_rows], np.int64)

    processor, model = _load_model(args.model, device, args.use_kernels)
    first_inputs, first_indices, first_dom_offsets = _prepare(
        processor, model, subset[int(subset_rows[0])], records_path, args.max_length,
        need_dom_offsets=not args.instruction_only_cache,
    )
    first_lengths = np.asarray([len(index) for index in first_indices], np.int32)
    source_count = sum(1 for _ in iter_jsonl(args.source_records))
    source_lengths = global_modality_lengths(args.source_features, source_count)
    expected_modality_lengths = np.asarray(source_lengths[global_indices], np.int32).copy()
    if not np.array_equal(first_lengths[:2], expected_modality_lengths[0, :2]):
        raise ValueError(
            f"fixed-prompt image/DOM lengths changed unexpectedly: "
            f"{first_lengths[:2].tolist()} != {expected_modality_lengths[0, :2].tolist()}"
        )
    expected_modality_lengths[:, 2] = first_lengths[2]
    modality_lengths = cache_modality_lengths(
        expected_modality_lengths, args.instruction_only_cache
    )
    token_lengths = modality_lengths.sum(axis=1, dtype=np.int64)
    offsets = np.concatenate((np.zeros((1,), np.int64), np.cumsum(token_lengths)))

    output = Path(args.output) / f"worker{args.rank:02d}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "subset_rows.npy", subset_rows)
    np.save(output / "record_indices.npy", global_indices)
    np.save(output / "offsets.npy", offsets)
    np.save(output / "modality-lengths.npy", modality_lengths)
    tokens = _open_array(
        output / "tokens-bf16.npy", np.uint16, (int(offsets[-1]), 4096), args.resume
    )
    y64 = y32 = None
    key64 = key64_valid = key64_positions = key64_detail_source = None
    key64_audit = key64_overflow = key64_done = None
    if not args.instruction_only_cache:
        key_bundle = [
            output / "key64-bf16.npy", output / "key64-valid.npy",
            output / "key64-positions.npy", output / "key64-detail-source.npy",
            output / "key64-audit.npy", output / "key64-overflow.npy",
            output / "key64-done.npy",
        ]
        key_bundle_complete = all(path.exists() for path in key_bundle)
        if args.resume and any(path.exists() for path in key_bundle) and not key_bundle_complete:
            raise RuntimeError(f"incomplete key64 resume bundle in {output}")
        y64 = _open_array(
            output / "y64-bf16.npy", np.uint16, (len(subset_rows), 64, 4096), args.resume
        )
        y32 = _open_array(
            output / "y32-bf16.npy", np.uint16, (len(subset_rows), 32, 4096), args.resume
        )
        key64 = _open_array(
            output / "key64-bf16.npy", np.uint16, (len(subset_rows), 64, 4096), args.resume
        )
        key64_valid = _open_array(
            output / "key64-valid.npy", np.bool_, (len(subset_rows), 64), args.resume
        )
        key64_positions = _open_array(
            output / "key64-positions.npy", np.float32, (len(subset_rows), 64), args.resume
        )
        key64_detail_source = _open_array(
            output / "key64-detail-source.npy", np.int32,
            (len(subset_rows), DETAIL_SLOTS), args.resume,
        )
        key64_audit = _open_array(
            output / "key64-audit.npy", np.int32, (len(subset_rows), 6), args.resume
        )
        key64_overflow = _open_array(
            output / "key64-overflow.npy", np.bool_, (len(subset_rows),), args.resume
        )
        key64_done = _open_array(
            output / "key64-done.npy", np.bool_, (len(subset_rows),), args.resume
        )
        if not args.resume or not key_bundle_complete:
            key64_done[:] = False
            key64_done.flush()
    done = _open_array(output / "done.npy", np.bool_, (len(subset_rows),), args.resume)
    if not args.resume:
        done[:] = False
        done.flush()

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
    completion = np.asarray(done, np.bool_)
    if key64_done is not None:
        completion = completion & np.asarray(key64_done, np.bool_)
    completed = int(completion.sum())
    try:
        for local, subset_row in enumerate(subset_rows):
            if bool(done[local]) and (key64_done is None or bool(key64_done[local])):
                continue
            record = subset[int(subset_row)]
            if local == 0:
                inputs, indices, dom_offsets = first_inputs, first_indices, first_dom_offsets
            else:
                inputs, indices, dom_offsets = _prepare(
                    processor, model, record, records_path, args.max_length,
                    need_dom_offsets=not args.instruction_only_cache,
                )
            image_grid_thw = inputs["image_grid_thw"][0].tolist()
            inputs = inputs.to(device)
            indices = tuple(index.to(device) for index in indices)
            capture.clear()
            try:
                with torch.inference_mode():
                    model.model(**inputs, use_cache=False, output_hidden_states=False)
            except _StopAtLayer:
                pass
            hidden = capture.pop("hidden", None)
            if hidden is None:
                raise RuntimeError(f"layer hook did not fire for {record['state_id']}")
            parts = [hidden[0].index_select(0, index) for index in indices]
            actual = np.asarray([len(part) for part in parts], np.int32)
            if not np.array_equal(actual, expected_modality_lengths[local]):
                raise ValueError(
                    f"modality length mismatch for {record['state_id']}: "
                    f"{actual.tolist()} != {expected_modality_lengths[local].tolist()}"
                )
            flat = parts[2] if args.instruction_only_cache else torch.cat(parts, dim=0)
            pooled = {}
            key_output = None
            if not args.instruction_only_cache:
                pooled = {
                    total: torch.cat([
                        adaptive_pool_torch(part, slots)
                        for part, slots in zip(parts, layout)
                    ])
                    for total, layout in SLOT_LAYOUTS.items()
                }
                key_output = build_key64(
                    parts[0], parts[1], parts[2],
                    dom=record["dom"],
                    dom_token_offsets=dom_offsets,
                    image_grid_thw=image_grid_thw,
                    spatial_merge_size=int(model.config.vision_config.spatial_merge_size),
                )
            start, stop = int(offsets[local]), int(offsets[local + 1])
            tokens[start:stop] = bf16_bits(flat)
            if y64 is not None and y32 is not None:
                if key_output is None:
                    raise RuntimeError("key64 output was not produced")
                y64[local] = bf16_bits(pooled[64])
                y32[local] = bf16_bits(pooled[32])
                key64[local] = bf16_bits(key_output.tokens)
                key64_valid[local] = key_output.valid.cpu().numpy()
                key64_positions[local] = key_output.positions.cpu().numpy()
                key64_detail_source[local] = key_output.detail_source.cpu().numpy()
                key64_audit[local] = np.asarray([
                    key_output.audit["candidate_spans"],
                    key_output.audit["selected_spans"],
                    key_output.audit["selected_tokens"],
                    key_output.audit["dom_nodes"],
                    key_output.audit["context_nodes"],
                    key_output.audit["separator_dom_tokens"],
                ], np.int32)
                key64_overflow[local] = key_output.audit["detail_overflow"]
                key64_done[local] = True
            done[local] = True
            completed += 1
            if completed % args.flush_every == 0 or completed == len(subset_rows):
                arrays = [tokens]
                if y64 is not None and y32 is not None:
                    arrays.extend((
                        y64, y32, key64, key64_valid, key64_positions,
                        key64_detail_source, key64_audit, key64_overflow,
                    ))
                for array in arrays:
                    array.flush()
                if key64_done is not None:
                    key64_done.flush()
                done.flush()
                elapsed = time.time() - started
                logging.info(
                    "rank=%d completed=%d/%d rate=%.3f states/s",
                    args.rank, completed, len(subset_rows), completed / max(elapsed, 1e-6),
                )
            del inputs, hidden, flat, parts, pooled, key_output
    finally:
        handle.remove()
    elapsed = time.time() - started
    summary = {
        "protocol": "fixed_task_independent_observation_v1",
        "rank": args.rank,
        "world_size": args.world_size,
        "records": len(subset_rows),
        "completed": completed,
        "tokens": int(offsets[-1]),
        "logical_full_h_bf16_gib": float(int(offsets[-1]) * 4096 * 2 / 1024**3),
        "elapsed_seconds": elapsed,
        "states_per_second": completed / max(elapsed, 1e-6),
        "layer": args.layer,
        "instruction_tokens": int(first_lengths[2]),
        "cache_mode": "instruction_only" if args.instruction_only_cache else "full_h_y64_y32",
        "fixed_instruction_sha256": hashlib.sha256(fixed_instruction.encode()).hexdigest(),
        "model": str(Path(args.model).resolve()),
        "subset_manifest": str(records_path.resolve()),
        "early_stop": args.early_stop,
        "use_kernels": args.use_kernels,
    }
    if key64_audit is not None:
        audit_values = np.asarray(key64_audit)
        valid_values = np.asarray(key64_valid)
        grid = [int(value) for value in first_inputs["image_grid_thw"][0].tolist()]
        merge = int(model.config.vision_config.spatial_merge_size)
        merged_hw = [grid[1] // merge, grid[2] // merge]
        summary["key64"] = {
            "protocol": "fixed_prompt_key64_v1",
            "layout": list(KEY64_LAYOUT),
            "merged_image_hw": merged_hw,
            "pooled_image_hw": [8, 4] if merged_hw[0] >= merged_hw[1] else [4, 8],
            "detail_overflow_count": int(np.asarray(key64_overflow).sum()),
            "detail_valid_min": int(valid_values[:, 32:44].sum(axis=1).min()),
            "detail_valid_max": int(valid_values[:, 32:44].sum(axis=1).max()),
            "context_valid_min": int(valid_values[:, 44:60].sum(axis=1).min()),
            "context_valid_max": int(valid_values[:, 44:60].sum(axis=1).max()),
            "separator_dom_tokens": int(audit_values[:, 5].sum()),
        }
    write_json(output / "fixed-prompt-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--source-records", required=True)
    parser.add_argument("--source-features", required=True)
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
    parser.add_argument("--instruction-only-cache", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error("rank must be in [0,world-size)")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    print(json.dumps(extract(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
