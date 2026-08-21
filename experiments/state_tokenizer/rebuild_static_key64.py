"""Rebuild key64_static_v2 from cached fixed-prompt Full-H without Qwen forward passes."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoProcessor

from .common import iter_jsonl, write_json
from .extract_full_h import _open_array
from .extract_qwen import _input_text, bf16_bits, modality_indices
from .ragged_store import bf16_tensor_view, discover_full_h_shards
from .key_pooling import KEY64_LAYOUT
from .static_key_pooling import (
    DETAIL_SLOTS,
    SLOT_POOLED_LITERAL,
    SLOT_RAW_LITERAL,
    STATIC_KEY64_PROTOCOL,
    build_static_key64,
)


def _dom_offsets(
    processor, model, record: dict, expected_dom_tokens: int,
    prompt_mode: str = "base",
):
    text = _input_text(
        processor, record["dom"], record["instruction"], prompt_mode
    )
    encoded = processor.tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    indices = modality_indices(processor, model, encoded["input_ids"])
    if len(indices[1]) != expected_dom_tokens:
        raise ValueError(
            f"DOM token mismatch for {record['state_id']}: "
            f"{len(indices[1])} != {expected_dom_tokens}"
        )
    dom_start = text.index(record["dom"])
    offsets = encoded["offset_mapping"][0].index_select(0, indices[1])
    return [
        (int(start) - dom_start, int(stop) - dom_start)
        for start, stop in offsets.tolist()
    ], len(indices[2])


def rebuild(args: argparse.Namespace) -> dict:
    records_path = Path(args.records).resolve()
    full_h_root = Path(args.full_h).resolve()
    output_root = Path(args.output).resolve()
    if full_h_root == output_root:
        raise ValueError("static rebuild output must be separate from the Full-H root")
    if args.flush_every < 1 or args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("flush-every/world-size must be positive and rank must be valid")
    records = list(iter_jsonl(records_path))
    # Keyed by state_id: the collection manifest has no global_index, and matching
    # on it is what guarantees the instruction belongs to the state being pooled.
    task_instructions = (
        {row["state_id"]: row["instruction"] for row in iter_jsonl(args.instruction_records)}
        if args.instruction_records else {}
    )
    if task_instructions:
        missing = [r["state_id"] for r in records if r["state_id"] not in task_instructions]
        if missing:
            raise ValueError(
                f"{len(missing)} states have no task instruction, e.g. {missing[0]}; "
                "instruction priority would silently degrade to plain rotation"
            )
    input_shards = discover_full_h_shards(full_h_root, len(records))
    assigned = [
        shard for index, shard in enumerate(input_shards)
        if index % args.world_size == args.rank
    ]
    if not assigned:
        raise ValueError("rank received no Full-H shards")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    config = json.loads((Path(args.model) / "config.json").read_text())
    model = SimpleNamespace(config=SimpleNamespace(image_token_id=config["image_token_id"]))
    merge = int(config["vision_config"]["spatial_merge_size"])
    grid = tuple(args.image_grid_thw)
    started = time.time()
    completed = 0
    total_raw = total_pooled = total_long_states = total_skipped = 0
    output_paths = []

    for shard in assigned:
        source_summary_path = shard.path / "fixed-prompt-summary.json"
        if not source_summary_path.exists():
            raise FileNotFoundError(source_summary_path)
        source_summary = json.loads(source_summary_path.read_text())
        if Path(source_summary["model"]).resolve() != Path(args.model).resolve():
            raise ValueError(f"Full-H model mismatch in {shard.path}")
        if Path(source_summary["subset_manifest"]).resolve() != records_path:
            raise ValueError(f"Full-H manifest mismatch in {shard.path}")
        expected_merged = [grid[1] // merge, grid[2] // merge]
        if source_summary.get("key64", {}).get("merged_image_hw") != expected_merged:
            raise ValueError(f"Full-H image grid mismatch in {shard.path}")
        source_tokens = np.load(shard.path / "tokens-bf16.npy", mmap_mode="c")
        source_offsets = np.load(shard.path / "offsets.npy", mmap_mode="r")
        source_lengths = np.load(shard.path / "modality-lengths.npy", mmap_mode="r")
        output = output_root / shard.path.name
        output.mkdir(parents=True, exist_ok=True)
        output_paths.append(str(output))
        count = len(shard.subset_rows)
        bundle = [
            output / "subset_rows.npy",
            output / "record_indices.npy",
            output / "key64-static-bf16.npy",
            output / "key64-static-valid.npy",
            output / "key64-static-positions.npy",
            output / "key64-static-slot-kind.npy",
            output / "key64-static-detail-ranges.npy",
            output / "key64-static-audit.npy",
            output / "done.npy",
        ]
        existing = [path.exists() for path in bundle]
        if args.resume and any(existing) and not all(existing):
            raise RuntimeError(f"incomplete static rebuild bundle in {output}")
        if args.resume and all(existing):
            if not np.array_equal(np.load(bundle[0]), shard.subset_rows):
                raise ValueError(f"subset row mismatch in {output}")
            if not np.array_equal(np.load(bundle[1]), shard.global_indices):
                raise ValueError(f"record index mismatch in {output}")
        else:
            np.save(bundle[0], shard.subset_rows)
            np.save(bundle[1], shard.global_indices)
        arrays = {
            "tokens": _open_array(
                output / "key64-static-bf16.npy", np.uint16,
                (count, 64, 4096), args.resume,
            ),
            "valid": _open_array(
                output / "key64-static-valid.npy", np.bool_,
                (count, 64), args.resume,
            ),
            "positions": _open_array(
                output / "key64-static-positions.npy", np.float32,
                (count, 64), args.resume,
            ),
            "slot_kind": _open_array(
                output / "key64-static-slot-kind.npy", np.uint8,
                (count, 64), args.resume,
            ),
            "detail_ranges": _open_array(
                output / "key64-static-detail-ranges.npy", np.int32,
                (count, DETAIL_SLOTS, 2), args.resume,
            ),
            "audit": _open_array(
                output / "key64-static-audit.npy", np.int32,
                (count, 8), args.resume,
            ),
            "done": _open_array(
                output / "done.npy", np.bool_, (count,), args.resume,
            ),
        }
        if not args.resume:
            arrays["done"][:] = False
            arrays["done"].flush()
        for local, subset_row in enumerate(shard.subset_rows):
            if bool(arrays["done"][local]):
                continue
            record = records[int(subset_row)]
            if int(record["global_index"]) != int(shard.global_indices[local]):
                raise ValueError(f"record identity mismatch at subset row {subset_row}")
            start, stop = int(source_offsets[local]), int(source_offsets[local + 1])
            flat = bf16_tensor_view(source_tokens[start:stop]).to(device)
            lengths = [int(value) for value in source_lengths[local]]
            if sum(lengths) != len(flat):
                raise ValueError(f"Full-H length mismatch in {record['state_id']}")
            image_stop = lengths[0]
            dom_stop = image_stop + lengths[1]
            image_hidden = flat[:image_stop]
            dom_hidden = flat[image_stop:dom_stop]
            prompt_hidden = flat[dom_stop:]
            dom_offsets, prompt_tokens = _dom_offsets(
                processor, model, record, len(dom_hidden), args.prompt_mode
            )
            if prompt_tokens != len(prompt_hidden):
                raise ValueError(
                    f"prompt token mismatch for {record['state_id']}: "
                    f"{prompt_tokens} != {len(prompt_hidden)}"
                )
            result = build_static_key64(
                image_hidden, dom_hidden, prompt_hidden,
                dom=record["dom"],
                dom_token_offsets=dom_offsets,
                image_grid_thw=grid,
                spatial_merge_size=merge,
                filter_filler=args.filter_filler,
                instruction=task_instructions.get(record["state_id"]),
            )
            arrays["tokens"][local] = bf16_bits(result.tokens)
            arrays["valid"][local] = result.valid.cpu().numpy()
            arrays["positions"][local] = result.positions.cpu().numpy()
            arrays["slot_kind"][local] = result.slot_kind.cpu().numpy()
            arrays["detail_ranges"][local] = result.detail_ranges.cpu().numpy()
            audit = result.audit
            arrays["audit"][local] = np.asarray([
                audit["candidate_spans"],
                audit["short_candidate_spans"],
                audit["long_candidate_spans"],
                audit["selected_short_tokens"],
                audit["skipped_short_spans"],
                audit["raw_slots"],
                audit["pooled_long_slots"],
                audit["valid_detail_slots"],
            ], np.int32)
            arrays["done"][local] = True
            total_raw += audit["raw_slots"]
            total_pooled += audit["pooled_long_slots"]
            total_long_states += int(audit["selected_long"] is not None)
            total_skipped += audit["skipped_short_spans"]
            completed += 1
            if completed % args.flush_every == 0:
                for array in arrays.values():
                    array.flush()
                logging.info("rank=%d completed=%d", args.rank, completed)
        for array in arrays.values():
            array.flush()
        write_json(output / "key64-static-summary.json", {
            "protocol": STATIC_KEY64_PROTOCOL,
            "filter_filler": args.filter_filler,
            "rank": args.rank,
            "world_size": args.world_size,
            "records": count,
            # Derived, not a literal: a hardcoded layout kept reporting the old
            # (32,12,16,4) after the prompt band was recycled into detail, so the
            # manifest disagreed with the arrays it described.
            "layout": list(KEY64_LAYOUT),
            "image_grid_thw": list(grid),
            "instruction_priority": bool(task_instructions),
            "spatial_merge_size": merge,
            "model": str(Path(args.model).resolve()),
            "device": str(device),
            "records_manifest": str(Path(args.records).resolve()),
            "prompt_mode": args.prompt_mode,
        })
    summary = {
        "protocol": STATIC_KEY64_PROTOCOL,
        "filter_filler": args.filter_filler,
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned_shards": output_paths,
        "completed": completed,
        "raw_detail_slots": total_raw,
        "pooled_long_slots": total_pooled,
        "states_with_long_pool": total_long_states,
        "skipped_short_spans": total_skipped,
        "elapsed_seconds": time.time() - started,
    }
    write_json(Path(args.output) / f"key64-static-rebuild-rank{args.rank:02d}.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--full-h", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--image-grid-thw", nargs=3, type=int, default=(1, 20, 14))
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--instruction-records",
                        help="collection manifest carrying the real task instruction. "
                             "The feature manifest holds the fixed observation "
                             "prompt by design, so passing it here would rank "
                             "candidates against the wrong text and silently do "
                             "nothing. Omit to keep the pre-v8 rotation exactly")
    parser.add_argument("--filter-filler", action="store_true",
                        help="strip runs of lorem filler from detail candidates so a "
                             "value padded by filler falls back under the raw-slot "
                             "threshold instead of being pooled; off reproduces v5 "
                             "bit for bit")
    parser.add_argument("--prompt-mode", choices=("base", "instruct"), default="base")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error("rank must be in [0,world-size)")
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    print(json.dumps(rebuild(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
