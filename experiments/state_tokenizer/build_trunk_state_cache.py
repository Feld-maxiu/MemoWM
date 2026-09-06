"""Precompute exact BF16 layer states used by the QFormer trainer."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .extract_qwen import _load_model
from .trunk_state_cache import (
    TRUNK_STATE_CACHE_PROTOCOL,
    cache_name,
    input_fingerprint,
)
from .trunk_states import text_trunk_states, trunk_states


def records(directory: Path) -> list[tuple[str, int, dict]]:
    result: list[tuple[str, int, dict]] = []
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata"])))
        sample_id = str(metadata["sample_id"])
        result.extend((sample_id, index, record) for index, record in enumerate(metadata["records"]))
    if not result:
        raise FileNotFoundError(f"no observation records under {directory}")
    return result


def valid_existing(path: Path, fingerprint: str, layer: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata"])))
            return (
                metadata.get("protocol") == TRUNK_STATE_CACHE_PROTOCOL
                and metadata.get("input_fingerprint") == fingerprint
                and int(metadata.get("layer", -1)) == layer
                and data["hidden_bf16_bits"].dtype == np.uint16
                and data["hidden_bf16_bits"].ndim == 2
                and data["hidden_bf16_bits"].shape[1] == 4096
            )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world-size")

    source = Path(args.store)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = records(source)
    rows = all_rows[args.rank::args.world_size]
    if args.limit is not None:
        rows = rows[: args.limit]
    print(
        f"[trunk-cache] rank={args.rank}/{args.world_size} rows={len(rows)}/{len(all_rows)}",
        flush=True,
    )

    processor = model = None
    done = skipped = tokens = 0
    for sample_id, index, record in rows:
        target = output / cache_name(sample_id, index)
        fingerprint = input_fingerprint(sample_id, index, record, args.layer)
        if args.resume and target.exists() and valid_existing(target, fingerprint, args.layer):
            skipped += 1
            continue
        if model is None:
            processor, model = _load_model(
                args.model, torch.device(args.device), False, dtype=torch.bfloat16
            )
        if record.get("observation_protocol") == "qwen35_9b_visual_transcription_v1":
            state = text_trunk_states(
                processor, model, str(record["text_observation"]),
                layer=args.layer, device=torch.device(args.device),
            )
        else:
            with Image.open(record["screenshot"]) as handle:
                image = handle.convert("RGB")
            state = trunk_states(
                processor, model, image, record["synthetic_axtree"],
                layer=args.layer, device=torch.device(args.device),
            )
        hidden_bits = state.hidden.contiguous().view(torch.uint16).cpu().numpy()
        metadata = np.asarray(json.dumps({
            "protocol": TRUNK_STATE_CACHE_PROTOCOL,
            "sample_id": sample_id,
            "record_index": index,
            "input_fingerprint": fingerprint,
            "layer": args.layer,
            "model": str(Path(args.model).resolve()),
            "tokens": len(state),
        }, sort_keys=True))
        with tempfile.NamedTemporaryFile(dir=output, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            np.savez(
                temporary,
                hidden_bf16_bits=hidden_bits,
                modality_ids=state.modality_ids.to(torch.uint8).cpu().numpy(),
                positions=state.positions.cpu().numpy(),
                metadata=metadata,
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        done += 1
        tokens += len(state)
        if done % 25 == 0 or done == len(rows):
            print(f"[trunk-cache] done={done} skipped={skipped} tokens={tokens}", flush=True)
    print(f"[trunk-cache] complete done={done} skipped={skipped} tokens={tokens}", flush=True)


if __name__ == "__main__":
    main()
