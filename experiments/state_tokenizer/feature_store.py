"""Readers for sharded state-tokenizer features."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .common import modality_slices


@dataclass(frozen=True)
class FeatureShard:
    path: Path
    indices: np.ndarray


def discover_shards(root: str | Path, expected_records: int | None = None) -> list[FeatureShard]:
    root = Path(root)
    shards = []
    all_indices = []
    for path in sorted(root.glob("worker*")):
        index_path = path / "record_indices.npy"
        done_path = path / "done.npy"
        if not index_path.exists() or not done_path.exists():
            continue
        indices = np.load(index_path)
        done = np.load(done_path, mmap_mode="r")
        if len(indices) != len(done) or not bool(np.asarray(done).all()):
            raise ValueError(f"incomplete feature shard: {path}")
        shards.append(FeatureShard(path, np.asarray(indices, np.int64)))
        all_indices.append(np.asarray(indices, np.int64))
    if not shards:
        raise FileNotFoundError(f"no complete worker shards under {root}")
    combined = np.concatenate(all_indices)
    if len(np.unique(combined)) != len(combined):
        raise ValueError("duplicate record indices across feature shards")
    if expected_records is not None:
        expected = np.arange(expected_records, dtype=np.int64)
        if not np.array_equal(np.sort(combined), expected):
            missing = np.setdiff1d(expected, combined)
            raise ValueError(f"feature store does not cover manifest; missing={missing[:20]}")
    return shards


def decode_bf16_bits(bits: np.ndarray, device: str | torch.device = "cpu") -> torch.Tensor:
    unsigned = np.asarray(bits, dtype=np.uint16)
    tensor = torch.from_numpy(unsigned.copy()).view(torch.bfloat16)
    return tensor.to(device=device, dtype=torch.float32)


def encode_bf16_bits(values: torch.Tensor) -> np.ndarray:
    return values.to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()


def aggregate_tokens(tokens: torch.Tensor, total_slots: int) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1] != total_slots:
        raise ValueError(f"expected (batch,{total_slots},width), got {tuple(tokens.shape)}")
    parts = []
    for section in modality_slices(total_slots):
        item = tokens[:, section, :]
        parts.extend((item.mean(dim=1), item.amax(dim=1)))
    return torch.cat(parts, dim=-1)


def materialize_probe_features(
    root: str | Path,
    representation: str,
    total_records: int,
    output: str | Path,
    *,
    batch_size: int = 32,
) -> Path:
    """Create a global-index-aligned FP16 matrix used by the linear probes."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        array = np.load(output, mmap_mode="r")
        expected_width = 6 * (512 if representation == "pca64" else 4096)
        if array.shape == (total_records, expected_width) and array.dtype == np.float16:
            return output
        raise ValueError(f"existing probe feature mismatch: {output}")
    width = 6 * (512 if representation == "pca64" else 4096)
    destination = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float16, shape=(total_records, width)
    )
    shards = discover_shards(root, total_records)
    for shard in shards:
        if representation == "h":
            source = np.load(shard.path / "h-stats-fp16.npy", mmap_mode="r")
            if source.shape[1:] != (6, 4096):
                raise ValueError(f"invalid H stats in {shard.path}")
            destination[shard.indices] = np.asarray(source).reshape(len(source), -1)
            continue
        if representation == "y64":
            filename, slots = "y64-bf16.npy", 64
        elif representation == "y32":
            filename, slots = "y32-bf16.npy", 32
        elif representation == "pca64":
            filename, slots = "x64-pca-bf16.npy", 64
        else:
            raise ValueError(f"unknown representation: {representation}")
        source = np.load(shard.path / filename, mmap_mode="r")
        for start in range(0, len(source), batch_size):
            stop = min(start + batch_size, len(source))
            tokens = decode_bf16_bits(source[start:stop])
            features = aggregate_tokens(tokens, slots).to(torch.float16).numpy()
            destination[shard.indices[start:stop]] = features
    destination.flush()
    return output


def iter_y64_batches(
    root: str | Path,
    *,
    batch_states: int = 16,
    selected_indices: set[int] | None = None,
    device: str | torch.device = "cpu",
) -> Iterator[tuple[np.ndarray, torch.Tensor]]:
    """Yield global indices and decoded Y64 batches from complete shards."""
    for shard in discover_shards(root):
        source = np.load(shard.path / "y64-bf16.npy", mmap_mode="r")
        if selected_indices is None:
            local_rows = np.arange(len(source), dtype=np.int64)
        else:
            local_rows = np.asarray([
                local for local, global_index in enumerate(shard.indices)
                if int(global_index) in selected_indices
            ], dtype=np.int64)
        for start in range(0, len(local_rows), batch_states):
            rows = local_rows[start:start + batch_states]
            if len(rows) == 0:
                continue
            yield shard.indices[rows], decode_bf16_bits(source[rows], device=device)
