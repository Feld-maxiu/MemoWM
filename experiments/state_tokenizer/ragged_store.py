"""Ragged Full-H storage and fixed-representation lookup for v2 probes."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

from .feature_store import discover_shards


def bf16_tensor_view(bits: np.ndarray) -> torch.Tensor:
    """Zero-conversion view of uint16 storage as BF16.

    Stores are opened copy-on-write so PyTorch sees a writable NumPy view while
    the on-disk cache remains immutable.
    """
    values = np.asarray(bits, dtype=np.uint16)
    return torch.from_numpy(values).view(torch.bfloat16)


@dataclass(frozen=True)
class FullHShard:
    path: Path
    subset_rows: np.ndarray
    global_indices: np.ndarray


def global_modality_lengths(features: str | Path, total_records: int) -> np.ndarray:
    destination = np.zeros((total_records, 3), dtype=np.int32)
    covered = np.zeros((total_records,), dtype=bool)
    for shard in discover_shards(features, total_records):
        values = np.load(shard.path / "modality-lengths.npy", mmap_mode="r")
        destination[shard.indices] = values
        covered[shard.indices] = True
    if not covered.all():
        raise ValueError("modality lengths do not cover the source manifest")
    return destination


def discover_full_h_shards(root: str | Path, expected_rows: int | None = None) -> list[FullHShard]:
    root = Path(root)
    shards = []
    all_rows = []
    for path in sorted(root.glob("worker*")):
        rows_path = path / "subset_rows.npy"
        indices_path = path / "record_indices.npy"
        done_path = path / "done.npy"
        if not rows_path.exists() or not indices_path.exists() or not done_path.exists():
            continue
        rows = np.asarray(np.load(rows_path), np.int64)
        indices = np.asarray(np.load(indices_path), np.int64)
        done = np.load(done_path, mmap_mode="r")
        offsets = np.load(path / "offsets.npy", mmap_mode="r")
        if len(rows) != len(indices) or len(done) != len(rows) or len(offsets) != len(rows) + 1:
            raise ValueError(f"invalid Full-H shard metadata: {path}")
        if not bool(np.asarray(done).all()):
            raise ValueError(f"incomplete Full-H shard: {path}")
        shards.append(FullHShard(path, rows, indices))
        all_rows.append(rows)
    if not shards:
        raise FileNotFoundError(f"no complete Full-H shards under {root}")
    combined = np.concatenate(all_rows)
    if len(np.unique(combined)) != len(combined):
        raise ValueError("duplicate subset rows across Full-H shards")
    if expected_rows is not None and not np.array_equal(np.sort(combined), np.arange(expected_rows)):
        missing = np.setdiff1d(np.arange(expected_rows), combined)
        raise ValueError(f"Full-H store misses subset rows: {missing[:20]}")
    return shards


class RaggedFullHStore:
    """Random access to sharded, flattened BF16 Full-H tokens."""

    def __init__(self, root: str | Path, expected_rows: int):
        self.shards = discover_full_h_shards(root, expected_rows)
        self.locations: dict[int, tuple[FullHShard, int]] = {}
        self._arrays: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for shard in self.shards:
            for local, row in enumerate(shard.subset_rows):
                self.locations[int(row)] = (shard, local)

    def _open(self, shard: FullHShard):
        if shard.path not in self._arrays:
            self._arrays[shard.path] = (
                np.load(shard.path / "tokens-bf16.npy", mmap_mode="c"),
                np.load(shard.path / "offsets.npy", mmap_mode="r"),
                np.load(shard.path / "modality-lengths.npy", mmap_mode="r"),
            )
        return self._arrays[shard.path]

    def get(self, subset_row: int) -> tuple[torch.Tensor, torch.Tensor]:
        shard, local = self.locations[int(subset_row)]
        tokens, offsets, modality_lengths = self._open(shard)
        start, stop = int(offsets[local]), int(offsets[local + 1])
        counts = torch.from_numpy(np.asarray(modality_lengths[local], np.int64))
        return (
            bf16_tensor_view(tokens[start:stop]),
            torch.repeat_interleave(torch.arange(3, dtype=torch.long), counts),
        )

    def get_modality(self, subset_row: int, modality: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= modality < 3:
            raise ValueError(modality)
        shard, local = self.locations[int(subset_row)]
        tokens, offsets, modality_lengths = self._open(shard)
        lengths = np.asarray(modality_lengths[local], np.int64)
        start = int(offsets[local] + lengths[:modality].sum())
        stop = start + int(lengths[modality])
        return (
            bf16_tensor_view(tokens[start:stop]),
            torch.full((stop - start,), modality, dtype=torch.long),
        )

    def modality_lengths(self) -> np.ndarray:
        result = np.empty((len(self.locations), 3), dtype=np.int32)
        for row, (shard, local) in self.locations.items():
            result[row] = self._open(shard)[2][local]
        return result

    def lengths(self) -> np.ndarray:
        result = np.empty((len(self.locations),), dtype=np.int32)
        for row, (shard, local) in self.locations.items():
            offsets = self._open(shard)[1]
            result[row] = int(offsets[local + 1] - offsets[local])
        return result


class FixedRepresentationStore:
    """Global-index lookup for existing Y64/Y32/PCA512 shards."""

    SPECS = {
        "y64": ("y64-bf16.npy", 64, 4096, (40, 20, 4)),
        "y32": ("y32-bf16.npy", 32, 4096, (20, 10, 2)),
        "x64": ("x64-pca-bf16.npy", 64, 512, (40, 20, 4)),
    }

    def __init__(self, root: str | Path, representation: str):
        if representation not in self.SPECS:
            raise ValueError(representation)
        self.filename, self.slots, self.width, self.layout = self.SPECS[representation]
        self.locations = {}
        self.arrays = {}
        for shard in discover_shards(root):
            if not (shard.path / self.filename).exists():
                raise FileNotFoundError(shard.path / self.filename)
            for local, global_index in enumerate(shard.indices):
                self.locations[int(global_index)] = (shard.path, local)

    def get(self, global_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, local = self.locations[int(global_index)]
        if path not in self.arrays:
            self.arrays[path] = np.load(path / self.filename, mmap_mode="c")
        tokens = bf16_tensor_view(self.arrays[path][local])
        modality_ids = torch.cat([
            torch.full((slots,), modality, dtype=torch.long)
            for modality, slots in enumerate(self.layout)
        ])
        return tokens, modality_ids


def normalized_modality_positions(modality_ids: torch.Tensor) -> torch.Tensor:
    """Return p=(i+0.5)/N independently inside each modality."""
    positions = torch.empty((len(modality_ids),), dtype=torch.float32)
    for modality in range(3):
        rows = torch.nonzero(modality_ids == modality, as_tuple=False).flatten()
        if len(rows):
            positions[rows] = (torch.arange(len(rows), dtype=torch.float32) + 0.5) / len(rows)
    if not bool(((positions > 0) & (positions < 1)).all()):
        raise ValueError("normalized positions must lie strictly inside (0,1)")
    return positions


def pad_token_batch(
    examples: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not examples:
        raise ValueError("cannot pad an empty batch")
    width = int(examples[0][0].shape[1])
    maximum = max(len(tokens) for tokens, _ in examples)
    batch = len(examples)
    tokens = torch.zeros((batch, maximum, width), dtype=torch.bfloat16)
    modalities = torch.zeros((batch, maximum), dtype=torch.long)
    positions = torch.zeros((batch, maximum), dtype=torch.float32)
    valid = torch.zeros((batch, maximum), dtype=torch.bool)
    for row, (value, modality) in enumerate(examples):
        if value.ndim != 2 or value.shape[1] != width or len(value) != len(modality):
            raise ValueError("inconsistent token example")
        stop = len(value)
        tokens[row, :stop] = value
        modalities[row, :stop] = modality
        positions[row, :stop] = normalized_modality_positions(modality)
        valid[row, :stop] = True
    return tokens, modalities, positions, valid
