"""Strict lookup for materialized WMA closed-loop codec reconstructions."""
from __future__ import annotations

import dataclasses
import json
from functools import lru_cache
from pathlib import Path

import numpy as np


PROTOCOL = "residualmem_wma_closed_loop_reconstruction_v1"


@dataclasses.dataclass(frozen=True)
class CodecReconstruction:
    state_id: str
    xbar: np.ndarray
    valid: np.ndarray


class WmaCodecReconstructionCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        with np.load(self.path, allow_pickle=False) as data:
            self.state_ids = np.asarray(data["state_ids"], dtype=np.str_)
            self.xbar = np.asarray(data["gated_xbar"], dtype=np.float32)
            self.valid = np.asarray(data["valid"], dtype=np.bool_)
            keys = np.asarray(data["lookup_keys"], dtype=np.str_)
            rows = np.asarray(data["lookup_rows"], dtype=np.int64)
            self.metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        if self.metadata.get("protocol") != PROTOCOL:
            raise ValueError(
                f"codec cache protocol {self.metadata.get('protocol')!r} != {PROTOCOL!r}"
            )
        if self.xbar.shape != (len(self.state_ids), 32, 512):
            raise ValueError(f"invalid codec xbar shape {self.xbar.shape}")
        if self.valid.shape != self.xbar.shape[:2]:
            raise ValueError(f"invalid codec validity shape {self.valid.shape}")
        if keys.shape != rows.shape or np.any(rows < 0) or np.any(rows >= len(self.xbar)):
            raise ValueError("invalid codec lookup table")
        self._row_by_key = {str(key): int(row) for key, row in zip(keys, rows)}
        if len(self._row_by_key) != len(keys):
            raise ValueError("codec lookup table contains duplicate keys")

    def __len__(self) -> int:
        return len(self.state_ids)

    def lookup(self, observation) -> CodecReconstruction:
        candidates = [str(value) for value in observation.image_ids if value]
        if observation.screenshot:
            candidates.append(Path(observation.screenshot).stem)
        matched = {self._row_by_key[value] for value in candidates if value in self._row_by_key}
        if not matched:
            raise KeyError(
                "observation is absent from the closed-loop codec cache; tried "
                + ", ".join(repr(value) for value in candidates)
            )
        if len(matched) != 1:
            raise ValueError(f"observation keys resolve to different codec rows: {matched}")
        row = matched.pop()
        return CodecReconstruction(
            state_id=str(self.state_ids[row]),
            xbar=self.xbar[row],
            valid=self.valid[row],
        )


@lru_cache(maxsize=None)
def load_wma_codec_reconstruction_cache(path: str | Path) -> WmaCodecReconstructionCache:
    return WmaCodecReconstructionCache(Path(path).resolve())
