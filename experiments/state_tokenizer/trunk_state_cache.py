"""Lossless on-disk cache for frozen Qwen3.5 layer-16 trunk states."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .trunk_states import TrunkStates


TRUNK_STATE_CACHE_PROTOCOL = "qwen35_full_trunk_states_bf16_v1"


def cache_name(sample_id: str, index: int) -> str:
    return f"{sample_id}-{index:04d}.npz"


def input_fingerprint(sample_id: str, index: int, record: dict, layer: int) -> str:
    """Bind a state to the exact image bytes, DOM text, prompt layer and row."""
    digest = hashlib.sha256()
    for value in (
        TRUNK_STATE_CACHE_PROTOCOL,
        sample_id,
        str(index),
        str(layer),
        str(record.get("synthetic_axtree", "")),
        str(record.get("text_observation", "")),
        str(record.get("observation_protocol", "")),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    screenshot_value = record.get("screenshot")
    if screenshot_value:
        screenshot = Path(str(screenshot_value))
        digest.update(hashlib.sha256(screenshot.read_bytes()).digest())
    else:
        digest.update(b"<NO_SCREENSHOT>")
    return digest.hexdigest()


class TrunkStateStore:
    """Load verified BF16 states once into CPU RAM, then transfer to the GPU."""

    def __init__(self, directory: str | Path, *, layer: int) -> None:
        self.directory = Path(directory)
        self.layer = int(layer)
        self._cpu: dict[tuple[str, int], TrunkStates] = {}

    def get(
        self, sample_id: str, index: int, record: dict, device: torch.device
    ) -> TrunkStates:
        key = (sample_id, int(index))
        if key not in self._cpu:
            path = self.directory / cache_name(*key)
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(np.asarray(data["metadata"])))
                if metadata.get("protocol") != TRUNK_STATE_CACHE_PROTOCOL:
                    raise ValueError(f"{path}: unexpected trunk cache protocol")
                expected = input_fingerprint(sample_id, index, record, self.layer)
                if metadata.get("input_fingerprint") != expected:
                    raise ValueError(f"{path}: input fingerprint mismatch")
                if int(metadata.get("layer", -1)) != self.layer:
                    raise ValueError(f"{path}: layer mismatch")
                bits = np.array(data["hidden_bf16_bits"], dtype=np.uint16, copy=True)
                modality = np.array(data["modality_ids"], dtype=np.int64, copy=True)
                positions = np.array(data["positions"], dtype=np.float32, copy=True)
            hidden = torch.from_numpy(bits).view(torch.bfloat16)
            state = TrunkStates(
                hidden=hidden,
                modality_ids=torch.from_numpy(modality),
                positions=torch.from_numpy(positions),
            )
            if state.hidden.ndim != 2 or state.hidden.shape[1] != 4096:
                raise ValueError(f"{path}: hidden shape {tuple(state.hidden.shape)}")
            if len(state.modality_ids) != len(state) or len(state.positions) != len(state):
                raise ValueError(f"{path}: inconsistent token lengths")
            if not torch.isfinite(state.hidden.float()).all() or not torch.isfinite(state.positions).all():
                raise ValueError(f"{path}: non-finite trunk state")
            self._cpu[key] = state
        state = self._cpu[key]
        return TrunkStates(
            hidden=state.hidden.to(device),
            modality_ids=state.modality_ids.to(device),
            positions=state.positions.to(device),
        )
