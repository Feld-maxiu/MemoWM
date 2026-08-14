from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.world_model.cache import CACHE_FILES, FrozenCache
from experiments.world_model.schema import PROTOCOL


def test_invalid_code_dtype_is_rejected_and_prompt_width_is_zero(tmp_path):
    np.save(tmp_path / CACHE_FILES["codes"], np.zeros((1, 64, 32), np.int16))
    np.save(tmp_path / CACHE_FILES["valid"], np.ones((1, 64), np.bool_))
    np.save(tmp_path / CACHE_FILES["global_indices"], np.arange(1, dtype=np.int64))
    np.savez_compressed(
        tmp_path / CACHE_FILES["transitions"],
        history_indices=np.empty((0, 7), np.int64),
        action_payloads=np.empty((0, 7, 40), np.uint8),
    )
    (tmp_path / CACHE_FILES["manifest"]).write_text(json.dumps({
        "protocol": PROTOCOL,
        "layout": [32, 16, 16, 0],
        "state_counts": {"train": 1, "validation": 0, "test": 0},
        "transitions": 0,
        "tasks": [],
        "episode_names": [],
        "artifact_sha256": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        FrozenCache(tmp_path)

