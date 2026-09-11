from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from residualmem.benchmarks.wma_codec_reconstruction import (
    PROTOCOL,
    WmaCodecReconstructionCache,
)


def _write_cache(path):
    np.savez_compressed(
        path,
        state_ids=np.asarray(["web_01-0000", "web_01-0001"]),
        lookup_keys=np.asarray(["web_01-0000", "web_01_img_001", "web_01_img_002"]),
        lookup_rows=np.asarray([0, 0, 1], dtype=np.int64),
        gated_xbar=np.arange(2 * 32 * 512, dtype=np.float32).reshape(2, 32, 512),
        valid=np.ones((2, 32), dtype=np.bool_),
        metadata=np.asarray(json.dumps({"protocol": PROTOCOL})),
    )


def test_lookup_uses_official_image_id(tmp_path):
    path = tmp_path / "codec.npz"
    _write_cache(path)
    cache = WmaCodecReconstructionCache(path)
    row = cache.lookup(SimpleNamespace(
        image_ids=("web_01_img_002",), screenshot="/unused/web_01_img_002.png"
    ))
    assert row.state_id == "web_01-0001"
    assert row.xbar.shape == (32, 512)
    assert row.valid.all()


def test_lookup_fails_closed_for_unknown_observation(tmp_path):
    path = tmp_path / "codec.npz"
    _write_cache(path)
    cache = WmaCodecReconstructionCache(path)
    with pytest.raises(KeyError, match="absent"):
        cache.lookup(SimpleNamespace(image_ids=("unknown",), screenshot=None))
