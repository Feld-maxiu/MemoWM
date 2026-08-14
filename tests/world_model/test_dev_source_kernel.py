"""The dense source kernel must reproduce the frozen baseline exactly.

Every downstream diagnostic that touches the source distribution -- the
residual model, the calibration comparison, the rank analysis -- is only
comparable to the published 8990.72 bits/transition if this holds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.world_model.cache import FrozenCache
from experiments.world_model.dev.source_kernel import (
    copy_kernel,
    fit_task,
    source_code_bits,
    source_kernel,
)

CACHE = Path("outputs/world_model/v8/cache")
BASELINE = Path("outputs/world_model/v8/baselines/validation/per_transition.npz")


def _cache():
    if not CACHE.exists():
        raise RuntimeError(f"frozen cache missing at {CACHE}")
    return FrozenCache(CACHE)


def test_copy_and_source_kernels_are_row_stochastic():
    cache = _cache()
    tables = fit_task(cache, 0, cache.indices_for_split("train"))
    positions = np.array([0, 1, 511, 1024, 2047])
    for kernel in (copy_kernel(tables, positions), source_kernel(tables, positions)):
        totals = kernel.sum(axis=-1)
        assert np.max(np.abs(totals - 1.0)) < 1e-9
        assert np.all(kernel > 0.0)


def test_dense_kernel_reproduces_frozen_source_code_bits():
    cache = _cache()
    with np.load(BASELINE) as handle:
        expected = handle["source_code_bits"]
        rows = handle["transition_indices"]
    actual = source_code_bits(
        cache, cache.indices_for_split("train"), rows,
    )
    assert actual.shape == expected.shape
    assert np.max(np.abs(actual - expected)) < 1e-9
