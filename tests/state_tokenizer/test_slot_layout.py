"""The 64-slot layout, checked from both dependency islands.

The layout is stated twice by necessity: the torch-side pooling reads it from
``experiments.state_tokenizer.slot_layout`` and the jax-side bottlenecks read
``residualmem.world_model.continuous_bottleneck``, and neither environment can
import the other's dependencies. A drift between the two mis-slices every group's
loss and normalization statistics without raising -- which is how a stale
(32, 12, 16, 4) survived the prompt band being recycled into detail, reporting
4890 valid "prompt" slots for a representation that has none.

Deliberately free of torch and jax imports so it runs under either interpreter.
"""
from __future__ import annotations

from experiments.state_tokenizer.slot_layout import GROUP_NAMES, KEY64_LAYOUT


def test_layout_sums_to_sixty_four():
    assert sum(KEY64_LAYOUT) == 64
    assert len(KEY64_LAYOUT) == len(GROUP_NAMES)
    assert all(size >= 0 for size in KEY64_LAYOUT)


def test_prompt_band_is_recycled_into_detail():
    sizes = dict(zip(GROUP_NAMES, KEY64_LAYOUT))
    assert sizes["prompt"] == 0
    assert sizes["detail"] == 16


def test_the_two_layout_definitions_agree():
    """Only runs where residualmem is importable; skipped otherwise."""
    try:
        from residualmem.world_model.continuous_bottleneck import (
            DEFAULT_GROUP_SIZES,
            GROUP_NAMES as MODEL_GROUPS,
        )
    except ImportError:
        import pytest

        pytest.skip("residualmem needs jax, absent in the extraction environment")
    assert DEFAULT_GROUP_SIZES == KEY64_LAYOUT
    assert MODEL_GROUPS == GROUP_NAMES
