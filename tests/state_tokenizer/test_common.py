from __future__ import annotations

import numpy as np

from experiments.state_tokenizer.common import (
    SLOT_LAYOUTS,
    adaptive_average_pool,
    aggregate_mean_max,
    derive_probe_labels,
    pool_modalities,
)


def test_adaptive_pool_is_deterministic_and_handles_short_inputs():
    values = np.arange(3 * 2, dtype=np.float32).reshape(3, 2)
    first, mask = adaptive_average_pool(values, 5)
    second, second_mask = adaptive_average_pool(values, 5)
    assert first.shape == (5, 2)
    assert np.array_equal(first, second)
    assert np.array_equal(mask, second_mask)
    assert mask.all()
    assert np.isfinite(first).all()


def test_empty_modality_uses_zero_slots_and_false_mask():
    pooled, mask = adaptive_average_pool(np.zeros((0, 4), np.float32), 3)
    assert pooled.shape == (3, 4)
    assert not mask.any()
    assert not pooled.any()


def test_modal_layouts_have_expected_shapes():
    modalities = [
        np.ones((9, 8), np.float32),
        np.ones((5, 8), np.float32) * 2,
        np.ones((2, 8), np.float32) * 3,
    ]
    for slots in (64, 32):
        pooled, mask = pool_modalities(modalities, slots)
        assert pooled.shape == (slots, 8)
        assert mask.shape == (slots,)
        assert mask.all()
        assert sum(SLOT_LAYOUTS[slots]) == slots
        assert aggregate_mean_max(pooled, slots).shape == (6 * 8,)


def test_probe_labels_capture_checkbox_transition():
    observation = {
        "dom_elements": (
            {"tag": "input_checkbox", "value": True, "text": "", "classes": "", "id": "c"},
            {"tag": "button", "value": "", "text": "Submit", "classes": "", "id": "b"},
        )
    }
    labels = derive_probe_labels(observation, [
        {"tag": "input_checkbox", "checked": True, "selected": None,
         "disabled": False, "interactive": True, "role": ""}
    ])
    assert labels["state"]["role_checkbox"]
    assert labels["state"]["role_button"]
    assert labels["state"]["any_checked"]
    assert labels["state"]["any_enabled"]
    assert "submit" in labels["visible_words"]
