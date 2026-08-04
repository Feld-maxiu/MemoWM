from __future__ import annotations

import math

from experiments.state_tokenizer.aggregate_value_only import retention
from experiments.state_tokenizer.slot_probe import DYNAMIC_STATE_LABELS


def _result(value: float, exact: float = 0.0):
    return {
        "test": {
            "value": {
                "macro_position_accuracy": value,
                "exact_value_accuracy": exact,
            }
        }
    }


def test_value_retention_uses_seed_paired_instruction_leakage():
    oracle = [_result(value) for value in (0.8, 0.9, 1.0)]
    leakage = [_result(value) for value in (0.2, 0.3, 0.4)]
    candidate = [_result(value) for value in (0.5, 0.6, 0.7)]
    result = retention(candidate, oracle, leakage, "macro_position_accuracy")
    assert math.isclose(result["ratio_of_means"], 0.5)
    assert math.isclose(result["paired_seed"]["mean"], 0.5)


def test_dynamic_state_objective_excludes_template_and_random_value_labels():
    assert "checkbox_0_checked" in DYNAMIC_STATE_LABELS
    assert "textbox_0_focused" in DYNAMIC_STATE_LABELS
    assert "textbox_0_nonempty" in DYNAMIC_STATE_LABELS
    assert "interactive_tampered" not in DYNAMIC_STATE_LABELS
    assert "has_random_value" not in DYNAMIC_STATE_LABELS
