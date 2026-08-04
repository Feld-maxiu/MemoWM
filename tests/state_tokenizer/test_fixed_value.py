import math

from experiments.state_tokenizer.aggregate_fixed_value import (
    COMPRESSED,
    chance_adjusted_retention,
    conservative_retention,
)
from experiments.state_tokenizer.extract_fixed_prompt import cache_modality_lengths


def _result(value: float):
    return {"test": {"value": {"macro_position_accuracy": value}}}


def test_fixed_value_retention_subtracts_random_digit_chance():
    oracle = [_result(value) for value in (0.8, 0.9, 1.0)]
    candidate = [_result(value) for value in (0.45, 0.50, 0.55)]
    result = chance_adjusted_retention(candidate, oracle, "macro_position_accuracy")
    assert math.isclose(result["ratio_of_means"], 0.5)
    assert math.isclose(result["relative_loss"], 0.5)


def test_fixed_value_conservative_retention_uses_metadata_above_chance():
    oracle = [_result(value) for value in (0.8, 0.9, 1.0)]
    candidate = [_result(value) for value in (0.5, 0.6, 0.7)]
    metadata = [_result(value) for value in (0.2, 0.3, 0.4)]
    result = conservative_retention(
        candidate, oracle, metadata, "macro_position_accuracy"
    )
    assert math.isclose(result["ratio_of_means"], 0.5)
    assert math.isclose(result["baseline_mean"], 0.3)


def test_instruction_only_is_treated_as_a_compressed_state_representation():
    assert "instruction_only" in COMPRESSED


def test_instruction_only_cache_zeroes_prefix_modalities():
    import numpy as np

    expected = np.asarray([[40, 20, 29], [30, 10, 29]], np.int32)
    cached = cache_modality_lengths(expected, True)
    assert cached.tolist() == [[0, 0, 29], [0, 0, 29]]
    assert expected.tolist() == [[40, 20, 29], [30, 10, 29]]
