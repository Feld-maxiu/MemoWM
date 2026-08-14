from __future__ import annotations

import math

import numpy as np

from experiments.world_model.baselines import (
    ALPHA,
    POSITIONS,
    SOURCE_PRIOR_CONCENTRATION,
    _task_rates,
)


def test_jeffreys_copy_and_source_probabilities_are_exact_and_finite():
    fit_source = np.zeros((2, 64, 32), np.uint8)
    fit_target = np.zeros_like(fit_source)
    eval_source = np.zeros((1, 64, 32), np.uint8)
    eval_target = np.zeros_like(eval_source)
    fit_mask = np.ones((2, 64), np.bool_)
    eval_mask = np.ones((1, 64), np.bool_)
    rates = _task_rates(
        fit_source, fit_target, fit_mask, fit_mask,
        eval_source, eval_target, eval_mask, eval_mask,
    )
    p_keep = (2 + ALPHA) / (2 + 2 * ALPHA)
    expected_copy = POSITIONS * -math.log2(p_keep)
    expected_source_probability = (
        2 + SOURCE_PRIOR_CONCENTRATION * p_keep
    ) / (2 + SOURCE_PRIOR_CONCENTRATION)
    expected_source = POSITIONS * -math.log2(expected_source_probability)
    assert abs(rates["copy_code_bits"][0] - expected_copy) < 1e-8
    assert abs(rates["source_code_bits"][0] - expected_source) < 1e-8
    assert all(np.isfinite(value).all() for value in rates.values())


def test_invalid_observation_mask_does_not_remove_any_code_codelength():
    fit_source = np.zeros((1, 64, 32), np.uint8)
    fit_target = np.ones_like(fit_source)
    eval_source = np.zeros((1, 64, 32), np.uint8)
    eval_target = np.ones_like(eval_source)
    all_valid = np.ones((1, 64), np.bool_)
    all_invalid = np.zeros((1, 64), np.bool_)
    left = _task_rates(
        fit_source, fit_target, all_valid, all_valid,
        eval_source, eval_target, all_valid, all_valid,
    )
    right = _task_rates(
        fit_source, fit_target, all_valid, all_valid,
        eval_source, eval_target, all_valid, all_invalid,
    )
    for name in ("marginal_code_bits", "copy_code_bits", "source_code_bits"):
        assert np.array_equal(left[name], right[name])

