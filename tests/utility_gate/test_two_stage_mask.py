import numpy as np
import pytest

from experiments.utility_gate.export_mask import keep_mask
from experiments.utility_gate.two_stage_mask import combine_utilities, restore_variants, select_states


def test_restore_changes_only_one_dropped_code_and_keeps_background():
    truth = np.array([[1, 2], [3, 4]], np.uint8)
    fill = np.array([[5, 6], [7, 8]], np.uint8)
    keep = np.array([[True, False], [False, True]])
    variants = restore_variants(truth, fill, keep, [1, 2])
    assert variants.tolist() == [[[1, 6], [7, 4]], [[1, 2], [7, 4]], [[1, 6], [3, 4]]]
    with pytest.raises(ValueError, match='dropped'):
        restore_variants(truth, fill, keep, [0])
    with pytest.raises(ValueError, match='duplicate'):
        restore_variants(truth, fill, keep, [1, 1])


def test_noop_fill_is_exactly_identical_when_restored():
    truth = np.array([[1, 2]], np.uint8)
    variants = restore_variants(truth, truth.copy(), np.zeros_like(truth, bool), [0, 1])
    assert np.array_equal(variants[0], variants[1]) and np.array_equal(variants[0], variants[2])


def test_two_pass_envelope_equals_sequential_rescue_and_never_newly_drops():
    original = np.array([1., 2., 3., 4.])
    effective, conditional, measured = combine_utilities(original, [4., 1., 0., 8.], [1, 1, 0, 1])
    assert effective.tolist() == [4., 2., 3., 8.]
    assert conditional[2] == original[2] and not measured[2]
    entropy = np.array([[3., 3., 3., 3.], [5., 1., 6., 7.]])
    initial = keep_mask(original, entropy, 1.)
    sequential = initial | keep_mask(conditional, entropy, 1.)
    final = keep_mask(effective, entropy, 1.)
    assert np.array_equal(sequential, final)
    assert not (initial & ~final).any()


def test_low_support_is_not_treated_as_measured_zero():
    effective, conditional, measured = combine_utilities([1., 2.], [100., 0.], [1, 0], minimum_labels=2)
    assert effective.tolist() == conditional.tolist() == [1., 2.]
    assert not measured.any()


def test_selection_covers_rare_positions_and_is_deterministic():
    drop = np.zeros((8, 1024), bool)
    drop[:, 0] = True
    drop[7, 10] = True
    selected = select_states(np.arange(8), drop, [str(i) for i in range(8)], 4, 35)
    assert 7 in selected and len(selected) == 4
    assert np.array_equal(selected, select_states(np.arange(8), drop, [str(i) for i in range(8)], 4, 35))
