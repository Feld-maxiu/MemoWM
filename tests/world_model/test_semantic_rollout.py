from __future__ import annotations

import numpy as np
import pytest

from experiments.world_model.semantic_utils import check_shape_alignment


class _Provider:
    def __init__(self, valid, *, slots=4):
        self.valid = np.asarray(valid, dtype=np.bool_)
        self.slots = slots

    def get(self, _row):
        return (
            np.ones((self.slots, 3), np.float32),
            np.zeros((self.slots,), np.int64),
            np.zeros((self.slots,), np.float32),
            self.valid,
        )


def test_rollout_alignment_allows_predicted_mask_errors():
    predicted = _Provider([True, False, True, False])
    clean = _Provider([True, True, False, False])
    check_shape_alignment(predicted, clean, [0])


def test_rollout_alignment_still_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        check_shape_alignment(
            _Provider([True, False, True], slots=3),
            _Provider([True, False, True, False], slots=4),
            [0],
        )
