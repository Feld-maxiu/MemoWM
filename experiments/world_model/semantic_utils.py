"""Dependency-free validation helpers for decoded rollout stores."""
from __future__ import annotations


def check_shape_alignment(candidate, reference, rows) -> None:
    """Verify global-index/shape alignment while allowing predicted validity.

    Closed-loop stores deliberately contain a predicted observation mask. The
    generic semantic-eval checker requires valid-slot support to equal the clean
    store, which would reject exactly the mask errors this diagnostic must
    measure. Global-index lookup and every returned tensor shape still have to
    agree.
    """
    for row in rows:
        left = candidate.get(int(row))
        right = reference.get(int(row))
        if len(left) != len(right):
            raise ValueError(f"row {row}: store tuple lengths differ")
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            if left_value.shape != right_value.shape:
                raise ValueError(
                    f"row {row}, field {index}: shape {tuple(left_value.shape)} "
                    f"!= {tuple(right_value.shape)}"
                )
