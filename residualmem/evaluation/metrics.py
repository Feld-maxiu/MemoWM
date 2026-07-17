from __future__ import annotations

from collections.abc import Sequence

from residualmem.types import CanonicalState, StateSchema


def reconstruction_metrics(
    target: Sequence[CanonicalState],
    reconstructed: Sequence[CanonicalState],
    schema: StateSchema,
) -> dict[str, float]:
    """Measure reconstruction against ground truth, never against a proxy signal."""
    if len(target) != len(reconstructed):
        raise ValueError(
            f"trajectory length mismatch: {len(target)} != {len(reconstructed)}"
        )
    if not target:
        raise ValueError("cannot evaluate an empty trajectory")

    correct = 0
    count = 0
    exact_states = 0
    for truth, pred in zip(target, reconstructed, strict=True):
        schema.validate(truth)
        schema.validate(pred)
        exact_states += truth == pred
        for actual, estimated in zip(truth.values, pred.values, strict=True):
            correct += actual == estimated
            count += 1
    return {
        "exact_field_accuracy": float(correct / count),
        "exact_state_accuracy": float(exact_states / len(target)),
    }
