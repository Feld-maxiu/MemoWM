from __future__ import annotations

from collections.abc import Sequence

from residualmem.codec.residual import field_distortion
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
    weighted_error = 0.0
    weighted_count = 0.0
    for truth, pred in zip(target, reconstructed, strict=True):
        schema.validate(truth)
        schema.validate(pred)
        for field, actual, estimated in zip(
            schema.fields, truth.values, pred.values, strict=True
        ):
            correct += actual == estimated
            count += 1
            weighted_error += field.weight * field_distortion(
                field, actual, estimated
            )
            weighted_count += field.weight
    return {
        "exact_field_accuracy": float(correct / count),
        "weighted_distortion": float(weighted_error / max(weighted_count, 1e-12)),
    }
