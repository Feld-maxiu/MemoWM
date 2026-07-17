"""ResidualMem: action-conditioned predictive memory with sparse corrections."""

from residualmem.schemas import crafter_schema
from residualmem.types import (
    CanonicalState,
    FieldSpec,
    Prediction,
    Predictor,
    StateSchema,
    Trajectory,
)

__version__ = "0.3.0"

__all__ = [
    "CanonicalState",
    "FieldSpec",
    "Prediction",
    "Predictor",
    "StateSchema",
    "Trajectory",
    "crafter_schema",
]
