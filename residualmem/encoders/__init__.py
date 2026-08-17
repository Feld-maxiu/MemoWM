"""Observation encoders producing the latent front-end target ``x_t``."""
from __future__ import annotations

from .base import ObservationEncoder, encoder_hash
from .structured import StructuredStateEncoder
from .normalization import GroupChannelNormalizer, NormalizedObservationEncoder
from .qwen import (
    HFQwenBackbone,
    MockBackbone,
    ObservationBackbone,
    QwenObservationEncoder,
)

__all__ = [
    "ObservationEncoder",
    "encoder_hash",
    "StructuredStateEncoder",
    "GroupChannelNormalizer",
    "NormalizedObservationEncoder",
    "ObservationBackbone",
    "MockBackbone",
    "HFQwenBackbone",
    "QwenObservationEncoder",
]
