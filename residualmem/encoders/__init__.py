"""Observation encoders; optional JAX-backed modules are imported lazily."""
from __future__ import annotations

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

_MODULE_BY_NAME = {
    "ObservationEncoder": "base", "encoder_hash": "base",
    "StructuredStateEncoder": "structured",
    "GroupChannelNormalizer": "normalization",
    "NormalizedObservationEncoder": "normalization",
    "ObservationBackbone": "qwen", "MockBackbone": "qwen",
    "HFQwenBackbone": "qwen", "QwenObservationEncoder": "qwen",
}


def __getattr__(name: str):
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    from importlib import import_module
    return getattr(import_module(f"{__name__}.{module_name}"), name)
