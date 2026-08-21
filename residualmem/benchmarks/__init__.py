"""Benchmark APIs, kept lazy so the Qwen-only runtime does not require JAX."""
from __future__ import annotations

from .worldmemarena_tokenizer import (
    WMA_OBSERVATION_PROTOCOL,
    WebObservation,
    fused_observation_text,
    genuine_user_text,
    observation_from_turn,
    synthetic_axtree,
)

__all__ = [
    "BenchmarkExample",
    "ExportProbe",
    "run_benchmark",
    "WMA_OBSERVATION_PROTOCOL",
    "WebObservation",
    "fused_observation_text",
    "genuine_user_text",
    "observation_from_turn",
    "synthetic_axtree",
]


def __getattr__(name: str):
    if name in {"BenchmarkExample", "ExportProbe", "run_benchmark"}:
        from . import base
        return getattr(base, name)
    raise AttributeError(name)
