"""Latent-side abstractions (technical-report / v0.4 line)."""
from __future__ import annotations

from .format import FORMAT_VERSION, LEGACY_MAGICS, MAGIC
from .tokenizer import PerceiverConfig, initialize_perceiver, resample
from .types import (
    DOMAINS,
    DomainId,
    LatentSpec,
    StateTokens,
    latent_state,
    make_latent_schema,
)

__all__ = [
    "DOMAINS",
    "DomainId",
    "LatentSpec",
    "StateTokens",
    "latent_state",
    "make_latent_schema",
    "MAGIC",
    "FORMAT_VERSION",
    "LEGACY_MAGICS",
    "PerceiverConfig",
    "initialize_perceiver",
    "resample",
]
