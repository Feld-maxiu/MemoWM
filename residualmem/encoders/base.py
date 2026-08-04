"""Observation-encoder interface for the latent front-end (report Section 5).

An ``ObservationEncoder`` maps a single observation to the frozen mixed target
``x_t`` (:class:`StateTokens`). Two hard constraints from the report are baked
into the interface:

* The encoder is **stateless across steps** -- ``encode`` sees one observation
  only. Stateful backends (e.g. a Qwen VLM) must clear their KV cache / context
  in :meth:`reset` so a long context never becomes unbilled memory.
* History propagates only through the world-model state or Residual Memory, never
  through the encoder.

The structured front-end (:mod:`residualmem.encoders.structured`) is the first
implementation; the Qwen3.5 + Perceiver front-end swaps in behind this same
interface at Stage 5.
"""
from __future__ import annotations

import hashlib
from typing import Any, Protocol, runtime_checkable

from ..latent.types import DomainId, LatentSpec, StateTokens


@runtime_checkable
class ObservationEncoder(Protocol):
    encoder_id: str
    domain: DomainId
    spec: LatentSpec

    def reset(self) -> None:
        """Clear any per-episode/per-step context (no-op for stateless encoders)."""

    def encode(self, observation: Any) -> StateTokens:
        """Return the frozen mixed target ``x_t`` for one observation."""

    @property
    def hash_bytes(self) -> bytes:
        """32-byte identity billed into the stream header (frozen-target discipline)."""


def encoder_hash(encoder_id: str, spec: LatentSpec, domain: DomainId) -> bytes:
    digest = hashlib.sha256()
    digest.update(encoder_id.encode("utf-8"))
    digest.update(spec.schema_id.encode("utf-8"))
    digest.update(f"{spec.token_dim}x{spec.num_tokens}".encode("utf-8"))
    digest.update(domain.name.encode("utf-8"))
    return digest.digest()
