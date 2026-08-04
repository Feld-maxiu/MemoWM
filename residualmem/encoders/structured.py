"""Structured (oracle) observation front-end.

Maps a canonical :class:`CanonicalState` (e.g. the 40-field Crafter schema) to the
frozen mixed target ``x_t`` (:class:`StateTokens`) and back. This is the first
implementation of the ``x_t`` seam: it lets the RSSM world model + latent codec
run end-to-end on structured input with no Qwen/multimodal dependency. The Qwen +
Perceiver front-end (Stage 5) swaps in behind the same :class:`ObservationEncoder`
protocol and produces the same ``StateTokens`` shape.

Per-field normalization mirrors ``world_model/gru.py:_state_features`` so the
target is bounded and well-scaled; the inverse (:meth:`decode_tokens`) denormalizes
a reconstructed ``x_tilde`` back to approximate field values for evidence rendering.
"""
from __future__ import annotations

import math

import numpy as np

from ..latent.types import LatentSpec, StateTokens, DomainId
from ..types import CanonicalState, StateSchema
from .base import encoder_hash


class StructuredStateEncoder:
    """Deterministic canonical-state <-> ``x_t`` codec (oracle front-end)."""

    def __init__(self, schema: StateSchema, latent_spec: LatentSpec, domain: DomainId):
        self.schema = schema
        self.domain = domain
        token_dim = len(schema.fields)
        # Structured x_t is a single state token whose width == field count.
        self.spec = LatentSpec(
            num_groups=latent_spec.num_groups,
            num_categories=latent_spec.num_categories,
            token_dim=token_dim,
            num_tokens=1,
        )
        self.encoder_id = f"structured-oracle-{schema.schema_id}"

    def reset(self) -> None:  # stateless
        pass

    @property
    def hash_bytes(self) -> bytes:
        return encoder_hash(self.encoder_id, self.spec, self.domain)

    def encode(self, observation: CanonicalState) -> StateTokens:
        self.schema.validate(observation)
        features = [self._normalize(spec, value)
                    for spec, value in zip(self.schema.fields, observation.values)]
        tokens = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return StateTokens(tokens, {"state": (0,)})

    def decode_tokens(self, tokens: np.ndarray) -> CanonicalState:
        """Inverse map: reconstructed x_tilde -> approximate canonical state."""
        flat = np.asarray(tokens, dtype=np.float32).reshape(-1)
        if flat.shape[0] != len(self.schema.fields):
            raise ValueError("token width does not match schema")
        values = [self._denormalize(spec, float(flat[index]))
                  for index, spec in enumerate(self.schema.fields)]
        return self.schema.make_state(values)

    def _normalize(self, spec, value) -> float:
        if value is None:
            return 0.0
        if spec.field_type == "bool":
            return 1.0 if int(value) else 0.0
        if spec.num_values is not None:
            return float(value) / max(1, spec.num_values - 1)
        # unbounded integer -> signed log compression
        v = float(value)
        return math.copysign(math.log1p(abs(v)), v)

    def _denormalize(self, spec, value: float):
        if spec.field_type == "bool":
            return 1 if value >= 0.5 else 0
        if spec.num_values is not None:
            scaled = round(value * max(1, spec.num_values - 1))
            return int(min(max(scaled, 0), spec.num_values - 1))
        magnitude = math.expm1(abs(value))
        return int(round(math.copysign(magnitude, value)))
