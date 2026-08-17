"""Latent state abstractions for ResidualMem v0.4 (technical-report line).

The technical report replaces the deterministic canonical state with a learned
*grouped categorical latent* ``z_t = (z_t^1, ..., z_t^N)`` where each group takes
one of ``C`` values. Structurally this is identical to a :class:`StateSchema`
whose fields are all ``categorical(num_values=C)`` -- so the existing schema
hashing/validation and the existing ``Predictor``/codec machinery apply directly
to latents (see ``latent_predictor`` and the exact codec as the lossless anchor).

``StateTokens`` carries a frozen state target and an explicit valid-slot mask.
The raw PCA state is ``x_t``; a fixed wrapper may convert it to normalized
``xbar_t`` before the world-model posterior ``q_phi(z_t | h_t, xbar_t, m_t)``.
``z_t`` is reserved for the grouped categorical stochastic latent. The front-end
(structured oracle first, Qwen3.5 + tokenizer later) remains swappable behind
this interface.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import numpy as np

from ..types import CanonicalState, FieldSpec, StateSchema


# Stable domain registry (technical report Eq. 4 / Section 6.3 domain token `d`).
# Indices are frozen so a trained domain embedding stays addressable.
DOMAINS: tuple[str, ...] = ("text", "web", "mobile", "crafter", "embodied")


@dataclasses.dataclass(frozen=True)
class DomainId:
    """A domain token `d` conditioning the world-model transition."""

    name: str

    def __post_init__(self) -> None:
        if self.name not in DOMAINS:
            raise ValueError(f"unknown domain {self.name!r}; expected one of {DOMAINS}")

    @property
    def index(self) -> int:
        return DOMAINS.index(self.name)

    @property
    def num_domains(self) -> int:
        return len(DOMAINS)


@dataclasses.dataclass(frozen=True)
class LatentSpec:
    """Shape of the grouped categorical latent plus its state-token target dims."""

    num_groups: int          # N latent groups
    num_categories: int      # C categories per group
    token_dim: int           # d_x of each state token
    num_tokens: int = 1      # K_x state tokens in x_t

    def __post_init__(self) -> None:
        if self.num_groups < 1:
            raise ValueError("num_groups must be >= 1")
        if self.num_categories < 2:
            raise ValueError("num_categories must be >= 2")
        if self.token_dim < 1 or self.num_tokens < 1:
            raise ValueError("token_dim and num_tokens must be >= 1")

    @property
    def schema_id(self) -> str:
        return f"latent-N{self.num_groups}-C{self.num_categories}"

    def make_schema(self) -> StateSchema:
        """A :class:`StateSchema` of N categorical(C) groups -- reuses hashing/validation."""
        fields = tuple(
            FieldSpec(f"z/{index}", "categorical", num_values=self.num_categories)
            for index in range(self.num_groups)
        )
        return StateSchema(self.schema_id, fields)


def make_latent_schema(num_groups: int, num_categories: int) -> StateSchema:
    """Convenience: build the latent :class:`StateSchema` directly."""
    return LatentSpec(num_groups, num_categories, token_dim=1).make_schema()


def latent_state(codes: Sequence[int], spec: LatentSpec) -> CanonicalState:
    """Wrap group codes as a validated latent :class:`CanonicalState`."""
    return spec.make_schema().make_state(tuple(int(code) for code in codes))


@dataclasses.dataclass(frozen=True)
class StateTokens:
    """The frozen mixed target ``x_t`` (report Eq. 9a).

    ``tokens`` has shape ``(num_tokens, token_dim)``. ``subspaces`` maps a name in
    {``sem``, ``ocr``, ``vis``, ``state``} to the row indices it occupies, so the
    codec's rate-distortion terms can weight subspaces differently without a free
    drifting latent. The structured front-end uses a single ``state`` subspace.
    """

    tokens: np.ndarray
    subspaces: Mapping[str, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        array = np.asarray(self.tokens, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"StateTokens.tokens must be 2D (K_x, d_x); got {array.shape}")
        object.__setattr__(self, "tokens", array)
        if not self.subspaces:
            object.__setattr__(
                self, "subspaces", {"state": tuple(range(array.shape[0]))}
            )
        valid = (
            np.ones((array.shape[0],), np.bool_)
            if self.valid is None else np.asarray(self.valid, np.bool_)
        )
        if valid.shape != (array.shape[0],):
            raise ValueError(
                f"StateTokens.valid must have shape ({array.shape[0]},); got {valid.shape}"
            )
        object.__setattr__(self, "valid", valid)

    @property
    def num_tokens(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def token_dim(self) -> int:
        return int(self.tokens.shape[1])

    def flat(self) -> np.ndarray:
        return self.tokens.reshape(-1)
