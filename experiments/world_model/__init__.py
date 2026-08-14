"""Frozen-v8 discrete world-model evidence gate.

This package is deliberately independent from :mod:`residualmem.world_model.rssm`.
It predicts the deterministic A2 codes ``y_t`` and the observation validity mask;
it does not introduce an RSSM latent or modify the frozen state tokenizer.
"""

from .schema import (
    ACTION_TYPE_IDS,
    TAG_IDS,
    VARIANTS,
    Action,
    canonicalize_action,
)

__all__ = [
    "ACTION_TYPE_IDS",
    "TAG_IDS",
    "VARIANTS",
    "Action",
    "canonicalize_action",
]
