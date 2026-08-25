"""Standalone, deterministic adapter from AMA trajectories to x_t inputs."""

from .adapter import (
    AMA_XT_OBSERVATION_PROTOCOL,
    AMAXTObservation,
    adapt_ama_step,
    adapt_ama_trajectory,
)
from .retrieval import XTRetrievalIndex, XTRetrievalResult
from .runtime import (
    QFormerRuntimeConfig,
    QFormerXTDocumentEncoder,
    QwenVLQueryEncoder,
    validate_artifact_pair,
)

__all__ = (
    "AMA_XT_OBSERVATION_PROTOCOL",
    "AMAXTObservation",
    "adapt_ama_step",
    "adapt_ama_trajectory",
    "QFormerRuntimeConfig",
    "QFormerXTDocumentEncoder",
    "QwenVLQueryEncoder",
    "XTRetrievalIndex",
    "XTRetrievalResult",
    "validate_artifact_pair",
)
