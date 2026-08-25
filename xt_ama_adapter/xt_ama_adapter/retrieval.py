"""In-memory top-k retrieval over x_t document vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .adapter import AMAXTObservation
from .runtime import EMBEDDING_DIMENSION, _l2_normalize


@dataclass(frozen=True)
class XTRetrievalResult:
    """One selected AMA step, retaining its exact Reader evidence text."""

    rank: int
    score: float
    record: AMAXTObservation


class XTRetrievalIndex:
    """Immutable per-episode vector index for x_t-RAG."""

    def __init__(self, records: Sequence[AMAXTObservation], embeddings: np.ndarray) -> None:
        if not records:
            raise ValueError("x_t index needs at least one AMA step")
        values = _l2_normalize(embeddings)
        if values.ndim != 2 or values.shape != (len(records), EMBEDDING_DIMENSION):
            raise ValueError(
                f"embeddings must have shape ({len(records)}, {EMBEDDING_DIMENSION}), got {values.shape}"
            )
        self.records = tuple(records)
        self.embeddings = values

    def retrieve_vector(self, query_embedding: np.ndarray, top_k: int = 5) -> tuple[XTRetrievalResult, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = _l2_normalize(np.asarray(query_embedding, dtype=np.float32))
        if query.ndim != 1:
            raise ValueError(f"query embedding must be 1-D, got {query.shape}")
        scores = self.embeddings @ query
        # Stable ordering makes tied retrieval traces reproducible by original
        # AMA step order.
        order = np.argsort(-scores, kind="stable")[: min(top_k, len(scores))]
        return tuple(
            XTRetrievalResult(rank=rank + 1, score=float(scores[index]), record=self.records[index])
            for rank, index in enumerate(order)
        )

    @staticmethod
    def reader_context(results: Sequence[XTRetrievalResult]) -> str:
        """Format selected original steps for the shared AMA QA Reader."""
        return "\n\n".join(result.record.step_text for result in results)
