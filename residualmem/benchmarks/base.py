"""Benchmark harness scaffolding (report Section 14), Stage 6.

This is a *runnable interface*, not an external-dataset integration. Real benchmarks
(EMemBench, WorldMemArena, LongMemEval-V2) plug in by:
  1. building a ``LatentMemoryFile`` + ``MemoryIndex`` from their episodes, and
  2. emitting a list of :class:`BenchmarkExample` (query + retrieval plan + a Reader
     action script + the expected answer).
The export/probe head that turns reconstructed latents into readable evidence is the
front-end's ``decode_tokens`` (see :class:`ExportProbe`); its output must never feed
back into memory storage. ``run_benchmark`` then drives the *existing* QueryEngine and
scores grounded answers, so no benchmark-specific query logic is needed.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Protocol, runtime_checkable

from ..query.engine import QueryEngine, ScriptedReaderPolicy
from ..query.types import QueryPlan, ReaderAction
from ..types import CanonicalState, StateSchema


@runtime_checkable
class ExportProbe(Protocol):
    """Decodes a reconstructed token target x_tilde to readable evidence.

    For the structured front-end this is an exact inverse; for the multimodal
    front-end it is a (frozen) probe head attached as the encoder's ``renderer``.
    """

    def decode_tokens(self, tokens: Any) -> CanonicalState:
        ...


@dataclasses.dataclass(frozen=True)
class BenchmarkExample:
    query: str
    plan: QueryPlan
    reader_actions: tuple[ReaderAction, ...]
    expected: str
    query_embedding: Any | None = None


def run_benchmark(
    memory,
    index,
    schema: StateSchema,
    examples: tuple[BenchmarkExample, ...],
    *,
    embedding_model_id: str | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Score a list of examples against a (latent or v0.3) memory + index."""
    results: list[dict[str, Any]] = []
    correct = 0
    for example in examples:
        reader = ScriptedReaderPolicy(example.plan, list(example.reader_actions))
        result = QueryEngine(memory, index, schema, top_k=top_k).run(
            example.query, reader, example.query_embedding, embedding_model_id)
        ok = result.answer.strip() == example.expected.strip()
        correct += int(ok)
        results.append({
            "query": example.query,
            "answer": result.answer,
            "expected": example.expected,
            "correct": ok,
            "wm_steps": result.trace.wm_steps,
            "memory_bytes": result.trace.memory_bytes,
            "index_bytes": result.trace.index_bytes,
            "evidence_bytes": result.trace.evidence_bytes,
        })
    return {"accuracy": correct / max(1, len(examples)), "n": len(examples), "results": results}
