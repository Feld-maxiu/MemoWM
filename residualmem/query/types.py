from __future__ import annotations

import dataclasses
from typing import Protocol

from ..codec.session import EvidenceChunk
from ..types import StateSchema


@dataclasses.dataclass(frozen=True)
class QueryPlan:
    query: str
    actions: tuple[str, ...] = ()
    changed_fields: tuple[str, ...] = ()
    literals: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    time_start: int | None = None
    time_end: int | None = None
    initial_fields: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Candidate:
    segment_id: int
    start_step: int
    end_step: int
    source: str
    structured_score: float
    dense_score: float | None
    matched_actions: tuple[str, ...] = ()
    matched_fields: tuple[str, ...] = ()
    matched_literals: tuple[str, ...] = ()
    matched_entities: tuple[str, ...] = ()
    time_overlap: bool = False


@dataclasses.dataclass(frozen=True)
class EvidenceRef:
    segment_id: int
    start_step: int
    end_step: int
    fields: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Open:
    segment_id: int
    fields: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Expand:
    fields: tuple[str, ...]
    max_steps: int | None = None
    segment_id: int | None = None


@dataclasses.dataclass(frozen=True)
class Reveal:
    start_step: int
    end_step: int
    fields: tuple[str, ...]
    segment_id: int | None = None


@dataclasses.dataclass(frozen=True)
class Switch:
    segment_id: int
    fields: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Answer:
    text: str
    evidence: tuple[EvidenceRef, ...]


ReaderAction = Open | Expand | Reveal | Switch | Answer


class ReaderPolicy(Protocol):
    """Abstract policy boundary; no concrete LLM dependency is required."""

    def plan(self, query: str, schema: StateSchema) -> QueryPlan:
        ...

    def act(
        self,
        query: str,
        candidates: tuple[Candidate, ...],
        evidence: tuple[EvidenceChunk, ...],
        trace: tuple[dict, ...],
    ) -> ReaderAction:
        ...
