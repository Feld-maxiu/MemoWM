from __future__ import annotations

import dataclasses
import json
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from ..codec.session import EvidenceChunk, MemoryFile, SegmentDecodeSession
from ..types import StateSchema
from .types import (
    Answer,
    Candidate,
    EvidenceRef,
    Expand,
    Open,
    QueryPlan,
    ReaderAction,
    ReaderPolicy,
    Reveal,
    Switch,
)

if TYPE_CHECKING:
    from ..retrieval.index import MemoryIndex


@dataclasses.dataclass(frozen=True)
class QueryTrace:
    query: str
    plan: QueryPlan
    candidates: tuple[Candidate, ...]
    actions: tuple[dict, ...]
    tool_calls: int
    wm_steps: int
    residual_records: int
    evidence_bytes: int
    index_bytes: int
    memory_bytes: int
    segments_opened: int
    latency_ms: float

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class QueryResult:
    answer: str
    evidence_refs: tuple[EvidenceRef, ...]
    evidence: tuple[EvidenceChunk, ...]
    candidates: tuple[Candidate, ...]
    trace: QueryTrace

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class ScriptedReaderPolicy:
    """Deterministic Reader used by tests and offline trace replay."""

    def __init__(self, plan: QueryPlan, actions: list[ReaderAction]):
        self.query_plan = plan
        self.actions = list(actions)
        self.position = 0

    def plan(self, query: str, schema: StateSchema) -> QueryPlan:
        del schema
        return dataclasses.replace(self.query_plan, query=query)

    def act(self, query, candidates, evidence, trace) -> ReaderAction:
        del query, candidates, evidence, trace
        if self.position >= len(self.actions):
            raise RuntimeError("scripted Reader exhausted before ANSWER")
        action = self.actions[self.position]
        self.position += 1
        return action


class QueryEngine:
    def __init__(
        self,
        memory: MemoryFile,
        index: MemoryIndex,
        schema: StateSchema,
        top_k: int = 8,
        max_tool_calls: int = 32,
        default_expand_steps: int = 8,
    ):
        if top_k < 1 or max_tool_calls < 1 or default_expand_steps < 1:
            raise ValueError("query limits must be positive")
        self.memory = memory
        self.index = index
        self.schema = schema
        self.top_k = top_k
        self.max_tool_calls = max_tool_calls
        self.default_expand_steps = default_expand_steps

    def run(
        self,
        query: str,
        reader: ReaderPolicy,
        query_embedding: np.ndarray | None = None,
        query_embedding_model_id: str | None = None,
    ) -> QueryResult:
        started = time.perf_counter()
        plan = reader.plan(query, self.schema)
        self._validate_fields(plan.initial_fields)
        candidates = self.index.retrieve(
            plan,
            query_embedding=query_embedding,
            query_embedding_model_id=query_embedding_model_id,
            top_k=self.top_k,
        )
        sessions: dict[int, SegmentDecodeSession] = {}
        evidence: list[EvidenceChunk] = []
        trace_actions: list[dict] = []
        active_segment: int | None = None
        tool_calls = 0

        if candidates:
            session, chunk = self._open(
                candidates[0].segment_id, plan.initial_fields, candidates, sessions
            )
            active_segment = session.segment_id
            evidence.append(chunk)
            trace_actions.append(
                _trace_action("AUTO_OPEN", chunk, segment_id=active_segment,
                              fields=plan.initial_fields)
            )
            tool_calls += 1

        while True:
            action = reader.act(
                query,
                candidates,
                tuple(evidence),
                tuple(trace_actions),
            )
            if isinstance(action, Answer):
                if not action.text.strip():
                    raise ValueError("Reader answer must be non-empty")
                self._validate_evidence_refs(action.evidence, evidence)
                trace_actions.append(
                    {"action": "ANSWER", "answer": action.text,
                     "evidence": [dataclasses.asdict(ref) for ref in action.evidence]}
                )
                return self._result(
                    query,
                    plan,
                    candidates,
                    evidence,
                    trace_actions,
                    sessions,
                    tool_calls,
                    started,
                    action,
                )
            if tool_calls >= self.max_tool_calls:
                raise RuntimeError("Reader exceeded max_tool_calls without ANSWER")
            if isinstance(action, (Open, Switch)):
                session, chunk = self._open(
                    action.segment_id, action.fields, candidates, sessions
                )
                active_segment = session.segment_id
                name = "OPEN" if isinstance(action, Open) else "SWITCH"
            elif isinstance(action, Expand):
                segment_id = active_segment if action.segment_id is None else action.segment_id
                session = self._session(segment_id, sessions)
                self._validate_fields(action.fields)
                step_cap = (
                    self.default_expand_steps
                    if action.max_steps is None
                    else action.max_steps
                )
                chunk = session.expand(action.fields, step_cap)
                active_segment = session.segment_id
                name = "EXPAND"
            elif isinstance(action, Reveal):
                segment_id = active_segment if action.segment_id is None else action.segment_id
                session = self._session(segment_id, sessions)
                self._validate_fields(action.fields)
                chunk = session.reveal(
                    action.start_step, action.end_step, action.fields
                )
                active_segment = session.segment_id
                name = "REVEAL"
            else:
                raise TypeError(f"unsupported Reader action: {type(action).__name__}")
            evidence.append(chunk)
            trace_actions.append(
                _trace_action(name, chunk, **dataclasses.asdict(action))
            )
            tool_calls += 1

    def _open(self, segment_id, fields, candidates, sessions):
        self._validate_fields(fields)
        if segment_id not in {candidate.segment_id for candidate in candidates}:
            raise ValueError("Reader can only open retrieved candidate segments")
        session = sessions.get(segment_id)
        if session is None:
            session = self.memory.open_session(segment_id)
            sessions[segment_id] = session
        return session, session.view(fields)

    @staticmethod
    def _session(segment_id, sessions):
        if segment_id is None or segment_id not in sessions:
            raise ValueError("EXPAND/REVEAL requires an open segment")
        return sessions[segment_id]

    def _validate_fields(self, fields: tuple[str, ...]) -> None:
        if not fields:
            raise ValueError("Reader must select at least one state field")
        unknown = sorted(set(fields) - set(self.schema.names))
        if unknown:
            raise ValueError(f"unknown fields: {unknown}")

    @staticmethod
    def _validate_evidence_refs(refs, evidence):
        if not refs:
            raise ValueError("Reader answer must cite exact evidence")
        observed: dict[tuple[int, int], set[str]] = defaultdict(set)
        for chunk in evidence:
            for frame in chunk.frames:
                observed[(chunk.segment_id, frame.step)].update(frame.values)
        for ref in refs:
            if ref.start_step > ref.end_step or not ref.fields:
                raise ValueError("invalid evidence reference")
            for step in range(ref.start_step, ref.end_step + 1):
                if not set(ref.fields) <= observed[(ref.segment_id, step)]:
                    raise ValueError("answer cites fields that were not shown to Reader")

    def _result(
        self,
        query,
        plan,
        candidates,
        evidence,
        trace_actions,
        sessions,
        tool_calls,
        started,
        answer,
    ):
        trace = QueryTrace(
            query=query,
            plan=plan,
            candidates=candidates,
            actions=tuple(trace_actions),
            tool_calls=tool_calls,
            wm_steps=sum(session.wm_steps for session in sessions.values()),
            residual_records=sum(
                session.residual_records for session in sessions.values()
            ),
            evidence_bytes=sum(_evidence_bytes(chunk) for chunk in evidence),
            index_bytes=self.index.index_bytes,
            memory_bytes=self.memory.file_bytes,
            segments_opened=len(sessions),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        return QueryResult(
            answer.text,
            answer.evidence,
            tuple(evidence),
            candidates,
            trace,
        )


def _evidence_bytes(chunk: EvidenceChunk) -> int:
    return len(
        json.dumps(
            dataclasses.asdict(chunk),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _trace_action(name: str, chunk: EvidenceChunk, **arguments) -> dict:
    return {
        "action": name,
        "arguments": arguments,
        "result": {
            "segment_id": chunk.segment_id,
            "start_step": chunk.start_step,
            "end_step": chunk.end_step,
            "reason": chunk.reason,
            "frame_count": len(chunk.frames),
        },
    }
