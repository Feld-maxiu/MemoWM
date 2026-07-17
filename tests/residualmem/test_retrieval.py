from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from residualmem.codec import MemoryFile, encode_memory
from residualmem.query import (
    Answer,
    EvidenceRef,
    Expand,
    QueryEngine,
    QueryPlan,
    ScriptedReaderPolicy,
)
from residualmem.retrieval import (
    MemoryIndex,
    build_memory_index,
    extract_segment_events,
)
from residualmem.types import FieldSpec, StateSchema, Trajectory
from residualmem.world_model import PersistencePredictor


def _fixture():
    schema = StateSchema(
        "retrieval-v3-exact",
        (
            FieldSpec("count", "integer"),
            FieldSpec("inventory/wood", "integer", num_values=10),
            FieldSpec("achievement/collect_wood", "bool"),
            FieldSpec("label", "literal"),
        ),
    )
    rows = (
        (0, 0, False, "start"),
        (1, 1, True, "forest"),
        (2, 1, True, "forest"),
        (3, 2, True, "table"),
        (4, 2, True, "table"),
        (5, 3, True, "furnace"),
        (6, 3, True, "furnace"),
        (7, 4, True, "done"),
    )
    trajectory = Trajectory(
        tuple(schema.make_state(row) for row in rows),
        (1, 2, 3, 4, 5, 6, 7),
        "retrieval",
    )
    return schema, trajectory


def _memory_and_index(tmp_path):
    schema, trajectory = _fixture()
    predictor = PersistencePredictor()
    memory_path = tmp_path / "memory.rsm"
    encode_memory(
        memory_path, trajectory, predictor, schema, segment_length=3
    )
    events = extract_segment_events(trajectory, schema, segment_length=3)
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32
    )
    index_path = tmp_path / "memory.rmi"
    build_memory_index(
        index_path,
        memory_path,
        events,
        schema,
        embeddings,
        "test-embedding-v1",
    )
    return schema, trajectory, predictor, memory_path, index_path


def test_segment_session_expands_to_next_residual_and_projects_fields(tmp_path):
    schema, trajectory, predictor, memory_path, _ = _memory_and_index(tmp_path)
    memory = MemoryFile(memory_path, predictor, schema)
    session = memory.open_session(0)

    anchor = session.view(("count",))
    assert anchor.frames[0].values == {"count": 0}
    chunk = session.expand(("count",), max_steps=8)
    assert chunk.reason == "next_residual"
    assert [frame.step for frame in chunk.frames] == [0, 1]
    assert chunk.frames[-1].values == {"count": 1}
    assert chunk.frames[-1].residual_applied

    while session.current_step < session.end_step:
        session.expand(("count",), max_steps=8)
    assert session.current_state == trajectory.states[3]
    revealed = session.reveal(1, 2, ("label",))
    assert [frame.values for frame in revealed.frames] == [
        {"label": "forest"},
        {"label": "forest"},
    ]


def test_structured_index_is_primary_and_dense_only_fills(tmp_path):
    schema, _, _, memory_path, index_path = _memory_and_index(tmp_path)
    index = MemoryIndex(index_path, schema, memory_path)
    try:
        plan = QueryPlan(
            "where did movement happen",
            actions=("move_left",),
            initial_fields=("count",),
        )
        candidates = index.retrieve(
            plan,
            np.asarray([0.0, 1.0], np.float32),
            "test-embedding-v1",
            top_k=2,
        )
        assert candidates[0].segment_id == 0
        assert candidates[0].source == "structured"
        assert candidates[1].segment_id == 1
        assert candidates[1].source == "dense"
    finally:
        index.close()

    connection = sqlite3.connect(index_path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()
    assert {"actions", "changed_fields", "literals", "entities", "segments"} <= tables
    assert "relations" not in tables
    assert "field_values" not in tables
    assert "event_steps" not in tables


def test_index_rejects_a_segment_layout_mismatch(tmp_path):
    schema, trajectory, _, memory_path, _ = _memory_and_index(tmp_path)
    wrong_events = extract_segment_events(trajectory, schema, segment_length=2)
    wrong_embeddings = np.ones((len(wrong_events), 2), np.float32)
    with pytest.raises(ValueError, match="Segment count|time range"):
        build_memory_index(
            tmp_path / "wrong.rmi",
            memory_path,
            wrong_events,
            schema,
            wrong_embeddings,
            "test-embedding-v1",
        )


def test_reader_progressively_expands_and_cites_shown_evidence(tmp_path):
    schema, _, predictor, memory_path, index_path = _memory_and_index(tmp_path)
    memory = MemoryFile(memory_path, predictor, schema)
    index = MemoryIndex(index_path, schema, memory_path)
    plan = QueryPlan(
        "when did count change",
        changed_fields=("count",),
        initial_fields=("count",),
    )
    reader = ScriptedReaderPolicy(
        plan,
        [
            Expand(("count",), max_steps=8),
            Answer(
                "count first changed at step 1",
                (EvidenceRef(0, 1, 1, ("count",)),),
            ),
        ],
    )
    try:
        result = QueryEngine(memory, index, schema, top_k=2).run(
            plan.query, reader
        )
    finally:
        index.close()

    assert result.answer == "count first changed at step 1"
    assert result.trace.tool_calls == 2
    assert result.trace.wm_steps == 1
    assert result.trace.residual_records == 1
    assert result.trace.segments_opened == 1
    assert result.trace.evidence_bytes > 0
