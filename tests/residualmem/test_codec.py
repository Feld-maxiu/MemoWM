from __future__ import annotations

from pathlib import Path

import pytest

from residualmem.codec.segment import (
    GLOBAL,
    DecodeError,
    LEGACY_MAGIC,
    LegacyFormatError,
    decode_memory,
    encode_memory,
)
from residualmem.types import FieldSpec, StateSchema, Trajectory
from residualmem.world_model.base import PersistencePredictor


def _schema() -> StateSchema:
    return StateSchema(
        "mixed-v3-exact",
        (
            FieldSpec("flag", "bool"),
            FieldSpec("kind", "categorical", num_values=5),
            FieldSpec("count", "integer"),
            FieldSpec("text", "literal"),
        ),
    )


def _trajectory(schema: StateSchema) -> Trajectory:
    rows = [
        (False, 0, 0, "start"),
        (True, 0, 2, "start"),
        (True, 4, -3, "novel"),
        (False, 2, 17, "novel"),
        (False, 2, 18, "start"),
        (True, 1, -100, "third"),
        (True, 3, 0, "third"),
        (False, 4, 1, "novel"),
    ]
    return Trajectory(
        tuple(schema.make_state(row) for row in rows),
        (1, 2, 3, 4, 5, 6, 7),
        "mixed",
    )


def test_exact_round_trip_across_segment_boundaries(tmp_path: Path):
    schema = _schema()
    source = _trajectory(schema)
    predictor = PersistencePredictor()
    path = tmp_path / "exact.rsm"

    written = encode_memory(
        path, source, predictor, schema, segment_length=3, action_values=17
    )
    decoded, read = decode_memory(path, predictor, schema)

    assert decoded.states == source.states
    assert decoded.actions == source.actions
    assert written == read
    assert written.total_bytes == path.stat().st_size


def test_v02_stream_is_explicitly_rejected(tmp_path: Path):
    schema = _schema()
    source = _trajectory(schema)
    predictor = PersistencePredictor()
    path = tmp_path / "legacy.rsm"

    encode_memory(path, source, predictor, schema, segment_length=3)
    legacy = bytearray(path.read_bytes())
    legacy[:8] = LEGACY_MAGIC
    path.write_bytes(legacy)

    with pytest.raises(LegacyFormatError, match="v0.2 streams are incompatible"):
        decode_memory(path, predictor, schema)


def test_corruption_and_versioned_hashes_are_rejected(tmp_path: Path):
    schema = _schema()
    source = _trajectory(schema)
    predictor = PersistencePredictor()
    path = tmp_path / "memory.rsm"
    encode_memory(path, source, predictor, schema, segment_length=3)

    corrupted = bytearray(path.read_bytes())
    corrupted[GLOBAL.size + 10] ^= 1
    damaged = tmp_path / "damaged.rsm"
    damaged.write_bytes(corrupted)
    with pytest.raises(DecodeError):
        decode_memory(damaged, predictor, schema)

    other_schema = StateSchema("other", schema.fields)
    with pytest.raises(DecodeError, match="schema hash mismatch"):
        decode_memory(path, predictor, other_schema)

    class OtherPredictor(PersistencePredictor):
        predictor_id = "other-predictor"

    with pytest.raises(DecodeError, match="world model hash mismatch"):
        decode_memory(path, OtherPredictor(), schema)


def test_external_actions_are_required_and_not_charged(tmp_path: Path):
    schema = _schema()
    source = _trajectory(schema)
    predictor = PersistencePredictor()
    path = tmp_path / "external.rsm"
    accounting = encode_memory(
        path,
        source,
        predictor,
        schema,
        segment_length=3,
        actions_external=True,
    )

    assert accounting.actions == 0
    with pytest.raises(DecodeError, match="external actions are required"):
        decode_memory(path, predictor, schema)
    decoded, _ = decode_memory(
        path, predictor, schema, external_actions=source.actions
    )
    assert decoded.states == source.states
    assert decoded.actions == source.actions
