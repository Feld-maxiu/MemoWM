from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..codec.segment import FORMAT_VERSION, GLOBAL, INDEX, MAGIC
from ..query.types import Candidate, QueryPlan
from ..types import StateSchema
from .events import SegmentEvents


INDEX_VERSION = 1


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def model_id(self) -> str:
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        ...


def embed_segment_documents(
    events: tuple[SegmentEvents, ...], provider: EmbeddingProvider
) -> np.ndarray:
    return np.asarray(provider.embed([event.document() for event in events]))


def build_memory_index(
    path: str | Path,
    memory_path: str | Path,
    events: tuple[SegmentEvents, ...],
    schema: StateSchema,
    embeddings: np.ndarray,
    embedding_model_id: str,
) -> Path:
    if not embedding_model_id:
        raise ValueError("embedding_model_id must be non-empty")
    if tuple(event.segment_id for event in events) != tuple(range(len(events))):
        raise ValueError("events must have contiguous segment ids")
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(events) or values.shape[1] < 1:
        raise ValueError("embeddings must have shape [segment_count, dimension]")
    if not np.isfinite(values).all():
        raise ValueError("embeddings must be finite")
    _validate_memory_layout(memory_path, events, schema)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE segments (
              segment_id INTEGER PRIMARY KEY,
              start_step INTEGER NOT NULL,
              end_step INTEGER NOT NULL,
              document TEXT NOT NULL,
              embedding BLOB NOT NULL,
              embedding_scale REAL NOT NULL
            );
            CREATE TABLE actions (
              value TEXT NOT NULL, segment_id INTEGER NOT NULL,
              PRIMARY KEY (value, segment_id),
              FOREIGN KEY(segment_id) REFERENCES segments(segment_id)
            );
            CREATE TABLE changed_fields (
              value TEXT NOT NULL, segment_id INTEGER NOT NULL,
              PRIMARY KEY (value, segment_id),
              FOREIGN KEY(segment_id) REFERENCES segments(segment_id)
            );
            CREATE TABLE literals (
              value TEXT NOT NULL, segment_id INTEGER NOT NULL,
              PRIMARY KEY (value, segment_id),
              FOREIGN KEY(segment_id) REFERENCES segments(segment_id)
            );
            CREATE TABLE entities (
              value TEXT NOT NULL, segment_id INTEGER NOT NULL,
              PRIMARY KEY (value, segment_id),
              FOREIGN KEY(segment_id) REFERENCES segments(segment_id)
            );
            CREATE INDEX actions_segment ON actions(segment_id);
            CREATE INDEX fields_segment ON changed_fields(segment_id);
            CREATE INDEX literals_segment ON literals(segment_id);
            CREATE INDEX entities_segment ON entities(segment_id);
            CREATE INDEX segment_time ON segments(start_step, end_step);
            """
        )
        metadata = {
            "index_version": str(INDEX_VERSION),
            "memory_sha256": _sha256_file(memory_path),
            "schema_hash": schema.hash_hex,
            "embedding_model_id": embedding_model_id,
            "embedding_dimension": str(values.shape[1]),
            "segment_count": str(len(events)),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        for event, embedding in zip(events, values, strict=True):
            quantized, scale = _quantize_embedding(embedding)
            connection.execute(
                "INSERT INTO segments VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event.segment_id,
                    event.start_step,
                    event.end_step,
                    event.document(),
                    quantized.tobytes(),
                    scale,
                ),
            )
            for table, event_values in (
                ("actions", event.actions),
                ("changed_fields", event.changed_fields),
                ("literals", event.literals),
                ("entities", event.entities),
            ):
                connection.executemany(
                    f"INSERT INTO {table}(value, segment_id) VALUES (?, ?)",
                    ((value, event.segment_id) for value in event_values),
                )
        connection.commit()
    finally:
        connection.close()
    return path


class MemoryIndex:
    def __init__(
        self,
        path: str | Path,
        schema: StateSchema,
        memory_path: str | Path | None = None,
    ):
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        try:
            self.metadata = dict(
                self.connection.execute("SELECT key, value FROM metadata")
            )
            if int(self.metadata.get("index_version", -1)) != INDEX_VERSION:
                raise ValueError("unsupported index version")
            if self.metadata.get("schema_hash") != schema.hash_hex:
                raise ValueError("index schema hash mismatch")
            if memory_path is not None:
                if self.metadata.get("memory_sha256") != _sha256_file(memory_path):
                    raise ValueError("index does not belong to this memory stream")
        except Exception:
            self.connection.close()
            raise

    @property
    def index_bytes(self) -> int:
        return self.path.stat().st_size

    @property
    def embedding_model_id(self) -> str:
        return self.metadata["embedding_model_id"]

    def close(self) -> None:
        self.connection.close()

    def retrieve(
        self,
        plan: QueryPlan,
        query_embedding: np.ndarray | None = None,
        query_embedding_model_id: str | None = None,
        top_k: int = 8,
    ) -> tuple[Candidate, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        matches: dict[int, dict[str, set[str] | bool]] = {}
        criterion_count = 0
        for table, values, label in (
            ("actions", tuple(dict.fromkeys(plan.actions)), "actions"),
            (
                "changed_fields",
                tuple(dict.fromkeys(plan.changed_fields)),
                "fields",
            ),
            ("literals", tuple(dict.fromkeys(plan.literals)), "literals"),
            ("entities", tuple(dict.fromkeys(plan.entities)), "entities"),
        ):
            criterion_count += len(values)
            for value in values:
                for row in self.connection.execute(
                    f"SELECT segment_id FROM {table} WHERE value = ?", (value,)
                ):
                    bucket = matches.setdefault(row[0], _empty_matches())
                    bucket[label].add(value)
        has_time = plan.time_start is not None or plan.time_end is not None
        if has_time:
            criterion_count += 1
            lower = 0 if plan.time_start is None else plan.time_start
            upper = (2**63 - 1) if plan.time_end is None else plan.time_end
            if lower > upper:
                raise ValueError("query time_start must not exceed time_end")
            for row in self.connection.execute(
                "SELECT segment_id FROM segments "
                "WHERE end_step >= ? AND start_step <= ?",
                (lower, upper),
            ):
                bucket = matches.setdefault(row[0], _empty_matches())
                bucket["time"] = True

        dense_scores = self._dense_scores(
            query_embedding, query_embedding_model_id
        )
        structured = []
        for segment_id, found in matches.items():
            row = self.connection.execute(
                "SELECT start_step, end_step FROM segments WHERE segment_id = ?",
                (segment_id,),
            ).fetchone()
            matched_count = sum(
                len(found[key]) for key in ("actions", "fields", "literals", "entities")
            ) + int(found["time"])
            structured.append(
                Candidate(
                    segment_id=segment_id,
                    start_step=row[0],
                    end_step=row[1],
                    source="structured",
                    structured_score=matched_count / max(1, criterion_count),
                    dense_score=dense_scores.get(segment_id),
                    matched_actions=tuple(sorted(found["actions"])),
                    matched_fields=tuple(sorted(found["fields"])),
                    matched_literals=tuple(sorted(found["literals"])),
                    matched_entities=tuple(sorted(found["entities"])),
                    time_overlap=bool(found["time"]),
                )
            )
        structured.sort(
            key=lambda item: (-item.structured_score, item.start_step, item.segment_id)
        )
        output = structured[:top_k]
        used = {candidate.segment_id for candidate in output}
        for segment_id, score in sorted(
            dense_scores.items(), key=lambda item: (-item[1], item[0])
        ):
            if len(output) >= top_k:
                break
            if segment_id in used:
                continue
            row = self.connection.execute(
                "SELECT start_step, end_step FROM segments WHERE segment_id = ?",
                (segment_id,),
            ).fetchone()
            output.append(
                Candidate(
                    segment_id,
                    row[0],
                    row[1],
                    "dense",
                    0.0,
                    score,
                )
            )
        return tuple(output)

    def _dense_scores(
        self,
        query_embedding: np.ndarray | None,
        query_embedding_model_id: str | None,
    ) -> dict[int, float]:
        if query_embedding is None:
            return {}
        if query_embedding_model_id != self.embedding_model_id:
            raise ValueError("query/index embedding model mismatch")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        dimension = int(self.metadata["embedding_dimension"])
        if query.shape != (dimension,) or not np.isfinite(query).all():
            raise ValueError("query embedding has wrong shape or non-finite values")
        norm = float(np.linalg.norm(query))
        if norm == 0:
            raise ValueError("query embedding must be non-zero")
        query = query / norm
        scores = {}
        for row in self.connection.execute(
            "SELECT segment_id, embedding, embedding_scale FROM segments"
        ):
            vector = np.frombuffer(row[1], dtype=np.int8).astype(np.float32)
            vector *= float(row[2])
            scores[int(row[0])] = float(np.dot(query, vector))
        return scores


def _empty_matches() -> dict[str, set[str] | bool]:
    return {
        "actions": set(),
        "fields": set(),
        "literals": set(),
        "entities": set(),
        "time": False,
    }


def _quantize_embedding(value: np.ndarray) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(value))
    normalized = value / norm if norm else np.zeros_like(value)
    peak = float(np.max(np.abs(normalized)))
    scale = peak / 127.0 if peak else 1.0
    quantized = np.rint(normalized / scale).clip(-127, 127).astype(np.int8)
    return quantized, scale


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_memory_layout(
    memory_path: str | Path,
    events: tuple[SegmentEvents, ...],
    schema: StateSchema,
) -> None:
    path = Path(memory_path)
    with path.open("rb") as handle:
        header = handle.read(GLOBAL.size)
        if len(header) != GLOBAL.size:
            raise ValueError("memory stream has a truncated global header")
        (
            magic,
            version,
            schema_hash,
            _,
            _,
            _,
            _,
            segment_count,
            index_offset,
        ) = GLOBAL.unpack(header)
        if magic != MAGIC or version != FORMAT_VERSION:
            raise ValueError("index building requires a v0.3 memory stream")
        if schema_hash != schema.hash_bytes:
            raise ValueError("memory/schema mismatch while building index")
        if index_offset + segment_count * INDEX.size != path.stat().st_size:
            raise ValueError("memory stream has an invalid index")
        handle.seek(index_offset)
        raw_index = handle.read(segment_count * INDEX.size)
    if segment_count != len(events):
        raise ValueError("event count does not match memory Segment count")
    for expected, event in enumerate(events):
        segment_id, _, _, start_step, num_steps = INDEX.unpack_from(
            raw_index, expected * INDEX.size
        )
        if (
            segment_id != event.segment_id
            or start_step != event.start_step
            or start_step + num_steps - 1 != event.end_step
        ):
            raise ValueError("event time range does not match memory Segment layout")
