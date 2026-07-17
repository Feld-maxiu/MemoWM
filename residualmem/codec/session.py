from __future__ import annotations

import dataclasses
import hashlib
import zlib
from pathlib import Path
from typing import Any

from ..types import CanonicalState, Predictor, StateSchema
from .residual import (
    LiteralDictionary,
    ResidualCursor,
    decode_residual_record,
)
from .segment import (
    FLAG_ACTIONS_EXTERNAL,
    FORMAT_VERSION,
    GLOBAL,
    INDEX,
    LEGACY_MAGIC,
    MAGIC,
    SEGMENT,
    DecodeError,
    LegacyFormatError,
    decode_actions,
)
from .state import decode_full_state, encode_full_state


@dataclasses.dataclass(frozen=True)
class SegmentIndexRow:
    segment_id: int
    offset: int
    total_bytes: int
    start_step: int
    num_steps: int

    @property
    def end_step(self) -> int:
        return self.start_step + self.num_steps - 1


@dataclasses.dataclass(frozen=True)
class SegmentData:
    row: SegmentIndexRow
    anchor: CanonicalState
    actions: tuple[int, ...]
    residual_payload: bytes
    final_hash: bytes


@dataclasses.dataclass(frozen=True)
class EvidenceFrame:
    step: int
    action: int | None
    values: dict[str, Any]
    residual_applied: bool


@dataclasses.dataclass(frozen=True)
class EvidenceChunk:
    segment_id: int
    start_step: int
    end_step: int
    reason: str
    frames: tuple[EvidenceFrame, ...]


class MemoryFile:
    """Validated random-access view of a ResidualMem v0.3 stream."""

    def __init__(
        self,
        path: str | Path,
        predictor: Predictor,
        schema: StateSchema,
        adapter_hash: bytes | None = None,
        external_actions: tuple[int, ...] | None = None,
        action_values: int = 17,
    ):
        self.path = Path(path)
        self.predictor = predictor
        self.schema = schema
        self.external_actions = external_actions
        self.action_values = action_values
        with self.path.open("rb") as handle:
            header = handle.read(GLOBAL.size)
            if len(header) != GLOBAL.size:
                raise DecodeError("truncated global header")
            (
                magic,
                version,
                schema_hash,
                model_hash,
                stored_adapter_hash,
                self.segment_length,
                _,
                segment_count,
                self.index_offset,
            ) = GLOBAL.unpack(header)
            if magic == LEGACY_MAGIC or version == 2:
                raise LegacyFormatError(
                    "ResidualMem v0.2 streams are incompatible with the v0.3 "
                    "exact-only schema; re-encode from the canonical trajectory"
                )
            if magic != MAGIC or version != FORMAT_VERSION:
                raise DecodeError("unsupported ResidualMem format")
            if schema_hash != schema.hash_bytes:
                raise DecodeError("schema hash mismatch")
            if model_hash != predictor.hash_bytes:
                raise DecodeError("world model hash mismatch")
            if adapter_hash is not None and stored_adapter_hash != adapter_hash:
                raise DecodeError("adapter hash mismatch")
            expected_size = self.index_offset + segment_count * INDEX.size
            if expected_size != self.path.stat().st_size:
                raise DecodeError("invalid index offset or file length")
            handle.seek(self.index_offset)
            raw_index = handle.read(segment_count * INDEX.size)

        self.rows = tuple(
            SegmentIndexRow(*INDEX.unpack_from(raw_index, index * INDEX.size))
            for index in range(segment_count)
        )
        for expected_id, row in enumerate(self.rows):
            if row.segment_id != expected_id:
                raise DecodeError("segment ids must be contiguous")
            if row.offset < GLOBAL.size or row.offset + row.total_bytes > self.index_offset:
                raise DecodeError("invalid segment index")

    @property
    def file_bytes(self) -> int:
        return self.path.stat().st_size

    @property
    def segment_count(self) -> int:
        return len(self.rows)

    def load_segment(self, segment_id: int) -> SegmentData:
        try:
            row = self.rows[segment_id]
        except IndexError as exc:
            raise KeyError(f"unknown segment {segment_id}") from exc
        if segment_id < 0 or row.segment_id != segment_id:
            raise KeyError(f"unknown segment {segment_id}")
        with self.path.open("rb") as handle:
            handle.seek(row.offset)
            packed = handle.read(row.total_bytes)
        if len(packed) != row.total_bytes or len(packed) < SEGMENT.size:
            raise DecodeError("truncated segment")
        (
            header_id,
            header_start,
            num_steps,
            anchor_bytes,
            action_bytes,
            residual_bytes,
            flags,
            final_hash,
            stored_crc,
        ) = SEGMENT.unpack_from(packed)
        if (header_id, header_start, num_steps) != (
            row.segment_id,
            row.start_step,
            row.num_steps,
        ):
            raise DecodeError("segment header/index mismatch")
        expected_total = SEGMENT.size + anchor_bytes + action_bytes + residual_bytes
        if expected_total != row.total_bytes:
            raise DecodeError("segment length mismatch")
        payload = packed[SEGMENT.size:]
        actual_crc = zlib.crc32(packed[: SEGMENT.size - 4] + payload) & 0xFFFFFFFF
        if actual_crc != stored_crc:
            raise DecodeError("segment CRC mismatch")
        anchor_end = anchor_bytes
        action_end = anchor_end + action_bytes
        anchor = decode_full_state(payload[:anchor_end], self.schema)
        action_count = num_steps - 1
        if flags & FLAG_ACTIONS_EXTERNAL:
            if self.external_actions is None:
                raise DecodeError("external actions are required")
            actions = self.external_actions[
                row.start_step: row.start_step + action_count
            ]
            if len(actions) != action_count:
                raise DecodeError("external action stream is too short")
        else:
            actions = decode_actions(
                payload[anchor_end:action_end], action_count, self.action_values
            )
        return SegmentData(
            row=row,
            anchor=anchor,
            actions=tuple(actions),
            residual_payload=payload[action_end:],
            final_hash=final_hash,
        )

    def open_session(self, segment_id: int) -> "SegmentDecodeSession":
        return SegmentDecodeSession(
            self.load_segment(segment_id),
            self.predictor.new_session(),
            self.schema,
        )


class SegmentDecodeSession:
    """Forward-only WM reconstruction with exact residual corrections."""

    def __init__(
        self,
        segment: SegmentData,
        predictor: Predictor,
        schema: StateSchema,
    ):
        self.segment = segment
        self.predictor = predictor
        self.schema = schema
        self.predictor.reset()
        self.cursor = ResidualCursor(
            segment.residual_payload, schema, segment.row.num_steps
        )
        self.dictionary = LiteralDictionary.from_anchor(segment.anchor, schema)
        self.current_local_step = 0
        self.current_state = segment.anchor
        self.cache: dict[int, tuple[CanonicalState, int | None, bool]] = {
            0: (segment.anchor, None, False)
        }
        self.wm_steps = 0
        self.residual_records = 0

    @property
    def segment_id(self) -> int:
        return self.segment.row.segment_id

    @property
    def current_step(self) -> int:
        return self.segment.row.start_step + self.current_local_step

    @property
    def end_step(self) -> int:
        return self.segment.row.end_step

    def view(self, fields: tuple[str, ...]) -> EvidenceChunk:
        selected = self._field_indices(fields)
        frame = self._frame(self.current_local_step, selected)
        return EvidenceChunk(
            self.segment_id, frame.step, frame.step, "view", (frame,)
        )

    def expand(
        self,
        fields: tuple[str, ...],
        max_steps: int = 8,
    ) -> EvidenceChunk:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        selected = self._field_indices(fields)
        start = self.current_local_step
        if start >= self.segment.row.num_steps - 1:
            return EvidenceChunk(
                self.segment_id,
                self.current_step,
                self.current_step,
                "segment_end",
                (self._frame(start, selected),),
            )
        cap = min(start + max_steps, self.segment.row.num_steps - 1)
        next_record = self.cursor.peek()
        target = cap
        reason = "step_cap"
        if next_record is not None and start < next_record.step <= cap:
            target = next_record.step
            reason = "next_residual"
        elif cap == self.segment.row.num_steps - 1:
            reason = "segment_end"

        frames = [self._frame(start, selected)]
        for local_step in range(start + 1, target + 1):
            action = self.segment.actions[local_step - 1]
            default = self.predictor.predict_next(
                self.current_state, action
            ).default_state
            self.schema.validate(default)
            record = self.cursor.peek()
            residual_applied = False
            if record is not None and record.step < local_step:
                raise DecodeError("residual cursor fell behind reconstruction")
            if record is not None and record.step == local_step:
                self.cursor.pop()
                self.current_state = decode_residual_record(
                    record, default, self.schema, self.dictionary
                )
                residual_applied = True
                self.residual_records += 1
            else:
                self.current_state = default
            self.current_local_step = local_step
            self.wm_steps += 1
            self.cache[local_step] = (
                self.current_state,
                action,
                residual_applied,
            )
            frames.append(self._frame(local_step, selected))

        if self.current_local_step == self.segment.row.num_steps - 1:
            digest = hashlib.sha256(
                encode_full_state(self.current_state, self.schema)
            ).digest()
            if digest != self.segment.final_hash:
                raise DecodeError("final state hash mismatch")
        return EvidenceChunk(
            self.segment_id,
            self.segment.row.start_step + start,
            self.current_step,
            reason,
            tuple(frames),
        )

    def reveal(
        self,
        start_step: int,
        end_step: int,
        fields: tuple[str, ...],
    ) -> EvidenceChunk:
        if start_step > end_step:
            raise ValueError("start_step must not exceed end_step")
        selected = self._field_indices(fields)
        local_start = start_step - self.segment.row.start_step
        local_end = end_step - self.segment.row.start_step
        if local_start < 0 or local_end > self.current_local_step:
            raise ValueError("reveal can only read already reconstructed states")
        frames = tuple(
            self._frame(step, selected)
            for step in range(local_start, local_end + 1)
        )
        return EvidenceChunk(
            self.segment_id, start_step, end_step, "reveal", frames
        )

    def _field_indices(self, fields: tuple[str, ...]) -> tuple[int, ...]:
        if not fields:
            raise ValueError("Reader must request at least one field")
        positions = {name: index for index, name in enumerate(self.schema.names)}
        unknown = sorted(set(fields) - set(positions))
        if unknown:
            raise ValueError(f"unknown fields: {unknown}")
        return tuple(positions[name] for name in dict.fromkeys(fields))

    def _frame(
        self, local_step: int, selected: tuple[int, ...]
    ) -> EvidenceFrame:
        state, action, residual_applied = self.cache[local_step]
        return EvidenceFrame(
            step=self.segment.row.start_step + local_step,
            action=action,
            values={self.schema.fields[index].name: state[index] for index in selected},
            residual_applied=residual_applied,
        )
