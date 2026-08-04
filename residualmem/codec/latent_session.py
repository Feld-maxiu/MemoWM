"""Random-access + progressive decode for the RSMEMV04 latent stream (report Section 10).

Mirrors ``codec/session.py`` (``MemoryFile`` / ``SegmentDecodeSession``) so the
existing :class:`QueryEngine` drives it unchanged -- it only needs ``open_session``
plus ``view``/``expand``/``reveal``. The difference is internal: each step runs the
RSSM forward, fills dropped groups with the prior mode, decodes stored groups from
the entropy stream, then renders the reconstructed latent to *approximate* canonical
fields via ``D_omega`` + the front-end inverse. Evidence values are lossy but the
grounding contract is unchanged (an answer may only cite fields actually shown).
"""
from __future__ import annotations

import dataclasses
import hashlib
import zlib
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from ..latent.format import FORMAT_VERSION, MAGIC
from ..types import CanonicalState, StateSchema
from ..world_model import rssm as R
from .entropy import RansDecoder, quantize_freqs
from .latent_residual import LatentStepper, _deserialize, _softmax_rows
from .segment import (
    FLAG_ACTIONS_EXTERNAL,
    GLOBAL,
    INDEX,
    SEGMENT,
    DecodeError,
    LegacyFormatError,
    decode_actions,
)
from .segment_v4 import LEGACY_MAGICS, rssm_stream_hash
from .session import EvidenceChunk, EvidenceFrame, SegmentIndexRow
from .state import decode_full_state, encode_full_state


@dataclasses.dataclass(frozen=True)
class LatentSegmentData:
    row: SegmentIndexRow
    anchor_codes: np.ndarray
    actions: tuple[int, ...]
    residual_payload: bytes
    final_hash: bytes


class LatentMemoryFile:
    """Validated random-access view of a latent stream. ``encoder`` is the frozen
    front-end supplying canonical field names, ``decode_tokens`` and its hash."""

    def __init__(self, path, params, config: R.RSSMConfig, domain, latent_schema: StateSchema,
                 encoder, *, external_actions=None, action_values: int = 17):
        self.path = Path(path)
        self.config = config
        self.domain = domain
        self.latent_schema = latent_schema
        self.encoder = encoder
        self.canonical_schema: StateSchema = encoder.schema
        self.external_actions = external_actions
        self.action_values = action_values
        self.stepper = LatentStepper(params, config, domain.index)

        data_size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            header = handle.read(GLOBAL.size)
            if len(header) != GLOBAL.size:
                raise DecodeError("truncated global header")
            (magic, version, schema_hash, model_hash, encoder_hash, self.segment_length,
             self.scale_bits, segment_count, self.index_offset) = GLOBAL.unpack(header)
            if magic in LEGACY_MAGICS:
                raise LegacyFormatError("v0.2/v0.3 streams are incompatible with the latent decoder")
            if magic != MAGIC or version != FORMAT_VERSION:
                raise DecodeError("unsupported ResidualMem latent format")
            if schema_hash != latent_schema.hash_bytes:
                raise DecodeError("latent schema hash mismatch")
            if model_hash != rssm_stream_hash(params, config, latent_schema, domain.name):
                raise DecodeError("world model hash mismatch")
            if encoder_hash != encoder.hash_bytes:
                raise DecodeError("front-end encoder hash mismatch")
            if self.index_offset + segment_count * INDEX.size != data_size:
                raise DecodeError("invalid index offset or file length")
            handle.seek(self.index_offset)
            raw_index = handle.read(segment_count * INDEX.size)
        self.rows = tuple(
            SegmentIndexRow(*INDEX.unpack_from(raw_index, i * INDEX.size))
            for i in range(segment_count))
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

    def load_segment(self, segment_id: int) -> LatentSegmentData:
        try:
            row = self.rows[segment_id]
        except IndexError as exc:
            raise KeyError(f"unknown segment {segment_id}") from exc
        with self.path.open("rb") as handle:
            handle.seek(row.offset)
            packed = handle.read(row.total_bytes)
        if len(packed) != row.total_bytes or len(packed) < SEGMENT.size:
            raise DecodeError("truncated segment")
        (hid, hstart, num_steps, anchor_bytes, action_bytes, residual_bytes,
         flags, final_hash, stored_crc) = SEGMENT.unpack_from(packed)
        if (hid, hstart, num_steps) != (row.segment_id, row.start_step, row.num_steps):
            raise DecodeError("segment header/index mismatch")
        if SEGMENT.size + anchor_bytes + action_bytes + residual_bytes != row.total_bytes:
            raise DecodeError("segment length mismatch")
        payload = packed[SEGMENT.size:]
        if zlib.crc32(packed[:SEGMENT.size - 4] + payload) & 0xFFFFFFFF != stored_crc:
            raise DecodeError("segment CRC mismatch")
        anchor = decode_full_state(payload[:anchor_bytes], self.latent_schema)
        action_count = num_steps - 1
        if flags & FLAG_ACTIONS_EXTERNAL:
            if self.external_actions is None:
                raise DecodeError("external actions are required")
            actions = self.external_actions[row.start_step: row.start_step + action_count]
            if len(actions) != action_count:
                raise DecodeError("external action stream is too short")
        else:
            actions = decode_actions(
                payload[anchor_bytes: anchor_bytes + action_bytes], action_count, self.action_values)
        anchor_codes = np.asarray([int(v) for v in anchor.values], np.int64)
        return LatentSegmentData(row, anchor_codes, tuple(actions),
                                 payload[anchor_bytes + action_bytes:], final_hash)

    def open_session(self, segment_id: int) -> "LatentDecodeSession":
        return LatentDecodeSession(
            self.load_segment(segment_id), self.stepper, self.latent_schema,
            self.encoder, self.scale_bits)

    def render_trajectory(self):
        """Decode + render every segment to an approximate canonical Trajectory
        (used to build the retrieval index). Reuses the progressive session."""
        from ..types import Trajectory
        states: list[CanonicalState] = []
        actions: list[int] = []
        all_fields = self.canonical_schema.names
        for segment_id in range(self.segment_count):
            session = self.open_session(segment_id)
            session.view(all_fields)
            while session.current_local_step < session.num_steps - 1:
                session.expand(all_fields, session.num_steps)
            seg_states = [session.cache[i][0] for i in range(session.num_steps)]
            if states:
                states[-1] = seg_states[0]
                states.extend(seg_states[1:])
            else:
                states.extend(seg_states)
            actions.extend(session.segment.actions)
        return Trajectory(tuple(states), tuple(actions), self.path.stem)


class LatentDecodeSession:
    """Forward-only latent reconstruction with prior-fill + entropy-decoded corrections."""

    def __init__(self, segment: LatentSegmentData, stepper: LatentStepper,
                 latent_schema: StateSchema, encoder, scale_bits: int):
        self.segment = segment
        self.stepper = stepper
        self.latent_schema = latent_schema
        self.encoder = encoder
        self.schema = encoder.schema
        self.num_steps = segment.row.num_steps
        record_map, entropy = _deserialize(
            segment.residual_payload, stepper.config.num_groups, self.num_steps)
        self.record_map = record_map
        self.record_steps = sorted(record_map)
        self.rans = RansDecoder(entropy, scale_bits) if entropy else None
        self.scale_bits = scale_bits

        self.h = stepper.h0
        self.prev_codes = segment.anchor_codes
        self.current_local_step = 0
        anchor_state = self._render(stepper.h0, segment.anchor_codes)
        self.cache: dict[int, tuple[CanonicalState, int | None, bool]] = {
            0: (anchor_state, None, False)}
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
        return EvidenceChunk(self.segment_id, frame.step, frame.step, "view", (frame,))

    def expand(self, fields: tuple[str, ...], max_steps: int = 8) -> EvidenceChunk:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        selected = self._field_indices(fields)
        start = self.current_local_step
        if start >= self.num_steps - 1:
            return EvidenceChunk(self.segment_id, self.current_step, self.current_step,
                                 "segment_end", (self._frame(start, selected),))
        cap = min(start + max_steps, self.num_steps - 1)
        next_rec = next((r for r in self.record_steps if r > start), None)
        target, reason = cap, "step_cap"
        if next_rec is not None and start < next_rec <= cap:
            target, reason = next_rec, "next_residual"
        elif cap == self.num_steps - 1:
            reason = "segment_end"

        frames = [self._frame(start, selected)]
        for local in range(start + 1, target + 1):
            action = self.segment.actions[local - 1]
            self.h, plog = self.stepper._dec(
                self.stepper.params, self.h,
                jnp.asarray(self.prev_codes.astype(np.int32)),
                jnp.asarray(int(action)))
            plog = np.asarray(plog, np.float64)
            ztil = plog.argmax(-1).astype(np.int64)
            residual_applied = False
            sent = self.record_map.get(local)
            if sent:
                probs = _softmax_rows(plog)
                for j in sent:
                    ztil[j] = self.rans.decode(quantize_freqs(probs[j], self.scale_bits))
                residual_applied = True
                self.residual_records += 1
            self.prev_codes = ztil
            state = self._render(self.h, ztil)
            self.current_local_step = local
            self.wm_steps += 1
            self.cache[local] = (state, action, residual_applied)
            frames.append(self._frame(local, selected))

        if self.current_local_step == self.num_steps - 1:
            final_state = self.latent_schema.make_state([int(v) for v in self.prev_codes])
            digest = hashlib.sha256(encode_full_state(final_state, self.latent_schema)).digest()
            if digest != self.segment.final_hash:
                raise DecodeError("final state hash mismatch")
        return EvidenceChunk(self.segment_id, self.segment.row.start_step + start,
                             self.current_step, reason, tuple(frames))

    def reveal(self, start_step: int, end_step: int, fields: tuple[str, ...]) -> EvidenceChunk:
        if start_step > end_step:
            raise ValueError("start_step must not exceed end_step")
        selected = self._field_indices(fields)
        local_start = start_step - self.segment.row.start_step
        local_end = end_step - self.segment.row.start_step
        if local_start < 0 or local_end > self.current_local_step:
            raise ValueError("reveal can only read already reconstructed states")
        frames = tuple(self._frame(step, selected) for step in range(local_start, local_end + 1))
        return EvidenceChunk(self.segment_id, start_step, end_step, "reveal", frames)

    def _render(self, h, codes) -> CanonicalState:
        x_hat = self.stepper.decode_x(h, codes)
        return self.encoder.decode_tokens(x_hat)

    def _field_indices(self, fields: tuple[str, ...]) -> tuple[int, ...]:
        if not fields:
            raise ValueError("Reader must request at least one field")
        positions = {name: index for index, name in enumerate(self.schema.names)}
        unknown = sorted(set(fields) - set(positions))
        if unknown:
            raise ValueError(f"unknown fields: {unknown}")
        return tuple(positions[name] for name in dict.fromkeys(fields))

    def _frame(self, local_step: int, selected: tuple[int, ...]) -> EvidenceFrame:
        state, action, residual_applied = self.cache[local_step]
        return EvidenceFrame(
            step=self.segment.row.start_step + local_step,
            action=action,
            values={self.schema.fields[index].name: state[index] for index in selected},
            residual_applied=residual_applied)
