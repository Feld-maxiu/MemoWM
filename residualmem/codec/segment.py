from __future__ import annotations

import dataclasses
import hashlib
import struct
import zlib
from pathlib import Path

from ..types import Predictor, StateSchema, Trajectory
from .bits import BitReader, BitWriter
from .residual import decode_residuals, encode_residuals
from .state import decode_full_state, encode_full_state, field_width


MAGIC = b"RSMEMV03"
LEGACY_MAGIC = b"RSMEMV02"
FORMAT_VERSION = 3
GLOBAL = struct.Struct("<8sH32s32s32sHIIQ")
SEGMENT = struct.Struct("<IQHIIIB32sI")
INDEX = struct.Struct("<IQIQH")
FLAG_ACTIONS_EXTERNAL = 1


class DecodeError(RuntimeError):
    pass


class LegacyFormatError(DecodeError):
    pass


@dataclasses.dataclass(frozen=True)
class BitAccounting:
    global_header: int = 0
    segment_headers: int = 0
    anchors: int = 0
    actions: int = 0
    residuals: int = 0
    checksums: int = 0
    index: int = 0

    @property
    def total_bytes(self) -> int:
        return sum(dataclasses.asdict(self).values())

    @property
    def total_bits(self) -> int:
        return self.total_bytes * 8

    def as_dict(self) -> dict[str, int]:
        return {**dataclasses.asdict(self), "total_bytes": self.total_bytes,
                "total_bits": self.total_bits}


def encode_actions(actions: tuple[int, ...], num_values: int = 17) -> bytes:
    writer = BitWriter()
    width = field_width(num_values)
    for action in actions:
        if not 0 <= action < num_values:
            raise ValueError(f"action {action} outside [0, {num_values})")
        writer.write_bits(action, width)
    return writer.finish()


def decode_actions(data: bytes, count: int, num_values: int = 17) -> tuple[int, ...]:
    reader = BitReader(data)
    width = field_width(num_values)
    output = tuple(reader.read_bits(width) for _ in range(count))
    if any(action >= num_values for action in output):
        raise DecodeError("invalid action id")
    return output


def encode_memory(
    path: str | Path,
    trajectory: Trajectory,
    predictor: Predictor,
    schema: StateSchema,
    segment_length: int = 64,
    adapter_hash: bytes | None = None,
    actions_external: bool = False,
    action_values: int = 17,
) -> BitAccounting:
    if not 1 <= segment_length <= 65534:
        raise ValueError("segment_length must fit uint16 with its anchor")
    for state in trajectory.states:
        schema.validate(state)
    adapter_hash = adapter_hash or bytes(32)
    if len(adapter_hash) != 32 or len(predictor.hash_bytes) != 32:
        raise ValueError("hashes must be 32 bytes")

    segments: list[bytes] = []
    index_rows: list[tuple[int, int, int, int, int]] = []
    accounting = BitAccounting(global_header=GLOBAL.size)
    offset = GLOBAL.size
    transition_ranges = list(
        range(0, len(trajectory.actions), segment_length)
    ) or [0]
    for segment_id, start in enumerate(transition_ranges):
        stop = min(start + segment_length, len(trajectory.actions))
        # Adjacent segments share only their boundary anchor. This preserves
        # every transition while keeping each segment independently decodable.
        targets = trajectory.states[start:stop + 1]
        actions = trajectory.actions[start:stop]
        anchor = encode_full_state(targets[0], schema)
        action_payload = b"" if actions_external else encode_actions(actions, action_values)
        residual, reconstructed = encode_residuals(
            targets, actions, predictor, schema)
        final_hash = hashlib.sha256(
            encode_full_state(reconstructed[-1], schema)).digest()
        flags = FLAG_ACTIONS_EXTERNAL if actions_external else 0
        header_without_crc = SEGMENT.pack(
            segment_id, start, len(targets), len(anchor), len(action_payload),
            len(residual), flags, final_hash, 0)
        payload = anchor + action_payload + residual
        crc = zlib.crc32(header_without_crc[:-4] + payload) & 0xFFFFFFFF
        header = SEGMENT.pack(
            segment_id, start, len(targets), len(anchor), len(action_payload),
            len(residual), flags, final_hash, crc)
        packed = header + payload
        segments.append(packed)
        index_rows.append((segment_id, offset, len(packed), start, len(targets)))
        offset += len(packed)
        accounting = _add_accounting(
            accounting,
            segment_headers=SEGMENT.size - 4,
            checksums=4,
            anchors=len(anchor),
            actions=len(action_payload),
            residuals=len(residual),
        )
    index_offset = offset
    index_payload = b"".join(INDEX.pack(*row) for row in index_rows)
    accounting = _add_accounting(accounting, index=len(index_payload))
    global_header = GLOBAL.pack(
        MAGIC, FORMAT_VERSION, schema.hash_bytes, predictor.hash_bytes,
        adapter_hash, segment_length, 0, len(segments), index_offset)
    data = global_header + b"".join(segments) + index_payload
    if len(data) != accounting.total_bytes:
        raise AssertionError((len(data), accounting))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return accounting


def decode_memory(
    path: str | Path,
    predictor: Predictor,
    schema: StateSchema,
    adapter_hash: bytes | None = None,
    external_actions: tuple[int, ...] | None = None,
    action_values: int = 17,
) -> tuple[Trajectory, BitAccounting]:
    data = Path(path).read_bytes()
    if len(data) < GLOBAL.size:
        raise DecodeError("truncated global header")
    (
        magic, version, schema_hash, model_hash, stored_adapter_hash,
        segment_length, _, segment_count, index_offset,
    ) = GLOBAL.unpack_from(data)
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
    if index_offset + segment_count * INDEX.size != len(data):
        raise DecodeError("invalid index offset or file length")

    rows = [
        INDEX.unpack_from(data, index_offset + index * INDEX.size)
        for index in range(segment_count)
    ]
    all_states = []
    all_actions: list[int] = []
    accounting = BitAccounting(global_header=GLOBAL.size, index=segment_count * INDEX.size)
    for expected_id, row in enumerate(rows):
        segment_id, offset, total_bytes, start_step, indexed_steps = row
        if segment_id != expected_id or offset + total_bytes > index_offset:
            raise DecodeError("invalid segment index")
        if total_bytes < SEGMENT.size:
            raise DecodeError("truncated segment")
        fields = SEGMENT.unpack_from(data, offset)
        (
            header_id, header_start, num_steps, anchor_bytes, action_bytes,
            residual_bytes, flags, final_hash, stored_crc,
        ) = fields
        if (header_id, header_start, num_steps) != (
            segment_id, start_step, indexed_steps
        ):
            raise DecodeError("segment header/index mismatch")
        expected_total = SEGMENT.size + anchor_bytes + action_bytes + residual_bytes
        if expected_total != total_bytes:
            raise DecodeError("segment length mismatch")
        payload_start = offset + SEGMENT.size
        payload = data[payload_start: offset + total_bytes]
        header_zero_crc = data[offset: offset + SEGMENT.size - 4] + bytes(4)
        actual_crc = zlib.crc32(header_zero_crc[:-4] + payload) & 0xFFFFFFFF
        if actual_crc != stored_crc:
            raise DecodeError("segment CRC mismatch")
        anchor_data = payload[:anchor_bytes]
        action_data = payload[anchor_bytes: anchor_bytes + action_bytes]
        residual_data = payload[anchor_bytes + action_bytes:]
        anchor = decode_full_state(anchor_data, schema)
        action_count = num_steps - 1
        if flags & FLAG_ACTIONS_EXTERNAL:
            if external_actions is None:
                raise DecodeError("external actions are required")
            actions = external_actions[start_step: start_step + action_count]
            if len(actions) != action_count:
                raise DecodeError("external action stream is too short")
        else:
            actions = decode_actions(action_data, action_count, action_values)
        states = decode_residuals(
            residual_data, anchor, tuple(actions), num_steps, predictor, schema)
        if hashlib.sha256(encode_full_state(states[-1], schema)).digest() != final_hash:
            raise DecodeError("final state hash mismatch")
        if all_states:
            # Adjacent segments share one exact boundary anchor.
            all_states[-1] = states[0]
            all_states.extend(states[1:])
        else:
            all_states.extend(states)
        all_actions.extend(actions)
        accounting = _add_accounting(
            accounting,
            segment_headers=SEGMENT.size - 4,
            checksums=4,
            anchors=anchor_bytes,
            actions=action_bytes,
            residuals=residual_bytes,
        )
    if accounting.total_bytes != len(data):
        raise DecodeError("bit accounting mismatch")
    return Trajectory(
        tuple(all_states), tuple(all_actions), Path(path).stem,
        {"segment_length": segment_length}), accounting


def _add_accounting(value: BitAccounting, **updates: int) -> BitAccounting:
    fields = dataclasses.asdict(value)
    for key, amount in updates.items():
        fields[key] += amount
    return BitAccounting(**fields)
