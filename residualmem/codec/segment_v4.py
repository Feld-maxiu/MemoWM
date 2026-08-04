"""RSMEMV04 latent memory container (report Section 10, anchors + segments).

Mirrors the v0.3 ``segment.py`` framing (global header, per-segment header with
CRC32 and final-state hash, index) but stores a *latent* stream: each segment's
anchor is the intra-coded latent ``z_tilde_0`` and the residual payload is the
droppable, entropy-coded correction stream from ``latent_residual``. Reuses the
v0.3 struct layouts, action packing, and :class:`BitAccounting`.

Header hashes bind the stream to the exact latent schema, the RSSM parameters, and
the front-end encoder (frozen-target discipline). The v0.2/v0.3 magics are rejected.
"""
from __future__ import annotations

import dataclasses
import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np

from ..latent.format import FORMAT_VERSION, MAGIC
from ..types import StateSchema, Trajectory
from ..world_model import rssm as R
from .segment import (
    FLAG_ACTIONS_EXTERNAL,
    GLOBAL,
    INDEX,
    SEGMENT,
    BitAccounting,
    DecodeError,
    LegacyFormatError,
    _add_accounting,
    decode_actions,
    encode_actions,
)
from .latent_residual import LatentStepper, decode_latent_segment, encode_latent_segment
from .send_mask import ExactMaskPolicy, MaskPolicy
from .state import decode_full_state, encode_full_state

LEGACY_MAGICS = (b"RSMEMV02", b"RSMEMV03")


def rssm_stream_hash(params, config: R.RSSMConfig, latent_schema: StateSchema,
                     domain_name: str) -> bytes:
    return R._content_hash(params, config, latent_schema.hash_bytes, domain_name)


def encode_latent_memory(
    path,
    trajectory: Trajectory,
    params,
    config: R.RSSMConfig,
    encoder,
    domain,
    latent_schema: StateSchema,
    *,
    mask_policy: MaskPolicy | None = None,
    segment_length: int = 64,
    actions_external: bool = False,
    action_values: int = 17,
    scale_bits: int = 16,
) -> tuple[BitAccounting, Trajectory]:
    if not 1 <= segment_length <= 65534:
        raise ValueError("segment_length must fit uint16 with its anchor")
    if not 1 <= scale_bits <= 30:
        raise ValueError("scale_bits out of range")
    mask_policy = mask_policy or ExactMaskPolicy()
    for state in trajectory.states:
        encoder.schema.validate(state) if hasattr(encoder, "schema") else None

    x_all = np.stack([encoder.encode(state).flat() for state in trajectory.states]).astype(np.float32)
    actions = np.asarray(trajectory.actions, np.int32)
    stepper = LatentStepper(params, config, domain.index)
    model_hash = rssm_stream_hash(params, config, latent_schema, domain.name)
    encoder_hash = encoder.hash_bytes
    if len(model_hash) != 32 or len(encoder_hash) != 32:
        raise ValueError("hashes must be 32 bytes")

    segments: list[bytes] = []
    index_rows: list[tuple[int, int, int, int, int]] = []
    latent_states: list = []
    accounting = BitAccounting(global_header=GLOBAL.size)
    offset = GLOBAL.size
    ranges = list(range(0, len(actions), segment_length)) or [0]
    for segment_id, start in enumerate(ranges):
        stop = min(start + segment_length, len(actions))
        x_window = x_all[start:stop + 1]
        seg_actions = actions[start:stop]
        num_steps = x_window.shape[0]
        anchor_codes, residual, recon = encode_latent_segment(
            stepper, x_window, seg_actions, mask_policy, scale_bits)
        anchor_state = latent_schema.make_state([int(v) for v in anchor_codes])
        final_state = latent_schema.make_state([int(v) for v in recon[-1]])
        anchor_bytes = encode_full_state(anchor_state, latent_schema)
        action_payload = b"" if actions_external else encode_actions(seg_actions, action_values)
        final_hash = hashlib.sha256(encode_full_state(final_state, latent_schema)).digest()
        flags = FLAG_ACTIONS_EXTERNAL if actions_external else 0
        header_wo_crc = SEGMENT.pack(
            segment_id, start, num_steps, len(anchor_bytes), len(action_payload),
            len(residual), flags, final_hash, 0)
        payload = anchor_bytes + action_payload + residual
        crc = zlib.crc32(header_wo_crc[:-4] + payload) & 0xFFFFFFFF
        header = SEGMENT.pack(
            segment_id, start, num_steps, len(anchor_bytes), len(action_payload),
            len(residual), flags, final_hash, crc)
        packed = header + payload
        segments.append(packed)
        index_rows.append((segment_id, offset, len(packed), start, num_steps))
        offset += len(packed)
        accounting = _add_accounting(
            accounting, segment_headers=SEGMENT.size - 4, checksums=4,
            anchors=len(anchor_bytes), actions=len(action_payload), residuals=len(residual))
        seg_states = [latent_schema.make_state([int(v) for v in codes]) for codes in recon]
        if latent_states:
            latent_states[-1] = seg_states[0]
            latent_states.extend(seg_states[1:])
        else:
            latent_states.extend(seg_states)

    index_offset = offset
    index_payload = b"".join(INDEX.pack(*row) for row in index_rows)
    accounting = _add_accounting(accounting, index=len(index_payload))
    global_header = GLOBAL.pack(
        MAGIC, FORMAT_VERSION, latent_schema.hash_bytes, model_hash, encoder_hash,
        segment_length, scale_bits, len(segments), index_offset)
    data = global_header + b"".join(segments) + index_payload
    if len(data) != accounting.total_bytes:
        raise AssertionError((len(data), accounting.total_bytes))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    latent_traj = Trajectory(
        tuple(latent_states), tuple(int(a) for a in actions),
        Path(path).stem, {"domain": domain.name, "mask_policy": mask_policy.policy_id})
    return accounting, latent_traj


def decode_latent_memory(
    path,
    params,
    config: R.RSSMConfig,
    domain,
    latent_schema: StateSchema,
    *,
    external_actions: tuple[int, ...] | None = None,
    action_values: int = 17,
) -> tuple[Trajectory, BitAccounting]:
    data = Path(path).read_bytes()
    if len(data) < GLOBAL.size:
        raise DecodeError("truncated global header")
    (magic, version, schema_hash, model_hash, encoder_hash, segment_length,
     scale_bits, segment_count, index_offset) = GLOBAL.unpack_from(data)
    if magic in LEGACY_MAGICS:
        raise LegacyFormatError("v0.2/v0.3 streams are incompatible with the latent decoder")
    if magic != MAGIC or version != FORMAT_VERSION:
        raise DecodeError("unsupported ResidualMem latent format")
    if schema_hash != latent_schema.hash_bytes:
        raise DecodeError("latent schema hash mismatch")
    if model_hash != rssm_stream_hash(params, config, latent_schema, domain.name):
        raise DecodeError("world model hash mismatch")
    if index_offset + segment_count * INDEX.size != len(data):
        raise DecodeError("invalid index offset or file length")

    stepper = LatentStepper(params, config, domain.index)
    rows = [INDEX.unpack_from(data, index_offset + i * INDEX.size) for i in range(segment_count)]
    all_states: list = []
    all_actions: list[int] = []
    accounting = BitAccounting(global_header=GLOBAL.size, index=segment_count * INDEX.size)
    for expected_id, row in enumerate(rows):
        segment_id, seg_offset, total_bytes, start_step, indexed_steps = row
        if segment_id != expected_id or seg_offset + total_bytes > index_offset:
            raise DecodeError("invalid segment index")
        fields = SEGMENT.unpack_from(data, seg_offset)
        (hid, hstart, num_steps, anchor_bytes, action_bytes, residual_bytes,
         flags, final_hash, stored_crc) = fields
        if (hid, hstart, num_steps) != (segment_id, start_step, indexed_steps):
            raise DecodeError("segment header/index mismatch")
        if SEGMENT.size + anchor_bytes + action_bytes + residual_bytes != total_bytes:
            raise DecodeError("segment length mismatch")
        payload = data[seg_offset + SEGMENT.size: seg_offset + total_bytes]
        header_zero_crc = data[seg_offset: seg_offset + SEGMENT.size - 4] + bytes(4)
        if zlib.crc32(header_zero_crc[:-4] + payload) & 0xFFFFFFFF != stored_crc:
            raise DecodeError("segment CRC mismatch")
        anchor = decode_full_state(payload[:anchor_bytes], latent_schema)
        action_count = num_steps - 1
        if flags & FLAG_ACTIONS_EXTERNAL:
            if external_actions is None:
                raise DecodeError("external actions are required")
            seg_actions = external_actions[start_step: start_step + action_count]
            if len(seg_actions) != action_count:
                raise DecodeError("external action stream is too short")
        else:
            seg_actions = decode_actions(
                payload[anchor_bytes: anchor_bytes + action_bytes], action_count, action_values)
        residual_payload = payload[anchor_bytes + action_bytes:]
        anchor_codes = np.asarray([int(v) for v in anchor.values], np.int64)
        recon = decode_latent_segment(
            stepper, anchor_codes, np.asarray(seg_actions, np.int32), num_steps,
            residual_payload, scale_bits)
        seg_states = [latent_schema.make_state([int(v) for v in codes]) for codes in recon]
        if hashlib.sha256(encode_full_state(seg_states[-1], latent_schema)).digest() != final_hash:
            raise DecodeError("final state hash mismatch")
        if all_states:
            all_states[-1] = seg_states[0]
            all_states.extend(seg_states[1:])
        else:
            all_states.extend(seg_states)
        all_actions.extend(seg_actions)
        accounting = _add_accounting(
            accounting, segment_headers=SEGMENT.size - 4, checksums=4,
            anchors=anchor_bytes, actions=action_bytes, residuals=residual_bytes)
    if accounting.total_bytes != len(data):
        raise DecodeError("bit accounting mismatch")
    return Trajectory(tuple(all_states), tuple(all_actions), Path(path).stem,
                      {"segment_length": segment_length, "domain": domain.name}), accounting
