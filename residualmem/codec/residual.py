from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import CanonicalState, FieldSpec, Predictor, StateSchema
from .bits import BitReader, BitWriter
from .state import field_width
from .varint import (
    decode_uvarint, encode_uvarint, read_uvarint, write_uvarint,
    zigzag_decode, zigzag_encode,
)


@dataclass
class LiteralDictionary:
    values: list[str]

    @classmethod
    def from_anchor(cls, anchor: CanonicalState, schema: StateSchema):
        values: list[str] = []
        for spec, value in zip(schema.fields, anchor.values):
            if spec.field_type == "literal" and value is not None and value not in values:
                values.append(value)
        return cls(values)

    def index(self, value: str) -> int | None:
        try:
            return self.values.index(value)
        except ValueError:
            return None

    def add(self, value: str) -> None:
        if value not in self.values:
            self.values.append(value)


def field_distortion(spec: FieldSpec, actual, default) -> float:
    if actual == default:
        return 0.0
    if spec.field_type == "integer":
        return abs(int(actual) - int(default)) / spec.scale
    return 1.0


def encode_residuals(
    targets: tuple[CanonicalState, ...],
    actions: tuple[int, ...],
    predictor: Predictor,
    schema: StateSchema,
    lambda_: float,
) -> tuple[bytes, tuple[CanonicalState, ...]]:
    if len(actions) != len(targets) - 1:
        raise ValueError("action/state length mismatch")
    predictor.reset()
    reconstructed = targets[0]
    outputs = [reconstructed]
    dictionary = LiteralDictionary.from_anchor(reconstructed, schema)
    records: list[bytes] = []
    last_record_step = 0
    mask_bytes = (len(schema.fields) + 7) // 8
    for step in range(1, len(targets)):
        default = predictor.predict_next(reconstructed, actions[step - 1]).default_state
        schema.validate(default)
        target = targets[step]
        mandatory: list[int] = []
        profitable: list[tuple[int, float]] = []
        for index, (spec, actual, predicted) in enumerate(
            zip(schema.fields, target.values, default.values)
        ):
            if actual == predicted:
                continue
            payload_bits = _payload_bit_length(spec, actual, predicted, dictionary)
            if spec.policy == "must":
                mandatory.append(index)
            else:
                gain = spec.weight * field_distortion(spec, actual, predicted)
                gain -= lambda_ * payload_bits
                if gain > 0:
                    profitable.append((index, gain))
        selected = sorted(set(mandatory) | {index for index, _ in profitable})
        if selected:
            payload, payload_bits, pending_literals = _encode_selected(
                selected, target, default, schema, dictionary)
            shared = (
                len(encode_uvarint(step - last_record_step)) * 8
                + mask_bytes * 8
                + len(encode_uvarint(payload_bits)) * 8
                + (len(payload) * 8 - payload_bits)
            )
            optional_gain = sum(gain for _, gain in profitable)
            if mandatory or optional_gain > lambda_ * shared:
                mask = bytearray(mask_bytes)
                for index in selected:
                    mask[index // 8] |= 1 << (index % 8)
                record = bytearray(encode_uvarint(step - last_record_step))
                record += mask
                record += encode_uvarint(payload_bits)
                record += payload
                records.append(bytes(record))
                reconstructed = _apply_values(default, selected, target, schema)
                for value in pending_literals:
                    dictionary.add(value)
                last_record_step = step
            else:
                reconstructed = default
        else:
            reconstructed = default
        outputs.append(reconstructed)
    payload = bytearray(encode_uvarint(len(records)))
    for record in records:
        payload += record
    return bytes(payload), tuple(outputs)


def decode_residuals(
    payload: bytes,
    anchor: CanonicalState,
    actions: tuple[int, ...],
    num_steps: int,
    predictor: Predictor,
    schema: StateSchema,
) -> tuple[CanonicalState, ...]:
    record_count, offset = decode_uvarint(payload)
    mask_bytes = (len(schema.fields) + 7) // 8
    records: dict[int, tuple[list[int], bytes, int]] = {}
    absolute_step = 0
    for _ in range(record_count):
        delta, offset = decode_uvarint(payload, offset)
        absolute_step += delta
        mask = payload[offset: offset + mask_bytes]
        if len(mask) != mask_bytes:
            raise EOFError("truncated residual mask")
        offset += mask_bytes
        bit_length, offset = decode_uvarint(payload, offset)
        byte_length = (bit_length + 7) // 8
        data = payload[offset: offset + byte_length]
        if len(data) != byte_length:
            raise EOFError("truncated residual payload")
        offset += byte_length
        selected = [
            index for index in range(len(schema.fields))
            if mask[index // 8] & (1 << (index % 8))
        ]
        if not 0 < absolute_step < num_steps or absolute_step in records:
            raise ValueError("invalid or duplicate residual step")
        records[absolute_step] = (selected, data, bit_length)
    if offset != len(payload):
        raise ValueError("trailing residual bytes")
    if len(actions) != num_steps - 1:
        raise ValueError("action/state length mismatch")
    predictor.reset()
    reconstructed = anchor
    outputs = [anchor]
    dictionary = LiteralDictionary.from_anchor(anchor, schema)
    for step in range(1, num_steps):
        default = predictor.predict_next(reconstructed, actions[step - 1]).default_state
        if step in records:
            selected, data, bit_length = records[step]
            reconstructed = _decode_selected(
                selected, data, bit_length, default, schema, dictionary)
        else:
            reconstructed = default
        outputs.append(reconstructed)
    return tuple(outputs)


def _payload_bit_length(spec: FieldSpec, actual, default, dictionary) -> int:
    if spec.field_type == "bool":
        return 0
    if spec.field_type in {"categorical", "vq"}:
        return field_width(int(spec.num_values) - 1)
    if spec.field_type == "integer":
        return len(encode_uvarint(zigzag_encode(int(actual) - int(default)))) * 8
    if spec.field_type == "literal":
        index = dictionary.index(actual)
        if index is not None:
            return 1 + len(encode_uvarint(index)) * 8
        raw = actual.encode("utf-8")
        return 1 + len(encode_uvarint(len(raw))) * 8 + len(raw) * 8
    raise ValueError(spec.field_type)


def _encode_selected(selected, target, default, schema, dictionary):
    writer = BitWriter()
    pending: list[str] = []
    for index in selected:
        spec = schema.fields[index]
        actual, predicted = target[index], default[index]
        if spec.field_type == "bool":
            continue
        if spec.field_type in {"categorical", "vq"}:
            residual = int(actual) - (1 if int(actual) > int(predicted) else 0)
            writer.write_bits(residual, field_width(int(spec.num_values) - 1))
        elif spec.field_type == "integer":
            write_uvarint(writer, zigzag_encode(int(actual) - int(predicted)))
        elif spec.field_type == "literal":
            known = dictionary.index(actual)
            writer.write_bits(int(known is not None), 1)
            if known is not None:
                write_uvarint(writer, known)
            else:
                raw = actual.encode("utf-8")
                write_uvarint(writer, len(raw))
                writer.write_bytes(raw)
                pending.append(actual)
        else:
            raise ValueError(spec.field_type)
    return writer.finish(), writer.bit_length, pending


def _decode_selected(selected, data, bit_length, default, schema, dictionary):
    reader = BitReader(data, bit_length)
    values = list(default.values)
    for index in selected:
        spec = schema.fields[index]
        predicted = default[index]
        if spec.field_type == "bool":
            values[index] = not bool(predicted)
        elif spec.field_type in {"categorical", "vq"}:
            residual = reader.read_bits(field_width(int(spec.num_values) - 1))
            actual = residual + (1 if residual >= int(predicted) else 0)
            if actual >= int(spec.num_values):
                raise ValueError(f"invalid residual value for {spec.name}")
            values[index] = actual
        elif spec.field_type == "integer":
            values[index] = int(predicted) + zigzag_decode(read_uvarint(reader))
        elif spec.field_type == "literal":
            if reader.read_bits(1):
                known = read_uvarint(reader)
                try:
                    values[index] = dictionary.values[known]
                except IndexError as exc:
                    raise ValueError("literal dictionary reference is invalid") from exc
            else:
                value = reader.read_bytes(read_uvarint(reader)).decode("utf-8")
                values[index] = value
                dictionary.add(value)
        else:
            raise ValueError(spec.field_type)
    if reader.remaining:
        raise ValueError("residual payload was not fully consumed")
    return schema.make_state(values)


def _apply_values(default, selected, target, schema):
    values = list(default.values)
    for index in selected:
        values[index] = target[index]
    return schema.make_state(values)
