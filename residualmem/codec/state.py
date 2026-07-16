from __future__ import annotations

import math

from ..types import CanonicalState, FieldSpec, StateSchema
from .bits import BitReader, BitWriter
from .varint import read_uvarint, write_uvarint, zigzag_decode, zigzag_encode


def field_width(num_values: int) -> int:
    return max(1, math.ceil(math.log2(num_values)))


def encode_full_state(state: CanonicalState, schema: StateSchema) -> bytes:
    schema.validate(state)
    writer = BitWriter()
    for spec, value in zip(schema.fields, state.values):
        if spec.optional:
            writer.write_bits(int(value is not None), 1)
            if value is None:
                continue
        _write_full_field(writer, spec, value)
    return writer.finish()


def decode_full_state(data: bytes, schema: StateSchema) -> CanonicalState:
    reader = BitReader(data)
    values = []
    for spec in schema.fields:
        if spec.optional and not reader.read_bits(1):
            values.append(None)
            continue
        values.append(_read_full_field(reader, spec))
    return schema.make_state(values)


def _write_full_field(writer: BitWriter, spec: FieldSpec, value) -> None:
    if spec.field_type == "bool":
        writer.write_bits(int(bool(value)), 1)
    elif spec.field_type in {"categorical", "vq"}:
        writer.write_bits(int(value), field_width(int(spec.num_values)))
    elif spec.field_type == "integer":
        write_uvarint(writer, zigzag_encode(int(value)))
    elif spec.field_type == "literal":
        payload = value.encode("utf-8")
        write_uvarint(writer, len(payload))
        writer.write_bytes(payload)
    else:
        raise ValueError(spec.field_type)


def _read_full_field(reader: BitReader, spec: FieldSpec):
    if spec.field_type == "bool":
        return bool(reader.read_bits(1))
    if spec.field_type in {"categorical", "vq"}:
        value = reader.read_bits(field_width(int(spec.num_values)))
        if value >= int(spec.num_values):
            raise ValueError(f"invalid {spec.name} value {value}")
        return value
    if spec.field_type == "integer":
        return zigzag_decode(read_uvarint(reader))
    if spec.field_type == "literal":
        return reader.read_bytes(read_uvarint(reader)).decode("utf-8")
    raise ValueError(spec.field_type)
