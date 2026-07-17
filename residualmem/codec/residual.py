from __future__ import annotations

from dataclasses import dataclass

from ..types import CanonicalState, Predictor, StateSchema
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


@dataclass(frozen=True)
class ResidualRecord:
    step: int
    selected: tuple[int, ...]
    payload: bytes
    bit_length: int


class ResidualCursor:
    """Lazy, forward-only parser over residual records."""

    def __init__(self, payload: bytes, schema: StateSchema, num_steps: int):
        self.payload = payload
        self.schema = schema
        self.num_steps = num_steps
        self.remaining, self.offset = decode_uvarint(payload)
        self.absolute_step = 0
        self._pending: ResidualRecord | None = None

    def peek(self) -> ResidualRecord | None:
        if self._pending is None and self.remaining:
            self._pending = self._read()
        if self._pending is None and self.offset != len(self.payload):
            raise ValueError("trailing residual bytes")
        return self._pending

    def pop(self) -> ResidualRecord | None:
        record = self.peek()
        self._pending = None
        return record

    def _read(self) -> ResidualRecord:
        delta, self.offset = decode_uvarint(self.payload, self.offset)
        if delta <= 0:
            raise ValueError("residual step deltas must be positive")
        self.absolute_step += delta
        mask_bytes = (len(self.schema.fields) + 7) // 8
        mask = self.payload[self.offset: self.offset + mask_bytes]
        if len(mask) != mask_bytes:
            raise EOFError("truncated residual mask")
        self.offset += mask_bytes
        bit_length, self.offset = decode_uvarint(self.payload, self.offset)
        byte_length = (bit_length + 7) // 8
        data = self.payload[self.offset: self.offset + byte_length]
        if len(data) != byte_length:
            raise EOFError("truncated residual payload")
        self.offset += byte_length
        if not 0 < self.absolute_step < self.num_steps:
            raise ValueError("invalid residual step")
        selected = tuple(
            index for index in range(len(self.schema.fields))
            if mask[index // 8] & (1 << (index % 8))
        )
        if not selected:
            raise ValueError("residual records must select at least one field")
        self.remaining -= 1
        return ResidualRecord(
            self.absolute_step, selected, data, bit_length
        )


def encode_residuals(
    targets: tuple[CanonicalState, ...],
    actions: tuple[int, ...],
    predictor: Predictor,
    schema: StateSchema,
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
        selected: list[int] = []
        for index, (spec, actual, predicted) in enumerate(
            zip(schema.fields, target.values, default.values)
        ):
            if actual == predicted:
                continue
            selected.append(index)
        if selected:
            payload, payload_bits, pending_literals = _encode_selected(
                selected, target, default, schema, dictionary)
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
    cursor = ResidualCursor(payload, schema, num_steps)
    if len(actions) != num_steps - 1:
        raise ValueError("action/state length mismatch")
    predictor.reset()
    reconstructed = anchor
    outputs = [anchor]
    dictionary = LiteralDictionary.from_anchor(anchor, schema)
    for step in range(1, num_steps):
        default = predictor.predict_next(reconstructed, actions[step - 1]).default_state
        record = cursor.peek()
        if record is not None and record.step == step:
            cursor.pop()
            reconstructed = decode_residual_record(
                record, default, schema, dictionary)
        else:
            reconstructed = default
        outputs.append(reconstructed)
    if cursor.peek() is not None:
        raise ValueError("residual record exceeds decoded trajectory")
    return tuple(outputs)


def decode_residual_record(
    record: ResidualRecord,
    default: CanonicalState,
    schema: StateSchema,
    dictionary: LiteralDictionary,
) -> CanonicalState:
    return _decode_selected(
        record.selected,
        record.payload,
        record.bit_length,
        default,
        schema,
        dictionary,
    )


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
