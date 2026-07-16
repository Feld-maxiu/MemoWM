from __future__ import annotations

from .bits import BitReader, BitWriter


def encode_uvarint(value: int) -> bytes:
    if value < 0:
        raise ValueError("uvarint cannot encode a negative value")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(output)


def decode_uvarint(data: bytes, offset: int = 0) -> tuple[int, int]:
    value = 0
    shift = 0
    for index in range(offset, len(data)):
        byte = data[index]
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, index + 1
        shift += 7
        if shift > 63:
            raise ValueError("uvarint is too large")
    raise EOFError("truncated uvarint")


def write_uvarint(writer: BitWriter, value: int) -> None:
    writer.write_bytes(encode_uvarint(value))


def read_uvarint(reader: BitReader) -> int:
    value = 0
    shift = 0
    while True:
        byte = reader.read_bits(8)
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value
        shift += 7
        if shift > 63:
            raise ValueError("uvarint is too large")


def zigzag_encode(value: int) -> int:
    return value * 2 if value >= 0 else -value * 2 - 1


def zigzag_decode(value: int) -> int:
    return value // 2 if value % 2 == 0 else -(value // 2) - 1
