from __future__ import annotations


class BitWriter:
    def __init__(self) -> None:
        self._data = bytearray()
        self._bits = 0

    @property
    def bit_length(self) -> int:
        return self._bits

    def write_bits(self, value: int, count: int) -> None:
        if count < 0 or value < 0 or value >= (1 << count if count else 1):
            raise ValueError((value, count))
        for offset in range(count):
            byte_index = self._bits // 8
            bit_index = self._bits % 8
            if byte_index == len(self._data):
                self._data.append(0)
            if (value >> offset) & 1:
                self._data[byte_index] |= 1 << bit_index
            self._bits += 1

    def write_bytes(self, value: bytes) -> None:
        for byte in value:
            self.write_bits(byte, 8)

    def finish(self) -> bytes:
        return bytes(self._data)


class BitReader:
    def __init__(self, data: bytes, bit_length: int | None = None) -> None:
        self.data = data
        self.bit_length = len(data) * 8 if bit_length is None else bit_length
        if self.bit_length > len(data) * 8:
            raise ValueError("bit length exceeds payload")
        self.offset = 0

    @property
    def remaining(self) -> int:
        return self.bit_length - self.offset

    def read_bits(self, count: int) -> int:
        if count < 0 or self.offset + count > self.bit_length:
            raise EOFError("truncated bit stream")
        value = 0
        for target in range(count):
            byte_index = self.offset // 8
            bit_index = self.offset % 8
            value |= ((self.data[byte_index] >> bit_index) & 1) << target
            self.offset += 1
        return value

    def read_bytes(self, count: int) -> bytes:
        return bytes(self.read_bits(8) for _ in range(count))
