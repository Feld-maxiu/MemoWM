"""Conditional entropy coder for latent residuals (report Eq. 19 / Eq. 22).

A 32-bit rANS coder that codes each sent latent group's symbol against the *prior*
categorical distribution ``p_theta(z_t^j | h_t)``. Because the coder consumes the
prior distribution symbol-by-symbol, and encoder + decoder both advance the RSSM
with the same decoder-side state ``z_tilde``, they reproduce identical frequency
tables -- so ``-log2 p`` coding is exactly reversible. This is where "a more
accurate world model spends fewer bits" appears in the real byte stream.

Frequency tables are quantized deterministically from prior probabilities to
``2^scale_bits`` with every symbol given mass >= 1 (any realized symbol is codable).
"""
from __future__ import annotations

import numpy as np

RANS_L = 1 << 23  # lower bound of the normalized rANS interval


def quantize_freqs(probs: np.ndarray, scale_bits: int = 16) -> np.ndarray:
    """Deterministic prob -> integer frequencies summing to exactly ``2^scale_bits``."""
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum()
    total = 1 << scale_bits
    freqs = np.floor(probs * total).astype(np.int64)
    freqs = np.maximum(freqs, 1)
    diff = total - int(freqs.sum())
    # Repair the sum on the most massive symbol (kept large, stays >= 1).
    order = np.argsort(freqs)[::-1]
    idx = 0
    while diff != 0:
        j = order[idx % len(order)]
        step = 1 if diff > 0 else -1
        if freqs[j] + step >= 1:
            freqs[j] += step
            diff -= step
        idx += 1
    return freqs


def _starts(freqs: np.ndarray) -> np.ndarray:
    starts = np.zeros(len(freqs) + 1, dtype=np.int64)
    np.cumsum(freqs, out=starts[1:])
    return starts


class RansEncoder:
    """Buffers (symbol, freqs) in forward order; emits a byte stream in :meth:`finish`."""

    def __init__(self, scale_bits: int = 16):
        self.scale_bits = scale_bits
        self._ops: list[tuple[int, int, int]] = []  # (symbol_start, freq)

    def encode(self, symbol: int, freqs: np.ndarray) -> None:
        starts = _starts(freqs)
        self._ops.append((int(starts[symbol]), int(freqs[symbol])))

    def finish(self) -> bytes:
        x = RANS_L
        renorm: list[int] = []
        x_max_base = (RANS_L >> self.scale_bits) << 8
        for start, freq in reversed(self._ops):
            x_max = x_max_base * freq
            while x >= x_max:
                renorm.append(x & 0xFF)
                x >>= 8
            x = ((x // freq) << self.scale_bits) + (x % freq) + start
        state = bytes((x >> (8 * i)) & 0xFF for i in range(4))
        return state + bytes(reversed(renorm))


class RansDecoder:
    """Streams symbols forward; caller supplies the same per-symbol ``freqs``."""

    def __init__(self, data: bytes, scale_bits: int = 16):
        if len(data) < 4:
            raise ValueError("rANS stream too short")
        self.scale_bits = scale_bits
        self.mask = (1 << scale_bits) - 1
        self.data = data
        self.pos = 4
        self.x = int.from_bytes(data[:4], "little")

    def decode(self, freqs: np.ndarray) -> int:
        starts = _starts(freqs)
        slot = self.x & self.mask
        symbol = int(np.searchsorted(starts, slot, side="right") - 1)
        freq = int(freqs[symbol])
        start = int(starts[symbol])
        self.x = freq * (self.x >> self.scale_bits) + slot - start
        while self.x < RANS_L:
            if self.pos >= len(self.data):
                raise ValueError("rANS stream exhausted")
            self.x = (self.x << 8) | self.data[self.pos]
            self.pos += 1
        return symbol

    @property
    def consumed(self) -> int:
        return self.pos
