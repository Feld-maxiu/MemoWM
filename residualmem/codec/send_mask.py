"""Send-mask policies (report Section 7.3 / Eq. 20).

The mask ``m_t^j`` decides, per latent group, whether the realized code is stored
(the decoder reads it) or dropped (the decoder fills the prior mode, accepting
distortion). We only ever consider *mismatched* groups -- a group the prior
already predicts is free. Removing Section 8's task-utility value head, the write
decision is pure rate-distortion over the state reconstruction:

* :class:`ExactMaskPolicy` -- send every mismatch (lambda -> 0). Lossless w.r.t.
  the realized latent ``z^+``; the codec round-trips it exactly.
* :class:`RateDistortionMaskPolicy` -- send a mismatch only when its conditional
  code length ``-log2 p_j(z^+_j)`` is within the rate budget ``lam`` (and under an
  optional per-step cap). Larger ``lam`` -> more stored -> lower distortion.

A learned head ``p_eta(m_t | h_t)`` trained by RD gradient can later refine which
groups are worth their bits; it plugs in through the same ``select`` interface.
"""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class MaskPolicy(Protocol):
    policy_id: str

    def select(self, prior_logits: np.ndarray, zplus: np.ndarray,
               prior_argmax: np.ndarray) -> list[int]:
        """Return the sorted group indices to store this step."""


def _mismatched(zplus: np.ndarray, prior_argmax: np.ndarray) -> list[int]:
    return [j for j in range(len(zplus)) if int(zplus[j]) != int(prior_argmax[j])]


def _bits_for(prior_logits_row: np.ndarray, symbol: int) -> float:
    row = np.asarray(prior_logits_row, dtype=np.float64)
    row = row - row.max()
    logp = row - math.log(float(np.exp(row).sum()))
    return -logp[symbol] / math.log(2.0)


class ExactMaskPolicy:
    policy_id = "exact"

    def select(self, prior_logits, zplus, prior_argmax) -> list[int]:
        return _mismatched(zplus, prior_argmax)


class RateDistortionMaskPolicy:
    """Store cheap corrections first; drop mismatches costlier than ``lam`` bits."""

    def __init__(self, lam: float = 8.0, max_groups: int | None = None):
        self.lam = float(lam)
        self.max_groups = max_groups
        self.policy_id = f"rd(lam={lam},cap={max_groups})"

    def select(self, prior_logits, zplus, prior_argmax) -> list[int]:
        candidates = []
        for j in _mismatched(zplus, prior_argmax):
            bits = _bits_for(prior_logits[j], int(zplus[j]))
            if bits <= self.lam:
                candidates.append((bits, j))
        candidates.sort()
        if self.max_groups is not None:
            candidates = candidates[: self.max_groups]
        return sorted(j for _, j in candidates)
