"""Gate groups over the 1024 code positions, with no third-party dependencies.

A state is quantised into ``codes[slot, subspace]`` of shape (32, 32). The send
gate decides, per group of positions, whether to transmit the true code or let
the decoder fall back to the world model's estimate. This module is the single
definition of how the 1024 positions are bundled into groups.

It lives alone, free of numpy/torch/jax, for the same reason ``slot_layout``
does: the world-model side runs under the jax venv and the reader side under the
torch env, neither can import the other's dependencies, and a drift between two
copies of the grouping would mis-attribute every utility label without raising.

Two facts this module encodes, both read off the quantiser rather than assumed:

* **Subspaces split inside a slot.** ``qformer_pq.encode`` reshapes
  ``(count, slots, 512)`` to ``(count, slots, num_subspaces, subspace_dim)``
  (``qformer_pq.py:172``), touching only the last axis. Slot ``s`` never mixes
  with slot ``s'``, so a slot is a contiguous block of positions.
* **Flat index is ``slot * num_subspaces + subspace``.** ``fit_codebook``
  addresses problem ``p`` as ``blocked[:, p // num_subspaces, p % num_subspaces]``
  (``qformer_pq.py:143``), which is also why ``occupancy`` ships as (1024, C).

Caveat worth keeping in mind when reading per-subspace results: the 16 dims a
subspace owns are *rotated* coordinates. Under ``rotation="shared"`` subspace m
owns PCA/OPQ components ``order[0][16m:16m+16]``, so "slot s, subspace m" has no
standing interpretation in raw xbar channels. Slot identity does survive.

The granularity is not a free choice, because the send mask has to be
transmitted. A raw mask costs one bit per group, so gating all 1024 positions
spends 1024 bits against a code rate near 5000-6000 -- about 20% of the budget,
before any code is saved. Gating the 32 slots costs 32 bits, about 0.6%. That
trade is what ``rate_utility_curve`` measures; this module only names the
options.
"""
from __future__ import annotations

PROTOCOL = "residualmem_utility_gate_v1"

# Fixed upstream by the Q-Former (32 queries) and the OPQ fit (M=32). Neither is
# a tunable here: the codebook, the world model's code head and the connector's
# rank embedding are all sized against them.
NUM_SLOTS = 32
NUM_SUBSPACES = 32
NUM_POSITIONS = NUM_SLOTS * NUM_SUBSPACES

# 32 subspaces per slot bundled four ways gives 8 subspaces per bundle. Chosen so
# that the middle granularity is a clean refinement of the slot: every bundle
# lies inside exactly one slot, so a slot-level result can be read as the sum of
# its four bundles (up to the additivity gap, which is measured, not assumed).
NUM_BUNDLES_PER_SLOT = 4
SUBSPACES_PER_BUNDLE = NUM_SUBSPACES // NUM_BUNDLES_PER_SLOT

SLOT_GRANULARITY = NUM_SLOTS                            # 32
BUNDLE_GRANULARITY = NUM_SLOTS * NUM_BUNDLES_PER_SLOT   # 128
POSITION_GRANULARITY = NUM_POSITIONS                    # 1024

GRANULARITIES = (SLOT_GRANULARITY, BUNDLE_GRANULARITY, POSITION_GRANULARITY)

GRANULARITY_NAMES = {
    SLOT_GRANULARITY: "slot",
    BUNDLE_GRANULARITY: "bundle",
    POSITION_GRANULARITY: "position",
}


def slot_of(position: int) -> int:
    return position // NUM_SUBSPACES


def subspace_of(position: int) -> int:
    return position % NUM_SUBSPACES


def position_of(slot: int, subspace: int) -> int:
    return slot * NUM_SUBSPACES + subspace


def group_of(granularity: int) -> tuple[int, ...]:
    """Map each of the 1024 positions to its group id under ``granularity``.

    Returns a tuple of length 1024 whose values cover ``range(granularity)``
    exactly. Deliberately a plain tuple so both interpreters can hold it without
    numpy.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"granularity {granularity} is not one of {GRANULARITIES}"
        )
    if granularity == POSITION_GRANULARITY:
        return tuple(range(NUM_POSITIONS))
    if granularity == SLOT_GRANULARITY:
        return tuple(slot_of(p) for p in range(NUM_POSITIONS))
    return tuple(
        slot_of(p) * NUM_BUNDLES_PER_SLOT + subspace_of(p) // SUBSPACES_PER_BUNDLE
        for p in range(NUM_POSITIONS)
    )


def positions_in_group(granularity: int, group: int) -> tuple[int, ...]:
    assignment = group_of(granularity)
    return tuple(p for p in range(NUM_POSITIONS) if assignment[p] == group)


def raw_mask_bits(granularity: int) -> int:
    """Cost of shipping an unmodelled send mask: one bit per group.

    The entropy-coded alternative -- ``-log2 p_eta(m | h_t)`` under a prior the
    decoder can evaluate -- is strictly cheaper, but it is a learned quantity and
    so cannot live in a dependency-free module. Both are reported; this is the
    one that needs no model.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"granularity {granularity} is not one of {GRANULARITIES}"
        )
    return granularity


assert NUM_POSITIONS == 1024, NUM_POSITIONS
assert NUM_SUBSPACES % NUM_BUNDLES_PER_SLOT == 0, NUM_BUNDLES_PER_SLOT
assert SUBSPACES_PER_BUNDLE == 8, SUBSPACES_PER_BUNDLE
assert BUNDLE_GRANULARITY == 128, BUNDLE_GRANULARITY
assert set(GRANULARITY_NAMES) == set(GRANULARITIES)
for _granularity in GRANULARITIES:
    _assignment = group_of(_granularity)
    assert len(_assignment) == NUM_POSITIONS
    assert set(_assignment) == set(range(_granularity)), _granularity
