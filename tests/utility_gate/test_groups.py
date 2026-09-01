"""The gate grouping, checked against the quantiser it claims to describe.

``groups`` asserts two things about ``qformer_pq`` that nothing else enforces:
subspaces are split inside a slot, and the flat position index is
``slot * 32 + subspace``. Both are load-bearing -- if either were wrong, every
per-position utility label would be attributed to the wrong code, and nothing
downstream would raise, because the shapes stay (32, 32) either way.

So the locality test does not read the convention out of ``qformer_pq``'s source;
it perturbs one slot of a synthetic state, runs the real ``encode``, and asserts
that only that slot's codes moved. That is the property the gate depends on,
tested as behaviour rather than as a comment.

Deliberately free of torch and jax imports so it runs under either interpreter.
``qformer_pq``'s encode/decode path is numpy-only; its single jax dependency is a
function-local import inside ``fit_codebook``, which this test never calls.
"""
from __future__ import annotations

import numpy as np

from experiments.utility_gate.groups import (
    BUNDLE_GRANULARITY,
    GRANULARITIES,
    NUM_POSITIONS,
    NUM_SLOTS,
    NUM_SUBSPACES,
    POSITION_GRANULARITY,
    SLOT_GRANULARITY,
    SUBSPACES_PER_BUNDLE,
    group_of,
    position_of,
    positions_in_group,
    raw_mask_bits,
    slot_of,
    subspace_of,
)

SUBSPACE_DIM = 16
NUM_CATEGORIES = 8


def _synthetic_codebook(seed: int = 0):
    rng = np.random.default_rng(seed)
    mean = np.zeros((NUM_SLOTS, NUM_SLOTS * SUBSPACE_DIM), np.float32)
    scale = np.ones((NUM_SLOTS, NUM_SLOTS * SUBSPACE_DIM), np.float32)
    centroids = rng.standard_normal(
        (NUM_SLOTS, NUM_SUBSPACES, NUM_CATEGORIES, SUBSPACE_DIM)
    ).astype(np.float32)
    return mean, scale, centroids


def test_every_granularity_partitions_the_positions():
    for granularity in GRANULARITIES:
        assignment = group_of(granularity)
        assert len(assignment) == NUM_POSITIONS
        assert set(assignment) == set(range(granularity))
        sizes = {g: assignment.count(g) for g in range(granularity)}
        assert len(set(sizes.values())) == 1, "groups must be equal-sized"


def test_index_round_trip():
    for position in range(NUM_POSITIONS):
        assert position_of(slot_of(position), subspace_of(position)) == position


def test_bundles_refine_slots():
    """Every bundle lies inside exactly one slot, so slot == sum of 4 bundles."""
    slots = group_of(SLOT_GRANULARITY)
    bundles = group_of(BUNDLE_GRANULARITY)
    by_bundle: dict[int, set[int]] = {}
    for position in range(NUM_POSITIONS):
        by_bundle.setdefault(bundles[position], set()).add(slots[position])
    assert all(len(owners) == 1 for owners in by_bundle.values())
    assert len(by_bundle) == BUNDLE_GRANULARITY
    assert all(
        len(positions_in_group(BUNDLE_GRANULARITY, g)) == SUBSPACES_PER_BUNDLE
        for g in range(BUNDLE_GRANULARITY)
    )


def test_raw_mask_bits_is_one_per_group():
    assert raw_mask_bits(SLOT_GRANULARITY) == 32
    assert raw_mask_bits(BUNDLE_GRANULARITY) == 128
    assert raw_mask_bits(POSITION_GRANULARITY) == 1024


def test_encode_is_slot_local():
    """Perturbing slot k changes only ``codes[:, k, :]``.

    This is the assumption that makes a slot a contiguous block of positions. It
    would fail if the quantiser rotated across slots rather than within them.
    """
    from experiments.state_tokenizer.qformer_pq import encode

    mean, scale, centroids = _synthetic_codebook()
    rng = np.random.default_rng(1)
    states = rng.standard_normal(
        (4, NUM_SLOTS, NUM_SLOTS * SUBSPACE_DIM)
    ).astype(np.float32)
    before = encode(states, mean, scale, centroids)

    perturbed = states.copy()
    perturbed[:, 7, :] += 5.0 * rng.standard_normal(
        (4, NUM_SLOTS * SUBSPACE_DIM)
    ).astype(np.float32)
    after = encode(perturbed, mean, scale, centroids)

    moved = np.flatnonzero((before != after).any(axis=(0, 2)))
    assert moved.tolist() == [7], moved.tolist()


def test_flat_index_matches_c_order_of_the_code_array():
    """``position_of`` must agree with how the codebook addresses its problems.

    ``fit_codebook`` fans the 1024 problems out as ``p // 32, p % 32`` and ships
    ``occupancy`` as (1024, C); anything that joins per-position quantities back
    to ``codes[slot, subspace]`` has to use the same order.
    """
    from experiments.state_tokenizer.qformer_pq import encode

    mean, scale, centroids = _synthetic_codebook()
    rng = np.random.default_rng(2)
    states = rng.standard_normal(
        (3, NUM_SLOTS, NUM_SLOTS * SUBSPACE_DIM)
    ).astype(np.float32)
    codes = encode(states, mean, scale, centroids)
    flat = codes.reshape(codes.shape[0], -1)
    for slot in range(NUM_SLOTS):
        for subspace in range(NUM_SUBSPACES):
            assert np.array_equal(
                flat[:, position_of(slot, subspace)], codes[:, slot, subspace]
            )
