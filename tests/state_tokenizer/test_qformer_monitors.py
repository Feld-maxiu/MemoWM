"""The collapse monitors, which are the gate on the Q-Former run.

These exist because the first version got the measurement wrong in a way that
looked like a catastrophic result. It normalized each state without removing
the mean across observations, so the pairwise cosine reported a shared offset
rather than a lack of spread. At step 400 that read 1.0000 for the learned
states against 0.5739 for the fixed pooling and flagged COLLAPSE-REGRESSED --
while the *centred* numbers were 0.0389 against -0.0225, i.e. close to
orthogonal in both, with 14.7 against 28.8 effective dimensions.

The two representations are not comparable uncentred: the pooled xbar is near
zero-mean by construction (frozen group/channel statistics) and a learned
resampler's output is not. ``test_a_shared_offset_does_not_look_like_collapse``
is the direct regression.

Numpy only.
"""
from __future__ import annotations

import numpy as np

from experiments.state_tokenizer.train_qformer_joint import spread


def _states(count=40, slots=8, width=16, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=(count, slots, width))


def test_a_shared_offset_does_not_look_like_collapse():
    """The bug: adding a constant to every state changed the verdict."""
    base = _states()
    offset = np.full_like(base[0], 50.0)
    shifted = base + offset

    plain, moved = spread(base), spread(shifted)
    assert abs(plain["pairwise_cosine"] - moved["pairwise_cosine"]) < 1e-6, (
        "a constant offset moved the cosine -- the mean is not being removed"
    )
    assert abs(plain["effective_rank"] - moved["effective_rank"]) < 1e-4

    # The offset is not ignored, it is reported separately -- it is a real
    # property of the representation, just not the one the gate is about.
    assert moved["mean_to_deviation"] > 10 * plain["mean_to_deviation"]


def test_effective_rank_catches_variation_trapped_in_a_subspace():
    """One of the two failure modes: the states differ, but along few axes."""
    rng = np.random.default_rng(2)
    basis = rng.normal(size=(3, 8 * 16))
    coefficients = rng.normal(size=(40, 3))
    low_rank = (coefficients @ basis).reshape(40, 8, 16)
    assert spread(low_rank)["effective_rank"] < 4.0, spread(low_rank)
    assert spread(_states())["effective_rank"] > 10.0


def test_mean_to_deviation_catches_states_that_barely_differ():
    """The other mode, and the reason effective rank alone is not enough.

    Centred effective rank is scale-invariant, so twenty near-identical states
    plus isotropic noise still read as full rank -- the variation is spread over
    every axis, it is simply tiny. Only the mean-to-deviation ratio sees that.
    """
    single = _states(count=1)[0]
    barely = np.stack([single] * 20) + _states(20, seed=1, scale=1e-6)
    assert spread(barely)["effective_rank"] > 10.0, "rank alone would call this healthy"
    assert spread(barely)["mean_to_deviation"] > 1e4
    assert spread(_states())["mean_to_deviation"] < 1.0


def test_spread_is_scale_invariant():
    """Doubling every state changes nothing about how distinguishable they are."""
    base = _states()
    for key, value in spread(base).items():
        assert abs(value - spread(base * 2.0)[key]) < 1e-6, key


def test_orthogonal_states_read_near_zero_cosine():
    identity = np.eye(30)[:, None, :]           # 30 mutually orthogonal states
    result = spread(identity)
    # Centring 30 orthogonal unit vectors leaves them at -1/(n-1) exactly.
    assert abs(result["pairwise_cosine"] + 1 / 29) < 1e-6, result
    assert result["effective_rank"] > 25.0
