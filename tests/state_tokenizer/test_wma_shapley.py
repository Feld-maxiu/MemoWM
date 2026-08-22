"""Shapley decomposition used by the WMA shift attribution.

The point of using Shapley here rather than raw arm differences is that the
retrieval head is nonlinear: ``f(V,T) != f(V,0) + f(0,T) - f(0,0)``, so the
obvious "visual cosine over text cosine" reading has no defensible denominator.
Shapley's efficiency axiom supplies one -- the attributions sum exactly to
``full - empty`` -- and these tests pin that property, since a decomposition that
does not sum correctly would still print plausible-looking percentages.

Runs in the torch environment only: the module under test loads the retrieval
head, so it imports torch at module scope. That is fine here -- unlike the slot
layout or the PCA binding, this code has no jax-side caller to keep in sync.
"""
from __future__ import annotations

import itertools

import numpy as np

from experiments.state_tokenizer.wma_shift_attribution import (
    modality_interaction,
    shapley,
)


def _values(fn, players, n=32, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, players))
    return {
        mask: fn(np.asarray(mask, float), base)
        for mask in itertools.product((0, 1), repeat=players)
    }


def test_efficiency_holds_for_a_nonlinear_value_function():
    # tanh of a weighted sum: no additive decomposition exists, which is the
    # case the tool is actually used in.
    values = _values(lambda m, b: np.tanh(b @ (m * np.array([1.0, -2.0]))), 2)
    result = shapley(values, 2)
    assert abs(result["efficiency_residual"]) < 1e-12
    assert result["max_abs_efficiency_residual"] < 1e-12
    assert abs(sum(result["phi"]) - result["grand"]) < 1e-12


def test_efficiency_holds_for_three_players():
    values = _values(lambda m, b: np.exp(b @ m) / (1.0 + np.abs(b @ m)), 3)
    result = shapley(values, 3)
    assert abs(result["efficiency_residual"]) < 1e-12
    assert abs(sum(result["phi"]) - result["grand"]) < 1e-12


def test_additive_value_function_gives_the_plain_marginals():
    weights = np.array([3.0, -1.0])
    values = _values(lambda m, b: b @ (m * weights), 2)
    result = shapley(values, 2)
    # With no interaction each attribution is just that player's own term.
    empty = values[(0, 0)]
    for player, mask in enumerate([(1, 0), (0, 1)]):
        marginal = float((values[mask] - empty).mean())
        assert abs(result["phi"][player] - marginal) < 1e-12


def test_symmetric_players_receive_equal_attribution():
    values = _values(lambda m, b: np.tanh(b[:, 0] * m[0] + b[:, 0] * m[1]), 2)
    result = shapley(values, 2)
    assert abs(result["phi"][0] - result["phi"][1]) < 1e-12


def test_two_player_formula_matches_the_closed_form():
    values = _values(lambda m, b: np.sin(b @ m) + b[:, 0] * m[0] * m[1], 2)
    result = shapley(values, 2)
    m00, m10, m01, m11 = (values[k] for k in ((0, 0), (1, 0), (0, 1), (1, 1)))
    phi_v = 0.5 * ((m10 - m00) + (m11 - m01))
    phi_t = 0.5 * ((m01 - m00) + (m11 - m10))
    assert abs(result["phi"][0] - phi_v.mean()) < 1e-12
    assert abs(result["phi"][1] - phi_t.mean()) < 1e-12


def test_interaction_is_zero_exactly_when_the_function_is_additive():
    additive = _values(lambda m, b: b @ m, 2)
    assert abs(modality_interaction(additive)["abs_mean"]) < 1e-12

    # A product term is pure interaction: neither player explains it alone.
    coupled = _values(lambda m, b: b[:, 0] * m[0] * m[1], 2)
    assert modality_interaction(coupled)["abs_mean"] > 0.1


def test_efficiency_survives_a_large_interaction():
    # Efficiency is an identity, so it holds even when the shares are
    # meaningless -- which is exactly why the interaction has to be reported
    # next to them rather than inferred from the residual.
    values = _values(lambda m, b: b[:, 0] * m[0] * m[1] * 10.0, 2)
    result = shapley(values, 2)
    assert abs(result["efficiency_residual"]) < 1e-12
    assert modality_interaction(values)["abs_mean"] > 1.0
