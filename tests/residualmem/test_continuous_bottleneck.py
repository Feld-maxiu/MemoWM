"""A1 continuous latent bottleneck invariants.

These assert *wiring* guarantees, not model quality: padding must stay exactly
zero everywhere, encoder attention must never touch invalid slots, decoder
routing must be independent of latent content, and the whole path must be
deterministic. They are the S0 implementation gate for the A1 experiment.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from residualmem.world_model import continuous_bottleneck as C


def _tiny(**overrides) -> C.ContinuousBottleneckConfig:
    kwargs = dict(
        num_input_slots=8,
        input_dim=16,
        num_e_tokens=3,
        e_dim=8,
        num_heads=2,
        ffn_hidden=12,
        group_sizes=(4, 2, 1, 1),
    )
    kwargs.update(overrides)
    return C.ContinuousBottleneckConfig(**kwargs)


def _batch(config, states=5, seed=0, invalid_from=6):
    rng = np.random.default_rng(seed)
    xbar = rng.normal(size=(states, config.num_input_slots, config.input_dim)).astype(np.float32)
    valid = np.ones((states, config.num_input_slots), bool)
    valid[:, invalid_from:] = False
    return xbar * valid[..., None], valid


def test_config_rejects_inconsistent_shapes():
    with pytest.raises(ValueError):
        _tiny(group_sizes=(4, 2, 1))  # does not sum to num_input_slots
    with pytest.raises(ValueError):
        _tiny(e_dim=9)  # not divisible by num_heads
    with pytest.raises(ValueError):
        _tiny(num_e_tokens=0)
    assert C.ContinuousBottleneckConfig(group_sizes=[32, 12, 16, 4]).group_sizes == (32, 12, 16, 4)


def test_initialization_is_deterministic_and_seed_dependent():
    config = _tiny()
    first = C.initialize_params(config, seed=1)
    again = C.initialize_params(config, seed=1)
    other = C.initialize_params(config, seed=2)
    assert set(first) == set(C.expected_param_shapes(config))
    for name, value in first.items():
        assert jnp.array_equal(value, again[name])
        assert tuple(value.shape) == C.expected_param_shapes(config)[name]
    assert not jnp.array_equal(first["encoder/e_queries"], other["encoder/e_queries"])


def test_default_contract_shapes():
    config = C.ContinuousBottleneckConfig()
    params = C.initialize_params(config, 0)
    xbar, valid = _batch(config, states=2, invalid_from=60)
    e_t = C.encode(params, xbar, valid, config)
    xbar_hat = C.decode(params, e_t, valid, config)
    assert e_t.shape == (2, 16, 512)
    assert xbar_hat.shape == (2, 64, 512)
    assert config.input_scalars == 64 * 512
    assert config.e_scalars == 16 * 512
    assert config.scalar_ratio == pytest.approx(0.25)


def test_batched_and_unbatched_agree_and_are_deterministic():
    config = _tiny()
    params = C.initialize_params(config, 0)
    xbar, valid = _batch(config)
    e_batch, attention = C.encode(params, xbar, valid, config, return_attention=True)
    assert e_batch.shape == (5, config.num_e_tokens, config.e_dim)
    assert attention.shape == (5, config.num_heads, config.num_e_tokens, config.num_input_slots)
    single = C.encode(params, xbar[0], valid[0], config)
    assert single.shape == (config.num_e_tokens, config.e_dim)
    assert jnp.allclose(single, e_batch[0], atol=1e-6)
    assert jnp.array_equal(
        C.reconstruct(params, xbar, valid, config),
        C.reconstruct(params, xbar, valid, config),
    )


def test_rejects_mismatched_shapes():
    config = _tiny()
    params = C.initialize_params(config, 0)
    xbar, valid = _batch(config)
    with pytest.raises(ValueError):
        C.encode(params, xbar[..., :-1], valid, config)
    with pytest.raises(ValueError):
        C.encode(params, xbar, valid[:-1], config)
    with pytest.raises(ValueError):
        C.decode(params, jnp.zeros((5, config.num_e_tokens + 1, config.e_dim)), valid, config)


def test_all_invalid_state_stays_finite():
    config = _tiny()
    params = C.initialize_params(config, 0)
    valid = np.zeros((config.num_input_slots,), bool)
    xbar = np.zeros((config.num_input_slots, config.input_dim), np.float32)
    e_t, attention = C.encode(params, xbar, valid, config, return_attention=True)
    xbar_hat = C.decode(params, e_t, valid, config)
    assert bool(jnp.all(jnp.isfinite(e_t)))
    assert float(jnp.sum(attention)) == 0.0
    assert float(jnp.max(jnp.abs(xbar_hat))) == 0.0


def test_invalid_slots_are_ignored_and_outputs_stay_zero():
    config = _tiny()
    params = C.initialize_params(config, 0)
    xbar, valid = _batch(config)
    corrupted = np.array(xbar)
    corrupted[~valid] = 1e3  # garbage in padding must not leak anywhere

    clean_e, clean_attention = C.encode(params, xbar, valid, config, return_attention=True)
    dirty_e, dirty_attention = C.encode(params, corrupted, valid, config, return_attention=True)
    assert jnp.array_equal(clean_e, dirty_e)
    assert jnp.array_equal(clean_attention, dirty_attention)

    # exact zero attention on invalid keys, exact zero on invalid outputs
    assert float(jnp.sum(clean_attention * (~valid)[:, None, None, :])) == 0.0
    prediction = C.reconstruct(params, corrupted, valid, config)
    assert int(np.count_nonzero(np.asarray(prediction)[~valid])) == 0
    row_sums = jnp.sum(clean_attention, axis=-1)
    assert float(jnp.max(jnp.abs(row_sums - 1.0))) < 1e-6


def test_decoder_routing_is_independent_of_latent_content():
    config = _tiny()
    params = C.initialize_params(config, 0)
    xbar, valid = _batch(config)
    e_t = C.encode(params, xbar, valid, config)

    first = C.decode(params, e_t, valid, config, return_attention=True)[1]
    second = C.decode(params, e_t * 7.0 - 3.0, valid, config, return_attention=True)[1]
    assert jnp.array_equal(first, second)
    assert first.shape == (config.num_heads, config.num_input_slots, config.num_e_tokens)
    # but the reconstruction must still respond to latent content
    assert not jnp.allclose(
        C.decode(params, e_t, valid, config),
        C.decode(params, e_t * 7.0 - 3.0, valid, config),
    )


def test_masked_mse_uses_valid_slots_times_token_dim():
    config = _tiny()
    xbar, valid = _batch(config, states=2)
    prediction = np.zeros_like(xbar)
    expected = float(
        np.sum(np.square(xbar) * valid[..., None]) / (valid.sum() * config.input_dim)
    )
    assert float(C.masked_mse(prediction, xbar, valid)) == pytest.approx(expected, rel=1e-6)
    with pytest.raises(ValueError):
        C.masked_mse(prediction[:-1], xbar, valid)


def test_group_metrics_slice_groups_and_match_zero_baseline():
    config = _tiny()
    xbar, valid = _batch(config, states=3)
    metrics = C.group_metrics(np.zeros_like(xbar), xbar, valid, config)
    assert metrics["all/mse"] == pytest.approx(metrics["all/zero_mse"], rel=1e-6)
    assert metrics["all/r2"] == pytest.approx(0.0, abs=1e-6)
    start = 0
    for name, size in zip(C.GROUP_NAMES, config.group_sizes):
        stop = start + size
        assert metrics[f"{name}/valid_slots"] == pytest.approx(
            float(valid[:, start:stop].sum())
        )
        start = stop
    # SSE accumulation over batches equals a single-shot evaluation
    totals = {}
    for chunk in (slice(0, 1), slice(1, 3)):
        partial = C.group_sse(
            np.zeros_like(xbar[chunk]), xbar[chunk], valid[chunk], config
        )
        for key, value in partial.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    assert C.metrics_from_sse(totals)["all/mse"] == pytest.approx(metrics["all/mse"], rel=1e-6)


def test_gradients_reach_addresses_and_queries():
    config = _tiny()
    params = C.initialize_params(config, 0)
    xbar, valid = _batch(config)

    def loss(current):
        return C.masked_mse(C.reconstruct(current, xbar, valid, config), xbar, valid)

    grads = jax.grad(loss)(params)
    for name in (
        "encoder/e_queries",
        "encoder/input_positions",
        "decoder/e_addresses",
        "decoder/output_queries",
    ):
        norm = float(jnp.sqrt(jnp.sum(jnp.square(grads[name]))))
        assert np.isfinite(norm) and norm > 0.0, name


def test_attention_diagnostics_report_expected_invariants():
    config = _tiny()
    params = C.initialize_params(config, 0)
    xbar, valid = _batch(config)
    _, attention = C.reconstruct(params, xbar, valid, config, return_attention=True)
    diagnostics = C.attention_diagnostics(
        attention["encoder"], attention["decoder"], valid, config
    )
    assert float(diagnostics["encoder/invalid_weight_sum"]) == 0.0
    assert float(diagnostics["encoder/row_sum_max_error"]) < 1e-6
    assert float(diagnostics["decoder/row_sum_max_error"]) < 1e-6
    assert 0.0 < float(diagnostics["decoder/effective_e_tokens"]) <= config.num_e_tokens
    mass = sum(float(diagnostics[f"encoder/mass_{name}"]) for name in C.GROUP_NAMES)
    assert mass == pytest.approx(1.0, abs=1e-5)
    assert "attention/top1_slot_accuracy" not in diagnostics  # A0-only metric
