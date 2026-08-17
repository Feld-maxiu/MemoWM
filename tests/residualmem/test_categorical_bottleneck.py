"""A2 grouped categorical (PQ-8) bottleneck invariants.

The two that the 4096-bit claim rests on: the decoder never sees anything but a
gathered codeword (no continuous side channel), and the integer codes alone
reproduce the decoder input bit-for-bit. The rest assert shapes, masking,
determinism and that straight-through gradients actually reach both the encoder
and the codebook.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from residualmem.world_model import categorical_bottleneck as Q


def _tiny(**overrides) -> Q.CategoricalBottleneckConfig:
    kwargs = dict(
        num_input_slots=8, input_dim=16, num_e_tokens=4, e_dim=8,
        num_heads=2, ffn_hidden=12, group_sizes=(4, 2, 1, 1),
        num_subspaces=2, num_categories=5, temperature=0.5,
    )
    kwargs.update(overrides)
    return Q.CategoricalBottleneckConfig(**kwargs)


def _batch(config, states=5, seed=0, invalid_from=6):
    rng = np.random.default_rng(seed)
    xbar = rng.normal(
        size=(states, config.num_input_slots, config.input_dim)
    ).astype(np.float32)
    valid = np.ones((states, config.num_input_slots), bool)
    valid[:, invalid_from:] = False
    return xbar * valid[..., None], valid


def test_config_validates_product_quantization_geometry():
    with pytest.raises(ValueError):
        _tiny(num_subspaces=3)  # e_dim not divisible
    with pytest.raises(ValueError):
        _tiny(num_categories=1)
    with pytest.raises(ValueError):
        _tiny(temperature=0.0)
    config = _tiny()
    assert config.subspace_dim == config.e_dim // config.num_subspaces
    assert config.codes_per_state == config.num_e_tokens * config.num_subspaces


def test_default_capacity_accounting_is_4096_bits():
    config = Q.CategoricalBottleneckConfig()
    assert (config.num_e_tokens, config.num_subspaces, config.num_categories) == (64, 8, 256)
    assert config.codes_per_state == 512
    assert config.code_bits == pytest.approx(4096.0)
    assert config.latent_tensor_bits == 64 * 512 * 32
    assert config.tensor_reduction == pytest.approx(256.0)


def test_codebook_shape_and_parameter_reuse():
    config = _tiny()
    params = Q.initialize_params(config, 0)
    assert params[Q.CODEBOOK].shape == (
        config.num_e_tokens, config.num_subspaces,
        config.num_categories, config.subspace_dim,
    )
    # every A1 key survives unchanged; only the codebook is new
    from residualmem.world_model.continuous_bottleneck import expected_param_shapes as a1
    assert set(a1(config.continuous)) | {Q.CODEBOOK} == set(params)
    assert jnp.array_equal(
        Q.initialize_params(config, 0)[Q.CODEBOOK], params[Q.CODEBOOK]
    )


def test_expanded_distance_matches_naive_broadcast():
    config = _tiny()
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config)
    _, _, diagnostics = Q.reconstruct(
        params, xbar, valid, config, return_codes=True, return_diagnostics=True
    )
    latent = Q._split_subspaces(diagnostics["latent"], config)
    naive = jnp.sum(jnp.square(latent[..., None, :] - params[Q.CODEBOOK][None]), -1)
    expanded = -diagnostics["logits"] * config.temperature
    assert float(jnp.max(jnp.abs(naive - expanded))) < 1e-4


def test_decoder_input_is_exactly_a_gathered_codeword():
    """No continuous bypass: hard forward == pure table lookup, bit for bit."""
    config = _tiny()
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config)
    _, codes, diagnostics = Q.reconstruct(
        params, xbar, valid, config, return_codes=True, return_diagnostics=True
    )
    assert codes.shape == (5, config.num_e_tokens, config.num_subspaces)
    assert int(codes.min()) >= 0 and int(codes.max()) < config.num_categories
    assert jnp.array_equal(Q.embed_codes(params, codes, config), diagnostics["quantized"])

    # and it really is the per-subspace table entry
    codebook = np.asarray(params[Q.CODEBOOK])
    quantized = np.asarray(diagnostics["quantized"]).reshape(
        5, config.num_e_tokens, config.num_subspaces, config.subspace_dim
    )
    codes_np = np.asarray(codes)
    for token in range(config.num_e_tokens):
        for sub in range(config.num_subspaces):
            expected = codebook[token, sub, codes_np[:, token, sub]]
            assert np.array_equal(quantized[:, token, sub], expected)


def test_codes_alone_reproduce_the_reconstruction():
    config = _tiny()
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config)
    prediction, codes = Q.reconstruct(params, xbar, valid, config, return_codes=True)
    replayed = Q.decode(
        params, Q.embed_codes(params, codes, config), valid, config.continuous
    )
    assert jnp.array_equal(prediction, replayed)


def test_soft_path_is_opt_in_and_differs_from_hard():
    config = _tiny()
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config)
    hard = Q.reconstruct(params, xbar, valid, config)
    soft = Q.reconstruct(params, xbar, valid, config, soft=True)
    assert not jnp.allclose(hard, soft)
    # the default path never builds the soft mixture
    assert jnp.array_equal(hard, Q.reconstruct(params, xbar, valid, config, soft=False))


def test_forward_is_deterministic_and_rng_free():
    config = _tiny()
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config)
    assert jnp.array_equal(
        Q.reconstruct(params, xbar, valid, config),
        Q.reconstruct(params, xbar, valid, config),
    )
    single = Q.reconstruct(params, xbar[0], valid[0], config)
    assert single.shape == (config.num_input_slots, config.input_dim)
    assert jnp.allclose(single, Q.reconstruct(params, xbar, valid, config)[0], atol=1e-6)


def test_masking_semantics_survive_quantization():
    config = _tiny()
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config)
    corrupted = np.array(xbar)
    corrupted[~valid] = 1e3

    clean, clean_codes = Q.reconstruct(params, xbar, valid, config, return_codes=True)
    dirty, dirty_codes = Q.reconstruct(params, corrupted, valid, config, return_codes=True)
    assert jnp.array_equal(clean_codes, dirty_codes)
    assert jnp.array_equal(clean, dirty)
    assert int(np.count_nonzero(np.asarray(clean)[~valid])) == 0


def test_straight_through_reaches_encoder_and_codebook():
    config = _tiny()
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config)

    def loss(current):
        return Q.masked_mse(Q.reconstruct(current, xbar, valid, config), xbar, valid)

    grads = jax.grad(loss)(params)
    for name in (Q.CODEBOOK, "encoder/e_queries", "encoder/input_positions",
                 "decoder/e_addresses", "decoder/output_queries"):
        norm = float(jnp.sqrt(jnp.sum(jnp.square(grads[name]))))
        assert np.isfinite(norm) and norm > 0.0, name

    # the distance-based logits give every codeword an alignment gradient, not
    # just the selected one -- this is what stands in for VQ's commitment loss
    touched = np.abs(np.asarray(grads[Q.CODEBOOK])).sum(-1) > 0
    assert touched.sum() > np.unique(np.asarray(_selected_pairs(config, xbar, valid, params))).size


def _selected_pairs(config, xbar, valid, params):
    _, codes = Q.reconstruct(params, xbar, valid, config, return_codes=True)
    return np.asarray(codes)


def test_temperature_calibration_is_monotone_and_hits_target():
    config = _tiny(num_categories=8)
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config, states=16)
    _, _, diagnostics = Q.reconstruct(
        params, xbar, valid, config, return_codes=True, return_diagnostics=True
    )
    latent = diagnostics["latent"]

    achieved = []
    for temperature in (0.01, 0.1, 1.0, 10.0):
        _, info = Q.calibrate_temperature(
            params, latent, config, low=temperature, high=temperature, iterations=1
        )
        achieved.append(info["achieved_median_max_prob"])
    assert achieved == sorted(achieved, reverse=True), achieved

    temperature, info = Q.calibrate_temperature(params, latent, config, target=0.8)
    assert temperature > 0
    assert info["achieved_median_max_prob"] == pytest.approx(0.8, abs=0.05)
    assert info["max_prob_p5"] <= info["achieved_median_max_prob"] <= info["max_prob_p95"]


def test_code_histogram_is_integer_and_batch_split_invariant():
    config = _tiny()
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config, states=6)
    _, codes = Q.reconstruct(params, xbar, valid, config, return_codes=True)

    histogram = Q.code_histogram(codes, config)
    assert histogram.dtype == np.int64
    assert histogram.shape == (
        config.num_e_tokens, config.num_subspaces, config.num_categories
    )
    assert histogram.sum() == 6 * config.codes_per_state
    split = Q.code_histogram(codes[:2], config) + Q.code_histogram(codes[2:], config)
    assert np.array_equal(histogram, split)

    health = Q.code_health(histogram, config)
    assert health["code/bits_per_state"] == pytest.approx(config.code_bits)
    assert 1.0 <= health["code/perplexity_median"] <= config.num_categories
    assert 0.0 <= health["code/dominant_share_max"] <= 1.0


def test_code_histogram_rejects_out_of_range_codes():
    config = _tiny()
    codes = np.zeros((2, config.num_e_tokens, config.num_subspaces), np.int64)
    codes[0, 0, 0] = config.num_categories
    with pytest.raises(ValueError):
        Q.code_histogram(codes, config)
    with pytest.raises(ValueError):
        Q.code_histogram(np.zeros((2, 3, 3), np.int64), config)


# --------------------------------------------------------------------------- #
# learned assignment
# --------------------------------------------------------------------------- #
def _learned(mode, **overrides):
    return _tiny(assignment=mode, temperature=1.0, **overrides)


def test_assignment_validation_and_selector_width():
    with pytest.raises(ValueError):
        _tiny(assignment="nearest")
    assert not _tiny().is_learned
    assert _learned("learned_slice").selector_dim == _tiny().subspace_dim
    assert _learned("learned_full").selector_dim == _tiny().e_dim


@pytest.mark.parametrize("mode", ["learned_slice", "learned_mix", "learned_full"])
def test_equivalence_initialisation_reproduces_pq_exactly(mode):
    """Step 0 of a learned run must be the PQ model it started from.

    PQ's logits are already linear in r; dropping the shared -||r||^2/tau term is
    exact because argmax and softmax are both shift-invariant. So the transform
    is an identity on decisions, not merely an argmax-preserving approximation.
    """
    pq = _tiny(temperature=0.7)
    pq_params = Q.initialize_params(pq, 0)
    xbar, valid = _batch(pq, states=6)
    pq_out, pq_codes, pq_diag = Q.reconstruct(
        pq_params, xbar, valid, pq, return_codes=True, return_diagnostics=True
    )

    config = _learned(mode)
    params = Q.learned_params_from_pq(pq_params, pq, config)
    out, codes, diag = Q.reconstruct(
        params, xbar, valid, config, return_codes=True, return_diagnostics=True
    )
    assert jnp.array_equal(codes, pq_codes)
    assert jnp.array_equal(diag["quantized"], pq_diag["quantized"])
    assert jnp.array_equal(out, pq_out)
    assert jnp.allclose(diag["probs"], pq_diag["probs"], atol=1e-5)
    # the distance diagnostic is meaningless without prototypes
    assert "selected_distance" in pq_diag and "selected_distance" not in diag


def test_learned_full_starts_blind_to_other_slices():
    """Equivalence init zeroes the cross-slice weights; training opens them."""
    pq = _tiny(temperature=0.7)
    pq_params = Q.initialize_params(pq, 0)
    config = _learned("learned_full")
    weight = np.asarray(Q.learned_params_from_pq(pq_params, pq, config)[Q.SELECTOR_WEIGHT])
    dim = config.subspace_dim
    for subspace in range(config.num_subspaces):
        start = subspace * dim
        outside = np.delete(weight[:, subspace], np.arange(start, start + dim), axis=-1)
        assert not outside.any(), subspace
        assert weight[:, subspace, :, start:start + dim].any()


@pytest.mark.parametrize("mode", ["learned_slice", "learned_mix", "learned_full"])
def test_learned_forward_is_hard_and_codes_replay(mode):
    config = _learned(mode)
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config)
    prediction, codes, diagnostics = Q.reconstruct(
        params, xbar, valid, config, return_codes=True, return_diagnostics=True
    )
    assert jnp.array_equal(Q.embed_codes(params, codes, config), diagnostics["quantized"])
    replayed = Q.decode(
        params, Q.embed_codes(params, codes, config), valid, config.continuous
    )
    assert jnp.array_equal(prediction, replayed)
    assert int(np.count_nonzero(np.asarray(prediction)[~valid])) == 0
    assert jnp.array_equal(prediction, Q.reconstruct(params, xbar, valid, config))


@pytest.mark.parametrize("mode", ["learned_slice", "learned_mix", "learned_full"])
def test_learned_gradients_split_between_selector_and_embedding(mode):
    config = _learned(mode)
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config)

    def loss(current):
        return Q.masked_mse(Q.reconstruct(current, xbar, valid, config), xbar, valid)

    grads = jax.grad(loss)(params)
    for name in (Q.SELECTOR_WEIGHT, Q.SELECTOR_BIAS, Q.EMBEDDING_TABLE):
        norm = float(jnp.sqrt(jnp.sum(jnp.square(grads[name]))))
        assert np.isfinite(norm) and norm > 0.0, name
    assert Q.CODEBOOK not in grads

    # the embedding is taught only through the categories actually chosen; the
    # soft surrogate reaches the selector instead, because E is stop-gradiented
    # inside it
    _, codes = Q.reconstruct(params, xbar, valid, config, return_codes=True)
    touched = np.abs(np.asarray(grads[Q.EMBEDDING_TABLE])).sum(-1) > 0
    chosen = np.zeros_like(touched)
    codes_np = np.asarray(codes)
    for token in range(config.num_e_tokens):
        for subspace in range(config.num_subspaces):
            chosen[token, subspace, np.unique(codes_np[:, token, subspace])] = True
    assert np.array_equal(touched, chosen)


def test_learned_mix_starts_as_identity_and_is_trainable():
    """The mixer only departs from A1's basis if reconstruction asks it to."""
    pq = _tiny(temperature=0.7)
    pq_params = Q.initialize_params(pq, 0)
    config = _learned("learned_mix")
    params = Q.learned_params_from_pq(pq_params, pq, config)
    assert jnp.array_equal(params[Q.SELECTOR_MIXER], jnp.eye(config.e_dim))
    assert params[Q.SELECTOR_WEIGHT].shape[-1] == config.subspace_dim

    xbar, valid = _batch(config)

    def loss(current):
        return Q.masked_mse(Q.reconstruct(current, xbar, valid, config), xbar, valid)

    norm = float(jnp.sqrt(jnp.sum(jnp.square(jax.grad(loss)(params)[Q.SELECTOR_MIXER]))))
    assert np.isfinite(norm) and norm > 0.0

    # a non-identity mixer genuinely changes which categories win
    rotated = dict(params)
    rotated[Q.SELECTOR_MIXER] = jnp.flip(params[Q.SELECTOR_MIXER], axis=-1)
    _, codes = Q.reconstruct(params, xbar, valid, config, return_codes=True)
    _, other = Q.reconstruct(rotated, xbar, valid, config, return_codes=True)
    assert not jnp.array_equal(codes, other)


def test_learned_parameter_counts_reflect_decoupling():
    """Decoupling selector from embedding inherently doubles the tables."""
    pq = _tiny()
    counts = {}
    for mode in ("pq", "learned_slice", "learned_mix", "learned_full"):
        config = _tiny(assignment=mode, temperature=1.0)
        params = Q.initialize_params(config, 0)
        quantizer = {Q.CODEBOOK, Q.SELECTOR_WEIGHT, Q.SELECTOR_BIAS,
                     Q.SELECTOR_MIXER, Q.EMBEDDING_TABLE}
        counts[mode] = sum(
            int(v.size) for k, v in params.items() if k in quantizer
        )
    table = pq.num_e_tokens * pq.num_subspaces * pq.num_categories
    assert counts["pq"] == table * pq.subspace_dim
    assert counts["learned_slice"] == table * (2 * pq.subspace_dim + 1)
    assert counts["learned_full"] == table * (pq.e_dim + pq.subspace_dim + 1)
    # the mixer buys a full-token view for one shared matrix, not per category
    assert counts["learned_mix"] == counts["learned_slice"] + pq.e_dim ** 2
    assert (counts["pq"] < counts["learned_slice"] < counts["learned_mix"]
            < counts["learned_full"])


def test_equivalence_init_rejects_mismatched_geometry():
    pq = _tiny(temperature=0.7)
    pq_params = Q.initialize_params(pq, 0)
    with pytest.raises(ValueError):
        Q.learned_params_from_pq(pq_params, pq, _tiny())  # target is not learned
    with pytest.raises(ValueError):
        Q.learned_params_from_pq(
            pq_params, pq, _learned("learned_full", num_categories=pq.num_categories + 1)
        )


# --------------------------------------------------------------------------- #
# detail-weighted reconstruction
# --------------------------------------------------------------------------- #
def test_group_slice_matches_the_configured_layout():
    config = _tiny()  # (4, 2, 1, 1)
    spans = [Q.group_slice(config, name) for name in Q.GROUP_NAMES]
    assert spans == [slice(0, 4), slice(4, 6), slice(6, 7), slice(7, 8)]
    assert Q.group_slice(Q.CategoricalBottleneckConfig(), "detail") == slice(32, 44)
    with pytest.raises(ValueError):
        Q.group_slice(config, "dom")


def test_detail_weight_zero_is_bit_identical_to_plain_mse():
    """Regression lock: the 20 configs already trained must not shift by a ulp."""
    config = _tiny()
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config)
    prediction = Q.reconstruct(params, xbar, valid, config)

    plain = Q.masked_mse(prediction, xbar, valid)
    weighted = plain + 0.0 * Q.masked_group_mse(prediction, xbar, valid, config, "detail")
    assert float(weighted) == float(plain)


def test_lambda_one_gives_detail_exactly_double_weight():
    """The whole experiment rests on this factor being 2, not 9.2.

    Checked on the gradient of the loss w.r.t. the prediction, which is where
    the weight actually acts. A group-local denominator would show up here as a
    ratio of ``1 + total/detail`` instead of 2.
    """
    config = _tiny()
    xbar, valid = _batch(config)
    prediction = jnp.asarray(xbar) + 0.5  # any point with a nonzero residual

    def natural(p):
        return Q.masked_mse(p, xbar, valid)

    def weighted(p):
        return natural(p) + 1.0 * Q.masked_group_mse(p, xbar, valid, config, "detail")

    ratio = jax.grad(weighted)(prediction) / jax.grad(natural)(prediction)
    detail = Q.group_slice(config, "detail")
    inside = np.asarray(ratio[:, detail][valid[:, detail]])
    outside = np.asarray(
        jnp.concatenate([ratio[:, :detail.start], ratio[:, detail.stop:]], axis=1)
    )[np.concatenate([valid[:, :detail.start], valid[:, detail.stop:]], axis=1)]
    assert np.allclose(inside, 2.0)
    assert np.allclose(outside, 1.0)


def test_detail_term_ignores_other_groups_and_padding():
    config = _tiny()
    xbar, valid = _batch(config)
    prediction = np.asarray(xbar) + 0.5
    reference = float(Q.masked_group_mse(prediction, xbar, valid, config, "detail"))

    detail = Q.group_slice(config, "detail")
    perturbed = np.array(prediction)
    perturbed[:, :detail.start] += 7.0        # image slots
    perturbed[:, detail.stop:] -= 3.0         # context + prompt slots
    assert float(Q.masked_group_mse(perturbed, xbar, valid, config, "detail")) == reference

    padded = np.array(prediction)
    padded[~valid] = 1e3
    assert float(Q.masked_group_mse(padded, xbar, valid, config, "detail")) == reference


def test_detail_denominator_is_global_not_group_local():
    """The failure mode this guards is silent: only the constant changes."""
    config = _tiny()
    xbar, valid = _batch(config)
    prediction = np.asarray(xbar) + 0.5

    detail = Q.group_slice(config, "detail")
    valid_all = float(np.asarray(valid).sum())
    valid_detail = float(np.asarray(valid[:, detail]).sum())
    assert valid_detail < valid_all  # otherwise the test proves nothing

    value = float(Q.masked_group_mse(prediction, xbar, valid, config, "detail"))
    group_local = float(
        Q.masked_mse(prediction[:, detail], xbar[:, detail], valid[:, detail])
    )
    assert value == pytest.approx(group_local * valid_detail / valid_all, rel=1e-6)


# --------------------------------------------------------------------------- #
# per-group weighted reconstruction
# --------------------------------------------------------------------------- #
def test_unit_weights_reproduce_plain_mse_and_partition_it():
    """All-ones must be the natural loss, or a weighted run is not a control.

    The group terms share one global denominator, so they *partition* the plain
    MSE. Per-group denominators would break both properties at once.
    """
    config = _tiny()
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config)
    prediction = Q.reconstruct(params, xbar, valid, config)

    plain = float(Q.masked_mse(prediction, xbar, valid))
    unit = float(Q.masked_weighted_mse(prediction, xbar, valid, config, (1, 1, 1, 1)))
    assert unit == pytest.approx(plain, rel=1e-6)

    parts = sum(
        float(Q.masked_group_mse(prediction, xbar, valid, config, name))
        for name in Q.GROUP_NAMES
    )
    assert parts == pytest.approx(plain, rel=1e-6)


def test_group_weights_are_per_element_multipliers():
    """w_g must scale an element's gradient by exactly w_g, whatever the layout."""
    config = _tiny()
    xbar, valid = _batch(config)
    prediction = jnp.asarray(xbar) + 0.5
    weights = (1.0, 2.0, 0.5, 0.5)

    def natural(p):
        return Q.masked_mse(p, xbar, valid)

    def weighted(p):
        return Q.masked_weighted_mse(p, xbar, valid, config, weights)

    ratio = jax.grad(weighted)(prediction) / jax.grad(natural)(prediction)
    for name, expected in zip(Q.GROUP_NAMES, weights):
        span = Q.group_slice(config, name)
        inside = np.asarray(ratio[:, span][valid[:, span]])
        assert np.allclose(inside, expected), (name, expected)


def test_group_weights_equal_detail_weight_when_only_detail_is_raised():
    """(1,2,1,1) and --detail-weight 1 are two spellings of the same objective."""
    config = _tiny()
    params = Q.initialize_params(config, 0)
    xbar, valid = _batch(config)
    prediction = Q.reconstruct(params, xbar, valid, config)

    additive = float(
        Q.masked_mse(prediction, xbar, valid)
        + 1.0 * Q.masked_group_mse(prediction, xbar, valid, config, "detail")
    )
    weighted = float(
        Q.masked_weighted_mse(prediction, xbar, valid, config, (1, 2, 1, 1))
    )
    assert weighted == pytest.approx(additive, rel=1e-6)


def test_group_weights_validate_length_and_layout():
    config = _tiny()
    with pytest.raises(ValueError):
        Q.slot_group_weights(config, (1, 2, 3))
    slots = np.asarray(Q.slot_group_weights(config, (1, 2, 0.5, 0.25)))
    assert slots.shape == (config.num_input_slots,)
    # (4, 2, 1, 1) layout
    assert slots.tolist() == [1, 1, 1, 1, 2, 2, 0.5, 0.25]


def test_zero_weight_removes_a_group_from_the_objective():
    config = _tiny()
    xbar, valid = _batch(config)
    prediction = np.asarray(xbar) + 0.5
    span = Q.group_slice(config, "image")

    reference = float(
        Q.masked_weighted_mse(prediction, xbar, valid, config, (0, 1, 1, 1))
    )
    perturbed = np.array(prediction)
    perturbed[:, span] += 9.0
    assert float(
        Q.masked_weighted_mse(perturbed, xbar, valid, config, (0, 1, 1, 1))
    ) == reference
