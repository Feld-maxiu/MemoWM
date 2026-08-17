from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from experiments.world_model.model import (
    ModelConfig,
    block_causal_mask,
    codelength_bits,
    initialize_params,
    predict,
)


def _config():
    return ModelConfig(
        num_tasks=2, max_history=3, num_latent_tokens=4,
        num_subspaces=2, num_categories=8, num_observation_slots=4,
        code_embedding_dim=4, d_model=8, num_layers=1, num_heads=2,
        mlp_dim=16, dropout=0.0, action_embedding_dim=4,
        byte_embedding_dim=3, payload_hidden_dim=5, max_payload_bytes=4,
    )


def _inputs(config):
    batch, history = 2, config.max_history
    codes = np.zeros((batch, history, 4, 2), np.uint8)
    valid = np.ones((batch, history, 4), np.bool_)
    present = np.ones((batch, history), np.bool_)
    actions = {
        "types": np.zeros((batch, history), np.uint8),
        "tags": np.zeros((batch, history), np.uint8),
        "refs": np.zeros((batch, history), np.uint8),
        "payloads": np.zeros((batch, history, 4), np.uint8),
        "lengths": np.zeros((batch, history), np.uint8),
    }
    return codes, valid, present, actions


def test_attention_is_bidirectional_within_time_and_causal_across_time():
    mask = np.asarray(block_causal_mask(3, 4))
    assert mask[2, 0] and mask[0, 3]
    assert not mask[0, 4]
    assert mask[8, 7]


def test_output_shapes_and_classifier_weight_is_tied():
    config = _config()
    params = initialize_params(config, 0)
    assert "code_head/w" not in params
    codes, valid, present, actions = _inputs(config)
    mask_logits, code_logits = predict(
        params, codes, valid, actions, np.zeros((2,), np.uint8),
        "full", config, history_present=present,
    )
    assert mask_logits.shape == (2, 4)
    assert code_logits.shape == (2, 4, 2, 8)


def test_uniform_codelength_charges_every_code_when_all_slots_invalid():
    mask_logits = jnp.zeros((1, 4))
    code_logits = jnp.zeros((1, 4, 2, 8))
    rates = codelength_bits(
        mask_logits, code_logits, jnp.zeros((1, 4), bool),
        jnp.zeros((1, 4, 2), jnp.int32),
    )
    assert float(rates["mask_bits"][0]) == 4.0
    assert float(rates["code_bits"][0]) == 4 * 2 * 3


def test_no_history_and_t_only_cannot_leak_older_episode_content():
    config = _config()
    params = initialize_params(config, 0)
    codes, valid, present, actions = _inputs(config)
    changed_codes = codes.copy()
    changed_codes[:, :-1] = 7
    changed_valid = valid.copy()
    changed_valid[:, :-1] = False
    changed_actions = {name: value.copy() for name, value in actions.items()}
    changed_actions["refs"][:, :-1] = 12
    for variant in ("no_history", "t_only"):
        first = predict(
            params, codes, valid, actions, np.zeros((2,), np.uint8), variant,
            config, history_present=present,
        )
        second = predict(
            params, changed_codes, changed_valid, changed_actions,
            np.zeros((2,), np.uint8), variant, config, history_present=present,
        )
        assert np.array_equal(np.asarray(first[0]), np.asarray(second[0]))
        assert np.array_equal(np.asarray(first[1]), np.asarray(second[1]))


def test_no_history_routes_removed_prefix_through_learned_masks():
    config = _config()
    params = initialize_params(config, 0)
    codes, valid, present, actions = _inputs(config)

    def objective(current):
        mask, logits = predict(
            current, codes, valid, actions, np.zeros((2,), np.uint8),
            "no_history", config, history_present=present,
        )
        return jnp.sum(mask) + jnp.sum(logits)

    gradients = jax.grad(objective)(params)
    assert float(jnp.linalg.norm(gradients["state_mask"])) > 0.0
    assert float(jnp.linalg.norm(gradients["action/mask"])) > 0.0


def test_all_five_variants_have_finite_gradients_with_one_parameter_tree():
    config = _config()
    params = initialize_params(config, 0)
    codes, valid, present, actions = _inputs(config)
    for variant in ("t_only", "no_action", "structural_action", "no_history", "full"):
        def objective(current):
            mask, logits = predict(
                current, codes, valid, actions, np.zeros((2,), np.uint8),
                variant, config, history_present=present,
            )
            return jnp.mean(mask) + jnp.mean(logits)
        grads = jax.grad(objective)(params)
        assert all(np.isfinite(np.asarray(value)).all()
                   for value in jax.tree_util.tree_leaves(grads))



def test_copy_gate_is_inert_when_disabled_and_exact_when_enabled():
    """The gate must not perturb any other parameter.

    ``statistics.py`` compares ``parameter_shapes`` across runs, so the gate has
    to be created inside ``initialize_params`` rather than injected afterwards.
    Creating it must not consume PRNG state, otherwise every other parameter
    would shift and a gated run could no longer be compared against an ungated
    one at the same seed.
    """
    config = _config()
    gated = dataclasses.replace(config, use_copy_gate=True)
    plain_params = initialize_params(config, 0)
    gated_params = initialize_params(gated, 0)

    assert set(gated_params) - set(plain_params) == {"copy_head/w", "copy_head/b"}
    assert not set(plain_params) - set(gated_params)
    tokens, subspaces = config.num_latent_tokens, config.num_subspaces
    assert gated_params["copy_head/w"].shape == (
        tokens, subspaces, config.code_embedding_dim
    )
    assert gated_params["copy_head/b"].shape == (tokens, subspaces)
    # Zero init => copy_logit 0 => 50/50 mixture at step 0.
    assert not np.any(np.asarray(gated_params["copy_head/w"]))
    assert not np.any(np.asarray(gated_params["copy_head/b"]))
    for name, value in plain_params.items():
        assert np.array_equal(np.asarray(value), np.asarray(gated_params[name])), name


def test_copy_gate_matches_the_head_the_bakeoff_trained():
    """Formal init must equal the dev harness's post-hoc injection, bit for bit.

    Every architecture number in worklog sections 12 and 24 was measured through
    ``dev/d_head_bakeoff.py::apply_head``. If ``train.py`` started from a
    different function those numbers would not carry over to the formal run.
    """
    from experiments.world_model.dev.d_head_bakeoff import apply_head

    config = _config()
    formal = initialize_params(dataclasses.replace(config, use_copy_gate=True), 0)
    injected = apply_head(initialize_params(config, 0), "tied_copy")
    assert set(formal) == set(injected)
    for name, value in formal.items():
        assert np.array_equal(np.asarray(value), np.asarray(injected[name])), name


def test_copy_gate_returns_normalised_logits_with_finite_gradients():
    """Mixing in the log domain must leave the result already normalised.

    ``codelength_bits`` applies ``log_softmax``; it is only idempotent here if
    the gate emits a normalised distribution. The identity branch uses -1e30
    rather than -inf so that ``logaddexp`` keeps a finite gradient path.
    """
    config = dataclasses.replace(_config(), use_copy_gate=True)
    params = initialize_params(config, 0)
    codes, valid, present, actions = _inputs(config)
    tasks = np.zeros(codes.shape[0], np.int32)

    _mask_logits, code_logits = predict(
        params, codes, valid, actions, tasks, "full", config,
        history_present=present, train=False,
    )
    total = jax.scipy.special.logsumexp(code_logits, axis=-1)
    assert float(jnp.abs(total).max()) < 1e-6
    assert bool(jnp.isfinite(code_logits).all())

    def loss(values):
        mask_logits, logits = predict(
            values, codes, valid, actions, tasks, "full", config,
            history_present=present, train=False,
        )
        # Targets equal the source codes, so the identity branch carries real
        # mass and the -1e30 path is actually exercised.
        targets = np.zeros(
            (codes.shape[0], config.num_latent_tokens, config.num_subspaces),
            np.int32,
        )
        rates = codelength_bits(mask_logits, logits, valid[:, -1], targets)
        return jnp.mean(rates["total_bits"])

    grads = jax.grad(loss)(params)
    assert all(np.isfinite(np.asarray(value)).all()
               for value in jax.tree_util.tree_leaves(grads))
    assert float(jnp.abs(grads["copy_head/w"]).max()) > 0.0
