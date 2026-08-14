from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from experiments.world_model.config import TrainingConfig
from experiments.world_model.model import ModelConfig, initialize_params
from experiments.world_model.train import (
    DeterministicSampler,
    _save_checkpoint,
    build_optimizer,
    load_checkpoint,
    make_update,
)


def _config():
    return ModelConfig(
        num_tasks=2, max_history=3, num_latent_tokens=4,
        num_subspaces=2, num_categories=8, num_observation_slots=4,
        code_embedding_dim=4, d_model=8, num_layers=1, num_heads=2,
        mlp_dim=16, dropout=0.0, action_embedding_dim=4,
        byte_embedding_dim=3, payload_hidden_dim=5, max_payload_bytes=4,
    )


def _batch():
    return {
        "history_codes": jnp.zeros((2, 3, 4, 2), jnp.uint8),
        "history_valid": jnp.ones((2, 3, 4), jnp.bool_),
        "history_present": jnp.ones((2, 3), jnp.bool_),
        "action_types": jnp.zeros((2, 3), jnp.uint8),
        "action_tags": jnp.zeros((2, 3), jnp.uint8),
        "action_refs": jnp.zeros((2, 3), jnp.uint8),
        "action_payloads": jnp.zeros((2, 3, 4), jnp.uint8),
        "action_lengths": jnp.zeros((2, 3), jnp.uint8),
        "task_ids": jnp.zeros((2,), jnp.uint8),
        "target_codes": jnp.zeros((2, 4, 2), jnp.uint8),
        "target_valid": jnp.ones((2, 4), jnp.bool_),
    }


def test_update_and_full_checkpoint_resume_state(tmp_path):
    model = _config()
    training = TrainingConfig(
        batch_size=2, learning_rate=3e-4, warmup_steps=1,
        max_steps=4, min_steps=1, eval_every=1, patience_steps=1,
    )
    params = initialize_params(model, 0)
    optimizer, _ = build_optimizer(params, training)
    state = optimizer.init(params)
    update = make_update(optimizer, "full", model, overfit=False)
    params, state, loss, metrics = update(
        params, state, _batch(), jax.random.PRNGKey(1)
    )
    assert np.isfinite(float(loss))
    assert np.isfinite(float(metrics["grad_norm_before_clip"]))
    sampler = DeterministicSampler(np.arange(5), 2, 0)
    sampler.next()
    path = tmp_path / "checkpoint.pkl"
    metadata = {
        "variant": "full", "seed": 0,
        "cache_manifest_sha256": "cache", "config_sha256": "config",
    }
    _save_checkpoint(
        path, params=params, opt_state=state, key=jax.random.PRNGKey(2),
        sampler=sampler, step=1, best_metric=float(loss), best_step=1,
        evals_without_improvement=0, metadata=metadata,
    )
    restored = load_checkpoint(path, metadata)
    assert restored["step"] == 1
    resumed = DeterministicSampler.from_state(restored["sampler"])
    assert np.array_equal(sampler.next(), resumed.next())
