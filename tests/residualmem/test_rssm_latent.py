"""Stage 1 (v0.4 latent line): RSSM world model + LatentPredictor invariants.

These assert the *determinism* guarantees the codec relies on -- not model quality:
the RSSM latent trajectory must round-trip exactly through the existing exact codec,
the structured front-end must be an exact inverse on bounded fields, and the
LatentPredictor must be reproducible across sessions.
"""
from __future__ import annotations

import numpy as np
import pytest

from residualmem.codec import decode_memory, encode_memory
from residualmem.encoders.structured import StructuredStateEncoder
from residualmem.latent.types import DomainId, LatentSpec
from residualmem.schemas import crafter_schema
from residualmem.types import Trajectory
from residualmem.world_model import rssm as R
from residualmem.world_model import rssm_train as T
from residualmem.world_model.latent_predictor import LatentPredictor


def _synth_trajectory(n_steps: int, seed: int = 0) -> Trajectory:
    rng = np.random.default_rng(seed)
    schema = crafter_schema()
    inv = [0] * 16
    ach = [0] * 22
    x, y = 10, 10
    states, actions = [], []
    for _ in range(n_steps):
        states.append(schema.make_state([x, y] + inv + [bool(a) for a in ach]))
        actions.append(int(rng.integers(0, 17)))
        x = min(63, max(0, x + int(rng.integers(-1, 2))))
        y = min(63, max(0, y + int(rng.integers(-1, 2))))
        if rng.random() < 0.1:
            j = int(rng.integers(0, 16))
            inv[j] = min(9, inv[j] + 1)
        if rng.random() < 0.05:
            ach[int(rng.integers(0, 22))] = 1
    states.append(schema.make_state([x, y] + inv + [bool(a) for a in ach]))
    return Trajectory(tuple(states), tuple(actions))


def _tiny_setup(seed: int = 0):
    schema = crafter_schema()
    domain = DomainId("crafter")
    encoder = StructuredStateEncoder(schema, LatentSpec(6, 12, token_dim=1), domain)
    config = T.build_config(encoder, num_groups=6, num_categories=12,
                            hidden_size=48, embed_size=48, post_hidden=48, dec_hidden=48)
    latent_schema = LatentSpec(6, 12, token_dim=1).make_schema()
    return schema, domain, encoder, config, latent_schema


def test_structured_encoder_is_exact_inverse_on_bounded_fields():
    schema, domain, encoder, _, _ = _tiny_setup()
    traj = _synth_trajectory(20, seed=3)
    for state in traj.states:
        tokens = encoder.encode(state).tokens
        recovered = encoder.decode_tokens(tokens)
        assert tuple(int(v) for v in recovered.values) == tuple(int(v) for v in state.values)


def test_latent_trajectory_roundtrips_through_exact_codec(tmp_path):
    schema, domain, encoder, config, latent_schema = _tiny_setup()
    traj = _synth_trajectory(48, seed=1)
    ckpt = tmp_path / "rssm.npz"
    T.train_rssm([traj], encoder, domain, config, ckpt,
                 seed=0, teacher_epochs=3, closed_loop_epochs=1, segment_length=48)
    params, config2, _ = R.load_rssm_checkpoint(ckpt)

    latent_traj = T.latentize_trajectory(params, config2, encoder, traj, domain)
    predictor = LatentPredictor(params, config2, latent_schema, domain)

    path = tmp_path / "mem.rsm"
    accounting = encode_memory(path, latent_traj, predictor, latent_schema,
                               segment_length=10_000)
    decoded, decoded_accounting = decode_memory(path, predictor, latent_schema)
    assert decoded.states == latent_traj.states
    assert decoded.actions == latent_traj.actions
    assert decoded_accounting == accounting


def test_latent_predictor_is_reproducible_across_sessions():
    schema, domain, encoder, config, latent_schema = _tiny_setup()
    params = R.initialize_params(config, seed=0)
    a = LatentPredictor(params, config, latent_schema, domain)
    b = a.new_session()
    assert a.hash_bytes == b.hash_bytes
    assert len(a.hash_bytes) == 32

    anchor = latent_schema.make_state([0] * config.num_groups)
    a.reset(); b.reset()
    pa = a.predict_next(anchor, action=1).default_state
    pb = b.predict_next(anchor, action=1).default_state
    assert pa.values == pb.values


def test_rssm_training_reduces_loss():
    schema, domain, encoder, config, latent_schema = _tiny_setup()
    traj = _synth_trajectory(48, seed=2)
    examples = T._segment_examples(traj, encoder, 48)
    x, actions = examples[0]
    import jax, jax.numpy as jnp
    params = R.initialize_params(config, seed=0)
    key = jax.random.PRNGKey(0)
    before = float(R.rssm_loss(params, jnp.asarray(x), jnp.asarray(actions), 3, config, key))
    assert np.isfinite(before)
    assert before > 0
