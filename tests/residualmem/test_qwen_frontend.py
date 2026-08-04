"""Stage 5 (v0.4 latent line): Qwen multimodal front-end runs through the x_t seam.

Uses the CPU MockBackbone so the whole path -- observation -> backbone features ->
Perceiver tokenizer -> StateTokens -> RSSM -> latent codec -> RSMEMV04 -> decode --
runs without downloading a real model. Confirms the front-end is a drop-in x_t
source; model quality is out of scope.
"""
from __future__ import annotations

import numpy as np

from residualmem.codec import decode_latent_memory, encode_latent_memory
from residualmem.codec.send_mask import ExactMaskPolicy
from residualmem.encoders.qwen import MockBackbone, QwenObservationEncoder
from residualmem.latent.tokenizer import PerceiverConfig, initialize_perceiver
from residualmem.latent.types import DomainId, LatentSpec
from residualmem.types import Trajectory
from residualmem.world_model import rssm as R
from residualmem.world_model.rssm_train import build_config


def _make_encoder(seed: int = 0):
    backbone = MockBackbone(output_dim=32, seq_len=6)
    tconfig = PerceiverConfig(num_latents=4, latent_dim=16, input_dim=32,
                              num_heads=4, num_layers=2)
    tparams = initialize_perceiver(tconfig, seed=seed)
    domain = DomainId("web")
    encoder = QwenObservationEncoder(
        backbone, tparams, tconfig, LatentSpec(8, 16, 1), domain)
    return encoder, domain


def test_qwen_frontend_produces_fixed_state_tokens():
    encoder, _ = _make_encoder()
    tokens = encoder.encode("the user clicked submit")
    assert tokens.tokens.shape == (4, 16)
    assert tokens.flat().shape == (64,)
    # deterministic for a given observation
    again = encoder.encode("the user clicked submit")
    assert np.allclose(tokens.tokens, again.tokens)
    assert len(encoder.hash_bytes) == 32
    # x_dim wiring
    assert encoder.spec.token_dim * encoder.spec.num_tokens == 64


def test_multimodal_latent_memory_roundtrips(tmp_path):
    encoder, domain = _make_encoder()
    observations = tuple(f"observation step {i}" for i in range(40))
    actions = tuple(int(i % 17) for i in range(39))
    traj = Trajectory(observations, actions)  # states are raw observations

    config = build_config(encoder, num_groups=8, num_categories=16,
                          hidden_size=48, embed_size=48, post_hidden=48, dec_hidden=48)
    assert config.x_dim == 64
    params = R.initialize_params(config, seed=1)
    latent_schema = LatentSpec(8, 16, 1).make_schema()

    path = tmp_path / "mm.rs4"
    accounting, latent_traj = encode_latent_memory(
        path, traj, params, config, encoder, domain, latent_schema,
        mask_policy=ExactMaskPolicy(), segment_length=16)
    decoded, decoded_acc = decode_latent_memory(path, params, config, domain, latent_schema)
    assert decoded.states == latent_traj.states
    assert decoded_acc == accounting
    assert accounting.total_bytes == path.stat().st_size
