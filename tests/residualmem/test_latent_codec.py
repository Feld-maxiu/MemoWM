"""Stage 2 (v0.4 latent line): entropy coder + latent codec invariants.

Asserts the codec-correctness properties the report requires: exact encoder/decoder
sync (lossless and lossy), bit accounting == real file size, the rate knob, legacy
rejection, corruption detection, and model-hash binding. Model *quality* is not
under test -- untrained params are fine because these are determinism guarantees.
"""
from __future__ import annotations

import numpy as np
import pytest

from residualmem.codec import (
    DecodeError,
    LegacyFormatError,
    decode_latent_memory,
    encode_latent_memory,
)
from residualmem.codec.entropy import RansDecoder, RansEncoder, quantize_freqs
from residualmem.codec.segment import GLOBAL
from residualmem.codec.send_mask import ExactMaskPolicy, RateDistortionMaskPolicy
from residualmem.encoders.structured import StructuredStateEncoder
from residualmem.latent.types import DomainId, LatentSpec
from residualmem.schemas import crafter_schema
from residualmem.types import Trajectory
from residualmem.world_model import rssm as R


def _synth(n_steps: int, seed: int = 0) -> Trajectory:
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
            inv[int(rng.integers(0, 16))] = min(9, inv[int(rng.integers(0, 16))] + 1)
        if rng.random() < 0.05:
            ach[int(rng.integers(0, 22))] = 1
    states.append(schema.make_state([x, y] + inv + [bool(a) for a in ach]))
    return Trajectory(tuple(states), tuple(actions))


def _setup(seed: int = 2):
    schema = crafter_schema()
    domain = DomainId("crafter")
    encoder = StructuredStateEncoder(schema, LatentSpec(8, 16, 1), domain)
    config = R.RSSMConfig(num_groups=8, num_categories=16, x_dim=40,
                          hidden_size=48, embed_size=48, post_hidden=48, dec_hidden=48)
    params = R.initialize_params(config, seed=seed)
    latent_schema = LatentSpec(8, 16, 1).make_schema()
    return schema, domain, encoder, config, params, latent_schema


def test_entropy_roundtrip_varying_distributions():
    rng = np.random.default_rng(0)
    for _ in range(50):
        C = int(rng.integers(2, 32))
        n = int(rng.integers(0, 40))
        rows = [rng.normal(0, rng.uniform(0.2, 4.0), size=C) for _ in range(n)]
        syms = [int(rng.integers(0, C)) for _ in range(n)]
        enc = RansEncoder()
        for s, row in zip(syms, rows):
            probs = np.exp(row - row.max()); probs /= probs.sum()
            enc.encode(s, quantize_freqs(probs))
        data = enc.finish()
        dec = RansDecoder(data)
        out = []
        for row in rows:
            probs = np.exp(row - row.max()); probs /= probs.sum()
            out.append(dec.decode(quantize_freqs(probs)))
        assert out == syms


@pytest.mark.parametrize("policy_name", ["exact", "rd"])
def test_latent_memory_roundtrips_and_accounts(tmp_path, policy_name):
    schema, domain, encoder, config, params, latent_schema = _setup()
    traj = _synth(140, seed=1)
    policy = ExactMaskPolicy() if policy_name == "exact" else RateDistortionMaskPolicy(lam=3.0)
    path = tmp_path / f"{policy_name}.rs4"
    accounting, latent_traj = encode_latent_memory(
        path, traj, params, config, encoder, domain, latent_schema,
        mask_policy=policy, segment_length=64)
    decoded, decoded_accounting = decode_latent_memory(path, params, config, domain, latent_schema)
    # encoder-side reconstruction == decoder-side reconstruction (sync), even when lossy
    assert decoded.states == latent_traj.states
    assert decoded.actions == latent_traj.actions
    assert decoded_accounting == accounting
    assert accounting.total_bytes == path.stat().st_size


def test_rate_knob_exact_is_at_least_rd(tmp_path):
    schema, domain, encoder, config, params, latent_schema = _setup()
    traj = _synth(140, seed=5)
    exact_acc, _ = encode_latent_memory(
        tmp_path / "e.rs4", traj, params, config, encoder, domain, latent_schema,
        mask_policy=ExactMaskPolicy(), segment_length=64)
    rd_acc, _ = encode_latent_memory(
        tmp_path / "r.rs4", traj, params, config, encoder, domain, latent_schema,
        mask_policy=RateDistortionMaskPolicy(lam=0.0), segment_length=64)
    assert exact_acc.residuals >= rd_acc.residuals


def test_corruption_is_detected(tmp_path):
    schema, domain, encoder, config, params, latent_schema = _setup()
    traj = _synth(80, seed=3)
    path = tmp_path / "m.rs4"
    encode_latent_memory(path, traj, params, config, encoder, domain, latent_schema,
                         mask_policy=ExactMaskPolicy(), segment_length=64)
    raw = bytearray(path.read_bytes())
    raw[-30] ^= 0xFF
    bad = tmp_path / "bad.rs4"
    bad.write_bytes(raw)
    with pytest.raises(DecodeError):
        decode_latent_memory(bad, params, config, domain, latent_schema)


def test_model_hash_mismatch_is_rejected(tmp_path):
    schema, domain, encoder, config, params, latent_schema = _setup(seed=2)
    traj = _synth(80, seed=3)
    path = tmp_path / "m.rs4"
    encode_latent_memory(path, traj, params, config, encoder, domain, latent_schema,
                         mask_policy=ExactMaskPolicy(), segment_length=64)
    other = R.initialize_params(config, seed=999)
    with pytest.raises(DecodeError):
        decode_latent_memory(path, other, config, domain, latent_schema)


def test_legacy_stream_is_rejected(tmp_path):
    schema, domain, encoder, config, params, latent_schema = _setup()
    header = bytearray(GLOBAL.size)
    header[:8] = b"RSMEMV03"
    path = tmp_path / "legacy.rs4"
    path.write_bytes(bytes(header))
    with pytest.raises(LegacyFormatError):
        decode_latent_memory(path, params, config, domain, latent_schema)
