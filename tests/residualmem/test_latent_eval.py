"""Stage 4 (v0.4 latent line): rate-distortion operating points + Memory Rate."""
from __future__ import annotations

import numpy as np

from residualmem.encoders.structured import StructuredStateEncoder
from residualmem.evaluation.runner import run_latent_rate_distortion
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
            inv[3] = min(9, inv[3] + 1)
    states.append(schema.make_state([x, y] + inv + [bool(a) for a in ach]))
    return Trajectory(tuple(states), tuple(actions))


def test_rate_distortion_report(tmp_path):
    schema = crafter_schema()
    domain = DomainId("crafter")
    encoder = StructuredStateEncoder(schema, LatentSpec(8, 16, 1), domain)
    config = R.RSSMConfig(num_groups=8, num_categories=16, x_dim=40,
                          hidden_size=48, embed_size=48, post_hidden=48, dec_hidden=48)
    params = R.initialize_params(config, seed=2)
    latent_schema = LatentSpec(8, 16, 1).make_schema()
    traj = _synth(96, seed=1)

    rows = run_latent_rate_distortion(
        traj, params, config, encoder, domain, latent_schema, tmp_path,
        lambdas=(1.0, 4.0), segment_length=32)

    names = {row["name"] for row in rows}
    assert {"wm_only", "exact"} <= names
    for row in rows:
        for key in ("bytes", "bytes_per_step", "memory_rate", "state_distortion",
                    "exact_field_accuracy", "residual_bytes"):
            assert key in row
    by_name = {row["name"]: row for row in rows}
    # storing nothing must never cost more residual bytes than storing every mismatch
    assert by_name["wm_only"]["residual_bytes"] <= by_name["exact"]["residual_bytes"]
    assert (tmp_path / "rate_distortion.json").exists()
    assert (tmp_path / "rate_distortion.csv").exists()
