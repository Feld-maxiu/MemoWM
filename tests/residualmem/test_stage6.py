"""Stage 6 (v0.4 latent line): fixed-budget joint fine-tune + benchmark harness.

Confirms the scaffolding runs end-to-end on CPU (no external datasets, no real Qwen):
joint_finetune continues training at a target budget, the benchmark harness drives
the existing QueryEngine and scores grounded answers, and the Pareto helper reduces
rate-distortion rows to a frontier.
"""
from __future__ import annotations

import numpy as np

from residualmem.benchmarks import BenchmarkExample, run_benchmark
from residualmem.codec import LatentMemoryFile, encode_latent_memory
from residualmem.codec.send_mask import ExactMaskPolicy
from residualmem.encoders.structured import StructuredStateEncoder
from residualmem.evaluation.runner import pareto_operating_points
from residualmem.latent.types import DomainId, LatentSpec
from residualmem.query import Answer, EvidenceRef, Expand, QueryPlan
from residualmem.retrieval import (
    MemoryIndex,
    build_latent_memory_index,
    extract_segment_events,
)
from residualmem.schemas import crafter_schema
from residualmem.types import Trajectory
from residualmem.world_model import rssm as R
from residualmem.world_model.rssm_train import build_config, joint_finetune, train_rssm


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
    states.append(schema.make_state([x, y] + inv + [bool(a) for a in ach]))
    return Trajectory(tuple(states), tuple(actions))


def test_joint_finetune_runs_at_a_budget(tmp_path):
    schema = crafter_schema()
    domain = DomainId("crafter")
    encoder = StructuredStateEncoder(schema, LatentSpec(8, 16, 1), domain)
    config = build_config(encoder, num_groups=8, num_categories=16,
                          hidden_size=48, embed_size=48, post_hidden=48, dec_hidden=48)
    traj = _synth(64, seed=1)
    ckpt = tmp_path / "rssm.npz"
    train_rssm([traj], encoder, domain, config, ckpt, seed=0,
               teacher_epochs=2, closed_loop_epochs=0, segment_length=32)
    metrics = joint_finetune(ckpt, [traj], encoder, domain, tmp_path / "ft.npz",
                             kl_beta=3.0, epochs=2, segment_length=32)
    assert metrics["kl_beta"] == 3.0
    assert "prior_predictive_accuracy" in metrics
    assert (tmp_path / "ft.npz").exists()


def test_benchmark_harness_scores_grounded_answers(tmp_path):
    schema = crafter_schema()
    domain = DomainId("crafter")
    encoder = StructuredStateEncoder(schema, LatentSpec(8, 16, 1), domain)
    config = R.RSSMConfig(num_groups=8, num_categories=16, x_dim=40,
                          hidden_size=48, embed_size=48, post_hidden=48, dec_hidden=48)
    params = R.initialize_params(config, seed=2)
    latent_schema = LatentSpec(8, 16, 1).make_schema()
    traj = _synth(80, seed=3)
    mem = tmp_path / "m.rs4"
    encode_latent_memory(mem, traj, params, config, encoder, domain, latent_schema,
                         mask_policy=ExactMaskPolicy(), segment_length=64)
    memory = LatentMemoryFile(mem, params, config, domain, latent_schema, encoder)
    events = extract_segment_events(memory.render_trajectory(), schema, segment_length=64)
    idx = tmp_path / "idx.rmi"
    build_latent_memory_index(idx, mem, events, schema, latent_schema,
                              np.zeros((len(events), 4), np.float32), "zeros-test")
    index = MemoryIndex(idx, schema, mem)

    example = BenchmarkExample(
        query="where at start?",
        plan=QueryPlan(query="where at start?", changed_fields=("pos_x",),
                       initial_fields=("pos_x", "pos_y")),
        reader_actions=(
            Expand(fields=("pos_x", "pos_y"), max_steps=2),
            Answer(text="ok", evidence=(EvidenceRef(0, 0, 0, ("pos_x", "pos_y")),)),
        ),
        expected="ok",
    )
    report = run_benchmark(memory, index, schema, (example,))
    index.close()
    assert report["n"] == 1
    assert report["accuracy"] == 1.0
    assert report["results"][0]["memory_bytes"] == mem.stat().st_size


def test_pareto_operating_points():
    rows = [
        {"name": "a", "bytes_per_step": 1.0, "state_distortion": 0.5},
        {"name": "b", "bytes_per_step": 2.0, "state_distortion": 0.6},  # dominated
        {"name": "c", "bytes_per_step": 3.0, "state_distortion": 0.2},
        {"name": "d", "bytes_per_step": 4.0, "state_distortion": 0.3},  # dominated
    ]
    frontier = [r["name"] for r in pareto_operating_points(rows)]
    assert frontier == ["a", "c"]
