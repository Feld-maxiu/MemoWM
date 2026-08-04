"""Stage 3 (v0.4 latent line): latent random-access, rendering, retrieval + query.

Verifies the progressive latent session round-trips (final-hash check passes),
that the *existing* QueryEngine drives the latent memory unchanged, and that the
grounding contract still holds (an answer may only cite fields actually shown).
"""
from __future__ import annotations

import numpy as np
import pytest

from residualmem.codec import LatentMemoryFile, encode_latent_memory
from residualmem.codec.send_mask import ExactMaskPolicy, RateDistortionMaskPolicy
from residualmem.encoders.structured import StructuredStateEncoder
from residualmem.latent.types import DomainId, LatentSpec
from residualmem.query import (
    Answer,
    EvidenceRef,
    Expand,
    QueryEngine,
    QueryPlan,
    ScriptedReaderPolicy,
)
from residualmem.retrieval import (
    MemoryIndex,
    build_latent_memory_index,
    extract_segment_events,
)
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
        if rng.random() < 0.05:
            ach[6] = 1
    states.append(schema.make_state([x, y] + inv + [bool(a) for a in ach]))
    return Trajectory(tuple(states), tuple(actions))


def _build(tmp_path, mask_policy=None):
    schema = crafter_schema()
    domain = DomainId("crafter")
    encoder = StructuredStateEncoder(schema, LatentSpec(8, 16, 1), domain)
    config = R.RSSMConfig(num_groups=8, num_categories=16, x_dim=40,
                          hidden_size=48, embed_size=48, post_hidden=48, dec_hidden=48)
    params = R.initialize_params(config, seed=2)
    latent_schema = LatentSpec(8, 16, 1).make_schema()
    traj = _synth(130, seed=1)
    mem = tmp_path / "m.rs4"
    encode_latent_memory(mem, traj, params, config, encoder, domain, latent_schema,
                         mask_policy=mask_policy or ExactMaskPolicy(), segment_length=64)
    memory = LatentMemoryFile(mem, params, config, domain, latent_schema, encoder)
    return schema, memory, mem, latent_schema


@pytest.mark.parametrize("policy", [ExactMaskPolicy(), RateDistortionMaskPolicy(lam=2.0)])
def test_random_access_sessions_pass_final_hash(tmp_path, policy):
    _, memory, _, _ = _build(tmp_path, policy)
    fields = memory.canonical_schema.names
    for segment_id in range(memory.segment_count):
        session = memory.open_session(segment_id)
        session.view(fields)
        while session.current_local_step < session.num_steps - 1:
            session.expand(fields, session.num_steps)  # raises on hash mismatch
        assert session.current_local_step == session.num_steps - 1


def test_query_engine_runs_on_latent_memory(tmp_path):
    schema, memory, mem, latent_schema = _build(tmp_path)
    rendered = memory.render_trajectory()
    assert len(rendered.states) == len(rendered.actions) + 1
    events = extract_segment_events(rendered, schema, segment_length=64)
    idx = tmp_path / "idx.rmi"
    build_latent_memory_index(idx, mem, events, schema, latent_schema,
                              np.zeros((len(events), 4), np.float32), "zeros-test")
    index = MemoryIndex(idx, schema, mem)
    plan = QueryPlan(query="q", changed_fields=("pos_x",), initial_fields=("pos_x", "pos_y"))
    reader = ScriptedReaderPolicy(plan, [
        Expand(fields=("pos_x", "pos_y"), max_steps=4),
        Answer(text="moved", evidence=(EvidenceRef(0, 0, 0, ("pos_x", "pos_y")),)),
    ])
    result = QueryEngine(memory, index, schema).run("q", reader)
    index.close()
    assert result.answer == "moved"
    assert result.trace.memory_bytes == mem.stat().st_size


def test_answer_cannot_cite_unshown_field(tmp_path):
    schema, memory, mem, latent_schema = _build(tmp_path)
    events = extract_segment_events(memory.render_trajectory(), schema, segment_length=64)
    idx = tmp_path / "idx.rmi"
    build_latent_memory_index(idx, mem, events, schema, latent_schema,
                              np.zeros((len(events), 4), np.float32), "zeros-test")
    index = MemoryIndex(idx, schema, mem)
    plan = QueryPlan(query="q", changed_fields=("pos_x",), initial_fields=("pos_x",))
    reader = ScriptedReaderPolicy(plan, [
        Answer(text="x", evidence=(EvidenceRef(0, 0, 0, ("inventory/coal",)),)),
    ])
    with pytest.raises(ValueError):
        QueryEngine(memory, index, schema).run("q", reader)
    index.close()


def test_latent_memory_rejects_wrong_encoder(tmp_path):
    schema, memory, mem, latent_schema = _build(tmp_path)
    domain = DomainId("crafter")
    config = R.RSSMConfig(num_groups=8, num_categories=16, x_dim=40,
                          hidden_size=48, embed_size=48, post_hidden=48, dec_hidden=48)
    params = R.initialize_params(config, seed=2)
    # A different domain -> different encoder hash -> must be rejected.
    other_encoder = StructuredStateEncoder(schema, LatentSpec(8, 16, 1), DomainId("web"))
    from residualmem.codec import DecodeError
    with pytest.raises(DecodeError):
        LatentMemoryFile(mem, params, config, domain, latent_schema, other_encoder)
