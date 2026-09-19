from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.state_tokenizer import run_longmemeval_local as runner
from residualmem.benchmarks.longmemeval_compact import compact_axtree_text
from residualmem.benchmarks.longmemeval_index import RetrievedObservation
from residualmem.benchmarks.longmemeval_text_index import (
    PROTOCOL,
    TextRetrievalHit,
    format_raw_state_context,
)


def _observation(trajectory, index, tree=None):
    record = {
        "trajectory_id": trajectory,
        "state_index": index,
        "step_idx": index,
        "url": "https://example.test/form",
        "incoming_action_text": f"click('{index}')",
        "axtree": tree if tree is not None else (
            "RootWebArea 'Form'\n"
            f"  [{index}] textbox 'Value' value='exact-{index}'"
        ),
    }
    return RetrievedObservation(
        trajectory, index, 0.0,
        np.full((2, 3), index + 1, np.float32), np.ones(2, np.bool_),
        np.asarray([1.0, 0.0], np.float32), "original stored anchor", record,
    )


def _index(observations):
    lookup = {(row.trajectory_id, row.record_index): row for row in observations}
    return SimpleNamespace(observation=lambda trajectory, index: lookup.get((trajectory, index)))


def _hit(trajectory, center, start, end, *, goal="task", actions="1. click('0')\n2. click('1')"):
    text = format_raw_state_context(
        trajectory_id=trajectory, goal=goal, center_index=center,
        slice_state_indexes=list(range(start, end)),
        full_action_sequence=actions, local_action_sequence="1. click('0')",
        states=[{"state_index": center, "axtree": "legacy window text"}],
    )
    return TextRetrievalHit(trajectory, center, 0.9, text, goal, start, end)


def _text(segments):
    return "".join(segment.text for segment in segments if segment.text is not None)


def test_snapshots_deduplicate_identities_and_bind_each_latent():
    observations = [_observation("a", i) for i in range(4)]
    hits = [_hit("a", 1, 0, 3), _hit("a", 2, 1, 4)]
    segments, context, diagnostics = runner.text_state_snapshot_memory(hits, _index(observations))
    assert [(row.trajectory_id, row.record_index) for row in context] == [
        ("a", 0), ("a", 1), ("a", 2), ("a", 3),
    ]
    latents = [segment.latent for segment in segments if segment.latent is not None]
    assert len(latents) == 4
    for row, original, (xbar, valid) in zip(context, observations, latents):
        np.testing.assert_array_equal(xbar, original.xbar)
        np.testing.assert_array_equal(valid, original.valid)
        assert row.anchor_text == compact_axtree_text(original.record)
        assert original.anchor_text == "original stored anchor"
    text = _text(segments)
    assert text.count('<observation trajectory="a"') == 4
    assert text.count("Full action sequence") == 1
    assert "1. click('0')\n2. click('1')" in text
    assert "Retrieved in window ranks: 1, 2" in text
    assert "legacy window text" not in text
    assert "independent sparse snapshots, not deltas" in text
    assert diagnostics == {
        "layout_protocol": runner.STATE_SNAPSHOT_PROTOCOL,
        "retrieved_windows": 2,
        "window_state_references": 6,
        "unique_states": 4,
        "latent_blocks": 4,
        "latent_tokens": 8,
    }


def test_snapshot_content_does_not_depend_on_window_or_hit_order():
    index = _index([_observation("a", i) for i in range(4)])
    hits = [_hit("a", 1, 0, 3), _hit("a", 2, 1, 4)]
    _, forward, _ = runner.text_state_snapshot_memory(hits, index)
    _, reverse, _ = runner.text_state_snapshot_memory(list(reversed(hits)), index)
    assert [(row.record_index, row.anchor_text) for row in forward] == [
        (row.record_index, row.anchor_text) for row in reverse
    ]
    for hit in hits:
        _, isolated, _ = runner.text_state_snapshot_memory([hit], index)
        for row in isolated:
            assert row.anchor_text == forward[row.record_index].anchor_text


def test_removed_and_reappearing_controls_are_not_elided():
    button = "RootWebArea 'Form'\n  [10] button 'Continue'"
    observations = [
        _observation("a", 0, button),
        _observation("a", 1, "RootWebArea 'Form'"),
        _observation("a", 2, button),
    ]
    segments, context, _ = runner.text_state_snapshot_memory([_hit("a", 1, 0, 3)], _index(observations))
    assert "Continue" in context[0].anchor_text
    assert "Continue" not in context[1].anchor_text
    assert "Continue" in context[2].anchor_text
    assert context[0].anchor_text == context[2].anchor_text
    assert "no new lines relative to" not in _text(segments)


def test_trajectory_order_and_duplicate_hits_preserve_state_identity():
    observations = [_observation(t, i) for t in ("a", "b") for i in range(3)]
    hits = [_hit("b", 1, 0, 2), _hit("a", 2, 1, 3), _hit("b", 2, 1, 3)]
    segments, context, _ = runner.text_state_snapshot_memory(hits + [hits[0]], _index(observations))
    assert [(row.trajectory_id, row.record_index) for row in context] == [
        ("b", 0), ("b", 1), ("b", 2), ("a", 1), ("a", 2),
    ]
    assert _text(segments).count("Full action sequence") == 2
    assert "Retrieved in window ranks: 1, 3, 4" in _text(segments)


def test_empty_retrieval_remains_empty_memory():
    segments, context, diagnostics = runner.text_state_snapshot_memory([], _index([]))
    assert segments == context == []
    assert diagnostics["latent_tokens"] == 0


@pytest.mark.parametrize("center,start,end", [(0, -1, 1), (1, 1, 1), (2, 0, 2)])
def test_invalid_window_fails(center, start, end):
    with pytest.raises(ValueError, match="invalid state window"):
        runner.text_state_snapshot_memory([_hit("a", center, start, end)], _index([]))


def test_missing_neighbor_latent_is_not_silently_skipped():
    with pytest.raises(ValueError, match="missing snapshot latent"):
        runner.text_state_snapshot_memory([_hit("a", 0, 0, 2)], _index([_observation("a", 0)]))


def test_wrong_observation_identity_fails():
    index = SimpleNamespace(observation=lambda *args: _observation("other", 0))
    with pytest.raises(ValueError, match="snapshot identity mismatch"):
        runner.text_state_snapshot_memory([_hit("a", 0, 0, 1)], index)


@pytest.mark.parametrize("change", [
    {"valid": np.zeros(2, np.bool_)},
    {"valid": np.ones(1, np.bool_)},
    {"record": {"trajectory_id": "other"}},
])
def test_invalid_snapshot_observation_fails(change):
    row = replace(_observation("a", 0), **change)
    with pytest.raises(ValueError, match="invalid snapshot observation"):
        runner.text_state_snapshot_memory([_hit("a", 0, 0, 1)], _index([row]))


def test_unrecognized_window_context_is_rejected():
    hit = replace(_hit("a", 0, 0, 1), context_text="no trajectory metadata")
    with pytest.raises(ValueError, match="invalid trajectory context"):
        runner.text_state_snapshot_memory([hit], _index([]))


@pytest.mark.parametrize("change", [{"goal": "another task"}, {"actions": "1. different action"}])
def test_conflicting_trajectory_metadata_is_rejected(change):
    hits = [_hit("a", 0, 0, 1), _hit("a", 1, 1, 2, **change)]
    with pytest.raises(ValueError, match="conflicting trajectory context"):
        runner.text_state_snapshot_memory(hits, _index([_observation("a", i) for i in range(2)]))


def test_snapshot_blocks_pass_through_the_real_reader_encoder():
    import torch
    from residualmem.latent.instruct_bridge import InputSoftTokenConnector, Qwen35LatentReader

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.ones(1, 2, dtype=torch.long),
                    "attention_mask": torch.ones(1, 2, dtype=torch.long)}

    embedding = torch.nn.Embedding(4, 4)
    model = SimpleNamespace(get_input_embeddings=lambda: embedding)
    reader = Qwen35LatentReader(
        model, SimpleNamespace(tokenizer=Tokenizer()),
        InputSoftTokenConnector(model_dim=4, slots=2), mode="input",
    )
    observations = [
        replace(_observation("a", i), xbar=np.full((2, 512), i + 1, np.float32))
        for i in range(3)
    ]
    segments, _, _ = runner.text_state_snapshot_memory(
        [_hit("a", 1, 0, 3)], _index(observations),
    )
    pieces, masks, spans = reader._encode_segments(segments, torch.device("cpu"))
    assert len(spans) == 3
    assert all(end - start == 2 for start, end in spans)
    assert sum(piece.shape[1] for piece in pieces) == sum(mask.shape[1] for mask in masks)
    assert all(torch.isfinite(piece).all() for piece in pieces)


def _runner_case(tmp_path, monkeypatch, *, metadata_overrides=None):
    data_root = tmp_path / "data"
    (data_root / "haystacks").mkdir(parents=True)
    (data_root / "questions.jsonl").write_text(json.dumps({
        "id": "q1", "domain": "web", "question": "What is shown?",
        "question_type": "static-environment", "answer": "ok",
        "eval_function": "test", "image": None,
    }) + "\n")
    (data_root / "haystacks/lme_v2_small.json").write_text(json.dumps({"q1": ["a"]}))
    cache = tmp_path / "cache"
    (cache / "trajectories").mkdir(parents=True)
    observations = [_observation("a", i) for i in range(2)]
    np.savez(
        cache / "trajectories/a.npz",
        xbar=np.stack([row.xbar for row in observations]),
        valid=np.stack([row.valid for row in observations]),
        key=np.stack([row.key for row in observations]),
        metadata=np.asarray(json.dumps({
            "trajectory_id": "a", "records": [row.record for row in observations],
        })),
    )
    text_index = tmp_path / "text-index.npz"
    metadata = {"protocol": PROTOCOL, "cache": str(cache), "compression": "compact_exact"}
    metadata.update(metadata_overrides or {})
    hit = _hit("a", 0, 0, 2)
    np.savez(
        text_index, embedding=np.asarray([[1.0, 0.0]], np.float32),
        trajectory_id=np.asarray(["a"]), center_index=np.asarray([0]),
        slice_start=np.asarray([0]), slice_end=np.asarray([2]),
        goal=np.asarray([hit.goal]), context_text=np.asarray([hit.context_text]),
        metadata=np.asarray(json.dumps(metadata)),
    )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    calls = []

    class Encoder:
        def __init__(self, *args, **kwargs):
            pass

        def encode_query(self, question):
            return np.asarray([1.0, 0.0], np.float32)

    class Reader:
        def __init__(self, *args, **kwargs):
            self.system_prompt = kwargs["system_prompt"]

        def answer(self, question, segments, **kwargs):
            calls.append((question, segments, kwargs, self.system_prompt))
            self.last_generation = {"prompt_tokens": len(segments)}
            return r"\boxed{ok}"

    monkeypatch.setattr(runner, "TextEmbeddingEncoder", Encoder)
    monkeypatch.setattr(runner, "LongMemEvalQFormerRuntime", lambda **kwargs: SimpleNamespace(
        model=None, processor=None, reader=SimpleNamespace(connector=None),
    ))
    monkeypatch.setattr(runner, "LongMemEvalReader", Reader)
    monkeypatch.setattr(runner, "_load_eval_function", lambda: (
        lambda spec, prediction, answer, **kwargs: prediction == answer,
        lambda spec: "test", lambda value: value[7:-1], bool,
    ))
    args = [
        "--data-root", str(data_root), "--cache", str(cache),
        "--model", str(model_dir), "--checkpoint", str(tmp_path / "checkpoint.pt"),
        "--embedding-model", str(model_dir), "--text-embedding-model", str(model_dir),
        "--text-index", str(text_index), "--retrieval-mode", "text",
        "--latent-payload", "--top-k", "1", "--device", "cpu",
    ]
    return args, calls


def _run(monkeypatch, args, output, extra=()):
    monkeypatch.setattr(sys, "argv", ["run_longmemeval_local", *args, "--output", str(output), *extra])
    runner.main()


def test_runner_records_actual_snapshot_context_and_keeps_retrieval(tmp_path, monkeypatch):
    args, calls = _runner_case(tmp_path, monkeypatch)
    legacy_path = tmp_path / "legacy.jsonl"
    snapshot_path = tmp_path / "snapshots.jsonl"
    _run(monkeypatch, args, legacy_path)
    _run(monkeypatch, args, snapshot_path, ["--text-memory-layout", "state-snapshots"])
    legacy = json.loads(legacy_path.read_text())
    snapshot = json.loads(snapshot_path.read_text())
    assert legacy["hits"] == snapshot["hits"]
    assert legacy["generation"] == snapshot["generation"]
    assert legacy["context_ids"] == [["a", 0]]
    assert snapshot["context_ids"] == [["a", 0], ["a", 1]]
    assert snapshot["context_observations"] == 2
    assert snapshot["hit_count"] == 1
    assert snapshot["memory_diagnostics"]["latent_blocks"] == 2
    assert "memory_diagnostics" not in legacy
    assert legacy["score"] is snapshot["score"] is True
    assert len(calls) == 2 and calls[0][3] == calls[1][3]
    assert calls[0][2] == calls[1][2]
    old_config = json.loads(legacy_path.with_suffix(".config.json").read_text())
    new_config = json.loads(snapshot_path.with_suffix(".config.json").read_text())
    assert "text_memory_layout" not in old_config["memory"]
    assert new_config["memory"]["text_memory_layout"] == "state-snapshots"
    assert new_config["memory"]["compact_protocol"] == runner.COMPACT_PROTOCOL
    assert old_config["reader_system_prompt_sha256"] == new_config["reader_system_prompt_sha256"]


def test_explicit_legacy_layout_preserves_default_config_and_payload(tmp_path, monkeypatch):
    args, calls = _runner_case(tmp_path, monkeypatch)
    first, second = tmp_path / "default.jsonl", tmp_path / "explicit.jsonl"
    _run(monkeypatch, args, first)
    _run(monkeypatch, args, second, ["--text-memory-layout", "legacy-windows"])
    assert json.loads(first.with_suffix(".config.json").read_text()) == json.loads(
        second.with_suffix(".config.json").read_text()
    )
    assert _text(calls[0][1]) == _text(calls[1][1])


@pytest.mark.parametrize("initial,next_layout", [
    ("legacy-windows", "state-snapshots"), ("state-snapshots", "legacy-windows"),
])
def test_runner_cannot_resume_into_another_layout(tmp_path, monkeypatch, initial, next_layout):
    args, calls = _runner_case(tmp_path, monkeypatch)
    output = tmp_path / "results.jsonl"
    _run(monkeypatch, args, output, ["--text-memory-layout", initial])
    original = output.read_bytes()
    with pytest.raises(ValueError, match="resume configuration changed"):
        _run(monkeypatch, args, output, ["--resume", "--text-memory-layout", next_layout])
    assert output.read_bytes() == original
    _run(monkeypatch, args, output, ["--resume", "--text-memory-layout", initial])
    assert len(calls) == 1


@pytest.mark.parametrize("extra", [
    ["--retrieval-mode", "hybrid"], ["--no-latent-payload"],
    ["--no-anchor"], ["--no-structured-memory"],
])
def test_snapshot_mode_rejects_incompatible_settings(tmp_path, monkeypatch, extra):
    args, calls = _runner_case(tmp_path, monkeypatch)
    output = tmp_path / "results.jsonl"
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, args, output, ["--text-memory-layout", "state-snapshots", *extra])
    assert error.value.code == 2
    assert not output.with_suffix(".config.json").exists()
    assert calls == []


@pytest.mark.parametrize("metadata", [{"compression": "raw"}, {"cache": "/different/cache"}])
def test_snapshot_mode_rejects_wrong_index_source(tmp_path, monkeypatch, metadata):
    args, calls = _runner_case(tmp_path, monkeypatch, metadata_overrides=metadata)
    output = tmp_path / "results.jsonl"
    with pytest.raises(SystemExit) as error:
        _run(monkeypatch, args, output, ["--text-memory-layout", "state-snapshots"])
    assert error.value.code == 2
    assert not output.with_suffix(".config.json").exists()
    assert calls == []


@pytest.mark.parametrize("layout", ["legacy-windows", "state-snapshots"])
def test_launcher_passes_layout_without_changing_old_default(tmp_path, layout):
    cache, model = tmp_path / "cache", tmp_path / "model"
    cache.mkdir()
    model.mkdir()
    index = tmp_path / "index.npz"
    index.touch()
    capture = tmp_path / "calls.jsonl"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['LME_TEST_CALLS'], 'a') as handle:\n"
        "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ, "LME_PYTHON": str(fake_python), "LME_CACHE": str(cache),
        "LME_TEXT_MODEL": str(model), "LME_TEXT_INDEX": str(index),
        "LME_OUTPUT": str(tmp_path / "output"), "LME_LOGS": str(tmp_path / "logs"),
        "LME_GPUS": "4,5", "LME_TEXT_MEMORY_LAYOUT": layout, "LME_TEST_CALLS": str(capture),
    }
    launcher = Path(runner.__file__).with_name("run_longmemeval_latent_text_rag_eval.sh")
    subprocess.run(["bash", str(launcher)], env=env, check=True, capture_output=True, text=True)
    calls = [json.loads(line) for line in capture.read_text().splitlines()]
    evaluations = [call for call in calls if "experiments.state_tokenizer.run_longmemeval_local" in call]
    merge = next(call for call in calls if "experiments.state_tokenizer.merge_longmemeval_eval" in call)
    assert len(evaluations) == 2
    for call in evaluations:
        if layout == "state-snapshots":
            assert call[call.index("--text-memory-layout") + 1] == layout
        else:
            assert "--text-memory-layout" not in call
    assert ("--anchor" if layout == "state-snapshots" else "--no-anchor") in merge
