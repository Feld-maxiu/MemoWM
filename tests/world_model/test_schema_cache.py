from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.state_tokenizer.common import sha256_file
from experiments.world_model.cache import (
    CACHE_FILES,
    FrozenCache,
    _Record,
    build_transition_archive,
)
from experiments.world_model.schema import (
    MAX_PAYLOAD_BYTES,
    PROTOCOL,
    SPLIT_IDS,
    TAG_IDS,
    canonicalize_action,
)


def _raw(kind="CLICK", ref=2, tag="button", text=None, policy="random"):
    return {"type": kind, "ref": ref, "tag": tag, "text": text, "policy": policy}


def test_select_option_uses_the_executed_parent_ref():
    tree = (
        '<ref=20 parent=0 tag=select text="menu"/>\n'
        '<ref=21 parent=20 tag=option text="choice"/>'
    )
    action = canonicalize_action(
        _raw("SELECT_OPTION", 21, "option", "choice"), tree
    )
    assert action.ref == 20
    assert action.tag_id == TAG_IDS["select"]
    assert action.payload == b"choice"


def test_payload_is_measured_in_utf8_bytes_and_never_truncated():
    allowed = "a" * MAX_PAYLOAD_BYTES
    assert canonicalize_action(_raw("FILL", 2, "input_text", allowed)).payload_length == 40
    with pytest.raises(ValueError):
        canonicalize_action(_raw("FILL", 2, "input_text", "é" * 21))


def test_ref_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        canonicalize_action(_raw(ref=64))


def _episode(name, split, start, length):
    rows = []
    for step in range(length):
        rows.append(_Record(
            global_index=start + step,
            task="task-a",
            episode_id=name,
            step=step,
            split=split,
            action=_raw(ref=2) if step < length - 1 else _raw(ref=3),
            compact_axtree="",
        ))
    return rows


def _write_fake_cache(root: Path):
    records = _episode("train-ep", "train", 0, 3)
    records += _episode("validation-ep", "validation", 3, 2)
    records += _episode("test-ep", "test", 5, 2)
    info = build_transition_archive(records, root / CACHE_FILES["transitions"])
    total = len(records)
    np.save(root / CACHE_FILES["codes"], np.zeros((total, 64, 32), np.uint8))
    valid = np.ones((total, 64), np.bool_)
    valid[:, 40:] = False
    np.save(root / CACHE_FILES["valid"], valid)
    np.save(root / CACHE_FILES["global_indices"], np.arange(total, dtype=np.int64))
    manifest = {
        "protocol": PROTOCOL,
        "format_version": 1,
        "layout": [32, 16, 16, 0],
        "state_counts": {"train": 3, "validation": 2, "test": 2},
        **info,
        "artifact_sha256": {},
        "test_locked": True,
    }
    (root / CACHE_FILES["manifest"]).write_text(json.dumps(manifest), encoding="utf-8")
    return records


def test_transition_join_uses_successor_and_history_is_left_padded(tmp_path):
    records = _write_fake_cache(tmp_path)
    cache = FrozenCache(tmp_path)
    assert cache.manifest["transition_counts"] == {
        "train": 2, "validation": 1, "test": 1
    }
    # The terminal action exists in every episode but creates no transition.
    assert cache.manifest["transitions"] == 4
    train = cache.indices_for_split("train")
    first = cache.transitions["history_indices"][train[0]]
    second = cache.transitions["history_indices"][train[1]]
    assert first.tolist() == [-1, -1, -1, -1, -1, -1, 0]
    assert second.tolist() == [-1, -1, -1, -1, -1, 0, 1]
    batch = cache.batch(train[:1])
    assert not batch["history_present"][0, :-1].any()
    assert batch["target_codes"].shape == (1, 64, 32)


def test_episode_cross_split_is_a_hard_error(tmp_path):
    rows = _episode("bad", "train", 0, 2)
    rows[1] = dataclasses.replace(rows[1], split="validation")
    with pytest.raises(ValueError):
        build_transition_archive(rows, tmp_path / "bad.npz")


def test_episode_must_start_at_zero_for_rollout_horizons(tmp_path):
    rows = _episode("bad-origin", "train", 0, 2)
    rows = [dataclasses.replace(row, step=row.step + 1) for row in rows]
    with pytest.raises(ValueError):
        build_transition_archive(rows, tmp_path / "bad-origin.npz")


def test_test_split_requires_matching_freeze_manifest(tmp_path):
    _write_fake_cache(tmp_path)
    cache = FrozenCache(tmp_path)
    with pytest.raises(PermissionError):
        cache.indices_for_split("test")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "protocol": "v8_wm_test_freeze_v1",
        "test_unlocked": True,
        "cache_manifest_sha256": sha256_file(tmp_path / CACHE_FILES["manifest"]),
        "selected_variant": "full",
        "checkpoints": [{}, {}, {}],
    }), encoding="utf-8")
    rows = cache.indices_for_split("test", test_freeze_manifest=freeze)
    assert len(rows) == 1

