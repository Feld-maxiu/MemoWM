from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from experiments.state_tokenizer.common import sha256_file
from experiments.world_model.cache import V8_EXPECTED_TRANSITIONS
from experiments.world_model.schema import PROTOCOL
from experiments.world_model.statistics import (
    _validate_run_artifact,
    paired_episode_bootstrap,
)


def test_constant_episode_gain_has_exact_positive_interval():
    differences = np.ones((3, 6), np.float64) * 2.5
    tasks = np.asarray([0, 0, 0, 1, 1, 1], np.uint8)
    episodes = np.asarray([0, 0, 1, 2, 3, 3], np.int32)
    result = paired_episode_bootstrap(
        differences, tasks, episodes, replicates=200, seed=0
    )
    assert result["micro"]["ci95"] == [2.5, 2.5]
    assert result["task_macro"]["ci95"] == [2.5, 2.5]


def _write_formal_run(tmp_path):
    transition_path = tmp_path / "per_transition.npz"
    np.savez(transition_path, total_bits=np.ones((2,), np.float64))
    resolved = {
        "protocol": PROTOCOL,
        "variant": "full",
        "seed": 0,
        "model": {"d_model": 256},
        "training": {"batch_size": 32, "max_steps": 20_000},
        "evaluation": {"batch_size": 32},
        "action": {"exclude_policy": True},
    }
    resolved_path = tmp_path / "resolved.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8"
    )
    run = {
        "protocol": PROTOCOL,
        "variant": "full",
        "seed": 0,
        "parameter_count": 10,
        "parameter_shapes": {"w": [2, 5]},
        "data": {
            "cache_manifest_sha256": "cache",
            "train_transitions": V8_EXPECTED_TRANSITIONS["train"],
            "selection": "validation",
            "selection_transitions": V8_EXPECTED_TRANSITIONS["validation"],
            "subset": None,
            "test_evaluated": False,
        },
        "training": {"batch_size": 32, "max_steps": 20_000},
        "artifacts": {
            "per_transition_sha256": sha256_file(transition_path),
            "resolved_config": "resolved.yaml",
            "resolved_config_sha256": sha256_file(resolved_path),
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return transition_path


def test_formal_run_validation_rejects_subsets_and_stale_config(tmp_path):
    transition_path = _write_formal_run(tmp_path)
    artifact, signature = _validate_run_artifact("full", 0, transition_path)
    assert artifact["parameter_count"] == 10
    assert signature["resolved_core"]["training"]["max_steps"] == 20_000

    run_path = tmp_path / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["data"]["subset"] = "10k"
    run_path.write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(ValueError):
        _validate_run_artifact("full", 0, transition_path)
