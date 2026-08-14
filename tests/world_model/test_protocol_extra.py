from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.world_model.baselines import evaluate_baselines, run
from experiments.world_model.cache import CACHE_FILES, FrozenCache
from experiments.world_model.model import ModelConfig, initialize_params, predict
from experiments.world_model.schema import (
    ACTION_TYPE_IDS,
    PROTOCOL,
    REF_MASK_ID,
    TAG_MASK_ID,
)


def test_formal_baseline_rejects_same_fit_and_eval_split_before_io(tmp_path):
    args = SimpleNamespace(
        fit_split="train", eval_split="train", allow_same_split_debug=False,
        allow_nontrain_fit_debug=False, cache=tmp_path, output=tmp_path,
    )
    with pytest.raises(ValueError):
        run(args)


def test_distributions_are_conditioned_on_task():
    class Cache:
        pass

    cache = Cache()
    cache.codes = np.zeros((8, 64, 32), np.uint8)
    cache.codes[[2, 3, 6, 7]] = 255
    cache.valid = np.ones((8, 64), np.bool_)
    histories = np.full((4, 7), -1, np.int64)
    histories[:, -1] = [0, 2, 4, 6]
    cache.transitions = {
        "history_indices": histories,
        "target_indices": np.asarray([1, 3, 5, 7], np.int64),
        "task_ids": np.asarray([0, 1, 0, 1], np.uint8),
    }
    fit, evaluate = np.asarray([0, 1]), np.asarray([2, 3])
    matched = evaluate_baselines(cache, fit, evaluate)
    cache.transitions["task_ids"] = np.asarray([0, 1, 1, 0], np.uint8)
    mismatched = evaluate_baselines(cache, fit, evaluate)
    assert np.all(matched["marginal_code_bits"] < mismatched["marginal_code_bits"])


def test_legacy_layout_is_rejected_before_arrays_are_opened(tmp_path):
    (tmp_path / CACHE_FILES["manifest"]).write_text(json.dumps({
        "protocol": PROTOCOL, "layout": [32, 12, 16, 4],
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        FrozenCache(tmp_path)


def _small_config():
    return ModelConfig(
        num_tasks=2, max_history=3, num_latent_tokens=4,
        num_subspaces=2, num_categories=8, num_observation_slots=4,
        code_embedding_dim=4, d_model=8, num_layers=1, num_heads=2,
        mlp_dim=16, dropout=0.0, action_embedding_dim=4,
        byte_embedding_dim=3, payload_hidden_dim=5, max_payload_bytes=4,
    )


def _inputs(config):
    batch, history = 2, config.max_history
    return (
        np.zeros((batch, history, 4, 2), np.uint8),
        np.ones((batch, history, 4), np.bool_),
        np.ones((batch, history), np.bool_),
        {
            "types": np.zeros((batch, history), np.uint8),
            "tags": np.zeros((batch, history), np.uint8),
            "refs": np.zeros((batch, history), np.uint8),
            "payloads": np.zeros((batch, history, 4), np.uint8),
            "lengths": np.zeros((batch, history), np.uint8),
        },
    )


def test_policy_is_excluded_and_pad_mask_ids_are_distinct():
    params = initialize_params(_small_config(), 0)
    assert all("policy" not in name for name in params)
    assert ACTION_TYPE_IDS["PAD"] != ACTION_TYPE_IDS["MASK"]
    assert REF_MASK_ID != 64 and TAG_MASK_ID != 8


def test_action_and_payload_masks_are_true_ablation_controls():
    config = _small_config()
    params = initialize_params(config, 0)
    codes, valid, present, actions = _inputs(config)
    changed = {name: value.copy() for name, value in actions.items()}
    changed["types"].fill(2)
    changed["tags"].fill(6)
    changed["refs"].fill(12)
    changed["payloads"].fill(123)
    changed["lengths"].fill(4)
    for variant in ("t_only", "no_action"):
        left = predict(
            params, codes, valid, actions, np.zeros((2,), np.uint8),
            variant, config, history_present=present,
        )
        right = predict(
            params, codes, valid, changed, np.zeros((2,), np.uint8),
            variant, config, history_present=present,
        )
        assert np.array_equal(np.asarray(left[1]), np.asarray(right[1]))
    payload_only = {name: value.copy() for name, value in actions.items()}
    payload_only["payloads"].fill(123)
    payload_only["lengths"].fill(4)
    left = predict(
        params, codes, valid, actions, np.zeros((2,), np.uint8),
        "structural_action", config, history_present=present,
    )
    right = predict(
        params, codes, valid, payload_only, np.zeros((2,), np.uint8),
        "structural_action", config, history_present=present,
    )
    assert np.array_equal(np.asarray(left[1]), np.asarray(right[1]))
