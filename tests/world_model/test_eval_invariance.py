from __future__ import annotations

import dataclasses

import numpy as np

from experiments.world_model.config import (
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from experiments.world_model.model import ModelConfig, initialize_params
from experiments.world_model.schema import PROTOCOL
from experiments.world_model.train import evaluate_model


class _Cache:
    task_names = ("task-a", "task-b")

    def __init__(self):
        self.transitions = {
            "task_ids": np.asarray([0, 1, 0, 1, 0], np.uint8),
            "action_policies": np.asarray([[0], [1], [0], [1], [0]], np.uint8),
        }

    def batch(self, rows):
        rows = np.asarray(rows, np.int64)
        count = len(rows)
        codes = np.zeros((count, 3, 4, 2), np.uint8)
        targets = np.zeros((count, 4, 2), np.uint8)
        for local, row in enumerate(rows):
            codes[local].fill(int(row) % 8)
            targets[local].fill((int(row) + 1) % 8)
        return {
            "transition_indices": rows,
            "history_codes": codes,
            "history_valid": np.ones((count, 3, 4), np.bool_),
            "history_present": np.ones((count, 3), np.bool_),
            "action_types": np.zeros((count, 3), np.uint8),
            "action_tags": np.zeros((count, 3), np.uint8),
            "action_refs": np.zeros((count, 3), np.uint8),
            "action_payloads": np.zeros((count, 3, 4), np.uint8),
            "action_lengths": np.zeros((count, 3), np.uint8),
            "task_ids": self.transitions["task_ids"][rows],
            "target_codes": targets,
            "target_valid": np.ones((count, 4), np.bool_),
            "target_indices": rows,
            "episode_ids": rows.astype(np.int32),
            "steps": np.zeros((count,), np.uint8),
            "policies": self.transitions["action_policies"][rows, -1],
            "structural_action_bits": np.full((count,), 11, np.uint16),
            "full_action_bits": np.full((count,), 17, np.uint16),
        }


def _config(eval_batch):
    model = ModelConfig(
        num_tasks=2, max_history=3, num_latent_tokens=4,
        num_subspaces=2, num_categories=8, num_observation_slots=4,
        code_embedding_dim=4, d_model=8, num_layers=1, num_heads=2,
        mlp_dim=16, dropout=0.0, action_embedding_dim=4,
        byte_embedding_dim=3, payload_hidden_dim=5, max_payload_bytes=4,
    )
    return ExperimentConfig(
        protocol=PROTOCOL,
        model=model,
        training=TrainingConfig(),
        evaluation=EvaluationConfig(batch_size=eval_batch, bootstrap_replicates=10),
        action={},
    )


def test_fp64_evaluation_is_batch_partition_invariant():
    cache = _Cache()
    params = initialize_params(_config(1).model, 0)
    rows = np.arange(5, dtype=np.int64)
    one, _ = evaluate_model(cache, rows, params, "full", _config(1))
    three, _ = evaluate_model(cache, rows, params, "full", _config(3))
    for name in (
        "mask_bits_per_transition", "code_bits_per_transition",
        "total_bits_per_transition", "mask_accuracy", "code_accuracy",
    ):
        assert abs(one[name] - three[name]) < 1e-12

    audited, rates = evaluate_model(
        cache, rows, params, "full", _config(3), keep_per_transition=True
    )
    assert audited["episodes"] == 5
    assert audited["action_bill_bits_per_transition"] == 17.0
    assert audited["task_id_bill_bits_per_transition"] == 4.0
    assert np.array_equal(rates["action_bill_bits"], np.full(5, 17.0))
    assert np.array_equal(rates["task_id_bill_bits"], np.full(5, 4.0))
    assert np.allclose(
        rates["total_episodic_bits"], rates["total_bits"] + 21.0
    )
