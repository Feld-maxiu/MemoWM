from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..types import StateSchema, Trajectory
from .gru import (
    GRUConfig, GRUPredictor, initialize_params, save_gru_checkpoint, sequence_logits,
    sequence_loss,
)


def train_gru(
    trajectories: list[Trajectory],
    schema: StateSchema,
    output: str | Path,
    config: GRUConfig | None = None,
    seed: int = 0,
    epochs: int = 50,
    learning_rate: float = 3e-4,
    segment_length: int = 64,
) -> dict[str, float]:
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    config = config or GRUConfig()
    params = initialize_params(schema, config, seed)
    optimizer = optax.chain(optax.clip_by_global_norm(100.0), optax.adam(learning_rate))
    state = optimizer.init(params)
    value_and_grad = jax.jit(jax.value_and_grad(
        lambda p, inputs, targets, actions: sequence_loss(
            p, inputs, targets, actions, schema, config)))

    if segment_length < 1:
        raise ValueError("segment_length must be positive")
    teacher_examples = [
        example
        for trajectory in trajectories
        for example in _segment_examples(trajectory, segment_length)
    ]
    if not teacher_examples:
        raise ValueError("training needs at least one transition")
    params, state, losses = _train_examples(
        params, state, teacher_examples, epochs, optimizer, value_and_grad, seed)
    teacher_loss = float(np.mean(losses[-max(1, len(teacher_examples)):]))

    output = Path(output)
    save_gru_checkpoint(output, params, schema, config)
    metrics = evaluate_gru(GRUPredictor(params, schema, config), trajectories, schema)
    metrics.update(teacher_loss=teacher_loss)
    output.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def evaluate_gru(
    predictor: GRUPredictor,
    trajectories: list[Trajectory],
    schema: StateSchema,
) -> dict[str, float]:
    total = correct = 0
    absolute_error = 0.0
    integer_count = 0
    nll_values = []
    for trajectory in trajectories:
        inputs, targets, actions = _example(trajectory)
        logits = sequence_logits(
            predictor.params, jnp.asarray(inputs), jnp.asarray(actions),
            schema, predictor.config)
        for index, spec in enumerate(schema.fields):
            values = np.asarray(logits[index])
            labels = targets[:, index].astype(np.int32)
            if values.shape[-1] > 1:
                probs = np.asarray(jax.nn.log_softmax(values))
                nll_values.extend((-probs[np.arange(len(labels)), labels]).tolist())
                pred = values.argmax(-1)
            else:
                pred = np.rint(values[:, 0]).astype(np.int32)
            correct += int((pred == labels).sum())
            total += len(labels)
            if spec.field_type == "integer":
                absolute_error += float(np.abs(pred - labels).sum())
                integer_count += len(labels)
    return {
        "field_nll": float(np.mean(nll_values)) if nll_values else 0.0,
        "default_accuracy": correct / max(1, total),
        "integer_mae": absolute_error / max(1, integer_count),
    }


def _example(trajectory: Trajectory):
    return (
        np.asarray([state.values for state in trajectory.states[:-1]], np.float32),
        np.asarray([state.values for state in trajectory.states[1:]], np.float32),
        np.asarray(trajectory.actions, np.int32),
    )


def _segment_examples(trajectory: Trajectory, segment_length: int):
    for start in range(0, len(trajectory.actions), segment_length):
        stop = min(start + segment_length, len(trajectory.actions))
        yield (
            np.asarray(
                [state.values for state in trajectory.states[start:stop]],
                np.float32,
            ),
            np.asarray(
                [state.values for state in trajectory.states[start + 1:stop + 1]],
                np.float32,
            ),
            np.asarray(trajectory.actions[start:stop], np.int32),
        )


def _train_examples(params, state, examples, epochs, optimizer, value_and_grad, seed):
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    for _ in range(epochs):
        for index in rng.permutation(len(examples)):
            inputs, targets, actions = examples[index]
            loss, grads = value_and_grad(
                params, jnp.asarray(inputs), jnp.asarray(targets), jnp.asarray(actions))
            updates, state = optimizer.update(grads, state, params)
            params = optax.apply_updates(params, updates)
            losses.append(float(loss))
    return params, state, losses
