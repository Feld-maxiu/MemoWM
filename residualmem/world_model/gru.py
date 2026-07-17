from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..types import CanonicalState, Prediction, Predictor, StateSchema


@dataclasses.dataclass(frozen=True)
class GRUConfig:
    hidden_size: int = 512
    layers: int = 2
    action_values: int = 17
    update_bias: float = -1.0


def initialize_params(schema: StateSchema, config: GRUConfig, seed: int = 0):
    if config.layers != 2:
        raise ValueError("ResidualMem v0.3 implements exactly two GRU layers")
    key = jax.random.PRNGKey(seed)
    keys = iter(jax.random.split(key, 32 + len(schema.fields) * 2))
    hidden = config.hidden_size
    input_size = len(schema.fields) + config.action_values + 1

    def weight(shape, scale=None):
        scale = scale or np.sqrt(2.0 / max(1, shape[0]))
        return jax.random.truncated_normal(next(keys), -2, 2, shape) * scale

    params: dict[str, jax.Array] = {
        "input/w": weight((input_size, hidden)),
        "input/b": jnp.zeros((hidden,), jnp.float32),
    }
    for layer in range(2):
        params[f"gru{layer}/norm_scale"] = jnp.ones((2 * hidden,), jnp.float32)
        params[f"gru{layer}/w"] = weight((2 * hidden, 3 * hidden))
        params[f"gru{layer}/b"] = jnp.zeros((3 * hidden,), jnp.float32)
    for index, spec in enumerate(schema.fields):
        outputs = _field_outputs(spec)
        params[f"head{index}/w"] = weight((hidden, outputs), scale=0.01)
        params[f"head{index}/b"] = jnp.zeros((outputs,), jnp.float32)
    return params


def _field_outputs(spec) -> int:
    if spec.field_type == "bool":
        return 2
    if spec.num_values is not None:
        return int(spec.num_values)
    if spec.field_type == "literal":
        return 2
    return 1


def _state_features(states: jax.Array, schema: StateSchema) -> jax.Array:
    columns = []
    for index, spec in enumerate(schema.fields):
        value = states[..., index].astype(jnp.float32)
        if spec.num_values is not None:
            value = value / max(1, spec.num_values - 1)
        else:
            value = jnp.sign(value) * jnp.log1p(jnp.abs(value))
        columns.append(value)
    return jnp.stack(columns, -1)


def _rms_norm(value: jax.Array, scale: jax.Array) -> jax.Array:
    return value * jax.lax.rsqrt(jnp.mean(jnp.square(value), -1, keepdims=True) + 1e-4) * scale


def _gru_step(params, layer: int, carry: jax.Array, inputs: jax.Array, update_bias: float):
    value = jnp.concatenate([carry, inputs], -1)
    value = _rms_norm(value, params[f"gru{layer}/norm_scale"])
    value = value @ params[f"gru{layer}/w"] + params[f"gru{layer}/b"]
    reset, candidate, update = jnp.split(value, 3, -1)
    candidate = jnp.tanh(jax.nn.sigmoid(reset) * candidate)
    update = jax.nn.sigmoid(update + update_bias)
    return update * candidate + (1.0 - update) * carry


def model_step(params, carry, state, action, dt, schema: StateSchema, config: GRUConfig):
    state = _state_features(state, schema)
    action = jax.nn.one_hot(action, config.action_values, dtype=jnp.float32)
    inputs = jnp.concatenate([state, action, jnp.asarray(dt, jnp.float32)[..., None]], -1)
    inputs = jax.nn.silu(inputs @ params["input/w"] + params["input/b"])
    h0 = _gru_step(params, 0, carry[0], inputs, config.update_bias)
    h1 = _gru_step(params, 1, carry[1], h0, config.update_bias)
    logits = tuple(
        h1 @ params[f"head{index}/w"] + params[f"head{index}/b"]
        for index in range(len(schema.fields)))
    return (h0, h1), logits


def sequence_logits(params, input_states, actions, schema: StateSchema, config: GRUConfig):
    hidden = config.hidden_size
    initial = (jnp.zeros((hidden,), jnp.float32), jnp.zeros((hidden,), jnp.float32))

    def step(carry, values):
        state, action = values
        carry, logits = model_step(params, carry, state, action, 1.0, schema, config)
        return carry, logits

    _, logits = jax.lax.scan(step, initial, (input_states, actions))
    return logits


def sequence_loss(params, input_states, target_states, actions, schema, config):
    logits = sequence_logits(params, input_states, actions, schema, config)
    losses = []
    for index, spec in enumerate(schema.fields):
        if _field_outputs(spec) == 1:
            losses.append(jnp.mean(jnp.square(logits[index][..., 0] - target_states[..., index])))
        else:
            labels = target_states[..., index].astype(jnp.int32)
            losses.append(jnp.mean(
                -jnp.take_along_axis(
                    jax.nn.log_softmax(logits[index]), labels[..., None], -1)[..., 0]))
    return jnp.mean(jnp.stack(losses))


class GRUPredictor(Predictor):
    def __init__(self, params, schema: StateSchema, config: GRUConfig):
        self.params = {key: jnp.asarray(value) for key, value in params.items()}
        self.schema = schema
        self.config = config
        self.predictor_id = "gru-v1-" + self._content_hash().hex()
        self.reset()

    def reset(self) -> None:
        hidden = self.config.hidden_size
        self.carry = (jnp.zeros((hidden,), jnp.float32), jnp.zeros((hidden,), jnp.float32))

    def new_session(self) -> Predictor:
        return GRUPredictor(self.params, self.schema, self.config)

    def predict_next(
        self, reconstructed: CanonicalState, action: int, dt: int = 1
    ) -> Prediction:
        values = jnp.asarray(reconstructed.values, jnp.float32)
        self.carry, logits = model_step(
            self.params, self.carry, values, jnp.asarray(action), jnp.asarray(dt),
            self.schema, self.config)
        defaults = []
        distributions: dict[str, Any] = {}
        for spec, field_logits, previous in zip(
            self.schema.fields, logits, reconstructed.values
        ):
            field_logits = np.asarray(field_logits, dtype=np.float32)
            distributions[spec.name] = field_logits
            if spec.field_type == "literal":
                defaults.append(previous if int(field_logits.argmax()) == 0 else "UNKNOWN")
            elif spec.field_type == "integer" and spec.num_values is None:
                defaults.append(int(np.rint(field_logits[0])))
            elif spec.field_type == "integer":
                probs = np.exp(field_logits - field_logits.max())
                probs /= probs.sum()
                defaults.append(int(np.searchsorted(np.cumsum(probs), 0.5)))
            elif spec.field_type == "bool":
                defaults.append(bool(field_logits.argmax()))
            else:
                defaults.append(int(field_logits.argmax()))
        return Prediction(self.schema.make_state(defaults), distributions)

    @property
    def hash_bytes(self) -> bytes:
        return self._content_hash()

    def _content_hash(self) -> bytes:
        digest = hashlib.sha256()
        digest.update(self.schema.hash_bytes)
        digest.update(json.dumps(dataclasses.asdict(self.config), sort_keys=True).encode())
        for name, value in sorted(self.params.items()):
            array = np.asarray(value)
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes(order="C"))
        return digest.digest()


def save_gru_checkpoint(path: str | Path, params, schema: StateSchema, config: GRUConfig):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({
        "schema_hash": schema.hash_hex,
        "schema": json.loads(schema.canonical_json()),
        "config": dataclasses.asdict(config),
    }, sort_keys=True)
    arrays = {name.replace("/", "__"): np.asarray(value) for name, value in params.items()}
    arrays["metadata"] = np.asarray(metadata)
    np.savez_compressed(path, **arrays)
    return path


def load_gru_checkpoint(path: str | Path, schema: StateSchema) -> GRUPredictor:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["schema_hash"] != schema.hash_hex:
            raise ValueError(
                "checkpoint schema hash mismatch; v0.2/RD checkpoints are not "
                "compatible with ResidualMem v0.3 exact-only schemas"
            )
        config = GRUConfig(**metadata["config"])
        params = {
            name.replace("__", "/"): data[name]
            for name in data.files if name != "metadata"
        }
    return GRUPredictor(params, schema, config)
