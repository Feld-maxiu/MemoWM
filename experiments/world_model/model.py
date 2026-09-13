from __future__ import annotations

import dataclasses
import math
from typing import Mapping

import jax
import jax.numpy as jnp

from .schema import (
    ACTION_TYPE_IDS,
    MAX_HISTORY,
    MAX_PAYLOAD_BYTES,
    NUM_CATEGORIES,
    NUM_LATENT_TOKENS,
    NUM_MODEL_REFS,
    NUM_MODEL_TAGS,
    NUM_OBSERVATION_SLOTS,
    NUM_SUBSPACES,
    REF_MASK_ID,
    TAG_MASK_ID,
    validate_variant,
)


LN2 = math.log(2.0)


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    num_tasks: int = 12
    max_history: int = MAX_HISTORY
    num_latent_tokens: int = NUM_LATENT_TOKENS
    num_subspaces: int = NUM_SUBSPACES
    num_categories: int = NUM_CATEGORIES
    num_observation_slots: int = NUM_OBSERVATION_SLOTS
    code_embedding_dim: int = 8
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 8
    mlp_dim: int = 1024
    dropout: float = 0.1
    action_embedding_dim: int = 32
    byte_embedding_dim: int = 16
    payload_hidden_dim: int = 64
    use_target_channel: bool = False
    use_copy_gate: bool = False
    use_persistence_aux: bool = False
    aux_persistence_weight: float = 1.0
    use_action_gate_bias: bool = False
    output_smoothing: float = 0.0
    exclude_source_in_change: bool = False
    max_payload_bytes: int = MAX_PAYLOAD_BYTES
    use_coordinate_channel: bool = False
    num_action_types: int = len(ACTION_TYPE_IDS)
    num_model_tags: int = NUM_MODEL_TAGS
    num_model_refs: int = NUM_MODEL_REFS
    coordinate_bins: int = 64
    remat: bool = False

    def __post_init__(self):
        integers = (
            self.num_tasks, self.max_history, self.num_latent_tokens,
            self.num_subspaces, self.num_categories, self.num_observation_slots,
            self.code_embedding_dim, self.d_model, self.num_layers,
            self.num_heads, self.mlp_dim, self.action_embedding_dim,
            self.byte_embedding_dim, self.payload_hidden_dim,
            self.max_payload_bytes,
        )
        if any(value < 1 for value in integers):
            raise ValueError("all model dimensions must be positive")
        if self.d_model != self.num_subspaces * self.code_embedding_dim:
            raise ValueError(
                "weight tying requires d_model == num_subspaces * code_embedding_dim"
            )
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        if not 0.0 <= self.output_smoothing < 1.0:
            raise ValueError("output_smoothing must lie in [0,1)")
        for name in ("num_action_types", "num_model_tags", "num_model_refs",
                     "coordinate_bins"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


def _normal(key, shape, std=0.02):
    return jax.random.normal(key, shape, jnp.float32) * std


def _weight(key, shape):
    return _normal(key, shape, 1.0 / math.sqrt(shape[0]))


def initialize_params(config: ModelConfig, seed: int = 0) -> dict:
    key = jax.random.PRNGKey(seed)

    def take():
        nonlocal key
        key, subkey = jax.random.split(key)
        return subkey

    action_width = config.action_embedding_dim * 3 + config.payload_hidden_dim
    if config.use_coordinate_channel:
        action_width += config.action_embedding_dim
    if config.use_target_channel:
        action_width += config.payload_hidden_dim
    params = {
        "code_embedding": _normal(take(), (
            config.num_latent_tokens,
            config.num_subspaces,
            config.num_categories,
            config.code_embedding_dim,
        )),
        "slot_embedding": _normal(take(), (config.num_latent_tokens, config.d_model)),
        "time_embedding": _normal(take(), (config.max_history, config.d_model)),
        "task_embedding": _normal(take(), (config.num_tasks, config.d_model)),
        "state_mask": _normal(take(), (config.d_model,)),
        "state_pad": _normal(take(), (config.d_model,)),
        "valid/w": _weight(take(), (config.num_observation_slots, config.d_model)),
        "valid/b": jnp.zeros((config.d_model,), jnp.float32),
        "action/type_embedding": _normal(take(), (
            config.num_action_types, config.action_embedding_dim
        )),
    }
    if not config.use_coordinate_channel:
        params["action/tag_embedding"] = _normal(take(), (
            config.num_model_tags, config.action_embedding_dim
        ))
        params["action/ref_embedding"] = _normal(take(), (
            config.num_model_refs, config.action_embedding_dim
        ))
    params.update({
        "action/byte_embedding": _normal(take(), (256, config.byte_embedding_dim)),
        "action/gru_wx": _weight(take(), (
            config.byte_embedding_dim, 3 * config.payload_hidden_dim
        )),
        "action/gru_wh": _weight(take(), (
            config.payload_hidden_dim, 3 * config.payload_hidden_dim
        )),
        "action/gru_b": jnp.zeros((3 * config.payload_hidden_dim,), jnp.float32),
        "action/payload_mask": _normal(take(), (config.payload_hidden_dim,)),
    })
    if config.use_coordinate_channel:
        params["action/x_embedding"] = _normal(take(), (
            config.coordinate_bins + 1, config.action_embedding_dim
        ))
        params["action/y_embedding"] = _normal(take(), (
            config.coordinate_bins + 1, config.action_embedding_dim
        ))
        params["action/delta_w"] = _weight(take(), (3, config.action_embedding_dim))
        params["action/delta_b"] = jnp.zeros((config.action_embedding_dim,), jnp.float32)
    if config.use_target_channel:
        params["action/target_mask"] = _normal(take(), (config.payload_hidden_dim,))
    params.update({
        "action/proj_w": _weight(take(), (action_width, config.d_model)),
        "action/proj_b": jnp.zeros((config.d_model,), jnp.float32),
        "action/mask": _normal(take(), (config.d_model,)),
        "action/pad": _normal(take(), (config.d_model,)),
        "mask_head/w": _weight(take(), (config.d_model, config.num_observation_slots)),
        "mask_head/b": jnp.zeros((config.num_observation_slots,), jnp.float32),
        "code_head/b": jnp.zeros((
            config.num_latent_tokens, config.num_subspaces, config.num_categories
        ), jnp.float32),
    })
    for layer in range(config.num_layers):
        prefix = f"layer{layer}"
        params.update({
            f"{prefix}/attn_norm_scale": jnp.ones((config.d_model,), jnp.float32),
            f"{prefix}/attn_norm_bias": jnp.zeros((config.d_model,), jnp.float32),
            f"{prefix}/qkv_w": _weight(take(), (config.d_model, 3 * config.d_model)),
            f"{prefix}/qkv_b": jnp.zeros((3 * config.d_model,), jnp.float32),
            f"{prefix}/attn_out_w": _weight(take(), (config.d_model, config.d_model)),
            f"{prefix}/attn_out_b": jnp.zeros((config.d_model,), jnp.float32),
            f"{prefix}/mlp_norm_scale": jnp.ones((config.d_model,), jnp.float32),
            f"{prefix}/mlp_norm_bias": jnp.zeros((config.d_model,), jnp.float32),
            f"{prefix}/mlp_w1": _weight(take(), (config.d_model, config.mlp_dim)),
            f"{prefix}/mlp_b1": jnp.zeros((config.mlp_dim,), jnp.float32),
            f"{prefix}/mlp_w2": _weight(take(), (config.mlp_dim, config.d_model)),
            f"{prefix}/mlp_b2": jnp.zeros((config.d_model,), jnp.float32),
        })
    params["final_norm_scale"] = jnp.ones((config.d_model,), jnp.float32)
    params["final_norm_bias"] = jnp.zeros((config.d_model,), jnp.float32)
    if config.use_copy_gate:
        params["copy_head/w"] = jnp.zeros((
            config.num_latent_tokens, config.num_subspaces,
            config.code_embedding_dim,
        ), jnp.float32)
        params["copy_head/b"] = jnp.zeros((
            config.num_latent_tokens, config.num_subspaces
        ), jnp.float32)
        if config.use_persistence_aux:
            params["persistence_head/w"] = jnp.zeros((
                config.num_latent_tokens, config.num_subspaces,
                2 * config.code_embedding_dim,
            ), jnp.float32)
            params["persistence_head/b"] = jnp.zeros((
                config.num_latent_tokens, config.num_subspaces,
            ), jnp.float32)
            params["persistence_gate_scale"] = jnp.ones((), jnp.float32)
        if config.use_action_gate_bias:
            params["copy_action_bias"] = jnp.zeros(
                (config.num_action_types,), jnp.float32
            )
    return params


def parameter_count(params: Mapping[str, jax.Array]) -> int:
    return int(sum(int(value.size) for value in jax.tree_util.tree_leaves(params)))


def parameter_shapes(params: Mapping[str, jax.Array]) -> dict[str, tuple[int, ...]]:
    return {name: tuple(value.shape) for name, value in params.items()}


def _layer_norm(values, scale, bias, eps=1e-5):
    values = jnp.asarray(values, jnp.float32)
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + eps) * scale + bias


def _dropout(values, key, rate: float, train: bool):
    if not train or rate == 0.0:
        return values
    keep = 1.0 - rate
    mask = jax.random.bernoulli(key, keep, values.shape)
    return values * mask.astype(values.dtype) / keep


def block_causal_mask(max_history: int, tokens_per_time: int) -> jax.Array:
    times = jnp.repeat(jnp.arange(max_history), tokens_per_time)
    return times[None, :] <= times[:, None]


def _embed_codes(params, codes, config: ModelConfig):
    codes = jnp.asarray(codes, jnp.int32)
    expected = (
        codes.shape[0], config.max_history,
        config.num_latent_tokens, config.num_subspaces,
    )
    if codes.shape != expected:
        raise ValueError(f"history_codes shape {codes.shape} != {expected}")
    token = jnp.arange(config.num_latent_tokens)[None, None, :, None]
    group = jnp.arange(config.num_subspaces)[None, None, None, :]
    pieces = params["code_embedding"][token, group, codes]
    return pieces.reshape(codes.shape[:-1] + (config.d_model,))


def _byte_gru(params, payloads, lengths, config: ModelConfig):
    payloads = jnp.asarray(payloads, jnp.int32)
    lengths = jnp.asarray(lengths, jnp.int32)
    batch_shape = payloads.shape[:-1]
    flat = payloads.reshape((-1, config.max_payload_bytes))
    flat_lengths = lengths.reshape((-1,))
    embedded = params["action/byte_embedding"][flat]
    hidden = jnp.zeros((flat.shape[0], config.payload_hidden_dim), jnp.float32)

    def step(hidden, item):
        index, values = item
        x_part = values @ params["action/gru_wx"]
        h_part = hidden @ params["action/gru_wh"]
        x_z, x_r, x_n = jnp.split(x_part, 3, axis=-1)
        h_z, h_r, h_n = jnp.split(h_part, 3, axis=-1)
        b_z, b_r, b_n = jnp.split(params["action/gru_b"], 3, axis=-1)
        z = jax.nn.sigmoid(x_z + h_z + b_z)
        r = jax.nn.sigmoid(x_r + h_r + b_r)
        candidate = jnp.tanh(x_n + r * h_n + b_n)
        updated = z * hidden + (1.0 - z) * candidate
        active = index < flat_lengths
        return jnp.where(active[:, None], updated, hidden), None

    hidden, _ = jax.lax.scan(
        step,
        hidden,
        (jnp.arange(config.max_payload_bytes), jnp.swapaxes(embedded, 0, 1)),
    )
    return hidden.reshape(batch_shape + (config.payload_hidden_dim,))


def _byte_channel(params, byte_ids, lengths, mask_param, mode, shape, config):
    if mode == "full":
        return _byte_gru(params, byte_ids, lengths, config)
    if mode == "mask":
        return jnp.broadcast_to(mask_param, shape + (config.payload_hidden_dim,))
    raise ValueError(mode)


def actions_from_batch(batch, config: ModelConfig) -> dict:
    actions = {
        "types": batch["action_types"],
        "payloads": batch["action_payloads"],
        "lengths": batch["action_lengths"],
    }
    if config.use_coordinate_channel:
        for name in ("x", "y", "dx", "dy", "has_coord", "has_delta"):
            actions[name] = batch[f"action_{name}"]
    else:
        actions["tags"] = batch["action_tags"]
        actions["refs"] = batch["action_refs"]
    return actions


def _coordinate_bin(values, present, bins: int):
    index = jnp.clip((jnp.asarray(values, jnp.float32) * bins).astype(jnp.int32),
                     0, bins - 1)
    return jnp.where(jnp.asarray(present, jnp.bool_), index, bins)


def _signed_log(values):
    values = jnp.asarray(values, jnp.float32)
    return jnp.sign(values) * jnp.log1p(jnp.abs(values))


def _action_embedding(
    params, actions, config: ModelConfig, payload_mode: str,
    target_mode: str = "mask",
):
    type_ids = jnp.asarray(actions["types"], jnp.int32)
    if config.use_coordinate_channel:
        structural = (
            params["action/type_embedding"][type_ids],
            params["action/x_embedding"][
                _coordinate_bin(actions["x"], actions["has_coord"], config.coordinate_bins)
            ],
            params["action/y_embedding"][
                _coordinate_bin(actions["y"], actions["has_coord"], config.coordinate_bins)
            ],
            jnp.stack(
                (_signed_log(actions["dx"]), _signed_log(actions["dy"]),
                 jnp.asarray(actions["has_delta"], jnp.float32)),
                axis=-1,
            ) @ params["action/delta_w"] + params["action/delta_b"],
        )
    else:
        structural = (
            params["action/type_embedding"][type_ids],
            params["action/tag_embedding"][jnp.asarray(actions["tags"], jnp.int32)],
            params["action/ref_embedding"][jnp.asarray(actions["refs"], jnp.int32)],
        )
    payload = _byte_channel(
        params, actions["payloads"], actions["lengths"],
        params["action/payload_mask"], payload_mode, type_ids.shape, config,
    )
    if "action/target_mask" in params:
        target = _byte_channel(
            params, actions["targets"], actions["target_lengths"],
            params["action/target_mask"], target_mode, type_ids.shape, config,
        )
        combined = jnp.concatenate((*structural, target, payload), axis=-1)
    else:
        combined = jnp.concatenate((*structural, payload), axis=-1)
    return combined @ params["action/proj_w"] + params["action/proj_b"]


def _prepare_inputs(
    params,
    history_codes,
    history_valid,
    history_present,
    actions,
    task_id,
    variant: str,
    config: ModelConfig,
):
    variant = validate_variant(variant, allow_dev=True)
    data_present = jnp.asarray(history_present, jnp.bool_)
    if data_present.shape != (history_codes.shape[0], config.max_history):
        raise ValueError(f"history_present has wrong shape {data_present.shape}")
    if not bool(jnp.shape(history_valid) == (
        history_codes.shape[0], config.max_history, config.num_observation_slots
    )):
        raise ValueError(f"history_valid has wrong shape {jnp.shape(history_valid)}")

    fixed_context = variant in (
        "no_history", "t_only", "state_only", "struct_no_history"
    )
    present = jnp.ones_like(data_present) if fixed_context else data_present
    actual_state = _embed_codes(params, history_codes, config)
    valid_summary = (
        jnp.asarray(history_valid, jnp.float32) @ params["valid/w"] + params["valid/b"]
    )
    state = actual_state + valid_summary[:, :, None, :]
    last_time = (
        jnp.arange(config.max_history) == config.max_history - 1
    )[None, :, None, None]
    if variant == "t_only":
        state = jnp.broadcast_to(params["state_mask"], state.shape)
    elif variant in ("no_history", "state_only", "struct_no_history"):
        state = jnp.where(last_time, state, params["state_mask"])
    if not fixed_context:
        state = jnp.where(
            data_present[:, :, None, None],
            state,
            params["state_pad"][None, None, None, :],
        )

    if variant in ("t_only", "no_action", "state_only"):
        action = jnp.broadcast_to(
            params["action/mask"],
            (history_codes.shape[0], config.max_history, config.d_model),
        )
    else:
        structural_only = variant in ("structural_action", "struct_no_history")
        action = _action_embedding(
            params, actions, config,
            payload_mode="mask" if (
                structural_only or variant == "semantic_action"
            ) else "full",
            target_mode="mask" if structural_only else "full",
        )
    if variant in ("no_history", "struct_no_history"):
        action = jnp.where(
            last_time[..., 0], action, params["action/mask"][None, None, :]
        )
    if not fixed_context:
        action = jnp.where(
            data_present[:, :, None], action, params["action/pad"][None, None, :]
        )

    task_id = jnp.asarray(task_id, jnp.int32)
    task = params["task_embedding"][task_id][:, None, None, :]
    time = params["time_embedding"][None, :, None, :]
    slot = params["slot_embedding"][None, None, :, :]
    hidden = state + action[:, :, None, :] + task + time + slot
    hidden = hidden * present[:, :, None, None]
    return hidden, present


def _transformer_layer(params, hidden, present, layer, config, key, train):
    batch = hidden.shape[0]
    sequence = config.max_history * config.num_latent_tokens
    x = hidden.reshape(batch, sequence, config.d_model)
    active = jnp.repeat(present, config.num_latent_tokens, axis=1)
    prefix = f"layer{layer}"
    normalized = _layer_norm(
        x, params[f"{prefix}/attn_norm_scale"], params[f"{prefix}/attn_norm_bias"]
    )
    qkv = normalized @ params[f"{prefix}/qkv_w"] + params[f"{prefix}/qkv_b"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    head_dim = config.d_model // config.num_heads

    def heads(values):
        return values.reshape(batch, sequence, config.num_heads, head_dim)

    q, k, v = heads(q), heads(k), heads(v)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(head_dim)
    allowed = block_causal_mask(config.max_history, config.num_latent_tokens)
    allowed = allowed[None, None, :, :] & active[:, None, None, :]
    attention = jax.nn.softmax(jnp.where(allowed, logits, -1e30), axis=-1)
    attention = attention * allowed
    attention = attention / jnp.maximum(attention.sum(axis=-1, keepdims=True), 1e-30)
    key_attn, key_resid, key_mlp = jax.random.split(key, 3)
    attention = _dropout(attention, key_attn, config.dropout, train)
    attended = jnp.einsum("bhqk,bkhd->bqhd", attention, v).reshape(
        batch, sequence, config.d_model
    )
    attended = attended @ params[f"{prefix}/attn_out_w"] + params[f"{prefix}/attn_out_b"]
    x = x + _dropout(attended, key_resid, config.dropout, train)
    mlp_input = _layer_norm(
        x, params[f"{prefix}/mlp_norm_scale"], params[f"{prefix}/mlp_norm_bias"]
    )
    mlp = jax.nn.gelu(
        mlp_input @ params[f"{prefix}/mlp_w1"] + params[f"{prefix}/mlp_b1"]
    )
    mlp = mlp @ params[f"{prefix}/mlp_w2"] + params[f"{prefix}/mlp_b2"]
    x = x + _dropout(mlp, key_mlp, config.dropout, train)
    x = x * active[:, :, None]
    return x.reshape(hidden.shape)


def predict(
    params,
    history_codes,
    history_valid,
    actions,
    task_id,
    ablation_mask: str,
    config: ModelConfig,
    *,
    history_present=None,
    rng=None,
    train: bool = False,
    source_log_prior=None,
    return_aux: bool = False,
):
    history_codes = jnp.asarray(history_codes, jnp.int32)
    if history_present is None:
        history_present = jnp.any(jnp.asarray(history_valid, jnp.bool_), axis=-1)
    if rng is None:
        rng = jax.random.PRNGKey(0)
    hidden, present = _prepare_inputs(
        params, history_codes, history_valid, history_present,
        actions, task_id, ablation_mask, config,
    )
    keys = jax.random.split(rng, config.num_layers)
    layer_fn = (
        jax.checkpoint(_transformer_layer, static_argnums=(3, 4, 6))
        if config.remat else _transformer_layer
    )
    for layer in range(config.num_layers):
        hidden = layer_fn(
            params, hidden, present, layer, config, keys[layer], train
        )
    hidden = _layer_norm(hidden, params["final_norm_scale"], params["final_norm_bias"])
    current = hidden[:, -1]
    pooled = jnp.mean(current, axis=1)
    mask_logits = pooled @ params["mask_head/w"] + params["mask_head/b"]
    pieces = current.reshape(
        current.shape[0], config.num_latent_tokens,
        config.num_subspaces, config.code_embedding_dim,
    )
    code_logits = jnp.einsum(
        "bige,igce->bigc", pieces,
        params.get("code_head/w", params["code_embedding"]),
    ) + params["code_head/b"]
    if source_log_prior is not None:
        code_logits = code_logits + source_log_prior
    aux_keep_logit = None
    if "copy_head/w" in params:
        source_codes = jnp.asarray(history_codes, jnp.int32)[:, -1]
        copy_logit = jnp.einsum(
            "bige,ige->big", pieces, params["copy_head/w"]
        ) + params["copy_head/b"]
        aux_keep_logit = None
        if (
            getattr(config, "use_persistence_aux", False)
            and "persistence_head/w" in params
        ):
            # code_embedding is the tied (i, g, C, e) head weight, so the
            # per-axis source embedding is a gather over the category axis.
            one_hot_source = jax.nn.one_hot(
                source_codes, config.num_categories,
                dtype=code_logits.dtype,
            )
            source_embedding = jnp.einsum(
                "bigc,igce->bige", one_hot_source, params["code_embedding"]
            )
            features = jnp.concatenate([pieces, source_embedding], axis=-1)
            aux_keep_logit = jnp.einsum(
                "bige,ige->big", features, params["persistence_head/w"]
            ) + params["persistence_head/b"]
            copy_logit = (
                copy_logit + params["persistence_gate_scale"] * aux_keep_logit
            )
        if getattr(config, "use_action_gate_bias", False) and (
            "copy_action_bias" in params
        ):
            # The last action drives the source -> target transition, so its
            # type shifts the per-axis keep/change prior of the copy gate.
            action_type = jnp.asarray(actions["types"], jnp.int32)[:, -1]
            copy_logit = copy_logit + params["copy_action_bias"][action_type][
                :, None, None
            ]
        log_keep = jax.nn.log_sigmoid(copy_logit)[..., None]
        log_change = jax.nn.log_sigmoid(-copy_logit)[..., None]
        on_source = jax.nn.one_hot(
            source_codes, config.num_categories, dtype=code_logits.dtype
        )
        if config.exclude_source_in_change:
            # The keep path already covers the source code; the change path
            # answers "where does the code move to" and therefore
            # renormalizes over the non-source codes only (mirroring the
            # copy baseline's destination distribution, which also excludes
            # the source).  Without this the residual head wastes much of
            # its mass re-predicting "no change" on changed axes.
            residual = jax.nn.log_softmax(
                jnp.where(on_source > 0, -1e30, code_logits), axis=-1
            )
        else:
            residual = jax.nn.log_softmax(code_logits, axis=-1)
        keep_term = jnp.where(on_source > 0, log_keep, -1e30)
        code_logits = jnp.logaddexp(keep_term, log_change + residual)
    if config.output_smoothing > 0:
        # Fixed epsilon floor on the output distribution (no fitted
        # parameters): P = (1-eps) softmax + eps/C, returned as log-probs.
        log_probs = jax.nn.log_softmax(code_logits, axis=-1)
        code_logits = jnp.logaddexp(
            math.log1p(-config.output_smoothing) + log_probs,
            math.log(config.output_smoothing / config.num_categories),
        )
    if return_aux:
        return mask_logits, code_logits, aux_keep_logit
    return mask_logits, code_logits


def codelength_bits(mask_logits, code_logits, target_valid, target_codes, *,
                    code_log_probs=False):
    target_valid = jnp.asarray(target_valid, jnp.float32)
    target_codes = jnp.asarray(target_codes, jnp.int32)
    mask_matrix = (
        jax.nn.softplus(mask_logits) - target_valid * mask_logits
    ) / LN2
    if code_log_probs:
        log_probability = code_logits
    else:
        log_probability = jax.nn.log_softmax(code_logits, axis=-1)
    selected = jnp.take_along_axis(
        log_probability, target_codes[..., None], axis=-1
    )[..., 0]
    code_matrix = -selected / LN2
    mask_bits = jnp.sum(mask_matrix, axis=-1)
    code_bits = jnp.sum(code_matrix, axis=(-2, -1))
    return {
        "mask_matrix": mask_matrix,
        "code_matrix": code_matrix,
        "mask_bits": mask_bits,
        "code_bits": code_bits,
        "total_bits": mask_bits + code_bits,
    }


def loss_and_metrics(
    params, batch, variant: str, config: ModelConfig, *, rng=None, train=False
):
    actions = actions_from_batch(batch, config)
    mask_logits, code_logits, aux_keep_logit = predict(
        params,
        batch["history_codes"], batch["history_valid"], actions,
        batch["task_ids"], variant, config,
        history_present=batch.get("history_present"), rng=rng, train=train,
        return_aux=True,
    )
    rates = codelength_bits(
        mask_logits, code_logits, batch["target_valid"], batch["target_codes"],
        code_log_probs=config.output_smoothing > 0,
    )
    mask_prediction = mask_logits >= 0
    code_prediction = jnp.argmax(code_logits, axis=-1)
    metrics = {
        "loss": jnp.mean(rates["total_bits"]),
        "mask_bits": jnp.mean(rates["mask_bits"]),
        "code_bits": jnp.mean(rates["code_bits"]),
        "mask_accuracy": jnp.mean(
            mask_prediction == jnp.asarray(batch["target_valid"], jnp.bool_)
        ),
        "code_accuracy": jnp.mean(
            code_prediction == jnp.asarray(batch["target_codes"], jnp.int32)
        ),
    }
    if aux_keep_logit is not None:
        history_present = batch.get("history_present")
        if history_present is None:
            history_present = jnp.any(
                jnp.asarray(batch["history_valid"], jnp.bool_), axis=-1
            )
        has_history = jnp.asarray(history_present, jnp.bool_)[:, -1]
        source_codes = jnp.asarray(batch["history_codes"], jnp.int32)[:, -1]
        label = (
            jnp.asarray(batch["target_codes"], jnp.int32)
            == source_codes
        ).astype(jnp.float32)
        valid = jnp.broadcast_to(
            has_history[:, None, None], label.shape
        ).astype(jnp.float32)
        positives = label * valid
        persistence_rate = (
            jnp.sum(positives) / jnp.maximum(jnp.sum(valid), 1.0)
        )
        rate = jnp.clip(persistence_rate, 0.01, 0.99)
        weight_positive = 0.5 / rate
        weight_negative = 0.5 / (1.0 - rate)
        logits = aux_keep_logit
        bce = -(
            weight_positive * positives * jax.nn.log_sigmoid(logits)
            + weight_negative * (valid - positives)
            * jax.nn.log_sigmoid(-logits)
        )
        aux_loss = jnp.sum(bce) / jnp.maximum(jnp.sum(valid), 1.0)
        predicted_keep = (jax.nn.sigmoid(logits) >= 0.5).astype(jnp.float32)
        metrics["persistence_rate"] = persistence_rate
        metrics["persistence_accuracy"] = jnp.sum(
            (predicted_keep == label) * valid
        ) / jnp.maximum(jnp.sum(valid), 1.0)
        metrics["aux_loss"] = aux_loss
        total_loss = (
            metrics["loss"]
            + getattr(config, "aux_persistence_weight", 1.0) * aux_loss
        )
        return total_loss, (metrics, rates, mask_logits, code_logits)
    return metrics["loss"], (metrics, rates, mask_logits, code_logits)
