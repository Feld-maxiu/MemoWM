"""A world model with a recurrent state instead of a fixed history window.

``model.py`` flattens the last ``max_history`` observations into one sequence and
re-attends over all of it at every step. Context is hard-capped and attention is
quadratic in the window: on MolmoWeb a 32-step window is 1024 tokens, 1.07 GB of
attention per copy, and needs rematerialisation to fit at all -- while still
seeing all of only 90% of trajectories, and 15.8% at the inherited window of 7.

Here the time axis is carried by a state rather than re-attended:

    per step   tokens_t = embed(z_t) + slot + action_t          (B, K, d)
    spatial    a few transformer layers over the K slots only
    temporal   h_t = GRU(h_{t-1}, tokens_t), one GRU shared across slots
    head       the same tied rank-8 code head and copy gate, applied to h_t

Context is then unbounded and per-step cost is O(1) in trajectory length. The
head is deliberately unchanged so that bits/transition means the same thing as
in the windowed model and the two can be compared on the same dev transitions.

The spatial layers run on all steps at once -- they do not mix time -- and only
the GRU is scanned, so a chunk of 64 steps costs (B*L, heads, K, K) of attention:
67 MB at B=32, against the windowed model's 1.07 GB. The recurrence is the cheap
part; dropping the quadratic window is what pays for it.

Parameters are initialised in a fixed order from one seed, as in ``model.py``, so
that a run is reproducible up to XLA's own nondeterminism -- measured at ~31 bits
between two otherwise identical runs on this corpus.
"""
from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp

from .model import (
    ModelConfig,
    _byte_channel,
    _coordinate_bin,
    _dropout,
    _layer_norm,
    _signed_log,
    _normal,
    _weight,
)


@dataclasses.dataclass(frozen=True)
class RecurrentConfig(ModelConfig):
    """``ModelConfig`` plus the two knobs the recurrent stack adds.

    ``max_history`` is inherited but means something different here: it is only
    the length of the time-embedding table, used to tell the model where in a
    chunk it is. Context itself is unbounded.
    """
    #: Transformer layers over the K slots within a step. The remaining
    #: ``num_layers - spatial_layers`` budget goes to the GRU, which is one layer
    #: by construction, so this is where depth is spent.
    spatial_layers: int = 6
    #: Steps per chunk. A boundary restarts h, so this bounds the context the
    #: model can actually use during training even though inference need not.
    chunk_length: int = 64

    def __post_init__(self):
        super().__post_init__()
        if self.spatial_layers < 1:
            raise ValueError("spatial_layers must be positive")
        if self.chunk_length < 2:
            raise ValueError("chunk_length must be at least 2")


def initialize_params(config: RecurrentConfig, seed: int = 0) -> dict:
    key = jax.random.PRNGKey(seed)

    def take():
        nonlocal key
        key, subkey = jax.random.split(key)
        return subkey

    action_width = config.action_embedding_dim * 3 + config.payload_hidden_dim
    if config.use_coordinate_channel:
        action_width += config.action_embedding_dim

    params = {
        "code_embedding": _normal(take(), (
            config.num_latent_tokens, config.num_subspaces,
            config.num_categories, config.code_embedding_dim,
        )),
        "slot_embedding": _normal(take(), (config.num_latent_tokens, config.d_model)),
        "task_embedding": _normal(take(), (config.num_tasks, config.d_model)),
        "state_mask": _normal(take(), (config.d_model,)),
        "valid/w": _weight(take(), (config.num_observation_slots, config.d_model)),
        "valid/b": jnp.zeros((config.d_model,), jnp.float32),
        "action/type_embedding": _normal(take(), (
            config.num_action_types, config.action_embedding_dim
        )),
        "action/byte_embedding": _normal(take(), (256, config.byte_embedding_dim)),
        "action/gru_wx": _weight(take(), (
            config.byte_embedding_dim, 3 * config.payload_hidden_dim
        )),
        "action/gru_wh": _weight(take(), (
            config.payload_hidden_dim, 3 * config.payload_hidden_dim
        )),
        "action/gru_b": jnp.zeros((3 * config.payload_hidden_dim,), jnp.float32),
        "action/payload_mask": _normal(take(), (config.payload_hidden_dim,)),
    }
    if config.use_coordinate_channel:
        params["action/x_embedding"] = _normal(take(), (
            config.coordinate_bins + 1, config.action_embedding_dim
        ))
        params["action/y_embedding"] = _normal(take(), (
            config.coordinate_bins + 1, config.action_embedding_dim
        ))
        params["action/delta_w"] = _weight(take(), (3, config.action_embedding_dim))
        params["action/delta_b"] = jnp.zeros((config.action_embedding_dim,), jnp.float32)
    params.update({
        "action/proj_w": _weight(take(), (action_width, config.d_model)),
        "action/proj_b": jnp.zeros((config.d_model,), jnp.float32),
        "action/mask": _normal(take(), (config.d_model,)),
        "mask_head/w": _weight(take(), (config.d_model, config.num_observation_slots)),
        "mask_head/b": jnp.zeros((config.num_observation_slots,), jnp.float32),
        "code_head/b": jnp.zeros((
            config.num_latent_tokens, config.num_subspaces, config.num_categories
        ), jnp.float32),
    })
    for layer in range(config.spatial_layers):
        prefix = f"spatial{layer}"
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
    # The temporal GRU. Shared across slots: a slot is a learned query, and the
    # same update rule should apply to all of them, which also keeps the
    # parameter count near the windowed model's for a fair comparison.
    params.update({
        "state/gru_wx": _weight(take(), (config.d_model, 3 * config.d_model)),
        "state/gru_wh": _weight(take(), (config.d_model, 3 * config.d_model)),
        "state/gru_b": jnp.zeros((3 * config.d_model,), jnp.float32),
        "state/h0": _normal(take(), (config.num_latent_tokens, config.d_model)),
        "state/norm_scale": jnp.ones((config.d_model,), jnp.float32),
        "state/norm_bias": jnp.zeros((config.d_model,), jnp.float32),
    })
    if config.use_copy_gate:
        params["copy_head/w"] = jnp.zeros((
            config.num_latent_tokens, config.num_subspaces, config.code_embedding_dim
        ), jnp.float32)
        params["copy_head/b"] = jnp.zeros((
            config.num_latent_tokens, config.num_subspaces
        ), jnp.float32)
    return params


def _embed_step_codes(params, codes, config: RecurrentConfig):
    """``(B, L, K, M)`` code ids to ``(B, L, K, d)``."""
    codes = jnp.asarray(codes, jnp.int32)
    pieces = jnp.take_along_axis(
        params["code_embedding"][None, None], codes[..., None, None], axis=4
    )
    return pieces[..., 0, :].reshape(codes.shape[:3] + (config.d_model,))


def _action_tokens(params, actions, config: RecurrentConfig, payload_mode: str):
    """One ``(B, L, d)`` vector per step, same channels as the windowed model."""
    type_ids = jnp.asarray(actions["types"], jnp.int32)
    structural = (
        params["action/type_embedding"][type_ids],
        params["action/x_embedding"][
            _coordinate_bin(actions["x"], actions["has_coord"], config.coordinate_bins)
        ],
        params["action/y_embedding"][
            _coordinate_bin(actions["y"], actions["has_coord"], config.coordinate_bins)
        ],
        jnp.stack((
            _signed_log(actions["dx"]), _signed_log(actions["dy"]),
            jnp.asarray(actions["has_delta"], jnp.float32),
        ), axis=-1) @ params["action/delta_w"] + params["action/delta_b"],
    )
    payload = _byte_channel(
        params, actions["payloads"], actions["lengths"],
        params["action/payload_mask"], payload_mode, type_ids.shape, config,
    )
    combined = jnp.concatenate((*structural, payload), axis=-1)
    return combined @ params["action/proj_w"] + params["action/proj_b"]


def _spatial_layer(params, hidden, layer, config, key, train):
    """Self-attention across the K slots of one step. No time mixing."""
    prefix = f"spatial{layer}"
    batch, slots, width = hidden.shape
    normalized = _layer_norm(
        hidden, params[f"{prefix}/attn_norm_scale"], params[f"{prefix}/attn_norm_bias"]
    )
    qkv = normalized @ params[f"{prefix}/qkv_w"] + params[f"{prefix}/qkv_b"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    head_dim = config.d_model // config.num_heads

    def heads(values):
        return values.reshape(batch, slots, config.num_heads, head_dim)

    q, k, v = heads(q), heads(k), heads(v)
    logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(head_dim)
    attention = jax.nn.softmax(logits, axis=-1)
    key_attn, key_resid, key_mlp = jax.random.split(key, 3)
    attention = _dropout(attention, key_attn, config.dropout, train)
    attended = jnp.einsum("bhqk,bkhd->bqhd", attention, v).reshape(batch, slots, width)
    attended = attended @ params[f"{prefix}/attn_out_w"] + params[f"{prefix}/attn_out_b"]
    hidden = hidden + _dropout(attended, key_resid, config.dropout, train)
    mlp_input = _layer_norm(
        hidden, params[f"{prefix}/mlp_norm_scale"], params[f"{prefix}/mlp_norm_bias"]
    )
    mlp = jax.nn.gelu(mlp_input @ params[f"{prefix}/mlp_w1"] + params[f"{prefix}/mlp_b1"])
    mlp = mlp @ params[f"{prefix}/mlp_w2"] + params[f"{prefix}/mlp_b2"]
    return hidden + _dropout(mlp, key_mlp, config.dropout, train)


def _gru_step(params, state, inputs):
    """Standard GRU, applied per slot with shared weights."""
    gates_x = inputs @ params["state/gru_wx"] + params["state/gru_b"]
    gates_h = state @ params["state/gru_wh"]
    width = gates_x.shape[-1] // 3
    reset = jax.nn.sigmoid(gates_x[..., :width] + gates_h[..., :width])
    update = jax.nn.sigmoid(gates_x[..., width:2 * width] + gates_h[..., width:2 * width])
    candidate = jnp.tanh(gates_x[..., 2 * width:] + reset * gates_h[..., 2 * width:])
    return (1.0 - update) * state + update * candidate


def predict(params, codes, code_valid, valid, actions, task_id, ablation_mask,
            config: RecurrentConfig, *, rng=None, train: bool = False):
    """``(mask_logits, code_logits)`` for every step of every chunk.

    ``codes`` is ``(B, L, K, M)`` and the outputs are ``(B, L, K)`` and
    ``(B, L, K, M, C)``: one prediction per step, not one per chunk.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    batch, length = codes.shape[0], codes.shape[1]

    state = _embed_step_codes(params, codes, config)
    summary = jnp.asarray(code_valid, jnp.float32) @ params["valid/w"] + params["valid/b"]
    state = state + summary[:, :, None, :] + params["slot_embedding"][None, None]
    if ablation_mask in ("t_only", "state_only"):
        state = jnp.broadcast_to(params["state_mask"], state.shape)

    if ablation_mask in ("t_only", "no_action"):
        action = jnp.broadcast_to(params["action/mask"], (batch, length, config.d_model))
    else:
        action = _action_tokens(
            params, actions, config,
            "mask" if ablation_mask == "structural_action" else "full",
        )
    tokens = state + action[:, :, None, :] + params["task_embedding"][task_id][:, :, None, :]
    tokens = tokens * jnp.asarray(valid, jnp.float32)[:, :, None, None]

    # Spatial layers see (B*L) independent steps; time is not mixed here.
    flat = tokens.reshape(batch * length, config.num_latent_tokens, config.d_model)
    keys = jax.random.split(rng, config.spatial_layers + 1)
    layer_fn = (
        jax.checkpoint(_spatial_layer, static_argnums=(2, 3, 5))
        if config.remat else _spatial_layer
    )
    for layer in range(config.spatial_layers):
        flat = layer_fn(params, flat, layer, config, keys[layer], train)
    tokens = flat.reshape(batch, length, config.num_latent_tokens, config.d_model)

    # Temporal recurrence. `no_history` cuts the carry, which is this model's
    # analogue of the windowed variant that masks all but the last observation.
    carry_scale = 0.0 if ablation_mask in ("no_history", "struct_no_history") else 1.0
    initial = jnp.broadcast_to(
        params["state/h0"], (batch, config.num_latent_tokens, config.d_model)
    )

    def step(carry, item):
        inputs, live = item
        updated = _gru_step(params, carry * carry_scale, inputs)
        updated = jnp.where(live[:, None, None], updated, carry)
        return updated, updated

    _final, hidden = jax.lax.scan(
        step, initial,
        (tokens.transpose(1, 0, 2, 3), jnp.asarray(valid, jnp.bool_).T),
    )
    hidden = hidden.transpose(1, 0, 2, 3)
    hidden = _layer_norm(hidden, params["state/norm_scale"], params["state/norm_bias"])

    pooled = jnp.mean(hidden, axis=2)
    mask_logits = pooled @ params["mask_head/w"] + params["mask_head/b"]
    pieces = hidden.reshape(
        batch, length, config.num_latent_tokens,
        config.num_subspaces, config.code_embedding_dim,
    )
    code_logits = jnp.einsum(
        "btige,igce->btigc", pieces, params["code_embedding"]
    ) + params["code_head/b"]
    if "copy_head/w" in params:
        source = jnp.asarray(codes, jnp.int32)
        copy_logit = jnp.einsum(
            "btige,ige->btig", pieces, params["copy_head/w"]
        ) + params["copy_head/b"]
        log_keep = jax.nn.log_sigmoid(copy_logit)[..., None]
        log_change = jax.nn.log_sigmoid(-copy_logit)[..., None]
        residual = jax.nn.log_softmax(code_logits, axis=-1)
        on_source = jax.nn.one_hot(source, config.num_categories, dtype=code_logits.dtype)
        keep_term = jnp.where(on_source > 0, log_keep, -1e30)
        code_logits = jnp.logaddexp(keep_term, log_change + residual)
    return mask_logits, code_logits
