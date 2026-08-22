"""A0 no-bottleneck slot reconstruction primitives (JAX)."""
from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

# One source, not a second literal. The layout was repeated verbatim in several
# places and every copy kept reporting the pre-recycling (32, 12, 16, 4) after
# the four prompt slots were folded into detail -- nothing raised, the group
# metrics were simply sliced over the wrong spans. A1/A2 were corrected then;
# this default was missed.
from .continuous_bottleneck import DEFAULT_GROUP_SIZES


@dataclasses.dataclass(frozen=True)
class SlotReconstructionConfig:
    num_slots: int = 64
    token_dim: int = 512
    num_heads: int = 8
    ffn_hidden: int = 1024
    group_sizes: tuple[int, ...] = DEFAULT_GROUP_SIZES

    def __post_init__(self):
        object.__setattr__(self, "group_sizes", tuple(self.group_sizes))
        if self.num_slots < 1 or self.token_dim < 1 or self.num_heads < 1:
            raise ValueError("slot reconstruction dimensions must be positive")
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        if sum(self.group_sizes) != self.num_slots:
            raise ValueError("group sizes must sum to num_slots")


def initialize_params(config: SlotReconstructionConfig, seed: int = 0):
    key = jax.random.PRNGKey(seed)

    def normal(shape, std=0.02):
        nonlocal key
        key, subkey = jax.random.split(key)
        return jax.random.normal(subkey, shape, jnp.float32) * std

    def weight(shape):
        return normal(shape, 1.0 / math.sqrt(shape[0]))

    identity = jnp.eye(config.token_dim, dtype=jnp.float32)
    return {
        "slot_embedding": normal((config.num_slots, config.token_dim)),
        "decoder_queries": normal((config.num_slots, config.token_dim)),
        "q_norm/scale": jnp.ones((config.token_dim,), jnp.float32),
        "q_norm/bias": jnp.zeros((config.token_dim,), jnp.float32),
        "memory_norm/scale": jnp.ones((config.token_dim,), jnp.float32),
        "memory_norm/bias": jnp.zeros((config.token_dim,), jnp.float32),
        "ffn_norm/scale": jnp.ones((config.token_dim,), jnp.float32),
        "ffn_norm/bias": jnp.zeros((config.token_dim,), jnp.float32),
        "attn/q": weight((config.token_dim, config.token_dim)),
        "attn/k": weight((config.token_dim, config.token_dim)),
        "attn/v": weight((config.token_dim, config.token_dim)),
        "attn/out": weight((config.token_dim, config.token_dim)),
        "ffn/w1": weight((config.token_dim, config.ffn_hidden)),
        "ffn/b1": jnp.zeros((config.ffn_hidden,), jnp.float32),
        "ffn/w2": weight((config.ffn_hidden, config.token_dim)),
        "ffn/b2": jnp.zeros((config.token_dim,), jnp.float32),
        # OutputProjection consumes the raw Pre-Norm residual stream; there is no
        # final LayerNorm. Identity initialization avoids adding an artificial
        # output bottleneck to the wiring test.
        "output/w": identity,
        "output/b": jnp.zeros((config.token_dim,), jnp.float32),
    }


def layer_norm(values, scale, bias, eps=1e-5):
    mean = values.mean(axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + eps) * scale + bias


def reconstruct(
    params,
    xbar,
    valid_mask,
    config: SlotReconstructionConfig,
    *,
    slot_embedding_scale=1.0,
    value_source="normalized_memory",
    routing_mode="content_softmax",
    return_attention=False,
):
    """Pre-Norm query-to-slot reconstruction with no latent/global bottleneck."""
    xbar = jnp.asarray(xbar, jnp.float32)
    valid_mask = jnp.asarray(valid_mask, jnp.bool_)
    if xbar.shape[-2:] != (config.num_slots, config.token_dim):
        raise ValueError(f"unexpected xbar shape: {xbar.shape}")
    if valid_mask.shape != xbar.shape[:-1]:
        raise ValueError(f"valid mask {valid_mask.shape} does not match {xbar.shape}")
    batch_shape = xbar.shape[:-2]
    xbar = xbar * valid_mask[..., None]
    memory = xbar + slot_embedding_scale * params["slot_embedding"]
    queries = jnp.broadcast_to(
        params["decoder_queries"], batch_shape + params["decoder_queries"].shape
    )

    q_norm = layer_norm(queries, params["q_norm/scale"], params["q_norm/bias"])
    memory_norm = layer_norm(
        memory, params["memory_norm/scale"], params["memory_norm/bias"]
    )
    if value_source == "normalized_memory":
        value_input = memory_norm
    elif value_source == "raw_xbar":
        value_input = xbar
    else:
        raise ValueError(f"unknown value_source: {value_source}")
    heads = config.num_heads
    head_dim = config.token_dim // heads
    v = (value_input @ params["attn/v"]).reshape(
        batch_shape + (config.num_slots, heads, head_dim)
    )
    if routing_mode == "hard_diagonal":
        eye = jnp.eye(config.num_slots, dtype=jnp.float32)
        attention = jnp.broadcast_to(
            eye, batch_shape + (heads, config.num_slots, config.num_slots)
        )
        attention = attention * valid_mask[..., None, None, :]
        attended = v.reshape(
            batch_shape + (config.num_slots, config.token_dim)
        )
        # Oracle routing removes the decoder-query residual so the remaining
        # path is exactly W_V -> W_O -> Pre-Norm FFN -> OutputProjection.
        hidden = attended @ params["attn/out"]
    else:
        q = (q_norm @ params["attn/q"]).reshape(
            batch_shape + (config.num_slots, heads, head_dim)
        )
        if routing_mode == "content_softmax":
            key_input = memory_norm
        elif routing_mode == "slot_key":
            key_input = jnp.broadcast_to(
                slot_embedding_scale * params["slot_embedding"], memory.shape
            )
            value_input = xbar
            v = (value_input @ params["attn/v"]).reshape(
                batch_shape + (config.num_slots, heads, head_dim)
            )
        else:
            raise ValueError(f"unknown routing_mode: {routing_mode}")
        k = (key_input @ params["attn/k"]).reshape(
            batch_shape + (config.num_slots, heads, head_dim)
        )
        logits = jnp.einsum("...qhd,...khd->...hqk", q, k) / math.sqrt(head_dim)
        logits = jnp.where(valid_mask[..., None, None, :], logits, -1e30)
        attention = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("...hqk,...khd->...qhd", attention, v).reshape(
            batch_shape + (config.num_slots, config.token_dim)
        )
        hidden = queries + attended @ params["attn/out"]
    ffn_input = layer_norm(
        hidden, params["ffn_norm/scale"], params["ffn_norm/bias"]
    )
    ffn = jax.nn.gelu(ffn_input @ params["ffn/w1"] + params["ffn/b1"])
    hidden = hidden + ffn @ params["ffn/w2"] + params["ffn/b2"]
    # Deliberately no final LayerNorm before output projection.
    xbar_hat = hidden @ params["output/w"] + params["output/b"]
    xbar_hat = xbar_hat * valid_mask[..., None]
    if return_attention:
        return xbar_hat, attention
    return xbar_hat


def masked_mse(prediction, target, valid_mask):
    prediction = jnp.asarray(prediction, jnp.float32)
    target = jnp.asarray(target, jnp.float32)
    valid = jnp.asarray(valid_mask, jnp.float32)
    if prediction.shape != target.shape or valid.shape != target.shape[:-1]:
        raise ValueError("prediction/target/mask shapes do not match")
    denominator = jnp.maximum(valid.sum() * target.shape[-1], 1.0)
    return jnp.sum(jnp.square(prediction - target) * valid[..., None]) / denominator


def group_metrics(
    prediction,
    target,
    valid_mask,
    config: SlotReconstructionConfig,
):
    """Return natural/group MSE, zero baselines, and R² values."""
    output = {}

    def metrics_for(name, predicted, expected, valid):
        mse = masked_mse(predicted, expected, valid)
        zero = masked_mse(jnp.zeros_like(expected), expected, valid)
        r2 = 1.0 - mse / jnp.maximum(zero, 1e-12)
        output[f"{name}/mse"] = mse
        output[f"{name}/zero_mse"] = zero
        output[f"{name}/r2"] = r2
        output[f"{name}/valid_slots"] = valid.sum()

    metrics_for("all", prediction, target, valid_mask)
    start = 0
    names = ("image", "detail", "context", "prompt")
    for name, size in zip(names, config.group_sizes):
        stop = start + size
        metrics_for(
            name,
            prediction[..., start:stop, :],
            target[..., start:stop, :],
            valid_mask[..., start:stop],
        )
        start = stop
    return output


def attention_diagnostics(attention, valid_mask):
    """Diagnostic only: decoder query-to-same-slot attention alignment."""
    weights = attention.mean(axis=-3)  # (..., query, key)
    diagonal = jnp.diagonal(weights, axis1=-2, axis2=-1)
    valid = jnp.asarray(valid_mask, jnp.float32)
    diagonal_mean = jnp.sum(diagonal * valid) / jnp.maximum(valid.sum(), 1.0)
    top1 = jnp.argmax(weights, axis=-1)
    expected = jnp.arange(weights.shape[-2])
    top1_accuracy = jnp.sum((top1 == expected) * valid) / jnp.maximum(valid.sum(), 1.0)
    invalid_attention = jnp.sum(weights * (~jnp.asarray(valid_mask, jnp.bool_))[..., None, :])
    return {
        "attention/diagonal_mean": diagonal_mean,
        "attention/top1_slot_accuracy": top1_accuracy,
        "attention/invalid_weight_sum": invalid_attention,
    }
