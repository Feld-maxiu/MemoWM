"""Perceiver-Resampler state tokenizer (report Eq. 9): maps a variable-length
observation feature sequence ``H_t`` (e.g. Qwen hidden states) to a fixed set of
``K_x`` state tokens of width ``d_x`` -- the frozen mixed target ``x_t``.

Hand-rolled functional JAX (param dict + explicit multi-head cross-attention) to
match the rest of the codebase and stay CPU-friendly. Deterministic (no dropout),
so it is safe to freeze after Stage-1 training and reuse at encode/decode time.
"""
from __future__ import annotations

import dataclasses
import hashlib

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class PerceiverConfig:
    num_latents: int          # K_x
    latent_dim: int           # d_x
    input_dim: int            # d_model of the observation features
    num_heads: int = 4
    num_layers: int = 2
    mlp_ratio: int = 2

    def __post_init__(self) -> None:
        if self.latent_dim % self.num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")


def initialize_perceiver(config: PerceiverConfig, seed: int = 0) -> dict:
    key = jax.random.PRNGKey(seed)
    keys = iter(jax.random.split(key, 16 + config.num_layers * 8))
    d = config.latent_dim

    def weight(shape, scale=None):
        scale = scale or np.sqrt(2.0 / max(1, shape[0]))
        return jax.random.truncated_normal(next(keys), -2, 2, shape) * scale

    params = {
        "latents": weight((config.num_latents, d), scale=0.02),
        "in/w": weight((config.input_dim, d)),
        "in/b": jnp.zeros((d,), jnp.float32),
    }
    hidden = d * config.mlp_ratio
    for layer in range(config.num_layers):
        params[f"l{layer}/q"] = weight((d, d))
        params[f"l{layer}/k"] = weight((d, d))
        params[f"l{layer}/v"] = weight((d, d))
        params[f"l{layer}/o"] = weight((d, d))
        params[f"l{layer}/norm1"] = jnp.ones((d,), jnp.float32)
        params[f"l{layer}/norm2"] = jnp.ones((d,), jnp.float32)
        params[f"l{layer}/mlp1"] = weight((d, hidden))
        params[f"l{layer}/mlp1_b"] = jnp.zeros((hidden,), jnp.float32)
        params[f"l{layer}/mlp2"] = weight((hidden, d))
        params[f"l{layer}/mlp2_b"] = jnp.zeros((d,), jnp.float32)
    return params


def _rms_norm(value, scale):
    return value * jax.lax.rsqrt(jnp.mean(jnp.square(value), -1, keepdims=True) + 1e-4) * scale


def _cross_attention(params, layer, latents, context, config: PerceiverConfig):
    d = config.latent_dim
    heads = config.num_heads
    head_dim = d // heads
    q = (latents @ params[f"l{layer}/q"]).reshape(-1, heads, head_dim)
    k = (context @ params[f"l{layer}/k"]).reshape(-1, heads, head_dim)
    v = (context @ params[f"l{layer}/v"]).reshape(-1, heads, head_dim)
    scores = jnp.einsum("qhd,khd->hqk", q, k) / np.sqrt(head_dim)
    weights = jax.nn.softmax(scores, -1)
    out = jnp.einsum("hqk,khd->qhd", weights, v).reshape(-1, d)
    return out @ params[f"l{layer}/o"]


def resample(params, features, config: PerceiverConfig):
    """features: (T, input_dim) -> x_t: (num_latents, latent_dim)."""
    context = features @ params["in/w"] + params["in/b"]
    latents = params["latents"]
    for layer in range(config.num_layers):
        normed = _rms_norm(latents, params[f"l{layer}/norm1"])
        latents = latents + _cross_attention(params, layer, normed, context, config)
        normed = _rms_norm(latents, params[f"l{layer}/norm2"])
        hidden = jax.nn.silu(normed @ params[f"l{layer}/mlp1"] + params[f"l{layer}/mlp1_b"])
        latents = latents + hidden @ params[f"l{layer}/mlp2"] + params[f"l{layer}/mlp2_b"]
    return latents


def perceiver_hash(params, config: PerceiverConfig) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"perceiver-v04\0")
    digest.update(repr(dataclasses.asdict(config)).encode())
    for name, value in sorted(params.items()):
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.digest()
