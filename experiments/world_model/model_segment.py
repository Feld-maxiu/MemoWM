"""The windowed transformer, applied once per episode segment instead of once
per transition.

``model.py`` attends over the last ``max_history`` observations and then reads a
prediction off the final time step only. Every transition therefore re-encodes
its whole window: at window W each observation is processed W times, and the
attention cost is quadratic in W for a single prediction.

The block-causal mask already makes step *t* invisible to steps before it, so
nothing stops the same forward pass from producing a prediction at *every* step.
Doing that turns cost-per-prediction from O(W^2) into O(L) amortised over a
segment of length L, which is what makes a long context affordable: a 64-step
segment costs about what the windowed model spends on eight 32-step windows,
and gives 64 predictions instead of eight.

Context is then the segment prefix rather than a sliding window. On MolmoWeb a
64-step segment leaves 97% of transitions seeing their entire trajectory prefix,
so a Transformer-XL memory over previous segments would only reach the remaining
3% -- it is deliberately not implemented here, because the measurement it would
serve is already covered by comparing against ``model_recurrent``, which has
genuinely unbounded context and lost by 668 bits.

Everything below the head is imported from ``model.py`` unchanged, so this is the
same model under a different reduction, not a new one.
"""
from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from .model import (
    ModelConfig,
    _layer_norm,
    _prepare_inputs,
    _transformer_layer,
    initialize_params as initialize_windowed_params,
)


@dataclasses.dataclass(frozen=True)
class SegmentConfig(ModelConfig):
    """``ModelConfig`` where ``max_history`` is the segment length.

    Kept as a separate type so that a config meant for segment training cannot
    be handed to ``train.py``'s per-transition loop, where ``max_history`` means
    a sliding window and the numbers would not be comparable.
    """
    @property
    def segment_length(self) -> int:
        return self.max_history


def initialize_params(config: SegmentConfig, seed: int = 0) -> dict:
    """Identical to the windowed model's: same shapes, same PRNG order."""
    return initialize_windowed_params(config, seed)


def predict(params, codes, code_valid, valid, actions, task_id, ablation_mask,
            config: SegmentConfig, *, rng=None, train: bool = False):
    """``(mask_logits, code_logits)`` at every step of the segment.

    ``codes`` is ``(B, L, K, M)``; the outputs are ``(B, L, K)`` and
    ``(B, L, K, M, C)``. The windowed model's own ``predict`` is the special case
    that keeps only ``[:, -1]``.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
    batch, length = codes.shape[0], codes.shape[1]
    present = jnp.asarray(valid, jnp.bool_)
    # `_prepare_inputs` takes one task per row; a segment never spans tasks
    # because it never spans episodes.
    hidden, present = _prepare_inputs(
        params, codes, code_valid, present, actions,
        jnp.asarray(task_id, jnp.int32)[:, 0], ablation_mask, config,
    )
    keys = jax.random.split(rng, config.num_layers)
    layer_fn = (
        jax.checkpoint(_transformer_layer, static_argnums=(3, 4, 6))
        if config.remat else _transformer_layer
    )
    for layer in range(config.num_layers):
        hidden = layer_fn(params, hidden, present, layer, config, keys[layer], train)
    hidden = _layer_norm(hidden, params["final_norm_scale"], params["final_norm_bias"])

    # The one departure from model.predict: every step, not just the last.
    pooled = jnp.mean(hidden, axis=2)
    mask_logits = pooled @ params["mask_head/w"] + params["mask_head/b"]
    pieces = hidden.reshape(
        batch, length, config.num_latent_tokens,
        config.num_subspaces, config.code_embedding_dim,
    )
    code_logits = jnp.einsum(
        "btige,igce->btigc", pieces,
        params.get("code_head/w", params["code_embedding"]),
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
