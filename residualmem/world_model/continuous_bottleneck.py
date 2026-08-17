"""A1 continuous latent bottleneck primitives (JAX).

A0 (``slot_reconstruction.py``) has no bottleneck: 64 memory slots are read by 64
decoder queries. A1 adds exactly one new element -- a deterministic continuous
information bottleneck::

    (xbar_t, m_t) -> e_t in R^{N x d_e} -> xbar_hat_t

There is deliberately no categorical sampling, no ``z_t``, no prior, no KL, no
transition, no action and no temporal sequence. Leading batch dimensions are
independent states, never time. ``e_t`` stays continuous FP32; the compression
claim is scalar-dimensional (``N*d_e`` vs ``64*512``), not a bitrate claim.

Wiring follows the A0 conclusion that a routing address must have a stable path
that state content cannot perturb:

* Encoder (content-dependent keys are allowed here, because latent queries must
  decide *what* to absorb): ``K_{t,s} = W_K LN(xbar_{t,s} + p_s)`` and
  ``V_{t,s} = W_V xbar_{t,s}``. The input slot address ``p_s`` enters the key
  only, so the reconstructed content path stays free of learned address vectors.
* Decoder (first version, hard-coded): ``K_i = W_K a_i^latent`` from a learned
  per-index latent address and ``V_{t,i} = W_V e_{t,i}``. Decoder attention is
  therefore a sample-independent learned ``64 x N`` routing matrix.
"""
from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp

GROUP_NAMES: tuple[str, ...] = ("image", "detail", "context", "prompt")
DEFAULT_GROUP_SIZES = (32, 16, 16, 0)
PROTOCOL = "a1_continuous_bottleneck_v1"


@dataclasses.dataclass(frozen=True)
class ContinuousBottleneckConfig:
    """Shapes of the A1 continuous bottleneck.

    ``num_e_tokens`` (N) and ``e_dim`` (d_e) are the only capacity knobs the A1
    experiment sweeps. The default is the A1-Target point; the first executed
    experiment is A1-Wide (``num_e_tokens=64``), which has no scalar compression.
    """

    num_input_slots: int = 64
    input_dim: int = 512
    num_e_tokens: int = 16
    e_dim: int = 512
    num_heads: int = 8
    ffn_hidden: int = 1024
    # Default only; real runs inherit this from the A1 checkpoint. Kept in sync
    # with the shared layout so a run that omits the checkpoint does not silently
    # slice a (32,16,16,0) store as if it were (32,12,16,4).
    group_sizes: tuple[int, ...] = DEFAULT_GROUP_SIZES

    def __post_init__(self):
        object.__setattr__(self, "group_sizes", tuple(int(s) for s in self.group_sizes))
        sizes = (
            self.num_input_slots,
            self.input_dim,
            self.num_e_tokens,
            self.e_dim,
            self.num_heads,
            self.ffn_hidden,
        )
        if any(value < 1 for value in sizes):
            raise ValueError("continuous bottleneck dimensions must be positive")
        if self.e_dim % self.num_heads:
            raise ValueError("e_dim must be divisible by num_heads")
        if self.input_dim % self.num_heads:
            raise ValueError("input_dim must be divisible by num_heads")
        if sum(self.group_sizes) != self.num_input_slots:
            raise ValueError("group sizes must sum to num_input_slots")

    @property
    def input_scalars(self) -> int:
        return self.num_input_slots * self.input_dim

    @property
    def e_scalars(self) -> int:
        return self.num_e_tokens * self.e_dim

    @property
    def scalar_ratio(self) -> float:
        return self.e_scalars / self.input_scalars


def initialize_params(config: ContinuousBottleneckConfig, seed: int = 0) -> dict:
    key = jax.random.PRNGKey(seed)

    def normal(shape, std=0.02):
        nonlocal key
        key, subkey = jax.random.split(key)
        return jax.random.normal(subkey, shape, jnp.float32) * std

    def weight(shape):
        return normal(shape, 1.0 / math.sqrt(shape[0]))

    d_in = config.input_dim
    d_e = config.e_dim
    hidden = config.ffn_hidden
    return {
        # ---- encoder: 64 input slots -> N continuous latent tokens ----
        "encoder/input_positions": normal((config.num_input_slots, d_in)),
        "encoder/e_queries": normal((config.num_e_tokens, d_e)),
        "encoder/q_norm/scale": jnp.ones((d_e,), jnp.float32),
        "encoder/q_norm/bias": jnp.zeros((d_e,), jnp.float32),
        "encoder/memory_norm/scale": jnp.ones((d_in,), jnp.float32),
        "encoder/memory_norm/bias": jnp.zeros((d_in,), jnp.float32),
        "encoder/attn/q": weight((d_e, d_e)),
        "encoder/attn/k": weight((d_in, d_e)),
        "encoder/attn/v": weight((d_in, d_e)),
        "encoder/attn/out": weight((d_e, d_e)),
        "encoder/ffn_norm/scale": jnp.ones((d_e,), jnp.float32),
        "encoder/ffn_norm/bias": jnp.zeros((d_e,), jnp.float32),
        "encoder/ffn/w1": weight((d_e, hidden)),
        "encoder/ffn/b1": jnp.zeros((hidden,), jnp.float32),
        "encoder/ffn/w2": weight((hidden, d_e)),
        "encoder/ffn/b2": jnp.zeros((d_e,), jnp.float32),
        # ---- decoder: N continuous latent tokens -> 64 output slots ----
        # Latent addresses are indexed by latent position only; they never see
        # e_t, so decoder routing cannot be perturbed by state content.
        "decoder/e_addresses": normal((config.num_e_tokens, d_e)),
        "decoder/output_queries": normal((config.num_input_slots, d_in)),
        "decoder/q_norm/scale": jnp.ones((d_in,), jnp.float32),
        "decoder/q_norm/bias": jnp.zeros((d_in,), jnp.float32),
        "decoder/attn/q": weight((d_in, d_in)),
        "decoder/attn/k": weight((d_e, d_in)),
        "decoder/attn/v": weight((d_e, d_in)),
        "decoder/attn/out": weight((d_in, d_in)),
        "decoder/ffn_norm/scale": jnp.ones((d_in,), jnp.float32),
        "decoder/ffn_norm/bias": jnp.zeros((d_in,), jnp.float32),
        "decoder/ffn/w1": weight((d_in, hidden)),
        "decoder/ffn/b1": jnp.zeros((hidden,), jnp.float32),
        "decoder/ffn/w2": weight((hidden, d_in)),
        "decoder/ffn/b2": jnp.zeros((d_in,), jnp.float32),
        # OutputProjection consumes the raw Pre-Norm residual stream; there is no
        # final LayerNorm. Identity init matches A0 so the bottleneck, not the
        # output head, is the thing under test.
        "decoder/output/w": jnp.eye(d_in, dtype=jnp.float32),
        "decoder/output/b": jnp.zeros((d_in,), jnp.float32),
    }


def expected_param_shapes(config: ContinuousBottleneckConfig) -> dict[str, tuple[int, ...]]:
    return {
        name: tuple(value.shape) for name, value in initialize_params(config, 0).items()
    }


def parameter_count(params) -> int:
    return int(sum(int(jnp.asarray(value).size) for value in params.values()))


def layer_norm(values, scale, bias, eps=1e-5):
    mean = values.mean(axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    return (values - mean) * jax.lax.rsqrt(variance + eps) * scale + bias


def _masked_softmax(logits, valid_mask):
    """Softmax over keys with exact-zero weight on invalid keys.

    Rows whose keys are all invalid return exactly zero (finite), instead of the
    uniform distribution a plain masked softmax would produce.
    """
    keep = valid_mask[..., None, None, :]
    logits = jnp.where(keep, logits, -1e30)
    weights = jax.nn.softmax(logits, axis=-1)
    weights = weights * keep
    total = jnp.sum(weights, axis=-1, keepdims=True)
    return weights / jnp.maximum(total, 1e-30)


def _check_inputs(xbar, valid_mask, config):
    xbar = jnp.asarray(xbar, jnp.float32)
    valid_mask = jnp.asarray(valid_mask, jnp.bool_)
    if xbar.shape[-2:] != (config.num_input_slots, config.input_dim):
        raise ValueError(f"unexpected xbar shape: {xbar.shape}")
    if valid_mask.shape != xbar.shape[:-1]:
        raise ValueError(f"valid mask {valid_mask.shape} does not match {xbar.shape}")
    return xbar, valid_mask


def encode(
    params,
    xbar,
    valid_mask,
    config: ContinuousBottleneckConfig,
    *,
    return_attention=False,
):
    """xbar_t -> e_t. Address enters the key; only content enters the value."""
    xbar, valid_mask = _check_inputs(xbar, valid_mask, config)
    batch_shape = xbar.shape[:-2]
    heads = config.num_heads
    head_dim = config.e_dim // heads

    xbar_safe = xbar * valid_mask[..., None]
    memory_key = layer_norm(
        xbar_safe + params["encoder/input_positions"],
        params["encoder/memory_norm/scale"],
        params["encoder/memory_norm/bias"],
    )
    memory_value = xbar_safe

    queries = jnp.broadcast_to(
        params["encoder/e_queries"], batch_shape + params["encoder/e_queries"].shape
    )
    q_norm = layer_norm(
        queries, params["encoder/q_norm/scale"], params["encoder/q_norm/bias"]
    )
    q = (q_norm @ params["encoder/attn/q"]).reshape(
        batch_shape + (config.num_e_tokens, heads, head_dim)
    )
    k = (memory_key @ params["encoder/attn/k"]).reshape(
        batch_shape + (config.num_input_slots, heads, head_dim)
    )
    v = (memory_value @ params["encoder/attn/v"]).reshape(
        batch_shape + (config.num_input_slots, heads, head_dim)
    )
    logits = jnp.einsum("...qhd,...khd->...hqk", q, k) / math.sqrt(head_dim)
    attention = _masked_softmax(logits, valid_mask)
    attended = jnp.einsum("...hqk,...khd->...qhd", attention, v).reshape(
        batch_shape + (config.num_e_tokens, config.e_dim)
    )
    hidden = queries + attended @ params["encoder/attn/out"]
    ffn_input = layer_norm(
        hidden, params["encoder/ffn_norm/scale"], params["encoder/ffn_norm/bias"]
    )
    ffn = jax.nn.gelu(ffn_input @ params["encoder/ffn/w1"] + params["encoder/ffn/b1"])
    e_t = hidden + ffn @ params["encoder/ffn/w2"] + params["encoder/ffn/b2"]
    # Deliberately no final LayerNorm: e_t is the information path and must keep
    # its scale.
    if return_attention:
        return e_t, attention
    return e_t


def decoder_attention(params, config: ContinuousBottleneckConfig):
    """Learned ``(heads, 64, N)`` routing matrix.

    Content-independent by construction: both the output queries and the latent
    addresses are parameters, so this does not depend on e_t or on the batch.
    """
    heads = config.num_heads
    head_dim = config.input_dim // heads
    q_norm = layer_norm(
        params["decoder/output_queries"],
        params["decoder/q_norm/scale"],
        params["decoder/q_norm/bias"],
    )
    q = (q_norm @ params["decoder/attn/q"]).reshape(
        (config.num_input_slots, heads, head_dim)
    )
    k = (params["decoder/e_addresses"] @ params["decoder/attn/k"]).reshape(
        (config.num_e_tokens, heads, head_dim)
    )
    logits = jnp.einsum("qhd,khd->hqk", q, k) / math.sqrt(head_dim)
    return jax.nn.softmax(logits, axis=-1)


def decode(
    params,
    e_t,
    valid_mask,
    config: ContinuousBottleneckConfig,
    *,
    return_attention=False,
):
    """e_t -> xbar_hat_t through fixed latent addresses."""
    e_t = jnp.asarray(e_t, jnp.float32)
    valid_mask = jnp.asarray(valid_mask, jnp.bool_)
    if e_t.shape[-2:] != (config.num_e_tokens, config.e_dim):
        raise ValueError(f"unexpected e_t shape: {e_t.shape}")
    batch_shape = e_t.shape[:-2]
    if valid_mask.shape != batch_shape + (config.num_input_slots,):
        raise ValueError(f"valid mask {valid_mask.shape} does not match e_t {e_t.shape}")
    heads = config.num_heads
    head_dim = config.input_dim // heads

    attention = decoder_attention(params, config)
    v = (e_t @ params["decoder/attn/v"]).reshape(
        batch_shape + (config.num_e_tokens, heads, head_dim)
    )
    attended = jnp.einsum("hqk,...khd->...qhd", attention, v).reshape(
        batch_shape + (config.num_input_slots, config.input_dim)
    )
    queries = jnp.broadcast_to(
        params["decoder/output_queries"],
        batch_shape + params["decoder/output_queries"].shape,
    )
    hidden = queries + attended @ params["decoder/attn/out"]
    ffn_input = layer_norm(
        hidden, params["decoder/ffn_norm/scale"], params["decoder/ffn_norm/bias"]
    )
    ffn = jax.nn.gelu(ffn_input @ params["decoder/ffn/w1"] + params["decoder/ffn/b1"])
    hidden = hidden + ffn @ params["decoder/ffn/w2"] + params["decoder/ffn/b2"]
    xbar_hat = hidden @ params["decoder/output/w"] + params["decoder/output/b"]
    xbar_hat = xbar_hat * valid_mask[..., None]
    if return_attention:
        return xbar_hat, attention
    return xbar_hat


def reconstruct(
    params,
    xbar,
    valid_mask,
    config: ContinuousBottleneckConfig,
    *,
    return_attention=False,
    return_latent=False,
):
    """Full A1 path: xbar_t -> e_t -> xbar_hat_t."""
    e_t, encoder_weights = encode(
        params, xbar, valid_mask, config, return_attention=True
    )
    xbar_hat, decoder_weights = decode(
        params, e_t, valid_mask, config, return_attention=True
    )
    if not return_attention and not return_latent:
        return xbar_hat
    extras = []
    if return_attention:
        extras.append({"encoder": encoder_weights, "decoder": decoder_weights})
    if return_latent:
        extras.append(e_t)
    return (xbar_hat, *extras)


def masked_mse(prediction, target, valid_mask):
    """Natural per-valid-slot MSE; denominator is ``valid slots x token_dim``."""
    prediction = jnp.asarray(prediction, jnp.float32)
    target = jnp.asarray(target, jnp.float32)
    valid = jnp.asarray(valid_mask, jnp.float32)
    if prediction.shape != target.shape or valid.shape != target.shape[:-1]:
        raise ValueError("prediction/target/mask shapes do not match")
    denominator = jnp.maximum(valid.sum() * target.shape[-1], 1.0)
    return jnp.sum(jnp.square(prediction - target) * valid[..., None]) / denominator


def group_slice(config: ContinuousBottleneckConfig, group: str) -> slice:
    """Slot range owned by ``group`` under the configured layout."""
    if group not in GROUP_NAMES:
        raise ValueError(f"unknown group {group!r}; expected one of {GROUP_NAMES}")
    start = 0
    for name, size in zip(GROUP_NAMES, config.group_sizes):
        if name == group:
            return slice(start, start + size)
        start += size
    raise AssertionError("unreachable")  # pragma: no cover


def masked_group_mse(prediction, target, valid_mask,
                     config: ContinuousBottleneckConfig, group: str = "detail"):
    """One group's SSE over the *global* valid-element denominator.

    Deliberately not the group's own denominator. Added as
    ``L_nat + lambda * L_group``, this scaling makes ``lambda = 1`` mean exactly
    "count this group twice", i.e. a 2x weight per element. Normalising by the
    group's own valid elements would instead give detail ``1 + 363552/44279 =
    9.2x`` -- off by 4.6x, and invisible in the training curve, surfacing only
    as unexplained image/context degradation.
    """
    prediction = jnp.asarray(prediction, jnp.float32)
    target = jnp.asarray(target, jnp.float32)
    valid = jnp.asarray(valid_mask, jnp.float32)
    if prediction.shape != target.shape or valid.shape != target.shape[:-1]:
        raise ValueError("prediction/target/mask shapes do not match")
    denominator = jnp.maximum(valid.sum() * target.shape[-1], 1.0)
    span = group_slice(config, group)
    error = jnp.square(prediction[..., span, :] - target[..., span, :])
    return jnp.sum(error * valid[..., span, None]) / denominator


def slot_group_weights(config: ContinuousBottleneckConfig, weights) -> jnp.ndarray:
    """Expand per-group weights to a per-slot vector under the configured layout."""
    if len(weights) != len(GROUP_NAMES):
        raise ValueError(
            f"expected one weight per group {GROUP_NAMES}, got {len(weights)}"
        )
    return jnp.concatenate([
        jnp.full((size,), float(weight), jnp.float32)
        for size, weight in zip(config.group_sizes, weights)
    ])


def masked_weighted_mse(prediction, target, valid_mask,
                        config: ContinuousBottleneckConfig, weights):
    """Per-group weighted MSE sharing the natural loss's global denominator.

    Because every group is divided by the *same* total-valid-element count, the
    four group terms partition ``masked_mse`` rather than re-normalising each
    group. Two consequences that make the weights readable:

    * all weights at 1 reproduces ``masked_mse`` exactly, so a weighted run is a
      controlled variation on the natural objective rather than a new one;
    * ``w_g`` is a direct per-element importance multiplier -- ``w_detail = 2``
      means a detail element's squared error counts twice an image element's,
      independent of how many slots each group happens to own.

    Per-group denominators would instead make the weights depend on group
    occupancy, so the same numbers would mean something different on data with a
    different slot mix.
    """
    prediction = jnp.asarray(prediction, jnp.float32)
    target = jnp.asarray(target, jnp.float32)
    valid = jnp.asarray(valid_mask, jnp.float32)
    if prediction.shape != target.shape or valid.shape != target.shape[:-1]:
        raise ValueError("prediction/target/mask shapes do not match")
    denominator = jnp.maximum(valid.sum() * target.shape[-1], 1.0)
    weighted = valid * slot_group_weights(config, weights)
    return jnp.sum(jnp.square(prediction - target) * weighted[..., None]) / denominator


def group_sse(prediction, target, valid_mask, config: ContinuousBottleneckConfig):
    """Sums, not means: batched evaluation must aggregate SSE, then divide once.

    Valid-slot counts differ between states, so averaging per-batch MSE values
    would silently weight small batches more heavily.
    """
    prediction = jnp.asarray(prediction, jnp.float32)
    target = jnp.asarray(target, jnp.float32)
    valid = jnp.asarray(valid_mask, jnp.float32)
    output = {}

    def accumulate(name, predicted, expected, mask):
        error = jnp.sum(jnp.square(predicted - expected) * mask[..., None])
        zero = jnp.sum(jnp.square(expected) * mask[..., None])
        output[f"{name}/sse"] = error
        output[f"{name}/zero_sse"] = zero
        output[f"{name}/valid_slots"] = jnp.sum(mask)
        output[f"{name}/valid_scalars"] = jnp.sum(mask) * target.shape[-1]

    accumulate("all", prediction, target, valid)
    start = 0
    for name, size in zip(GROUP_NAMES, config.group_sizes):
        stop = start + size
        accumulate(
            name,
            prediction[..., start:stop, :],
            target[..., start:stop, :],
            valid[..., start:stop],
        )
        start = stop
    return output


def metrics_from_sse(totals: dict[str, float]) -> dict[str, float]:
    """Turn accumulated SSE/counts into MSE, zero baseline and R²."""
    metrics: dict[str, float] = {}
    for name in ("all",) + GROUP_NAMES:
        scalars = max(float(totals[f"{name}/valid_scalars"]), 1.0)
        mse = float(totals[f"{name}/sse"]) / scalars
        zero = float(totals[f"{name}/zero_sse"]) / scalars
        metrics[f"{name}/mse"] = mse
        metrics[f"{name}/zero_mse"] = zero
        metrics[f"{name}/r2"] = 1.0 - mse / max(zero, 1e-12)
        metrics[f"{name}/valid_slots"] = float(totals[f"{name}/valid_slots"])
    return metrics


def group_metrics(prediction, target, valid_mask, config: ContinuousBottleneckConfig):
    """Single-shot convenience wrapper around :func:`group_sse`."""
    totals = {
        name: float(value)
        for name, value in group_sse(prediction, target, valid_mask, config).items()
    }
    return metrics_from_sse(totals)


def _entropy_stats(weights, axis_size, eps=1e-12):
    entropy = -jnp.sum(weights * jnp.log(weights + eps), axis=-1)
    return entropy, jnp.exp(entropy), math.log(max(axis_size, 2))


def encoder_diagnostics(attention, valid_mask, config: ContinuousBottleneckConfig):
    """Aggregate encoder cross-attention diagnostics (report-only unless noted).

    ``invalid_weight_sum`` and ``row_sum_max_error`` are integrity gates; entropy
    and per-group mass are diagnostic, since specialization may legitimately
    produce non-uniform attention.
    """
    valid = jnp.asarray(valid_mask, jnp.float32)
    weights = jnp.asarray(attention, jnp.float32).mean(axis=-3)  # (..., N, slots)
    row_sums = jnp.sum(weights, axis=-1)
    invalid = jnp.sum(weights * (1.0 - valid)[..., None, :])
    entropy, effective, _ = _entropy_stats(weights, config.num_input_slots)
    valid_slots = jnp.maximum(jnp.sum(valid, axis=-1), 1.0)
    normalized = entropy / jnp.log(jnp.maximum(valid_slots, 2.0))[..., None]

    output = {
        "encoder/invalid_weight_sum": invalid,
        "encoder/row_sum_max_error": jnp.max(jnp.abs(row_sums - 1.0)),
        "encoder/normalized_entropy": jnp.mean(normalized),
        "encoder/effective_slots": jnp.mean(effective),
        "encoder/effective_slot_fraction": jnp.mean(effective / valid_slots[..., None]),
    }
    start = 0
    for name, size in zip(GROUP_NAMES, config.group_sizes):
        stop = start + size
        output[f"encoder/mass_{name}"] = jnp.mean(
            jnp.sum(weights[..., start:stop], axis=-1)
        )
        start = stop
    # Query collapse: mean pairwise cosine between latent-query attention rows.
    normed = weights / jnp.maximum(
        jnp.linalg.norm(weights, axis=-1, keepdims=True), 1e-12
    )
    similarity = jnp.einsum("...qs,...rs->...qr", normed, normed)
    count = config.num_e_tokens
    off_diagonal = (jnp.sum(similarity, axis=(-2, -1)) - jnp.sum(
        jnp.diagonal(similarity, axis1=-2, axis2=-1), axis=-1
    ))
    pairs = max(count * (count - 1), 1)
    output["encoder/query_pairwise_cosine"] = jnp.mean(off_diagonal / pairs)
    return output


def decoder_diagnostics(attention, config: ContinuousBottleneckConfig):
    weights = jnp.asarray(attention, jnp.float32).mean(axis=-3)  # (64, N)
    row_sums = jnp.sum(weights, axis=-1)
    entropy, effective, _ = _entropy_stats(weights, config.num_e_tokens)
    token_mass = jnp.mean(weights, axis=-2)
    return {
        "decoder/row_sum_max_error": jnp.max(jnp.abs(row_sums - 1.0)),
        "decoder/normalized_entropy": jnp.mean(
            entropy / math.log(max(config.num_e_tokens, 2))
        ),
        "decoder/effective_e_tokens": jnp.mean(effective),
        "decoder/effective_e_token_fraction": jnp.mean(effective) / config.num_e_tokens,
        "decoder/min_token_mass": jnp.min(token_mass),
        "decoder/max_token_mass": jnp.max(token_mass),
    }


def attention_diagnostics(
    encoder_attention,
    decoder_weights,
    valid_mask,
    config: ContinuousBottleneckConfig,
):
    """Both sides at once.

    A0's diagonal/top-1 metrics are intentionally absent: encoder and decoder
    axes have different lengths in A1, so slot-to-slot identity is undefined.
    """
    output = encoder_diagnostics(encoder_attention, valid_mask, config)
    output.update(decoder_diagnostics(decoder_weights, config))
    return output


def latent_statistics(e_t):
    e_t = jnp.asarray(e_t, jnp.float32)
    return {
        "e_t/mean": jnp.mean(e_t),
        "e_t/std": jnp.std(e_t),
        "e_t/rms": jnp.sqrt(jnp.mean(jnp.square(e_t))),
        "e_t/max_abs": jnp.max(jnp.abs(e_t)),
    }
