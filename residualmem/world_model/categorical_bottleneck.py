"""A2 grouped categorical bottleneck primitives (JAX), product-quantized.

A1 sends a continuous ``r_t`` of shape ``(64, 512)`` from encoder to decoder. A2
adds exactly one new element -- discretization -- and changes nothing else::

    (xbar_t, m_t) -> r_t -> z_t -> e~_t -> xbar_hat_t

Each of the 64 latent tokens is split into ``M`` subspaces of ``e_dim // M``
dims; each subspace independently selects one of ``C`` codewords (product
quantization). With ``M=8, C=256`` a state is 64 tokens x 8 codes x 8 bits =
4096 bits, plus 64 external mask bits. That is a nominal 256x reduction against
the raw FP32 latent tensor (64*512*32 = 1,048,576 bits) -- a *tensor footprint*
comparison, not a measured bitrate: nothing here is entropy coded.

There is no prior, no KL, no transition, no action, no history and no time. The
quantizer is deterministic (argmax, no RNG), so training and evaluation take the
identical path and A2 adds only "discretization" as a variable, not stochasticity.

Two invariants the rest of the experiment leans on:

* **No continuous bypass.** The decoder only ever sees gathered codebook entries.
  ``quantize`` builds the forward value with a gather, never a matmul, so the
  bit-exactness of "decoder input == codebook entry" does not depend on matmul
  precision. The soft path exists only as a diagnostic and is never the
  authoritative reconstruction.
* **Codes are sufficient.** ``embed_codes`` reproduces the decoder input from the
  integer codes alone, bit-for-bit, which is what makes the 4096-bit claim real.

The codebook doubles as classification prototype and decoder embedding, so no
separate logit head exists. A consequence worth stating: because ``probs``
depends on the codebook through the distances, *every* codeword receives
gradient through the softmax path, not just the selected one. That alignment
pressure is what replaces VQ-VAE's explicit commitment/codebook losses here, and
it is also why a temporarily unused codeword can still recover.
"""
from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np

from residualmem.world_model.continuous_bottleneck import (
    DEFAULT_GROUP_SIZES,
    GROUP_NAMES,
    ContinuousBottleneckConfig,
    attention_diagnostics,
    decode,
    decoder_attention,
    encode,
    group_metrics,
    group_slice,
    group_sse,
    initialize_params as initialize_continuous_params,
    masked_group_mse,
    masked_mse,
    masked_weighted_mse,
    metrics_from_sse,
    slot_group_weights,
)

__all__ = [
    "GROUP_NAMES",
    "PROTOCOL",
    "FORMAT_VERSION",
    "CODEBOOK",
    "SELECTOR_WEIGHT",
    "SELECTOR_BIAS",
    "SELECTOR_MIXER",
    "EMBEDDING_TABLE",
    "ASSIGNMENTS",
    "CategoricalBottleneckConfig",
    "initialize_params",
    "expected_param_shapes",
    "parameter_count",
    "quantize",
    "embed_codes",
    "learned_params_from_pq",
    "reconstruct",
    "calibrate_temperature",
    "code_histogram",
    "code_health",
    "attention_diagnostics",
    "decode",
    "decoder_attention",
    "encode",
    "group_metrics",
    "group_slice",
    "group_sse",
    "masked_group_mse",
    "masked_mse",
    "masked_weighted_mse",
    "metrics_from_sse",
    "slot_group_weights",
]

PROTOCOL = "a2_categorical_bottleneck_v1"
FORMAT_VERSION = 1
CODEBOOK = "quantizer/codebook"
SELECTOR_WEIGHT = "selector/weight"
SELECTOR_BIAS = "selector/bias"
SELECTOR_MIXER = "selector/mixer"
EMBEDDING_TABLE = "embedding/table"
ASSIGNMENTS = ("pq", "learned_slice", "learned_mix", "learned_full")


@dataclasses.dataclass(frozen=True)
class CategoricalBottleneckConfig:
    """A1 geometry plus the categorical-assignment knobs.

    ``num_e_tokens`` is the same axis A1 calls latent tokens; A2 does not
    introduce a second name for it.

    ``assignment`` picks how a category is chosen:

    ``pq``
        Nearest prototype under Euclidean distance, with the prototype table
        doubling as the decoder embedding. Geometry-driven.
    ``learned_slice``
        A free linear classifier over the subspace's own slice, decoupled from
        the decoder embedding.
    ``learned_mix``
        A single shared ``e_dim x e_dim`` mixer rotates the latent basis before
        the same slice-restricted classifiers run. Each group therefore sees an
        arbitrary linear combination of the whole token, but the mixer is
        estimated from every token and group at once, so it costs ~0.26M
        parameters instead of the ~16.8M a per-category full view would.
    ``learned_full``
        Every ``(token, group, category)`` gets its own full-width weight
        vector. Maximally expressive and maximally parameter-hungry.

    The two learned modes have no ``temperature``: at equivalence
    initialisation it is folded into the weights and bias, and thereafter the
    logit scale is free.
    """

    num_input_slots: int = 64
    input_dim: int = 512
    num_e_tokens: int = 64
    e_dim: int = 512
    num_heads: int = 8
    ffn_hidden: int = 1024
    # Default only; real runs inherit this from the A1 checkpoint. Kept in sync
    # with the shared layout so a run that omits the checkpoint does not silently
    # slice a (32,16,16,0) store as if it were (32,12,16,4).
    group_sizes: tuple[int, ...] = DEFAULT_GROUP_SIZES
    num_subspaces: int = 8
    num_categories: int = 256
    temperature: float = 1.0
    assignment: str = "pq"

    def __post_init__(self):
        object.__setattr__(self, "group_sizes", tuple(int(s) for s in self.group_sizes))
        # Reuse A1's validation for the shared geometry.
        self.continuous
        if self.num_subspaces < 1:
            raise ValueError("num_subspaces must be positive")
        if self.e_dim % self.num_subspaces:
            raise ValueError("e_dim must be divisible by num_subspaces")
        if self.num_categories < 2:
            raise ValueError("num_categories must be at least 2")
        if not self.temperature > 0:
            raise ValueError("temperature must be positive")
        if self.assignment not in ASSIGNMENTS:
            raise ValueError(f"assignment must be one of {ASSIGNMENTS}")

    @property
    def is_learned(self) -> bool:
        return self.assignment != "pq"

    @property
    def selector_dim(self) -> int:
        """Input width of the categorical classifier."""
        if self.assignment == "learned_full":
            return self.e_dim
        return self.subspace_dim


    @property
    def continuous(self) -> ContinuousBottleneckConfig:
        return ContinuousBottleneckConfig(
            num_input_slots=self.num_input_slots,
            input_dim=self.input_dim,
            num_e_tokens=self.num_e_tokens,
            e_dim=self.e_dim,
            num_heads=self.num_heads,
            ffn_hidden=self.ffn_hidden,
            group_sizes=self.group_sizes,
        )

    @property
    def subspace_dim(self) -> int:
        return self.e_dim // self.num_subspaces

    @property
    def codes_per_state(self) -> int:
        return self.num_e_tokens * self.num_subspaces

    @property
    def code_bits(self) -> float:
        """Nominal payload bits; the 64 mask bits are NOT counted here."""
        return self.codes_per_state * math.log2(self.num_categories)

    @property
    def latent_tensor_bits(self) -> int:
        return self.num_e_tokens * self.e_dim * 32

    @property
    def tensor_reduction(self) -> float:
        """Against the raw FP32 latent tensor. Not a measured bitrate."""
        return self.latent_tensor_bits / self.code_bits


def initialize_params(config: CategoricalBottleneckConfig, seed: int = 0) -> dict:
    """A1 parameters plus the assignment-specific tables.

    ``pq`` gets a single codebook that doubles as prototype and embedding; the
    runner overwrites it with per-subspace K-means centroids fitted on the train
    split only. The learned modes get a separate classifier and embedding, which
    the runner overwrites with the equivalence initialisation from a PQ
    checkpoint.
    """
    params = dict(initialize_continuous_params(config.continuous, seed))
    key = jax.random.PRNGKey(seed + 1)
    tokens, subspaces = config.num_e_tokens, config.num_subspaces
    categories, dim = config.num_categories, config.subspace_dim
    if not config.is_learned:
        params[CODEBOOK] = jax.random.normal(
            key, (tokens, subspaces, categories, dim), jnp.float32
        ) * 0.02
        return params
    weight_key, embed_key = jax.random.split(key)
    # Selector and embedding are independent: nothing requires the latent to sit
    # near the codeword the decoder will receive, so there is no commitment term
    # and no notion of quantization error in these modes.
    params[SELECTOR_WEIGHT] = jax.random.normal(
        weight_key, (tokens, subspaces, categories, config.selector_dim), jnp.float32
    ) * (1.0 / math.sqrt(config.selector_dim))
    params[SELECTOR_BIAS] = jnp.zeros((tokens, subspaces, categories), jnp.float32)
    if config.assignment == "learned_mix":
        # Identity start: the mixer only departs from the original basis if
        # reconstruction asks it to.
        params[SELECTOR_MIXER] = jnp.eye(config.e_dim, dtype=jnp.float32)
    params[EMBEDDING_TABLE] = jax.random.normal(
        embed_key, (tokens, subspaces, categories, dim), jnp.float32
    ) * 0.02
    return params


def expected_param_shapes(config: CategoricalBottleneckConfig) -> dict[str, tuple[int, ...]]:
    return {name: tuple(value.shape) for name, value in initialize_params(config, 0).items()}


def parameter_count(params) -> int:
    return int(sum(int(jnp.asarray(value).size) for value in params.values()))


def _split_subspaces(values, config: CategoricalBottleneckConfig):
    return values.reshape(
        values.shape[:-1] + (config.num_subspaces, config.subspace_dim)
    )


def _merge_subspaces(values, config: CategoricalBottleneckConfig):
    return values.reshape(values.shape[:-2] + (config.e_dim,))


def _gather_codewords(codebook, codes, config: CategoricalBottleneckConfig):
    """Exact lookup ``B[i, m, codes[..., i, m]]`` via a gather, never a matmul.

    A matmul against a one-hot would be at the mercy of the matmul precision
    (TF32 would round the codeword itself), which would quietly break the
    "decoder sees exactly a codebook entry" invariant.
    """
    tokens, subspaces = config.num_e_tokens, config.num_subspaces
    categories = config.num_categories
    flat = codebook.reshape(tokens * subspaces * categories, config.subspace_dim)
    offsets = (
        jnp.arange(tokens)[:, None] * (subspaces * categories)
        + jnp.arange(subspaces)[None, :] * categories
    )
    return flat[offsets + codes]


def _embedding_table(params, config: CategoricalBottleneckConfig):
    """The table the decoder reads from, whichever assignment mode is active."""
    return params[EMBEDDING_TABLE] if config.is_learned else params[CODEBOOK]


def _pq_logits(params, sub, config: CategoricalBottleneckConfig):
    """Negative squared distance to each prototype, scaled by temperature.

    ``||r - B||^2 = ||r||^2 + ||B||^2 - 2 r.B``. The expanded form is mandatory:
    the naive broadcast difference would materialise ``(batch, N, M, C, dsub)``,
    ~1 GiB at batch 32, while this peaks at ``(batch, N, M, C)`` ~ 17 MiB.
    """
    codebook = params[CODEBOOK]
    cross = jnp.einsum("...imd,imcd->...imc", sub, codebook)
    distance = (
        jnp.sum(jnp.square(sub), -1)[..., None]
        + jnp.sum(jnp.square(codebook), -1)
        - 2.0 * cross
    )
    return -distance / config.temperature, jnp.maximum(distance, 0.0)


def _learned_logits(params, r, sub, config: CategoricalBottleneckConfig):
    """A free linear classifier: which category best serves reconstruction?

    ``learned_full`` reads the whole latent token per category. ``learned_mix``
    instead rotates the basis once with a shared mixer and keeps the per-group
    slice restriction -- each group still sees an arbitrary combination of the
    whole token, but the combination is learned from every token and group at
    once rather than separately for each category.
    """
    weight = params[SELECTOR_WEIGHT]
    if config.assignment == "learned_full":
        logits = jnp.einsum("...id,imcd->...imc", r, weight)
    else:
        if config.assignment == "learned_mix":
            mixed = jnp.einsum("...id,de->...ie", r, params[SELECTOR_MIXER])
            sub = _split_subspaces(mixed, config)
        logits = jnp.einsum("...imd,imcd->...imc", sub, weight)
    return logits + params[SELECTOR_BIAS]


def quantize(
    params,
    r,
    config: CategoricalBottleneckConfig,
    *,
    soft=False,
    return_diagnostics=False,
):
    """r_t -> (e~_t, z_t). Forward is exactly a gathered table entry.

    Gradients match the textbook straight-through estimator
    ``st = probs + sg(hard - probs)``: ``d e~/d probs = E`` and ``d e~/d E`` only
    reaches the selected entries. The difference is that the forward value here
    is bit-exact, because ``soft - sg(soft)`` is identically ``0.0`` and the hard
    term is a gather.

    In the learned modes the selector and the embedding are separate parameters,
    so the reconstruction gradient splits cleanly: through ``hard`` it teaches
    each category what to hand the decoder, through ``straight`` it teaches the
    classifier when to pick that category.
    """
    r = jnp.asarray(r, jnp.float32)
    if r.shape[-2:] != (config.num_e_tokens, config.e_dim):
        raise ValueError(f"unexpected latent shape: {r.shape}")
    table = _embedding_table(params, config)
    sub = _split_subspaces(r, config)  # (..., N, M, dsub)

    distance = None
    if config.is_learned:
        logits = _learned_logits(params, r, sub, config)
    else:
        logits, distance = _pq_logits(params, sub, config)
    probs = jax.nn.softmax(logits, -1)
    codes = jnp.argmax(logits, -1)  # (..., N, M)

    if soft:
        # Diagnostic path only. Never the authoritative reconstruction: a full
        # probability simplex per subspace would be a continuous side channel
        # carrying far more than log2(C) bits.
        sub_hat = jnp.einsum("...imc,imcd->...imd", probs, table)
    else:
        hard = _gather_codewords(table, codes, config)
        straight = jnp.einsum(
            "...imc,imcd->...imd", probs, jax.lax.stop_gradient(table)
        )
        sub_hat = hard + (straight - jax.lax.stop_gradient(straight))

    e_tilde = _merge_subspaces(sub_hat, config)
    if not return_diagnostics:
        return e_tilde, codes
    top2 = jnp.sort(logits, axis=-1)[..., -2:]
    diagnostics = {
        "probs": probs,
        "logits": logits,
        "max_prob": jnp.max(probs, -1),
        "top1_top2_margin": top2[..., 1] - top2[..., 0],
    }
    if distance is not None:
        # Only meaningful under PQ, where the prototype the classifier compares
        # against is also the vector the decoder receives.
        diagnostics["selected_distance"] = jnp.min(distance, -1)
    return e_tilde, codes, diagnostics


def embed_codes(params, codes, config: CategoricalBottleneckConfig):
    """Rebuild the decoder input from integer codes alone (replay path)."""
    codes = jnp.asarray(codes, jnp.int32)
    if codes.shape[-2:] != (config.num_e_tokens, config.num_subspaces):
        raise ValueError(f"unexpected codes shape: {codes.shape}")
    table = _embedding_table(params, config)
    return _merge_subspaces(_gather_codewords(table, codes, config), config)


def learned_params_from_pq(pq_params, pq_config, config: CategoricalBottleneckConfig):
    """Exact-equivalence initialisation: reproduce PQ's decisions on step 0.

    PQ's logits are already a linear classifier::

        -||r_g - B_c||^2 / tau
            = (2 B_c.r_g - ||B_c||^2) / tau  -  ||r_g||^2 / tau

    The trailing term is identical across categories, and both argmax and
    softmax are invariant to a constant shift, so dropping it is exact rather
    than merely argmax-preserving. Setting ``W = 2B/tau``, ``b = -||B||^2/tau``
    and ``E = B`` therefore makes the learned model pick the same codes and hand
    the decoder the same vectors as the PQ model it starts from.

    For ``learned_full`` the weight is zero outside the subspace's own slice, so
    the cross-dimensional view is available but unused until training opens it.
    """
    if not config.is_learned:
        raise ValueError("target config must use a learned assignment")
    for field in ("num_e_tokens", "e_dim", "num_subspaces", "num_categories"):
        if getattr(pq_config, field) != getattr(config, field):
            raise ValueError(
                f"PQ source and learned target disagree on {field}: "
                f"{getattr(pq_config, field)} != {getattr(config, field)}"
            )
    codebook = jnp.asarray(pq_params[CODEBOOK], jnp.float32)
    tokens, subspaces = config.num_e_tokens, config.num_subspaces
    categories, dim = config.num_categories, config.subspace_dim

    slice_weight = 2.0 * codebook / pq_config.temperature
    bias = -jnp.sum(jnp.square(codebook), -1) / pq_config.temperature
    if config.assignment == "learned_full":
        weight = jnp.zeros((tokens, subspaces, categories, config.e_dim), jnp.float32)
        for subspace in range(subspaces):
            start = subspace * dim
            weight = weight.at[:, subspace, :, start:start + dim].set(
                slice_weight[:, subspace]
            )
    else:
        weight = slice_weight

    params = {name: value for name, value in pq_params.items() if name != CODEBOOK}
    params[SELECTOR_WEIGHT] = weight
    params[SELECTOR_BIAS] = bias
    if config.assignment == "learned_mix":
        # Identity mixer: step 0 reads the original slices, exactly as PQ did.
        params[SELECTOR_MIXER] = jnp.eye(config.e_dim, dtype=jnp.float32)
    params[EMBEDDING_TABLE] = codebook
    return params


def reconstruct(
    params,
    xbar,
    valid_mask,
    config: CategoricalBottleneckConfig,
    *,
    soft=False,
    return_attention=False,
    return_codes=False,
    return_diagnostics=False,
):
    """Full A2 path: xbar_t -> r_t -> z_t -> e~_t -> xbar_hat_t."""
    continuous = config.continuous
    r, encoder_weights = encode(params, xbar, valid_mask, continuous, return_attention=True)
    quantized = quantize(params, r, config, soft=soft, return_diagnostics=return_diagnostics)
    if return_diagnostics:
        e_tilde, codes, diagnostics = quantized
    else:
        e_tilde, codes = quantized
        diagnostics = None
    xbar_hat = decode(params, e_tilde, valid_mask, continuous)

    if not (return_attention or return_codes or return_diagnostics):
        return xbar_hat
    extras = [xbar_hat]
    if return_attention:
        extras.append({
            "encoder": encoder_weights,
            "decoder": decoder_attention(params, continuous),
        })
    if return_codes:
        extras.append(codes)
    if return_diagnostics:
        diagnostics["latent"] = r
        diagnostics["quantized"] = e_tilde
        extras.append(diagnostics)
    return tuple(extras)


def calibrate_temperature(
    params,
    r,
    config: CategoricalBottleneckConfig,
    *,
    target=0.8,
    low=1e-4,
    high=1e6,
    iterations=60,
):
    """Bisect tau so the median max posterior probability hits ``target``.

    Calibrating on ``max_prob`` rather than the top-1/top-2 margin matters: the
    margin only reflects the runner-up, while ``max_prob`` is set by all ``C-1``
    competitors. ``max_prob`` decreases monotonically in tau, so bisection is safe.

    Too small a tau saturates the softmax and ``d probs/d logits -> 0`` starves
    both the encoder and the codebook (the assignment-level straight-through has
    no identity gradient path, unlike VQ-VAE's vector STE). Too large and the
    straight-through bias dominates.
    """
    codebook = params[CODEBOOK]
    sub = _split_subspaces(jnp.asarray(r, jnp.float32), config)
    cross = jnp.einsum("...imd,imcd->...imc", sub, codebook)
    distance = (
        jnp.sum(jnp.square(sub), -1)[..., None]
        + jnp.sum(jnp.square(codebook), -1)
        - 2.0 * cross
    )
    # Shift-invariant: only gaps to the nearest codeword matter.
    gaps = distance - jnp.min(distance, -1, keepdims=True)

    def median_max_prob(temperature):
        weights = jnp.exp(-gaps / temperature)
        return float(jnp.median(1.0 / jnp.sum(weights, -1)))

    history = []
    lowest, highest = low, high
    for _ in range(iterations):
        middle = math.sqrt(low * high)  # geometric bisection over a wide range
        value = median_max_prob(middle)
        history.append({"temperature": middle, "median_max_prob": value})
        if value > target:
            low = middle
        else:
            high = middle
    temperature = math.sqrt(low * high)
    weights = jnp.exp(-gaps / temperature)
    max_prob = 1.0 / jnp.sum(weights, -1)
    achieved = float(jnp.median(max_prob))
    # Bracket exhaustion must be reported, not silently returned as a bound: it
    # means no temperature reaches the target, which happens when the codebook is
    # degenerate (duplicate centroids leave a near-zero nearest/runner-up gap).
    hit_bound = (
        temperature <= lowest * 1.01 or temperature >= highest * 0.99
    )
    return temperature, {
        "target_median_max_prob": target,
        "achieved_median_max_prob": achieved,
        "max_prob_p5": float(jnp.percentile(max_prob, 5)),
        "max_prob_p95": float(jnp.percentile(max_prob, 95)),
        "search_low": lowest,
        "search_high": highest,
        "hit_search_bound": bool(hit_bound),
        "converged": bool(not hit_bound and abs(achieved - target) <= 0.05),
        "iterations": iterations,
        "history_tail": history[-3:],
    }


def code_histogram(codes, config: CategoricalBottleneckConfig):
    """Per ``(token, subspace)`` category counts, shape ``(N, M, C)``, int64.

    Host-side and exact: counts are accumulated across batches by the runner, and
    the numerics protocol forbids float32 reductions for reported quantities.
    """
    codes = np.asarray(codes, np.int64)
    if codes.shape[-2:] != (config.num_e_tokens, config.num_subspaces):
        raise ValueError(f"unexpected codes shape: {codes.shape}")
    if codes.size and (codes.min() < 0 or codes.max() >= config.num_categories):
        raise ValueError("codes outside [0, num_categories)")
    flat = codes.reshape(-1, config.num_e_tokens * config.num_subspaces)
    offsets = np.arange(flat.shape[1], dtype=np.int64) * config.num_categories
    counts = np.bincount(
        (flat + offsets).ravel(),
        minlength=config.num_e_tokens * config.num_subspaces * config.num_categories,
    )
    return counts.reshape(
        config.num_e_tokens, config.num_subspaces, config.num_categories
    ).astype(np.int64)


def code_health(histogram, config: CategoricalBottleneckConfig) -> dict[str, float]:
    """Codebook usage summary in float64. Reported, never gated.

    Uniform category usage is not a goal, so none of these become pass/fail
    criteria; they exist to reveal collapse.
    """
    histogram = np.asarray(histogram, np.float64)
    total = np.maximum(histogram.sum(-1, keepdims=True), 1.0)
    frequency = histogram / total
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(frequency > 0, frequency * np.log2(frequency), 0.0)
    entropy = -terms.sum(-1)
    perplexity = np.power(2.0, entropy)
    active = (histogram > 0).sum(-1)
    dominant = frequency.max(-1)

    def spread(name, values):
        values = np.asarray(values, np.float64)
        return {
            f"code/{name}_p5": float(np.percentile(values, 5)),
            f"code/{name}_median": float(np.median(values)),
            f"code/{name}_p95": float(np.percentile(values, 95)),
        }

    output = {
        "code/subspaces": float(config.codes_per_state),
        "code/categories": float(config.num_categories),
        "code/bits_per_state": float(config.code_bits),
    }
    output.update(spread("perplexity", perplexity))
    output.update(spread("active_categories", active))
    output.update(spread("dominant_share", dominant))
    output["code/active_categories_min"] = float(active.min())
    output["code/dominant_share_max"] = float(dominant.max())
    output["code/collapsed_subspaces"] = float((active <= 1).sum())
    return output
