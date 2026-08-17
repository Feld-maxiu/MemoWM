"""Probabilistic latent world model (technical-report Section 6): an RSSM with a
deterministic recurrent core, a grouped categorical stochastic state, and a
prior / posterior / decoder head trio.

Written in the same hand-rolled functional-JAX style as ``gru.py`` (param dict +
``lax.scan``) rather than flax, to match the existing codebase and stay
CPU-friendly. A causal-Transformer core is left as a later ablation, mirroring how
the v0.3 line kept the Transformer optional.

Notation (report Eqs. 11-15):
    h_t   deterministic state    h_t = f_theta(h_{t-1}, z_{t-1}, u_{t-1}, d)
    p_t   prior    p_theta(z_t | h_t)              -- used by the codec at decode time
    q_t   posterior q_phi(z_t | h_t, xbar_t, m_t) -- produces categorical code z_t
    xbar_hat decoder D_omega(h_t, z_t)             -- reconstructs normalized xbar_t
z_t is N groups of C categories. Only the decoder-side state advances the next
step; the codec never lets an unsent posterior code bypass the transition.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class RSSMConfig:
    num_groups: int = 8          # N
    num_categories: int = 16     # C
    x_dim: int = 40              # flattened width of x_t (num_tokens * token_dim)
    hidden_size: int = 256
    embed_size: int = 128        # z embedding dim
    domain_embed: int = 16
    post_hidden: int = 128
    dec_hidden: int = 256
    action_values: int = 17
    num_domains: int = 5
    update_bias: float = -1.0
    kl_beta: float = 1.0
    kl_free_bits: float = 0.1    # nats per group
    recon_weight: float = 1.0
    slot_posterior: bool = False
    state_num_tokens: int = 1
    state_token_dim: int = 40
    posterior_heads: int = 4
    recon_group_sizes: tuple[int, ...] = ()
    recon_group_weights: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "recon_group_sizes", tuple(self.recon_group_sizes))
        object.__setattr__(self, "recon_group_weights", tuple(self.recon_group_weights))
        if self.state_num_tokens * self.state_token_dim != self.x_dim:
            raise ValueError("state_num_tokens * state_token_dim must equal x_dim")
        if self.slot_posterior:
            if self.posterior_heads < 1 or self.post_hidden % self.posterior_heads:
                raise ValueError("post_hidden must be divisible by posterior_heads")
            if self.state_num_tokens < 2:
                raise ValueError("slot posterior requires at least two state tokens")
        if self.recon_group_sizes:
            if sum(self.recon_group_sizes) != self.state_num_tokens:
                raise ValueError("reconstruction group sizes must sum to state_num_tokens")
            if self.recon_group_weights:
                if len(self.recon_group_weights) != len(self.recon_group_sizes):
                    raise ValueError("reconstruction group weights must match group sizes")
                if any(weight < 0 for weight in self.recon_group_weights):
                    raise ValueError("reconstruction group weights must be non-negative")
                if sum(self.recon_group_weights) <= 0:
                    raise ValueError("reconstruction group weights must sum positive")

    @property
    def latent_dim(self) -> int:
        return self.num_groups * self.num_categories


# --------------------------------------------------------------------------- init


def initialize_params(config: RSSMConfig, seed: int = 0) -> dict[str, jax.Array]:
    key = jax.random.PRNGKey(seed)
    keys = iter(jax.random.split(key, 64))
    h = config.hidden_size

    def weight(shape, scale=None):
        scale = scale or np.sqrt(2.0 / max(1, shape[0]))
        return jax.random.truncated_normal(next(keys), -2, 2, shape) * scale

    trans_in = config.embed_size + config.action_values + 1 + config.domain_embed
    params: dict[str, jax.Array] = {
        "domain/emb": weight((config.num_domains, config.domain_embed), scale=0.1),
        "zembed/w": weight((config.latent_dim, config.embed_size)),
        "zembed/b": jnp.zeros((config.embed_size,), jnp.float32),
        "trans_in/w": weight((trans_in, h)),
        "trans_in/b": jnp.zeros((h,), jnp.float32),
        "gru/norm_scale": jnp.ones((2 * h,), jnp.float32),
        "gru/w": weight((2 * h, 3 * h)),
        "gru/b": jnp.zeros((3 * h,), jnp.float32),
        "prior/w": weight((h, config.latent_dim), scale=0.01),
        "prior/b": jnp.zeros((config.latent_dim,), jnp.float32),
        "xenc/w": weight((config.x_dim, config.post_hidden)),
        "xenc/b": jnp.zeros((config.post_hidden,), jnp.float32),
        "post/w": weight((h + config.post_hidden, config.latent_dim), scale=0.01),
        "post/b": jnp.zeros((config.latent_dim,), jnp.float32),
        "dec/w1": weight((h + config.embed_size, config.dec_hidden)),
        "dec/b1": jnp.zeros((config.dec_hidden,), jnp.float32),
        "dec/w2": weight((config.dec_hidden, config.x_dim), scale=0.01),
        "dec/b2": jnp.zeros((config.x_dim,), jnp.float32),
    }
    if config.slot_posterior:
        params["xenc/w"] = weight((config.state_token_dim, config.post_hidden))
        params["post_q/w"] = weight((config.hidden_size, config.post_hidden))
        params["post_k/w"] = weight((config.post_hidden, config.post_hidden))
        params["post_v/w"] = weight((config.post_hidden, config.post_hidden))
        params["post_attn_out/w"] = weight((config.post_hidden, config.post_hidden))
    return params


# ------------------------------------------------------------------------- pieces


def _rms_norm(value, scale):
    return value * jax.lax.rsqrt(jnp.mean(jnp.square(value), -1, keepdims=True) + 1e-4) * scale


def _gru(params, h_prev, gin, update_bias):
    value = jnp.concatenate([h_prev, gin], -1)
    value = _rms_norm(value, params["gru/norm_scale"])
    value = value @ params["gru/w"] + params["gru/b"]
    reset, candidate, update = jnp.split(value, 3, -1)
    candidate = jnp.tanh(jax.nn.sigmoid(reset) * candidate)
    update = jax.nn.sigmoid(update + update_bias)
    return update * candidate + (1.0 - update) * h_prev


def embed_onehot(params, onehot, config: RSSMConfig):
    flat = onehot.reshape(onehot.shape[:-2] + (config.latent_dim,))
    return jax.nn.silu(flat @ params["zembed/w"] + params["zembed/b"])


def embed_codes(params, codes, config: RSSMConfig):
    onehot = jax.nn.one_hot(codes, config.num_categories, dtype=jnp.float32)
    return embed_onehot(params, onehot, config)


def transition(params, h_prev, zemb_prev, action, dt, domain_idx, config: RSSMConfig):
    a = jax.nn.one_hot(action, config.action_values, dtype=jnp.float32)
    dom = params["domain/emb"][domain_idx]
    dt = jnp.asarray(dt, jnp.float32).reshape((1,))
    inp = jnp.concatenate([zemb_prev, a, dt, dom], -1)
    gin = jax.nn.silu(inp @ params["trans_in/w"] + params["trans_in/b"])
    return _gru(params, h_prev, gin, config.update_bias)


def prior_logits(params, h, config: RSSMConfig):
    flat = h @ params["prior/w"] + params["prior/b"]
    return flat.reshape(flat.shape[:-1] + (config.num_groups, config.num_categories))


def posterior_logits(
    params, h, xbar_flat, config: RSSMConfig, valid_mask=None,
):
    """q_phi(z_t | h_t, xbar_t, m_t); z_t is reserved for categorical codes."""
    if config.slot_posterior:
        tokens = xbar_flat.reshape(
            xbar_flat.shape[:-1] + (config.state_num_tokens, config.state_token_dim)
        )
        if valid_mask is None:
            valid_mask = jnp.ones(tokens.shape[:-1], jnp.bool_)
        valid_mask = jnp.asarray(valid_mask, jnp.bool_)
        if valid_mask.shape != tokens.shape[:-1]:
            raise ValueError(
                f"valid mask shape {valid_mask.shape} does not match tokens {tokens.shape[:-1]}"
            )
        xh = jax.nn.silu(tokens @ params["xenc/w"] + params["xenc/b"])
        heads = config.posterior_heads
        head_dim = config.post_hidden // heads
        query = (h @ params["post_q/w"]).reshape(h.shape[:-1] + (heads, head_dim))
        keys = (xh @ params["post_k/w"]).reshape(
            xh.shape[:-1] + (heads, head_dim)
        )
        values = (xh @ params["post_v/w"]).reshape(
            xh.shape[:-1] + (heads, head_dim)
        )
        logits = jnp.einsum("...hd,...khd->...hk", query, keys) / np.sqrt(head_dim)
        logits = jnp.where(valid_mask[..., None, :], logits, -1e30)
        weights = jax.nn.softmax(logits, axis=-1)
        attended = jnp.einsum("...hk,...khd->...hd", weights, values)
        attended = attended.reshape(h.shape[:-1] + (config.post_hidden,))
        xh = attended @ params["post_attn_out/w"]
    else:
        xh = jax.nn.silu(xbar_flat @ params["xenc/w"] + params["xenc/b"])
    flat = jnp.concatenate([h, xh], -1) @ params["post/w"] + params["post/b"]
    return flat.reshape(flat.shape[:-1] + (config.num_groups, config.num_categories))


def decode_tokens(params, h, zemb, config: RSSMConfig):
    hidden = jax.nn.silu(jnp.concatenate([h, zemb], -1) @ params["dec/w1"] + params["dec/b1"])
    return hidden @ params["dec/w2"] + params["dec/b2"]


def reconstruction_loss(xbar_hat, xbar_t, valid_mask, config: RSSMConfig):
    """Masked reconstruction in normalized xbar space, optionally group-balanced."""
    if not config.slot_posterior:
        return jnp.mean(jnp.square(xbar_hat - xbar_t))
    predicted = xbar_hat.reshape((config.state_num_tokens, config.state_token_dim))
    target = xbar_t.reshape((config.state_num_tokens, config.state_token_dim))
    valid = jnp.asarray(valid_mask, jnp.float32)
    squared = jnp.square(predicted - target)
    if not config.recon_group_sizes:
        return jnp.sum(squared * valid[:, None]) / jnp.maximum(
            valid.sum() * config.state_token_dim, 1.0
        )
    group_losses = []
    start = 0
    for size in config.recon_group_sizes:
        stop = start + size
        group_valid = valid[start:stop]
        group_losses.append(
            jnp.sum(squared[start:stop] * group_valid[:, None])
            / jnp.maximum(group_valid.sum() * config.state_token_dim, 1.0)
        )
        start = stop
    losses = jnp.stack(group_losses)
    if config.recon_group_weights:
        weights = jnp.asarray(config.recon_group_weights, jnp.float32)
        weights = weights / weights.sum()
    else:
        counts = jnp.asarray(config.recon_group_sizes, jnp.float32)
        weights = counts / counts.sum()
    return jnp.sum(losses * weights)


def st_sample(logits, key):
    """Straight-through Gumbel-softmax hard sample. Returns (onehot, codes)."""
    gumbel = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + 1e-9) + 1e-9)
    soft = jax.nn.softmax(logits + gumbel, -1)
    codes = jnp.argmax(soft, -1)
    hard = jax.nn.one_hot(codes, logits.shape[-1], dtype=jnp.float32)
    onehot = hard + soft - jax.lax.stop_gradient(soft)  # straight-through
    return onehot, codes


def kl_grouped(q_logits, p_logits, free_bits):
    logq = jax.nn.log_softmax(q_logits, -1)
    logp = jax.nn.log_softmax(p_logits, -1)
    q = jnp.exp(logq)
    kl_per_group = jnp.sum(q * (logq - logp), -1)          # (..., N)
    kl_per_group = jnp.maximum(kl_per_group, free_bits)     # free bits
    return jnp.mean(jnp.sum(kl_per_group, -1))


# --------------------------------------------------------------- sequence (train)


def rssm_sequence(
    params, xbar_seq, actions, domain_idx, config: RSSMConfig, key,
    closed_loop: bool = False, valid_seq=None,
):
    """Scan one trajectory; return (recon_loss, kl_loss). Advances teacher-forced
    with the posterior sample, or (closed_loop) with the prior mode."""
    T = xbar_seq.shape[0]
    if valid_seq is None:
        valid_seq = jnp.ones((T, config.state_num_tokens), jnp.bool_)
    else:
        valid_seq = jnp.asarray(valid_seq, jnp.bool_)
        if valid_seq.shape != (T, config.state_num_tokens):
            raise ValueError(
                f"valid_seq must be {(T, config.state_num_tokens)}; got {valid_seq.shape}"
            )
    h0 = jnp.zeros((config.hidden_size,), jnp.float32)
    z0 = jnp.zeros((config.embed_size,), jnp.float32)
    prev_actions = jnp.concatenate([jnp.zeros((1,), jnp.int32), actions.astype(jnp.int32)])
    is_first = jnp.concatenate([jnp.ones((1,), jnp.float32), jnp.zeros((T - 1,), jnp.float32)])
    keys = jax.random.split(key, T)

    def step(carry, xs):
        h_prev, zemb_prev = carry
        xbar_t, valid_t, a_prev, first, k = xs
        if config.slot_posterior:
            xbar_t = (
                xbar_t.reshape((config.state_num_tokens, config.state_token_dim))
                * valid_t[:, None]
            ).reshape((config.x_dim,))
        h_trans = transition(params, h_prev, zemb_prev, a_prev, 1.0, domain_idx, config)
        h = jnp.where(first > 0.5, jnp.zeros_like(h_trans), h_trans)
        plog = prior_logits(params, h, config)
        qlog = posterior_logits(params, h, xbar_t, config, valid_t)
        onehot, codes = st_sample(qlog, k)
        zemb_post = embed_onehot(params, onehot, config)
        xbar_hat = decode_tokens(params, h, zemb_post, config)
        recon = reconstruction_loss(xbar_hat, xbar_t, valid_t, config)
        kl = kl_grouped(qlog, plog, config.kl_free_bits)
        if closed_loop:
            prior_codes = jnp.argmax(plog, -1)
            zemb_next = embed_codes(params, prior_codes, config)
        else:
            zemb_next = zemb_post
        return (h, zemb_next), (recon, kl)

    _, (recon, kl) = jax.lax.scan(
        step, (h0, z0), (xbar_seq, valid_seq, prev_actions, is_first, keys))
    return jnp.mean(recon), jnp.mean(kl)


def rssm_loss(
    params, xbar_seq, actions, domain_idx, config: RSSMConfig, key,
    closed_loop: bool = False, valid_seq=None,
):
    recon, kl = rssm_sequence(
        params, xbar_seq, actions, domain_idx, config, key,
        closed_loop, valid_seq,
    )
    return config.recon_weight * recon + config.kl_beta * kl


def latentize_codes(
    params, xbar_seq, actions, domain_idx, config: RSSMConfig, valid_seq=None,
):
    """Deterministic posterior encode -> categorical codes z_t (T, N)."""
    T = xbar_seq.shape[0]
    valid_seq = (
        jnp.ones((T, config.state_num_tokens), jnp.bool_)
        if valid_seq is None else jnp.asarray(valid_seq, jnp.bool_)
    )
    h0 = jnp.zeros((config.hidden_size,), jnp.float32)
    z0 = jnp.zeros((config.embed_size,), jnp.float32)
    prev_actions = jnp.concatenate([jnp.zeros((1,), jnp.int32), actions.astype(jnp.int32)])
    is_first = jnp.concatenate([jnp.ones((1,), jnp.float32), jnp.zeros((T - 1,), jnp.float32)])

    def step(carry, xs):
        h_prev, zemb_prev = carry
        xbar_t, valid_t, a_prev, first = xs
        if config.slot_posterior:
            xbar_t = (
                xbar_t.reshape((config.state_num_tokens, config.state_token_dim))
                * valid_t[:, None]
            ).reshape((config.x_dim,))
        h_trans = transition(params, h_prev, zemb_prev, a_prev, 1.0, domain_idx, config)
        h = jnp.where(first > 0.5, jnp.zeros_like(h_trans), h_trans)
        qlog = posterior_logits(params, h, xbar_t, config, valid_t)
        codes = jnp.argmax(qlog, -1)
        zemb = embed_codes(params, codes, config)
        return (h, zemb), codes

    _, codes = jax.lax.scan(
        step, (h0, z0), (xbar_seq, valid_seq, prev_actions, is_first)
    )
    return codes


def posterior_rollout(
    params, xbar_seq, actions, domain_idx, config: RSSMConfig, valid_seq=None,
):
    """Return categorical z_t codes and reconstructed normalized xbar_t."""
    T = xbar_seq.shape[0]
    valid_seq = (
        jnp.ones((T, config.state_num_tokens), jnp.bool_)
        if valid_seq is None else jnp.asarray(valid_seq, jnp.bool_)
    )
    h0 = jnp.zeros((config.hidden_size,), jnp.float32)
    z0 = jnp.zeros((config.embed_size,), jnp.float32)
    prev_actions = jnp.concatenate([jnp.zeros((1,), jnp.int32), actions.astype(jnp.int32)])
    is_first = jnp.concatenate([jnp.ones((1,), jnp.float32), jnp.zeros((T - 1,), jnp.float32)])

    def step(carry, xs):
        h_prev, zemb_prev = carry
        xbar_t, valid_t, a_prev, first = xs
        if config.slot_posterior:
            xbar_t = (
                xbar_t.reshape((config.state_num_tokens, config.state_token_dim))
                * valid_t[:, None]
            ).reshape((config.x_dim,))
        h_trans = transition(params, h_prev, zemb_prev, a_prev, 1.0, domain_idx, config)
        h = jnp.where(first > 0.5, jnp.zeros_like(h_trans), h_trans)
        qlog = posterior_logits(params, h, xbar_t, config, valid_t)
        codes = jnp.argmax(qlog, -1)
        zemb = embed_codes(params, codes, config)
        xbar_hat = decode_tokens(params, h, zemb, config)
        return (h, zemb), (codes, xbar_hat)

    _, (codes, xbar_hat) = jax.lax.scan(
        step, (h0, z0), (xbar_seq, valid_seq, prev_actions, is_first)
    )
    return codes, xbar_hat


# --------------------------------------------------------------------- checkpoint


def _content_hash(params, config: RSSMConfig, latent_schema_hash: bytes,
                  domain_name: str) -> bytes:
    digest = hashlib.sha256()
    digest.update(b"rssm-v04\0")
    digest.update(latent_schema_hash)
    digest.update(domain_name.encode("utf-8"))
    digest.update(json.dumps(dataclasses.asdict(config), sort_keys=True).encode())
    for name, value in sorted(params.items()):
        array = np.asarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.digest()


def save_rssm_checkpoint(path, params, config: RSSMConfig, latent_schema, domain_name):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps({
        "config": dataclasses.asdict(config),
        "latent_schema_hash": latent_schema.hash_hex,
        "latent_schema": json.loads(latent_schema.canonical_json()),
        "domain": domain_name,
    }, sort_keys=True)
    arrays = {name.replace("/", "__"): np.asarray(value) for name, value in params.items()}
    arrays["metadata"] = np.asarray(metadata)
    np.savez_compressed(path, **arrays)
    return path


def load_rssm_checkpoint(path):
    with np.load(Path(path), allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        config = RSSMConfig(**metadata["config"])
        params = {name.replace("__", "/"): data[name]
                  for name in data.files if name != "metadata"}
    return params, config, metadata
