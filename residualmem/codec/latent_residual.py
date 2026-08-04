"""Latent residual codec loop (report Sections 7.3-7.4, 12.1-12.2).

For one segment the encoder walks the RSSM forward: each step it forms the prior
``p_t``, the posterior code ``z^+_t``, chooses which mismatched groups to store via
a :class:`MaskPolicy`, entropy-codes the stored codes against the prior, and
advances the deterministic state with the *reconstructed* latent ``z_tilde_t``
(dropped groups filled with the prior mode). The decoder replays the identical
forward pass -- it never sees ``x_t`` -- reading the mask, decoding stored codes,
and filling the rest, so ``z_tilde`` is bit-identical on both sides. The anchor
(step 0) is intra-coded from the posterior at ``h=0``.

Exact-mask mode is the lossless special case: every mismatch is stored and the
reconstruction equals ``z^+`` exactly.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from ..world_model import rssm as R
from .entropy import RansDecoder, RansEncoder, quantize_freqs
from .send_mask import ExactMaskPolicy, MaskPolicy
from .varint import decode_uvarint, encode_uvarint


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class LatentStepper:
    """Jitted RSSM steps shared by encoder and decoder (guarantees identical h)."""

    def __init__(self, params, config: R.RSSMConfig, domain_idx: int):
        self.params = {k: jnp.asarray(v) for k, v in params.items()}
        self.config = config
        self.h0 = jnp.zeros((config.hidden_size,), jnp.float32)

        def enc_step(params, h, prev_codes, action, x):
            zemb = R.embed_codes(params, prev_codes, config)
            hn = R.transition(params, h, zemb, action, 1.0, domain_idx, config)
            return hn, R.prior_logits(params, hn, config), R.posterior_logits(params, hn, x, config)

        def dec_step(params, h, prev_codes, action):
            zemb = R.embed_codes(params, prev_codes, config)
            hn = R.transition(params, h, zemb, action, 1.0, domain_idx, config)
            return hn, R.prior_logits(params, hn, config)

        def anchor_post(params, x):
            return R.posterior_logits(params, self.h0, x, config)

        self._enc = jax.jit(enc_step)
        self._dec = jax.jit(dec_step)
        self._anchor = jax.jit(anchor_post)

        def render_step(params, h, codes):
            zemb = R.embed_codes(params, codes, config)
            return R.decode_tokens(params, h, zemb, config)

        self._render = jax.jit(render_step)

    def anchor_codes(self, x0) -> np.ndarray:
        return np.asarray(self._anchor(self.params, jnp.asarray(x0))).argmax(-1).astype(np.int64)

    def decode_x(self, h, codes) -> np.ndarray:
        """Reconstruct the token target x_tilde = D_omega(h, z_tilde) for rendering."""
        return np.asarray(self._render(self.params, h, jnp.asarray(np.asarray(codes), jnp.int32)))


def encode_latent_segment(stepper: LatentStepper, x_window: np.ndarray, actions: np.ndarray,
                          mask_policy: MaskPolicy, scale_bits: int = 16):
    """Return (anchor_codes, payload_bytes, recon_states) for one segment."""
    config = stepper.config
    x = jnp.asarray(np.asarray(x_window, np.float32))
    steps = x.shape[0]
    anchor = stepper.anchor_codes(x[0])
    h = stepper.h0
    prev = jnp.asarray(anchor, jnp.int32)
    recon = [anchor.copy()]
    enc = RansEncoder(scale_bits)
    records: list[tuple[int, list[int]]] = []
    last = 0
    for local in range(1, steps):
        h, plog, qlog = stepper._enc(
            stepper.params, h, prev, jnp.asarray(int(actions[local - 1])), x[local])
        plog = np.asarray(plog, np.float64)
        pargmax = plog.argmax(-1)
        zplus = np.asarray(qlog).argmax(-1)
        sent = mask_policy.select(plog, zplus, pargmax)
        ztil = pargmax.copy().astype(np.int64)
        if sent:
            probs = _softmax_rows(plog)
            for j in sent:
                ztil[j] = int(zplus[j])
                enc.encode(int(zplus[j]), quantize_freqs(probs[j], scale_bits))
            records.append((local - last, sent))
            last = local
        recon.append(ztil)
        prev = jnp.asarray(ztil, jnp.int32)
    entropy = enc.finish() if records else b""
    payload = _serialize(records, config.num_groups, entropy)
    return anchor, payload, recon


def decode_latent_segment(stepper: LatentStepper, anchor_codes: np.ndarray, actions: np.ndarray,
                          num_steps: int, payload: bytes, scale_bits: int = 16):
    """Replay a segment from its intra-coded anchor. Returns list of code arrays."""
    config = stepper.config
    record_map, entropy = _deserialize(payload, config.num_groups, num_steps)
    h = stepper.h0
    prev = jnp.asarray(np.asarray(anchor_codes, np.int32), jnp.int32)
    recon = [np.asarray(anchor_codes, np.int64)]
    rans = RansDecoder(entropy, scale_bits) if entropy else None
    for local in range(1, num_steps):
        h, plog = stepper._dec(stepper.params, h, prev, jnp.asarray(int(actions[local - 1])))
        plog = np.asarray(plog, np.float64)
        ztil = plog.argmax(-1).astype(np.int64)
        sent = record_map.get(local)
        if sent:
            probs = _softmax_rows(plog)
            for j in sent:
                ztil[j] = rans.decode(quantize_freqs(probs[j], scale_bits))
        recon.append(ztil)
        prev = jnp.asarray(ztil.astype(np.int32), jnp.int32)
    if rans is not None and rans.consumed != len(entropy):
        raise ValueError("entropy stream not fully consumed")
    return recon


def _serialize(records, num_groups: int, entropy: bytes) -> bytes:
    mask_bytes = (num_groups + 7) // 8
    buf = bytearray(encode_uvarint(len(records)))
    for delta, sent in records:
        buf += encode_uvarint(delta)
        mask = bytearray(mask_bytes)
        for j in sent:
            mask[j // 8] |= 1 << (j % 8)
        buf += mask
    buf += encode_uvarint(len(entropy))
    buf += entropy
    return bytes(buf)


def _deserialize(payload: bytes, num_groups: int, num_steps: int):
    mask_bytes = (num_groups + 7) // 8
    count, offset = decode_uvarint(payload)
    record_map: dict[int, list[int]] = {}
    absolute = 0
    for _ in range(count):
        delta, offset = decode_uvarint(payload, offset)
        if delta <= 0:
            raise ValueError("residual step deltas must be positive")
        absolute += delta
        if not 0 < absolute < num_steps:
            raise ValueError("residual step out of range")
        mask = payload[offset:offset + mask_bytes]
        if len(mask) != mask_bytes:
            raise EOFError("truncated residual mask")
        offset += mask_bytes
        sent = [j for j in range(num_groups) if mask[j // 8] & (1 << (j % 8))]
        if not sent:
            raise ValueError("residual record must store at least one group")
        record_map[absolute] = sent
    length, offset = decode_uvarint(payload, offset)
    entropy = payload[offset:offset + length]
    if len(entropy) != length:
        raise EOFError("truncated entropy stream")
    if offset + length != len(payload):
        raise ValueError("trailing residual bytes")
    return record_map, entropy
