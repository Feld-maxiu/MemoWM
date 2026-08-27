"""Q-Former state tokenizer -- report Eq. (9), the learnable ``P_rho``.

Replaces the fixed pooling in ``static_key_pooling.build_static_key64``. That
pooler reduces the image band with ``F.adaptive_avg_pool2d`` onto a 4x8 spatial
grid, which on a 1280x720 screenshot averages 27.5 patch tokens per slot. Two
different web pages both carry a browser chrome band, a content area and a lot
of white, so averaging *by position* is the operation that removes page
identity: measured pairwise cosine on the image band is 0.5877 while that band
holds 74.3% of the state's energy.

Three things change here, and only the first is "a learned projection":

* queries address content, not grid position;
* **cross-attention is not causal.** In the pooled pipeline an image token's
  layer-16 state cannot see text that comes after it, so the image slots are
  structurally immune to the serializer -- and the observation prompt, which
  sits last in the sequence and owns zero slots, influences nothing at all.
  Every query here sees every position, prompt included;
* the slot budget follows content instead of a hardcoded (32, 16, 16, 0), under
  which the context band was measured valid in exactly 3 of its 16 slots.

The output keeps the pipeline's shape contract -- ``(K_x, 512)`` with ``K_x=64``
-- because ``_check_latents``, the A2 slot-position table and the reader
connector's ``rank_embedding`` all hard-require it. So this buys quality, not
compression; a smaller ``K_x`` would mean rewriting those three.

Ported from the JAX ``residualmem.latent.tokenizer.resample`` with latent
self-attention added: the JAX version is pure cross-attention, so its queries
can neither divide work nor avoid duplicating each other.
"""
from __future__ import annotations

import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

QFORMER_PROTOCOL = "qwen35_qformer_state_tokenizer_v1"


class FourierPositionEncoding(nn.Module):
    """Fixed encoding for normalized positions in (0,1).

    Lifted from ``experiments.state_tokenizer.slot_reader`` so this module does
    not import from ``experiments`` -- ``residualmem`` is the library side.
    """

    def __init__(self, hidden: int, minimum_cycles: float = 1.0,
                 maximum_cycles: float = 128.0) -> None:
        super().__init__()
        if hidden % 2:
            raise ValueError("position encoding width must be even")
        frequencies = torch.exp(torch.linspace(
            math.log(minimum_cycles), math.log(maximum_cycles), hidden // 2
        ))
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 2:
            raise ValueError(f"expected (batch, tokens), got {tuple(positions.shape)}")
        angles = 2 * math.pi * positions.unsqueeze(-1) * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class QFormerBlock(nn.Module):
    """Pre-norm cross-attention to ``H_t``, self-attention among queries, FFN."""

    def __init__(self, hidden: int, heads: int, mlp_ratio: int = 4,
                 self_attention: bool = False, qk_norm: bool = False) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden {hidden} must be divisible by heads {heads}")
        self.heads = heads
        self.head_dim = hidden // heads
        self.cross_norm = nn.LayerNorm(hidden)
        self.cross_q = nn.Linear(hidden, hidden, bias=False)
        self.cross_k = nn.Linear(hidden, hidden, bias=False)
        self.cross_v = nn.Linear(hidden, hidden, bias=False)
        self.cross_out = nn.Linear(hidden, hidden, bias=False)
        # Self-attention among the latents lets queries divide work -- and is
        # also a mixing operator: each query adds a weighted average of all the
        # others. Measured at 16 queries, holding everything else fixed, it is
        # the difference between an effective rank that falls to 4 and stays
        # there and one that dips to 10 and climbs back to 26, at identical
        # validation CE. So it costs 16.8M parameters to lose 6x the spread and
        # buys nothing, and it is off by default. The JAX ``resample()`` this is
        # ported from has no such sublayer either.
        self.self_attention = bool(self_attention)
        # Parameter-free RMS norm on q and k before the dot product. Off by
        # default and deliberately parameter-free: it adds nothing to the
        # state_dict, so every checkpoint trained without it still loads and the
        # same weights can be swept with it on and off.
        #
        # It exists because the amplification is localized and nothing else
        # explains it. Backward hooks put the gradient arriving at the latents
        # at ~1e-3 and the gradient leaving into the context at 1e13-inf: 16
        # orders of magnitude appear inside these four blocks, not in any loss
        # term, which is why every attempt to fix this by reweighting the
        # objective failed. It is confined to blocks.0's cross_q/cross_k, the
        # queries and the input side; blocks 1-3 are normal and cross_v never
        # appears -- its gradient is p^T @ grad_out with p a probability
        # distribution, so it cannot amplify, which places the blow-up on the
        # softmax jacobian side. Falsified on the way here: fp16 teacher cache,
        # massive trunk activations (|max| 73 on offenders and clean rows
        # alike), LayerNorm's 1/std (min variance 0.0971 vs 0.1002), the
        # position encoding (norm is a constant 22.627, independent of index),
        # the modality embedding (0.07-0.61 across every checkpoint, largest on
        # the arms that never diverged), and any dependence on the observation
        # weight (all ten arms have both survived and died).
        self.qk_norm = bool(qk_norm)
        if self.self_attention:
            self.self_norm = nn.LayerNorm(hidden)
            self.self_q = nn.Linear(hidden, hidden, bias=False)
            self.self_k = nn.Linear(hidden, hidden, bias=False)
            self.self_v = nn.Linear(hidden, hidden, bias=False)
            self.self_out = nn.Linear(hidden, hidden, bias=False)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * mlp_ratio),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden * mlp_ratio, hidden),
        )

    def _split(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def _merge(self, value: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = value.shape
        return value.transpose(1, 2).reshape(batch, length, self.heads * self.head_dim)

    def _scores(self, value: torch.Tensor) -> torch.Tensor:
        """RMS-normalize a split q or k over the head dimension, if enabled."""
        if not self.qk_norm:
            return value
        scale = torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return (value.float() * scale).to(value.dtype)

    def forward(self, latents, context, key_mask):
        normed = self.cross_norm(latents)
        attended = F.scaled_dot_product_attention(
            self._scores(self._split(self.cross_q(normed))),
            self._scores(self._split(self.cross_k(context))),
            self._split(self.cross_v(context)),
            # (batch, 1, 1, tokens): broadcasts over heads and queries.
            attn_mask=key_mask[:, None, None, :],
        )
        latents = latents + self.cross_out(self._merge(attended))

        if self.self_attention:
            normed = self.self_norm(latents)
            attended = F.scaled_dot_product_attention(
                self._scores(self._split(self.self_q(normed))),
                self._scores(self._split(self.self_k(normed))),
                self._split(self.self_v(normed)),
            )
            latents = latents + self.self_out(self._merge(attended))

        return latents + self.ffn(self.ffn_norm(latents))


class StateQFormer(nn.Module):
    """``H_t`` of any length -> ``(num_queries, output_dim)`` state tokens."""

    connector_type = QFORMER_PROTOCOL

    def __init__(
        self,
        *,
        num_queries: int = 16,
        output_dim: int = 512,
        input_dim: int = 4096,
        hidden: int = 1024,
        heads: int = 8,
        layers: int = 4,
        modalities: int = 3,
        self_attention: bool = False,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.output_dim = int(output_dim)
        self.input_dim = int(input_dim)
        self.self_attention = bool(self_attention)
        self.qk_norm = bool(qk_norm)
        self.queries = nn.Parameter(torch.empty(self.num_queries, hidden))
        nn.init.normal_(self.queries, std=0.02)
        self.input_projection = nn.Linear(input_dim, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.modality_embedding = nn.Embedding(modalities, hidden)
        nn.init.zeros_(self.modality_embedding.weight)
        self.position_encoding = FourierPositionEncoding(hidden)
        self.blocks = nn.ModuleList(
            QFormerBlock(hidden, heads, self_attention=self.self_attention,
                         qk_norm=self.qk_norm)
            for _ in range(layers)
        )
        self.output_projection = nn.Linear(hidden, self.output_dim, bias=False)
        self.output_norm = nn.LayerNorm(self.output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_ids: torch.Tensor,
        positions: torch.Tensor,
        key_mask: torch.Tensor,
    ) -> torch.Tensor:
        """``(B,T,input_dim)`` -> ``(B,num_queries,output_dim)``.

        ``key_mask`` is True on real tokens. It exists only to neutralize
        right-padding when several observations of different length share a
        batch -- there is no per-observation notion of an invalid input token,
        unlike the pooled path where unfilled detail/context slots were real.
        """
        if hidden_states.ndim != 3:
            raise ValueError(f"expected (batch, tokens, dim), got {tuple(hidden_states.shape)}")
        if hidden_states.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected width {self.input_dim}, got {hidden_states.shape[-1]}"
            )
        for name, value in (("modality_ids", modality_ids),
                            ("positions", positions), ("key_mask", key_mask)):
            if value.shape != hidden_states.shape[:2]:
                raise ValueError(
                    f"{name} must be {tuple(hidden_states.shape[:2])}, got {tuple(value.shape)}"
                )
        if not bool(key_mask.any(dim=1).all()):
            raise ValueError("every observation must contribute at least one token")

        dtype = self.input_projection.weight.dtype
        context = self.input_norm(self.input_projection(hidden_states.to(dtype)))
        context = context + self.modality_embedding(modality_ids)
        context = context + self.position_encoding(positions.to(dtype)).to(dtype)
        # Zero the padded positions as well as masking them. The mask alone is
        # enough for attention, but a padded row still reaches LayerNorm, and
        # leaving it live makes "output is invariant to padding" a property of
        # the mask rather than of the module.
        context = context * key_mask[..., None].to(dtype)

        latents = self.queries.to(dtype).expand(len(hidden_states), -1, -1)
        for block in self.blocks:
            latents = block(latents, context, key_mask)
        return self.output_norm(self.output_projection(latents))


class QFormerStateReader(nn.Module):
    """``P_rho`` plus the reader connector, trained jointly, saved as one artifact.

    They are one artifact because they are one decision: the resampler's output
    space is only meaningful to the connector that was trained against it, and
    the repo has already paid once for letting two halves of a coordinate system
    drift apart under identical file names (see ``pca_binding``).

    ``encode`` is the write side -- query-independent, so its output can be
    stored once per observation and reused for every question. That property is
    load-bearing: the WorldMemArena adapter encodes at ingest, caches the latent
    under a memory id, and never re-encodes. Conditioning this on the question
    would invalidate that whole storage model.
    """

    connector_type = "qwen35_qformer_reader_v1"

    def __init__(self, qformer: StateQFormer, connector: nn.Module,
                 retrieval_head: nn.Module | None = None) -> None:
        super().__init__()
        self.qformer = qformer
        self.connector = connector
        # Optional, and trained jointly when present. It is the same module A1
        # uses, so the semantic anchor and the retrieval head are one thing:
        # its attention pooling reduces the slots to a single vector *before*
        # any cosine, which is what keeps the anchor from asking every slot to
        # match the same 4096-d target and homogenizing them.
        self.retrieval_head = retrieval_head

    def project(self, xbar, valid):
        if self.retrieval_head is None:
            raise ValueError("this reader was built without a retrieval head")
        return self.retrieval_head(xbar, valid)

    def encode(self, hidden_states, modality_ids, positions, key_mask):
        """-> ``(xbar, valid)``. Every query is always valid, unlike the pooled
        path where unfilled detail and context slots were genuinely empty."""
        xbar = self.qformer(hidden_states, modality_ids, positions, key_mask)
        valid = torch.ones(xbar.shape[:-1], dtype=torch.bool, device=xbar.device)
        return xbar, valid

    def forward(self, hidden_states, modality_ids, positions, key_mask):
        xbar, valid = self.encode(hidden_states, modality_ids, positions, key_mask)
        return self.connector(xbar, valid), xbar, valid


def qformer_hash(module: nn.Module) -> str:
    """Content hash of the frozen weights, for the provenance chain.

    ``pca_binding`` exists because two identically-named artifact trees once
    silently produced wrong coordinates. A learned tokenizer needs the same
    guard; this replaces ``pca_sha256`` in the artifact metadata.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().to(torch.float32).cpu().numpy().tobytes())
    return digest.hexdigest()
