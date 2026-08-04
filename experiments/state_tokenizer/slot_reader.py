"""One-query, one-cross-attention diagnostic readers for v2."""
from __future__ import annotations

import math

import torch
from torch import nn


class ProbeHeads(nn.Module):
    def __init__(
        self,
        hidden: int,
        *,
        dynamic: int,
        static: int,
        dom_words: int,
        overlap_words: int,
        tasks: int,
    ):
        super().__init__()
        self.dynamic = nn.Linear(hidden, dynamic)
        self.static = nn.Linear(hidden, static)
        self.dom_words = nn.Linear(hidden, dom_words) if dom_words else None
        self.overlap_words = nn.Linear(hidden, overlap_words) if overlap_words else None
        self.digits = nn.Linear(hidden, 5 * 10)
        self.task = nn.Linear(hidden, tasks)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        result = {
            "dynamic": self.dynamic(hidden),
            "static": self.static(hidden),
            "digits": self.digits(hidden).reshape(len(hidden), 5, 10),
            "task": self.task(hidden),
        }
        if self.dom_words is not None:
            result["dom_words"] = self.dom_words(hidden)
        if self.overlap_words is not None:
            result["overlap_words"] = self.overlap_words(hidden)
        return result


class FourierPositionEncoding(nn.Module):
    """Fixed 256-D encoding for normalized positions in (0,1)."""

    def __init__(self, hidden: int = 256, minimum_cycles: float = 1.0, maximum_cycles: float = 128.0):
        super().__init__()
        if hidden % 2:
            raise ValueError("position encoding width must be even")
        frequencies = torch.exp(torch.linspace(
            math.log(minimum_cycles), math.log(maximum_cycles), hidden // 2
        ))
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 2:
            raise ValueError(f"expected (batch,tokens), got {tuple(positions.shape)}")
        angles = 2 * math.pi * positions.unsqueeze(-1) * self.frequencies
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class SlotAwareReader(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        dynamic: int,
        static: int,
        dom_words: int,
        overlap_words: int,
        tasks: int,
        hidden: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.modality_embedding = nn.Embedding(3, hidden)
        self.position_encoding = FourierPositionEncoding(hidden)
        self.query = nn.Parameter(torch.empty(1, 1, hidden))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden)
        self.heads = ProbeHeads(
            hidden, dynamic=dynamic, static=static, dom_words=dom_words,
            overlap_words=overlap_words, tasks=tasks,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        modality_ids: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not bool(valid.any(dim=1).all()):
            raise ValueError("every sample must contain at least one valid token")
        values = self.input_projection(tokens)
        values = self.input_norm(values)
        values = values + self.modality_embedding(modality_ids)
        values = values + self.position_encoding(positions).to(values.dtype)
        query = self.query.expand(len(tokens), -1, -1)
        pooled, _ = self.attention(
            query, values, values, key_padding_mask=~valid, need_weights=False
        )
        return self.heads(self.output_norm(pooled[:, 0]))


class MetadataReader(nn.Module):
    """Small leakage baseline for task-ID and task+step metadata."""

    def __init__(
        self,
        input_dim: int,
        *,
        dynamic: int,
        static: int,
        dom_words: int,
        overlap_words: int,
        tasks: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden))
        self.heads = ProbeHeads(
            hidden, dynamic=dynamic, static=static, dom_words=dom_words,
            overlap_words=overlap_words, tasks=tasks,
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.heads(self.encoder(features))

