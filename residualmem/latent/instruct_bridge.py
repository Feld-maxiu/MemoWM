"""Frozen v9 Instruct latent bridges for retrieval and native QA.

The tokenizer output is always ``(xbar, valid)`` with shape ``64 x 512``.
Retrieval maps it into the official Qwen3-VL embedding space; reader bridges
map it into Qwen3.5-9B-Instruct's width.  No WorldMemArena sample is required
to train these modules.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


BRIDGE_CACHE_PROTOCOL = "qwen35_instruct_bridge_cache_v2"
FUSED_OBSERVATION_TEACHER_PROTOCOL = "qwen3_vl_fused_observation_v1"
BROWSERGYM_TEACHER_TEXT_PROTOCOL = "browsergym_task_plus_dom_v1"
RETRIEVAL_HEAD_TYPE = "qwen_vl_retrieval_v1"
RETRIEVAL_BRIDGE_PROTOCOL = "qwen35_instruct_retrieval_bridge_v1"
READER_BRIDGE_PROTOCOL = "qwen35_instruct_reader_bridge_v1"


def _check_latents(xbar: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xbar = torch.as_tensor(xbar, dtype=torch.float32)
    valid = torch.as_tensor(valid, dtype=torch.bool, device=xbar.device)
    if xbar.shape[-2:] != (64, 512):
        raise ValueError(f"expected xbar (...,64,512), got {tuple(xbar.shape)}")
    if valid.shape != xbar.shape[:-1]:
        raise ValueError(f"valid mask {tuple(valid.shape)} does not match {tuple(xbar.shape)}")
    if not torch.isfinite(xbar).all():
        raise ValueError("xbar contains non-finite values")
    return xbar, valid


def masked_softmax(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
    weights = torch.softmax(logits, dim=-1) * valid.to(logits.dtype)
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)


class MaskedAttentionRetrievalHead(nn.Module):
    """Shared Xbar/A2 head: masked attention pooling -> normalized 4096-D."""

    head_type = RETRIEVAL_HEAD_TYPE

    def __init__(self, input_dim: int = 512, output_dim: int = 4096) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_norm = nn.LayerNorm(input_dim)
        self.score = nn.Linear(input_dim, 1, bias=False)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim),
        )

    def forward(self, xbar: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        xbar, valid = _check_latents(xbar, valid)
        if not valid.any(dim=-1).all():
            raise ValueError("each state needs at least one valid slot")
        values = self.input_norm(xbar)
        weights = masked_softmax(self.score(values).squeeze(-1), valid)
        pooled = torch.sum(weights[..., None] * values, dim=-2)
        return F.normalize(self.projection(pooled), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scale = values.float().square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        return values * scale.to(values.dtype) * self.weight


class InputSoftTokenConnector(nn.Module):
    """Convert xbar slots to Qwen input embeddings (4096-D soft tokens)."""

    connector_type = "qwen35_input_soft_token_v1"

    def __init__(self, input_dim: int = 512, model_dim: int = 4096, slots: int = 64) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.output_norm = RMSNorm(model_dim)
        # Zero init preserves the meaning of state content at step zero; rank
        # identity is learned only when the reader objective needs it.
        self.rank_embedding = nn.Parameter(torch.zeros(slots, model_dim))

    def forward(self, xbar: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        xbar, valid = _check_latents(xbar, valid)
        output = self.output_norm(self.mlp(self.input_norm(xbar)))
        output = output + self.rank_embedding
        return output * valid[..., None]


class Layer16Restorer(nn.Module):
    """Invert train-only normalization and PCA512 back to layer-16 width."""

    def __init__(
        self,
        *,
        slot_mean: np.ndarray | torch.Tensor,
        slot_scale: np.ndarray | torch.Tensor,
        pca_mean: np.ndarray | torch.Tensor,
        components: np.ndarray | torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("slot_mean", torch.as_tensor(slot_mean, dtype=torch.float32))
        self.register_buffer("slot_scale", torch.as_tensor(slot_scale, dtype=torch.float32))
        self.register_buffer("pca_mean", torch.as_tensor(pca_mean, dtype=torch.float32))
        self.register_buffer("components", torch.as_tensor(components, dtype=torch.float32))
        if self.slot_mean.shape != (64, 512) or self.slot_scale.shape != (64, 512):
            raise ValueError("normalizer slot statistics must be (64,512)")
        if self.pca_mean.shape != (4096,) or self.components.shape != (4096, 512):
            raise ValueError("PCA artifact must contain mean (4096) and components (4096,512)")

    @classmethod
    def from_artifacts(cls, normalization: str | Path, pca: str | Path):
        with np.load(normalization, allow_pickle=False) as norm:
            layout = tuple(int(x) for x in norm["layout"])
            group_index = np.repeat(np.arange(len(layout)), layout)
            slot_mean = np.asarray(norm["mean"], np.float32)[group_index]
            slot_scale = np.asarray(norm["scale"], np.float32)[group_index]
        with np.load(pca, allow_pickle=False) as artifact:
            return cls(
                slot_mean=slot_mean,
                slot_scale=slot_scale,
                pca_mean=artifact["mean"],
                components=artifact["components"],
            )

    def forward(self, xbar: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        xbar, valid = _check_latents(xbar, valid)
        pca = (xbar * self.slot_scale + self.slot_mean) * valid[..., None]
        hidden = pca @ self.components.transpose(0, 1) + self.pca_mean
        return hidden * valid[..., None]


class Layer16Connector(nn.Module):
    """Restored layer-16 states plus a zero-initialized trainable correction."""

    connector_type = "qwen35_layer16_residual_v1"

    def __init__(self, restorer: Layer16Restorer) -> None:
        super().__init__()
        self.restorer = restorer
        self.adapter = nn.Linear(512, 4096, bias=False)
        nn.init.zeros_(self.adapter.weight)

    def forward(self, xbar: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        xbar, valid = _check_latents(xbar, valid)
        return (self.restorer(xbar, valid) + self.adapter(xbar)) * valid[..., None]


class Qwen35LatentReader:
    """Native Qwen3.5 answer path conditioned on retrieved latent states."""

    def __init__(self, model, processor, connector: nn.Module, *, mode: str) -> None:
        if mode not in {"input", "layer16"}:
            raise ValueError("reader mode must be 'input' or 'layer16'")
        self.model = model
        self.processor = processor
        self.connector = connector
        self.mode = mode

    def answer(
        self,
        question: str,
        states: list[tuple[np.ndarray, np.ndarray]],
        *,
        max_new_tokens: int = 96,
    ) -> str:
        if not states:
            return "Not mentioned in memory."
        device = next(self.model.parameters()).device
        xbar = torch.as_tensor(np.stack([row[0] for row in states]), device=device)
        valid = torch.as_tensor(np.stack([row[1] for row in states]), device=device)
        with torch.inference_mode():
            latent = self.connector(xbar, valid).reshape(1, -1, 4096)
        latent_valid = valid.reshape(1, -1)
        prompt = (
            "Answer the question from the retrieved memory-state tokens. Keep the answer concise. "
            "If absent, say exactly 'Not mentioned in memory.'.\nQuestion: " + question
        )
        encoded = self.processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        embeddings = self.model.get_input_embeddings()(input_ids)
        prefix_len = latent.shape[1]
        full_attention = torch.cat((latent_valid.to(attention.dtype), attention), dim=1)
        hook_handle = None
        if self.mode == "input":
            inputs_embeds = torch.cat((latent.to(embeddings.dtype), embeddings), dim=1)
        else:
            # Placeholder embeddings establish positions/cache.  At the input of
            # layer 17, the first-pass prefix is replaced by restored layer-16
            # states. Cached single-token decoding passes are left untouched.
            placeholders = torch.zeros_like(latent, dtype=embeddings.dtype)
            inputs_embeds = torch.cat((placeholders, embeddings), dim=1)
            target_layer = self.model.model.language_model.layers[16]

            def replace_prefix(_module, args):
                hidden = args[0]
                if hidden.shape[1] < prefix_len:
                    return None
                updated = hidden.clone()
                updated[:, :prefix_len] = latent.to(updated.dtype)
                return (updated, *args[1:])

            hook_handle = target_layer.register_forward_pre_hook(replace_prefix)
        try:
            with torch.inference_mode():
                generated = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=full_attention,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
        finally:
            if hook_handle is not None:
                hook_handle.remove()
        return self.processor.tokenizer.decode(generated[0], skip_special_tokens=True).strip()


class TorchA2Reconstructor(nn.Module):
    """Inference-only Torch port of the frozen JAX A2 path.

    This avoids importing JAX into the Qwen environment. Parameters are loaded
    losslessly from the A2 NPZ and the equations mirror
    ``categorical_bottleneck.reconstruct``.
    """

    def __init__(self, params: dict[str, np.ndarray], config: dict[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.config.setdefault("assignment", "pq")
        self._buffer_names: dict[str, str] = {}
        for index, (name, value) in enumerate(params.items()):
            buffer_name = f"p_{index:03d}"
            self.register_buffer(buffer_name, torch.as_tensor(value, dtype=torch.float32))
            self._buffer_names[name] = buffer_name

    @classmethod
    def from_checkpoint(cls, path: str | Path):
        with np.load(path, allow_pickle=False) as checkpoint:
            if "metadata" not in checkpoint.files:
                raise ValueError("A2 checkpoint has no metadata")
            metadata = json.loads(str(np.asarray(checkpoint["metadata"]).item()))
            if metadata.get("protocol") != "a2_categorical_bottleneck_v1":
                raise ValueError("not an A2 v1 checkpoint")
            config = dict(metadata["config"])
            config.setdefault("assignment", "pq")
            params = {
                name.replace("__", "/"): np.asarray(checkpoint[name], np.float32)
                for name in checkpoint.files if name != "metadata"
            }
        return cls(params, config)

    def _p(self, name: str) -> torch.Tensor:
        return getattr(self, self._buffer_names[name])

    @staticmethod
    def _ln(x, scale, bias):
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).square().mean(-1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + 1e-5) * scale + bias

    def _encode(self, xbar, valid):
        c = self.config
        heads = int(c["num_heads"])
        d_e = int(c["e_dim"])
        head_dim = d_e // heads
        safe = xbar * valid[..., None]
        key_input = self._ln(
            safe + self._p("encoder/input_positions"),
            self._p("encoder/memory_norm/scale"), self._p("encoder/memory_norm/bias"),
        )
        queries = self._p("encoder/e_queries").expand(xbar.shape[0], -1, -1)
        q_norm = self._ln(
            queries, self._p("encoder/q_norm/scale"), self._p("encoder/q_norm/bias")
        )
        q = (q_norm @ self._p("encoder/attn/q")).reshape(xbar.shape[0], -1, heads, head_dim)
        k = (key_input @ self._p("encoder/attn/k")).reshape(xbar.shape[0], -1, heads, head_dim)
        v = (safe @ self._p("encoder/attn/v")).reshape(xbar.shape[0], -1, heads, head_dim)
        logits = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(head_dim)
        mask = valid[:, None, None, :]
        attention = torch.softmax(logits.masked_fill(~mask, -1e30), -1) * mask
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-30)
        attended = torch.einsum("bhqk,bkhd->bqhd", attention, v).reshape(xbar.shape[0], -1, d_e)
        hidden = queries + attended @ self._p("encoder/attn/out")
        ffn_input = self._ln(
            hidden, self._p("encoder/ffn_norm/scale"), self._p("encoder/ffn_norm/bias")
        )
        ffn = F.gelu(
            ffn_input @ self._p("encoder/ffn/w1") + self._p("encoder/ffn/b1"),
            approximate="tanh",
        )
        return hidden + ffn @ self._p("encoder/ffn/w2") + self._p("encoder/ffn/b2")

    def _quantize(self, latent):
        c = self.config
        batch, tokens, width = latent.shape
        subspaces = int(c["num_subspaces"])
        subdim = width // subspaces
        sub = latent.reshape(batch, tokens, subspaces, subdim)
        assignment = str(c.get("assignment", "pq"))
        if assignment == "pq":
            table = self._p("quantizer/codebook")
            cross = torch.einsum("bimd,imcd->bimc", sub, table)
            distance = sub.square().sum(-1)[..., None] + table.square().sum(-1) - 2 * cross
            logits = -distance / float(c["temperature"])
        else:
            table = self._p("embedding/table")
            weight = self._p("selector/weight")
            if assignment == "learned_full":
                logits = torch.einsum("bid,imcd->bimc", latent, weight)
            else:
                classifier_input = sub
                if assignment == "learned_mix":
                    mixed = torch.einsum("bid,de->bie", latent, self._p("selector/mixer"))
                    classifier_input = mixed.reshape(batch, tokens, subspaces, subdim)
                logits = torch.einsum("bimd,imcd->bimc", classifier_input, weight)
            logits = logits + self._p("selector/bias")
        codes = logits.argmax(-1)
        expanded = table.unsqueeze(0).expand(batch, -1, -1, -1, -1)
        index = codes[..., None, None].expand(-1, -1, -1, 1, subdim)
        return torch.gather(expanded, 3, index).squeeze(3).reshape(batch, tokens, width)

    def _decode(self, latent, valid):
        c = self.config
        batch = latent.shape[0]
        heads = int(c["num_heads"])
        input_dim = int(c["input_dim"])
        head_dim = input_dim // heads
        queries_base = self._p("decoder/output_queries")
        q_norm = self._ln(
            queries_base, self._p("decoder/q_norm/scale"), self._p("decoder/q_norm/bias")
        )
        q = (q_norm @ self._p("decoder/attn/q")).reshape(-1, heads, head_dim)
        k = (self._p("decoder/e_addresses") @ self._p("decoder/attn/k")).reshape(-1, heads, head_dim)
        attention = torch.softmax(torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(head_dim), -1)
        v = (latent @ self._p("decoder/attn/v")).reshape(batch, -1, heads, head_dim)
        attended = torch.einsum("hqk,bkhd->bqhd", attention, v).reshape(batch, -1, input_dim)
        hidden = queries_base.expand(batch, -1, -1) + attended @ self._p("decoder/attn/out")
        ffn_input = self._ln(
            hidden, self._p("decoder/ffn_norm/scale"), self._p("decoder/ffn_norm/bias")
        )
        ffn = F.gelu(
            ffn_input @ self._p("decoder/ffn/w1") + self._p("decoder/ffn/b1"),
            approximate="tanh",
        )
        hidden = hidden + ffn @ self._p("decoder/ffn/w2") + self._p("decoder/ffn/b2")
        return (hidden @ self._p("decoder/output/w") + self._p("decoder/output/b")) * valid[..., None]

    def forward(self, xbar: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        xbar, valid = _check_latents(xbar, valid)
        return self._decode(self._quantize(self._encode(xbar, valid)), valid)


@dataclasses.dataclass(frozen=True)
class BridgeArtifact:
    artifact_type: str
    protocol: str
    metadata: dict[str, Any]


def save_bridge(path: str | Path, module: nn.Module, *, protocol: str, **metadata: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "protocol": protocol,
        "artifact_type": getattr(module, "head_type", getattr(module, "connector_type", type(module).__name__)),
        "metadata": metadata,
        "state_dict": module.state_dict(),
    }, path)


def load_bridge(path: str | Path, module: nn.Module, *, expected_protocol: str) -> BridgeArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("protocol") != expected_protocol:
        raise ValueError(f"bridge protocol {payload.get('protocol')!r} != {expected_protocol!r}")
    module.load_state_dict(payload["state_dict"], strict=True)
    return BridgeArtifact(
        artifact_type=str(payload.get("artifact_type", "")),
        protocol=str(payload["protocol"]),
        metadata=dict(payload.get("metadata") or {}),
    )


def cache_metadata_json(**extra: Any) -> str:
    return json.dumps({
        "protocol": BRIDGE_CACHE_PROTOCOL,
        "teacher_protocol": FUSED_OBSERVATION_TEACHER_PROTOCOL,
        "teacher_text_protocol": BROWSERGYM_TEACHER_TEXT_PROTOCOL,
        **extra,
    }, sort_keys=True)
