"""Qwen3-32B soft-token bridge and provenance-checked artifact format."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

QWEN32_INPUT_BRIDGE_PROTOCOL = "qwen32_input_soft_token_bridge_k32_v1"
QWEN32_SCALED_INPUT_BRIDGE_PROTOCOL = "qwen32_scaled_input_soft_token_bridge_k32_v2"
QWEN32_RMS_CALIBRATED_BRIDGE_PROTOCOL = "qwen32_rms_calibrated_input_soft_token_bridge_k32_v3"
QWEN32_MODEL_DIMENSION = 5120
QFORMER_STATE_DIMENSION = 512
# Official AMA prompt-budget truncation marker (model_client._truncate_prompt_to_token_budget).
TRUNCATION_MARKER = "\n...[truncated]...\n"


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = float(eps)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scale = values.float().square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        return values * scale.to(values.dtype) * self.weight


class Qwen32InputSoftTokenBridge(nn.Module):
    """Map rank-major top-k K32 states into Qwen3-32B embeddings."""

    protocol = QWEN32_INPUT_BRIDGE_PROTOCOL

    def __init__(self, *, input_dim: int = 512, model_dim: int = 5120,
                 slots: int = 32, max_memory_ranks: int = 4) -> None:
        super().__init__()
        if (input_dim, model_dim) != (512, 5120):
            raise ValueError("v1 bridge is fixed to 512 -> 5120")
        if slots != 32:
            raise ValueError("v1 bridge requires K=32 QFormer states")
        if max_memory_ranks < 1:
            raise ValueError("max_memory_ranks must be positive")
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.slots = int(slots)
        self.max_memory_ranks = int(max_memory_ranks)
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.output_projection = nn.Linear(model_dim, model_dim)
        self.output_norm = RMSNorm(model_dim)
        self.slot_embedding = nn.Parameter(torch.zeros(slots, model_dim))
        self.memory_rank_embedding = nn.Parameter(torch.zeros(max_memory_ranks, model_dim))

    def forward(self, xbar: torch.Tensor, valid: torch.Tensor, *,
                return_content: bool = False):
        xbar = torch.as_tensor(xbar, dtype=torch.float32)
        valid = torch.as_tensor(valid, dtype=torch.bool, device=xbar.device)
        if xbar.ndim == 3:
            xbar, valid = xbar[:, None], valid[:, None]
        if xbar.ndim != 4 or xbar.shape[-2:] != (self.slots, self.input_dim):
            raise ValueError(
                f"expected (batch,memories,{self.slots},{self.input_dim}), got {tuple(xbar.shape)}"
            )
        if valid.shape != xbar.shape[:-1]:
            raise ValueError(f"valid {tuple(valid.shape)} does not match {tuple(xbar.shape)}")
        if xbar.shape[1] > self.max_memory_ranks:
            raise ValueError(f"received {xbar.shape[1]} memories, maximum is {self.max_memory_ranks}")
        if not torch.isfinite(xbar).all():
            raise ValueError("xbar contains non-finite values")
        if not valid.any(dim=(-1, -2)).all():
            raise ValueError("each batch row needs at least one valid latent slot")
        values = self.input_norm(xbar)
        values = self.input_projection(values)
        values = torch.nn.functional.gelu(values)
        values = self.output_projection(values)
        content = self.output_norm(values)
        content = content * valid[..., None]
        values = content + self.slot_embedding[None, None]
        values = values + self.memory_rank_embedding[None, :xbar.shape[1], None]
        values = values * valid[..., None]
        values = values.flatten(1, 2)
        mask = valid.flatten(1, 2)
        if return_content:
            return values, mask, content.flatten(1, 2)
        return values, mask


class Qwen32ScaledInputSoftTokenBridge(Qwen32InputSoftTokenBridge):
    """V2 bridge with a bounded, learnable reader-embedding scale."""

    protocol = QWEN32_SCALED_INPUT_BRIDGE_PROTOCOL

    def __init__(self, *, initial_output_scale: float,
                 min_output_scale: float = 0.005,
                 max_output_scale: float = 0.05, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not (0 < min_output_scale < initial_output_scale < max_output_scale):
            raise ValueError(
                "output scales must satisfy 0 < min < initial < max"
            )
        self.initial_output_scale = float(initial_output_scale)
        self.min_output_scale = float(min_output_scale)
        self.max_output_scale = float(max_output_scale)
        fraction = ((self.initial_output_scale - self.min_output_scale) /
                    (self.max_output_scale - self.min_output_scale))
        self.output_scale_logit = nn.Parameter(torch.tensor(
            math.log(fraction / (1.0 - fraction)), dtype=torch.float32
        ))

    def output_scale(self) -> torch.Tensor:
        span = self.max_output_scale - self.min_output_scale
        return self.min_output_scale + span * self.output_scale_logit.sigmoid()

    def forward(self, xbar: torch.Tensor, valid: torch.Tensor, *,
                return_content: bool = False):
        result = super().forward(xbar, valid, return_content=return_content)
        values, mask = result[:2]
        values = values * self.output_scale().to(values.dtype)
        if return_content:
            return values, mask, result[2]
        return values, mask


class Qwen32RMSCalibratedInputSoftTokenBridge(Qwen32InputSoftTokenBridge):
    """V3 bridge whose valid output tokens match the Reader embedding RMS."""

    protocol = QWEN32_RMS_CALIBRATED_BRIDGE_PROTOCOL

    def __init__(self, *, target_rms: float, rms_eps: float = 1e-12,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not math.isfinite(target_rms) or target_rms <= 0:
            raise ValueError("target_rms must be finite and positive")
        if not math.isfinite(rms_eps) or rms_eps <= 0:
            raise ValueError("rms_eps must be finite and positive")
        self.register_buffer("target_rms", torch.tensor(float(target_rms)))
        self.rms_eps = float(rms_eps)

    def forward(self, xbar: torch.Tensor, valid: torch.Tensor, *,
                return_content: bool = False):
        result = super().forward(xbar, valid, return_content=return_content)
        values, mask = result[:2]
        inverse = values.float().square().mean(-1, keepdim=True).clamp_min(
            self.rms_eps
        ).rsqrt()
        values = values * inverse.to(values.dtype) * self.target_rms.to(values.dtype)
        values = values * mask[..., None]
        if return_content:
            return values, mask, result[2]
        return values, mask


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_sha256(prefix: str, suffix: str, *, enable_thinking: bool) -> str:
    payload = json.dumps({"prefix": prefix, "suffix": suffix,
                          "enable_thinking": bool(enable_thinking)},
                         ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def ama_openend_user_prompt(question: str, *, memory_sentinel: str) -> str:
    if not question.strip():
        raise ValueError("question must be non-empty")
    if not memory_sentinel or memory_sentinel in question:
        raise ValueError("memory sentinel must be unique and non-empty")
    return (
        f"{memory_sentinel}\n\n## Questions\nQuestion 1: {question}\n\n"
        f"## Instructions\nProvide a direct and concise answer.\n\n"
        f"Answer[1]: [your answer here]"
    )


def render_ama_openend_parts(tokenizer, question: str, *, enable_thinking: bool):
    sentinel = "\x00AMA_LATENT_MEMORY\x00"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": ama_openend_user_prompt(
            question, memory_sentinel=sentinel)}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=bool(enable_thinking),
    )
    if rendered.count(sentinel) != 1:
        raise ValueError("chat template did not preserve exactly one memory sentinel")
    prefix, suffix = rendered.split(sentinel)
    return prefix, suffix, prompt_sha256(prefix, suffix, enable_thinking=enable_thinking)


def latent_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Give masked padding no RoPE position and do not let it advance positions."""
    positions = attention_mask.long().cumsum(-1) - 1
    return positions.masked_fill(~attention_mask.bool(), 0)


class Qwen32LatentReader:
    """Direct-input Qwen3-32B reader; no original memory text is accepted."""

    def __init__(self, model, tokenizer, bridge: Qwen32InputSoftTokenBridge, *,
                 enable_thinking: bool = True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        embedding = model.get_input_embeddings()
        if int(embedding.weight.shape[-1]) != QWEN32_MODEL_DIMENSION:
            raise ValueError("reader embedding width is not Qwen3-32B's 5120")
        self.bridge = bridge.to(embedding.weight.device)
        self.enable_thinking = bool(enable_thinking)

    def build_inputs(self, question: str, xbar: torch.Tensor, valid: torch.Tensor):
        xbar_tensor = torch.as_tensor(xbar, dtype=torch.float32)
        valid_tensor = torch.as_tensor(valid, dtype=torch.bool)
        if xbar_tensor.ndim not in (2, 3):
            raise ValueError("answer expects one (memories,32,512) latent collection")
        segments = [("latent", (xbar_tensor[index], valid_tensor[index]))
                    for index in range(xbar_tensor.shape[0])]
        inputs, _audit = self.build_inputs_segments(question, segments)
        return inputs

    def build_inputs_segments(self, question: str, segments, *,
                              max_model_len: int | None = None,
                              max_new_tokens: int | None = None):
        """Build reader inputs from rank-ordered memory segments.

        ``segments`` items are ``("latent", (xbar, valid))`` or ``("text", str)``,
        listed in memory-rank order.  Latent rows are packed through the bridge
        (``memory_rank_embedding`` follows the latent-rank index); text rows are
        embedded with the reader tokenizer and joined with blank lines.  Per the
        official RAG convention each text row is a full step document and rows
        are never shortened individually.

        When the whole prompt would exceed ``max_model_len - max_new_tokens``
        tokens, only the text block is shortened to head 70% / tail 30% with the
        official ``TRUNCATION_MARKER``; latent rows keep their fixed
        ``bridge.slots``-token footprint.

        Returns ``(inputs, audit)`` where ``audit`` records the budget, the
        text token counts before/after and whether truncation fired.
        """
        prefix, suffix, _ = render_ama_openend_parts(
            self.tokenizer, question, enable_thinking=self.enable_thinking
        )
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        embedding = self.model.get_input_embeddings()
        device, dtype = embedding.weight.device, embedding.weight.dtype

        latent_states: list[tuple[torch.Tensor, torch.Tensor]] = []
        text_rows: list[str] = []
        for kind, payload in segments:
            if kind == "latent":
                latent_states.append(payload)
            elif kind == "text":
                text_rows.append(str(payload))
            else:
                raise ValueError(f"unknown memory segment kind {kind!r}")

        text_ids: list[int] = []
        for row in text_rows:
            if text_ids:
                text_ids.extend(self.tokenizer.encode("\n\n", add_special_tokens=False))
            text_ids.extend(self.tokenizer.encode(row, add_special_tokens=False))

        if latent_states:
            xbar_tensor = torch.stack([
                torch.as_tensor(xbar, dtype=torch.float32) for xbar, _ in latent_states
            ]).to(device)
            valid_tensor = torch.stack([
                torch.as_tensor(valid, dtype=torch.bool) for _, valid in latent_states
            ]).to(device)
            if xbar_tensor.ndim != 3 or xbar_tensor.shape[-2:] != (
                    self.bridge.slots, self.bridge.input_dim):
                raise ValueError(
                    "latent segments expect (memories,"
                    f"{self.bridge.slots},{self.bridge.input_dim}) states"
                )
            latent, latent_mask = self.bridge(xbar_tensor[None], valid_tensor[None])
            latent = latent.to(dtype)[0]
            latent_mask = latent_mask[0]
            slots = self.bridge.slots
            latent_chunks = [latent[row * slots:(row + 1) * slots]
                             for row in range(latent.shape[0])]
            latent_masks = [latent_mask[row * slots:(row + 1) * slots]
                            for row in range(latent_mask.shape[0])]
        else:
            latent_chunks, latent_masks = [], []

        budget = None
        allowed = None
        if max_model_len is not None or max_new_tokens is not None:
            if max_model_len is None or max_new_tokens is None:
                raise ValueError("max_model_len and max_new_tokens must be given together")
            if max_new_tokens < 1:
                raise ValueError("max_new_tokens must be positive")
            budget = int(max_model_len) - int(max_new_tokens)
            if budget <= 0:
                raise ValueError("max_new_tokens leaves no prompt budget")
            fixed = len(prefix_ids) + len(suffix_ids) + sum(
                chunk.shape[0] for chunk in latent_chunks)
            if fixed > budget:
                raise ValueError(
                    f"prompt scaffold plus latent memory exceeds budget: "
                    f"{fixed} > {budget}; increase max_model_len or reduce "
                    "max_new_tokens"
                )
            # The latent footprint and the prompt scaffold are fixed costs; the
            # whole slack goes to the text block (official head/tail semantics).
            allowed = budget - fixed
        text_tokens_before = len(text_ids)
        prompt_truncated = False
        if allowed is not None and text_tokens_before > allowed:
            prompt_truncated = True
            marker_ids = self.tokenizer.encode(TRUNCATION_MARKER, add_special_tokens=False)
            keep = allowed - len(marker_ids)
            if keep <= 0:
                text_ids = []
            else:
                head = keep * 7 // 10
                tail = keep - head
                text_ids = (text_ids[:head] + marker_ids
                            + (text_ids[-tail:] if tail else []))

        embeds = [embedding(torch.tensor([prefix_ids], dtype=torch.long, device=device))]
        masks = [torch.ones((1, len(prefix_ids)), dtype=torch.bool, device=device)]
        latent_row = 0
        text_placed = False
        for kind, _payload in segments:
            if kind == "latent":
                embeds.append(latent_chunks[latent_row][None])
                masks.append(latent_masks[latent_row][None])
                latent_row += 1
            elif not text_placed:
                text_placed = True
                if text_ids:
                    ids = torch.tensor([text_ids], dtype=torch.long, device=device)
                    embeds.append(embedding(ids))
                    masks.append(torch.ones_like(ids, dtype=torch.bool))
        embeds.append(embedding(torch.tensor([suffix_ids], dtype=torch.long, device=device)))
        masks.append(torch.ones((1, len(suffix_ids)), dtype=torch.bool, device=device))

        attention_mask = torch.cat(masks, dim=1)
        inputs = {
            "inputs_embeds": torch.cat(embeds, dim=1),
            "attention_mask": attention_mask,
            "position_ids": latent_position_ids(attention_mask),
        }
        audit = {
            "prompt_tokens": int(attention_mask.shape[1]),
            "prompt_budget": budget,
            "prompt_truncated": prompt_truncated,
            "text_tokens_before": text_tokens_before,
            "text_tokens_after": len(text_ids),
            "latent_memories": len(latent_states),
            "text_rows": len(text_rows),
        }
        return inputs, audit

    def answer(self, question: str, xbar: torch.Tensor, valid: torch.Tensor, *,
               max_new_tokens: int = 256) -> str:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        inputs = self.build_inputs(question, xbar, valid)
        self.model.eval()
        self.bridge.eval()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=(self.tokenizer.pad_token_id
                              if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id),
            )
        token_ids = generated.sequences[0] if hasattr(generated, "sequences") else generated[0]
        # Transformers versions differ on whether inputs_embeds generation
        # returns only new ids or prepends placeholder prompt ids.  The new
        # suffix can never exceed max_new_tokens, so retaining the tail is
        # stable under either convention.
        if len(token_ids) > max_new_tokens:
            token_ids = token_ids[-max_new_tokens:]
        return self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def save_qwen32_bridge(path: str | Path, bridge: Qwen32InputSoftTokenBridge, *,
                       qformer_artifact_sha256: str,
                       retrieval_head_artifact_sha256: str,
                       reader_model_sha256: str, data_manifest_sha256: str,
                       prompt_sha256_value: str, **metadata: Any) -> None:
    required = {
        "qformer_artifact_sha256": qformer_artifact_sha256,
        "retrieval_head_artifact_sha256": retrieval_head_artifact_sha256,
        "reader_model_sha256": reader_model_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "prompt_sha256": prompt_sha256_value,
    }
    if any(not isinstance(value, str) or len(value) != 64 for value in required.values()):
        raise ValueError("bridge provenance fields must be 64-character SHA256 strings")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scaled_metadata = {}
    if isinstance(bridge, Qwen32ScaledInputSoftTokenBridge):
        scaled_metadata = {
            "initial_output_scale": bridge.initial_output_scale,
            "min_output_scale": bridge.min_output_scale,
            "max_output_scale": bridge.max_output_scale,
            "learned_output_scale": float(bridge.output_scale().detach().cpu()),
        }
    elif isinstance(bridge, Qwen32RMSCalibratedInputSoftTokenBridge):
        scaled_metadata = {
            "target_reader_embedding_rms": float(bridge.target_rms.detach().cpu()),
            "rms_calibration_eps": bridge.rms_eps,
        }
    torch.save({
        "protocol": bridge.protocol,
        "artifact_type": type(bridge).__name__,
        "metadata": {**required, "input_dim": bridge.input_dim,
                     "model_dim": bridge.model_dim, "slots": bridge.slots,
                     "max_memory_ranks": bridge.max_memory_ranks,
                     **scaled_metadata, **metadata},
        "state_dict": bridge.state_dict(),
    }, target)


def load_qwen32_bridge(path: str | Path, *, expected_qformer_sha256: str | None = None,
                       expected_retrieval_head_sha256: str | None = None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    protocol = payload.get("protocol")
    if protocol not in {
        QWEN32_INPUT_BRIDGE_PROTOCOL, QWEN32_SCALED_INPUT_BRIDGE_PROTOCOL,
        QWEN32_RMS_CALIBRATED_BRIDGE_PROTOCOL,
    }:
        raise ValueError(f"unexpected bridge protocol {payload.get('protocol')!r}")
    metadata = dict(payload.get("metadata") or {})
    for name in ("qformer_artifact_sha256", "retrieval_head_artifact_sha256",
                 "reader_model_sha256", "data_manifest_sha256", "prompt_sha256"):
        if not isinstance(metadata.get(name), str) or len(metadata[name]) != 64:
            raise ValueError(f"bridge checkpoint lacks valid {name}")
    if expected_qformer_sha256 is not None and metadata["qformer_artifact_sha256"] != expected_qformer_sha256:
        raise ValueError("bridge QFormer artifact hash mismatch")
    if expected_retrieval_head_sha256 is not None and metadata["retrieval_head_artifact_sha256"] != expected_retrieval_head_sha256:
        raise ValueError("bridge retrieval-head artifact hash mismatch")
    common = {
        "input_dim": int(metadata.get("input_dim", -1)),
        "model_dim": int(metadata.get("model_dim", -1)),
        "slots": int(metadata.get("slots", -1)),
        "max_memory_ranks": int(metadata.get("max_memory_ranks", -1)),
    }
    if protocol == QWEN32_SCALED_INPUT_BRIDGE_PROTOCOL:
        bridge = Qwen32ScaledInputSoftTokenBridge(
            initial_output_scale=float(metadata.get("initial_output_scale", -1)),
            min_output_scale=float(metadata.get("min_output_scale", -1)),
            max_output_scale=float(metadata.get("max_output_scale", -1)),
            **common,
        )
    elif protocol == QWEN32_RMS_CALIBRATED_BRIDGE_PROTOCOL:
        bridge = Qwen32RMSCalibratedInputSoftTokenBridge(
            target_rms=float(metadata.get("target_reader_embedding_rms", -1)),
            rms_eps=float(metadata.get("rms_calibration_eps", 1e-12)),
            **common,
        )
    else:
        bridge = Qwen32InputSoftTokenBridge(**common)
    bridge.load_state_dict(payload["state_dict"], strict=True)
    return bridge, metadata
