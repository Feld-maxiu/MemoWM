"""Qwen3.5 multimodal front-end (report Section 5), Stage 5.

Swaps the ``x_t`` source from the structured oracle to a frozen VLM behind the
same :class:`ObservationEncoder` seam: ``observation -> backbone hidden states H_t
-> Perceiver resampler -> StateTokens x_t``. Everything downstream (RSSM, codec,
retrieval) is unchanged.

Hard constraints from the report are honored: the backbone is frozen and stateless
across steps (fresh forward per observation, KV cache never carried over), so a long
context can never become unbilled memory.

Two backbones are provided:
* :class:`HFQwenBackbone` -- the real frozen Qwen loaded from HuggingFace (lazy
  ``transformers``/``torch`` import; download the weights via ``model_id``).
* :class:`MockBackbone` -- deterministic random features for CPU wiring tests, so
  the whole path runs without downloading a 9B model.
"""
from __future__ import annotations

import hashlib
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from ..latent.tokenizer import PerceiverConfig, perceiver_hash, resample
from ..latent.types import DomainId, LatentSpec, StateTokens
from .base import encoder_hash


@runtime_checkable
class ObservationBackbone(Protocol):
    backbone_id: str
    output_dim: int

    def encode_features(self, observation: Any) -> np.ndarray:
        """Return frozen hidden states of shape (seq_len, output_dim)."""

    def reset(self) -> None:
        """Clear any per-step context (report: no cross-step KV carryover)."""


class MockBackbone:
    """Deterministic random features keyed by the observation (CPU wiring tests)."""

    def __init__(self, output_dim: int = 64, seq_len: int = 8):
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.backbone_id = f"mock-d{output_dim}-t{seq_len}"

    def reset(self) -> None:
        pass

    def encode_features(self, observation: Any) -> np.ndarray:
        digest = hashlib.sha256(repr(observation).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.normal(size=(self.seq_len, self.output_dim)).astype(np.float32)


class HFQwenBackbone:
    """Frozen Qwen loaded from HuggingFace. Text-first; images via the processor.

    ``transformers``/``torch`` are imported lazily so importing this module never
    requires them. Download weights by passing the HF ``model_id`` (e.g. a Qwen3.5
    id); nothing here runs a forward pass until :meth:`encode_features` is called.
    """

    def __init__(self, model_id: str, *, layer: int = -1, device: str = "cpu",
                 dtype: str = "float32", trust_remote_code: bool = True):
        self.model_id = model_id
        self.layer = layer
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.backbone_id = f"hf:{model_id}@layer{layer}"
        self._model = None
        self._tokenizer = None
        self.output_dim = None  # filled after the model loads

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: F401  (lazy)
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code)
        self._model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code,
            output_hidden_states=True).to(self.device).eval()
        for param in self._model.parameters():
            param.requires_grad_(False)
        self.output_dim = int(self._model.config.hidden_size)

    def reset(self) -> None:
        pass  # each encode_features call is an independent, context-free forward

    def encode_features(self, observation: Any) -> np.ndarray:
        self._ensure_loaded()
        import torch

        text = observation if isinstance(observation, str) else str(observation)
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        hidden = outputs.hidden_states[self.layer][0]  # (seq_len, hidden)
        return hidden.float().cpu().numpy().astype(np.float32)


class QwenObservationEncoder:
    """Frozen backbone + Perceiver tokenizer -> StateTokens (``x_t``)."""

    def __init__(self, backbone: ObservationBackbone, tokenizer_params, tokenizer_config,
                 latent_spec: LatentSpec, domain: DomainId, *, renderer=None):
        self.backbone = backbone
        self.tokenizer_params = {k: jnp.asarray(v) for k, v in tokenizer_params.items()}
        self.tokenizer_config = tokenizer_config
        self.domain = domain
        self.renderer = renderer  # optional probe head for evidence rendering
        self.spec = LatentSpec(
            num_groups=latent_spec.num_groups,
            num_categories=latent_spec.num_categories,
            token_dim=tokenizer_config.latent_dim,
            num_tokens=tokenizer_config.num_latents)
        self.encoder_id = f"qwen-frontend[{backbone.backbone_id}]"
        self._tok_hash = perceiver_hash(self.tokenizer_params, tokenizer_config)

        def _resample(params, features):
            return resample(params, features, tokenizer_config)

        self._resample = jax.jit(_resample)

    def reset(self) -> None:
        self.backbone.reset()

    def encode(self, observation: Any) -> StateTokens:
        features = np.asarray(self.backbone.encode_features(observation), np.float32)
        if features.ndim != 2 or features.shape[1] != self.tokenizer_config.input_dim:
            raise ValueError(
                f"backbone features {features.shape} do not match tokenizer input_dim "
                f"{self.tokenizer_config.input_dim}")
        tokens = np.asarray(self._resample(self.tokenizer_params, features), np.float32)
        return StateTokens(tokens, {"state": tuple(range(tokens.shape[0]))})

    @property
    def hash_bytes(self) -> bytes:
        base = encoder_hash(self.encoder_id, self.spec, self.domain)
        return hashlib.sha256(base + self._tok_hash).digest()

    def decode_tokens(self, tokens: np.ndarray):
        if self.renderer is None:
            raise NotImplementedError(
                "multimodal evidence rendering needs a probe head; attach a `renderer` "
                "with decode_tokens(x)->CanonicalState (report Section 12 export/probe head)")
        return self.renderer.decode_tokens(tokens)
