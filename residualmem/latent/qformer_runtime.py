"""A Q-Former tokenizer behind ``FrozenV9InstructTokenizer``'s contract.

The WorldMemArena adapter builds one state per observation and never re-encodes
-- it caches the latent under a memory id and serves it for every subsequent
question. That storage model is what forces the resampler to stay
query-independent, and it is also why swapping the tokenizer is a drop-in: the
adapter only needs ``encode(observation) -> something with .xbar/.valid``.

So this exposes exactly that, and everything downstream -- the retrieval head,
the reader connector, the official retrieval and answer stages -- runs
unchanged. The alternative was another bespoke offline harness, and the last
two of those each produced a number that turned out to measure a confound I had
introduced myself.

Differences from the frozen pooling it replaces, all of them intended:

* the full variable-length layer-16 sequence is resampled by learned queries
  instead of being reduced by fixed spatial pooling and rule-based selection;
* every slot is valid, where the pooled path left the context band empty in 13
  of its 16 slots on every WorldMemArena observation;
* there is no PCA and no frozen group/channel normalization -- the resampler
  emits the 512-wide state directly, so ``a2_xbar`` is unavailable and the
  ``-A2-`` baselines cannot run against it.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

QFORMER_TOKENIZER_PROTOCOL = "qwen35_qformer_state_v1"


@dataclasses.dataclass(frozen=True)
class QFormerTokenizerOutput:
    """The subset of ``FrozenTokenizerOutput`` the adapter actually reads."""

    xbar: np.ndarray
    valid: np.ndarray
    a2_xbar: np.ndarray | None = None
    key64: np.ndarray | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


class QFormerInstructTokenizer:
    """``WebObservation -> (queries, 512)`` learned state, query-independent."""

    protocol = QFORMER_TOKENIZER_PROTOCOL

    def __init__(
        self,
        *,
        model_path: str | Path,
        checkpoint: str | Path,
        queries: int = 16,
        hidden: int = 1024,
        heads: int = 8,
        layers: int = 4,
        self_attention: bool = False,
        device: str = "cuda:0",
        layer: int = 16,
        max_length: int = 8192,
    ) -> None:
        from experiments.state_tokenizer.extract_qwen import _load_model
        from experiments.state_tokenizer.trunk_states import NUM_MODALITIES

        from .instruct_bridge import InputSoftTokenConnector, MaskedAttentionRetrievalHead
        from .qformer import QFORMER_PROTOCOL, QFormerStateReader, StateQFormer

        self.device = torch.device(device)
        self.layer = int(layer)
        self.max_length = int(max_length)
        self.processor, self.model = _load_model(str(model_path), self.device, False)

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("protocol") != QFORMER_PROTOCOL:
            raise ValueError(
                f"{checkpoint}: protocol {payload.get('protocol')!r}, "
                f"expected {QFORMER_PROTOCOL!r}"
            )
        state = payload["state_dict"]
        # The retrieval head is trained jointly, as L_sem's projection. Without
        # it the states are in a coordinate system no head was ever fitted to,
        # and the one on disk was fitted to the PCA path -- feeding it these
        # states would retrieve noise while looking like a working pipeline.
        # This repo has already paid once for a silent coordinate mismatch
        # (see pca_binding); refuse instead.
        if not any(k.startswith("retrieval_head.") for k in state):
            raise ValueError(
                f"{checkpoint} carries no retrieval head, so it was trained "
                "without L_sem. Its states have no matched retriever and cannot "
                "be evaluated through the retrieval pipeline."
            )
        module = QFormerStateReader(
            StateQFormer(num_queries=queries, hidden=hidden, heads=heads,
                         layers=layers, modalities=NUM_MODALITIES,
                         self_attention=self_attention),
            InputSoftTokenConnector(slots=queries),
            MaskedAttentionRetrievalHead(),
        )
        module.load_state_dict(state, strict=True)
        self.reader = module.to(self.device).eval()
        self.connector = self.reader.connector
        self.retrieval_head = self.reader.retrieval_head
        self.metadata = dict(payload.get("metadata") or {})

    def _blank_image(self) -> Image.Image:
        return Image.new("RGB", (1280, 720), "white")

    def encode(
        self,
        observation,
        *,
        validate: bool = True,
        image: Image.Image | None = None,
        return_key64: bool = False,
    ) -> QFormerTokenizerOutput:
        from residualmem.benchmarks.worldmemarena_tokenizer import (
            DEFAULT_AXTREE_STYLE,
            synthetic_axtree,
            validate_observation,
        )

        from experiments.state_tokenizer.trunk_states import collate, trunk_states

        if validate:
            validate_observation(observation)
        dom = synthetic_axtree(observation, DEFAULT_AXTREE_STYLE)
        if image is not None:
            picture = image.convert("RGB")
        elif observation.screenshot:
            with Image.open(observation.screenshot) as handle:
                picture = handle.convert("RGB")
        else:
            picture = self._blank_image()

        states = trunk_states(
            self.processor, self.model, picture, dom,
            layer=self.layer, max_length=self.max_length, device=self.device,
        )
        with torch.no_grad():
            xbar, valid = self.reader.encode(*collate([states]))
        return QFormerTokenizerOutput(
            xbar=xbar[0].float().cpu().numpy(),
            valid=valid[0].cpu().numpy(),
            metadata={
                "protocol": self.protocol,
                "original_sequence_length": len(states),
                "queries": int(xbar.shape[1]),
                "synthetic_axtree": dom,
                "checkpoint": self.metadata,
            },
        )
