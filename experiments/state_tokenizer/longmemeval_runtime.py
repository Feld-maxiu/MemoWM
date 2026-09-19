"""Standalone LongMemEval runtime for the ResidualMem components.

This module intentionally does not import ``QFormerInstructTokenizer`` or
modify the WMA adapter.  LongMemEval supplies a real text AXTree, so the
runtime calls the frozen trunk and the learned Q-Former directly with that
tree.  It can be used by cache-building/evaluation scripts without changing
the WMA reproduction contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from residualmem.benchmarks.longmemeval import canonicalize_axtree


class LongMemEvalQFormerRuntime:
    """Encode LME observations and produce their retrieval keys."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        checkpoint: str | Path,
        retrieval_head: str | Path | None = None,
        device: str = "cuda:0",
        layer: int | None = None,
        max_length: int = 8192,
        allow_truncate: bool = False,
    ) -> None:
        from experiments.state_tokenizer.extract_qwen import _load_model
        from experiments.state_tokenizer.trunk_states import NUM_MODALITIES
        from residualmem.latent.instruct_bridge import (
            InputSoftTokenConnector,
            MaskedAttentionRetrievalHead,
            RETRIEVAL_BRIDGE_PROTOCOL,
            load_bridge,
        )
        from residualmem.latent.qformer import QFormerStateReader, StateQFormer

        self.device = torch.device(device)
        self.max_length = int(max_length)
                                                                               
                                                                              
                                                                     
        self.allow_truncate = bool(allow_truncate)
        self.truncated_observations = 0
        self.processor, self.model = _load_model(str(model_path), self.device, False)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("state_dict")
        if not isinstance(state, dict):
            raise ValueError(f"{checkpoint}: missing state_dict")
        metadata = dict(payload.get("metadata") or {})
        queries = int(metadata.get("queries", 32))
        hidden = int(metadata.get("hidden", 1024))
        heads = int(metadata.get("heads", 8))
        layers = int(metadata.get("layers", 4))
        qk_norm = bool(metadata.get("qk_norm", False))
        self.layer = int(layer if layer is not None else metadata.get("layer", 16))
        self.reader = QFormerStateReader(
            StateQFormer(
                num_queries=queries,
                hidden=hidden,
                heads=heads,
                layers=layers,
                modalities=NUM_MODALITIES,
                self_attention=bool(metadata.get("self_attention", False)),
                qk_norm=qk_norm,
            ),
            InputSoftTokenConnector(slots=queries),
            MaskedAttentionRetrievalHead(),
        ).to(self.device)
        self.reader.load_state_dict(state, strict=True)
        retrieval_artifact = None
        if retrieval_head is not None:
            retrieval_artifact = load_bridge(
                retrieval_head,
                self.reader.retrieval_head,
                expected_protocol=RETRIEVAL_BRIDGE_PROTOCOL,
            )
        self.reader.eval()
        self.retrieval_head = self.reader.retrieval_head
        self.metadata = {
            "protocol": "longmemeval-runtime-v1",
            "checkpoint": str(Path(checkpoint).resolve()),
            "retrieval_head": (
                str(Path(retrieval_head).resolve()) if retrieval_head is not None else None
            ),
            "retrieval_head_source": (
                "standalone" if retrieval_artifact is not None else "joint_checkpoint"
            ),
            "retrieval_head_metadata": (
                retrieval_artifact.metadata if retrieval_artifact is not None else None
            ),
            "model": str(Path(model_path).resolve()),
            "queries": queries,
            "layer": self.layer,
            "max_length": self.max_length,
            "allow_truncate": self.allow_truncate,
            "qk_norm": qk_norm,
        }

    @torch.inference_mode()
    def encode(self, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(xbar, valid, key)`` for one prepared observation."""
        from experiments.state_tokenizer.trunk_states import collate, trunk_states

        screenshot = Path(str(record["screenshot"])).resolve()
        with Image.open(screenshot) as handle:
            image = handle.convert("RGB")
        dom = canonicalize_axtree(
            str(record.get("axtree") or record.get("synthetic_axtree") or "")
        )
        states = trunk_states(
            self.processor,
            self.model,
            image,
            dom,
            layer=self.layer,
            max_length=self.max_length,
            device=self.device,
            allow_truncate=self.allow_truncate,
        )
        if states.truncated:
            self.truncated_observations += 1
        xbar, valid = self.reader.encode(*collate([states]))
        key = self.retrieval_head(xbar, valid)
        return (
            xbar[0].float().cpu().numpy(),
            valid[0].cpu().numpy().astype(np.bool_),
            key[0].float().cpu().numpy(),
        )


def load_prepared_records(store: str | Path):
    """Iterate ``(trajectory_id, record_index, record)`` from a prepared store."""
    root = Path(store) / "trajectories"
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no prepared trajectories under {root}")
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata"])))
        trajectory_id = str(metadata["sample_id"])
        for index, record in enumerate(metadata.get("records") or []):
            yield trajectory_id, index, record
