"""Runtime bridge for AMA-adapted observations and the existing x_t artifacts.

The implementation deliberately keeps heavyweight imports lazy.  Importing this
module from the AMA evaluation environment is safe; constructing either encoder
requires the residual-mem checkout and its ``qwen-vl`` Python environment.

The selected artifact pair is explicit rather than interchangeable:

* ``qformer-K16-nosa.pt`` produces 16 x 512 x_t states;
* ``retrieval-head-qformer-wma.pt`` was fitted on that Q-Former cache and maps
  those states to normalized 4096-D ``qwen3_vl_fused_observation_v1`` vectors.

Using a PCA head or a 64-slot Q-Former with this head would be a coordinate
mismatch, so the loader checks both protocols and state-dict shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import sys

import numpy as np

from .adapter import AMAXTObservation


QWEN_VL_FUSED_OBSERVATION_PROTOCOL = "qwen3_vl_fused_observation_v1"
QFORMER_PROTOCOL = "qwen35_qformer_state_tokenizer_v1"
RETRIEVAL_HEAD_PROTOCOL = "qwen35_instruct_retrieval_bridge_v1"
EMBEDDING_DIMENSION = 4096


@dataclass(frozen=True)
class QFormerRuntimeConfig:
    """Paths and architecture values required by the frozen x_t runtime."""

    residualmem_root: str
    qwen35_model_path: str
    qformer_checkpoint: str
    retrieval_head_checkpoint: str
    query_model_path: str
    # CPU is the safe default: a caller must opt in explicitly before any GPU
    # can be touched.  This matters on the shared server, where existing jobs
    # may already own the cards.
    device: str = "cpu"
    queries: int = 16
    qformer_hidden: int = 1024
    qformer_heads: int = 8
    qformer_layers: int = 4
    self_attention: bool = False
    layer: int = 16
    max_length: int = 8192

    def required_paths(self) -> tuple[Path, ...]:
        return tuple(
            Path(path)
            for path in (
                self.residualmem_root,
                self.qwen35_model_path,
                self.qformer_checkpoint,
                self.retrieval_head_checkpoint,
                self.query_model_path,
            )
        )

    def check_paths(self) -> None:
        missing = [str(path) for path in self.required_paths() if not path.exists()]
        if missing:
            raise FileNotFoundError("x_t runtime paths do not exist: " + ", ".join(missing))


def _enable_residualmem(root: str) -> None:
    resolved = str(Path(root).resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim not in (1, 2) or values.shape[-1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"expected (...,{EMBEDDING_DIMENSION}) embedding, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("embedding contains non-finite values")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embedding contains a zero-norm row")
    return values / norms


def validate_artifact_pair(config: QFormerRuntimeConfig) -> dict[str, Any]:
    """Validate the selected Q-Former and retrieval head without a GPU forward.

    This reads the two checkpoints on CPU. It proves protocol, slot count and
    4096-D head compatibility before a 9B model is loaded on GPU.
    """
    config.check_paths()
    _enable_residualmem(config.residualmem_root)
    import torch

    qformer_payload = torch.load(
        config.qformer_checkpoint, map_location="cpu", weights_only=True
    )
    head_payload = torch.load(
        config.retrieval_head_checkpoint, map_location="cpu", weights_only=True
    )
    if qformer_payload.get("protocol") != QFORMER_PROTOCOL:
        raise ValueError(
            f"Q-Former protocol {qformer_payload.get('protocol')!r}; expected {QFORMER_PROTOCOL!r}"
        )
    metadata = dict(qformer_payload.get("metadata") or {})
    artifact_queries = metadata.get("queries")
    if artifact_queries != config.queries:
        raise ValueError(
            f"Q-Former checkpoint has queries={artifact_queries!r}; config requests {config.queries}"
        )
    if head_payload.get("protocol") != RETRIEVAL_HEAD_PROTOCOL:
        raise ValueError(
            f"retrieval-head protocol {head_payload.get('protocol')!r}; "
            f"expected {RETRIEVAL_HEAD_PROTOCOL!r}"
        )
    head_metadata = dict(head_payload.get("metadata") or {})
    if head_metadata.get("teacher_protocol") != QWEN_VL_FUSED_OBSERVATION_PROTOCOL:
        raise ValueError(
            "retrieval head is not aligned to qwen3_vl_fused_observation_v1"
        )
    cache_name = str(head_metadata.get("cache", "")).lower()
    if "qformer" not in cache_name:
        raise ValueError(
            "retrieval head was not trained on a Q-Former cache; refusing coordinate mismatch"
        )
    state = head_payload.get("state_dict") or {}
    projection = state.get("projection.2.weight")
    if projection is None or tuple(projection.shape) != (EMBEDDING_DIMENSION, 1024):
        actual = None if projection is None else tuple(projection.shape)
        raise ValueError(f"retrieval head projection must be (4096, 1024), got {actual}")
    return {
        "qformer_protocol": qformer_payload["protocol"],
        "qformer_queries": artifact_queries,
        "qformer_checkpoint": str(Path(config.qformer_checkpoint).resolve()),
        "retrieval_head_protocol": head_payload["protocol"],
        "retrieval_head_checkpoint": str(Path(config.retrieval_head_checkpoint).resolve()),
        "teacher_protocol": head_metadata["teacher_protocol"],
        "embedding_dimension": EMBEDDING_DIMENSION,
    }


class QFormerXTDocumentEncoder:
    """Frozen Qwen3.5 + Q-Former + matched retrieval head for AMA steps."""

    def __init__(self, config: QFormerRuntimeConfig) -> None:
        self.config = config
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        validate_artifact_pair(self.config)
        _enable_residualmem(self.config.residualmem_root)
        import torch
        from PIL import Image
        from experiments.state_tokenizer.extract_qwen import _load_model
        from experiments.state_tokenizer.trunk_states import NUM_MODALITIES
        from residualmem.latent.instruct_bridge import (
            InputSoftTokenConnector,
            MaskedAttentionRetrievalHead,
        )
        from residualmem.latent.qformer import QFormerStateReader, StateQFormer

        self._torch = torch
        self._image = Image
        self._device = torch.device(self.config.device)
        self.processor, self.model = _load_model(
            self.config.qwen35_model_path, self._device, False
        )
        # The Q-Former checkpoint contains its reader connector too.  It is
        # loaded strictly even though retrieval only needs qformer.encode(), so
        # all coordinates remain tied to the saved artifact.
        reader = QFormerStateReader(
            StateQFormer(
                num_queries=self.config.queries,
                hidden=self.config.qformer_hidden,
                heads=self.config.qformer_heads,
                layers=self.config.qformer_layers,
                modalities=NUM_MODALITIES,
                self_attention=self.config.self_attention,
            ),
            InputSoftTokenConnector(slots=self.config.queries),
            retrieval_head=None,
        )
        qformer_payload = torch.load(
            self.config.qformer_checkpoint, map_location="cpu", weights_only=True
        )
        reader.load_state_dict(qformer_payload["state_dict"], strict=True)
        self.reader = reader.to(self._device).eval()

        head = MaskedAttentionRetrievalHead()
        head_payload = torch.load(
            self.config.retrieval_head_checkpoint, map_location="cpu", weights_only=True
        )
        head.load_state_dict(head_payload["state_dict"], strict=True)
        self.retrieval_head = head.to(self._device).eval()
        self._loaded = True

    def encode_with_latents(
        self, records: Sequence[AMAXTObservation]
    ) -> tuple[np.ndarray, tuple[tuple[np.ndarray, np.ndarray], ...]]:
        """Encode AMA steps once for both retrieval and the latent Reader.

        The vectors are used only to rank steps.  The matching ``(xbar, valid)``
        state is retained for each selected rank and passed directly to the
        connector-backed Qwen reader at answer time.
        """
        if not records:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32), ()
        self._load()
        _enable_residualmem(self.config.residualmem_root)
        from residualmem.benchmarks.worldmemarena_tokenizer import (
            WebObservation,
            synthetic_axtree,
        )
        from experiments.state_tokenizer.trunk_states import collate, trunk_states

        vectors = []
        latents = []
        for record in records:
            web_observation = WebObservation(**record.worldmem_payload())
            if not web_observation.has_observation:
                raise ValueError(f"AMA step {record.step_index} has no encodable observation")
            dom = synthetic_axtree(web_observation)
            image = self._image.new("RGB", (1280, 720), "white")
            states = trunk_states(
                self.processor,
                self.model,
                image,
                dom,
                layer=self.config.layer,
                max_length=self.config.max_length,
                device=self._device,
            )
            with self._torch.no_grad():
                xbar, valid = self.reader.encode(*collate([states]))
                vector = self.retrieval_head(xbar, valid)[0]
            vectors.append(vector.float().cpu().numpy())
            latents.append((
                xbar[0].float().cpu().numpy(),
                valid[0].cpu().numpy(),
            ))
        return _l2_normalize(np.stack(vectors, axis=0)), tuple(latents)

    def encode(self, records: Sequence[AMAXTObservation]) -> np.ndarray:
        """Compatibility helper for retrieval-only callers."""
        return self.encode_with_latents(records)[0]

class QwenVLQueryEncoder:
    """Qwen3-VL-Embedding query encoder in the retrieval-head teacher space."""

    def __init__(self, config: QFormerRuntimeConfig) -> None:
        self.config = config
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        self.config.check_paths()
        import torch
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            self.config.query_model_path,
            device=self.config.device,
            trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16},
        )

    def encode(self, questions: Iterable[str]) -> np.ndarray:
        """Encode AMA questions as normalized 4096-D vectors.

        The document side was distilled to the Qwen3-VL fused-observation
        protocol.  Questions are text-only payloads in that same embedding
        model, never passed through the x_t document encoder.
        """
        values = [str(question).strip() for question in questions]
        if any(not value for value in values):
            raise ValueError("query question must be non-empty")
        if not values:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        self._load()
        output = self._model.encode(
            [{"text": value} for value in values],
            batch_size=min(8, len(values)),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return _l2_normalize(np.asarray(output, dtype=np.float32))
