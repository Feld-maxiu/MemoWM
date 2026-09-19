"""Official-style raw-state text retrieval for LongMemEval-V2 Web Small."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Iterable

import numpy as np


PROTOCOL = "longmemeval-official-raw-state-text-v1"
DEFAULT_QUERY_INSTRUCTION = (
    "Given a question about past agent trajectories, retrieve relevant "
    "memory entries that help answer it."
)


@dataclasses.dataclass(frozen=True)
class TextRetrievalHit:
    trajectory_id: str
    center_index: int
    score: float
    context_text: str
    goal: str
    slice_start: int
    slice_end: int


class LongMemEvalTextIndex:
    """Immutable global index over one raw-state slice per observation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        with np.load(self.path, allow_pickle=False) as data:
            self.embedding = np.asarray(data["embedding"], dtype=np.float32)
            self.trajectory_id = np.asarray(data["trajectory_id"], dtype=str)
            self.center_index = np.asarray(data["center_index"], dtype=np.int64)
            self.slice_start = np.asarray(data["slice_start"], dtype=np.int64)
            self.slice_end = np.asarray(data["slice_end"], dtype=np.int64)
            self.context_text = np.asarray(data["context_text"], dtype=str)
            self.goal = np.asarray(data["goal"], dtype=str)
            self.metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        if self.metadata.get("protocol") != PROTOCOL:
            raise ValueError(f"text index protocol mismatch: {self.metadata.get('protocol')}")
        count = len(self.embedding)
        for name, value in (
            ("trajectory_id", self.trajectory_id),
            ("center_index", self.center_index),
            ("slice_start", self.slice_start),
            ("slice_end", self.slice_end),
            ("context_text", self.context_text),
            ("goal", self.goal),
        ):
            if len(value) != count:
                raise ValueError(f"text index {name} is misaligned")
        self.embedding /= np.maximum(
            np.linalg.norm(self.embedding, axis=1, keepdims=True), 1e-12
        )

    def __len__(self) -> int:
        return len(self.embedding)

    def query(
        self,
        query_embedding: np.ndarray,
        *,
        trajectory_ids: Iterable[str] | None = None,
        top_k: int = 6,
    ) -> list[TextRetrievalHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape != (self.embedding.shape[1],):
            raise ValueError("query embedding has the wrong dimension")
        norm = float(np.linalg.norm(query))
        if norm == 0 or not np.isfinite(query).all():
            raise ValueError("query embedding must be finite and nonzero")
        allowed = None if trajectory_ids is None else {str(value) for value in trajectory_ids}
        indexes = np.arange(len(self.embedding))
        if allowed is not None:
            indexes = indexes[np.isin(self.trajectory_id, sorted(allowed))]
        if not len(indexes):
            return []
        scores = self.embedding[indexes] @ (query / norm)
        order = np.argsort(-scores, kind="stable")[:top_k]
        output: list[TextRetrievalHit] = []
        for position in order:
            row = int(indexes[int(position)])
            output.append(TextRetrievalHit(
                trajectory_id=str(self.trajectory_id[row]),
                center_index=int(self.center_index[row]),
                score=float(scores[int(position)]),
                context_text=str(self.context_text[row]),
                goal=str(self.goal[row]),
                slice_start=int(self.slice_start[row]),
                slice_end=int(self.slice_end[row]),
            ))
        return output


class TextEmbeddingEncoder:
    """SentenceTransformer wrapper with official query instruction formatting."""

    def __init__(self, model_path: str | Path, *, device: str = "cuda:0") -> None:
        from sentence_transformers import SentenceTransformer
        import torch

        self.model_path = Path(model_path).resolve()
        self.device = device
        config_path = self.model_path / "config.json"
        model_type = ""
        if config_path.is_file():
            model_type = str(json.loads(config_path.read_text()).get("model_type") or "")
        self.multimodal_text_payload = model_type.startswith("qwen3_vl")
        self.model = SentenceTransformer(
            str(self.model_path),
            device=device,
            trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16},
        )

    def _payload(self, texts: list[str]) -> list[str] | list[dict[str, str]]:
        if self.multimodal_text_payload:
            return [{"text": text} for text in texts]
        return texts

    def encode_texts(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 16,
        max_input_tokens: int = 4096,
    ) -> np.ndarray:
        if max_input_tokens < 1:
            raise ValueError("max_input_tokens must be positive")
        values = []
        for text in texts:
            token_ids = self.model.tokenizer.encode(
                str(text), add_special_tokens=False
            )
            if len(token_ids) > max_input_tokens:
                text = self.model.tokenizer.decode(
                    token_ids[:max_input_tokens],
                    skip_special_tokens=True,
                )
            values.append(str(text))
        if not values:
            return np.zeros((0, 4096), dtype=np.float32)
        vectors = np.asarray(self.model.encode(
            self._payload(values),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(values) > batch_size,
        ), dtype=np.float32)
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("text encoder returned invalid embeddings")
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        return vectors

    def encode_query(
        self,
        question: str,
        *,
        instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ) -> np.ndarray:
        formatted = f"Instruct: {instruction.strip()}\nQuery:{str(question).strip()}"
        vectors = self.encode_texts([formatted], batch_size=1)
        if vectors.shape[0] != 1:
            raise ValueError("query encoder returned no vector")
        return vectors[0]


def format_raw_state_context(
    *,
    trajectory_id: str,
    goal: str,
    center_index: int,
    slice_state_indexes: list[int],
    full_action_sequence: str,
    local_action_sequence: str,
    states: list[dict],
) -> str:
    """Format one hit in the same shape used by the official RAG context."""
    blocks = []
    for state in states:
        action = state.get("incoming_action_text") or state.get("action") or "<none>"
        blocks.append(
            "\n".join([
                f"State {state.get('state_index', '')} (step {state.get('step_idx', '')})",
                f"- URL: {state.get('url', '<unknown>')}",
                f"- Action: {action}",
                "- AXTree:",
                str(state.get("axtree") or state.get("synthetic_axtree") or ""),
            ])
        )
    return "\n".join([
        f"- Trajectory: {trajectory_id}",
        f"- Goal: {goal}",
        f"- Center state index: {center_index}",
        "",
        "Full action sequence",
        full_action_sequence,
        "",
        "Local slice action sequence",
        local_action_sequence,
        "",
        "Relevant state slices",
        "\n\n".join(blocks),
    ])
