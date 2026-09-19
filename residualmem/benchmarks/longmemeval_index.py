"""In-memory index for the standalone LongMemEval latent cache."""
from __future__ import annotations

import collections
import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit

import numpy as np

from .longmemeval_compact import compact_state_text


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).casefold())


class _BM25Index:
    """Small deterministic BM25 index over the retrieval sidecar text."""

    def __init__(self, documents: Iterable[str], *, k1: float = 1.2, b: float = 0.75):
        tokens = [_tokenize(document) for document in documents]
        self.k1 = float(k1)
        self.b = float(b)
        self.lengths = np.asarray([len(row) for row in tokens], dtype=np.float32)
        self.average_length = float(self.lengths.mean()) if len(self.lengths) else 0.0
        frequencies: list[collections.Counter[str]] = [
            collections.Counter(row) for row in tokens
        ]
        document_frequency: collections.Counter[str] = collections.Counter()
        postings: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for row, terms in enumerate(frequencies):
            for term, frequency in terms.items():
                document_frequency[term] += 1
                postings[term].append((row, int(frequency)))
        count = len(tokens)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        self.postings = dict(postings)

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.lengths), dtype=np.float64)
        query_terms = collections.Counter(_tokenize(query))
        if not query_terms or not len(self.lengths):
            return scores
        average = max(self.average_length, 1e-6)
        for term, query_frequency in query_terms.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for row, frequency in postings:
                length_norm = 1.0 - self.b + self.b * self.lengths[row] / average
                denominator = frequency + self.k1 * length_norm
                if denominator <= 0:
                    continue
                scores[row] += (
                    idf
                    * query_frequency
                    * (frequency * (self.k1 + 1.0))
                    / denominator
                )
        return scores


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return values
    deviation = float(values.std())
    if not np.isfinite(deviation) or deviation <= 1e-12:
        return np.zeros_like(values)
    return (values - float(values.mean())) / deviation


def _page_title(record: dict) -> str:
    tree = str(record.get("axtree") or record.get("synthetic_axtree") or "")
    match = re.search(r"RootWebArea ['\"](.+?)['\"]", tree)
    return " ".join(match.group(1).split()) if match else ""


def _compact_action(record: dict) -> str:
    value = record.get("incoming_action_text") or record.get("action")
    if value is None:
        return ""
    if isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(str(value).split())[:160]


@dataclasses.dataclass(frozen=True)
class RetrievedObservation:
    trajectory_id: str
    record_index: int
    score: float
    xbar: np.ndarray
    valid: np.ndarray
    key: np.ndarray
    anchor_text: str
    record: dict
    latent_score: float = 0.0
    bm25_score: float = 0.0


class LongMemEvalLatentIndex:
    """Exact cosine retrieval over precomputed Q-Former keys."""

    def __init__(self, cache_dir: str | Path, *, sidecar: str = "stored"):
        """``sidecar`` selects the text payload bound to each latent row.

        ``stored`` keeps the sparse exact-value anchor produced at write time.
        ``compact`` replaces it with the rules-1-3 compact exact-value view of
        the same observation, so the reader receives latent semantics plus a
        denser exact-value payload.  Retrieval keys are untouched either way.
        """
        if sidecar not in {"stored", "compact"}:
            raise ValueError(f"unknown sidecar source: {sidecar}")
        self.cache_dir = Path(cache_dir).resolve()
        self._rows: list[RetrievedObservation] = []
        self._netlocs_by_trajectory: dict[str, set[str]] = collections.defaultdict(set)
        for path in sorted((self.cache_dir / "trajectories").glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(np.asarray(data["metadata"])))
                xbar = np.asarray(data["xbar"], np.float32)
                valid = np.asarray(data["valid"], np.bool_)
                keys = np.asarray(data["key"], np.float32)
            records = metadata.get("records") or []
            if xbar.ndim != 3 or valid.ndim != 2 or keys.ndim != 2:
                raise ValueError(f"{path}: invalid cache dimensions")
            if not (len(xbar) == len(valid) == len(keys) == len(records)):
                raise ValueError(f"{path}: cache arrays and records are misaligned")
            trajectory_id = str(metadata.get("trajectory_id", path.stem))
            for index, record in enumerate(records):
                key = keys[index]
                norm = float(np.linalg.norm(key))
                if norm == 0 or not np.isfinite(key).all():
                    raise ValueError(f"{path}[{index}]: invalid retrieval key")
                anchor_text = str(record.get("anchor_text") or "")
                if sidecar == "compact":
                    anchor_text = (
                        "Exact-value sidecar for the compressed observation "
                        "(not a complete UI listing):\n"
                        + compact_state_text(record)
                    )
                self._rows.append(RetrievedObservation(
                    trajectory_id=trajectory_id,
                    record_index=index,
                    score=0.0,
                    xbar=xbar[index],
                    valid=valid[index],
                    key=key / norm,
                    anchor_text=anchor_text,
                    record=dict(record),
                ))
                netloc = urlsplit(str(record.get("url") or "")).netloc
                if netloc:
                    self._netlocs_by_trajectory[trajectory_id].add(netloc)
        self._bm25 = _BM25Index(row.anchor_text for row in self._rows)

    @property
    def size(self) -> int:
        return len(self._rows)

    def observation(
        self, trajectory_id: str, record_index: int
    ) -> RetrievedObservation | None:
        """Return the stored row for one observation identity, if present."""
        lookup = getattr(self, "_by_key", None)
        if lookup is None:
            lookup = {
                (row.trajectory_id, row.record_index): row for row in self._rows
            }
            self._by_key = lookup
        return lookup.get((str(trajectory_id), int(record_index)))

    def trajectory_ids_for_netloc(self, netloc: str) -> set[str]:
        requested = str(netloc).strip().casefold()
        if not requested:
            return set()
        return {
            trajectory_id
            for trajectory_id, found in self._netlocs_by_trajectory.items()
            if requested in {value.casefold() for value in found}
        }

    def trajectory_goal_texts(self) -> dict[str, str]:
        """Return one query-independent summary per trajectory for coarse routing."""
        goals: dict[str, str] = {}
        urls: dict[str, str] = {}
        titles: dict[str, list[str]] = collections.defaultdict(list)
        actions: dict[str, list[str]] = collections.defaultdict(list)
        for row in self._rows:
            record = row.record
            trajectory_id = row.trajectory_id
            goal = str(record.get("goal") or "").strip()
            if goal and trajectory_id not in goals:
                goals[trajectory_id] = goal
            url = str(record.get("url") or "").strip()
            if url and trajectory_id not in urls:
                urls[trajectory_id] = url
            title = _page_title(record)
            if title and title not in titles[trajectory_id]:
                titles[trajectory_id].append(title)
            action = _compact_action(record)
            if action and action not in actions[trajectory_id]:
                actions[trajectory_id].append(action)

        output: dict[str, str] = {}
        for trajectory_id in sorted(goals | urls | titles | actions):
            parts = []
            if goals.get(trajectory_id):
                parts.append(f"Task: {goals[trajectory_id]}")
            if urls.get(trajectory_id):
                parts.append(f"Start URL: {urls[trajectory_id]}")
            if titles[trajectory_id]:
                parts.append("Pages: " + " | ".join(titles[trajectory_id][:8]))
            if actions[trajectory_id]:
                parts.append("Actions: " + " | ".join(actions[trajectory_id][:6]))
            output[trajectory_id] = " ".join(parts) or trajectory_id
        return output

    def query(
        self,
        query_key: np.ndarray,
        *,
        trajectory_ids: Iterable[str] | None = None,
        top_k: int = 6,
    ) -> list[RetrievedObservation]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = np.asarray(query_key, np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm == 0 or not np.isfinite(query).all():
            raise ValueError("query key must be finite and non-zero")
        allowed = None if trajectory_ids is None else {str(item) for item in trajectory_ids}
        rows = [row for row in self._rows if allowed is None or row.trajectory_id in allowed]
        scored = [dataclasses.replace(row, score=float(np.dot(query / norm, row.key))) for row in rows]
        scored.sort(key=lambda row: (-row.score, row.trajectory_id, row.record_index))
        return scored[:top_k]

    def query_hybrid(
        self,
        query_key: np.ndarray,
        query_text: str,
        *,
        trajectory_ids: Iterable[str] | None = None,
        site: str | None = None,
        top_k: int = 6,
        bm25_weight: float = 0.5,
        trajectory_top_k: int = 1,
        trajectory_aggregate_k: int = 3,
        trajectory_prior: dict[str, float] | None = None,
        trajectory_prior_weight: float = 0.0,
        trajectory_hub_weight: float = 0.0,
    ) -> list[RetrievedObservation]:
        """Fuse one latent channel and one anchor-BM25 channel per observation.

        The two scores are standardized within the allowed candidate set, then
        fused on the same observation identity. When ``trajectory_top_k`` is 1,
        the returned observations are guaranteed to come from one trajectory.
        """
        if top_k < 1 or trajectory_top_k < 1 or trajectory_aggregate_k < 1:
            raise ValueError("top_k and trajectory sizes must be positive")
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be in [0, 1]")
        if not 0.0 <= trajectory_prior_weight <= 1.0:
            raise ValueError("trajectory_prior_weight must be in [0, 1]")
        if not 0.0 <= trajectory_hub_weight <= 1.0:
            raise ValueError("trajectory_hub_weight must be in [0, 1]")
        query = np.asarray(query_key, np.float32).reshape(-1)
        norm = float(np.linalg.norm(query))
        if norm == 0 or not np.isfinite(query).all():
            raise ValueError("query key must be finite and non-zero")

        allowed = None if trajectory_ids is None else {str(item) for item in trajectory_ids}
        if site is not None:
            site_ids = self.trajectory_ids_for_netloc(site)
            allowed = site_ids if allowed is None else allowed & site_ids
        row_indices = np.asarray([
            index for index, row in enumerate(self._rows)
            if allowed is None or row.trajectory_id in allowed
        ], dtype=np.int64)
        if not len(row_indices):
            raise ValueError("no observations remain after trajectory/site filtering")

        latent = np.asarray([
            float(np.dot(query / norm, self._rows[index].key))
            for index in row_indices
        ], dtype=np.float64)
        bm25 = self._bm25.scores(query_text)[row_indices]
        fused = (
            (1.0 - bm25_weight) * _standardize(latent)
            + bm25_weight * _standardize(bm25)
        )

        scores_by_trajectory: dict[str, list[float]] = collections.defaultdict(list)
        for position, row_index in enumerate(row_indices):
            scores_by_trajectory[self._rows[int(row_index)].trajectory_id].append(
                float(fused[position])
            )
        trajectory_scores = {}
        for trajectory_id, values in scores_by_trajectory.items():
            ordered = sorted(values, reverse=True)
            trajectory_scores[trajectory_id] = float(
                np.mean(ordered[:trajectory_aggregate_k])
            )
        if trajectory_hub_weight > 0.0:
            global_top = float(fused.max())
            for trajectory_id, values in scores_by_trajectory.items():
                support = np.mean(
                    np.asarray(values, dtype=np.float64) >= global_top - 0.5
                )
                trajectory_scores[trajectory_id] -= (
                    trajectory_hub_weight * float(support)
                )
        if trajectory_prior is not None and trajectory_prior_weight > 0.0:
            prior_ids = sorted(trajectory_scores)
            prior_values = np.asarray(
                [float(trajectory_prior.get(trajectory_id, 0.0)) for trajectory_id in prior_ids],
                dtype=np.float64,
            )
            standardized_observation = _standardize(
                np.asarray([trajectory_scores[t] for t in prior_ids], dtype=np.float64)
            )
            standardized_prior = _standardize(prior_values)
            for position, trajectory_id in enumerate(prior_ids):
                trajectory_scores[trajectory_id] = float(
                    (1.0 - trajectory_prior_weight) * standardized_observation[position]
                    + trajectory_prior_weight * standardized_prior[position]
                )
        selected_trajectories = {
            trajectory_id
            for trajectory_id, _ in sorted(
                trajectory_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )[:trajectory_top_k]
        }

        candidates: list[tuple[float, int, int, float, float]] = []
        for position, row_index in enumerate(row_indices):
            row = self._rows[int(row_index)]
            if row.trajectory_id not in selected_trajectories:
                continue
            candidates.append((
                float(fused[position]),
                int(row_index),
                position,
                float(latent[position]),
                float(bm25[position]),
            ))
        candidates.sort(key=lambda item: (-item[0], self._rows[item[1]].trajectory_id,
                                          self._rows[item[1]].record_index))
        output = []
        for fused_score, row_index, _position, latent_score, bm25_score in candidates[:top_k]:
            output.append(dataclasses.replace(
                self._rows[row_index],
                score=fused_score,
                latent_score=latent_score,
                bm25_score=bm25_score,
            ))
        return output

    def expanded_context(
        self,
        hits: Iterable[RetrievedObservation],
        *,
        radius: int = 1,
        max_observations: int = 18,
    ) -> list[RetrievedObservation]:
        """Expand hits by adjacent states without crossing trajectories."""
        if radius < 0 or max_observations < 1:
            raise ValueError("radius must be non-negative and max_observations positive")
        by_key = {(row.trajectory_id, row.record_index): row for row in self._rows}
        output: list[RetrievedObservation] = []
        seen: set[tuple[str, int]] = set()
        for hit in hits:
            for index in range(max(0, hit.record_index - radius), hit.record_index + radius + 1):
                key = (hit.trajectory_id, index)
                row = by_key.get(key)
                if row is None or key in seen:
                    continue
                seen.add(key)
                output.append(hit if index == hit.record_index else row)
                if len(output) >= max_observations:
                    return output
        return output


def diverse_hits(candidates, *, top_k=8, relevance_weight=.9, max_per_trajectory=2,
                 min_index_distance=2):
    """Rerank only latent-key candidates; no anchor/text/answer scoring."""
    if top_k < 1 or not 0 <= relevance_weight <= 1 or max_per_trajectory < 1:
        raise ValueError("invalid diversity settings")
    selected = []
    remaining = list(candidates)
    while remaining and len(selected) < top_k:
        eligible = [row for row in remaining if
                    sum(hit.trajectory_id == row.trajectory_id for hit in selected) < max_per_trajectory
                    and not any(hit.trajectory_id == row.trajectory_id and
                                abs(hit.record_index - row.record_index) <= min_index_distance
                                for hit in selected)]
        if not eligible:
            break
        def score(row):
            redundancy = max((float(row.key @ hit.key) for hit in selected), default=0.0)
            return relevance_weight * row.score - (1 - relevance_weight) * redundancy
        best = min(eligible, key=lambda row: (-score(row), row.trajectory_id, row.record_index))
        selected.append(best)
        remaining = [row for row in remaining if
                     (row.trajectory_id, row.record_index) != (best.trajectory_id, best.record_index)]
    return selected


class LocalVLQueryEncoder:
    """Query-side encoder paired with the frozen VL document teacher."""

    def __init__(self, model_path: str | Path, *, device: str = "cuda:0"):
        from sentence_transformers import SentenceTransformer
        import torch

        self.model_path = str(Path(model_path).resolve())
        self.model = SentenceTransformer(
            self.model_path,
            device=device,
            trust_remote_code=True,
            model_kwargs={"dtype": torch.bfloat16},
        )

    def encode(self, question: str, image: str | Path | None = None) -> np.ndarray:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty")
        payload: dict[str, str] = {"text": question}
        if image is not None:
            image_path = Path(image).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            payload["image"] = str(image_path)
        vector = np.asarray(self.model.encode(
            [payload], convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )[0], dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("query encoder returned an invalid vector")
        return vector

    def encode_many(self, texts: Iterable[str]) -> np.ndarray:
        """Encode query-independent summaries in one batch."""
        values = [str(text).strip() or "No task summary recorded." for text in texts]
        if not values:
            return np.zeros((0, 4096), dtype=np.float32)
        vectors = np.asarray(self.model.encode(
            [{"text": text} for text in values],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ), dtype=np.float32)
        if vectors.shape != (len(values), 4096) or not np.isfinite(vectors).all():
            raise ValueError("query encoder returned invalid batch vectors")
        return vectors
