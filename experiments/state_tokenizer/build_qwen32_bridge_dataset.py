"""Build retrieval-realistic HumanTrajs examples for the frozen Qwen3-32B reader.

This stage deliberately runs *after* the Q-Former cache and standalone
retrieval head have been accepted.  Each question is embedded in the fused
teacher coordinate system, all unique states from its trajectory are ranked by
the bound retrieval head, and the actual top-k K32 states are materialized.
Gold states are measured, never forced into the result.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from residualmem.latent.instruct_bridge import (
    BRIDGE_CACHE_PROTOCOL,
    FUSED_OBSERVATION_TEACHER_PROTOCOL,
    RETRIEVAL_BRIDGE_PROTOCOL,
    MaskedAttentionRetrievalHead,
)
from xt_ama_adapter.qwen32_bridge import file_sha256


PROTOCOL = "humantrajs_qwen32_retrieved_latents_v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("instruction") is not None:
                raise ValueError(f"{path}:{line_number}: instruction leakage")
            if row.get("verification_status") != "pseudo_web_text_verified":
                raise ValueError(f"{path}:{line_number}: QA is not text verified")
            if row.get("qa_input_protocol") != "vl-pseudo-web-observation-v1":
                raise ValueError(f"{path}:{line_number}: unexpected QA protocol")
            if row.get("split") not in {"train", "validation", "test"}:
                raise ValueError(f"{path}:{line_number}: invalid split")
            rows.append(row)
    if not rows:
        raise ValueError("QA input is empty")
    return rows


def _load_state_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        required = {"xbar", "valid", "split", "sample_id", "step_index", "metadata"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"QFormer cache lacks {sorted(missing)}")
        arrays = {name: np.asarray(data[name]) for name in required - {"metadata"}}
        metadata = json.loads(str(np.asarray(data["metadata"]).item()))
    if metadata.get("protocol") != BRIDGE_CACHE_PROTOCOL:
        raise ValueError("QFormer cache protocol mismatch")
    if metadata.get("teacher_protocol") != FUSED_OBSERVATION_TEACHER_PROTOCOL:
        raise ValueError("QFormer cache is not aligned to the fused teacher")
    if metadata.get("queries") != 32 or metadata.get("qk_norm") is not True:
        raise ValueError("QFormer cache must be K32 with qk_norm=true")
    if arrays["xbar"].shape[1:] != (32, 512):
        raise ValueError(f"unexpected xbar shape {arrays['xbar'].shape}")
    if arrays["valid"].shape != arrays["xbar"].shape[:2]:
        raise ValueError("valid mask shape does not match xbar")
    return arrays, metadata


def _load_head(path: Path, cache_metadata: dict[str, Any], device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("protocol") != RETRIEVAL_BRIDGE_PROTOCOL:
        raise ValueError("retrieval-head protocol mismatch")
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("qformer_artifact_sha256") != cache_metadata.get(
        "qformer_artifact_sha256"
    ):
        raise ValueError("retrieval head and QFormer cache artifact hashes differ")
    if metadata.get("teacher_protocol") != FUSED_OBSERVATION_TEACHER_PROTOCOL:
        raise ValueError("retrieval head teacher protocol mismatch")
    if metadata.get("queries") != 32 or metadata.get("qk_norm") is not True:
        raise ValueError("retrieval head must be bound to K32/QK-norm states")
    head = MaskedAttentionRetrievalHead()
    head.load_state_dict(payload["state_dict"], strict=True)
    return head.to(device).eval(), metadata


def _unique_states(arrays: dict[str, np.ndarray]):
    """Deduplicate multiple QA rows that point at the same trajectory step."""
    by_trajectory: dict[str, list[int]] = collections.defaultdict(list)
    seen: dict[tuple[str, int], int] = {}
    for index, (sample_id, step_index) in enumerate(
        zip(arrays["sample_id"].astype(str), arrays["step_index"].astype(int))
    ):
        key = (sample_id, int(step_index))
        previous = seen.get(key)
        if previous is not None:
            if not np.array_equal(arrays["xbar"][previous], arrays["xbar"][index]):
                raise ValueError(f"duplicate state {key} has different xbar values")
            if not np.array_equal(arrays["valid"][previous], arrays["valid"][index]):
                raise ValueError(f"duplicate state {key} has different valid masks")
            continue
        seen[key] = index
        by_trajectory[sample_id].append(index)
    for indices in by_trajectory.values():
        indices.sort(key=lambda i: int(arrays["step_index"][i]))
    return by_trajectory


def _encode_questions(model_path: str, questions: list[str], device: str,
                      batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_path,
        device=device,
        trust_remote_code=True,
        model_kwargs={"dtype": torch.bfloat16},
    )
    values = model.encode(
        [{"text": question} for question in questions],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(questions), 4096) or not np.isfinite(values).all():
        raise ValueError(f"unexpected query embedding array {values.shape}")
    return values


def _query_model_manifest_hash(model_path: Path) -> str:
    """Stable identity without rereading all 8B weight shards."""
    names = ["config.json", "modules.json", "model.safetensors.index.json",
             "tokenizer_config.json", "tokenizer.json"]
    digest = hashlib.sha256()
    for name in names:
        path = model_path / name
        if not path.exists():
            continue
        digest.update(name.encode())
        digest.update(str(path.stat().st_size).encode())
        digest.update(bytes.fromhex(file_sha256(path)))
    value = digest.hexdigest()
    if value == hashlib.sha256().hexdigest():
        raise ValueError(f"no query-model manifest files found under {model_path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="accepted QFormer xbar cache")
    parser.add_argument("--head", required=True, help="accepted, cache-bound retrieval head")
    parser.add_argument("--qa", required=True, help="verified HumanTrajs QA JSONL")
    parser.add_argument("--query-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("top-k must be positive")

    cache_path, head_path, qa_path = map(Path, (args.cache, args.head, args.qa))
    arrays, cache_metadata = _load_state_cache(cache_path)
    rows = _read_jsonl(qa_path)
    by_trajectory = _unique_states(arrays)
    device = torch.device(args.device)
    head, head_metadata = _load_head(head_path, cache_metadata, device)

    doc_vectors = []
    with torch.inference_mode():
        for start in range(0, len(arrays["xbar"]), args.head_batch_size):
            stop = start + args.head_batch_size
            vector = head(
                torch.as_tensor(arrays["xbar"][start:stop], dtype=torch.float32,
                                device=device),
                torch.as_tensor(arrays["valid"][start:stop], dtype=torch.bool,
                                device=device),
            )
            doc_vectors.append(F.normalize(vector.float(), dim=-1).cpu().numpy())
    doc_vectors_np = np.concatenate(doc_vectors).astype(np.float32)

    questions = [str(row["question"]).strip() for row in rows]
    answers = [str(row["answer"]).strip() for row in rows]
    oracle_text = [str(row.get("text_observation", "")).strip() for row in rows]
    if any(not value for value in questions + answers + oracle_text):
        raise ValueError("question, answer and verified observation text must be non-empty")
    query_vectors = _encode_questions(
        args.query_model, questions, args.device, args.query_batch_size
    )

    selected_xbar, selected_valid, selected_steps, selected_scores = [], [], [], []
    gold_hit, candidate_counts = [], []
    for row, query in zip(rows, query_vectors):
        trajectory = str(row["trajectory_id"])
        candidates = by_trajectory.get(trajectory)
        if not candidates:
            raise ValueError(f"trajectory {trajectory} has no cached states")
        scores = doc_vectors_np[candidates] @ query
        order = np.argsort(-scores, kind="stable")[: args.top_k]
        chosen = [candidates[int(position)] for position in order]
        count = len(chosen)
        xbar = np.zeros((args.top_k, 32, 512), dtype=np.float32)
        valid = np.zeros((args.top_k, 32), dtype=bool)
        steps = np.full(args.top_k, -1, dtype=np.int64)
        values = np.full(args.top_k, -np.inf, dtype=np.float32)
        xbar[:count] = arrays["xbar"][chosen]
        valid[:count] = arrays["valid"][chosen]
        steps[:count] = arrays["step_index"][chosen]
        values[:count] = scores[order]
        gold = {int(step) for group in row.get("gold_step_groups", []) for step in group}
        if not gold:
            raise ValueError(f"trajectory {trajectory}: QA has no gold steps")
        selected_xbar.append(xbar)
        selected_valid.append(valid)
        selected_steps.append(steps)
        selected_scores.append(values)
        gold_hit.append(bool(gold.intersection(map(int, steps[:count]))))
        candidate_counts.append(len(candidates))

    split = np.asarray([row["split"] for row in rows])
    counts = collections.Counter(split.tolist())
    hit = np.asarray(gold_hit, dtype=bool)
    recall = {
        name: float(hit[split == name].mean()) if np.any(split == name) else None
        for name in ("train", "validation", "test")
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        xbar=np.stack(selected_xbar),
        valid=np.stack(selected_valid),
        retrieved_step_index=np.stack(selected_steps),
        retrieval_score=np.stack(selected_scores),
        retrieval_gold_hit=hit,
        candidate_count=np.asarray(candidate_counts, dtype=np.int64),
        trajectory_id=np.asarray([row["trajectory_id"] for row in rows]),
        question=np.asarray(questions),
        answer=np.asarray(answers),
        oracle_text=np.asarray(oracle_text),
        split=split,
        metadata=np.asarray(json.dumps({
            "protocol": PROTOCOL,
            "top_k": args.top_k,
            "slots": 32,
            "state_dimension": 512,
            "selection": "retrieval_head_topk_no_gold_forcing",
            "split_source": "humantrajs_trajectory_manifest",
            "official_ama_test_included": False,
            "qa_sha256": file_sha256(qa_path),
            "qformer_cache_sha256": file_sha256(cache_path),
            "qformer_artifact_sha256": cache_metadata["qformer_artifact_sha256"],
            "retrieval_head_artifact_sha256": file_sha256(head_path),
            "retrieval_head_best_step": head_metadata.get("best_step"),
            "query_model_manifest_sha256": _query_model_manifest_hash(
                Path(args.query_model)
            ),
            "counts": dict(counts),
            "gold_recall_at_k": recall,
        }, sort_keys=True)),
    )
    print(json.dumps({"output": str(output.resolve()), "rows": len(rows),
                      "counts": dict(counts), "gold_recall_at_k": recall},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
