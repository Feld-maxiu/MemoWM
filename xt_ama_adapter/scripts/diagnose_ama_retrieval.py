"""Diagnose AMA web retrieval quality before committing to the ablation arms.

For every WEB question in the per-episode caches this script reports the
head-based top-k step ranking (the exact ranking the answer stage consumes),
the top1-top2 score margin (a collapsed head shows up as near-zero margins
everywhere) and, optionally, the overlap with the official AMA-Bench embedding
retrieval (mean-pooled, 512-token truncated documents, cosine top-k, see
``third_party/AMA-Bench/src/method/embedding_mem.py``).

Run this before the five ablation arms; if the head has no discriminability,
fix retrieval first instead of tuning prompts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.state_tokenizer.run_ama_web_latent import (
    _episode_ids,
    _load_cache_payloads,
    web_episodes,
)

DEFAULT_EMBEDDING_MODEL = "/data1/models/qwen3-embedding-4B"
MARGIN_EPSILON = 0.01


def _official_top_indices(texts: list[str], question: str, model: Any,
                          tokenizer: Any, device: torch.device,
                          top_k: int) -> list[int]:
    """Official embedding_mem retrieval: mean pooling, 512-token cap, cosine."""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    mask = (inputs["attention_mask"].unsqueeze(-1)
            .expand(outputs.last_hidden_state.size()).float())
    summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    docs = (summed / counts).cpu().numpy().astype(np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9
    question_embedding = tokenizer([question], padding=True, truncation=True,
                                   max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        question_output = model(**question_embedding)
    mask = (question_embedding["attention_mask"].unsqueeze(-1)
            .expand(question_output.last_hidden_state.size()).float())
    query = (torch.sum(question_output.last_hidden_state * mask, dim=1)
             / torch.clamp(mask.sum(dim=1), min=1e-9)).cpu().numpy()
    query = query.astype(np.float32)[0]
    query /= np.linalg.norm(query) + 1e-9
    scores = docs @ query
    return np.argsort(-scores)[:top_k].tolist()


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    skipped = _episode_ids(args.skip_episode_ids)
    eligible = [row for row in web_episodes(Path(args.test_file))
                if int(row["episode_id"]) not in skipped]
    files = sorted(Path(args.cache_dir).glob("episode-*.pt"))
    payloads = _load_cache_payloads(files, needs_texts=args.official_overlap)

    official_model = official_tokenizer = None
    device = torch.device(args.device)
    if args.official_overlap:
        import transformers
        official_tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.embedding_model, trust_remote_code=True)
        official_model = transformers.AutoModel.from_pretrained(
            args.embedding_model, torch_dtype=torch.float32,
            trust_remote_code=True).to(device).eval()

    rows: list[dict[str, Any]] = []
    margins: list[float] = []
    top1_scores: list[float] = []
    position_counter: Counter[int] = Counter()
    overlaps: list[float] = []
    for episode_id in sorted(payloads):
        payload = payloads[episode_id]
        docs = payload["document_embeddings"].float().numpy()
        queries = payload["query_embeddings"].float().numpy()
        step_indices = payload["step_indices"].tolist()
        step_texts = payload.get("step_texts") or []
        for qa_index, question in enumerate(payload["questions"]):
            scores = docs @ queries[qa_index]
            order = np.argsort(-scores)[:args.top_k]
            margin = (float(scores[order[0]] - scores[order[1]])
                      if len(order) > 1 else 0.0)
            margins.append(margin)
            top1_scores.append(float(scores[order[0]]))
            position_counter[int(order[0])] += 1
            row: dict[str, Any] = {
                "episode_id": episode_id,
                "qa_index": qa_index,
                "question": question,
                "top_step_indices": [int(step_indices[position])
                                     for position in order],
                "top_positions": [int(position) for position in order],
                "top_scores": [float(scores[position]) for position in order],
                "top1_top2_margin": margin,
            }
            if args.official_overlap:
                official_top = _official_top_indices(
                    step_texts, question, official_model, official_tokenizer,
                    device, args.top_k)
                overlap = len(set(order.tolist()) & set(official_top))
                overlaps.append(overlap / min(args.top_k, len(step_texts)))
                row["official_top_positions"] = official_top
                row["official_overlap_ratio"] = overlap / min(
                    args.top_k, len(step_texts))
            rows.append(row)

    report: dict[str, Any] = {
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "episodes": len(payloads),
        "questions": len(rows),
        "top_k": args.top_k,
        "mean_top1_score": float(np.mean(top1_scores)) if top1_scores else 0.0,
        "mean_top1_top2_margin": float(np.mean(margins)) if margins else 0.0,
        "ambiguous_margin_ratio": (
            float(np.mean([margin < MARGIN_EPSILON for margin in margins]))
            if margins else 0.0
        ),
        "head_discriminative": bool(margins) and float(np.mean(margins)) > MARGIN_EPSILON,
        "top1_position_histogram": dict(sorted(
            position_counter.items(), key=lambda item: -item[1])),
        "official_overlap": (None if not args.official_overlap else {
            "embedding_model": args.embedding_model,
            "mean_overlap_ratio": float(np.mean(overlaps)) if overlaps else 0.0,
        }),
    }
    return {"report": report, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--skip-episode-ids", default="184")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--no-official-overlap", dest="official_overlap",
                        action="store_false",
                        help="skip the official embedding retrieval comparison")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    result = diagnose(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result["report"], indent=2, sort_keys=True))
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in result["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result["report"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
