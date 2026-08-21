"""Strict Raw-Fused vs Xbar retrieval smoke on WorldMemArena web QA.

This intentionally skips answer generation and LLM judging.  It exercises the
official session/checkpoint runner, the official Qwen3-VL document/query
encoder, the frozen Qwen3.5 tokenizer, and the trained Xbar retrieval head.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def _top_rows(adapter, record) -> list[dict]:
    kinds = {row["memory_id"]: row.get("row_kind", "unknown") for row in adapter._rounds}
    return [
        {
            "rank": int(item.rank),
            "memory_id": item.memory_id,
            "score": float(item.score),
            "row_kind": kinds.get(item.memory_id, "unknown"),
            "image_path": item.image_path,
            "text": item.text[:300],
        }
        for item in record.retrieval.items
    ]


def _paired_observation_metrics(raw, xbar) -> dict:
    raw_by_id = {
        row["memory_id"]: np.asarray(vector, np.float32)
        for row, vector in zip(raw._rounds, raw._embeddings)
        if row.get("row_kind") == "fused_observation"
    }
    xbar_by_id = {
        row["memory_id"]: np.asarray(vector, np.float32)
        for row, vector in zip(xbar._rounds, xbar._embeddings)
        if row.get("row_kind") == "residualmem_xbar"
    }
    if set(raw_by_id) != set(xbar_by_id):
        raise ValueError(
            f"observation row mismatch: raw-only={sorted(set(raw_by_id)-set(xbar_by_id))}, "
            f"xbar-only={sorted(set(xbar_by_id)-set(raw_by_id))}"
        )
    ids = sorted(raw_by_id)
    if not ids:
        return {"states": 0}
    teacher = np.stack([raw_by_id[key] for key in ids])
    student = np.stack([xbar_by_id[key] for key in ids])
    teacher /= np.maximum(np.linalg.norm(teacher, axis=1, keepdims=True), 1e-12)
    student /= np.maximum(np.linalg.norm(student, axis=1, keepdims=True), 1e-12)
    similarity = student @ teacher.T
    order = np.argsort(-similarity, axis=1)
    ranks = np.asarray([
        int(np.flatnonzero(order[index] == index)[0]) + 1
        for index in range(len(ids))
    ])
    return {
        "states": len(ids),
        "paired_cosine_mean": float(np.diag(similarity).mean()),
        "paired_cosine_min": float(np.diag(similarity).min()),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "memory_ids": ids,
    }


def run(args: argparse.Namespace) -> dict:
    from eval_framework.datasets.worldmemarena import load_worldmemarena
    from eval_framework.evaluators.qa import _ranking_metrics
    from eval_framework.memory_adapters.qwen_embed_adapter import (
        QwenVLFusedObservationAdapter,
    )
    from eval_framework.memory_adapters.residualmem_instruct_adapter import (
        ResidualMemInstructAdapter,
    )
    from eval_framework.pipeline.runner import run_eval_sample
    from residualmem.latent.frozen_v8_runtime import FrozenV9InstructTokenizer

    bundle = load_worldmemarena(Path(args.dataset), split=args.split)
    sample = next((row for row in bundle.samples if row.sample_id == args.sample_id), None)
    if sample is None:
        raise KeyError(f"sample {args.sample_id!r} not found")
    if getattr(sample, "_subcategory", "") != "agent/arena/web":
        raise ValueError(f"sample {args.sample_id!r} is not a web sample")
    checkpoint = sample.normalized_checkpoints[args.checkpoint_index]
    checkpoint = dataclasses.replace(
        checkpoint, questions=checkpoint.questions[: args.num_queries]
    )
    sample = dataclasses.replace(sample, normalized_checkpoints=(checkpoint,))
    session_index = {row.session_id: index for index, row in enumerate(sample.sessions)}
    max_sessions = max(session_index[sid] for sid in checkpoint.covered_sessions) + 1

    raw = QwenVLFusedObservationAdapter()
    tokenizer = FrozenV9InstructTokenizer(
        model_path=args.qwen35_model,
        pca_path=args.pca,
        normalization_path=args.normalization,
        device=args.tokenizer_device,
        use_kernels=False,
    )
    xbar = ResidualMemInstructAdapter(
        baseline_name="ResidualMem-Instruct-Xbar-Input-RAG",
        tokenizer=tokenizer,
    )

    dummy_answer = lambda _question, _retrieval: ""
    _raw_sessions, raw_qa = run_eval_sample(
        raw, sample, top_k=args.top_k, answer_fn=dummy_answer,
        max_sessions=max_sessions,
    )
    _xbar_sessions, xbar_qa = run_eval_sample(
        xbar, sample, top_k=args.top_k, answer_fn=dummy_answer,
        max_sessions=max_sessions,
    )
    if len(raw_qa) != len(xbar_qa) or len(raw_qa) != len(checkpoint.questions):
        raise ValueError("raw/xbar QA count mismatch")

    per_query = []
    for raw_record, xbar_record in zip(raw_qa, xbar_qa):
        if raw_record.question != xbar_record.question:
            raise ValueError("raw/xbar query order mismatch")
        raw_ids = [item.memory_id for item in raw_record.retrieval.items]
        xbar_ids = [item.memory_id for item in xbar_record.retrieval.items]
        common = set(raw_ids) & set(xbar_ids)
        raw_recall, raw_ndcg = _ranking_metrics(raw_record)
        xbar_recall, xbar_ndcg = _ranking_metrics(xbar_record)
        raw_rows = _top_rows(raw, raw_record)
        xbar_rows = _top_rows(xbar, xbar_record)
        raw_observation = {row["memory_id"] for row in raw_rows if row["row_kind"] == "fused_observation"}
        xbar_observation = {row["memory_id"] for row in xbar_rows if row["row_kind"] == "residualmem_xbar"}
        per_query.append({
            "question": raw_record.question,
            "gold_answer": raw_record.gold_answer,
            "gold_evidence_memory_ids": list(raw_record.gold_evidence_memory_ids),
            "top1_same": bool(raw_ids and xbar_ids and raw_ids[0] == xbar_ids[0]),
            "top10_overlap": len(common),
            "top10_overlap_fraction": len(common) / max(args.top_k, 1),
            "observation_top10_overlap": len(raw_observation & xbar_observation),
            "raw_recall_at": raw_recall,
            "xbar_recall_at": xbar_recall,
            "raw_ndcg_at": raw_ndcg,
            "xbar_ndcg_at": xbar_ndcg,
            "raw_rows": raw_rows,
            "xbar_rows": xbar_rows,
        })

    def mean(path: tuple[str, ...]) -> float:
        values = []
        for row in per_query:
            value = row
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.mean(values)) if values else 0.0

    raw_kinds = Counter(row.get("row_kind", "unknown") for row in raw._rounds)
    xbar_kinds = Counter(row.get("row_kind", "unknown") for row in xbar._rounds)
    return {
        "protocol": "worldmemarena_raw_fused_vs_xbar_retrieval_smoke_v1",
        "sample_id": sample.sample_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "covered_sessions": list(checkpoint.covered_sessions),
        "max_sessions": max_sessions,
        "queries": len(per_query),
        "top_k": args.top_k,
        "raw_memory_rows": len(raw._rounds),
        "xbar_memory_rows": len(xbar._rounds),
        "raw_row_kinds": dict(raw_kinds),
        "xbar_row_kinds": dict(xbar_kinds),
        "paired_observation_alignment": _paired_observation_metrics(raw, xbar),
        "aggregate": {
            "top1_agreement": mean(("top1_same",)),
            "top10_overlap_fraction": mean(("top10_overlap_fraction",)),
            "raw_recall_at_10": mean(("raw_recall_at", "10")),
            "xbar_recall_at_10": mean(("xbar_recall_at", "10")),
            "raw_ndcg_at_10": mean(("raw_ndcg_at", "10")),
            "xbar_ndcg_at_10": mean(("xbar_ndcg_at", "10")),
        },
        "per_query": per_query,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sample-id", default="web_01")
    parser.add_argument("--split", choices=("all", "small"), default="all")
    parser.add_argument("--checkpoint-index", type=int, default=0)
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--qwen35-model", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--tokenizer-device", default="cuda:1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    print(json.dumps({
        "output": str(output.resolve()),
        **report["aggregate"],
        "paired_observation_alignment": report["paired_observation_alignment"],
    }, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
