"""Stage C: is a latent as good to answer from as the text it replaces?

End-to-end QA on WorldMemArena cannot answer this. Only 0.30 of the ten rows
retrieved per question is an observation row -- the only kind that carries a
latent -- so whatever the connector does is drowned by the round-text rows that
surround it. The connector's own question is separable, and this measures it
directly by taking retrieval out of the loop.

For every question whose retrieval surfaced an observation row, that one row is
handed to the reader twice: once as the text the row holds, once as the 64 soft
tokens the connector produces from its ``xbar``. Same question, same frozen
Qwen3.5, same official judge. The gap between the two conditions is the
connector's fidelity, and nothing else.

Two things this deliberately does not do. It does not use gold-evidence matching
to pick the row -- the benchmark's matcher falls back to session-level identity,
so "gold" there does not mean the row actually contains the answer. And it does
not claim a token saving: 64 soft tokens stand in for a median 88-token
observation, a 0.72x ratio, because the connector emits one token per slot with
no pooling anywhere.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np
import torch

from residualmem.latent.instruct_bridge import (
    READER_BRIDGE_PROTOCOL,
    InputSoftTokenConnector,
    MemorySegment,
    Qwen35LatentReader,
    load_bridge,
)


def load_observations(directory: Path) -> dict[tuple[str, str], tuple]:
    """``(sample_id, memory_id) -> (xbar, valid, text)`` from the extraction."""
    table: dict[tuple[str, str], tuple] = {}
    for path in sorted(directory.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata"])))
            keys = sorted(k for k in data.files if k.startswith("m11/xbar/"))
            records = metadata.get("records") or []
            if len(records) != len(keys):
                raise ValueError(f"{path.name}: {len(records)} records, {len(keys)} states")
            for key, record in zip(keys, records):
                index = key.rsplit("/", 1)[1]
                memory_id = record.get("memory_id")
                if not memory_id:
                    continue
                table[(path.stem, str(memory_id))] = (
                    np.asarray(data[key], np.float32),
                    np.asarray(data[f"m11/valid/{index}"], bool),
                    str(record.get("fused_text", "")),
                )
    if not table:
        raise FileNotFoundError(f"no observations under {directory}")
    return table


def select(pipeline: Path, table: dict) -> list[dict]:
    """Questions whose retrieval surfaced an observation we hold a latent for."""
    chosen = []
    for line in pipeline.open():
        record = json.loads(line)
        sample = record["sample_id"]
        for item in record["retrieval"]["items"]:
            key = (sample, item["memory_id"])
            if key in table:
                chosen.append({
                    "sample_id": sample,
                    "question": record["question"],
                    "gold_answer": record.get("gold_answer", ""),
                    "gold_contents": record.get("gold_evidence_contents", []),
                    "rank": int(item["rank"]),
                    "key": key,
                })
                break   # the top-ranked observation row only
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True, help="pipeline_qa.jsonl from a run")
    parser.add_argument("--xbar-dir", required=True)
    parser.add_argument("--connector", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="0 = every eligible question")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(args.pipeline).resolve().parents[2]))
    from eval_framework.judges import evaluate_qa_llm

    table = load_observations(Path(args.xbar_dir))
    questions = select(Path(args.pipeline), table)
    if args.limit:
        questions = questions[: args.limit]
    print(f"[stageC] {len(questions)} questions retrieved an observation we hold", flush=True)

    from experiments.state_tokenizer.extract_qwen import _load_model

    device = torch.device(args.device)
    processor, model = _load_model(args.model, device, False)
    connector = InputSoftTokenConnector()
    load_bridge(args.connector, connector, expected_protocol=READER_BRIDGE_PROTOCOL)
    connector.to(device).eval()
    reader = Qwen35LatentReader(model, processor, connector, mode="input")

    rows = []
    labels = {"text": collections.Counter(), "latent": collections.Counter()}
    for position, entry in enumerate(questions, 1):
        xbar, valid, text = table[entry["key"]]
        conditions = {
            "text": MemorySegment(text=text),
            "latent": MemorySegment(latent=(xbar, valid)),
        }
        answers = {
            name: reader.answer(entry["question"], [segment])
            for name, segment in conditions.items()
        }
        judged = {}
        for name, answer in answers.items():
            verdict = evaluate_qa_llm(
                entry["question"], entry["gold_answer"],
                "\n".join(str(x) for x in entry["gold_contents"]), answer,
            )
            label = str(verdict.get("evaluation_result", "Omission"))
            labels[name][label] += 1
            judged[name] = {"answer": answer, "label": label}
        rows.append({**{k: entry[k] for k in ("sample_id", "question", "rank")},
                     "text_tokens": int(processor.tokenizer(text, return_tensors="pt",
                                                            add_special_tokens=False)
                                        ["input_ids"].shape[1]),
                     "latent_tokens": 64,
                     "judged": judged})
        if position % 20 == 0:
            print(f"[stageC] {position}/{len(questions)}", flush=True)

    report = {
        "protocol": "connector_fidelity_v1",
        "questions": len(rows),
        "labels": {name: dict(counter) for name, counter in labels.items()},
        "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 56)
    print(f"{'condition':12s}{'Correct':>10s}{'Halluc':>10s}{'Omission':>10s}{'tokens':>10s}")
    print("-" * 56)
    for name in ("text", "latent"):
        counter = labels[name]
        total = max(sum(counter.values()), 1)
        cost = (np.mean([r["text_tokens"] for r in rows]) if name == "text" else 64.0)
        print(f"{name:12s}{counter['Correct'] / total:10.4f}"
              f"{counter['Hallucination'] / total:10.4f}"
              f"{counter['Omission'] / total:10.4f}{cost:10.1f}")
    print("=" * 56)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
