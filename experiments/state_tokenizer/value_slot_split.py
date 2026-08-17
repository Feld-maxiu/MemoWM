"""Split value-exact accuracy by where the value physically landed in the state.

The five-digit value reaches a detail slot one of two ways: as its own
``raw_literal`` token, or averaged into a ``pooled_literal`` long span together
with whatever filler shared that DOM attribute. Pooling happens in the tokenizer,
*before* any quantization, so if the pooled bucket reads far worse than the raw
one the ceiling is a tokenizer problem and no codebook or loss weighting can lift
it.

This is the gate on rebuilding the dataset: if the two buckets are close, the
pooling hypothesis is wrong and the rebuild is not worth its cost.

Note ``key64-static-detail-ranges.npy`` stores *token index* ranges, not
character offsets, so the comparison is done in token space with the same
tokenizer the encoder ran with.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .common import iter_jsonl, write_json
from .semantic_eval import REPRESENTATION, _load_frozen_probe
from .slot_probe import TokenProvider, build_targets, predict

SLOT_RAW_LITERAL = 1
DETAIL_START = 32


def _slot_tables(features: str) -> tuple[dict, dict]:
    ranges, kinds = {}, {}
    for worker in sorted(glob.glob(str(Path(features) / "worker*"))):
        index = np.load(Path(worker) / "record_indices.npy")
        span = np.load(Path(worker) / "key64-static-detail-ranges.npy")
        kind = np.load(Path(worker) / "key64-static-slot-kind.npy")
        for position, global_index in enumerate(index):
            ranges[int(global_index)] = span[position]
            kinds[int(global_index)] = kind[position]
    return ranges, kinds


def _placement(record, tokenizer, ranges, kinds) -> str | None:
    """``raw`` if every token of the value got its own slot, else ``pooled``."""
    digits = "".join(str(d) for d in record["v2"]["random_digits"])
    start = record["dom"].find(digits)
    if start < 0:
        return None
    encoded = tokenizer(record["dom"], return_offsets_mapping=True, add_special_tokens=False)
    stop = start + len(digits)
    wanted = {
        position for position, (low, high) in enumerate(encoded["offset_mapping"])
        if low < stop and high > start
    }
    if not wanted:
        return None
    global_index = int(record["global_index"])
    raw: set[int] = set()
    for slot, (low, high) in enumerate(ranges[global_index]):
        if low >= 0 and kinds[global_index][DETAIL_START + slot] == SLOT_RAW_LITERAL:
            raw |= set(range(int(low), int(high)))
    return "raw" if wanted <= raw else "pooled"


def run(args: argparse.Namespace) -> dict:
    records = list(iter_jsonl(args.records))
    targets = build_targets(records, json.loads(Path(args.summary).read_text()))
    rows = targets.splits[args.split]
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    provider = TokenProvider(REPRESENTATION, records, features=args.clean, full_h=None)
    model, _ = _load_frozen_probe(
        Path(args.probe_checkpoint), targets, provider.input_dim, device
    )
    predictions = predict(model, REPRESENTATION, provider, None, rows, targets, device)

    # same masking slot_probe uses, so "examples" matches the headline metric
    value_rows = np.all(targets.digits[rows] >= 0, axis=1)
    labels = targets.digits[rows][value_rows]
    scores = predictions["digits"][value_rows]
    correct = np.all(scores.argmax(axis=2) == labels, axis=1)
    per_position = scores.argmax(axis=2) == labels

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    ranges, kinds = _slot_tables(args.clean)
    value_row_ids = np.asarray(rows)[value_rows]

    buckets: dict[str, list[int]] = {"raw": [], "pooled": []}
    by_task: dict[tuple[str, str], list[int]] = {}
    # scroll-text-2 concatenates up to 23 "state-NNNNN" markers and the target
    # regex takes the first, so its "value" is a memorisation artefact rather
    # than a task quantity; it has to be separable from the honest cases.
    for position, row in enumerate(value_row_ids):
        record = records[int(row)]
        placement = _placement(record, tokenizer, ranges, kinds)
        if placement is not None:
            buckets[placement].append(position)
            by_task.setdefault((record["task"], placement), []).append(position)

    report = {
        "split": args.split,
        "value_examples": int(len(labels)),
        "buckets": {},
        "note": "raw = value owns its detail slots; pooled = averaged into a long span",
    }
    for name, positions in buckets.items():
        if not positions:
            continue
        picked = np.asarray(positions)
        report["buckets"][name] = {
            "states": int(len(picked)),
            "exact_value_accuracy": float(correct[picked].mean()),
            "macro_position_accuracy": float(per_position[picked].mean()),
        }
    report["by_task"] = {
        f"{task}|{placement}": {
            "states": int(len(positions)),
            "exact_value_accuracy": float(correct[np.asarray(positions)].mean()),
            "multi_marker_states": int(sum(
                len(re.findall(r"state-\\d{5}", records[int(value_row_ids[p])]["dom"])) > 1
                for p in positions)),
        }
        for (task, placement), positions in sorted(by_task.items())
    }
    honest = [p for (task, placement), ps in by_task.items()
              if "scroll-text" not in task for p in ps]
    honest_pooled = [p for (task, placement), ps in by_task.items()
                     if "scroll-text" not in task and placement == "pooled" for p in ps]
    honest_raw = [p for (task, placement), ps in by_task.items()
                  if "scroll-text" not in task and placement == "raw" for p in ps]
    report["excluding_scroll_text"] = {
        "states": len(honest),
        "overall_exact": float(correct[np.asarray(honest)].mean()) if honest else None,
        "raw_states": len(honest_raw),
        "raw_exact": float(correct[np.asarray(honest_raw)].mean()) if honest_raw else None,
        "pooled_states": len(honest_pooled),
        "pooled_exact": float(correct[np.asarray(honest_pooled)].mean()) if honest_pooled else None,
    }
    if {"raw", "pooled"} <= set(report["buckets"]):
        raw, pooled = report["buckets"]["raw"], report["buckets"]["pooled"]
        report["gap_exact"] = raw["exact_value_accuracy"] - pooled["exact_value_accuracy"]
        report["ceiling_if_pooled_matched_raw"] = raw["exact_value_accuracy"]
    write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation",
                        choices=("train", "validation", "test"))
    parser.add_argument("--tokenizer",
                        default="/root/nas/users/luzheng/workspace/models/Qwen3.5-9B")
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
