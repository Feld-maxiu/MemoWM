"""Model-free check: does the instruction's target label reach a raw detail slot?

This is the earliest honest signal about whether the AXTree switch worked. It
touches no probe and no learned weights -- it asks only whether the characters
naming the control the instruction points at survived into a slot that stores a
literal, rather than being pooled away or crowded out by the twelve-slot budget.

Two details make this easy to get wrong, and both produced a badly wrong number
during development:

* ``key64-static-detail-ranges`` stores **token indices**, not character offsets.
  Slicing the state text with them yields single stray characters and a coverage
  near zero. They have to be mapped through the DOM token offsets first.
* raw literal slots hold **one token each**, so a multi-token target such as
  ``s6WcI`` is spread over several slots. Testing slots individually reports
  nearly everything as missing; the covered characters have to be unioned first.

The real instruction lives only in the collection manifest -- the fixed-prompt
manifest carries the task-independent observation prompt by design.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from transformers import AutoProcessor

from .binding_probe import instruction_targets
from .common import iter_jsonl, write_json
from .extract_qwen import _input_text, modality_indices
from .static_key_pooling import SLOT_RAW_LITERAL

IMAGE_SLOTS = 32
# The two tasks whose instruction names a control by a literal label. Elsewhere
# the "target" is a slider value or an ordinal ("the 3rd checkbox"), which this
# measure does not apply to.
BINDING_TASKS = ("miniwob/click-checkboxes-v1", "miniwob/click-option-v1")


def dom_token_offsets(processor, model, record: dict) -> list[tuple[int, int]]:
    text = _input_text(processor, record["dom"], record["instruction"])
    encoded = processor.tokenizer(
        text, add_special_tokens=True, return_offsets_mapping=True, return_tensors="pt"
    )
    indices = modality_indices(processor, model, encoded["input_ids"])
    start = text.index(record["dom"])
    offsets = encoded["offset_mapping"][0].index_select(0, indices[1])
    return [(int(a) - start, int(b) - start) for a, b in offsets.tolist()]


def run(args: argparse.Namespace) -> dict:
    model_root = Path(args.model)
    processor = AutoProcessor.from_pretrained(model_root, local_files_only=True)
    config = json.loads((model_root / "config.json").read_text())
    model = SimpleNamespace(config=SimpleNamespace(image_token_id=config["image_token_id"]))

    encoded = {r["global_index"]: r for r in iter_jsonl(args.records)}
    # Keyed by state_id, not global_index: the collection manifest predates the
    # split step that assigns indices, so only v8's happens to carry one.
    instructions = {
        r["state_id"]: r["instruction"] for r in iter_jsonl(args.instruction_records)
    }

    hits = total = 0
    per_task = collections.Counter()
    for shard in sorted(Path(args.static_features).glob("worker*")):
        index = np.load(shard / "record_indices.npy")
        ranges = np.load(shard / "key64-static-detail-ranges.npy", mmap_mode="r")
        kinds = np.load(shard / "key64-static-slot-kind.npy", mmap_mode="r")
        for row, global_index in enumerate(index):
            record = encoded[int(global_index)]
            if record["task"] not in BINDING_TASKS:
                continue
            if args.split and record["split"] != args.split:
                continue
            targets = instruction_targets(instructions[record["state_id"]]) - {"nothing"}
            if not targets:
                continue
            text = record["dom"]
            offsets = dom_token_offsets(processor, model, record)
            covered = np.zeros(len(text) + 1, bool)
            for slot in range(ranges.shape[1]):
                if int(kinds[row, IMAGE_SLOTS + slot]) != SLOT_RAW_LITERAL:
                    continue
                first, last = int(ranges[row, slot, 0]), int(ranges[row, slot, 1])
                for token in range(max(first, 0), min(last, len(offsets))):
                    a, b = offsets[token]
                    if 0 <= a < b <= len(text):
                        covered[a:b] = True
            for target in targets:
                found = False
                at = text.find(target)
                while at != -1:
                    if covered[at:at + len(target)].all():
                        found = True
                        break
                    at = text.find(target, at + 1)
                total += 1
                hits += found
                per_task[(record["task"], "covered" if found else "missing")] += 1
            if args.limit and total >= args.limit:
                break

    report = {
        "protocol": "target_raw_slot_coverage_v1",
        "split": args.split,
        "tasks": list(BINDING_TASKS),
        "targets": total,
        "covered": hits,
        "coverage": hits / max(total, 1),
        "per_task": {f"{task}/{state}": count for (task, state), count in sorted(per_task.items())},
        "note": "detail ranges are token indices; raw slots hold one token each, so "
                "the covered characters are unioned before testing a target",
    }
    write_json(args.output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, help="manifest the features were built from")
    parser.add_argument("--instruction-records", required=True,
                        help="collection manifest carrying the real task instruction")
    parser.add_argument("--static-features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation",
                        choices=("train", "validation", "test", ""))
    parser.add_argument("--limit", type=int)
    report = run(parser.parse_args())
    print(json.dumps({
        "targets": report["targets"],
        "coverage": round(report["coverage"], 4),
        "per_task": report["per_task"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
