"""Attach v2 probe targets to the v5 manifest without resampling it.

``build_v2_subset`` couples two things: deriving the probe labels, and drawing a
3,500-state subset with its own split sizes. Only the first is wanted here -- the
7:2:1 task-stratified split is already fixed and must not be redrawn.

The vocabularies are the part that needs care. ``v2_data._word_vocab`` counts
over whatever records it is handed, so passing the full manifest would let words
that only ever appear in validation or test define the probe targets. They are
therefore fitted on the train split alone, exactly like the PCA and the
normalizer.

One subtlety: the feature manifest carries ``fixed_prompt.OBSERVATION_PROMPT``,
a task-independent instruction, because that is what the encoder was run with.
Deriving ``instruction_overlap_words`` from it would yield an empty vocabulary --
the fixed prompt shares no words with any DOM. The probe labels are therefore
derived from the *original* collection instruction, which is orthogonal to how
the states were encoded.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import iter_jsonl, sha256_file, write_json
from .v2_data import (
    UNSUPPORTED_LABELS,
    _word_vocab,
    derive_v2_targets,
    label_coverage,
)


def run(args: argparse.Namespace) -> dict:
    instructions = {}
    if args.instruction_records:
        for record in iter_jsonl(args.instruction_records):
            instructions[record["state_id"]] = record["instruction"]

    records = []
    for record in iter_jsonl(args.records):
        enriched = dict(record)
        source = dict(record)
        if instructions:
            if record["state_id"] not in instructions:
                raise ValueError(f"no original instruction for {record['state_id']}")
            source["instruction"] = instructions[record["state_id"]]
        enriched["v2"] = derive_v2_targets(source)
        records.append(enriched)

    train = [record for record in records if record["split"] == "train"]
    if not train:
        raise ValueError("no train records; cannot fit probe vocabularies")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    overlap = _word_vocab(train, "instruction_overlap_words", 64)
    dom = _word_vocab(train, "dom_only_words", 128)
    if not overlap:
        raise ValueError(
            "instruction_overlap vocabulary is empty; pass --instruction-records "
            "pointing at the original collection manifest so the labels are not "
            "derived from the fixed observation prompt"
        )
    summary = {
        "records": len(records),
        "split_counts": dict(Counter(record["split"] for record in records)),
        "task_counts": dict(Counter(record["task"] for record in records)),
        "unique_episodes": len({record["episode_id"] for record in records}),
        "source_manifest": str(Path(args.records).resolve()),
        "source_sha256": sha256_file(args.records),
        "instruction_source": (
            str(Path(args.instruction_records).resolve())
            if args.instruction_records else "fixed prompt (same manifest)"
        ),
        "subset_sha256": sha256_file(output),
        "dynamic_coverage": label_coverage(records),
        # train-only, so a held-out word can never define a probe target
        "instruction_overlap_vocab": overlap,
        "dom_only_vocab": dom,
        "vocab_fitted_on": "train",
        "unsupported_labels": UNSUPPORTED_LABELS,
        "note": "v2 targets derived in place; the 7:2:1 split is untouched",
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--instruction-records",
                        help="collection manifest holding the original per-task "
                             "instructions; required because the feature manifest "
                             "carries the fixed observation prompt instead")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps({
        "records": summary["records"],
        "split_counts": summary["split_counts"],
        "dom_vocab": len(summary["dom_only_vocab"]),
        "overlap_vocab": len(summary["instruction_overlap_vocab"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
