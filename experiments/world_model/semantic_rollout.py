"""Apply the existing clean-trained frozen probe to closed-loop reconstruction stores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.state_tokenizer.common import iter_jsonl, write_json
from experiments.state_tokenizer.semantic_eval import (
    REPRESENTATION,
    _flatten,
    _load_frozen_probe,
    split_overlap_words,
)
from experiments.state_tokenizer.slot_probe import (
    TokenProvider,
    build_targets,
    evaluate_predictions,
    predict,
)

from .semantic_utils import check_shape_alignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--probe-checkpoint", required=True)
    parser.add_argument("--rollout-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alignment-states", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    records = list(iter_jsonl(args.records))
    targets = build_targets(records, json.loads(Path(args.summary).read_text()))
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    clean = TokenProvider(REPRESENTATION, records, features=args.clean, full_h=None)
    model, probe_config = _load_frozen_probe(
        Path(args.probe_checkpoint), targets, clean.input_dim, device
    )
    rollout = json.loads(Path(args.rollout_json).read_text(encoding="utf-8"))
    if not rollout.get("reconstruction_stores"):
        raise ValueError("rollout has no reconstruction stores")

    results = {}
    for horizon, store_path in sorted(
        rollout["reconstruction_stores"].items(), key=lambda item: int(item[0])
    ):
        store_summary = json.loads(
            (Path(store_path) / "summary.json").read_text(encoding="utf-8")
        )
        rows = np.asarray(store_summary["global_indices"], np.int64)
        candidate = TokenProvider(
            REPRESENTATION, records, features=store_path, full_h=None
        )
        check_shape_alignment(candidate, clean, rows[: args.alignment_states])
        clean_predictions = predict(
            model, REPRESENTATION, clean, None, rows, targets, device
        )
        candidate_predictions = predict(
            model, REPRESENTATION, candidate, None, rows, targets, device
        )
        clean_metrics = _flatten(evaluate_predictions(targets, rows, clean_predictions))
        candidate_metrics = _flatten(
            evaluate_predictions(targets, rows, candidate_predictions)
        )
        clean_metrics.update(split_overlap_words(clean_metrics))
        candidate_metrics.update(split_overlap_words(candidate_metrics))
        results[horizon] = {
            "states": len(rows),
            "clean": clean_metrics,
            "predicted_prefix": candidate_metrics,
            "delta": {
                key: candidate_metrics[key] - clean_metrics[key]
                for key in candidate_metrics.keys() & clean_metrics.keys()
            },
        }
    report = {
        "protocol": "v8_wm_rollout_frozen_semantic_probe_v1",
        "probe_checkpoint": str(Path(args.probe_checkpoint).resolve()),
        "probe_config": probe_config,
        "rollout": str(Path(args.rollout_json).resolve()),
        "headline": "predicted-prefix semantic recovery by exact rollout horizon",
        "results": results,
        "note": (
            "The A2 code axis is distributed; this decoded frozen-probe result is "
            "the valid semantic/detail diagnostic, not a code-axis rate decomposition."
        ),
    }
    write_json(args.output, report)
    print(json.dumps({
        horizon: {"states": value["states"]}
        for horizon, value in results.items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
