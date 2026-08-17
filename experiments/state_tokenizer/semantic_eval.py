"""Semantic reconstruction evaluation with frozen probes.

R^2 says how faithful a reconstruction is element-wise; it cannot say how much
*task information* survived. This script answers that by fitting one probe on
clean states and then reading every reconstruction with that same frozen probe.

Frozen, not refit, is the whole point. Refitting a probe per representation
measures what is still linearly decodable, and quantization distortion is
systematic enough that a fresh probe would simply learn to undo it -- which
would make a lossy code look untouched. Holding the readout fixed measures what
was actually preserved.

The headline number is retention relative to A1, not to clean::

    R_disc(A2 | A1) = (m(A2) - b) / (m(A1) - b)

because the tokenizer and the continuous bridge have already spent part of the
semantic budget before discretization ever happens. Retention against clean is
reported too, so the loss decomposes as clean -> A1 -> A2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import iter_jsonl, write_json
from .slot_probe import (
    TargetBundle,
    TokenProvider,
    build_targets,
    evaluate_predictions,
    predict,
)
from .slot_reader import SlotAwareReader
from .v2_data import DYNAMIC_LABELS, STATIC_LABELS

REPRESENTATION = "key64_static_pca"


def _load_frozen_probe(path: Path, targets: TargetBundle, input_dim: int, device):
    """Rebuild the trained probe from a ``slot_probe.train`` checkpoint."""
    payload = torch.load(path, map_location=device)
    model = SlotAwareReader(
        input_dim,
        dynamic=len(DYNAMIC_LABELS), static=len(STATIC_LABELS),
        dom_words=len(targets.dom_vocab), overlap_words=len(targets.overlap_vocab),
        tasks=len(targets.task_names),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("config", {})


def _check_alignment(candidate: TokenProvider, reference: TokenProvider, rows) -> None:
    """The stores must describe the same states, or the comparison is meaningless.

    Reconstruction shards are written with a different interleaving than the
    original store, so alignment is checked through the global-index lookup
    rather than by shard order.
    """
    for row in rows:
        left = candidate.get(int(row))[0].float()
        right = reference.get(int(row))[0].float()
        if left.shape != right.shape:
            raise ValueError(f"row {row}: shape {tuple(left.shape)} != {tuple(right.shape)}")
        # padding slots are exactly zero in both, so the support must agree
        if not torch.equal((left != 0).any(-1), (right != 0).any(-1)):
            raise ValueError(f"row {row}: valid-slot support differs between stores")


def _flatten(metrics: dict, prefix: str = "") -> dict[str, float]:
    """Pull the scalar leaves out of slot_probe's nested metric tree."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}/"))
        elif isinstance(value, (int, float)) and np.isfinite(value):
            flat[name] = float(value)
    return flat


EPISODE_SPECIFIC_PREVALENCE = 0.05


def split_overlap_words(metrics: dict[str, float]) -> dict[str, float]:
    """Report generic page furniture separately from episode-specific targets.

    ``overlap_words`` mixes two kinds of word, and averaging them hides the one
    that matters. ``submit``/``login``/``cancel`` sit on nearly every page, so a
    representation can score well on them without having read anything; the
    MiniWoB target labels (``bg``, ``jt``, ``dq`` -- the ``bg`` in "Select bg and
    click Submit") name one control in one episode, so recovering them is the
    real evidence that the state kept "what this control is called".

    Split by measured prevalence rather than a hard-coded word list: the
    vocabulary is refitted per dataset, so any fixed list silently rots. Words
    present in under ``EPISODE_SPECIFIC_PREVALENCE`` of states are the targets.
    """
    generic, specific = [], []
    for key, value in metrics.items():
        if not key.endswith("/average_precision"):
            continue
        if not key.startswith("overlap_words/per_label/"):
            continue
        word = key.split("/")[2]
        prevalence = metrics.get(f"overlap_words/per_label/{word}/prevalence")
        if prevalence is None or not np.isfinite(value):
            continue
        (specific if prevalence < EPISODE_SPECIFIC_PREVALENCE else generic).append(value)

    out: dict[str, float] = {}
    if generic:
        out["overlap_words/generic_macro_average_precision"] = float(np.mean(generic))
        out["overlap_words/generic_labels"] = float(len(generic))
    if specific:
        out["overlap_words/episode_target_macro_average_precision"] = float(
            np.mean(specific)
        )
        out["overlap_words/episode_target_labels"] = float(len(specific))
    return out


def _retention(candidate: float, reference: float, chance: float) -> float | None:
    """Chance-corrected retention, or ``None`` when the ratio would be unreadable.

    Defined only where the reference actually beats chance. If ``reference <=
    chance`` there is no headroom to retain, and the ratio does not merely become
    noisy -- a negative denominator flips its direction, so a *better* candidate
    scores *lower*. Reporting that as a retention is worse than reporting nothing.
    """
    denominator = reference - chance
    if denominator < 1e-9:
        return None
    return (candidate - chance) / denominator


def run(args: argparse.Namespace) -> dict:
    records = list(iter_jsonl(args.records))
    targets = build_targets(records, json.loads(Path(args.summary).read_text()))
    rows = targets.splits[args.split]
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    stores = {"clean": args.clean}
    for entry in args.store:
        name, _, path = entry.partition("=")
        if not path:
            raise ValueError(f"--store expects name=path, got {entry!r}")
        stores[name] = path

    reference_provider = TokenProvider(
        REPRESENTATION, records, features=args.clean, full_h=None
    )
    model, probe_config = _load_frozen_probe(
        Path(args.probe_checkpoint), targets, reference_provider.input_dim, device
    )

    results: dict[str, dict] = {}
    for name, path in stores.items():
        provider = TokenProvider(REPRESENTATION, records, features=path, full_h=None)
        if provider.input_dim != reference_provider.input_dim:
            raise ValueError(f"{name}: input dim {provider.input_dim} != reference")
        _check_alignment(provider, reference_provider, rows[: args.alignment_states])
        predictions = predict(
            model, REPRESENTATION, provider, None, rows, targets, device
        )
        flat = _flatten(evaluate_predictions(targets, rows, predictions))
        flat.update(split_overlap_words(flat))
        results[name] = flat

    chance = _flatten(
        evaluate_predictions(
            targets, rows,
            # A constant (zero) score for every state: ranking metrics collapse to
            # prevalence, accuracies to the majority-free floor. Evaluated over the
            # *whole* split -- on a single state an average precision is degenerate
            # (it came out at 1.0 for overlap_words), which silently flips the sign
            # of every retention ratio built on it.
            {key: np.zeros_like(value) for key, value in
             predict(model, REPRESENTATION, reference_provider, None,
                     rows, targets, device).items()},
        )
    ) if args.chance_from_zero else {}
    if chance:
        chance.update(split_overlap_words(chance))

    clean = results["clean"]
    a1 = results.get(args.continuous_name)
    comparison: dict[str, dict] = {}
    for name, metrics in results.items():
        entry: dict[str, dict] = {"metrics": metrics, "retention_vs_clean": {}}
        if a1 is not None and name not in {"clean", args.continuous_name}:
            entry["retention_vs_continuous"] = {}
        for key, value in metrics.items():
            base = chance.get(key, 0.0)
            keep = _retention(value, clean.get(key, float("nan")), base)
            if keep is not None:
                entry["retention_vs_clean"][key] = keep
            if "retention_vs_continuous" in entry and key in a1:
                keep = _retention(value, a1[key], base)
                if keep is not None:
                    entry["retention_vs_continuous"][key] = keep
        comparison[name] = entry

    report = {
        "protocol": "semantic_frozen_probe_v1",
        "probe_checkpoint": str(Path(args.probe_checkpoint).resolve()),
        "probe_config": probe_config,
        "representation": REPRESENTATION,
        "split": args.split,
        "states": int(len(rows)),
        "stores": {name: str(Path(path).resolve()) for name, path in stores.items()},
        "continuous_reference": args.continuous_name,
        "headline": "retention_vs_continuous",
        # Recorded, not just applied: a retention ratio is only readable if the
        # baseline it was corrected against can be inspected. A chance value above
        # the reference silently inverts the ratio's direction.
        "chance": chance,
        "note": "one probe fitted on clean train, applied frozen to every store; "
                "test stays closed until the A2 configuration is frozen",
        "comparison": comparison,
    }
    write_json(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--summary", required=True, help="fixed-subset summary json")
    parser.add_argument("--clean", required=True, help="original feature store")
    parser.add_argument("--store", action="append", default=[],
                        help="name=path of a reconstruction store; repeatable")
    parser.add_argument("--probe-checkpoint", required=True,
                        help="checkpoint written by slot_probe.train on the clean store")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="validation",
                        choices=("train", "validation", "test"),
                        help="test must stay closed until A2 is frozen")
    parser.add_argument("--continuous-name", default="a1",
                        help="store treated as the discretization upper bound")
    parser.add_argument("--alignment-states", type=int, default=64)
    parser.add_argument("--chance-from-zero", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps({
        "split": report["split"], "states": report["states"],
        "stores": list(report["stores"]),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
