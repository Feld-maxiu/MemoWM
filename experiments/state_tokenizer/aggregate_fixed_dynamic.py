"""Aggregate fixed-prompt checked/focused/nonempty probes across three seeds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import write_json
from .slot_probe import DYNAMIC_STATE_LABELS


SEEDS = (0, 1, 2)
REPRESENTATIONS = (
    "full_h", "instruction_only", "task_only", "task_step", "y64", "y32", "x64"
)
OPTIONAL_COMPRESSED = (
    "key64", "key64_pca", "key64_static", "key64_static_pca",
)
# Fixed prompt token states are causally contextualized by the preceding image/DOM.
# They are therefore a compact page representation, not an instruction-semantics
# leakage baseline.  Only metadata that never reads page contents is used below.
LEAKAGE_BASELINES = ("task_only", "task_step")


def mean_std(values) -> dict:
    array = np.asarray(values, np.float64)
    finite = array[np.isfinite(array)]
    return {
        "mean": float(finite.mean()) if len(finite) else float("nan"),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "values": [float(value) for value in array],
        "finite": int(len(finite)),
    }


def load_results(root: Path, representation: str) -> list[dict]:
    results = []
    for seed in SEEDS:
        path = root / f"result-{representation}-seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text())
        if result.get("protocol") != "slot_aware_v3_fixed_prompt_dynamic":
            raise ValueError(f"not a fixed-prompt dynamic result: {path}")
        if result["representation"] != representation or int(result["seed"]) != seed:
            raise ValueError(f"result identity mismatch: {path}")
        results.append(result)
    return results


def state_metric(result: dict) -> float:
    return float(result["test"]["dynamic"]["state_eligible_task_macro_average_precision"])


def label_metric(result: dict, label: str) -> float:
    return float(result["test"]["dynamic"]["per_label"][label]["task_macro_average_precision"])


def summarize(results: list[dict]) -> dict:
    dynamic = [result["test"]["dynamic"] for result in results]
    eligible = list(dynamic[0]["state_eligible_labels"])
    return {
        "state_task_macro_average_precision": mean_std([state_metric(result) for result in results]),
        "eligible_labels": eligible,
        "labels": {
            label: {
                "task_macro_average_precision": mean_std([
                    label_metric(result, label) for result in results
                ]),
                "global_average_precision": mean_std([
                    result["test"]["dynamic"]["per_label"][label]["average_precision"]
                    for result in results
                ]),
                "task_prevalence": float(dynamic[0]["per_label"][label]["task_prevalence"]),
                "examples": int(dynamic[0]["per_label"][label]["examples"]),
            }
            for label in DYNAMIC_STATE_LABELS
        },
        "epochs_ran": [int(result["epochs_ran"]) for result in results],
    }


def strongest_leakage(summaries: dict, label: str | None = None) -> dict:
    if label is None:
        values = {
            name: summaries[name]["state_task_macro_average_precision"]["mean"]
            for name in LEAKAGE_BASELINES
        }
    else:
        values = {
            name: summaries[name]["labels"][label]["task_macro_average_precision"]["mean"]
            for name in LEAKAGE_BASELINES
        }
    finite = {name: value for name, value in values.items() if np.isfinite(value)}
    if not finite:
        return {"representation": None, "score": float("nan"), "all": values}
    name = max(finite, key=finite.get)
    return {"representation": name, "score": finite[name], "all": values}


def retention(candidate: list[dict], oracle: list[dict], leakage: list[dict]) -> dict:
    candidate_values = np.asarray([state_metric(result) for result in candidate])
    oracle_values = np.asarray([state_metric(result) for result in oracle])
    leakage_values = np.asarray([state_metric(result) for result in leakage])
    denominators = oracle_values - leakage_values
    paired = np.where(denominators > 0, (candidate_values - leakage_values) / denominators, np.nan)
    mean_denominator = float(oracle_values.mean() - leakage_values.mean())
    return {
        "definition": "(M_compressed - M_strongest_leakage) / (M_full_h - M_strongest_leakage)",
        "ratio_of_means": (
            float(candidate_values.mean() - leakage_values.mean()) / mean_denominator
            if mean_denominator > 0 else float("nan")
        ),
        "paired_seed": mean_std(paired),
        "candidate_minus_leakage": mean_std(candidate_values - leakage_values),
        "oracle_minus_leakage": mean_std(oracle_values - leakage_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results_root = Path(args.results)
    representations = list(REPRESENTATIONS)
    compression_names = ["y64", "y32", "x64"]
    for name in OPTIONAL_COMPRESSED:
        paths = [results_root / f"result-{name}-seed{seed}.json" for seed in SEEDS]
        if all(path.exists() for path in paths):
            representations.append(name)
            compression_names.append(name)
        elif any(path.exists() for path in paths):
            raise FileNotFoundError(f"incomplete optional representation: {name}")
    raw = {name: load_results(results_root, name) for name in representations}
    summaries = {name: summarize(results) for name, results in raw.items()}
    leakage = strongest_leakage(summaries)
    leakage_name = leakage["representation"]
    compression = {
        name: retention(raw[name], raw["full_h"], raw[leakage_name])
        for name in compression_names
    }
    per_label_leakage = {
        label: strongest_leakage(summaries, label)
        for label in summaries["full_h"]["eligible_labels"]
    }
    report = {
        "protocol": "slot_aware_v3_fixed_prompt_dynamic",
        "objective_labels": list(DYNAMIC_STATE_LABELS),
        "primary_labels": summaries["full_h"]["eligible_labels"],
        "primary_metric": "task-macro average precision over eligible checked/focused/nonempty labels",
        "instruction_only_interpretation": (
            "The fixed-prompt token hidden states are causally contextualized by the preceding "
            "image/DOM and are reported as a compact state representation, not as a leakage baseline."
        ),
        "leakage_baseline_definition": (
            "strongest metadata-only baseline among task-ID and task+step; neither reads page content"
        ),
        "strongest_group_leakage": leakage,
        "summaries": summaries,
        "compression_retention": compression,
        "per_label_strongest_leakage": per_label_leakage,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "aggregate-fixed-dynamic.json", report)
    with (output / "fixed-dynamic-per-label.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "representation", "task_prevalence", "AP_mean", "AP_std"])
        for label in DYNAMIC_STATE_LABELS:
            for name in representations:
                item = summaries[name]["labels"][label]
                writer.writerow([
                    label, name, item["task_prevalence"],
                    item["task_macro_average_precision"]["mean"],
                    item["task_macro_average_precision"]["std"],
                ])
    print(json.dumps({
        "output": str(output / "aggregate-fixed-dynamic.json"),
        "state_ap": {
            name: summaries[name]["state_task_macro_average_precision"]["mean"]
            for name in representations
        },
        "strongest_leakage": leakage,
        "retention": {name: item["ratio_of_means"] for name, item in compression.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
