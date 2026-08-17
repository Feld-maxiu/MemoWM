"""Aggregate fixed-prompt exact random-value probes across three seeds."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import write_json


SEEDS = (0, 1, 2)
REPRESENTATIONS = (
    "full_h", "instruction_only", "task_only", "task_step", "y64", "y32", "x64"
)
COMPRESSED = ("instruction_only", "y64", "y32", "x64")
OPTIONAL_COMPRESSED = (
    "key64", "key64_pca", "key64_static", "key64_static_pca",
)
METADATA_BASELINES = ("task_only", "task_step")
CHANCE = {
    "macro_position_accuracy": 0.1,
    "exact_value_accuracy": 1e-5,
}


def mean_std(values) -> dict:
    array = np.asarray(values, np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": [float(value) for value in array],
    }


def load_results(root: Path, representation: str) -> list[dict]:
    output = []
    for seed in SEEDS:
        path = root / f"result-{representation}-seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text())
        if result.get("protocol") != "slot_aware_v3_fixed_prompt_value_only":
            raise ValueError(f"not a fixed-prompt value result: {path}")
        if result["representation"] != representation or int(result["seed"]) != seed:
            raise ValueError(f"result identity mismatch: {path}")
        output.append(result)
    return output


def metric(result: dict, name: str) -> float:
    return float(result["test"]["value"][name])


def summarize(results: list[dict]) -> dict:
    values = [result["test"]["value"] for result in results]
    return {
        "macro_position_accuracy": mean_std([
            item["macro_position_accuracy"] for item in values
        ]),
        "exact_value_accuracy": mean_std([item["exact_value_accuracy"] for item in values]),
        "position_accuracy": [
            mean_std([item["position_accuracy"][position] for item in values])
            for position in range(5)
        ],
        "examples": int(values[0]["examples"]),
        "epochs_ran": [int(result["epochs_ran"]) for result in results],
    }


def chance_adjusted_retention(
    candidate: list[dict], oracle: list[dict], metric_name: str,
) -> dict:
    chance = CHANCE[metric_name]
    candidate_values = np.asarray([metric(result, metric_name) for result in candidate])
    oracle_values = np.asarray([metric(result, metric_name) for result in oracle])
    denominators = oracle_values - chance
    if bool((denominators <= 0).any()):
        raise ValueError(f"Full-H does not beat chance for {metric_name}: {denominators.tolist()}")
    paired = (candidate_values - chance) / denominators
    raw_ratio = float(candidate_values.mean() - chance) / float(oracle_values.mean() - chance)
    ratio_of_means = max(0.0, raw_ratio)
    return {
        "definition": "(M_compressed - M_chance) / (M_full_h - M_chance)",
        "chance": chance,
        "raw_ratio_of_means": raw_ratio,
        "ratio_of_means": ratio_of_means,
        "relative_loss": 1.0 - ratio_of_means,
        "below_baseline": bool(raw_ratio < 0),
        "absolute_metric_drop": float(oracle_values.mean() - candidate_values.mean()),
        "paired_seed": mean_std(paired),
    }


def conservative_retention(
    candidate: list[dict], oracle: list[dict], metadata: list[dict], metric_name: str,
) -> dict:
    chance = CHANCE[metric_name]
    candidate_values = np.asarray([metric(result, metric_name) for result in candidate])
    oracle_values = np.asarray([metric(result, metric_name) for result in oracle])
    metadata_values = np.asarray([metric(result, metric_name) for result in metadata])
    baseline_values = np.maximum(metadata_values, chance)
    denominators = oracle_values - baseline_values
    if bool((denominators <= 0).any()):
        raise ValueError(f"Full-H does not beat conservative baseline for {metric_name}")
    paired = (candidate_values - baseline_values) / denominators
    baseline_mean = max(float(metadata_values.mean()), chance)
    raw_ratio = (
        float(candidate_values.mean() - baseline_mean)
        / float(oracle_values.mean() - baseline_mean)
    )
    ratio_of_means = max(0.0, raw_ratio)
    return {
        "definition": (
            "(M_compressed - max(M_strongest_metadata,M_chance)) / "
            "(M_full_h - max(M_strongest_metadata,M_chance))"
        ),
        "theoretical_chance": chance,
        "metadata_mean": float(metadata_values.mean()),
        "baseline_mean": baseline_mean,
        "raw_ratio_of_means": raw_ratio,
        "ratio_of_means": ratio_of_means,
        "relative_loss": 1.0 - ratio_of_means,
        "below_baseline": bool(raw_ratio < 0),
        "absolute_metric_drop": float(oracle_values.mean() - candidate_values.mean()),
        "paired_seed": mean_std(paired),
    }


def strongest_metadata(raw: dict[str, list[dict]], metric_name: str) -> dict:
    values = {
        name: float(np.mean([metric(result, metric_name) for result in raw[name]]))
        for name in METADATA_BASELINES
    }
    representation = max(values, key=values.get)
    return {"representation": representation, "score": values[representation], "all": values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results_root = Path(args.results)
    representations = list(REPRESENTATIONS)
    compressed = list(COMPRESSED)
    for name in OPTIONAL_COMPRESSED:
        paths = [results_root / f"result-{name}-seed{seed}.json" for seed in SEEDS]
        if all(path.exists() for path in paths):
            representations.append(name)
            compressed.append(name)
        elif any(path.exists() for path in paths):
            raise FileNotFoundError(f"incomplete optional representation: {name}")
    raw = {name: load_results(results_root, name) for name in representations}
    summaries = {name: summarize(results) for name, results in raw.items()}
    chance_compression = {
        name: {
            metric_name: chance_adjusted_retention(raw[name], raw["full_h"], metric_name)
            for metric_name in CHANCE
        }
        for name in compressed
    }
    metadata = {
        metric_name: strongest_metadata(raw, metric_name)
        for metric_name in CHANCE
    }
    compression = {
        name: {
            metric_name: conservative_retention(
                raw[name], raw["full_h"],
                raw[metadata[metric_name]["representation"]], metric_name,
            )
            for metric_name in CHANCE
        }
        for name in compressed
    }
    report = {
        "protocol": "slot_aware_v3_fixed_prompt_value_retention",
        "seeds": list(SEEDS),
        "primary_metric": "macro_position_accuracy",
        "strict_metric": "exact_value_accuracy",
        "chance_baselines": CHANCE,
        "strongest_metadata_baselines": metadata,
        "summaries": summaries,
        "compression_retention": compression,
        "chance_adjusted_compression_retention": chance_compression,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "aggregate-fixed-value.json", report)
    with (output / "fixed-value-summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "representation", "position_accuracy_mean", "position_accuracy_std",
            "exact_accuracy_mean", "exact_accuracy_std", "position_retention",
            "position_relative_loss", "exact_retention", "exact_relative_loss",
            "chance_position_retention", "chance_exact_retention",
        ])
        for name in representations:
            item = compression.get(name, {})
            writer.writerow([
                name,
                summaries[name]["macro_position_accuracy"]["mean"],
                summaries[name]["macro_position_accuracy"]["std"],
                summaries[name]["exact_value_accuracy"]["mean"],
                summaries[name]["exact_value_accuracy"]["std"],
                item.get("macro_position_accuracy", {}).get("ratio_of_means", ""),
                item.get("macro_position_accuracy", {}).get("relative_loss", ""),
                item.get("exact_value_accuracy", {}).get("ratio_of_means", ""),
                item.get("exact_value_accuracy", {}).get("relative_loss", ""),
                chance_compression.get(name, {}).get(
                    "macro_position_accuracy", {}
                ).get("ratio_of_means", ""),
                chance_compression.get(name, {}).get(
                    "exact_value_accuracy", {}
                ).get("ratio_of_means", ""),
            ])
    print(json.dumps({
        "output": str(output / "aggregate-fixed-value.json"),
        "position_accuracy": {
            name: summaries[name]["macro_position_accuracy"]["mean"]
            for name in representations
        },
        "exact_accuracy": {
            name: summaries[name]["exact_value_accuracy"]["mean"]
            for name in representations
        },
        "position_retention": {
            name: compression[name]["macro_position_accuracy"]["ratio_of_means"]
            for name in compressed
        },
        "exact_retention": {
            name: compression[name]["exact_value_accuracy"]["ratio_of_means"]
            for name in compressed
        },
        "chance_position_retention": {
            name: chance_compression[name]["macro_position_accuracy"]["ratio_of_means"]
            for name in compressed
        },
        "metadata_baselines": metadata,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
