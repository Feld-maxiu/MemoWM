"""Aggregate value-only probes and compute leakage-adjusted compression retention."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import write_json


SEEDS = (0, 1, 2)
REPRESENTATIONS = ("full_h", "instruction_only", "y64", "y32", "x64")


def mean_std(values) -> dict:
    array = np.asarray(values, np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "values": [float(value) for value in array],
    }


def load_results(root: str | Path, representation: str) -> list[dict]:
    root = Path(root)
    results = []
    for seed in SEEDS:
        path = root / f"result-{representation}-seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text())
        if result.get("protocol") != "slot_aware_v2_value_only":
            raise ValueError(f"not a value-only result: {path}")
        if result["representation"] != representation or int(result["seed"]) != seed:
            raise ValueError(f"result identity mismatch: {path}")
        results.append(result)
    return results


def metric(result: dict, name: str = "macro_position_accuracy") -> float:
    return float(result["test"]["value"][name])


def retention(candidate: list[dict], oracle: list[dict], leakage: list[dict], metric_name: str) -> dict:
    candidate_values = np.asarray([metric(item, metric_name) for item in candidate])
    oracle_values = np.asarray([metric(item, metric_name) for item in oracle])
    leakage_values = np.asarray([metric(item, metric_name) for item in leakage])
    denominators = oracle_values - leakage_values
    if bool((denominators <= 0).any()):
        raise ValueError(f"oracle does not beat leakage for {metric_name}: {denominators.tolist()}")
    paired = (candidate_values - leakage_values) / denominators
    ratio_of_means = (
        float(candidate_values.mean() - leakage_values.mean())
        / float(oracle_values.mean() - leakage_values.mean())
    )
    return {
        "definition": "(M_compressed - M_instruction_only) / (M_full_h - M_instruction_only)",
        "ratio_of_means": ratio_of_means,
        "paired_seed": mean_std(paired),
        "candidate_minus_leakage": mean_std(candidate_values - leakage_values),
    }


def summarize(results: list[dict]) -> dict:
    values = [item["test"]["value"] for item in results]
    return {
        "macro_position_accuracy": mean_std([item["macro_position_accuracy"] for item in values]),
        "exact_value_accuracy": mean_std([item["exact_value_accuracy"] for item in values]),
        "position_accuracy": [
            mean_std([item["position_accuracy"][position] for item in values])
            for position in range(5)
        ],
        "examples": int(values[0]["examples"]),
        "epochs_ran": [int(item["epochs_ran"]) for item in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = {name: load_results(args.results, name) for name in REPRESENTATIONS}
    summaries = {name: summarize(items) for name, items in raw.items()}
    compression = {}
    for name in ("y64", "y32", "x64"):
        compression[name] = {
            "macro_position_accuracy": retention(
                raw[name], raw["full_h"], raw["instruction_only"], "macro_position_accuracy"
            ),
            "exact_value_accuracy": retention(
                raw[name], raw["full_h"], raw["instruction_only"], "exact_value_accuracy"
            ),
        }
    report = {
        "protocol": "slot_aware_v2_value_only_retention",
        "seeds": list(SEEDS),
        "primary_metric": "macro_position_accuracy",
        "leakage_baseline": "instruction_only",
        "summaries": summaries,
        "compression_retention": compression,
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "aggregate-value-only.json", report)
    with (output / "value-only-summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "representation", "value_accuracy_mean", "value_accuracy_std",
            "exact_accuracy_mean", "exact_accuracy_std", "retention_ratio_of_means",
            "retention_paired_mean", "retention_paired_std",
        ])
        for name in REPRESENTATIONS:
            retention_item = compression.get(name, {}).get("macro_position_accuracy", {})
            writer.writerow([
                name,
                summaries[name]["macro_position_accuracy"]["mean"],
                summaries[name]["macro_position_accuracy"]["std"],
                summaries[name]["exact_value_accuracy"]["mean"],
                summaries[name]["exact_value_accuracy"]["std"],
                retention_item.get("ratio_of_means", ""),
                retention_item.get("paired_seed", {}).get("mean", ""),
                retention_item.get("paired_seed", {}).get("std", ""),
            ])
    print(json.dumps({
        "output": str(output / "aggregate-value-only.json"),
        "value_accuracy": {
            name: summaries[name]["macro_position_accuracy"]["mean"]
            for name in REPRESENTATIONS
        },
        "retention": {
            name: compression[name]["macro_position_accuracy"]["ratio_of_means"]
            for name in compression
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
