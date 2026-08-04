"""Aggregate three-seed v2 probes and enforce the staged hard gates."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .common import write_json


SEEDS = (0, 1, 2)
LEAKAGE_BASELINES = ("task_only", "instruction_only", "task_step")


def _load(results: Path, representation: str) -> list[dict]:
    loaded = []
    for seed in SEEDS:
        path = results / f"result-{representation}-seed{seed}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        item = json.loads(path.read_text())
        if item["representation"] != representation or int(item["seed"]) != seed:
            raise ValueError(f"result identity mismatch: {path}")
        loaded.append(item)
    return loaded


def _mean_std(values) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    return {
        "mean": float(np.nanmean(array)),
        "std": float(np.nanstd(array, ddof=1)) if len(array) > 1 else 0.0,
        "values": [float(value) for value in array],
    }


def _group(result: dict, name: str) -> float:
    test = result["test"]
    if name == "dynamic":
        return float(test["dynamic"]["eligible_task_macro_average_precision"])
    if name == "value":
        return float(test["value"]["macro_position_accuracy"])
    if name == "dom_words":
        return float(test["dom_words"]["macro_average_precision"])
    if name == "overlap_words":
        return float(test["overlap_words"]["macro_average_precision"])
    if name == "task":
        return float(test["task"]["accuracy"])
    raise ValueError(name)


def summarize(results: list[dict]) -> dict:
    groups = {}
    for name in ("dynamic", "value", "dom_words", "overlap_words", "task"):
        if all(name not in {"dom_words", "overlap_words"} or name in item["test"] for item in results):
            groups[name] = _mean_std([_group(item, name) for item in results])
    eligible = results[0]["test"]["dynamic"]["eligible_labels"]
    labels = {}
    for label in eligible:
        label_results = [item["test"]["dynamic"]["per_label"][label] for item in results]
        labels[label] = {
            "task_macro_average_precision": _mean_std([
                item["task_macro_average_precision"] for item in label_results
            ]),
            "task_prevalence": float(label_results[0]["task_prevalence"]),
            "global_prevalence": float(label_results[0]["prevalence"]),
            "examples": int(label_results[0]["examples"]),
        }
    return {"groups": groups, "eligible_dynamic_labels": eligible, "dynamic_labels": labels}


def paired_retention(candidate: list[dict], reference: list[dict], group: str) -> dict:
    if group == "value":
        baseline = 0.1
    elif group == "dynamic":
        label_names = reference[0]["test"]["dynamic"]["eligible_labels"]
        baseline = float(np.nanmean([
            reference[0]["test"]["dynamic"]["per_label"][label]["task_prevalence"]
            for label in label_names
        ]))
    else:
        raise ValueError(group)
    ratios = []
    for compressed, oracle in zip(candidate, reference):
        denominator = _group(oracle, group) - baseline
        ratios.append((_group(compressed, group) - baseline) / denominator if denominator > 0 else float("nan"))
    result = _mean_std(ratios)
    result["baseline"] = baseline
    return result


def per_label_retention(candidate: list[dict], reference: list[dict]) -> dict:
    labels = reference[0]["test"]["dynamic"]["eligible_labels"]
    output = {}
    for label in labels:
        baseline = float(reference[0]["test"]["dynamic"]["per_label"][label]["task_prevalence"])
        ratios = []
        for compressed, oracle in zip(candidate, reference):
            compressed_ap = compressed["test"]["dynamic"]["per_label"][label]["task_macro_average_precision"]
            oracle_ap = oracle["test"]["dynamic"]["per_label"][label]["task_macro_average_precision"]
            denominator = oracle_ap - baseline
            ratios.append((compressed_ap - baseline) / denominator if denominator > 0 else float("nan"))
        output[label] = {**_mean_std(ratios), "baseline": baseline}
    return output


def strongest_baseline(summaries: dict[str, dict], group: str) -> dict:
    values = {
        name: summaries[name]["groups"][group]["mean"] for name in LEAKAGE_BASELINES
    }
    name = max(values, key=values.get)
    return {"representation": name, "score": values[name], "all": values}


def qualification_gate(summaries: dict[str, dict]) -> dict:
    full = summaries["full_h"]
    eligible = int(sum(
        np.isfinite(item["task_macro_average_precision"]["mean"])
        for item in full["dynamic_labels"].values()
    ))
    dynamic_baseline = strongest_baseline(summaries, "dynamic")
    value_baseline = strongest_baseline(summaries, "value")
    checks = {
        "at_least_6_eligible_dynamic_labels": bool(eligible >= 6),
        "dynamic_beats_leakage_by_0.10": bool(
            full["groups"]["dynamic"]["mean"] - dynamic_baseline["score"] >= 0.10
        ),
        "value_beats_leakage_by_0.10": bool(
            full["groups"]["value"]["mean"] - value_baseline["score"] >= 0.10
        ),
    }
    return {
        "name": "full_h_qualification",
        "passed": all(checks.values()),
        "failure_status": None if all(checks.values()) else "INCONCLUSIVE_PROBE",
        "checks": checks,
        "eligible_dynamic_labels": eligible,
        "dynamic_leakage_baseline": dynamic_baseline,
        "value_leakage_baseline": value_baseline,
        "full_h_dynamic": full["groups"]["dynamic"],
        "full_h_value": full["groups"]["value"],
    }


def y64_gate(
    candidate: list[dict], reference: list[dict], summaries: dict[str, dict]
) -> dict:
    dynamic_retention = paired_retention(candidate, reference, "dynamic")
    value_retention = paired_retention(candidate, reference, "value")
    labels = per_label_retention(candidate, reference)
    finite_label_ratios = [item["mean"] for item in labels.values() if np.isfinite(item["mean"])]
    fraction = float(np.mean(np.asarray(finite_label_ratios) >= 0.70)) if finite_label_ratios else 0.0
    leakage = strongest_baseline(summaries, "dynamic")
    candidate_dynamic = float(np.mean([_group(item, "dynamic") for item in candidate]))
    mean_retention = float(np.mean([dynamic_retention["mean"], value_retention["mean"]]))
    checks = {
        "dynamic_retention_at_least_0.80": bool(dynamic_retention["mean"] >= 0.80),
        "value_retention_at_least_0.80": bool(value_retention["mean"] >= 0.80),
        "mean_retention_at_least_0.85": bool(mean_retention >= 0.85),
        "at_least_80pct_labels_retain_0.70": bool(fraction >= 0.80),
        "dynamic_beats_leakage_by_0.05": bool(candidate_dynamic - leakage["score"] >= 0.05),
    }
    return {
        "name": "y64_hard_gate",
        "passed": all(checks.values()),
        "failure_status": None if all(checks.values()) else "Y64_GATE_FAILED_STOP",
        "checks": checks,
        "dynamic_retention": dynamic_retention,
        "value_retention": value_retention,
        "mean_retention": mean_retention,
        "per_label_retention": labels,
        "fraction_labels_retaining_0.70": fraction,
        "dynamic_leakage_baseline": leakage,
        "candidate_dynamic": candidate_dynamic,
    }


def write_label_csv(path: Path, summaries: dict[str, dict]) -> None:
    representations = list(summaries)
    labels = summaries[representations[0]]["eligible_dynamic_labels"]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "representation", "prevalence", "AP_mean", "AP_std"])
        for label in labels:
            for representation in representations:
                item = summaries[representation]["dynamic_labels"][label]
                writer.writerow([
                    label, representation, item["task_prevalence"],
                    item["task_macro_average_precision"]["mean"],
                    item["task_macro_average_precision"]["std"],
                ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=("qualification", "y64", "downstream"), required=True)
    args = parser.parse_args()
    results_path = Path(args.results)
    representations = list(LEAKAGE_BASELINES) + ["full_h"]
    if args.stage in {"y64", "downstream"}:
        representations.append("y64")
    if args.stage == "downstream":
        representations.extend(("y32", "x64"))
    raw = {name: _load(results_path, name) for name in representations}
    summaries = {name: summarize(items) for name, items in raw.items()}
    qualification = qualification_gate(summaries)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"stage": args.stage, "summaries": summaries, "qualification_gate": qualification}
    failed = not qualification["passed"]
    if args.stage in {"y64", "downstream"} and not failed:
        report["y64_gate"] = y64_gate(raw["y64"], raw["full_h"], summaries)
        failed = not report["y64_gate"]["passed"]
    if args.stage == "downstream" and not failed:
        report["downstream_retention"] = {
            name: {
                "dynamic": paired_retention(raw[name], raw["full_h"], "dynamic"),
                "value": paired_retention(raw[name], raw["full_h"], "value"),
                "per_label": per_label_retention(raw[name], raw["full_h"]),
            }
            for name in ("y32", "x64")
        }
    write_json(output / f"aggregate-{args.stage}.json", report)
    write_label_csv(output / f"per-label-{args.stage}.csv", summaries)
    print(json.dumps({
        "stage": args.stage,
        "qualification_passed": qualification["passed"],
        "y64_passed": report.get("y64_gate", {}).get("passed"),
        "output": str(output / f"aggregate-{args.stage}.json"),
    }, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
