"""Render the three frozen v8 World Model evidence-gate figures."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANT_ORDER = (
    "t_only", "no_action", "structural_action", "no_history", "full"
)


def _read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _save(fig, output: Path, stem: str) -> None:
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _mean_error(values: list[float]) -> tuple[float, float]:
    values_array = np.asarray(values, np.float64)
    return float(values_array.mean()), (
        float(values_array.std(ddof=1)) if len(values_array) > 1 else 0.0
    )


def model_vs_baselines(baseline: dict, runs: list[dict], output: Path) -> None:
    labels = ["marginal", "copy", "source"]
    means = [
        float(baseline["bits_per_transition"][name]["total_bits_per_transition"])
        for name in labels
    ]
    errors = [0.0] * len(labels)
    grouped: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        grouped[run["variant"]].append(
            float(run["best_selection"]["total_bits_per_transition"])
        )
    for variant in VARIANT_ORDER:
        if variant in grouped:
            mean, error = _mean_error(grouped[variant])
            labels.append(variant.replace("_", "-"))
            means.append(mean)
            errors.append(error)
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    positions = np.arange(len(labels))
    colors = ["#8c8c8c"] * 3 + ["#4472c4"] * (len(labels) - 3)
    axis.bar(positions, means, yerr=errors, capsize=3, color=colors)
    axis.set_xticks(positions, labels, rotation=24, ha="right")
    axis.set_ylabel("Held-out ideal codelength (bits / transition)")
    axis.set_title("Frozen v8 discrete WM versus train-fit baselines")
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output, "model_vs_baselines")


def causal_gains(statistics: dict, output: Path) -> None:
    requested = (
        ("history_gain", "history"),
        ("structural_action_gain_random_policy", "structural action\n(random policy)"),
        ("payload_gain", "payload"),
        ("structural_action_net_gain", "structural action\n(after billing)"),
        ("payload_net_gain", "payload\n(after billing)"),
    )
    comparisons = statistics["comparisons"]
    available = [(key, label) for key, label in requested if key in comparisons]
    if not available:
        raise ValueError("statistics JSON has none of the required gain comparisons")
    means, lower, upper = [], [], []
    for key, _label in available:
        value = comparisons[key]
        point = float(value["point_mean"])
        low, high = map(float, value["micro"]["ci95"])
        means.append(point)
        lower.append(point - low)
        upper.append(high - point)
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    positions = np.arange(len(available))
    colors = ["#4472c4" if "net" not in key else "#ed7d31" for key, _ in available]
    axis.bar(
        positions, means, yerr=np.asarray([lower, upper]), capsize=4, color=colors
    )
    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xticks(positions, [label for _key, label in available])
    axis.set_ylabel("Paired gain (bits / transition; positive is better)")
    axis.set_title("History, action, and payload evidence")
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output, "history_action_payload_gains")


def scale_and_rollout(
    scale_runs: list[dict], rollouts: list[dict], output: Path
) -> None:
    if not scale_runs or not rollouts:
        raise ValueError("scale_and_rollout requires both scale runs and rollouts")
    scale_values: dict[int, list[float]] = defaultdict(list)
    for run in scale_runs:
        scale_values[int(run["data"]["train_transitions"])].append(
            float(run["best_selection"]["total_bits_per_transition"])
        )
    sizes = sorted(scale_values)
    scale_mean, scale_error = zip(*[_mean_error(scale_values[size]) for size in sizes])

    horizon_values: dict[int, list[float]] = defaultdict(list)
    copy_values: dict[int, list[float]] = defaultdict(list)
    source_values: dict[int, list[float]] = defaultdict(list)
    for rollout in rollouts:
        for horizon_text, value in rollout["summary_by_horizon"].items():
            horizon = int(horizon_text)
            horizon_values[horizon].append(float(value["cumulative_bits"]))
            if "copy_cumulative_bits" in value:
                copy_values[horizon].append(float(value["copy_cumulative_bits"]))
                source_values[horizon].append(float(value["source_cumulative_bits"]))
    horizons = sorted(horizon_values)
    rollout_mean, rollout_error = zip(
        *[_mean_error(horizon_values[horizon]) for horizon in horizons]
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].errorbar(sizes, scale_mean, yerr=scale_error, marker="o", capsize=3)
    axes[0].set_xlabel("Train transitions (episode-complete subset)")
    axes[0].set_ylabel("Validation bits / transition")
    axes[0].set_title("Data scale at fixed 20k updates")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        horizons, rollout_mean, yerr=rollout_error, marker="o", capsize=3,
        label="closed-loop WM",
    )
    if copy_values:
        axes[1].plot(
            horizons,
            [np.mean(copy_values[horizon]) for horizon in horizons],
            marker="s", linestyle="--", label="copy baseline",
        )
        axes[1].plot(
            horizons,
            [np.mean(source_values[horizon]) for horizon in horizons],
            marker="^", linestyle="--", label="source baseline",
        )
    axes[1].set_xticks(horizons)
    axes[1].set_xlabel("Closed-loop horizon")
    axes[1].set_ylabel("Mean cumulative bits")
    axes[1].set_title("Known-action closed-loop rollout")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    _save(fig, output, "data_scale_and_closed_loop")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--run-json", action="append", required=True)
    parser.add_argument("--statistics-json")
    parser.add_argument("--scale-run-json", action="append", default=[])
    parser.add_argument("--rollout-json", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    produced = []
    model_vs_baselines(
        _read(args.baseline_json), [_read(path) for path in args.run_json], output
    )
    produced.append("model_vs_baselines")
    if args.statistics_json:
        causal_gains(_read(args.statistics_json), output)
        produced.append("history_action_payload_gains")
    has_scale = bool(args.scale_run_json)
    has_rollout = bool(args.rollout_json)
    if has_scale != has_rollout:
        raise ValueError(
            "scale and closed-loop figure requires both scale runs and rollouts"
        )
    if has_scale:
        scale_and_rollout(
            [_read(path) for path in args.scale_run_json],
            [_read(path) for path in args.rollout_json],
            output,
        )
        produced.append("data_scale_and_closed_loop")
    print(json.dumps({
        "output": str(output.resolve()),
        "figures": produced,
    }, indent=2))


if __name__ == "__main__":
    main()
