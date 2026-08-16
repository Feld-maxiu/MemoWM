"""Summarise dev runs with a selection-noise-robust statistic.

``best`` in a run artifact is the raw minimum over the eval trajectory. Section
23.2 of the worklog showed that for these runs the trajectory has a wide flat
basin -- for 8L/2048, every eval from step 88,000 to 240,000 sits within 30
bits of the minimum -- so the raw minimum is the smallest of ~60 near-equal
noisy samples. That makes it a *downward-biased* estimator whose bias grows
with the number of evaluations, which is a large part of why two runs of an
identical config differed by 110.13 bits.

The headline statistic here is therefore the minimum of a centred 5-eval moving
average of dev code bits. Averaging neighbouring evals cuts the sampling noise
before the minimum is taken, so it is far less sensitive to how many lottery
tickets a run drew. The raw minimum is still reported alongside, since every
earlier section of the worklog quotes it.

Comparisons remain valid only when the runs being compared have the same number
of evaluations; the data-scaling cells are configured so that they do (120).
"""
from __future__ import annotations

import argparse
import glob
import json


def smoothed_best(history, window: int = 5):
    """Minimum of a centred moving average; returns (value, step, evals)."""
    values = [h["code_bits_per_transition"] for h in history]
    steps = [h["step"] for h in history]
    if len(values) < window:
        best = min(range(len(values)), key=values.__getitem__)
        return values[best], steps[best], len(values)
    half = window // 2
    best_value, best_step = float("inf"), None
    for i in range(half, len(values) - half):
        mean = sum(values[i - half:i + half + 1]) / window
        if mean < best_value:
            best_value, best_step = mean, steps[i]
    return best_value, best_step, len(values)


def summarise(path: str, window: int) -> dict:
    data = json.loads(open(path, encoding="utf-8").read())
    history = data.get("history") or []
    value, step, evals = smoothed_best(history, window)
    split = data.get("split", {})
    capacity = data.get("capacity", {})
    budget = data.get("budget", {})
    return {
        "run": path.split("/")[-1].removesuffix(".json"),
        "layers": capacity.get("num_layers"),
        "mlp": capacity.get("mlp_dim"),
        "dropout": capacity.get("dropout"),
        "weight_decay": budget.get("weight_decay"),
        "updates": budget.get("max_steps"),
        "evals": evals,
        "fit": split.get("fit_transitions"),
        "fit_subsample": split.get("fit_subsample"),
        "epochs": split.get("epochs"),
        "smoothed_best": value,
        "smoothed_best_step": step,
        "raw_best": data.get("best", {}).get("code_bits_per_transition"),
        "raw_best_step": data.get("best_step"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="dev result JSONs (globs ok)")
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    rows = []
    for pattern in args.paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            try:
                rows.append(summarise(path, args.window))
            except (OSError, json.JSONDecodeError, KeyError) as error:
                print(f"skip {path}: {error}")

    header = (f"{'run':<30}{'L':>3}{'MLP':>6}{'drop':>6}{'upd':>9}{'ev':>5}"
              f"{'fit':>8}{'ep':>5}{'smoothed':>10}{'@step':>9}{'raw':>10}{'diff':>8}")
    print(header)
    for r in sorted(rows, key=lambda r: r["smoothed_best"]):
        epochs = f"{r['epochs']:.0f}" if r["epochs"] else "-"
        print(f"{r['run']:<30}{r['layers'] or 0:>3}{r['mlp'] or 0:>6}"
              f"{r['dropout'] if r['dropout'] is not None else 0:>6.2f}"
              f"{r['updates'] or 0:>9,}{r['evals']:>5}{r['fit'] or 0:>8,}{epochs:>5}"
              f"{r['smoothed_best']:>10.2f}{r['smoothed_best_step'] or 0:>9,}"
              f"{r['raw_best']:>10.2f}{r['smoothed_best'] - r['raw_best']:>+8.2f}")


if __name__ == "__main__":
    main()
