"""Print the headline rows of a ``semantic_eval`` report.

The report carries every per-label leaf, which is the right thing to archive and
the wrong thing to read: the handful of numbers a decision actually turns on get
lost in a few hundred lines of per-task prevalence counts. This pulls out the
summary metrics and shows them next to the R^2 guardrails from the training
records, so a run can be judged against the criteria in one screen.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADLINE = (
    ("value/exact_value_accuracy", "value exact"),
    ("value/macro_position_accuracy", "value position"),
    ("overlap_words/macro_average_precision", "overlap words AP"),
    ("dynamic/eligible_task_macro_average_precision", "dynamic AP"),
    ("dynamic/state_eligible_task_macro_average_precision", "dynamic AP (state)"),
    ("task/accuracy", "task accuracy"),
)
GROUPS = ("all", "detail", "image", "context", "prompt")


def _r2(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    metrics = json.loads(Path(path).read_text())["splits"]["validation"]
    return {name: metrics[f"{name}/r2"] for name in GROUPS if f"{name}/r2" in metrics}


def run(args: argparse.Namespace) -> None:
    report = json.loads(Path(args.report).read_text())
    comparison = report["comparison"]
    names = args.store or [n for n in comparison if n not in {"clean"}]
    reference = report["continuous_reference"]

    width = max(len(label) for _, label in HEADLINE) + 2
    header = f"{'metric':{width}s}" + "".join(f"{n:>12s}" for n in ["clean"] + names)
    print(f"\n== frozen probe, {report['split']}({report['states']}) ==")
    print(header)
    for key, label in HEADLINE:
        row = f"{label:{width}s}"
        for name in ["clean"] + names:
            value = comparison[name]["metrics"].get(key)
            row += f"{value:12.4f}" if value is not None else f"{'-':>12s}"
        print(row)
        # R_disc: how much of what A1 preserved survives discretization
        row = f"{'  R_disc':{width}s}" + f"{'':12s}"
        for name in names:
            keep = comparison[name].get("retention_vs_continuous", {}).get(key)
            row += f"{keep:12.4f}" if keep is not None else (
                f"{'(ref)':>12s}" if name == reference else f"{'-':>12s}")
        print(row)

    if not args.record:
        return
    records = dict(entry.split("=", 1) for entry in args.record)
    print(f"\n== validation R^2 ==")
    print(f"{'group':{width}s}" + "".join(f"{n:>12s}" for n in records))
    values = {name: _r2(path) for name, path in records.items()}
    for group in GROUPS:
        row = f"{group:{width}s}"
        for name in records:
            value = values[name].get(group)
            row += f"{value:12.5f}" if value is not None else f"{'-':>12s}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="semantic_eval output json")
    parser.add_argument("--store", action="append", help="restrict/order the columns")
    parser.add_argument("--record", action="append", default=[],
                        help="name=path of a training json, for the R^2 guardrails")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
