from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from residualmem.codec.segment import decode_memory, encode_memory
from residualmem.codec.state import encode_full_state
from residualmem.evaluation.metrics import reconstruction_metrics
from residualmem.types import Predictor, StateSchema, Trajectory


def independent_state_bytes(trajectory: Trajectory, schema: StateSchema) -> int:
    return sum(4 + len(encode_full_state(state, schema)) for state in trajectory.states)


def store_all_fixed_bytes(trajectory: Trajectory, schema: StateSchema) -> int:
    action_bytes = 4 * len(trajectory.actions)
    state_bytes = 8 * len(schema.fields) * len(trajectory.states)
    return action_bytes + state_bytes


def run_codec_suite(
    trajectory: Trajectory,
    schema: StateSchema,
    variants: list[dict[str, Any]],
    output_dir: str | Path,
    segment_length: int = 64,
) -> list[dict[str, Any]]:
    if not trajectory.actions:
        raise ValueError("evaluation needs at least one transition")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for variant in variants:
        name = str(variant["name"])
        predictor: Predictor = variant["predictor"]
        path = output_dir / f"{name}.rsm"
        accounting = encode_memory(
            path=path,
            trajectory=trajectory,
            predictor=predictor,
            schema=schema,
            segment_length=segment_length,
            actions_external=bool(variant.get("actions_external", False)),
        )
        external_actions = (
            trajectory.actions if variant.get("actions_external", False) else None
        )
        decoded, decoded_accounting = decode_memory(
            path, predictor, schema, external_actions=external_actions
        )
        if decoded_accounting != accounting:
            raise AssertionError("encoder/decoder bit accounting differs")
        metrics = reconstruction_metrics(trajectory.states, decoded.states, schema)
        rows.append(
            {
                "name": name,
                "bytes": path.stat().st_size,
                "bytes_per_step": path.stat().st_size / len(trajectory.actions),
                **accounting.as_dict(),
                **metrics,
            }
        )

    for name, total in (
        ("store_all_fixed64", store_all_fixed_bytes(trajectory, schema)),
        ("independent_typed_state", independent_state_bytes(trajectory, schema)),
    ):
        rows.append(
            {
                "name": name,
                "bytes": total,
                "bytes_per_step": total / len(trajectory.actions),
                "exact_field_accuracy": 1.0,
                "exact_state_accuracy": 1.0,
            }
        )

    (output_dir / "results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    keys = sorted({key for row in rows for key in row})
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_results(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    names = [row["name"] for row in rows]
    bytes_per_step = [row["bytes_per_step"] for row in rows]

    fig, axis = plt.subplots(figsize=(max(8, len(names) * 1.25), 4.8))
    axis.bar(names, bytes_per_step)
    axis.set_ylabel("Bytes per transition")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "compression.png", dpi=180)
    plt.close(fig)
