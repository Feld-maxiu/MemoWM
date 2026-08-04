from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from residualmem.codec.segment import decode_memory, encode_memory
from residualmem.codec.segment_v4 import encode_latent_memory
from residualmem.codec.latent_session import LatentMemoryFile
from residualmem.codec.send_mask import ExactMaskPolicy, RateDistortionMaskPolicy
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


def store_all_latent_bytes(latent_trajectory: Trajectory, latent_schema: StateSchema) -> int:
    """Denominator for Memory Rate (report Eq. 47a): every latent group stored fully."""
    state_bytes = sum(len(encode_full_state(state, latent_schema))
                      for state in latent_trajectory.states)
    action_bytes = 2 * len(latent_trajectory.actions)
    return state_bytes + action_bytes


def run_latent_rate_distortion(
    trajectory: Trajectory,
    params,
    config,
    encoder,
    domain,
    latent_schema: StateSchema,
    output_dir: str | Path,
    *,
    lambdas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
    segment_length: int = 64,
) -> list[dict[str, Any]]:
    """Encode the trajectory at several operating points, decode+render, and report
    real bytes vs. reconstruction quality (report: State Quality vs Total Memory Bytes).

    Variants: ``wm_only`` (store nothing -> WM prior rollout), ``rd_lam*`` (rate knob),
    ``exact`` (lossless w.r.t. the realized latent). Memory Rate = bytes / Store-All-Latent.
    No task/utility gating is involved -- the send decision is pure rate-distortion.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_schema = encoder.schema
    variants = (
        [("wm_only", RateDistortionMaskPolicy(lam=-1.0))]
        + [(f"rd_lam{lam:g}", RateDistortionMaskPolicy(lam=lam)) for lam in lambdas]
        + [("exact", ExactMaskPolicy())]
    )
    rows: list[dict[str, Any]] = []
    store_all = None
    for name, policy in variants:
        path = output_dir / f"{name}.rs4"
        accounting, latent_traj = encode_latent_memory(
            path, trajectory, params, config, encoder, domain, latent_schema,
            mask_policy=policy, segment_length=segment_length)
        if name == "exact":
            store_all = store_all_latent_bytes(latent_traj, latent_schema)
        memory = LatentMemoryFile(path, params, config, domain, latent_schema, encoder)
        approx = memory.render_trajectory()
        metrics = reconstruction_metrics(trajectory.states, approx.states, canonical_schema)
        rows.append({
            "name": name,
            "bytes": path.stat().st_size,
            "bytes_per_step": path.stat().st_size / max(1, len(trajectory.actions)),
            "residual_bytes": accounting.residuals,
            **metrics,
        })
    for row in rows:
        row["memory_rate"] = row["bytes"] / max(1, store_all)
        row["state_distortion"] = 1.0 - row.get("exact_field_accuracy", 0.0)

    (output_dir / "rate_distortion.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    keys = sorted({key for row in rows for key in row})
    with (output_dir / "rate_distortion.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_rate_distortion(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    ordered = sorted(rows, key=lambda r: r["bytes_per_step"])
    fig, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.plot([r["bytes_per_step"] for r in ordered],
              [r["state_distortion"] for r in ordered], marker="o")
    for row in ordered:
        axis.annotate(row["name"], (row["bytes_per_step"], row["state_distortion"]),
                      fontsize=7, xytext=(4, 4), textcoords="offset points")
    axis.set_xlabel("Actual bytes per transition")
    axis.set_ylabel("State distortion (1 - exact field accuracy)")
    axis.set_title("ResidualMem latent rate-distortion")
    fig.tight_layout()
    fig.savefig(output_dir / "rate_distortion.png", dpi=180)
    plt.close(fig)


def pareto_operating_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rate-performance frontier: points not dominated on both bytes and distortion
    (report Stage 6 operating points). Sorted by increasing bytes_per_step."""
    ordered = sorted(rows, key=lambda r: (r["bytes_per_step"], r["state_distortion"]))
    frontier: list[dict[str, Any]] = []
    best = float("inf")
    for row in ordered:
        if row["state_distortion"] < best - 1e-12:
            frontier.append(row)
            best = row["state_distortion"]
    return frontier
