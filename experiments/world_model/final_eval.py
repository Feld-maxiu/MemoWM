"""Formal final evaluation of the WebWorld-trained world model on AMA.

Produces the report artifact for the transfer story:
  - AMA validation, uncalibrated (raw v2fix checkpoint)
  - AMA validation, calibrated (post-hoc per-action-type copy-gate deltas,
    fitted on the AMA train split; validation untouched)
  - WebWorld reference validation, uncalibrated (the AMA-fitted bias is
    not applied on the source domain)
  - baseline snapshot for the comparison table
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
from pathlib import Path

import jax
import numpy as np

from .calibrate_gate_action import _with_action_bias
from .config import load_config
from .cache import FrozenCache
from .schema_web import ACTION_TYPE_IDS
from .train import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--reference-cache")
    parser.add_argument("--baseline-json")
    parser.add_argument("--bias-json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    device = jax.local_devices()[args.device_index]
    with open(args.checkpoint, "rb") as handle:
        checkpoint = pickle.load(handle)
    params = jax.device_put(dict(checkpoint["params"]), device)
    checkpoint_step = int(checkpoint.get("step", -1))

    cache = FrozenCache(args.cache)
    config = load_config(args.config, num_tasks=len(cache.task_names))
    cache.max_history = config.model.max_history
    rows = cache.indices_for_split("validation")

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "protocol": "wm-final-eval-v1",
        "selection_cache": str(args.cache),
        "reference_cache": args.reference_cache,
    }

    # 1) AMA validation, uncalibrated.
    summary_plain, per_plain = evaluate_model(
        cache, rows, params, "full", config, keep_per_transition=True
    )
    report["ama_validation_uncalibrated"] = summary_plain

    # 2) AMA validation, calibrated with per-action-type gate deltas.
    per_transition = {"ama_uncalibrated": per_plain}
    summary_cal = None
    if args.bias_json:
        with open(args.bias_json) as handle:
            bias_spec = json.load(handle)
        bias = np.full(
            len(ACTION_TYPE_IDS),
            bias_spec.get("default_delta_for_unlisted_types", 0.0),
            np.float32,
        )
        for name, value in bias_spec["fitted_delta"].items():
            bias[ACTION_TYPE_IDS[name]] = float(value)
        calibrated_params = _with_action_bias(params, bias)
        biased_config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, use_action_gate_bias=True),
        )
        summary_cal, per_cal = evaluate_model(
            cache, rows, calibrated_params, "full", biased_config,
            keep_per_transition=True,
        )
        report["ama_validation_calibrated"] = {
            **summary_cal,
            "bias_json": str(args.bias_json),
            "bias_method": bias_spec.get("method"),
        }
        per_transition["ama_calibrated"] = per_cal

    # 3) WebWorld reference validation, uncalibrated (no AMA-fitted bias).
    if args.reference_cache:
        reference = FrozenCache(args.reference_cache)
        reference.max_history = config.model.max_history
        reference_rows = reference.indices_for_split("validation")
        summary_ref, per_ref = evaluate_model(
            reference, reference_rows, params, "full", config,
            keep_per_transition=True,
        )
        report["webworld_reference_uncalibrated"] = summary_ref
        per_transition["webworld_uncalibrated"] = per_ref

    # 4) Baseline snapshot for the comparison table.
    if args.baseline_json:
        with open(args.baseline_json) as handle:
            report["baselines_snapshot"] = json.load(handle)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "report.json", "w") as handle:
        json.dump(report, handle, indent=1)
    np.savez_compressed(output / "per_transition.npz", **{
        f"{name}/{key}": value
        for name, part in per_transition.items()
        for key, value in part.items()
    })

    print(json.dumps({
        "ama_uncalibrated": round(
            report["ama_validation_uncalibrated"]["code_bits_per_transition"], 2),
        "ama_calibrated": None if summary_cal is None else round(
            summary_cal["code_bits_per_transition"], 2),
        "webworld_reference": None if not args.reference_cache else round(
            report["webworld_reference_uncalibrated"]["code_bits_per_transition"], 2),
        "artifacts": [str(output / "report.json"),
                      str(output / "per_transition.npz")],
    }, indent=1))


if __name__ == "__main__":
    main()
