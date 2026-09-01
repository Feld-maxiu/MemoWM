"""Train and evaluate the world model from the packed single-file dataset.

    unpack   worldmemarena_wm_train.npz -> a cache directory train.py can read
    evaluate a checkpoint on the dataset's evaluation split

Usage:
    python scripts_wm_dataset.py unpack   --dataset D.npz --output CACHE
    python scripts_wm_dataset.py evaluate --dataset D.npz --cache CACHE \
                                          --checkpoint best.pkl --config CFG
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

TRANSITION_COLUMNS = (
    "history_indices", "target_indices", "task_ids", "episode_ids", "steps",
    "split_ids", "action_types", "action_payloads", "action_lengths",
    "action_x", "action_y", "action_dx", "action_dy",
    "action_has_coord", "action_has_delta",
    "structural_action_bits", "full_action_bits",
)


def unpack(dataset: Path, output: Path) -> dict:
    data = np.load(dataset, allow_pickle=True)
    meta = json.loads(str(data["metadata"]))
    if "split_ids" in data:
        roles = np.asarray(data["split_ids"], np.int8)
    else:
        roles = np.zeros(len(data["target_indices"]), np.int8)

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    np.save(output / "codes.npy", data["codes"])
    np.save(output / "valid.npy", data["valid"])
    np.save(output / "global_indices.npy",
            np.arange(len(data["codes"]), dtype=np.int64))

    columns = {name: data[name] for name in TRANSITION_COLUMNS if name in data}
    columns["split_ids"] = np.where(roles == 1, 1, 0).astype(np.uint8)
    np.savez(output / "transitions.npz", **columns)

    episodes = int(np.max(columns["episode_ids"])) + 1
    states = int(len(data["codes"]))
    evaluated = int((roles == 1).sum())
    manifest = {
        "protocol": meta["protocol"],
        "format_version": meta["format_version"],
        "codes": [str(dataset)],
        "num_latent_tokens": meta["num_latent_tokens"],
        "num_subspaces": meta["num_subspaces"],
        "num_categories": meta["num_categories"],
        "layout": [meta["num_latent_tokens"]],
        "fixed_width_bits": meta["fixed_width_bits"],
        "max_history": meta["max_history"],
        "tasks": ["state"],
        "episodes": episodes,
        "episode_names": [f"ep_{i:06d}" for i in range(episodes)],
        # state_counts partitions the states, so every state sits under train.
        # Which transitions are evaluated is carried by transitions/split_ids,
        # which is what indices_for_split reads.
        "state_counts": {"train": states},
        "transition_counts": {"train": int(len(roles)), "validation": evaluated},
        "transitions": int(len(roles)),
        "artifact_sha256": {},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    unpacker = sub.add_parser("unpack")
    unpacker.add_argument("--dataset", required=True)
    unpacker.add_argument("--output", required=True)

    scorer = sub.add_parser("evaluate")
    scorer.add_argument("--dataset", required=True)
    scorer.add_argument("--cache", required=True)
    scorer.add_argument("--checkpoint", required=True)
    scorer.add_argument("--config", required=True)
    scorer.add_argument("--platform", default="gpu")
    scorer.add_argument("--device-index", type=int, default=0)
    scorer.add_argument("--output")

    args = parser.parse_args()

    if args.command == "unpack":
        manifest = unpack(Path(args.dataset), Path(args.output))
        print(json.dumps({k: manifest[k] for k in
                          ("transitions", "transition_counts", "fixed_width_bits",
                           "max_history", "num_categories")},
                         ensure_ascii=False, indent=2))
        return

    import jax
    from experiments.world_model.cache import FrozenCache
    from experiments.world_model.config import load_config
    from experiments.world_model.train import evaluate_model, load_checkpoint

    data = np.load(args.dataset, allow_pickle=True)
    width = json.loads(str(data["metadata"]))["fixed_width_bits"]

    cache = FrozenCache(args.cache)
    config = load_config(args.config, num_tasks=len(cache.task_names))
    cache.max_history = config.model.max_history
    jax.config.update("jax_default_matmul_precision",
                      config.training.matmul_precision)
    device = jax.devices(args.platform)[args.device_index]
    checkpoint = load_checkpoint(args.checkpoint)
    params = jax.device_put(checkpoint["params"], device)

    rows = cache.indices_for_split("validation", test_freeze_manifest=None)
    summary, _ = evaluate_model(cache, rows, params,
                                checkpoint["metadata"]["variant"], config)
    bits = summary["total_bits_per_transition"]
    report = {
        "transitions": int(len(rows)),
        "fixed_width_bits": width,
        "bits_per_transition": bits,
        "compression_ratio": width / bits,
        "code_accuracy": summary["code_accuracy"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
