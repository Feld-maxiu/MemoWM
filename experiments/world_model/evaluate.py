from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from experiments.state_tokenizer.common import sha256_file, write_json

from .cache import CACHE_FILES, FrozenCache
from .config import load_config
from .train import _write_per_episode, evaluate_model, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", default="configs/world_model/v8_discrete.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--test-freeze-manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    cache = FrozenCache(args.cache)
    config = load_config(args.config, num_tasks=len(cache.task_names))
    cache.max_history = config.model.max_history
    jax.config.update("jax_default_matmul_precision", config.training.matmul_precision)
    device = jax.devices(args.platform)[args.device_index]
    checkpoint = load_checkpoint(args.checkpoint)
    variant = checkpoint["metadata"]["variant"]
    params = jax.device_put(checkpoint["params"], device)
    rows = cache.indices_for_split(
        args.split, test_freeze_manifest=args.test_freeze_manifest
    )
    summary, per_transition = evaluate_model(
        cache, rows, params, variant, config, keep_per_transition=True
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "per_transition.npz"
    np.savez_compressed(archive, **per_transition)
    _write_per_episode(output / "per_episode.jsonl", cache, per_transition)
    report = {
        "protocol": "v8_discrete_wm_evaluation_v1",
        "split": args.split,
        "variant": variant,
        "seed": checkpoint["metadata"]["seed"],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "cache_manifest_sha256": sha256_file(Path(args.cache) / CACHE_FILES["manifest"]),
        "summary": summary,
        "per_transition": archive.name,
        "per_transition_sha256": sha256_file(archive),
        "test_freeze_manifest": (
            str(Path(args.test_freeze_manifest).resolve())
            if args.test_freeze_manifest else None
        ),
    }
    write_json(output / "evaluation.json", report)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
