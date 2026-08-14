"""Create the immutable manifest that is the sole key for opening v8 test data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from experiments.state_tokenizer.common import sha256_file, write_json

from .cache import CACHE_FILES, FrozenCache
from .config import load_config, resolved_dict
from .train import load_checkpoint


TEST_FREEZE_PROTOCOL = "v8_wm_test_freeze_v1"


def resolved_config_sha256(config, *, variant: str, seed: int) -> str:
    """Hash the canonical resolved config exactly as training writes it."""
    payload = yaml.safe_dump(
        resolved_dict(config, variant=variant, seed=seed), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--statistics-json", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument(
        "--curriculum-decision",
        choices=("not_triggered", "kept", "rejected"), required=True,
    )
    parser.add_argument("--curriculum-decision-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache = FrozenCache(args.cache, verify_hashes=True)
    formal_config = load_config(args.config, num_tasks=len(cache.task_names))
    cache_manifest = Path(args.cache) / CACHE_FILES["manifest"]
    cache_manifest_sha256 = sha256_file(cache_manifest)
    baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
    if not (
        baseline.get("protocol") == "v8_task_conditioned_baselines_v1"
        and baseline.get("fit_split") == "train"
        and baseline.get("eval_split") == "validation"
        and baseline.get("cache_manifest_sha256") == cache_manifest_sha256
    ):
        raise ValueError("baseline is not the matching train-fit validation artifact")

    statistics = json.loads(Path(args.statistics_json).read_text(encoding="utf-8"))
    if (
        statistics.get("protocol") != "v8_task_stratified_episode_bootstrap_v1"
        or statistics.get("test_used") is not False
    ):
        raise ValueError("statistics artifact has the wrong protocol or used test data")
    if not statistics.get("all_required_gates_passed"):
        raise ValueError("required validation gates did not pass; refusing test unlock")

    curriculum_artifact = None
    if args.curriculum_decision != "not_triggered":
        if not args.curriculum_decision_json:
            raise ValueError("kept/rejected curriculum requires its decision JSON")
        curriculum_path = Path(args.curriculum_decision_json)
        curriculum = json.loads(curriculum_path.read_text(encoding="utf-8"))
        if not (
            curriculum.get("protocol") == "v8_wm_curriculum_decision_v1"
            and curriculum.get("decision") == args.curriculum_decision
        ):
            raise ValueError("curriculum decision artifact disagrees with requested state")
        curriculum_artifact = {
            "path": str(curriculum_path.resolve()),
            "sha256": sha256_file(curriculum_path),
        }
    elif args.curriculum_decision_json:
        raise ValueError("not_triggered cannot be paired with a curriculum decision JSON")

    if len(args.checkpoint) != 3:
        raise ValueError("test freeze requires exactly three final checkpoints")
    checkpoints = []
    variants, seeds = set(), set()
    curriculum_flags = []
    for path in args.checkpoint:
        value = load_checkpoint(path)
        metadata = value["metadata"]
        if metadata.get("cache_manifest_sha256") != cache_manifest_sha256:
            raise ValueError(f"checkpoint {path} was trained on a different cache")
        variant = metadata["variant"]
        seed = int(metadata["seed"])
        expected_config_sha256 = resolved_config_sha256(
            formal_config, variant=variant, seed=seed
        )
        if metadata.get("config_sha256") != expected_config_sha256:
            raise ValueError(
                f"checkpoint {path} was not trained with the frozen formal config"
            )
        variants.add(variant)
        seeds.add(seed)
        curriculum_flags.append("curriculum" in metadata)
        checkpoints.append({
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
            "variant": variant,
            "seed": seed,
            "step": int(value["step"]),
            "resolved_config_sha256": expected_config_sha256,
            "curriculum": metadata.get("curriculum"),
        })
    if len(variants) != 1 or seeds != {0, 1, 2}:
        raise ValueError(
            f"final checkpoints must share one variant and use seeds 0/1/2; "
            f"got variants={variants}, seeds={seeds}"
        )
    if args.curriculum_decision == "kept" and not all(curriculum_flags):
        raise ValueError("kept curriculum requires three curriculum checkpoints")
    if args.curriculum_decision != "kept" and any(curriculum_flags):
        raise ValueError("rejected/not-triggered curriculum must use base checkpoints")
    selected = statistics.get("history_selection")
    if variants != {selected}:
        raise ValueError(
            f"statistics selected {selected}, but checkpoints are {sorted(variants)}"
        )

    root = Path(__file__).resolve().parent
    source_files = [
        root / "schema.py", root / "cache.py", root / "subsets.py",
        root / "baselines.py", root / "config.py", root / "model.py",
        root / "train.py", root / "evaluate.py", root / "rollout.py",
        root / "semantic_rollout.py", root / "semantic_utils.py",
        root / "statistics.py", root / "curriculum.py", root / "curriculum_decision.py",
        root / "figures.py", root / "freeze.py",
    ]
    missing = [str(path) for path in source_files if not path.exists()]
    if missing:
        raise ValueError(f"cannot freeze missing implementation files: {missing}")
    manifest = {
        "protocol": TEST_FREEZE_PROTOCOL,
        "test_unlocked": True,
        "cache_manifest": str(cache_manifest.resolve()),
        "cache_manifest_sha256": cache_manifest_sha256,
        "config": str(Path(args.config).resolve()),
        "config_sha256": sha256_file(args.config),
        "baseline_json": str(Path(args.baseline_json).resolve()),
        "baseline_sha256": sha256_file(args.baseline_json),
        "statistics_json": str(Path(args.statistics_json).resolve()),
        "statistics_sha256": sha256_file(args.statistics_json),
        "validation_gates": statistics["gates"],
        "selected_variant": next(iter(variants)),
        "curriculum_decision": args.curriculum_decision,
        "curriculum_decision_artifact": curriculum_artifact,
        "checkpoints": sorted(checkpoints, key=lambda item: item["seed"]),
        "source_sha256": {
            path.name: sha256_file(path) for path in source_files
        },
        "frozen_decisions": [
            "tokenizer/cache", "architecture", "hyperparameters", "ablations",
            "checkpoint selection", "history selection", "curriculum decision",
        ],
        "test_policy": "one_time_sign_confirmation_no_tuning",
        "cache_summary": {
            "states": cache.manifest["state_counts"],
            "transitions": cache.manifest["transition_counts"],
        },
    }
    write_json(args.output, manifest)
    print(json.dumps({
        "test_unlocked": True,
        "selected_variant": manifest["selected_variant"],
        "checkpoint_seeds": sorted(seeds),
        "output": str(Path(args.output).resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
