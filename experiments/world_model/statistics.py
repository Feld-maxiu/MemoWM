"""Three-seed, task-stratified episode paired bootstrap for WM evidence gates."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from experiments.state_tokenizer.common import sha256_file, write_json

from .cache import V8_EXPECTED_TRANSITIONS
from .schema import POLICY_IDS, PROTOCOL, VARIANTS


def _load(path: str | Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    return {name: archive[name] for name in archive.files}


def _align(reference: dict, candidate: dict, label: str):
    for name in ("transition_indices", "task_ids", "episode_ids", "policies"):
        if name not in candidate or not np.array_equal(reference[name], candidate[name]):
            raise ValueError(f"{label} is not paired/aligned on {name}")


def paired_episode_bootstrap(
    differences: np.ndarray,
    task_ids: np.ndarray,
    episode_ids: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 0,
) -> dict:
    """Bootstrap paired gains, drawing one seed then complete episodes/task."""
    differences = np.asarray(differences, np.float64)
    if differences.ndim == 1:
        differences = differences[None, :]
    seeds, transitions = differences.shape
    if task_ids.shape != (transitions,) or episode_ids.shape != (transitions,):
        raise ValueError("bootstrap metadata shape mismatch")
    rng = np.random.default_rng(seed)
    seed_draw = rng.integers(seeds, size=replicates)
    micro_sum = np.zeros((replicates,), np.float64)
    micro_count = np.zeros((replicates,), np.float64)
    task_means = np.zeros((replicates, len(np.unique(task_ids))), np.float64)

    for task_column, task in enumerate(sorted(np.unique(task_ids).tolist())):
        task_rows = np.flatnonzero(task_ids == task)
        episodes = np.unique(episode_ids[task_rows])
        sums = np.zeros((seeds, len(episodes)), np.float64)
        counts = np.zeros((len(episodes),), np.int64)
        for index, episode in enumerate(episodes):
            local = task_rows[episode_ids[task_rows] == episode]
            sums[:, index] = differences[:, local].sum(axis=1, dtype=np.float64)
            counts[index] = len(local)
        task_sum = np.zeros((replicates,), np.float64)
        task_count = np.zeros((replicates,), np.float64)
        for start in range(0, replicates, 256):
            stop = min(start + 256, replicates)
            draws = rng.integers(len(episodes), size=(stop - start, len(episodes)))
            chosen_seed = seed_draw[start:stop, None]
            task_sum[start:stop] = sums[chosen_seed, draws].sum(axis=1)
            task_count[start:stop] = counts[draws].sum(axis=1)
        micro_sum += task_sum
        micro_count += task_count
        task_means[:, task_column] = task_sum / np.maximum(task_count, 1)
    micro = micro_sum / np.maximum(micro_count, 1)
    macro = task_means.mean(axis=1)

    def interval(values):
        return {
            "bootstrap_mean": float(values.mean(dtype=np.float64)),
            "ci95": [
                float(np.percentile(values, 2.5)),
                float(np.percentile(values, 97.5)),
            ],
        }

    point_by_seed = differences.mean(axis=1, dtype=np.float64)
    return {
        "seeds": seeds,
        "replicates": replicates,
        "point_mean": float(point_by_seed.mean(dtype=np.float64)),
        "point_by_seed": point_by_seed.tolist(),
        "micro": interval(micro),
        "task_macro": interval(macro),
    }


def _stack(runs: dict[str, dict[int, dict]], variant: str, field: str) -> np.ndarray:
    return np.stack([runs[variant][seed][field] for seed in (0, 1, 2)])


def _validate_run_artifact(variant: str, seed: int, path: Path) -> tuple[dict, dict]:
    run_path = path.with_name("run.json")
    if not run_path.exists():
        raise FileNotFoundError(f"{path}: sibling run.json is required")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    data = run.get("data", {})
    run_artifacts = run.get("artifacts", {})
    if not (
        run.get("protocol") == PROTOCOL
        and run.get("variant") == variant
        and int(run.get("seed", -1)) == seed
        and data.get("selection") == "validation"
        and data.get("test_evaluated") is False
        and data.get("subset") is None
        # Diagnostic runs stamp dev_run=true. Formal runs predating this flag
        # omit it entirely, so .get() returns None and they stay admissible.
        and data.get("dev_run") is not True
        and int(data.get("train_transitions", -1))
        == V8_EXPECTED_TRANSITIONS["train"]
        and int(data.get("selection_transitions", -1))
        == V8_EXPECTED_TRANSITIONS["validation"]
        and run_artifacts.get("per_transition_sha256") == sha256_file(path)
    ):
        raise ValueError(f"{path}: run.json does not authorize this validation artifact")

    if run_artifacts.get("resolved_config") != "resolved.yaml":
        raise ValueError(f"{path}: run must declare its sibling resolved.yaml")
    resolved_path = run_path.with_name("resolved.yaml")
    if not resolved_path.exists():
        raise FileNotFoundError(f"{path}: sibling resolved.yaml is required")
    resolved_sha256 = sha256_file(resolved_path)
    if run_artifacts.get("resolved_config_sha256") != resolved_sha256:
        raise ValueError(f"{path}: resolved config hash is absent or stale")
    resolved = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    if not (
        isinstance(resolved, dict)
        and resolved.get("protocol") == PROTOCOL
        and resolved.get("variant") == variant
        and int(resolved.get("seed", -1)) == seed
    ):
        raise ValueError(f"{path}: resolved config identity disagrees with the run")
    resolved_training = resolved.get("training")
    if not isinstance(resolved_training, dict) or any(
        run.get("training", {}).get(name) != value
        for name, value in resolved_training.items()
    ):
        raise ValueError(f"{path}: run training metadata disagrees with resolved config")
    resolved_core = dict(resolved)
    resolved_core.pop("variant")
    resolved_core.pop("seed")
    signature = {
        "resolved_core": resolved_core,
        "parameter_count": int(run.get("parameter_count", -1)),
        "parameter_shapes": run.get("parameter_shapes"),
    }
    if signature["parameter_count"] <= 0 or not isinstance(
        signature["parameter_shapes"], dict
    ):
        raise ValueError(f"{path}: missing parameter provenance")
    artifact = {
        "variant": variant,
        "seed": seed,
        "npz": str(path.resolve()),
        "npz_sha256": sha256_file(path),
        "run_json": str(run_path.resolve()),
        "run_json_sha256": sha256_file(run_path),
        "resolved_config": str(resolved_path.resolve()),
        "resolved_config_sha256": resolved_sha256,
        "parameter_count": signature["parameter_count"],
        "cache_manifest_sha256": run["data"]["cache_manifest_sha256"],
    }
    return artifact, signature


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", nargs=3, action="append", metavar=("VARIANT", "SEED", "NPZ"),
        required=True,
    )
    parser.add_argument("--baseline", required=True, help="baseline per_transition.npz")
    parser.add_argument(
        "--expected-final-variant", "--final-variant",
        dest="expected_final_variant", choices=("full", "no_history"),
    )
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runs: dict[str, dict[int, dict]] = defaultdict(dict)
    artifacts = []
    formal_signatures = []
    reference = None
    for variant, seed_text, path_text in args.run:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}")
        seed = int(seed_text)
        if seed in runs[variant]:
            raise ValueError(f"duplicate {variant} seed {seed}")
        path = Path(path_text)
        value = _load(path)
        if reference is None:
            reference = value
        else:
            _align(reference, value, f"{variant}/{seed}")
        runs[variant][seed] = value
        artifact, signature = _validate_run_artifact(variant, seed, path)
        artifacts.append(artifact)
        formal_signatures.append(signature)
    expected = set(VARIANTS)
    if set(runs) != expected:
        raise ValueError(f"final statistics require variants {sorted(expected)}")
    for variant in VARIANTS:
        if set(runs[variant]) != {0, 1, 2}:
            raise ValueError(f"{variant} requires exactly seeds 0/1/2")
    cache_hashes = {artifact["cache_manifest_sha256"] for artifact in artifacts}
    if len(cache_hashes) != 1:
        raise ValueError("run artifacts do not share one frozen cache")

    serialized_signatures = {
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in formal_signatures
    }
    if len(serialized_signatures) != 1:
        raise ValueError(
            "final runs do not share one architecture and formal hyperparameter set"
        )
    formal_signature = next(iter(serialized_signatures))
    formal_signature_sha256 = hashlib.sha256(
        formal_signature.encode("utf-8")
    ).hexdigest()

    baseline_path = Path(args.baseline)
    baseline = _load(baseline_path)
    _align(reference, baseline, "baseline")
    baseline_json_path = baseline_path.with_name("baseline.json")
    baseline_json = json.loads(baseline_json_path.read_text(encoding="utf-8"))
    if not (
        baseline_json.get("fit_split") == "train"
        and baseline_json.get("eval_split") == "validation"
        and baseline_json.get("cache_manifest_sha256") == next(iter(cache_hashes))
        and baseline_json.get("per_transition_sha256") == sha256_file(baseline_path)
    ):
        raise ValueError("baseline artifact is not the matching held-out validation fit")
    task_ids = reference["task_ids"]
    episode_ids = reference["episode_ids"]
    policies = reference["policies"]
    comparisons = {}

    def compare(name, left, right, selection=None):
        difference = left - right
        selected_tasks, selected_episodes = task_ids, episode_ids
        if selection is not None:
            difference = difference[:, selection]
            selected_tasks = task_ids[selection]
            selected_episodes = episode_ids[selection]
        comparisons[name] = paired_episode_bootstrap(
            difference, selected_tasks, selected_episodes,
            replicates=args.replicates, seed=args.seed,
        )

    compare(
        "history_gain",
        _stack(runs, "no_history", "total_bits"),
        _stack(runs, "full", "total_bits"),
    )
    selected_model = (
        "full" if comparisons["history_gain"]["micro"]["ci95"][0] > 0
        else "no_history"
    )
    if (
        args.expected_final_variant is not None
        and args.expected_final_variant != selected_model
    ):
        raise ValueError(
            f"history CI selects {selected_model}, not "
            f"{args.expected_final_variant}"
        )
    final = _stack(runs, selected_model, "total_bits")
    compare(
        "c1_vs_copy", np.broadcast_to(baseline["copy_total_bits"], final.shape), final
    )
    compare(
        "c1_vs_source", np.broadcast_to(baseline["source_total_bits"], final.shape), final
    )
    marginal = np.broadcast_to(baseline["marginal_total_bits"], final.shape)
    compare("t_only_vs_marginal", marginal, _stack(runs, "t_only", "total_bits"))

    no_action = _stack(runs, "no_action", "total_bits")
    structural = _stack(runs, "structural_action", "total_bits")
    structural_bill = _stack(runs, "structural_action", "action_bill_bits")
    random_selection = policies == POLICY_IDS["random"]
    if not np.any(random_selection):
        raise ValueError("validation set has no random-policy transitions")
    compare("structural_action_gain", no_action, structural)
    compare(
        "structural_action_gain_random_policy",
        no_action, structural, random_selection,
    )
    compare("structural_action_net_gain", no_action, structural + structural_bill)
    compare(
        "structural_action_net_gain_random_policy",
        no_action, structural + structural_bill, random_selection,
    )

    full = _stack(runs, "full", "total_bits")
    full_bill = _stack(runs, "full", "action_bill_bits")
    compare("payload_gain", structural, full)
    compare(
        "payload_net_gain", structural + structural_bill, full + full_bill
    )

    gates = {
        "c1_beats_copy": comparisons["c1_vs_copy"]["micro"]["ci95"][0] > 0,
        "c1_beats_source": comparisons["c1_vs_source"]["micro"]["ci95"][0] > 0,
        "c2_random_structural_action": (
            comparisons["structural_action_gain_random_policy"]["micro"]["ci95"][0]
            > 0
        ),
    }
    report = {
        "protocol": "v8_task_stratified_episode_bootstrap_v1",
        "comparisons": comparisons,
        "gates": gates,
        "all_required_gates_passed": all(gates.values()),
        "history_selection": selected_model,
        "transition_bearing_tasks": int(len(np.unique(task_ids))),
        "random_policy_transitions": int(np.sum(random_selection)),
        "artifacts": {
            "runs": artifacts,
            "formal_run_signature_sha256": formal_signature_sha256,
            "baseline_npz": str(baseline_path.resolve()),
            "baseline_npz_sha256": sha256_file(baseline_path),
            "baseline_json": str(baseline_json_path.resolve()),
            "baseline_json_sha256": sha256_file(baseline_json_path),
        },
        "test_used": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
