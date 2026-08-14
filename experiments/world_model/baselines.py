"""Strict train-fit/held-out discrete transition baselines for frozen v8 codes."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.common import sha256_file, write_json
from experiments.state_tokenizer.slot_layout import GROUP_NAMES, KEY64_LAYOUT

from .cache import CACHE_FILES, FrozenCache
from .schema import NUM_CATEGORIES, NUM_LATENT_TOKENS, NUM_SUBSPACES, POLICY_NAMES


ALPHA = 0.5
SOURCE_PRIOR_CONCENTRATION = NUM_CATEGORIES * ALPHA
POSITIONS = NUM_LATENT_TOKENS * NUM_SUBSPACES


def _lookup(sorted_keys: np.ndarray, counts: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_keys, wanted)
    found = positions < len(sorted_keys)
    safe = np.minimum(positions, max(len(sorted_keys) - 1, 0))
    if len(sorted_keys):
        found &= sorted_keys[safe] == wanted
    output = np.zeros(wanted.shape, np.float64)
    if len(sorted_keys):
        output[found] = counts[safe[found]]
    return output


def _binary_nll(probability_one: np.ndarray, targets: np.ndarray) -> np.ndarray:
    probability = np.where(targets, probability_one, 1.0 - probability_one)
    return -np.log2(np.maximum(probability, np.finfo(np.float64).tiny))


def _task_rates(
    fit_source: np.ndarray,
    fit_target: np.ndarray,
    fit_source_mask: np.ndarray,
    fit_target_mask: np.ndarray,
    eval_source: np.ndarray,
    eval_target: np.ndarray,
    eval_source_mask: np.ndarray,
    eval_target_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit one task's tables and return per-evaluation-transition codelengths."""
    fit_source = fit_source.reshape(len(fit_source), POSITIONS)
    fit_target = fit_target.reshape(len(fit_target), POSITIONS)
    eval_source = eval_source.reshape(len(eval_source), POSITIONS)
    eval_target = eval_target.reshape(len(eval_target), POSITIONS)
    n_fit, n_eval = len(fit_source), len(eval_source)
    if n_fit < 1 or n_eval < 1:
        raise ValueError("each evaluated task needs nonempty fit and evaluation rows")

    offsets = np.arange(POSITIONS, dtype=np.int64) * NUM_CATEGORIES
    target_keys = (fit_target.astype(np.int64) + offsets).ravel()
    marginal_counts = np.bincount(
        target_keys, minlength=POSITIONS * NUM_CATEGORIES
    ).reshape(POSITIONS, NUM_CATEGORIES)
    eval_positions = np.arange(POSITIONS)[None, :]
    marginal_probability = (
        marginal_counts[eval_positions, eval_target] + ALPHA
    ) / (n_fit + NUM_CATEGORIES * ALPHA)
    marginal_code = -np.log2(marginal_probability).sum(axis=1, dtype=np.float64)

    fit_same = fit_source == fit_target
    keep_count = fit_same.sum(axis=0, dtype=np.int64)
    change_count = n_fit - keep_count
    changed_keys = (fit_target.astype(np.int64) + offsets)[~fit_same]
    changed_destination = np.bincount(
        changed_keys, minlength=POSITIONS * NUM_CATEGORIES
    ).reshape(POSITIONS, NUM_CATEGORIES)
    p_keep = (keep_count + ALPHA) / (n_fit + 2 * ALPHA)
    eval_same = eval_source == eval_target
    destination_numerator = changed_destination[eval_positions, eval_target] + ALPHA
    destination_denominator = (
        change_count[None, :]
        + NUM_CATEGORIES * ALPHA
        - (changed_destination[eval_positions, eval_source] + ALPHA)
    )
    destination_probability = destination_numerator / destination_denominator
    copy_probability = np.where(
        eval_same,
        p_keep[None, :],
        (1.0 - p_keep[None, :]) * destination_probability,
    )
    copy_code = -np.log2(copy_probability).sum(axis=1, dtype=np.float64)

    # Sparse source-conditioned counts avoid a >6 GiB dense table.
    position_pair_offsets = np.arange(POSITIONS, dtype=np.int64) * (
        NUM_CATEGORIES * NUM_CATEGORIES
    )
    fit_pair_keys = (
        position_pair_offsets
        + fit_source.astype(np.int64) * NUM_CATEGORIES
        + fit_target.astype(np.int64)
    ).ravel()
    pair_keys, pair_counts = np.unique(fit_pair_keys, return_counts=True)
    fit_source_keys = (offsets + fit_source.astype(np.int64)).ravel()
    source_keys, source_counts = np.unique(fit_source_keys, return_counts=True)
    eval_pair_keys = (
        position_pair_offsets
        + eval_source.astype(np.int64) * NUM_CATEGORIES
        + eval_target.astype(np.int64)
    )
    eval_source_keys = offsets + eval_source.astype(np.int64)
    pair_n = _lookup(pair_keys, pair_counts, eval_pair_keys)
    source_n = _lookup(source_keys, source_counts, eval_source_keys)
    source_probability = (
        pair_n + SOURCE_PRIOR_CONCENTRATION * copy_probability
    ) / (source_n + SOURCE_PRIOR_CONCENTRATION)
    source_code = -np.log2(source_probability).sum(axis=1, dtype=np.float64)

    slots = fit_target_mask.shape[1]
    slot_offsets = np.arange(slots, dtype=np.int64) * 2
    mask_counts = np.bincount(
        (fit_target_mask.astype(np.int64) + slot_offsets).ravel(),
        minlength=slots * 2,
    ).reshape(slots, 2)
    marginal_mask_probability_one = (mask_counts[:, 1] + ALPHA) / (n_fit + 2 * ALPHA)
    marginal_mask_matrix = _binary_nll(
        marginal_mask_probability_one[None, :], eval_target_mask
    )

    mask_pair_offsets = np.arange(slots, dtype=np.int64) * 4
    mask_pair_counts = np.bincount(
        (
            mask_pair_offsets
            + fit_source_mask.astype(np.int64) * 2
            + fit_target_mask.astype(np.int64)
        ).ravel(),
        minlength=slots * 4,
    ).reshape(slots, 2, 2)
    source_mask_counts = mask_pair_counts.sum(axis=-1)
    p_mask_one = (mask_pair_counts[:, :, 1] + ALPHA) / (
        source_mask_counts + 2 * ALPHA
    )
    copy_mask_matrix = _binary_nll(
        p_mask_one[np.arange(slots)[None, :], eval_source_mask.astype(np.int64)],
        eval_target_mask,
    )
    return {
        "marginal_code_bits": marginal_code,
        "copy_code_bits": copy_code,
        "source_code_bits": source_code,
        "marginal_mask_matrix": marginal_mask_matrix,
        "copy_mask_matrix": copy_mask_matrix,
    }


def evaluate_baselines(
    cache: FrozenCache,
    fit_rows: np.ndarray,
    eval_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    fit_rows = np.asarray(fit_rows, np.int64)
    eval_rows = np.asarray(eval_rows, np.int64)
    fit_source_indices = cache.transitions["history_indices"][fit_rows, -1]
    fit_target_indices = cache.transitions["target_indices"][fit_rows]
    eval_source_indices = cache.transitions["history_indices"][eval_rows, -1]
    eval_target_indices = cache.transitions["target_indices"][eval_rows]
    if np.any(fit_source_indices < 0) or np.any(eval_source_indices < 0):
        raise ValueError("a transition's current source state cannot be padding")

    output = {
        name: np.zeros((len(eval_rows),), np.float64)
        for name in (
            "marginal_code_bits", "copy_code_bits", "source_code_bits",
            "marginal_mask_bits", "copy_mask_bits",
        )
    }
    output["marginal_mask_group_bits"] = np.zeros(
        (len(eval_rows), len(KEY64_LAYOUT)), np.float64
    )
    output["copy_mask_group_bits"] = np.zeros_like(output["marginal_mask_group_bits"])

    fit_tasks = cache.transitions["task_ids"][fit_rows]
    eval_tasks = cache.transitions["task_ids"][eval_rows]
    for task in sorted(set(eval_tasks.tolist())):
        fit_local = np.flatnonzero(fit_tasks == task)
        eval_local = np.flatnonzero(eval_tasks == task)
        if not len(fit_local):
            raise ValueError(f"fit split has no transitions for task {task}")
        rates = _task_rates(
            np.asarray(cache.codes[fit_source_indices[fit_local]], np.uint8),
            np.asarray(cache.codes[fit_target_indices[fit_local]], np.uint8),
            np.asarray(cache.valid[fit_source_indices[fit_local]], np.bool_),
            np.asarray(cache.valid[fit_target_indices[fit_local]], np.bool_),
            np.asarray(cache.codes[eval_source_indices[eval_local]], np.uint8),
            np.asarray(cache.codes[eval_target_indices[eval_local]], np.uint8),
            np.asarray(cache.valid[eval_source_indices[eval_local]], np.bool_),
            np.asarray(cache.valid[eval_target_indices[eval_local]], np.bool_),
        )
        for name in ("marginal_code_bits", "copy_code_bits", "source_code_bits"):
            output[name][eval_local] = rates[name]
        for prefix in ("marginal", "copy"):
            matrix = rates[f"{prefix}_mask_matrix"]
            output[f"{prefix}_mask_bits"][eval_local] = matrix.sum(
                axis=1, dtype=np.float64
            )
            start = 0
            for group, size in enumerate(KEY64_LAYOUT):
                stop = start + size
                output[f"{prefix}_mask_group_bits"][eval_local, group] = matrix[
                    :, start:stop
                ].sum(axis=1, dtype=np.float64)
                start = stop
    for prefix in ("marginal", "copy"):
        output[f"{prefix}_total_bits"] = (
            output[f"{prefix}_mask_bits"] + output[f"{prefix}_code_bits"]
        )
    output["source_mask_bits"] = output["copy_mask_bits"].copy()
    output["source_total_bits"] = output["source_mask_bits"] + output["source_code_bits"]
    return output


def _mean(values: np.ndarray, selection: np.ndarray | None = None) -> float:
    if selection is not None:
        values = values[selection]
    return float(np.mean(values, dtype=np.float64)) if len(values) else float("nan")


def _summary(cache: FrozenCache, rows: np.ndarray, rates: dict[str, np.ndarray]) -> dict:
    policies = cache.transitions["action_policies"][rows, -1]
    report = {}
    for name in ("marginal", "copy", "source"):
        entry = {
            "mask_bits_per_transition": _mean(rates[f"{name}_mask_bits"]),
            "code_bits_per_transition": _mean(rates[f"{name}_code_bits"]),
            "total_bits_per_transition": _mean(rates[f"{name}_total_bits"]),
        }
        entry["by_policy"] = {
            policy: {
                "transitions": int(np.sum(policies == policy_id)),
                "total_bits_per_transition": _mean(
                    rates[f"{name}_total_bits"], policies == policy_id
                ),
            }
            for policy_id, policy in enumerate(POLICY_NAMES)
        }
        report[name] = entry
    for name in ("marginal", "copy"):
        report[name]["mask_bits_by_observation_group"] = {
            group: _mean(rates[f"{name}_mask_group_bits"][:, index])
            for index, group in enumerate(GROUP_NAMES)
        }
    return report


def _change_diagnostics(cache: FrozenCache, rows: np.ndarray, seed: int) -> dict:
    source_indices = cache.transitions["history_indices"][rows, -1]
    target_indices = cache.transitions["target_indices"][rows]
    source_codes = np.asarray(cache.codes[source_indices], np.uint8)
    target_codes = np.asarray(cache.codes[target_indices], np.uint8)
    source_mask = np.asarray(cache.valid[source_indices], np.bool_)
    target_mask = np.asarray(cache.valid[target_indices], np.bool_)
    tasks = cache.transitions["task_ids"][rows]
    rng = np.random.default_rng(seed)
    same_task_permutation = np.empty((len(rows),), np.int64)
    for task in sorted(set(tasks.tolist())):
        members = np.flatnonzero(tasks == task)
        same_task_permutation[members] = rng.permutation(members)
    global_permutation = rng.permutation(len(rows))

    def rates(code_right, mask_right):
        return {
            "code": float(np.mean(source_codes != code_right)),
            "mask": float(np.mean(source_mask != mask_right)),
        }

    consecutive = rates(target_codes, target_mask)
    start = 0
    consecutive["mask_by_observation_group"] = {}
    for name, size in zip(GROUP_NAMES, KEY64_LAYOUT):
        stop = start + size
        consecutive["mask_by_observation_group"][name] = (
            float(np.mean(source_mask[:, start:stop] != target_mask[:, start:stop]))
            if size else 0.0
        )
        start = stop
    return {
        "consecutive": consecutive,
        "same_task_random": rates(
            target_codes[same_task_permutation], target_mask[same_task_permutation]
        ),
        "global_random": rates(
            target_codes[global_permutation], target_mask[global_permutation]
        ),
        "note": (
            "A2 latent code axes are distributed and are not observation groups; "
            "only mask changes are grouped as image/detail/context/prompt."
        ),
    }


def _write_per_episode(
    path: Path, cache: FrozenCache, rows: np.ndarray, rates: dict[str, np.ndarray]
) -> None:
    grouped: dict[int, list[int]] = defaultdict(list)
    for local, episode in enumerate(cache.transitions["episode_ids"][rows]):
        grouped[int(episode)].append(local)
    with path.open("w", encoding="utf-8") as handle:
        for episode in sorted(grouped):
            local = np.asarray(grouped[episode], np.int64)
            task = int(cache.transitions["task_ids"][rows[local[0]]])
            value = {
                "episode_id": cache.episode_names[episode],
                "task": cache.task_names[task],
                "transitions": len(local),
                "task_id_bill_bits": 4.0,
            }
            for name in ("marginal", "copy", "source"):
                observation_bits = float(
                    rates[f"{name}_total_bits"][local].sum(dtype=np.float64)
                )
                value[f"{name}_total_bits"] = observation_bits
                value[f"{name}_episodic_total_bits"] = observation_bits + 4.0
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> dict:
    if args.fit_split == args.eval_split and not args.allow_same_split_debug:
        raise ValueError("fit and evaluation split must differ outside explicit debugging")
    if args.fit_split != "train" and not args.allow_nontrain_fit_debug:
        raise ValueError("formal baselines must fit on train")
    cache = FrozenCache(args.cache, verify_hashes=args.verify_cache_hashes)
    fit_rows = cache.indices_for_split(args.fit_split)
    eval_rows = cache.indices_for_split(
        args.eval_split, test_freeze_manifest=args.test_freeze_manifest
    )
    rates = evaluate_baselines(cache, fit_rows, eval_rows)
    episode_ids = cache.transitions["episode_ids"][eval_rows]
    task_bill = np.zeros((len(eval_rows),), np.float64)
    unique_episodes, first_indices = np.unique(episode_ids, return_index=True)
    task_bill[first_indices] = 4.0
    episodic_rates = {
        f"{name}_episodic_total_bits": rates[f"{name}_total_bits"] + task_bill
        for name in ("marginal", "copy", "source")
    }
    rate_summary = _summary(cache, eval_rows, rates)
    for name in ("marginal", "copy", "source"):
        rate_summary[name]["task_billed_total_bits_per_transition"] = _mean(
            episodic_rates[f"{name}_episodic_total_bits"]
        )
    structural_action_bits = cache.transitions["structural_action_bits"][eval_rows]
    full_action_bits = cache.transitions["full_action_bits"][eval_rows]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    transition_path = output / "per_transition.npz"
    np.savez_compressed(
        transition_path,
        transition_indices=eval_rows,
        task_ids=cache.transitions["task_ids"][eval_rows],
        episode_ids=cache.transitions["episode_ids"][eval_rows],
        policies=cache.transitions["action_policies"][eval_rows, -1],
        structural_action_bits=structural_action_bits,
        full_action_bits=full_action_bits,
        task_id_bill_bits=task_bill,
        **episodic_rates,
        **rates,
    )
    _write_per_episode(output / "per_episode.jsonl", cache, eval_rows, rates)
    report = {
        "protocol": "v8_task_conditioned_baselines_v1",
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "fit_transitions": len(fit_rows),
        "eval_transitions": len(eval_rows),
        "smoothing": {"jeffreys_alpha": ALPHA},
        "source_backoff": {"prior": "copy", "concentration": SOURCE_PRIOR_CONCENTRATION},
        "bits_per_transition": rate_summary,
        "change_rate": _change_diagnostics(cache, eval_rows, args.seed),
        "fixed_width_audit": {"code_bits": 16_384, "mask_bits": 64, "total_bits": 16_448},
        "side_information_audit": {
            "episodes": int(len(unique_episodes)),
            "task": {
                "num_task_ids": len(cache.task_names),
                "fixed_bits_per_episode": 4,
                "amortized_bits_per_transition": _mean(task_bill),
            },
            "action": {
                "used_by_baselines": False,
                "structural_fixed_bits_per_transition": _mean(
                    structural_action_bits
                ),
                "full_fixed_bits_per_transition": _mean(full_action_bits),
                "note": "Action costs are audited but are not part of baseline conditioning.",
            },
        },
        "cache_manifest_sha256": sha256_file(Path(args.cache) / CACHE_FILES["manifest"]),
        "per_transition": transition_path.name,
        "per_transition_sha256": sha256_file(transition_path),
        "per_episode": "per_episode.jsonl",
        "code_grouping_warning": (
            "All 2048 distributed A2 codes are charged. Observation validity never "
            "removes code loss, and code axes are not sliced into semantic groups."
        ),
    }
    write_json(output / "baseline.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fit-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--test-freeze-manifest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-cache-hashes", action="store_true")
    parser.add_argument("--allow-same-split-debug", action="store_true")
    parser.add_argument("--allow-nontrain-fit-debug", action="store_true")
    return parser


def main() -> None:
    report = run(build_parser().parse_args())
    print(json.dumps(report["bits_per_transition"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
