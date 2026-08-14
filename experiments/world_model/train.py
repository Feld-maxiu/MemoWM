"""Train and evaluate one frozen-v8 discrete WM variant with pure JAX/Optax."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from experiments.state_tokenizer.common import sha256_file, write_json

from .cache import CACHE_FILES, FrozenCache
from .config import ExperimentConfig, TrainingConfig, load_config, resolved_dict
from .model import (
    ModelConfig,
    initialize_params,
    loss_and_metrics,
    parameter_count,
    parameter_shapes,
)
from .schema import DEV_VARIANTS, POLICY_NAMES, PROTOCOL, validate_variant


MODEL_BATCH_KEYS = (
    "history_codes", "history_valid", "history_present",
    "action_types", "action_tags", "action_refs", "action_payloads",
    "action_lengths", "task_ids", "target_codes", "target_valid",
)


class DeterministicSampler:
    """Permutation sampler whose complete state is checkpointed."""

    def __init__(self, rows: np.ndarray, batch_size: int, seed: int):
        self.rows = np.asarray(rows, np.int64)
        if not len(self.rows):
            raise ValueError("training rows are empty")
        self.batch_size = int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.rows)
        self.cursor = 0
        self.epochs = 0

    def next(self) -> np.ndarray:
        chunks = []
        needed = self.batch_size
        while needed:
            available = len(self.order) - self.cursor
            take = min(needed, available)
            chunks.append(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            needed -= take
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(self.rows)
                self.cursor = 0
                self.epochs += 1
        return np.concatenate(chunks)

    def state_dict(self) -> dict:
        return {
            "rows": self.rows,
            "batch_size": self.batch_size,
            "rng_state": self.rng.bit_generator.state,
            "order": self.order,
            "cursor": self.cursor,
            "epochs": self.epochs,
        }

    @classmethod
    def from_state(cls, state: dict) -> "DeterministicSampler":
        value = cls(state["rows"], int(state["batch_size"]), 0)
        value.rng.bit_generator.state = state["rng_state"]
        value.order = np.asarray(state["order"], np.int64)
        value.cursor = int(state["cursor"])
        value.epochs = int(state["epochs"])
        return value


def _atomic_pickle(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _save_checkpoint(
    path: Path,
    *,
    params,
    opt_state,
    key,
    sampler: DeterministicSampler,
    step: int,
    best_metric: float,
    best_step: int,
    evals_without_improvement: int,
    metadata: dict,
) -> None:
    _atomic_pickle(path, {
        "protocol": PROTOCOL,
        "params": jax.device_get(params),
        "opt_state": jax.device_get(opt_state),
        "key": np.asarray(jax.device_get(key)),
        "sampler": sampler.state_dict(),
        "step": int(step),
        "best_metric": float(best_metric),
        "best_step": int(best_step),
        "evals_without_improvement": int(evals_without_improvement),
        "metadata": metadata,
    })


def load_checkpoint(path: str | Path, expected_metadata: dict | None = None) -> dict:
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    if value.get("protocol") != PROTOCOL:
        raise ValueError(f"unsupported checkpoint protocol {value.get('protocol')!r}")
    if expected_metadata is not None:
        for key in ("variant", "seed", "cache_manifest_sha256", "config_sha256"):
            if value["metadata"].get(key) != expected_metadata.get(key):
                raise ValueError(
                    f"checkpoint {key} mismatch: {value['metadata'].get(key)!r} != "
                    f"{expected_metadata.get(key)!r}"
                )
    return value


def _decay_mask(params):
    # AdamW excludes every bias, normalization scalar and embedding/mask token.
    return {
        name: bool(name.endswith("_w") or name.endswith("/w"))
        for name in params
    }


def build_optimizer(params, training: TrainingConfig, *, overfit: bool = False):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=training.learning_rate,
        warmup_steps=training.warmup_steps,
        decay_steps=training.max_steps,
        end_value=0.0,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(training.gradient_clip),
        optax.adamw(
            learning_rate=schedule,
            b1=training.adam_beta1,
            b2=training.adam_beta2,
            eps=training.adam_epsilon,
            weight_decay=0.0 if overfit else training.weight_decay,
            mask=_decay_mask(params),
        ),
    )
    return optimizer, schedule


def _device_batch(batch: dict) -> dict:
    return {name: jnp.asarray(batch[name]) for name in MODEL_BATCH_KEYS}


def make_update(optimizer, variant: str, config: ModelConfig, *, overfit: bool):
    def update(params, opt_state, batch, key):
        def objective(current):
            loss, (metrics, _rates, _mask_logits, _code_logits) = loss_and_metrics(
                current, batch, variant, config, rng=key, train=not overfit
            )
            compact = {
                "loss": metrics["loss"],
                "mask_bits": metrics["mask_bits"],
                "code_bits": metrics["code_bits"],
                "mask_accuracy": metrics["mask_accuracy"],
                "code_accuracy": metrics["code_accuracy"],
            }
            return loss, compact

        (loss, metrics), grads = jax.value_and_grad(objective, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        grad_norm = optax.global_norm(grads)
        metrics = {**metrics, "grad_norm_before_clip": grad_norm}
        return params, opt_state, loss, metrics

    return jax.jit(update)


def make_eval(variant: str, config: ModelConfig):
    def evaluate(params, batch):
        _loss, (metrics, rates, _mask_logits, _code_logits) = loss_and_metrics(
            params, batch, variant, config, rng=jax.random.PRNGKey(0), train=False
        )
        return {
            "mask_bits": rates["mask_bits"],
            "code_bits": rates["code_bits"],
            "total_bits": rates["total_bits"],
            "mask_accuracy": metrics["mask_accuracy"],
            "code_accuracy": metrics["code_accuracy"],
        }

    return jax.jit(evaluate)


def evaluate_model(
    cache: FrozenCache,
    rows: np.ndarray,
    params,
    variant: str,
    config: ExperimentConfig,
    *,
    keep_per_transition: bool = False,
) -> tuple[dict, dict[str, np.ndarray] | None]:
    evaluator = make_eval(variant, config.model)
    totals = defaultdict(float)
    transition_parts = defaultdict(list)
    task_totals: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    policy_totals: dict[int, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))
    start_time = time.time()
    for start in range(0, len(rows), config.evaluation.batch_size):
        selected = rows[start:start + config.evaluation.batch_size]
        host_batch = cache.batch(selected)
        result = jax.device_get(evaluator(params, _device_batch(host_batch)))
        mask_bits = np.asarray(result["mask_bits"], np.float64)
        code_bits = np.asarray(result["code_bits"], np.float64)
        total_bits = np.asarray(result["total_bits"], np.float64)
        count = len(selected)
        totals["count"] += count
        totals["mask_bits"] += mask_bits.sum(dtype=np.float64)
        totals["code_bits"] += code_bits.sum(dtype=np.float64)
        totals["total_bits"] += total_bits.sum(dtype=np.float64)
        # Device accuracies are batch means; weight them by their exact element counts.
        totals["mask_correct"] += float(result["mask_accuracy"]) * count
        totals["code_correct"] += float(result["code_accuracy"]) * count
        for local in range(count):
            task = int(host_batch["task_ids"][local])
            policy = int(host_batch["policies"][local])
            for bucket in (task_totals[task], policy_totals[policy]):
                bucket["count"] += 1
                bucket["mask_bits"] += mask_bits[local]
                bucket["code_bits"] += code_bits[local]
                bucket["total_bits"] += total_bits[local]
        if keep_per_transition:
            transition_parts["transition_indices"].append(np.asarray(selected, np.int64))
            for name, values in (
                ("mask_bits", mask_bits), ("code_bits", code_bits),
                ("total_bits", total_bits),
            ):
                transition_parts[name].append(values)
            for name in (
                "target_indices", "episode_ids", "task_ids", "policies",
                "structural_action_bits", "full_action_bits", "steps",
            ):
                transition_parts[name].append(np.asarray(host_batch[name]))

    count = max(totals["count"], 1.0)

    def summarize(bucket):
        denominator = max(bucket["count"], 1.0)
        return {
            "transitions": int(bucket["count"]),
            "mask_bits_per_transition": float(bucket["mask_bits"] / denominator),
            "code_bits_per_transition": float(bucket["code_bits"] / denominator),
            "total_bits_per_transition": float(bucket["total_bits"] / denominator),
        }

    summary = {
        "transitions": int(totals["count"]),
        "mask_bits_per_transition": float(totals["mask_bits"] / count),
        "code_bits_per_transition": float(totals["code_bits"] / count),
        "total_bits_per_transition": float(totals["total_bits"] / count),
        "mask_accuracy": float(totals["mask_correct"] / count),
        "code_accuracy": float(totals["code_correct"] / count),
        "elapsed_seconds": time.time() - start_time,
        "by_task": {
            cache.task_names[task]: summarize(bucket)
            for task, bucket in sorted(task_totals.items())
        },
        "by_policy": {
            POLICY_NAMES[policy]: summarize(bucket)
            for policy, bucket in sorted(policy_totals.items())
        },
    }
    per_transition = None
    if keep_per_transition:
        per_transition = {
            name: np.concatenate(parts) for name, parts in transition_parts.items()
        }
        if variant in ("full", "no_history"):
            bill = per_transition["full_action_bits"].astype(np.float64)
        elif variant == "structural_action":
            bill = per_transition["structural_action_bits"].astype(np.float64)
        else:
            bill = np.zeros_like(per_transition["total_bits"])
        episode_ids = per_transition["episode_ids"]
        task_bill = np.zeros_like(per_transition["total_bits"], dtype=np.float64)
        _, first_indices = np.unique(episode_ids, return_index=True)
        task_bill[first_indices] = 4.0
        per_transition["action_bill_bits"] = bill
        per_transition["task_id_bill_bits"] = task_bill
        per_transition["action_billed_total_bits"] = (
            per_transition["total_bits"] + bill
        )
        per_transition["total_episodic_bits"] = (
            per_transition["action_billed_total_bits"] + task_bill
        )
        summary["action_bill_bits_per_transition"] = float(bill.mean(dtype=np.float64))
        summary["action_billed_total_bits_per_transition"] = float(
            per_transition["action_billed_total_bits"].mean(dtype=np.float64)
        )
        summary["episodes"] = int(len(first_indices))
        summary["task_id_bill_bits_per_transition"] = float(
            task_bill.mean(dtype=np.float64)
        )
        summary["total_episodic_bits_per_transition"] = float(
            per_transition["total_episodic_bits"].mean(dtype=np.float64)
        )
    return summary, per_transition


def _task_balanced_overfit_rows(cache: FrozenCache, count: int, seed: int) -> np.ndarray:
    train = cache.indices_for_split("train")
    tasks = cache.transitions["task_ids"][train]
    rng = np.random.default_rng(seed)
    by_task = {task: rng.permutation(train[tasks == task]).tolist()
               for task in sorted(set(tasks.tolist()))}
    selected = []
    while len(selected) < count:
        progressed = False
        for task in sorted(by_task):
            if by_task[task]:
                selected.append(by_task[task].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"cannot select {count} overfit transitions")
    return np.asarray(selected, np.int64)


def _training_rows(cache: FrozenCache, args) -> np.ndarray:
    if args.overfit_transitions:
        return _task_balanced_overfit_rows(cache, args.overfit_transitions, args.seed)
    if args.subset_archive:
        archive = np.load(args.subset_archive, allow_pickle=False)
        if not args.subset_name or args.subset_name not in archive.files:
            raise ValueError(
                f"--subset-name must select one of {archive.files}"
            )
        rows = np.asarray(archive[args.subset_name], np.int64)
        formal_train = set(cache.indices_for_split("train").tolist())
        if any(int(row) not in formal_train for row in rows):
            raise ValueError("subset contains a non-train transition")
        return rows
    return cache.indices_for_split("train")


def _apply_training_overrides(training: TrainingConfig, args) -> TrainingConfig:
    values = {}
    for field, argument in (
        ("batch_size", args.batch_size),
        ("learning_rate", args.learning_rate),
        ("warmup_steps", args.warmup_steps),
        ("max_steps", args.max_steps),
        ("min_steps", args.min_steps),
        ("eval_every", args.eval_every),
        ("patience_steps", args.patience_steps),
        ("weight_decay", args.weight_decay),
        ("gradient_clip", args.gradient_clip),
    ):
        if argument is not None:
            values[field] = argument
    if args.overfit_transitions:
        values.setdefault("min_steps", values.get("max_steps", training.max_steps))
        values.setdefault("patience_steps", values.get("max_steps", training.max_steps))
        # Keep validation cadence legal for tiny sanity invocations.
        candidate_eval = values.get("eval_every", training.eval_every)
        candidate_patience = values["patience_steps"]
        if candidate_patience % candidate_eval:
            values["patience_steps"] = candidate_eval * math.ceil(
                candidate_patience / candidate_eval
            )
    return dataclasses.replace(training, **values)


def _write_per_episode(path: Path, cache: FrozenCache, values: dict[str, np.ndarray]):
    grouped: dict[int, list[int]] = defaultdict(list)
    for local, episode in enumerate(values["episode_ids"]):
        grouped[int(episode)].append(local)
    with path.open("w", encoding="utf-8") as handle:
        for episode in sorted(grouped):
            local = np.asarray(grouped[episode], np.int64)
            task = int(values["task_ids"][local[0]])
            item = {
                "episode_id": cache.episode_names[episode],
                "task": cache.task_names[task],
                "transitions": len(local),
                "observation_bits": float(
                    values["total_bits"][local].sum(dtype=np.float64)
                ),
                "action_bill_bits": float(
                    values["action_bill_bits"][local].sum(dtype=np.float64)
                ),
                "action_billed_total_bits": float(
                    values["action_billed_total_bits"][local].sum(dtype=np.float64)
                ),
                "task_id_bill_bits": 4.0,
                "total_episodic_bits": float(
                    values["total_episodic_bits"][local].sum(dtype=np.float64)
                ),
            }
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> dict:
    allow_dev = bool(getattr(args, "allow_dev_variant", False))
    variant = validate_variant(args.variant, allow_dev=allow_dev)
    if variant in DEV_VARIANTS and not allow_dev:
        raise ValueError(
            f"{variant!r} is a diagnostic variant; pass --allow-dev-variant"
        )
    cache = FrozenCache(args.cache, verify_hashes=args.verify_cache_hashes)
    config = load_config(args.config, num_tasks=len(cache.task_names))
    training = _apply_training_overrides(config.training, args)
    config = dataclasses.replace(config, training=training)
    jax.config.update("jax_default_matmul_precision", training.matmul_precision)
    if args.platform:
        devices = jax.devices(args.platform)
    else:
        devices = jax.devices()
    device = devices[args.device_index]

    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"run directory {output} is nonempty; use --resume")
    output.mkdir(parents=True, exist_ok=True)
    resolved = resolved_dict(
        config, variant=variant, seed=args.seed, allow_dev=allow_dev
    )
    resolved_path = output / "resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved, sort_keys=True), encoding="utf-8")
    cache_manifest_path = Path(args.cache) / CACHE_FILES["manifest"]
    metadata = {
        "variant": variant,
        "seed": args.seed,
        "cache_manifest_sha256": sha256_file(cache_manifest_path),
        "config_sha256": sha256_file(resolved_path),
    }

    train_rows = _training_rows(cache, args)
    overfit = bool(args.overfit_transitions)
    selection_rows = train_rows if overfit else cache.indices_for_split(
        "validation", test_freeze_manifest=None
    )
    params = jax.device_put(initialize_params(config.model, args.seed), device)
    optimizer, schedule = build_optimizer(params, training, overfit=overfit)
    opt_state = optimizer.init(params)
    sampler = DeterministicSampler(train_rows, training.batch_size, args.seed)
    key = jax.random.PRNGKey(args.seed + 1)
    step = 0
    best_metric = float("inf")
    best_step = 0
    evals_without_improvement = 0
    last_path, best_path = output / "last.pkl", output / "best.pkl"
    if args.resume:
        checkpoint = load_checkpoint(last_path, metadata)
        params = jax.device_put(checkpoint["params"], device)
        opt_state = jax.device_put(checkpoint["opt_state"], device)
        key = jax.device_put(checkpoint["key"], device)
        sampler = DeterministicSampler.from_state(checkpoint["sampler"])
        step = int(checkpoint["step"])
        best_metric = float(checkpoint["best_metric"])
        best_step = int(checkpoint["best_step"])
        evals_without_improvement = int(checkpoint["evals_without_improvement"])

    update = make_update(optimizer, variant, config.model, overfit=overfit)
    initial, _ = evaluate_model(
        cache, selection_rows, params, variant, config, keep_per_transition=False
    )
    if not args.resume:
        best_metric = initial["total_bits_per_transition"]
        _save_checkpoint(
            best_path, params=params, opt_state=opt_state, key=key,
            sampler=sampler, step=step, best_metric=best_metric,
            best_step=best_step,
            evals_without_improvement=evals_without_improvement,
            metadata=metadata,
        )
    history_path = output / "metrics.jsonl"
    start_time = time.time()
    stop_reason = "max_steps"
    with history_path.open("a", encoding="utf-8") as history:
        while step < training.max_steps:
            selected = sampler.next()
            batch = _device_batch(cache.batch(selected))
            key, update_key = jax.random.split(key)
            params, opt_state, _loss, train_metrics = update(
                params, opt_state, batch, update_key
            )
            step += 1
            if step % training.eval_every and step != training.max_steps:
                continue
            validation, _ = evaluate_model(
                cache, selection_rows, params, variant, config,
                keep_per_transition=False,
            )
            metric = validation["total_bits_per_transition"]
            record = {
                "step": step,
                "learning_rate": float(schedule(step)),
                "train_batch": {name: float(value) for name, value in train_metrics.items()},
                "selection": validation,
                "wall_seconds": time.time() - start_time,
            }
            history.write(json.dumps(record, sort_keys=True) + "\n")
            history.flush()
            improved = metric < best_metric
            if improved:
                best_metric, best_step = metric, step
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
            _save_checkpoint(
                last_path, params=params, opt_state=opt_state, key=key,
                sampler=sampler, step=step, best_metric=best_metric,
                best_step=best_step,
                evals_without_improvement=evals_without_improvement,
                metadata=metadata,
            )
            if improved:
                _save_checkpoint(
                    best_path, params=params, opt_state=opt_state, key=key,
                    sampler=sampler, step=step, best_metric=best_metric,
                    best_step=best_step,
                    evals_without_improvement=evals_without_improvement,
                    metadata=metadata,
                )
            if (
                not overfit and step >= training.min_steps
                and evals_without_improvement >= training.patience_evals
            ):
                stop_reason = "validation_patience"
                break

    if not best_path.exists():
        _save_checkpoint(
            best_path, params=params, opt_state=opt_state, key=key,
            sampler=sampler, step=step, best_metric=best_metric,
            best_step=best_step, evals_without_improvement=evals_without_improvement,
            metadata=metadata,
        )
    best = load_checkpoint(best_path, metadata)
    best_params = jax.device_put(best["params"], device)
    final, per_transition = evaluate_model(
        cache, selection_rows, best_params, variant, config,
        keep_per_transition=True,
    )
    transition_path = output / "per_transition.npz"
    np.savez_compressed(transition_path, **per_transition)
    _write_per_episode(output / "per_episode.jsonl", cache, per_transition)

    relative_drop = 1.0 - final["total_bits_per_transition"] / max(
        initial["total_bits_per_transition"], 1e-30
    )
    gates = {}
    if overfit:
        gates["overfit_nll_drop_90pct"] = relative_drop >= 0.90
    if args.baseline_json:
        baseline = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        baseline_rates = baseline["bits_per_transition"]
        gates["beats_copy_point"] = (
            final["total_bits_per_transition"]
            < baseline_rates["copy"]["total_bits_per_transition"]
        )
        gates["beats_source_point"] = (
            final["total_bits_per_transition"]
            < baseline_rates["source"]["total_bits_per_transition"]
        )
    result = {
        "protocol": PROTOCOL,
        "variant": variant,
        "seed": args.seed,
        "device": str(device),
        "parameter_count": parameter_count(best_params),
        "parameter_shapes": {
            name: list(shape) for name, shape in parameter_shapes(best_params).items()
        },
        "data": {
            "cache": str(Path(args.cache).resolve()),
            "cache_manifest_sha256": metadata["cache_manifest_sha256"],
            "train_transitions": len(train_rows),
            "selection": "overfit_train" if overfit else "validation",
            "selection_transitions": len(selection_rows),
            "subset": args.subset_name,
            "test_evaluated": False,
            # Diagnostic runs are permanently disqualified from formal
            # statistics; statistics._validate_run_artifact rejects this flag.
            "dev_run": allow_dev,
        },
        "training": {
            **dataclasses.asdict(training),
            "steps_ran": step,
            "best_step": best_step,
            "stop_reason": stop_reason,
            "wall_seconds": time.time() - start_time,
        },
        "initial": initial,
        "best_selection": final,
        "relative_nll_drop": relative_drop,
        "gates": gates,
        "all_gates_passed": all(gates.values()) if gates else None,
        "artifacts": {
            "resolved_config": "resolved.yaml",
            "resolved_config_sha256": metadata["config_sha256"],
            "best_checkpoint": "best.pkl",
            "last_checkpoint": "last.pkl",
            "metrics": "metrics.jsonl",
            "per_transition": "per_transition.npz",
            "per_transition_sha256": sha256_file(transition_path),
            "per_episode": "per_episode.jsonl",
        },
        "numerics": {
            "activations": "float32",
            "matmul_precision": training.matmul_precision,
            "metric_accumulation": "float64_host",
            "eval_batch_size": config.evaluation.batch_size,
        },
        "rate_scope": {
            "primary": "held_out_ideal_nll_bits_per_transition",
            "all_2048_codes_charged": True,
            "invalid_observation_slots_remove_code_loss": False,
            "actual_entropy_coder": False,
            "task_id_bits_per_episode": 4,
        },
    }
    write_json(output / "run.json", result)
    if overfit and not gates["overfit_nll_drop_90pct"]:
        raise RuntimeError(
            f"overfit gate failed: NLL dropped {relative_drop:.3%}, required 90%"
        )
    if args.enforce_c1_point_gate and not (
        gates.get("beats_copy_point") and gates.get("beats_source_point")
    ):
        raise RuntimeError("M1 point gate failed against copy/source baseline")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", default="configs/world_model/v8_discrete.yaml")
    parser.add_argument("--variant", choices=(
        "t_only", "no_action", "structural_action", "no_history", "full",
        "state_only", "struct_no_history",
    ), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--subset-archive")
    parser.add_argument("--subset-name")
    parser.add_argument("--overfit-transitions", type=int)
    parser.add_argument("--baseline-json")
    parser.add_argument("--enforce-c1-point-gate", action="store_true")
    parser.add_argument("--verify-cache-hashes", action="store_true")
    parser.add_argument("--resume", action="store_true")
    # Diagnostic escape hatch. Unlocks DEV_VARIANTS and stamps the run as a dev
    # run so statistics.py can never absorb it into a formal comparison.
    parser.add_argument("--allow-dev-variant", action="store_true")
    # Explicit overrides are useful for M0/M1 smoke tests; formal runs leave them unset.
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--min-steps", type=int)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--patience-steps", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--gradient-clip", type=float)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps({
        "variant": result["variant"],
        "seed": result["seed"],
        "best_selection": result["best_selection"],
        "gates": result["gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
