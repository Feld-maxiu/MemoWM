"""Closed-loop 1--7 step rollout from real episode initial states and known actions."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from experiments.state_tokenizer.common import sha256_file, write_json

from .cache import FrozenCache
from .config import load_config
from .model import actions_from_batch, codelength_bits, predict
from .train import model_batch_keys, _device_batch, load_checkpoint


def _make_rollout_step(variant, config):
    def step(params, batch):
        actions = actions_from_batch(batch, config.model)
        mask_logits, code_logits = predict(
            params,
            batch["history_codes"], batch["history_valid"], actions,
            batch["task_ids"], variant, config.model,
            history_present=batch["history_present"], train=False,
        )
        rates = codelength_bits(
            mask_logits, code_logits, batch["target_valid"],
            batch["target_codes"],
            code_log_probs=getattr(config.model, "output_smoothing", 0) > 0,
        )
        mask = mask_logits >= 0
        codes = jnp.argmax(code_logits, axis=-1).astype(jnp.uint8)
        return {
            "predicted_valid": mask,
            "predicted_codes": codes,
            "mask_bits": rates["mask_bits"],
            "code_bits": rates["code_bits"],
            "total_bits": rates["total_bits"],
            "mask_accuracy": jnp.mean(
                mask == jnp.asarray(batch["target_valid"], jnp.bool_), axis=-1
            ),
            "code_accuracy": jnp.mean(
                codes == jnp.asarray(batch["target_codes"], jnp.uint8), axis=(-2, -1)
            ),
        }
    return jax.jit(step)


def _replace_prefix(cache, host_batch, predicted):
    histories = cache.transitions["history_indices"][host_batch["transition_indices"]]
    for local in range(len(histories)):
        for time, state_index in enumerate(histories[local]):
            state_index = int(state_index)
            if state_index in predicted:
                codes, valid = predicted[state_index]
                host_batch["history_codes"][local, time] = codes
                host_batch["history_valid"][local, time] = valid


def _baseline_lookup(path: str | None):
    if not path:
        return None
    archive = np.load(path, allow_pickle=False)
    rows = archive["transition_indices"]
    return {
        int(row): {
            "copy": float(archive["copy_total_bits"][index]),
            "source": float(archive["source_total_bits"][index]),
        }
        for index, row in enumerate(rows)
    }


def _summarize(values: dict[str, np.ndarray]) -> dict:
    summary = {}
    cumulative_by_episode = defaultdict(float)
    cumulative_baseline = {"copy": defaultdict(float), "source": defaultdict(float)}
    for horizon in sorted(np.unique(values["horizons"]).tolist()):
        local = np.flatnonzero(values["horizons"] == horizon)
        cumulative = []
        cumulative_copy, cumulative_source = [], []
        for index in local:
            episode = int(values["episode_ids"][index])
            cumulative_by_episode[episode] += float(values["total_bits"][index])
            cumulative.append(cumulative_by_episode[episode])
            if "copy_baseline_bits" in values:
                cumulative_baseline["copy"][episode] += float(
                    values["copy_baseline_bits"][index]
                )
                cumulative_baseline["source"][episode] += float(
                    values["source_baseline_bits"][index]
                )
                cumulative_copy.append(cumulative_baseline["copy"][episode])
                cumulative_source.append(cumulative_baseline["source"][episode])
        entry = {
            "transitions": len(local),
            "step_bits": float(values["total_bits"][local].mean(dtype=np.float64)),
            "cumulative_bits": float(np.mean(cumulative, dtype=np.float64)),
            "mask_accuracy": float(
                values["mask_accuracy"][local].mean(dtype=np.float64)
            ),
            "code_accuracy": float(
                values["code_accuracy"][local].mean(dtype=np.float64)
            ),
        }
        if cumulative_copy:
            entry.update({
                "copy_cumulative_bits": float(np.mean(cumulative_copy, dtype=np.float64)),
                "source_cumulative_bits": float(
                    np.mean(cumulative_source, dtype=np.float64)
                ),
                "gain_vs_copy_cumulative_bits": float(
                    np.mean(cumulative_copy, dtype=np.float64)
                    - np.mean(cumulative, dtype=np.float64)
                ),
                "gain_vs_source_cumulative_bits": float(
                    np.mean(cumulative_source, dtype=np.float64)
                    - np.mean(cumulative, dtype=np.float64)
                ),
            })
        summary[str(horizon)] = entry
    return summary


def _copy_positions(store, indices):
    cache = {}
    rows = []
    for index in indices:
        worker, local = store.locations[int(index)]
        if worker not in cache:
            cache[worker] = np.load(worker / "key64-static-positions.npy", mmap_mode="r")
        rows.append(np.asarray(cache[worker][local], np.float32))
    return np.stack(rows)


def write_reconstruction_stores(args, values, output: Path) -> dict:
    """Decode predicted A2 codes into drop-in raw-PCA stores, one per horizon."""
    from residualmem.encoders.normalization import GroupChannelNormalizer
    from residualmem.world_model import categorical_bottleneck as Q
    from residualmem.world_model import continuous_bottleneck as C
    from experiments.state_tokenizer.a1_continuous_bottleneck import FeatureStore
    from experiments.state_tokenizer.a2_categorical_bottleneck import (
        load_checkpoint as load_a2_checkpoint,
        read_checkpoint_config,
    )

    stored = read_checkpoint_config(args.a2_checkpoint)
    a2_config = Q.CategoricalBottleneckConfig(
        **{**stored, "group_sizes": tuple(stored["group_sizes"])}
    )
    device = jax.devices(args.platform)[args.device_index]
    params = jax.device_put(load_a2_checkpoint(args.a2_checkpoint, a2_config), device)
    normalizer = GroupChannelNormalizer.from_npz(args.normalization)
    source = FeatureStore(args.source_features)
    stores = {}
    for horizon in sorted(np.unique(values["horizons"]).tolist()):
        selected = np.flatnonzero(values["horizons"] == horizon)
        indices = values["target_indices"][selected].astype(np.int64)
        recovered_parts = []
        for start in range(0, len(selected), args.batch_size):
            local = selected[start:start + args.batch_size]
            codes = jnp.asarray(values["predicted_codes"][local], jnp.int32)
            valid = jnp.asarray(values["predicted_valid"][local], jnp.bool_)
            latent = Q.embed_codes(params, codes, a2_config)
            xbar = C.decode(params, latent, valid, a2_config.continuous)
            recovered_parts.append(np.asarray(xbar, np.float32))
        xbar = np.concatenate(recovered_parts)
        valid = values["predicted_valid"][selected].astype(np.bool_)
        recovered = normalizer.denormalize(xbar, valid)
        if np.count_nonzero(recovered[~valid]):
            raise ValueError(f"horizon {horizon} reconstruction leaked into padding")
        root = output / f"recon_h{horizon}"
        worker = root / "worker00"
        worker.mkdir(parents=True, exist_ok=True)
        np.save(worker / "key64-static-pca-bf16.npy",
                recovered.astype(ml_dtypes.bfloat16).view(np.uint16))
        np.save(worker / "key64-static-valid.npy", valid)
        np.save(worker / "record_indices.npy", indices)
        np.save(worker / "subset_rows.npy", indices)
        np.save(worker / "done.npy", np.ones((len(indices),), np.bool_))
        np.save(worker / "key64-static-pca-done.npy", np.ones((len(indices),), np.bool_))
        np.save(worker / "key64-static-positions.npy", _copy_positions(source, indices))
        manifest = {
            "protocol": "v8_wm_rollout_reconstruction_store_v1",
            "horizon": horizon,
            "states": len(indices),
            "global_indices": indices.tolist(),
            "a2_checkpoint_sha256": sha256_file(args.a2_checkpoint),
            "note": "predicted codes/mask decoded to raw PCA; frozen probes only",
        }
        write_json(root / "summary.json", manifest)
        stores[str(horizon)] = str(root.resolve())
    return stores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", default="configs/world_model/v8_discrete.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--test-freeze-manifest")
    parser.add_argument("--baseline-per-transition")
    parser.add_argument("--output", required=True)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--write-reconstruction-stores", action="store_true")
    parser.add_argument("--a2-checkpoint")
    parser.add_argument("--normalization")
    parser.add_argument("--source-features")
    args = parser.parse_args()

    cache = FrozenCache(args.cache)
    config = load_config(args.config, num_tasks=len(cache.task_names))
    checkpoint = load_checkpoint(args.checkpoint)
    variant = checkpoint["metadata"]["variant"]
    device = jax.devices(args.platform)[args.device_index]
    params = jax.device_put(checkpoint["params"], device)
    split_rows = cache.indices_for_split(
        args.split, test_freeze_manifest=args.test_freeze_manifest
    )
    baseline = _baseline_lookup(args.baseline_per_transition)
    rollout_step = _make_rollout_step(variant, config)
    batch_keys = model_batch_keys(config.model)
    predicted = {}
    parts = defaultdict(list)
    for horizon in config.evaluation.rollout_horizons:
        rows = split_rows[cache.transitions["steps"][split_rows] == horizon - 1]
        for start in range(0, len(rows), args.batch_size):
            selected = rows[start:start + args.batch_size]
            host = cache.batch(selected)
            _replace_prefix(cache, host, predicted)
            result = jax.device_get(
                rollout_step(params, _device_batch(host, batch_keys))
            )
            for local, target in enumerate(host["target_indices"]):
                predicted[int(target)] = (
                    np.asarray(result["predicted_codes"][local], np.uint8),
                    np.asarray(result["predicted_valid"][local], np.bool_),
                )
            parts["transition_indices"].append(np.asarray(selected, np.int64))
            parts["target_indices"].append(np.asarray(host["target_indices"], np.int64))
            parts["episode_ids"].append(np.asarray(host["episode_ids"], np.int32))
            parts["task_ids"].append(np.asarray(host["task_ids"], np.uint8))
            parts["horizons"].append(np.full((len(selected),), horizon, np.uint8))
            for name in (
                "predicted_codes", "predicted_valid", "mask_bits", "code_bits",
                "total_bits", "mask_accuracy", "code_accuracy",
            ):
                parts[name].append(np.asarray(result[name]))
            if baseline is not None:
                parts["copy_baseline_bits"].append(np.asarray([
                    baseline[int(row)]["copy"] for row in selected
                ], np.float64))
                parts["source_baseline_bits"].append(np.asarray([
                    baseline[int(row)]["source"] for row in selected
                ], np.float64))
    values = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "rollout.npz"
    np.savez_compressed(archive, **values)
    stores = {}
    if args.write_reconstruction_stores:
        missing = [name for name in ("a2_checkpoint", "normalization", "source_features")
                   if getattr(args, name) is None]
        if missing:
            raise ValueError(f"reconstruction stores require: {missing}")
        stores = write_reconstruction_stores(args, values, output)
    report = {
        "protocol": "v8_wm_closed_loop_rollout_v1",
        "split": args.split,
        "variant": variant,
        "seed": checkpoint["metadata"]["seed"],
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "teacher_forcing": False,
        "known_actions": True,
        "summary_by_horizon": _summarize(values),
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "reconstruction_stores": stores,
    }
    write_json(output / "rollout.json", report)
    print(json.dumps(report["summary_by_horizon"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
