"""A0 no-compression slot reconstruction wiring test on real Static PCA states."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax

from residualmem.encoders.normalization import GroupChannelNormalizer
from residualmem.world_model.slot_reconstruction import (
    SlotReconstructionConfig,
    attention_diagnostics,
    group_metrics,
    initialize_params,
    masked_mse,
    reconstruct,
)

from .common import iter_jsonl, sha256_file, write_json
from .slot_layout import KEY64_LAYOUT


def select_train_rows(records: list[dict], num_states: int, seed: int) -> list[int]:
    if num_states < 1:
        raise ValueError("num-states must be positive")
    rng = np.random.default_rng(seed)
    by_task = defaultdict(list)
    for row, record in enumerate(records):
        if record["split"] == "train":
            by_task[record["task"]].append(row)
    for rows in by_task.values():
        rng.shuffle(rows)
    selected = []
    tasks = sorted(by_task)
    cursor = {task: 0 for task in tasks}
    while len(selected) < num_states:
        progressed = False
        for task in tasks:
            if cursor[task] < len(by_task[task]):
                selected.append(by_task[task][cursor[task]])
                cursor[task] += 1
                progressed = True
                if len(selected) == num_states:
                    break
        if not progressed:
            raise ValueError(f"requested {num_states} states but train split is exhausted")
    return selected


def load_feature_rows(root: str | Path, global_indices: list[int]):
    root = Path(root)
    locations = {}
    for worker in sorted(root.glob("worker*")):
        indices = np.load(worker / "record_indices.npy")
        done = np.load(worker / "done.npy", mmap_mode="r")
        pca_done = np.load(worker / "key64-static-pca-done.npy", mmap_mode="r")
        if not bool(np.asarray(done).all()) or not bool(np.asarray(pca_done).all()):
            raise ValueError(f"incomplete final feature shard: {worker}")
        for local, index in enumerate(indices):
            locations[int(index)] = (worker, local)
    missing = [index for index in global_indices if index not in locations]
    if missing:
        raise KeyError(f"feature store misses global indices: {missing[:20]}")
    x_rows, masks = [], []
    arrays = {}
    for index in global_indices:
        worker, local = locations[index]
        if worker not in arrays:
            arrays[worker] = (
                np.load(worker / "key64-static-pca-bf16.npy", mmap_mode="r"),
                np.load(worker / "key64-static-valid.npy", mmap_mode="r"),
            )
        bits, valid = arrays[worker]
        x_rows.append(
            np.asarray(bits[local], np.uint16).copy().view(ml_dtypes.bfloat16).astype(np.float32)
        )
        masks.append(np.asarray(valid[local], np.bool_).copy())
    return np.stack(x_rows), np.stack(masks)


def tree_l2(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)))


def numpy_metrics(values: dict) -> dict[str, float]:
    return {name: float(np.asarray(value)) for name, value in values.items()}


def evaluate(
    params,
    xbar,
    valid,
    x_t,
    normalizer: GroupChannelNormalizer,
    config: SlotReconstructionConfig,
    value_source: str,
    routing_mode: str,
):
    prediction, attention = reconstruct(
        params, xbar, valid, config,
        value_source=value_source,
        routing_mode=routing_mode,
        return_attention=True,
    )
    metrics = numpy_metrics(group_metrics(prediction, xbar, valid, config))
    metrics.update(numpy_metrics(attention_diagnostics(attention, valid)))
    prediction_np = np.asarray(prediction)
    recovered = normalizer.denormalize(prediction_np, np.asarray(valid))
    difference = recovered[np.asarray(valid)] - x_t[np.asarray(valid)]
    metrics["denormalized/rmse"] = float(np.sqrt(np.mean(difference.astype(np.float64) ** 2)))
    metrics["invalid_output_nonzero"] = int(np.count_nonzero(prediction_np[~np.asarray(valid)]))
    without_slots = reconstruct(
        params, xbar, valid, config,
        slot_embedding_scale=0.0,
        value_source=value_source,
        routing_mode=routing_mode,
    )
    metrics["slot_embedding_zeroed/mse"] = float(
        masked_mse(without_slots, xbar, valid)
    )
    metrics["slot_embedding_zeroed/r2"] = 1.0 - (
        metrics["slot_embedding_zeroed/mse"] / max(metrics["all/zero_mse"], 1e-12)
    )
    return metrics


def run(args: argparse.Namespace) -> dict:
    records = list(iter_jsonl(args.records))
    selected_rows = select_train_rows(records, args.num_states, args.seed)
    global_indices = [int(records[row]["global_index"]) for row in selected_rows]
    x_t, valid = load_feature_rows(args.features, global_indices)
    normalizer = GroupChannelNormalizer.from_npz(args.normalization)
    if (normalizer.num_tokens, normalizer.token_dim) != (64, 512):
        raise ValueError("A0 requires a 64×512 normalizer")
    xbar = normalizer.normalize(x_t, valid)
    if not np.isfinite(xbar).all() or np.count_nonzero(xbar[~valid]):
        raise ValueError("normalization produced invalid or nonzero padding values")

    config = SlotReconstructionConfig(
        num_slots=64,
        token_dim=512,
        num_heads=args.num_heads,
        ffn_hidden=args.ffn_hidden,
        group_sizes=tuple(KEY64_LAYOUT),
    )
    devices = jax.devices(args.platform)
    if not 0 <= args.device_index < len(devices):
        raise ValueError(f"device-index {args.device_index} unavailable for {args.platform}")
    device = devices[args.device_index]
    xbar_j = jax.device_put(jnp.asarray(xbar), device)
    valid_j = jax.device_put(jnp.asarray(valid), device)
    if args.init_checkpoint:
        with np.load(args.init_checkpoint, allow_pickle=False) as checkpoint:
            params = {
                name.replace("__", "/"): jnp.asarray(checkpoint[name])
                for name in checkpoint.files if name != "metadata"
            }
    else:
        params = initialize_params(config, args.seed)
    params = jax.device_put(params, device)
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.clip_norm),
        optax.adamw(args.learning_rate, weight_decay=args.weight_decay),
    )
    opt_state = optimizer.init(params)

    def loss_fn(current):
        prediction = reconstruct(
            current, xbar_j, valid_j, config,
            value_source=args.value_source,
            routing_mode=args.routing_mode,
        )
        return masked_mse(prediction, xbar_j, valid_j)

    loss_grad = jax.jit(jax.value_and_grad(loss_fn))
    first_loss, first_grads = loss_grad(params)
    initial_gradient_norm = tree_l2(first_grads)
    slot_gradient_norm = tree_l2(first_grads["slot_embedding"])
    query_gradient_norm = tree_l2(first_grads["decoder_queries"])
    history = []
    for step in range(args.steps):
        loss, grads = loss_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if (
            step == 0
            or (step + 1) % args.log_every == 0
            or step + 1 == args.steps
        ):
            history.append({"step": step + 1, "loss": float(loss)})
            if float(loss) <= args.early_stop_mse:
                break

    metrics = evaluate(
        params, xbar_j, valid_j, x_t, normalizer, config,
        args.value_source, args.routing_mode,
    )
    group_names = ("image", "detail", "context", "prompt")
    group_pass = {
        group: (
            metrics[f"{group}/mse"] < args.group_mse_gate
            and metrics[f"{group}/r2"] > args.group_r2_gate
        )
        for group in group_names
    }
    gates = {
        "all_mse": metrics["all/mse"] < args.total_mse_gate,
        "all_r2": metrics["all/r2"] > args.total_r2_gate,
        "denormalized_rmse": metrics["denormalized/rmse"] < args.raw_rmse_gate,
        "invalid_output_zero": metrics["invalid_output_nonzero"] == 0,
        "all_groups": all(group_pass.values()),
    }
    gates["passed"] = all(gates.values())
    result = {
        "protocol": "a0_slot_reconstruction_v1",
        "num_states": args.num_states,
        "seed": args.seed,
        "selected_rows": selected_rows,
        "selected_global_indices": global_indices,
        "selected_tasks": [records[row]["task"] for row in selected_rows],
        "feature_root": str(Path(args.features).resolve()),
        "records_sha256": sha256_file(args.records),
        "normalization_artifact": str(Path(args.normalization).resolve()),
        "init_checkpoint": (
            str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None
        ),
        "normalization_hash": normalizer.hash_bytes.hex(),
        "pca_sha256": normalizer.pca_sha256,
        "device": str(device),
        "config": {
            **dataclass_to_dict(config),
            "key_source": "normalized_memory",
            "value_source": args.value_source,
            "routing_mode": args.routing_mode,
            "pre_norm": True,
            "final_layer_norm": False,
        },
        "optimizer": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "clip_norm": args.clip_norm,
            "steps_requested": args.steps,
            "steps_ran": history[-1]["step"],
        },
        "initial_loss": float(first_loss),
        "initial_gradient_norm": initial_gradient_norm,
        "slot_embedding_gradient_norm": slot_gradient_norm,
        "decoder_query_gradient_norm": query_gradient_norm,
        "history": history,
        "metrics": metrics,
        "group_pass": group_pass,
        "gates": gates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    checkpoint = {
        name.replace("/", "__"): np.asarray(value)
        for name, value in params.items()
    }
    checkpoint["metadata"] = np.asarray(json.dumps({
        "protocol": result["protocol"],
        "config": result["config"],
        "normalization_hash": result["normalization_hash"],
        "pca_sha256": result["pca_sha256"],
    }, sort_keys=True))
    np.savez_compressed(output.with_suffix(".npz"), **checkpoint)
    return result


def dataclass_to_dict(config):
    import dataclasses
    return dataclasses.asdict(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-states", type=int, default=64)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-hidden", type=int, default=1024)
    parser.add_argument(
        "--value-source",
        choices=("normalized_memory", "raw_xbar"),
        default="normalized_memory",
    )
    parser.add_argument(
        "--routing-mode",
        choices=("content_softmax", "hard_diagonal", "slot_key"),
        default="content_softmax",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--early-stop-mse", type=float, default=1e-5)
    parser.add_argument("--total-mse-gate", type=float, default=1e-4)
    parser.add_argument("--total-r2-gate", type=float, default=0.999)
    parser.add_argument("--group-mse-gate", type=float, default=1e-3)
    parser.add_argument("--group-r2-gate", type=float, default=0.99)
    parser.add_argument("--raw-rmse-gate", type=float, default=1e-2)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "output": str(Path(args.output).resolve()),
        "num_states": result["num_states"],
        "steps_ran": result["optimizer"]["steps_ran"],
        "initial_loss": result["initial_loss"],
        "final_mse": result["metrics"]["all/mse"],
        "final_r2": result["metrics"]["all/r2"],
        "group_pass": result["group_pass"],
        "gates": result["gates"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
