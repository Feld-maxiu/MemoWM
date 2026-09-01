"""Train the recurrent world model and score it on the same dev transitions.

Separate from ``train.py`` rather than a flag on it: the batch is a set of
episode chunks instead of a set of transitions, so the sampler, the loss
reduction and the evaluation all differ. Everything that can be shared is
imported -- the optimiser, the checkpoint format, the episode split -- so the
budget is derived the same way and the numbers land on the same ruler.

The reported rate is bits per *transition*, averaged over exactly the rows a
windowed run would have scored. Padding steps and steps outside the split are
run so the recurrent state is correct, and then excluded from the average.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from experiments.state_tokenizer.common import sha256_file, write_json

from .cache import CACHE_FILES, FrozenCache
from .config import TrainingConfig
from .episodes import EpisodeView
from .model import codelength_bits, parameter_count, parameter_shapes
from . import model_recurrent, model_segment

#: Both take the same batch and return per-step logits of the same shape, so the
#: loop below is architecture-agnostic and the two land on the same ruler.
ARCHITECTURES = {
    "recurrent": (model_recurrent.RecurrentConfig,
                  model_recurrent.initialize_params, model_recurrent.predict),
    "segment": (model_segment.SegmentConfig,
                model_segment.initialize_params, model_segment.predict),
}
from .schema_web import WEB_CACHE_PROTOCOL
from .train import _atomic_pickle, build_optimizer, episode_split


ACTION_KEYS = ("x", "y", "dx", "dy", "has_coord", "has_delta")


def actions_of(batch) -> dict:
    actions = {
        "types": jnp.asarray(batch["action_types"]),
        "payloads": jnp.asarray(batch["action_payloads"]),
        "lengths": jnp.asarray(batch["action_lengths"]),
    }
    for name in ACTION_KEYS:
        actions[name] = jnp.asarray(batch[f"action_{name}"])
    return actions


def chunk_bits(params, batch, variant, config, forward, *, rng, train):
    """Total scored bits and the number of scored transitions in this batch."""
    mask_logits, code_logits = forward(
        params, jnp.asarray(batch["codes"]), jnp.asarray(batch["code_valid"]),
        jnp.asarray(batch["valid"]), actions_of(batch),
        jnp.asarray(batch["task_ids"], jnp.int32), variant, config,
        rng=rng, train=train,
    )
    count, length = code_logits.shape[0], code_logits.shape[1]
    flat = codelength_bits(
        mask_logits.reshape(count * length, -1),
        code_logits.reshape((count * length,) + code_logits.shape[2:]),
        jnp.asarray(batch["target_valid"]).reshape(count * length, -1),
        jnp.asarray(batch["target_codes"]).reshape(
            count * length, config.num_latent_tokens, config.num_subspaces),
    )
    weight = jnp.asarray(batch["loss_mask"], jnp.float32).reshape(-1)
    return {
        "total": jnp.sum(flat["total_bits"] * weight),
        "code": jnp.sum(flat["code_bits"] * weight),
        "mask": jnp.sum(flat["mask_bits"] * weight),
        "count": jnp.sum(weight),
    }


class ChunkSampler:
    """Shuffles chunks; state is fully checkpointed, as in train.DeterministicSampler."""

    def __init__(self, count: int, batch_size: int, seed: int):
        self.count, self.batch_size = int(count), int(batch_size)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.count)
        self.cursor, self.epochs = 0, 0

    def next(self) -> np.ndarray:
        picked, needed = [], self.batch_size
        while needed:
            take = min(needed, len(self.order) - self.cursor)
            picked.append(self.order[self.cursor:self.cursor + take])
            self.cursor += take
            needed -= take
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(self.count)
                self.cursor, self.epochs = 0, self.epochs + 1
        return np.concatenate(picked)


def evaluate(params, view, chunks, variant, config, forward, batch_size) -> dict:
    totals = {"total": 0.0, "code": 0.0, "mask": 0.0, "count": 0.0}
    for start in range(0, len(chunks["state_indices"]), batch_size):
        window = {k: v[start:start + batch_size] for k, v in chunks.items()}
        result = chunk_bits(
            params, view.materialise(window), variant, config, forward,
            rng=jax.random.PRNGKey(0), train=False,
        )
        for name in totals:
            totals[name] += float(result[name])
    scored = max(totals["count"], 1.0)
    return {
        "transitions": int(totals["count"]),
        "total_bits_per_transition": totals["total"] / scored,
        "code_bits_per_transition": totals["code"] / scored,
        "mask_bits_per_transition": totals["mask"] / scored,
    }


def run(args) -> dict:
    started = time.time()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if raw.get("protocol") != WEB_CACHE_PROTOCOL:
        raise SystemExit(f"unexpected protocol {raw.get('protocol')!r}")
    cache = FrozenCache(args.cache)
    config_type, build_params, forward = ARCHITECTURES[args.architecture]
    model_config = config_type(
        num_tasks=len(cache.task_names), **{
            k: v for k, v in raw["model"].items() if k != "num_tasks"
        })
    training = TrainingConfig(**raw["training"])
    # Applied in one call: TrainingConfig validates min_steps <= max_steps in
    # __post_init__, so setting them one at a time raises halfway through
    # whenever the new max is below the old min.
    overrides = {
        name: getattr(args, name)
        for name in ("batch_size", "learning_rate", "warmup_steps", "max_steps",
                     "min_steps", "eval_every", "patience_steps")
        if getattr(args, name, None) is not None
    }
    training = dataclasses.replace(training, **overrides) if overrides else training

    chunk_length = getattr(model_config, 'chunk_length', model_config.max_history)
    view = EpisodeView(cache, chunk_length=chunk_length)
    train_rows = cache.indices_for_split("train", test_freeze_manifest=None)
    if args.dev_fraction:
        train_rows, selection_rows = episode_split(
            cache, train_rows, 1.0 - args.dev_fraction, args.seed
        )
    else:
        selection_rows = cache.indices_for_split("validation", test_freeze_manifest=None)
    fit_chunks = view.chunks_for(train_rows)
    dev_chunks = view.chunks_for(selection_rows)

    device = jax.devices(args.platform)[args.device_index]
    params = jax.device_put(build_params(model_config, args.seed), device)
    optimizer, _schedule = build_optimizer(params, training)
    opt_state = optimizer.init(params)
    sampler = ChunkSampler(len(fit_chunks["state_indices"]), training.batch_size, args.seed)

    @jax.jit
    def update(params, opt_state, batch, key):
        def objective(current):
            result = chunk_bits(current, batch, args.variant, model_config, forward,
                                rng=key, train=True)
            return result["total"] / jnp.maximum(result["count"], 1.0)
        loss, grads = jax.value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    key = jax.random.PRNGKey(args.seed + 1)
    best, best_step, stale = float("inf"), 0, 0
    for step in range(1, training.max_steps + 1):
        batch = view.materialise({k: v[sampler.next()] for k, v in fit_chunks.items()})
        key, subkey = jax.random.split(key)
        params, opt_state, loss = update(params, opt_state, batch, subkey)
        if step % training.eval_every and step != training.max_steps:
            continue
        record = evaluate(params, view, dev_chunks, args.variant,
                          model_config, forward, args.eval_chunks)
        rate = record["total_bits_per_transition"]
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "step": step, "train_loss": float(loss), "selection": record,
                "wall_seconds": time.time() - started,
            }) + "\n")
        print(f"  step {step:>7}  dev {rate:>9.1f}  train {float(loss):>9.1f}", flush=True)
        if rate < best:
            best, best_step, stale = rate, step, 0
            _atomic_pickle(output / "best.pkl", {
                "protocol": WEB_CACHE_PROTOCOL, "params": jax.device_get(params),
                "step": step, "metadata": {"variant": args.variant, "seed": args.seed},
            })
        else:
            stale += training.eval_every
        _atomic_pickle(output / "last.pkl", {
            "protocol": WEB_CACHE_PROTOCOL, "params": jax.device_get(params),
            "step": step, "metadata": {"variant": args.variant, "seed": args.seed},
        })
        if step >= training.min_steps and stale >= training.patience_steps:
            print(f"  early stop: {stale} steps without improvement", flush=True)
            break

    result = {
        "protocol": WEB_CACHE_PROTOCOL,
        "architecture": args.architecture,
        "variant": args.variant,
        "seed": args.seed,
        "parameter_count": parameter_count(params),
        "parameter_shapes": {k: list(v) for k, v in parameter_shapes(params).items()},
        "data": {
            "cache": str(Path(args.cache).resolve()),
            "cache_manifest_sha256": sha256_file(Path(args.cache) / CACHE_FILES["manifest"]),
            "train_transitions": int(fit_chunks["loss_mask"].sum()),
            "selection_transitions": int(dev_chunks["loss_mask"].sum()),
            "selection": "train_dev_split" if args.dev_fraction else "validation",
            "dev_fraction": args.dev_fraction,
            "test_evaluated": False,
            "dev_run": True,
        },
        "episode_view": view.coverage(),
        "model": dataclasses.asdict(model_config),
        "training": {**dataclasses.asdict(training), "best_step": best_step,
                     "wall_seconds": time.time() - started},
        "best_dev_bits_per_transition": best,
    }
    write_json(output / "run.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--architecture", choices=tuple(ARCHITECTURES), default="recurrent")
    parser.add_argument("--variant", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--dev-fraction", type=float)
    parser.add_argument("--eval-chunks", type=int, default=32)
    for name in ("batch_size", "warmup_steps", "max_steps", "min_steps",
                 "eval_every", "patience_steps"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int)
    parser.add_argument("--learning-rate", type=float)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({k: result[k] for k in
                      ("parameter_count", "best_dev_bits_per_transition")}, indent=2))


if __name__ == "__main__":
    main()
