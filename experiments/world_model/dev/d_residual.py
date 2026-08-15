"""A2: does anything remain that the source lookup table cannot capture?

The pure neural WM has to relearn the source kernel from scratch, and A1 showed
it recovers only 2.7% of it. That makes ``L_full - L_source`` a mixture of two
opposite quantities. This diagnostic removes the relearning cost entirely:

    logits = log K_p(.|c_p) + Delta_theta(y_t, u_t, T)

with ``Delta_theta`` zero-initialised, so the model *starts* exactly at the
source baseline and the only thing it can learn is what the lookup table does
not already know. The headline number is therefore
``L_source - L_source+neural`` directly, rather than a difference of two
differently-misspecified models.

Everything runs on an 80/20 episode split of TRAIN. The formal validation split
is never touched.

Primary metric is code bits: the mask head is not zero-initialised (only the
code path is), and A1 established the entire source deficit lives on the code
channel.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..baselines import POSITIONS
from ..cache import FrozenCache
from ..config import load_config
from ..model import codelength_bits, initialize_params, predict
from ..schema import NUM_CATEGORIES
from ..train import DeterministicSampler, build_optimizer
from .artifacts import dev_path, write_dev_json
from .source_kernel import fit_task, source_kernel

MODEL_KEYS = (
    "history_codes", "history_valid", "history_present",
    "action_types", "action_tags", "action_refs", "action_payloads",
    "action_lengths", "task_ids", "target_codes", "target_valid",
)


def episode_split(cache: FrozenCache, rows: np.ndarray, fraction: float, seed: int):
    """Split train rows 80/20 by whole episode, stratified by task."""
    rng = np.random.default_rng(seed)
    episodes = cache.transitions["episode_ids"][rows]
    tasks = cache.transitions["task_ids"][rows]
    held: set[int] = set()
    for task in np.unique(tasks):
        unique = np.unique(episodes[tasks == task])
        shuffled = rng.permutation(unique)
        take = max(1, int(round(len(shuffled) * (1.0 - fraction))))
        held.update(shuffled[:take].tolist())
    mask = np.isin(episodes, np.fromiter(held, np.int64, len(held)))
    return rows[~mask], rows[mask]


def build_log_prior(cache: FrozenCache, fit_rows: np.ndarray, block: int = 256):
    """Device-resident ``log K`` -> (num_tasks, POSITIONS, 256, 256) float32."""
    num_tasks = len(cache.task_names)
    table = np.zeros((num_tasks, POSITIONS, NUM_CATEGORIES, NUM_CATEGORIES), np.float32)
    present = np.unique(cache.transitions["task_ids"][fit_rows])
    for task in present:
        tables = fit_task(cache, int(task), fit_rows)
        for start in range(0, POSITIONS, block):
            positions = np.arange(start, min(start + block, POSITIONS))
            kernel = source_kernel(tables, positions)
            table[int(task), positions] = np.log(kernel).astype(np.float32)
    return jnp.asarray(table), present


def gather_prior(log_prior, task_ids, history_codes):
    """Per-transition source log-rows -> (B, 64, 32, 256)."""
    source = jnp.asarray(history_codes, jnp.int32)[:, -1].reshape(-1, POSITIONS)
    batch = source.shape[0]
    rows = log_prior[
        jnp.asarray(task_ids, jnp.int32)[:, None],
        jnp.arange(POSITIONS)[None, :],
        source,
    ]
    return rows.reshape(batch, 64, 32, NUM_CATEGORIES)


def make_loss(variant, model_config, *, train: bool):
    # log_prior is an explicit argument, not a closure: baking a 6.4 GB table
    # into the jitted graph as a captured constant doubles device memory.
    def compute(params, batch, key, log_prior):
        actions = {
            "types": batch["action_types"], "tags": batch["action_tags"],
            "refs": batch["action_refs"], "payloads": batch["action_payloads"],
            "lengths": batch["action_lengths"],
        }
        prior = gather_prior(log_prior, batch["task_ids"], batch["history_codes"])
        mask_logits, code_logits = predict(
            params, batch["history_codes"], batch["history_valid"], actions,
            batch["task_ids"], variant, model_config,
            history_present=batch["history_present"], rng=key, train=train,
            source_log_prior=prior,
        )
        rates = codelength_bits(
            mask_logits, code_logits, batch["target_valid"], batch["target_codes"]
        )
        return jnp.mean(rates["total_bits"]), rates
    return compute


def evaluate(cache, rows, params, loss_fn, batch_size, log_prior):
    code = mask = 0.0
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        batch = cache.batch(selected)
        device = {name: jnp.asarray(batch[name]) for name in MODEL_KEYS}
        _loss, rates = loss_fn(params, device, jax.random.PRNGKey(0), log_prior)
        code += float(np.asarray(rates["code_bits"], np.float64).sum())
        mask += float(np.asarray(rates["mask_bits"], np.float64).sum())
    n = max(len(rows), 1)
    return {
        "code_bits_per_transition": code / n,
        "mask_bits_per_transition": mask / n,
        "total_bits_per_transition": (code + mask) / n,
        "transitions": int(len(rows)),
    }


def apply_budget(config, args):
    """Override the training budget and (optionally) backbone capacity.

    ``build_optimizer`` decays the cosine schedule over ``training.max_steps``,
    so raising only the loop bound would leave the learning rate annealed to
    zero at the old horizon and the extra steps would be wasted. The schedule
    and the loop must move together.

    ``--no-early-stop`` makes every configuration run exactly ``max_steps``.
    Without it a capacity sweep is not a fixed-budget comparison: cells stop at
    different steps and see different amounts of data.
    """
    patience = args.patience_steps
    if getattr(args, "no_early_stop", False):
        # Larger than any reachable idle streak, so patience can never fire.
        patience = args.max_steps + args.eval_every
    training = dataclasses.replace(
        config.training,
        max_steps=args.max_steps,
        patience_steps=patience,
        eval_every=args.eval_every,
    )
    if training.patience_steps % training.eval_every:
        raise ValueError("patience_steps must be a multiple of eval_every")
    model = config.model
    if getattr(args, "num_layers", None):
        model = dataclasses.replace(model, num_layers=args.num_layers)
    if getattr(args, "mlp_dim", None):
        model = dataclasses.replace(model, mlp_dim=args.mlp_dim)
    return dataclasses.replace(config, training=training, model=model)


def run(args: argparse.Namespace) -> dict:
    jax.config.update("jax_default_matmul_precision", "highest")
    device = jax.devices(args.platform)[args.device_index]
    cache = FrozenCache(args.cache)
    config = load_config(args.config, num_tasks=len(cache.task_names))
    config = apply_budget(config, args)
    train_rows = cache.indices_for_split("train")
    fit_rows, dev_rows = episode_split(cache, train_rows, args.fit_fraction, args.seed)

    started = time.time()
    log_prior, _present = build_log_prior(cache, fit_rows)
    log_prior = jax.device_put(log_prior, device)
    prior_seconds = time.time() - started

    params = initialize_params(config.model, args.seed)
    # Zero-initialised untied output head: Delta is exactly 0 at step 0, so the
    # code channel reproduces the source baseline bit for bit.
    params["code_head/w"] = jnp.zeros_like(params["code_embedding"])
    params = jax.device_put(params, device)

    train_loss = make_loss(args.variant, config.model, train=True)
    eval_loss = jax.jit(make_loss(args.variant, config.model, train=False))
    initial = evaluate(
        cache, dev_rows, params, eval_loss, config.evaluation.batch_size, log_prior
    )

    optimizer, schedule = build_optimizer(params, config.training)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, batch, key, log_prior):
        grads = jax.grad(lambda p: train_loss(p, batch, key, log_prior)[0])(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    sampler = DeterministicSampler(fit_rows, config.training.batch_size, args.seed)
    key = jax.random.PRNGKey(args.seed)
    training = config.training
    history = []
    best = dict(initial)
    best_step = 0
    since_improvement = 0
    stop_reason = "max_steps"
    step = 0
    for step in range(1, training.max_steps + 1):
        batch = cache.batch(sampler.next())
        device_batch = {name: jnp.asarray(batch[name]) for name in MODEL_KEYS}
        key, subkey = jax.random.split(key)
        params, opt_state = update(
            params, opt_state, device_batch, subkey, log_prior
        )
        if step % training.eval_every and step != training.max_steps:
            continue
        metrics = evaluate(
            cache, dev_rows, params, eval_loss,
            config.evaluation.batch_size, log_prior,
        )
        metrics["step"] = step
        metrics["learning_rate"] = float(schedule(step))
        history.append(metrics)
        if metrics["code_bits_per_transition"] < best["code_bits_per_transition"]:
            best, best_step, since_improvement = dict(metrics), step, 0
        else:
            since_improvement += training.eval_every
        print(json.dumps(metrics, sort_keys=True), flush=True)
        # A run that ends on max_steps is budget-truncated and its number is an
        # upper bound, not a converged value.
        if step >= training.min_steps and since_improvement >= training.patience_steps:
            stop_reason = "patience"
            break

    return {
        "diagnostic": "source_plus_neural_residual",
        "variant": args.variant,
        "seed": args.seed,
        "budget": {
            "max_steps": config.training.max_steps,
            "patience_steps": config.training.patience_steps,
            "eval_every": config.training.eval_every,
            "steps_ran": step,
            "stop_reason": stop_reason,
            "budget_truncated": stop_reason == "max_steps",
        },
        "split": {
            "basis": "train episodes, task-stratified",
            "fit_transitions": int(len(fit_rows)),
            "dev_transitions": int(len(dev_rows)),
            "formal_validation_touched": False,
        },
        "prior_build_seconds": prior_seconds,
        "R0_source_only": initial,
        "best": best,
        "best_step": best_step,
        "history": history,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="outputs/world_model/v8/cache")
    parser.add_argument("--config", default="configs/world_model/v8_discrete.yaml")
    parser.add_argument("--variant", required=True, choices=(
        "state_only", "struct_no_history", "no_history",
    ))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--patience-steps", type=int, default=10000)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--mlp-dim", type=int)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--fit-fraction", type=float, default=0.8)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    write_dev_json(dev_path(args.output), result)
    print(json.dumps({
        "variant": result["variant"],
        "R0_source_only": result["R0_source_only"],
        "best": result["best"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
