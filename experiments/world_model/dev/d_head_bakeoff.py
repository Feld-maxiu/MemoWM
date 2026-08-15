"""A3: which output head can actually represent the persistence structure?

A1 showed the neural model recovers only 2.7% of what the source table knows at
a matched information set. Two candidate causes, both about the output head:

* it is *tied* to the input code embedding, so "keep the current code" is only
  expressible as a rank-8 Gram matrix;
* it has no explicit persistence term, while the source baseline is 85-92%
  copy prior by mass.

This is a 2x2 over {tied, untied} x {no gate, copy gate}, run entirely inside
an 80/20 episode split of TRAIN. The formal validation split is never touched,
so choosing a head here costs no validation look.

Measured outcome (2026-08-14, variant struct_no_history, seed 0, 20k steps):
tied 9207.46, untied 9549.15, tied+copy 8772.70, untied+copy 8818.27 code
bits/transition, against source 9073.71 on the same split. The copy gate wins
with 18,432 extra parameters; untying alone is actively harmful.

``widen`` is deliberately absent: the rank screen showed the best free rank-8
basis is nearly lossless while the learned tied basis is not, so raising
``code_embedding_dim`` (which would force ``d_model`` 256->512 and ~4x the
parameters) answers a question the evidence has already settled.
"""
from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..cache import FrozenCache
from ..config import load_config
from ..model import codelength_bits, initialize_params, parameter_count, predict
from ..train import DeterministicSampler, build_optimizer
from .artifacts import dev_path, write_dev_json
from .d_residual import MODEL_KEYS, OPTIONAL_KEYS, apply_budget, episode_split

HEADS = ("tied", "untied", "tied_copy", "untied_copy")


def apply_head(params, head: str):
    """Attach the optional head parameters for one 2x2 cell."""
    params = dict(params)
    if head.startswith("untied"):
        # Initialised to the tied values, so every cell starts from the same
        # function and only the freedom to diverge differs.
        params["code_head/w"] = jnp.array(params["code_embedding"])
    if head.endswith("copy"):
        tokens, subspaces, _categories, dim = params["code_embedding"].shape
        params["copy_head/w"] = jnp.zeros((tokens, subspaces, dim), jnp.float32)
        params["copy_head/b"] = jnp.zeros((tokens, subspaces), jnp.float32)
    return params


def make_loss(variant, model_config, *, train: bool):
    def compute(params, batch, key):
        actions = {
            "types": batch["action_types"], "tags": batch["action_tags"],
            "refs": batch["action_refs"], "payloads": batch["action_payloads"],
            "lengths": batch["action_lengths"],
            "targets": batch.get("action_targets"),
            "target_lengths": batch.get("action_target_lengths"),
        }
        mask_logits, code_logits = predict(
            params, batch["history_codes"], batch["history_valid"], actions,
            batch["task_ids"], variant, model_config,
            history_present=batch["history_present"], rng=key, train=train,
        )
        rates = codelength_bits(
            mask_logits, code_logits, batch["target_valid"], batch["target_codes"]
        )
        return jnp.mean(rates["total_bits"]), rates
    return compute


def evaluate(cache, rows, params, loss_fn, batch_size):
    code = mask = 0.0
    for start in range(0, len(rows), batch_size):
        batch = cache.batch(rows[start:start + batch_size])
        device = {name: jnp.asarray(batch[name]) for name in MODEL_KEYS}
        device.update({
            name: jnp.asarray(batch[name]) for name in OPTIONAL_KEYS
            if batch.get(name) is not None
        })
        _loss, rates = loss_fn(params, device, jax.random.PRNGKey(0))
        code += float(np.asarray(rates["code_bits"], np.float64).sum())
        mask += float(np.asarray(rates["mask_bits"], np.float64).sum())
    n = max(len(rows), 1)
    return {
        "code_bits_per_transition": code / n,
        "mask_bits_per_transition": mask / n,
        "total_bits_per_transition": (code + mask) / n,
        "transitions": int(len(rows)),
    }


def run(args: argparse.Namespace) -> dict:
    jax.config.update("jax_default_matmul_precision", "highest")
    device = jax.devices(args.platform)[args.device_index]
    cache = FrozenCache(args.cache)
    config = load_config(args.config, num_tasks=len(cache.task_names))
    config = apply_budget(config, args)
    fit_rows, dev_rows = episode_split(
        cache, cache.indices_for_split("train"), args.fit_fraction, args.seed
    )

    params = apply_head(initialize_params(config.model, args.seed), args.head)
    params = jax.device_put(params, device)
    train_loss = make_loss(args.variant, config.model, train=True)
    eval_loss = jax.jit(make_loss(args.variant, config.model, train=False))
    initial = evaluate(cache, dev_rows, params, eval_loss, config.evaluation.batch_size)

    optimizer, schedule = build_optimizer(params, config.training)
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params, opt_state, batch, key):
        grads = jax.grad(lambda p: train_loss(p, batch, key)[0])(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state

    sampler = DeterministicSampler(fit_rows, config.training.batch_size, args.seed)
    key = jax.random.PRNGKey(args.seed)
    training = config.training
    started = time.time()
    history, best, best_step = [], dict(initial), 0
    since_improvement = 0
    stop_reason = "max_steps"
    step = 0
    for step in range(1, training.max_steps + 1):
        batch = cache.batch(sampler.next())
        device_batch = {name: jnp.asarray(batch[name]) for name in MODEL_KEYS}
        device_batch.update({
            name: jnp.asarray(batch[name]) for name in OPTIONAL_KEYS
            if batch.get(name) is not None
        })
        key, subkey = jax.random.split(key)
        params, opt_state = update(params, opt_state, device_batch, subkey)
        if step % training.eval_every and step != training.max_steps:
            continue
        metrics = evaluate(
            cache, dev_rows, params, eval_loss, config.evaluation.batch_size
        )
        metrics["step"] = step
        metrics["learning_rate"] = float(schedule(step))
        history.append(metrics)
        if metrics["code_bits_per_transition"] < best["code_bits_per_transition"]:
            best, best_step, since_improvement = dict(metrics), step, 0
        else:
            since_improvement += training.eval_every
        print(json.dumps(metrics, sort_keys=True), flush=True)
        # A run that ends on max_steps is budget-truncated: its number is an
        # upper bound, not a converged value.
        if step >= training.min_steps and since_improvement >= training.patience_steps:
            stop_reason = "patience"
            break

    return {
        "diagnostic": "head_bakeoff",
        "head": args.head,
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
        "parameter_count": int(parameter_count(params)),
        "capacity": {
            "d_model": config.model.d_model,
            "num_layers": config.model.num_layers,
            "mlp_dim": config.model.mlp_dim,
            "code_embedding_dim": config.model.code_embedding_dim,
        },
        "split": {
            "basis": "train episodes, task-stratified",
            "fit_transitions": int(len(fit_rows)),
            "dev_transitions": int(len(dev_rows)),
            "formal_validation_touched": False,
        },
        "wall_seconds": time.time() - started,
        "initial": initial,
        "best": best,
        "best_step": best_step,
        "history": history,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="outputs/world_model/v8/cache")
    parser.add_argument("--config", default="configs/world_model/v8_discrete.yaml")
    parser.add_argument("--head", required=True, choices=HEADS)
    parser.add_argument("--variant", default="struct_no_history")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--patience-steps", type=int, default=10000)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--use-target-channel", action="store_true")
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
        "head": result["head"], "best": result["best"],
        "parameter_count": result["parameter_count"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
