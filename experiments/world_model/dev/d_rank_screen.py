"""Rank screen: can the deployed output head even represent the source kernel?

The code head factorises its logits as ``h_piece . e_{c'}`` with both sides
``code_embedding_dim``-dimensional, so for *any* set of contexts the matrix of
code logits has rank <= ``code_embedding_dim`` (= 8). That is a hard structural
ceiling, independent of backbone capacity.

``d_head_bakeoff``'s docstring asserts a prior "rank screen" settled this ("the
best free rank-8 basis is nearly lossless"). That screen was never committed;
the only surviving numbers (``refine-logs/FINDINGS_20260814.md:122``) say the
opposite -- 0.42 bit/code, ~850 bit/transition -- and the inference drawn from
them is retracted on ``:111``. So the question is reopened here, and this time
the script is checked in.

What is measured: for each (task, position) the dense source kernel
``K_p(c'|c)`` is the conditional a head would have to express to match the
source baseline through the code channel. We fit the best rank-r logit
factorisation to it and report the residual KL, weighted by how often each
source code actually occurs in the fit split.

Two families are fitted, because they differ exactly by the fix that was
adopted after the original screen:

* ``plain``     logits = U V^T                 -- rank-r alone
* ``copy``      p = pi*1[c'=c] + (1-pi)*softmax(U V^T)  -- the *deployed* head

The copy-gate family has never been rank-screened. Persistence carries 85-92%
of the mass (worklog section 3), and ``1[c'=c]`` is full rank, so the copy gate
may already buy most of what raising the rank would buy -- or not, which is the
point of measuring.

Read-only w.r.t. the cache; uses train fit rows only, never validation.
"""
from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..baselines import cache_positions
from ..cache import FrozenCache
from ..schema import NUM_CATEGORIES
from .artifacts import dev_path, write_dev_json
from .d_residual import episode_split
from .source_kernel import fit_task, pair_counts, source_kernel

LN2 = float(np.log(2.0))


def collect_targets(cache, fit_rows, positions_per_task, seed):
    """Dense source kernels + empirical row weights for sampled (task, position)."""
    POSITIONS = cache_positions(cache)  # noqa: N806
    rng = np.random.default_rng(seed)
    tasks = np.unique(cache.transitions["task_ids"][fit_rows])
    kernels, weights, task_of, position_of, task_weight = [], [], [], [], []
    for task in tasks:
        tables = fit_task(cache, int(task), fit_rows)
        chosen = np.sort(rng.choice(POSITIONS, positions_per_task, replace=False))
        kernels.append(source_kernel(tables, chosen))            # (P,256,256)
        # How often each source code actually occurs at that position: rows the
        # model never sees should not count toward the representation penalty.
        weights.append(pair_counts(tables, chosen).sum(-1).astype(np.float64))
        task_of.extend([int(task)] * len(chosen))
        position_of.extend(chosen.tolist())
        task_weight.extend([float(tables.n_fit)] * len(chosen))
    return (
        np.concatenate(kernels, 0), np.concatenate(weights, 0),
        np.asarray(task_of), np.asarray(position_of), np.asarray(task_weight),
    )


def fit_rank(K, w, rank, *, use_copy, steps, learning_rate, seed, report=None):
    """Best rank-``rank`` (optionally copy-gated) fit; returns bits/code."""
    P, C, _ = K.shape
    keys = jax.random.split(jax.random.PRNGKey(seed), 2)
    scale = rank ** -0.5
    params = {
        "U": jax.random.normal(keys[0], (P, C, rank), jnp.float32) * scale,
        "V": jax.random.normal(keys[1], (P, C, rank), jnp.float32) * scale,
    }
    if use_copy:
        # pi = sigmoid(a), per context -- the deployed gate is also a function
        # of the context, so a free per-row pi is the right idealisation.
        params["a"] = jnp.zeros((P, C), jnp.float32)

    K = jnp.asarray(K, jnp.float32)
    logK = jnp.log(jnp.clip(K, 1e-30, None))
    w = jnp.asarray(w / np.maximum(w.sum(-1, keepdims=True), 1e-12), jnp.float32)
    # -1e30 rather than -inf: same mixture, but gradients stay finite.
    log_indicator = jnp.where(jnp.eye(C, dtype=bool), 0.0, -1e30)[None]

    def loss(p):
        logits = jnp.einsum("pcr,pdr->pcd", p["U"], p["V"])
        logq = jax.nn.log_softmax(logits, axis=-1)
        if use_copy:
            logq = jnp.logaddexp(
                jax.nn.log_sigmoid(p["a"])[..., None] + log_indicator,
                jax.nn.log_sigmoid(-p["a"])[..., None] + logq,
            )
        kl = jnp.sum(K * (logK - logq), axis=-1)          # (P,C) nats
        return jnp.sum(w * kl) / P

    optimizer = optax.adam(optax.cosine_decay_schedule(learning_rate, steps))
    state = optimizer.init(params)

    @jax.jit
    def step(params, state):
        value, grads = jax.value_and_grad(loss)(params)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, value

    trace = []
    for i in range(1, steps + 1):
        params, state, value = step(params, state)
        if i % max(steps // 8, 1) == 0 or i == steps:
            bits = float(value) / LN2
            trace.append({"step": i, "bits_per_code": bits})
            if report:
                report(f"    step {i:5d}  {bits:.4f} bit/code")
    return trace[-1]["bits_per_code"], trace


def run(args) -> dict:
    cache = FrozenCache(args.cache)
    POSITIONS = cache_positions(cache)  # noqa: N806
    fit_rows, _dev_rows = episode_split(
        cache, cache.indices_for_split("train"), args.fit_fraction, args.seed
    )
    started = time.time()
    K, w, task_of, position_of, _tw = collect_targets(
        cache, fit_rows, args.positions_per_task, args.seed
    )
    print(f"targets: {K.shape[0]} (task,position) pairs from {len(fit_rows)} fit rows",
          flush=True)

    # Reference: the kernel's own conditional entropy, i.e. what a *perfect*
    # head would still have to pay. The rank penalty is on top of this.
    weights = w / np.maximum(w.sum(-1, keepdims=True), 1e-12)
    entropy = float(
        (weights * -(K * np.log(np.clip(K, 1e-30, None))).sum(-1)).sum(-1).mean()
    ) / LN2

    results = []
    for rank in args.ranks:
        for use_copy in (False, True):
            family = "copy" if use_copy else "plain"
            print(f"  rank {rank:3d}  {family}", flush=True)
            bits, trace = fit_rank(
                K, w, rank, use_copy=use_copy, steps=args.steps,
                learning_rate=args.learning_rate, seed=args.seed,
                report=(print if args.verbose else None),
            )
            results.append({
                "rank": rank, "family": family,
                "bits_per_code": bits,
                "bits_per_transition": bits * POSITIONS,
                "trace": trace,
            })
            print(f"    -> {bits:.4f} bit/code = {bits * POSITIONS:.1f} bit/transition",
                  flush=True)

    return {
        "diagnostic": "rank_screen",
        "what": (
            "residual KL of the best rank-r logit factorisation of the dense "
            "source kernel K_p(c'|c), weighted by empirical source-code "
            "frequency; 'copy' adds the deployed explicit copy gate"
        ),
        "deployed_rank": 8,
        "seed": args.seed,
        "sample": {
            "pairs": int(K.shape[0]),
            "positions_per_task": args.positions_per_task,
            "tasks": sorted(set(task_of.tolist())),
            "fit_transitions": int(len(fit_rows)),
            "formal_validation_touched": False,
        },
        "kernel_conditional_entropy_bits_per_code": entropy,
        "kernel_conditional_entropy_bits_per_transition": entropy * POSITIONS,
        "optimisation": {
            "steps": args.steps, "learning_rate": args.learning_rate,
            "note": "adam + cosine decay; reported value is the final loss",
        },
        "results": results,
        "wall_seconds": time.time() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="outputs/world_model/v8/cache")
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--positions-per-task", type=int, default=24)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--fit-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    write_dev_json(dev_path(args.output), result)
    print(json.dumps(
        {f"{r['family']}_r{r['rank']}": round(r["bits_per_transition"], 1)
         for r in result["results"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
