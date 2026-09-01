from __future__ import annotations

import argparse
import json
import time

import numpy as np

from experiments.state_tokenizer.pq_reconstruction_probe import (
    consecutive_pairs,
    load_by_episode,
)
from experiments.state_tokenizer.qformer_pq import (
    decode,
    encode,
    fit_codebook,
    fit_rotation,
    fit_standardizer,
    reconstruction_metrics,
    rotate,
)


def fit_ridge(previous, current, penalty):
    slots, dim = previous.shape[1], previous.shape[2]
    weights = np.empty((slots, dim + 1, dim), np.float32)
    for slot in range(slots):
        design = np.concatenate(
            [previous[:, slot, :], np.ones((len(previous), 1), np.float32)], axis=1
        ).astype(np.float64)
        gram = design.T @ design
        gram[np.arange(dim), np.arange(dim)] += penalty
        weights[slot] = np.linalg.solve(
            gram, design.T @ current[:, slot, :].astype(np.float64)
        ).astype(np.float32)
    return weights


def apply_ridge(previous, weights):
    out = np.empty_like(previous)
    for slot in range(previous.shape[1]):
        out[:, slot, :] = (previous[:, slot, :] @ weights[slot, :-1]
                           + weights[slot, -1])
    return out


def quantise_and_score(fit_target, eval_target, eval_reference, eval_offset,
                       *, subspaces, categories, seed, iterations):
    mean, scale = fit_standardizer(fit_target)
    standard = ((fit_target - mean) / scale).astype(np.float32)
    bases, order = fit_rotation(standard, num_subspaces=subspaces, mode="shared")
    centroids, _ = fit_codebook(
        rotate(standard, bases, order), num_subspaces=subspaces,
        num_categories=categories, seed=seed, iterations=iterations,
        problem_batch=64,
    )
    codes = encode(eval_target, mean, scale, centroids, bases=bases, order=order)
    rebuilt = decode(codes, mean, scale, centroids, bases=bases, order=order)
    return reconstruction_metrics(eval_reference, rebuilt + eval_offset), codes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True)
    parser.add_argument("--n-fit", type=int, default=50000)
    parser.add_argument("--n-eval", type=int, default=20000)
    parser.add_argument("--num-subspaces", type=int, default=32)
    parser.add_argument("--num-categories", type=int, default=64)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--ridge-penalty", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.time()
    fit, fit_ep, fit_step, ev, ev_ep, ev_step = load_by_episode(
        args.states, args.n_fit, args.n_eval, args.seed)
    fit_prev, fit_cur = consecutive_pairs(fit_ep, fit_step)
    ev_prev, ev_cur = consecutive_pairs(ev_ep, ev_step)
    width = 32 * args.num_subspaces * int(np.log2(args.num_categories))
    print(f"[load] fit {len(fit_cur)} pairs  eval {len(ev_cur)} pairs  "
          f"width {width}  {time.time()-started:.0f}s", flush=True)

    ridge = fit_ridge(fit[fit_prev], fit[fit_cur], args.ridge_penalty)
    fit_hat = apply_ridge(fit[fit_prev], ridge)
    ev_hat = apply_ridge(ev[ev_prev], ridge)
    predictor = reconstruction_metrics(ev[ev_cur], ev_hat)
    scale_ratio = float(ev[ev_cur].std() and
                        (ev[ev_cur] - ev_hat).std() / ev[ev_cur].std())
    print(f"[predictor] continuous R2 {predictor['r2']:.4f}  "
          f"residual sd / state sd = {scale_ratio:.4f}  "
          f"(expect ~0.46 and ~0.73)  {time.time()-started:.0f}s", flush=True)

    zeros = np.zeros((), np.float32)
    direct, _ = quantise_and_score(
        fit[fit_cur], ev[ev_cur], ev[ev_cur], zeros,
        subspaces=args.num_subspaces, categories=args.num_categories,
        seed=args.seed, iterations=args.kmeans_iterations)
    print(f"[direct]  quantise z_t          R2 {direct['r2']:.5f}  "
          f"MSE {direct['mse']:.6f}  {time.time()-started:.0f}s", flush=True)

    dpcm, _ = quantise_and_score(
        fit[fit_cur] - fit_hat, ev[ev_cur] - ev_hat, ev[ev_cur], ev_hat,
        subspaces=args.num_subspaces, categories=args.num_categories,
        seed=args.seed, iterations=args.kmeans_iterations)
    print(f"[dpcm]    quantise z_t - zhat_t R2 {dpcm['r2']:.5f}  "
          f"MSE {dpcm['mse']:.6f}  {time.time()-started:.0f}s", flush=True)

    print(f"\nMSE ratio dpcm/direct = {dpcm['mse']/direct['mse']:.4f}  "
          f"({100*(1-dpcm['mse']/direct['mse']):+.1f}% distortion at equal bits)")

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({
            "protocol": "residualmem_dpcm_probe_v1",
            "loop": "open",
            "fixed_width_bits": width,
            "eval_pairs": int(len(ev_cur)),
            "predictor": predictor,
            "residual_scale_ratio": scale_ratio,
            "direct": direct, "dpcm": dpcm,
            "mse_ratio_dpcm_over_direct": dpcm["mse"] / direct["mse"],
        }, handle, indent=2)
    print(f"[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
