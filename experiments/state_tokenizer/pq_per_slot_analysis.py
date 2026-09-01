from __future__ import annotations

import argparse
import json
import time

import numpy as np

from experiments.state_tokenizer.pq_reconstruction_probe import (
    consecutive_pairs,
    load_by_episode,
    load_plain,
)
from experiments.state_tokenizer.qformer_pq import (
    decode,
    encode,
    fit_codebook,
    fit_rotation,
    fit_standardizer,
)


def per_slot_r2(states, rebuilt):
    residual = np.square(states - rebuilt, dtype=np.float64).sum(axis=(0, 2))
    spread = np.square(
        states - states.mean(axis=0, keepdims=True), dtype=np.float64
    ).sum(axis=(0, 2))
    return 1.0 - residual / np.maximum(spread, 1e-12)


def per_slot_persistence(codes, pairs):
    previous, current = pairs
    return (codes[previous] == codes[current]).mean(axis=(0, 2))


def per_slot_bigram_bits(fit_codes, fit_pairs, eval_codes, eval_pairs,
                         categories, alpha=0.5):
    fit_prev, fit_cur = fit_pairs
    ev_prev, ev_cur = eval_pairs
    slots, subspaces = fit_codes.shape[1], fit_codes.shape[2]
    out = np.empty(slots)
    for slot in range(slots):
        total = 0.0
        for subspace in range(subspaces):
            joint = np.zeros((categories, categories))
            np.add.at(joint,
                      (fit_codes[fit_prev, slot, subspace],
                       fit_codes[fit_cur, slot, subspace]), 1.0)
            joint += alpha
            conditional = joint / joint.sum(1, keepdims=True)
            total += -np.log2(conditional[eval_codes[ev_prev, slot, subspace],
                                          eval_codes[ev_cur, slot, subspace]]).mean()
        out[slot] = total / subspaces
    return out


def identity_frame(slots, dim):
    bases = np.tile(np.eye(dim, dtype=np.float32), (slots, 1, 1))
    order = np.tile(np.arange(dim, dtype=np.int64), (slots, 1))
    return bases, order


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True)
    parser.add_argument("--wma-web")
    parser.add_argument("--n-fit", type=int, default=50000)
    parser.add_argument("--n-eval", type=int, default=20000)
    parser.add_argument("--num-subspaces", type=int, default=32)
    parser.add_argument("--num-categories", type=int, default=64)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.time()
    fit, fit_ep, fit_step, ev, ev_ep, ev_step = load_by_episode(
        args.states, args.n_fit, args.n_eval, args.seed)
    fit_pairs = consecutive_pairs(fit_ep, fit_step)
    eval_pairs = consecutive_pairs(ev_ep, ev_step)
    wma = load_plain(args.wma_web)
    print(f"[load] fit {fit.shape} eval {ev.shape} "
          f"({len(eval_pairs[0])} pairs)  {time.time()-started:.0f}s", flush=True)

    mean, scale = fit_standardizer(fit)
    fit_s = ((fit - mean) / scale).astype(np.float32)
    slots, dim = fit.shape[1], fit.shape[2]

    frames = {}
    frames["opq"] = fit_rotation(fit_s, num_subspaces=args.num_subspaces,
                                 mode="per-slot")
    frames["pq"] = identity_frame(slots, dim)

    measured = {}
    for name, (bases, order) in frames.items():
        from experiments.state_tokenizer.qformer_pq import rotate
        centroids, _ = fit_codebook(
            rotate(fit_s, bases, order),
            num_subspaces=args.num_subspaces, num_categories=args.num_categories,
            seed=args.seed, iterations=args.kmeans_iterations, problem_batch=64,
        )
        kwargs = dict(bases=bases, order=order)
        eval_codes = encode(ev, mean, scale, centroids, **kwargs)
        fit_codes = encode(fit, mean, scale, centroids, **kwargs)
        rebuilt = decode(eval_codes, mean, scale, centroids, **kwargs)
        entry = {
            "r2": per_slot_r2(ev, rebuilt).tolist(),
            "persistence": per_slot_persistence(eval_codes, eval_pairs).tolist(),
            "bigram_bits": per_slot_bigram_bits(
                fit_codes, fit_pairs, eval_codes, eval_pairs,
                args.num_categories).tolist(),
        }
        if wma is not None:
            entry["r2_wma"] = per_slot_r2(
                wma, decode(encode(wma, mean, scale, centroids, **kwargs),
                            mean, scale, centroids, **kwargs)).tolist()
        measured[name] = entry
        print(f"[{name}] slot R2 min {min(entry['r2']):.4f} "
              f"median {np.median(entry['r2']):.4f} max {max(entry['r2']):.4f}  "
              f"persistence {min(entry['persistence'])*100:.1f}-"
              f"{max(entry['persistence'])*100:.1f}%  "
              f"{time.time()-started:.0f}s", flush=True)

    gap = np.array(measured["opq"]["r2"]) - np.array(measured["pq"]["r2"])
    print("\nslot  R2_opq   R2_pq    gap     persist_opq  persist_pq  "
          "bits_opq  bits_pq")
    for slot in np.argsort(gap):
        print(f"{slot:4d}  {measured['opq']['r2'][slot]:.4f}  "
              f"{measured['pq']['r2'][slot]:.4f}  {gap[slot]:+.4f}   "
              f"{measured['opq']['persistence'][slot]*100:9.1f}%  "
              f"{measured['pq']['persistence'][slot]*100:9.1f}%  "
              f"{measured['opq']['bigram_bits'][slot]:7.3f}  "
              f"{measured['pq']['bigram_bits'][slot]:7.3f}")

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({
            "protocol": "residualmem_pq_per_slot_v1",
            "n_fit": int(len(fit)), "n_eval": int(len(ev)),
            "num_subspaces": args.num_subspaces,
            "num_categories": args.num_categories,
            "per_slot": measured,
            "gap_opq_minus_pq": gap.tolist(),
        }, handle, indent=2)
    print(f"\n[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
