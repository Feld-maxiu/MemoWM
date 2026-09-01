"""Sweep how much WorldMemArena the PQ codebook needs to cover its states.

Fitting the codebook on MolmoWeb alone leaves WorldMemArena web at R^2 0.4416
against 0.7885 in-domain -- and that is a transfer failure, not an intrinsic
difficulty: a codebook fitted on WMA web itself reaches 0.9517, i.e. WMA is the
*easier* corpus of the two. The MolmoWeb centroids simply do not cover where WMA
states land (nearest-centroid distance 3.13 against 1.76, and P(|z|>3) is 0.0244
against 0.0024 under MolmoWeb's own standardiser).

The same shape of failure is already on record for the PCA path -- inverse-PCA
R^2 0.4274 on WMA against 0.9178 in-domain -- and the fix that worked there was
a mixed fitting corpus: a 25% WMA share captured 94% of the available gain for
2.8% of the in-domain cost. This sweeps the same axis for the quantiser.

**WorldMemArena web never enters a fit.** The mixing corpus is the six non-web
subcategories, which is also what the Q-Former itself was trained on; web is only
ever evaluated. That keeps the evaluation domain out of a design decision, which
is the one thing the earlier arm-selection design got wrong.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .qformer_pq import (
    codebook_health,
    decode,
    encode,
    fit_codebook,
    fit_standardizer,
    load_splits,
    load_states,
    reconstruction_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molmoweb-states", action="append", required=True)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--wma-fitcorpus", action="append", required=True)
    parser.add_argument("--wma-web", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shares", type=float, nargs="+",
                        default=[0.0, 0.10, 0.25, 0.50, 1.0])
    parser.add_argument("--num-subspaces", type=int, default=32)
    parser.add_argument("--num-categories", type=int, default=256)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--problem-batch", type=int, default=256)
    parser.add_argument("--fit-states", type=int, default=20000,
                        help="total fit budget, held constant across shares so "
                             "the sweep varies composition and not size")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    molmo, molmo_ids, checkpoint = load_states(args.molmoweb_states)
    splits = load_splits(args.records)
    assignment = np.asarray([splits[i] for i in molmo_ids])
    molmo_train = molmo[assignment == "train"]
    molmo_val = molmo[assignment == "validation"]

    fitcorpus, _, fit_checkpoint = load_states(args.wma_fitcorpus)
    web, _, web_checkpoint = load_states([args.wma_web])
    if {fit_checkpoint, web_checkpoint} != {checkpoint}:
        raise SystemExit("all three sets must come from one checkpoint")

    logging.info("MolmoWeb train %d / validation %d | WMA fitcorpus %d | WMA web %d",
                 len(molmo_train), len(molmo_val), len(fitcorpus), len(web))

    rng = np.random.default_rng(args.seed)
    rows = []
    for share in args.shares:
        wanted_wma = int(round(args.fit_states * share))
        wanted_molmo = args.fit_states - wanted_wma
        if wanted_wma > len(fitcorpus):
            # Shrink the budget rather than backfilling with MolmoWeb: topping
            # the shortfall back up would turn share=1.0 into share=0.23 while
            # still being labelled 1.0.
            logging.warning("share %.2f wants %d WMA states but only %d exist; "
                            "fitting on %d instead of %d",
                            share, wanted_wma, len(fitcorpus),
                            wanted_molmo + len(fitcorpus), args.fit_states)
            wanted_wma = len(fitcorpus)

        parts = []
        if wanted_molmo:
            parts.append(molmo_train[rng.choice(len(molmo_train), wanted_molmo, replace=False)])
        if wanted_wma:
            parts.append(fitcorpus[rng.choice(len(fitcorpus), wanted_wma, replace=False)])
        train = np.concatenate(parts)

        mean, scale = fit_standardizer(train)
        centroids, occupancy = fit_codebook(
            (train - mean) / scale,
            num_subspaces=args.num_subspaces, num_categories=args.num_categories,
            seed=args.seed, iterations=args.kmeans_iterations,
            problem_batch=args.problem_batch,
        )

        entry = {"share": share, "wma_states": wanted_wma, "molmoweb_states": wanted_molmo,
                 "empty_clusters": int((occupancy == 0).sum())}
        for name, states in (("molmoweb_val", molmo_val), ("wma_web", web)):
            codes = encode(states, mean, scale, centroids)
            entry[name] = {
                **reconstruction_metrics(states, decode(codes, mean, scale, centroids)),
                "code_health": codebook_health(codes, args.num_categories),
            }
        rows.append(entry)
        logging.info("share %.2f: MolmoWeb val r2 %.4f | WMA web r2 %.4f",
                     share, entry["molmoweb_val"]["r2"], entry["wma_web"]["r2"])

    print(f"\n{'WMA 份额':>10}{'拟合状态数':>22}{'MolmoWeb val R²':>18}{'WMA web R²':>14}{'空簇':>8}")
    for entry in rows:
        print(f"{entry['share']:>10.2f}"
              f"{entry['molmoweb_states']:>12d}+{entry['wma_states']:<9d}"
              f"{entry['molmoweb_val']['r2']:>18.4f}{entry['wma_web']['r2']:>14.4f}"
              f"{entry['empty_clusters']:>8d}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "checkpoint": checkpoint,
        "fit_states": args.fit_states,
        "num_subspaces": args.num_subspaces,
        "num_categories": args.num_categories,
        "note": "WorldMemArena web is evaluated only and never enters a fit",
        "rows": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
