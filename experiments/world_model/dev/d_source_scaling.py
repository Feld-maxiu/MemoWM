"""Data-scaling curve for the *source* baseline, as a control for the neural one.

The neural scaling curve alone cannot distinguish "this task has more to learn
from more data" from "this model happens to keep improving". Refitting the
source-conditioned Markov baseline on the same nested subsets gives a reference
curve from a fixed-capacity, closed-form estimator: it can only improve by
having denser count tables.

If ``L_neural(N)`` keeps falling while ``L_source(N)`` flattens, the remaining
headroom is structure the lookup table cannot express and the model can --
which is the case for collecting more data. If both flatten, more data buys
little for either.

Everything is fit on train episodes and scored on the same fixed dev split as
the neural curve; the formal validation split is never touched. Cheap enough to
run on CPU: only count tables are refit, no training.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ..cache import FrozenCache
from .artifacts import dev_path, write_dev_json
from .d_residual import episode_split, subsample_fit
from .source_kernel import source_code_bits


def run(args) -> dict:
    cache = FrozenCache(args.cache)
    fit_rows, dev_rows = episode_split(
        cache, cache.indices_for_split("train"), args.fit_fraction, args.seed
    )
    started = time.time()
    points = []
    for fraction in args.fractions:
        subset = subsample_fit(cache, fit_rows, fraction, args.seed)
        bits = source_code_bits(cache, subset, dev_rows)
        episodes = int(len(np.unique(cache.transitions["episode_ids"][subset])))
        point = {
            "fit_subsample": fraction,
            "fit_transitions": int(len(subset)),
            "fit_episodes": episodes,
            "source_code_bits_per_transition": float(bits.mean()),
        }
        points.append(point)
        print(f"  {fraction:>5.0%}  fit={len(subset):>6,}  "
              f"source={point['source_code_bits_per_transition']:.2f}", flush=True)

    for previous, current in zip(points, points[1:]):
        current["delta_from_previous"] = (
            previous["source_code_bits_per_transition"]
            - current["source_code_bits_per_transition"]
        )
    return {
        "diagnostic": "source_data_scaling",
        "what": (
            "source-conditioned Markov baseline refit on nested subsets of the "
            "fit episodes, scored on the fixed dev split; control curve for the "
            "neural data-scaling experiment"
        ),
        "seed": args.seed,
        "split": {
            "basis": "train episodes, task-stratified, nested subsets",
            "dev_transitions": int(len(dev_rows)),
            "fit_transitions_at_full": int(len(fit_rows)),
            "formal_validation_touched": False,
        },
        "points": points,
        "wall_seconds": time.time() - started,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="outputs/world_model/v8/cache")
    parser.add_argument(
        "--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument("--fit-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run(args)
    write_dev_json(dev_path(args.output), result)
    print(json.dumps(result["points"], indent=2))


if __name__ == "__main__":
    main()
