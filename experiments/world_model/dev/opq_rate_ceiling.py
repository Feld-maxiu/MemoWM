"""What would the OPQ code channel cost a world model that predicted well?

Not a proof of a bound. It is the code length an *oracle-decoder* model pays: it
is handed a prediction of the next observation with a controlled error level,
turns it into a per-subspace distribution using the quantiser's own centroid
geometry (softmax over -d^2/T), and is charged the NLL of the true code. The
temperature is swept and the minimum reported, so the curve is the best that
decoder can do at each prediction quality.

Read it as: "if the world model predicted the next xbar this well, the code
channel would cost at least this much" -- a rate floor conditional on a fidelity
level, not an unconditional bound.

The prediction is simulated as

    x_hat = sqrt(R2) * x + sqrt((1 - R2) * var) * eps

and NOT as ``x + noise``. The latter keeps the truth as its centre, so its true
R2 never falls below 0.5 however much noise is added; a first version of this
script did that and reported 1.32x compression at a nominal R2 of 0.

Distances use |c|^2 - 2 x.c rather than the direct broadcast, which would be
(N, 32, 32, 256, 16) -- 50 GB at N=1500.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.qformer_pq import rotate



def _jsonl_lines(path):
    """Newline-split only. See common.read_jsonl: `splitlines` also breaks on
    U+2028, which 11 MolmoWeb page titles contain, cutting records in half."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield line


def load_xbar(encoded_glob: str) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    for path in sorted(glob.glob(encoded_glob)):
        with np.load(path, allow_pickle=True) as data:
            states = np.asarray(data["xbar"], np.float32)
            ids = [str(v) for v in np.asarray(data["state_ids"])]
        for name, row in zip(ids, states):
            table[name] = row
    return table


def validation_pairs(records: list[Path], xbar: dict) -> list[tuple[str, str]]:
    rows = [json.loads(line) for path in records
            for line in _jsonl_lines(path) if line.strip()]
    rows = [r for r in rows if r["split"] == "validation"]
    rows.sort(key=lambda r: (r["episode_id"], int(r["step"])))
    return [
        (a["state_id"], b["state_id"])
        for a, b in zip(rows, rows[1:])
        if a["episode_id"] == b["episode_id"]
        and int(b["step"]) == int(a["step"]) + 1
        and a["state_id"] in xbar and b["state_id"] in xbar
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq", required=True)
    parser.add_argument("--encoded-glob", required=True)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--transitions", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pq = np.load(args.pq, allow_pickle=True)
    mean, scale = np.asarray(pq["mean"]), np.asarray(pq["scale"])
    centroids = np.asarray(pq["centroids"])              # (slots, subs, 256, 16)
    bases, order = np.asarray(pq["rotation_bases"]), np.asarray(pq["rotation_order"])
    slots, subs = centroids.shape[0], centroids.shape[1]
    positions = slots * subs

    xbar = load_xbar(args.encoded_glob)
    print(f"  xbar {len(xbar)} states", flush=True)
    pairs = validation_pairs(args.records, xbar)
    print(f"  validation transitions {len(pairs)}", flush=True)

    count = min(args.transitions, len(pairs))
    target = np.stack([xbar[b] for _, b in pairs[:count]])

    def to_subspaces(values):
        # rotate(), not a reimplementation: qformer_pq uses `states @ bases.T`
        # and an earlier version of this file used `@ bases`. Both round-trip
        # exactly, because the basis is orthonormal, so a round-trip test does
        # not catch it -- but the codebook was fitted in the former, so the
        # latter puts every vector in the wrong place relative to the centroids
        # and collapsed quantised reconstruction from R^2 0.991 to 0.106.
        z = ((values - mean) / scale).astype(np.float32)
        return rotate(z, bases, order).reshape(len(z), slots, subs, -1)

    truth = to_subspaces(target)
    norms = (centroids ** 2).sum(-1)[None]

    def distance(values, batch=256):
        out = np.empty((len(values), slots, subs, centroids.shape[2]), np.float32)
        for start in range(0, len(values), batch):
            block = values[start:start + batch]
            out[start:start + batch] = norms - 2 * np.einsum(
                "bsmd,smcd->bsmc", block, centroids
            )
        return out

    true_codes = distance(truth).argmin(-1)
    variance = float(truth.var())
    rng = np.random.default_rng(args.seed)

    print(f"\n  {'nominal':>8} {'actual':>7} {'code hit':>9} {'bit/code':>9} "
          f"{'bit/transition':>15} {'vs fixed':>9}", flush=True)
    fixed = positions * 8
    for level in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0):
        if level >= 1.0:
            prediction = truth.copy()
        else:
            noise = rng.normal(
                0.0, np.sqrt(variance * (1.0 - level)), truth.shape
            ).astype(np.float32)
            prediction = np.sqrt(level) * truth + noise
        achieved = float(np.corrcoef(prediction.ravel(), truth.ravel())[0, 1] ** 2)
        squared = distance(prediction)
        hit = float((squared.argmin(-1) == true_codes).mean())
        squared = squared - squared.min(-1, keepdims=True)
        best = None
        for temperature in np.geomspace(1e-3, 50.0, 22):
            logits = -squared / temperature
            shifted = logits - logits.max(-1, keepdims=True)
            normaliser = np.log(np.exp(shifted).sum(-1))
            chosen = np.take_along_axis(shifted, true_codes[..., None], -1)[..., 0]
            bits = float(-(chosen - normaliser).mean() / np.log(2.0))
            best = bits if best is None else min(best, bits)
        total = best * positions
        print(f"  {level:>8.3f} {achieved:>7.3f} {hit * 100:>8.1f}% {best:>9.3f} "
              f"{total:>15.1f} {fixed / total:>8.2f}x", flush=True)


if __name__ == "__main__":
    main()
