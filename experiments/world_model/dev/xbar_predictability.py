"""How predictable is the next observation, before any quantiser is involved?

``opq_rate_ceiling`` says the code channel is cheap when the next observation is
predicted well, and the trained model sits at roughly R^2 0.10. That leaves two
very different explanations, and they call for different work:

* the corpus is simply this unpredictable in xbar space, or
* it is more predictable than that and the discrete code channel is throwing the
  prediction away.

Three references, all in the OPQ-rotated standardised space the codes live in,
all fitted on train episodes and reported on a held-out dev split:

``persist``   predict x_{t+1} = x_t. The continuous analogue of the copy
              baseline, and the thing 10.3% code persistence is a coarsening of.
``mean``      predict the training mean. R^2 0 by construction; a control.
``ridge``     the best linear map x_t -> x_{t+1}, fitted per slot. A lower bound
              on what a model could reach: it is linear, sees one step of
              history, and never sees the action.

A ridge that lands near ``persist`` says the corpus really is close to a random
walk in this representation and the model has little to find. A ridge well above
it says the ceiling curve's upper reaches are reachable and the loss is being
spent somewhere else.
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


def load_xbar(patterns: list[str]) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with np.load(path, allow_pickle=True) as data:
                states = np.asarray(data["xbar"], np.float32)
                ids = [str(v) for v in np.asarray(data["state_ids"])]
            for name, row in zip(ids, states):
                table[name] = row
    return table


def pairs_for(records: list[Path], xbar: dict, split: str) -> list[tuple[str, str]]:
    rows = [json.loads(line) for path in records
            for line in _jsonl_lines(path) if line.strip()]
    rows = [r for r in rows if r["split"] == split]
    rows.sort(key=lambda r: (r["episode_id"], int(r["step"])))
    return [
        (a["state_id"], b["state_id"])
        for a, b in zip(rows, rows[1:])
        if a["episode_id"] == b["episode_id"]
        and int(b["step"]) == int(a["step"]) + 1
        and a["state_id"] in xbar and b["state_id"] in xbar
    ]


def r_squared(prediction: np.ndarray, truth: np.ndarray, baseline: np.ndarray) -> float:
    """1 - SSE/SST, with SST taken against the *training* mean, not the batch's."""
    sse = float(np.square(prediction - truth).sum())
    sst = float(np.square(truth - baseline).sum())
    return 1.0 - sse / sst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq", required=True)
    parser.add_argument("--encoded-glob", action="append", required=True)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--max-fit", type=int, default=30000)
    args = parser.parse_args()

    pq = np.load(args.pq, allow_pickle=True)
    mean, scale = np.asarray(pq["mean"]), np.asarray(pq["scale"])
    bases, order = np.asarray(pq["rotation_bases"]), np.asarray(pq["rotation_order"])

    xbar = load_xbar(args.encoded_glob)
    print(f"  xbar {len(xbar)} states", flush=True)

    def rotated(ids: list[str]) -> np.ndarray:
        block = np.stack([xbar[i] for i in ids])
        # See opq_rate_ceiling: qformer_pq rotates with `@ bases.T`. R^2 is
        # invariant to an orthonormal change of basis, so the numbers this file
        # reported before the fix stand, but the transform is shared now anyway.
        z = ((block - mean) / scale).astype(np.float32)
        return rotate(z, bases, order)

    fit = pairs_for(args.records, xbar, "train")[:args.max_fit]
    dev = pairs_for(args.records, xbar, "validation")
    print(f"  fit {len(fit)} transitions, dev {len(dev)}", flush=True)

    fit_source, fit_target = rotated([a for a, _ in fit]), rotated([b for _, b in fit])
    dev_source, dev_target = rotated([a for a, _ in dev]), rotated([b for _, b in dev])
    training_mean = fit_target.mean(0, keepdims=True)

    slots, channels = fit_source.shape[1], fit_source.shape[2]
    prediction = np.empty_like(dev_target)
    eye = np.eye(channels, dtype=np.float64) * args.ridge
    for slot in range(slots):
        a = fit_source[:, slot].astype(np.float64)
        b = fit_target[:, slot].astype(np.float64)
        centre_a, centre_b = a.mean(0), b.mean(0)
        weights = np.linalg.solve((a - centre_a).T @ (a - centre_a) + eye,
                                  (a - centre_a).T @ (b - centre_b))
        prediction[:, slot] = (
            (dev_source[:, slot].astype(np.float64) - centre_a) @ weights + centre_b
        ).astype(np.float32)

    print(f"\n  {'predictor':<10} {'R^2 on dev':>11}")
    for name, values in (
        ("mean", np.broadcast_to(training_mean, dev_target.shape)),
        ("persist", dev_source),
        ("ridge", prediction),
    ):
        print(f"  {name:<10} {r_squared(values, dev_target, training_mean):>11.4f}",
              flush=True)


if __name__ == "__main__":
    main()
