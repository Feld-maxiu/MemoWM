"""Decode a trained model's code distribution back to xbar and score it.

``xbar_predictability`` measures what a per-slot ridge reaches on the same dev
transitions -- 0.4622 against persistence's 0.1437 -- and ``opq_rate_ceiling``
maps an R^2 to a code cost. Neither says where the trained model actually sits,
because the ceiling curve decodes with a distance softmax at an optimal
temperature while the model decodes with a rank-8 tied head. Those are different
estimator families: the curve's R^2=0 row costs 7,311 bits where the empirical
marginal baseline costs 7,936, so the curve reads systematically low and cannot
be compared to a code-space number directly.

This puts the model on the ridge's ruler instead. For each dev transition it
takes the model's per-subspace distribution over the 256 centroids, forms the
expected centroid, reassembles the rotated vector, inverts the rotation and
standardisation, and scores R^2 against the true next xbar.

Two outcomes, and they call for different work:

* R^2 near 0.10 -- the model has not learned what a linear map already finds.
  The suspect is the objective: a 256-way cross entropy per subspace charges
  "slightly wrong cell" and "entirely wrong cell" the same, so nothing in the
  gradient rewards moving closer in xbar space.
* R^2 near 0.46 with the rate still at 7,050 -- the model has learned it and the
  head is discarding it. The suspect is the rank-8 tied head, whose residual
  against the first-order kernel measured 224.4 bits, and which cannot be
  widened without changing d_model (model.py:89).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.state_tokenizer.qformer_pq import unrotate

from ..cache import FrozenCache
from ..config import load_config
from ..model import actions_from_batch, predict
from ..train import load_checkpoint, model_batch_keys



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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="a train.py output directory")
    parser.add_argument("--checkpoint", default="best.pkl")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--pq", required=True)
    parser.add_argument("--encoded-glob", action="append", required=True)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=4000)
    args = parser.parse_args()

    record = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    config = load_config(args.run / "resolved.yaml")
    params = jax.device_get(load_checkpoint(args.run / args.checkpoint)["params"])
    params = {k: jnp.asarray(v) for k, v in params.items()}
    variant = record["variant"]

    pq = np.load(args.pq, allow_pickle=True)
    mean, scale = np.asarray(pq["mean"]), np.asarray(pq["scale"])
    centroids = np.asarray(pq["centroids"])            # (slots, subs, 256, dim)
    bases, order = np.asarray(pq["rotation_bases"]), np.asarray(pq["rotation_order"])

    cache = FrozenCache(args.cache)
    rows = cache.indices_for_split(args.split, test_freeze_manifest=None)
    rows = rows[:args.limit]

    # The cache indexes states by position in the sorted record manifest, so the
    # same ordering is rebuilt here to recover each target's state_id.
    manifest = [json.loads(line) for path in args.records
                for line in _jsonl_lines(path) if line.strip()]
    manifest = [r for r in manifest if r["split"] in ("train", "validation")]
    manifest.sort(key=lambda r: (r["episode_id"], int(r["step"])))
    state_ids = [r["state_id"] for r in manifest]

    xbar = load_xbar(args.encoded_glob)
    keys = model_batch_keys(config.model)
    slots, subs, categories, dim = centroids.shape

    predictions, truths = [], []
    for start in range(0, len(rows), args.batch_size):
        block = rows[start:start + args.batch_size]
        batch = cache.batch(block)
        device = {name: jnp.asarray(batch[name]) for name in keys}
        _mask_logits, code_logits = predict(
            params, device["history_codes"], device["history_valid"],
            actions_from_batch(device, config.model), device["task_ids"],
            variant, config.model,
            history_present=device["history_present"],
            rng=jax.random.PRNGKey(0), train=False,
        )
        probability = np.asarray(jax.nn.softmax(code_logits, axis=-1), np.float32)
        # Expected centroid under the model's own distribution, per (slot, sub).
        expected = np.einsum("bigc,igcd->bigd", probability, centroids)
        rotated = expected.reshape(len(block), slots, subs * dim)
        # unrotate(), not a reimplementation -- see opq_rate_ceiling's note.
        recovered = unrotate(rotated, bases, order) * scale + mean
        predictions.append(recovered)
        truths.append(np.stack([
            xbar[state_ids[int(i)]]
            for i in cache.transitions["target_indices"][block]
        ]))

    prediction = np.concatenate(predictions)
    truth = np.concatenate(truths)
    centre = truth.mean(0, keepdims=True)
    sse = float(np.square(prediction - truth).sum())
    sst = float(np.square(truth - centre).sum())
    print(json.dumps({
        "run": str(args.run),
        "checkpoint": args.checkpoint,
        "variant": variant,
        "transitions": int(len(prediction)),
        "decoded_r2": 1.0 - sse / sst,
        "rate_bits_per_transition": record["best_selection"]["total_bits_per_transition"]
        if args.checkpoint == "best.pkl" else
        record.get("last_selection", {}).get("total_bits_per_transition"),
    }, indent=2))


if __name__ == "__main__":
    main()
