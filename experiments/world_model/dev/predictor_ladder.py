"""Is R^2 0.46 the corpus, or the model?

``decoded_r2`` puts the trained 12-layer Transformer at R^2 0.4588 on the next
observation, and ``xbar_predictability`` puts a per-slot linear map at 0.4622 --
a map with no action channel, no history beyond one step, and no nonlinearity.
A 30.86M-parameter model that exactly matches a linear regression has either
found the corpus's ceiling or is being stopped by something other than capacity.

This fits a ladder of continuous predictors on the same fit episodes and scores
them on the same held-out dev transitions, so the two readings can be separated:

``persist``     x_{t+1} = x_t
``linear``      per-slot ridge from x_t
``linear+act``  ridge from x_t and the action (type one-hot, point, delta)
``linear+hist`` ridge from x_t and x_{t-1}
``mlp``         a shared per-slot MLP from x_t, trained to convergence on dev

If the ladder is flat, the Transformer is at the corpus ceiling and the rate
follows from that -- more data and a wider head both stop mattering. If the MLP
or the action row climbs well past 0.46, the Transformer is leaving prediction
on the table and the suspect is the discrete objective: at R^2 0.46 the code hit
rate is 2%, so almost every gradient it sees says "wrong cell" with no gradient
toward being closer in xbar space.

Everything runs in the OPQ-rotated standardised space, via qformer_pq's own
``rotate`` -- R^2 is invariant to that orthonormal change of basis, but sharing
the transform keeps this comparable to the other two scripts.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.state_tokenizer.qformer_pq import rotate

from ..schema_web import NUM_ACTION_TYPES



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


def triples(records: list[Path], xbar: dict, split: str) -> list[tuple]:
    """``(previous, source, target, action)`` -- previous repeats source at step 0."""
    rows = [json.loads(line) for path in records
            for line in _jsonl_lines(path) if line.strip()]
    rows = [r for r in rows if r["split"] == split]
    rows.sort(key=lambda r: (r["episode_id"], int(r["step"])))
    by_key = {(r["episode_id"], int(r["step"])): r for r in rows}
    out = []
    for a, b in zip(rows, rows[1:]):
        if (a["episode_id"] != b["episode_id"]
                or int(b["step"]) != int(a["step"]) + 1
                or a["state_id"] not in xbar or b["state_id"] not in xbar):
            continue
        earlier = by_key.get((a["episode_id"], int(a["step"]) - 1), a)
        if earlier["state_id"] not in xbar:
            earlier = a
        out.append((earlier["state_id"], a["state_id"], b["state_id"], a["action"]))
    return out


def action_features(actions: list[dict]) -> np.ndarray:
    """Type one-hot, point with its validity flag, signed-log delta with its own."""
    count = len(actions)
    out = np.zeros((count, NUM_ACTION_TYPES + 6), np.float32)
    for i, action in enumerate(actions):
        out[i, min(int(action["type_id"]), NUM_ACTION_TYPES - 1)] = 1.0
        coord, delta = bool(action["has_coord"]), bool(action["has_delta"])
        out[i, NUM_ACTION_TYPES:NUM_ACTION_TYPES + 3] = (
            float(action["x"]) * coord, float(action["y"]) * coord, float(coord)
        )
        sign = lambda v: np.sign(v) * np.log1p(abs(v))          # noqa: E731
        out[i, NUM_ACTION_TYPES + 3:] = (
            sign(float(action["dx"])) * delta, sign(float(action["dy"])) * delta,
            float(delta),
        )
    return out


def ridge_r2(fit_x, fit_y, dev_x, dev_y, centre, penalty) -> float:
    """Per-slot ridge. fit_x/dev_x may carry extra channels beyond the slot."""
    prediction = np.empty_like(dev_y)
    for slot in range(fit_y.shape[1]):
        a = fit_x[:, slot].astype(np.float64)
        b = fit_y[:, slot].astype(np.float64)
        mid_a, mid_b = a.mean(0), b.mean(0)
        weights = np.linalg.solve(
            (a - mid_a).T @ (a - mid_a) + np.eye(a.shape[1]) * penalty,
            (a - mid_a).T @ (b - mid_b),
        )
        prediction[:, slot] = (
            (dev_x[:, slot].astype(np.float64) - mid_a) @ weights + mid_b
        ).astype(np.float32)
    sse = float(np.square(prediction - dev_y).sum())
    sst = float(np.square(dev_y - centre).sum())
    return 1.0 - sse / sst


def mlp_r2(fit_x, fit_y, dev_x, dev_y, centre, *, hidden, steps, seed, batch) -> float:
    """One MLP shared across slots. Slots become extra rows, not extra parameters."""
    flat_fit_x = fit_x.reshape(-1, fit_x.shape[-1])
    flat_fit_y = fit_y.reshape(-1, fit_y.shape[-1])
    flat_dev_x = dev_x.reshape(-1, dev_x.shape[-1])
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 4)
    width_in, width_out = flat_fit_x.shape[1], flat_fit_y.shape[1]
    params = {
        "w1": jax.random.normal(keys[0], (width_in, hidden)) / np.sqrt(width_in),
        "b1": jnp.zeros((hidden,)),
        "w2": jax.random.normal(keys[1], (hidden, hidden)) / np.sqrt(hidden),
        "b2": jnp.zeros((hidden,)),
        "w3": jax.random.normal(keys[2], (hidden, width_out)) / np.sqrt(hidden),
        "b3": jnp.zeros((width_out,)),
    }

    def apply(p, x):
        h = jax.nn.gelu(x @ p["w1"] + p["b1"])
        h = jax.nn.gelu(h @ p["w2"] + p["b2"])
        return h @ p["w3"] + p["b3"]

    optimizer = optax.adamw(optax.cosine_decay_schedule(1e-3, steps), weight_decay=1e-4)
    state = optimizer.init(params)

    @jax.jit
    def update(p, s, x, y):
        loss, grads = jax.value_and_grad(
            lambda q: jnp.mean(jnp.square(apply(q, x) - y))
        )(p)
        updates, s = optimizer.update(grads, s, p)
        return optax.apply_updates(p, updates), s, loss

    rng = np.random.default_rng(seed)
    for step in range(steps):
        pick = rng.integers(0, len(flat_fit_x), batch)
        params, state, loss = update(
            params, state, jnp.asarray(flat_fit_x[pick]), jnp.asarray(flat_fit_y[pick])
        )
        if step % max(steps // 6, 1) == 0:
            print(f"      mlp step {step:>6}  train mse {float(loss):.4f}", flush=True)

    chunks = [np.asarray(apply(params, jnp.asarray(flat_dev_x[i:i + 8192])))
              for i in range(0, len(flat_dev_x), 8192)]
    prediction = np.concatenate(chunks).reshape(dev_y.shape)
    sse = float(np.square(prediction - dev_y).sum())
    sst = float(np.square(dev_y - centre).sum())
    return 1.0 - sse / sst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq", required=True)
    parser.add_argument("--encoded-glob", action="append", required=True)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--max-fit", type=int, default=30000)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--mlp-steps", type=int, default=12000)
    parser.add_argument("--mlp-batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pq = np.load(args.pq, allow_pickle=True)
    mean, scale = np.asarray(pq["mean"]), np.asarray(pq["scale"])
    bases, order = np.asarray(pq["rotation_bases"]), np.asarray(pq["rotation_order"])
    xbar = load_xbar(args.encoded_glob)

    def rotated(ids):
        z = ((np.stack([xbar[i] for i in ids]) - mean) / scale).astype(np.float32)
        return rotate(z, bases, order)

    fit = triples(args.records, xbar, "train")[:args.max_fit]
    dev = triples(args.records, xbar, "validation")
    print(f"  fit {len(fit)}  dev {len(dev)}", flush=True)

    parts = {}
    for name, rows in (("fit", fit), ("dev", dev)):
        parts[f"{name}_prev"] = rotated([r[0] for r in rows])
        parts[f"{name}_src"] = rotated([r[1] for r in rows])
        parts[f"{name}_tgt"] = rotated([r[2] for r in rows])
        parts[f"{name}_act"] = action_features([r[3] for r in rows])
    centre = parts["fit_tgt"].mean(0, keepdims=True)
    slots = parts["fit_src"].shape[1]

    def broadcast(features, reference):
        return np.repeat(features[:, None, :], reference.shape[1], axis=1)

    results = {}
    sse = float(np.square(parts["dev_src"] - parts["dev_tgt"]).sum())
    sst = float(np.square(parts["dev_tgt"] - centre).sum())
    results["persist"] = 1.0 - sse / sst
    results["linear"] = ridge_r2(parts["fit_src"], parts["fit_tgt"],
                                 parts["dev_src"], parts["dev_tgt"], centre, args.ridge)
    results["linear+act"] = ridge_r2(
        np.concatenate([parts["fit_src"], broadcast(parts["fit_act"], parts["fit_src"])], -1),
        parts["fit_tgt"],
        np.concatenate([parts["dev_src"], broadcast(parts["dev_act"], parts["dev_src"])], -1),
        parts["dev_tgt"], centre, args.ridge)
    results["linear+hist"] = ridge_r2(
        np.concatenate([parts["fit_src"], parts["fit_prev"]], -1), parts["fit_tgt"],
        np.concatenate([parts["dev_src"], parts["dev_prev"]], -1),
        parts["dev_tgt"], centre, args.ridge)
    for name, value in results.items():
        print(f"  {name:<12} {value:>8.4f}", flush=True)

    print("  mlp ...", flush=True)
    results["mlp"] = mlp_r2(
        parts["fit_src"], parts["fit_tgt"], parts["dev_src"], parts["dev_tgt"], centre,
        hidden=args.hidden, steps=args.mlp_steps, seed=args.seed, batch=args.mlp_batch)
    results["mlp+act"] = mlp_r2(
        np.concatenate([parts["fit_src"], broadcast(parts["fit_act"], parts["fit_src"])], -1),
        parts["fit_tgt"],
        np.concatenate([parts["dev_src"], broadcast(parts["dev_act"], parts["dev_src"])], -1),
        parts["dev_tgt"], centre,
        hidden=args.hidden, steps=args.mlp_steps, seed=args.seed, batch=args.mlp_batch)

    print(f"\n  {'predictor':<12} {'R^2 on dev':>11}")
    for name, value in results.items():
        print(f"  {name:<12} {value:>11.4f}")
    print(f"\n  reference: the trained transformer decodes to 0.4588 "
          f"({slots} slots, same dev episodes)")


if __name__ == "__main__":
    main()
