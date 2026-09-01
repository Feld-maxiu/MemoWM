"""Dump the world model's per-position posterior for every transition in a cache.

The gate needs two things from the world model and nothing else:

* ``code_bits[n, slot, subspace]`` -- the rate ``-log2 p_theta(z+ | h, u)``, which
  is what a position costs to send; and
* ``wm_argmax[n, slot, subspace]`` -- the estimate the decoder falls back to when
  a position is omitted, i.e. ``Estimate[p_theta]`` of equation (23) read as the
  mode.

Both already exist inside ``model.codelength_bits``: it returns ``code_matrix``
of shape (B, K, M) in bits. Nothing recomputes anything here. The reason this is
a separate script rather than a flag on ``evaluate`` is that ``make_eval``
(``train.py:208-221``) reduces the matrix to a scalar per transition before
anything downstream sees it, and widening that return would mean touching the
training loop -- which is running.

Also dumped, because they are free here and are the natural features for a head
that has to predict utility without seeing the reader: per-position entropy,
top-1 log-probability, the margin between the top two codes, and whether the
argmax happens to equal the true code. That last one matters more than it looks:
where the model's mode is already right, substituting it is the identity, so the
utility of sending that position is exactly zero and the label has a spike there.

Runs under the jax interpreter. Writes one npz, which is the only thing the torch
side ever reads.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.state_tokenizer.common import sha256_file, write_json

from ..world_model.cache import CACHE_FILES, FrozenCache
from ..world_model.config import load_config
from ..world_model.model import codelength_bits, loss_and_metrics
from ..world_model.train import load_checkpoint

LN2 = math.log(2.0)


def _posterior_features(code_logits):
    """Per-position summaries of the categorical posterior, in bits."""
    logprob = jax.nn.log_softmax(code_logits, axis=-1)
    ordered = jnp.sort(logprob, axis=-1)
    top1 = ordered[..., -1]
    top2 = ordered[..., -2]
    entropy = -jnp.sum(jnp.exp(logprob) * logprob, axis=-1) / LN2
    return {
        "wm_argmax": jnp.argmax(logprob, axis=-1).astype(jnp.uint8),
        "entropy_bits": entropy,
        "top1_logprob_bits": -top1 / LN2,
        "margin_bits": (top1 - top2) / LN2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--verify-against", default=None,
                        help="an evaluation.json whose "
                             "summary.code_bits_per_transition this dump must "
                             "reproduce; the check that pins this script to the "
                             "established evaluation protocol")
    parser.add_argument("--verify-tolerance", type=float, default=1.0,
                        help="bits per transition; the run-to-run floor on this "
                             "pipeline is ~31 bits, so anything this script "
                             "computes differently would show up far above 1")
    args = parser.parse_args()

    cache = FrozenCache(args.cache)
    config = load_config(args.config, num_tasks=len(cache.task_names))
    # FrozenCache defaults to None and would hand a 32-step window to a 16-step
    # model; the shape error that follows points nowhere near the cause.
    cache.max_history = config.model.max_history
    jax.config.update("jax_default_matmul_precision", config.training.matmul_precision)
    device = jax.devices(args.platform)[args.device_index]

    checkpoint = load_checkpoint(args.checkpoint)
    variant = checkpoint["metadata"]["variant"]
    params = jax.device_put(checkpoint["params"], device)

    try:
        rows = np.asarray(cache.indices_for_split(args.split))
    except Exception:                                        # noqa: BLE001
        rows = np.arange(cache.transitions)
    if not len(rows):
        raise SystemExit(f"no transitions in split {args.split!r}")

    @jax.jit
    def step(params, batch):
        _loss, (_metrics, rates, _mask_logits, code_logits) = loss_and_metrics(
            params, batch, variant, config.model,
            rng=jax.random.PRNGKey(0), train=False,
        )
        return {"code_bits": rates["code_matrix"], **_posterior_features(code_logits)}

    parts: dict[str, list] = {}
    for start in range(0, len(rows), args.batch_size):
        block = rows[start:start + args.batch_size]
        batch = cache.batch(block)
        out = step(params, batch)
        for name, value in out.items():
            parts.setdefault(name, []).append(np.asarray(value))
        parts.setdefault("target_codes", []).append(
            np.asarray(batch["target_codes"], np.uint8))
        parts.setdefault("target_indices", []).append(
            np.asarray(batch["target_indices"], np.int64))
        parts.setdefault("transition_indices", []).append(
            np.asarray(block, np.int64))
        if start % (args.batch_size * 20) == 0:
            print(f"[dump] {min(start + args.batch_size, len(rows))}/{len(rows)}",
                  flush=True)

    dumped = {name: np.concatenate(blocks) for name, blocks in parts.items()}
    dumped["code_bits"] = dumped["code_bits"].astype(np.float32)
    bits_per_transition = float(dumped["code_bits"].sum(axis=(1, 2)).mean())
    agreement = float((dumped["wm_argmax"] == dumped["target_codes"]).mean())

    report = {
        "protocol": "residualmem_utility_gate_posterior_v1",
        "cache": str(Path(args.cache).resolve()),
        "cache_manifest_sha256": sha256_file(Path(args.cache) / CACHE_FILES["manifest"]),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "variant": variant,
        "split": args.split,
        "transitions": int(len(rows)),
        "code_bits_per_transition": bits_per_transition,
        "code_accuracy": agreement,
        "zero_utility_fraction": agreement,
        "shapes": {name: list(value.shape) for name, value in dumped.items()},
    }

    if args.verify_against:
        reference = json.loads(
            Path(args.verify_against).read_text(encoding="utf-8"))
        # evaluate.py writes {"summary": ...}; train.py's run.json writes
        # {"best_selection": ...}. Same quantity, two shapes -- accept either so
        # the check can be pointed at whichever artifact is fresh.
        for key in ("summary", "best_selection", "last_selection"):
            if isinstance(reference.get(key), dict) and \
                    "code_bits_per_transition" in reference[key]:
                block, source = reference[key], key
                break
        else:
            raise SystemExit(
                f"{args.verify_against} carries no code_bits_per_transition "
                f"under summary/best_selection/last_selection"
            )
        expected = float(block["code_bits_per_transition"])
        delta = abs(expected - bits_per_transition)
        report["verification"] = {
            "reference": str(Path(args.verify_against).resolve()),
            "reference_key": source,
            "expected_code_bits_per_transition": expected,
            "delta": delta, "tolerance": args.verify_tolerance,
            "passed": delta <= args.verify_tolerance,
        }
        if delta > args.verify_tolerance:
            raise SystemExit(
                f"dump gives {bits_per_transition:.2f} bits/transition but "
                f"{args.verify_against} records {expected:.2f} under {source!r} "
                f"(delta {delta:.2f} > {args.verify_tolerance}). This script and "
                f"the training loop disagree about the same checkpoint and "
                f"cache; resolve before using the dump."
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dumped)
    write_json(output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nargmax == true code for {agreement:.4f} of positions -- those are "
          f"exactly the positions whose omission is the identity, so their "
          f"utility is zero by construction.")


if __name__ == "__main__":
    main()
