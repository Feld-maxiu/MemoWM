"""What the gate actually costs once the decoder has to live with its own output.

Every rate in this project so far conditions each step on the **true** prefix.
Under all-send that is exact -- report §6.2, eq (23)(24): with $m\\equiv1$ the
decoder's state $\\tilde h_t$ equals the encoder's $h_t$, so teacher forcing is
not an approximation, it is the same computation. The moment anything is dropped
the two separate: the decoder reconstructs the dropped positions from the prior,
carries that approximation forward, and predicts the next state from a history
that is no longer the real one. Its predictions get worse, and worse predictions
cost bits.

So the published 16.8% saving is an **upper bound** and 4312.0 bits is a **lower
bound**. This measures the real number by feeding the reconstruction back.

Two points of semantics that decide whether the number means anything:

**The decoder gates with its own entropy.** The keep rule is
``|U_j| >= lambda * H_j``, and under closed loop $H_j$ comes from the drifted
posterior, not the teacher-forced one. Using open-loop entropy here would be
giving the decoder a quantity it does not have. A consequence worth expecting:
the keep fraction will not come out at exactly the 0.8359 the open-loop run
reported, and that is correct rather than a discrepancy.

**Validity is not part of the reconstruction.** ``rollout._replace_prefix``
overwrites ``history_valid`` alongside the codes, which is right for a rollout --
there the model is imagining whether a slot exists at all. Here the state
demonstrably exists and was demonstrably stored; only its *contents* are
approximate. Overwriting validity would silently conflate "predicted away" with
"dropped to save bits", so this splices codes alone.

The null control is the whole reason to trust the result: with ``--all-send``
every position is kept, so the reconstruction is bit-identical to the truth,
nothing is spliced, and the closed-loop rate must equal the open-loop rate
exactly. If it does not, the splice is wired wrong and no gated number from this
script is worth reading.

☠️ Expect the correction to be small on this corpus, and say so *before*
measuring rather than after. The external test set is 817 transitions over 139
episodes -- about 5.9 steps each -- against a 16-step window, so drift has almost
no room to accumulate. A small number here means the upper bound was tight, which
is a real result; it is not evidence that closing the loop does not matter, and a
large number would mean ``UTILITY_GATE.md`` §0's operating point needs rewriting.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np

from experiments.state_tokenizer.common import sha256_file
from experiments.utility_gate.export_mask import keep_mask, per_state_rate

from ..world_model.cache import CACHE_FILES, FrozenCache
from ..world_model.config import load_config
from ..world_model.model import loss_and_metrics
from ..world_model.train import load_checkpoint
from .dump_wm_posteriors import _posterior_features


def _splice_codes(cache, host_batch, reconstructed: dict) -> int:
    """Write reconstructed codes into the history, leaving validity alone.

    Deliberately not ``rollout._replace_prefix``: that one also replaces
    ``history_valid``. See the module docstring.

    ☠️ The cache was built with a 32-step window and the model takes 16, so
    ``FrozenCache.batch`` slices ``value[:, -keep:]`` -- the *most recent* keep
    slots. ``history_indices`` is still the full width, and indexing it directly
    against the sliced batch writes each state's codes into some other state's
    time slot. The all-send null control catches exactly this, which is what it
    is for.
    """
    histories = cache.transitions["history_indices"][host_batch["transition_indices"]]
    keep = cache.max_history
    if keep is not None and keep < histories.shape[1]:
        histories = histories[:, -keep:]
    if histories.shape[1] != host_batch["history_codes"].shape[1]:
        raise ValueError(
            f"history_indices width {histories.shape[1]} does not match the "
            f"batch's {host_batch['history_codes'].shape[1]} time slots")

    touched = 0
    for local in range(len(histories)):
        for time, state_index in enumerate(histories[local]):
            codes = reconstructed.get(int(state_index))
            if codes is not None:
                host_batch["history_codes"][local, time] = codes
                touched += 1
    return touched


def _pad(block: np.ndarray, size: int) -> tuple[np.ndarray, int]:
    """Pad a batch to a fixed length so jax compiles one shape, not thirty.

    Transitions are grouped by step and the groups have ragged sizes; without
    padding every distinct size triggers a recompile.
    """
    if len(block) >= size:
        return block, len(block)
    filler = np.repeat(block[-1:], size - len(block))
    return np.concatenate([block, filler]), len(block)


def run_pass(cache, params, variant, config, rows, *, utility, lam, closed: bool,
             batch_size: int, arm: str = "gate", seed: int = 35,
             log_every: int = 10) -> dict:
    """One sweep in ascending step order. ``closed`` decides whether to feed back."""
    rng = np.random.default_rng(seed)

    @jax.jit
    def step(params, batch):
        _loss, (_metrics, rates, _mask_logits, code_logits) = loss_and_metrics(
            params, batch, variant, config.model,
            rng=jax.random.PRNGKey(0), train=False,
        )
        return {"code_bits": rates["code_matrix"], **_posterior_features(code_logits)}

    steps = np.asarray(cache.transitions["steps"])[rows]
    reconstructed: dict[int, np.ndarray] = {}
    collected: dict[str, list] = defaultdict(list)
    spliced_total = 0

    for order, value in enumerate(sorted(set(steps.tolist()))):
        group = rows[steps == value]
        for start in range(0, len(group), batch_size):
            block = group[start:start + batch_size]
            padded, real = _pad(block, batch_size)
            host = cache.batch(padded)
            if closed and reconstructed:
                spliced_total += _splice_codes(cache, host, reconstructed)
            out = jax.device_get(step(params, host))

            code_bits = np.asarray(out["code_bits"], np.float64)[:real]
            entropy = np.asarray(out["entropy_bits"], np.float64)[:real]
            argmax = np.asarray(out["wm_argmax"], np.uint8)[:real]
            truth = np.asarray(host["target_codes"], np.uint8)[:real]
            targets = np.asarray(host["target_indices"], np.int64)[:real]

            keep = build_keep(arm, utility, entropy, lam, rng).reshape(truth.shape)
            collected["code_bits"].append(code_bits)
            collected["keep"].append(keep)
            collected["target_indices"].append(targets)
            collected["steps"].append(np.full(real, value, np.int32))
            collected["gated_bits"].append(
                (code_bits.reshape(real, -1) * keep.reshape(real, -1)).sum(1))

            if closed:
                # This is the object the decoder actually ends up holding.
                blended = np.where(keep, truth, argmax).astype(np.uint8)
                for local, target in enumerate(targets):
                    reconstructed[int(target)] = blended[local]
        if order % log_every == 0:
            print(f"[closed-loop] step {value} done "
                  f"({len(reconstructed)} states reconstructed)", flush=True)

    code_bits = np.concatenate(collected["code_bits"])
    keep = np.concatenate(collected["keep"])
    return {
        "states": int(len(code_bits)),
        "full_rate_bits": per_state_rate(code_bits),
        "gated_rate_bits": per_state_rate(code_bits, keep),
        "keep_fraction": float(keep.mean()),
        "history_positions_spliced": int(spliced_total),
        # Kept out of the summary dict's JSON by the caller; used to test whether
        # drift accumulates with depth into the episode, which is the mechanism
        # the whole "short episodes bound the correction" argument rests on.
        "_per_state": {
            "steps": np.concatenate(collected["steps"]),
            "gated_bits": np.concatenate(collected["gated_bits"]),
            "target_indices": np.concatenate(collected["target_indices"]),
        },
    }


def build_keep(arm: str, utility: np.ndarray, entropy: np.ndarray, lam: float,
               rng: np.random.Generator) -> np.ndarray:
    """The gate's mask, or a budget-matched control.

    ☠️ Budget matching is per state and exact, not approximate. An earlier
    comparison in this project reported an effect that turned out to be a
    keep-fraction mismatch between the two arms; at matched budget it vanished.
    Every control here drops the *same number of positions in the same state* as
    the gate does, so the only thing that varies is which ones.

    * ``gate``   -- ``|U_j| >= lambda * H_j``.
    * ``random`` -- a uniformly chosen subset of the same size. The null: is the
      gate's choice of positions doing anything, or would any 16% do?
    * ``anti``   -- the complement of the gate's ranking: keep exactly the
      positions the gate would have dropped first. The adversarial bound.
    """
    gate = keep_mask(utility, entropy, lam)
    if arm == "gate":
        return gate

    states, positions = gate.shape
    # Rank positions the way the gate does -- by how far |U| clears lambda * H --
    # so `anti` is the same criterion read backwards rather than a different one.
    margin = utility[None, :] - lam * np.asarray(entropy, np.float64).reshape(states, -1)
    if arm == "anti":
        order = np.argsort(margin, axis=1)              # worst first
    elif arm == "random":
        order = np.argsort(rng.random((states, positions)), axis=1)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    out = np.zeros_like(gate)
    budget = gate.sum(1)
    for row in range(states):
        out[row, order[row, :budget[row]]] = True
    return out


def drift_by_depth(openloop: dict, closed: dict, edges=(0, 1, 2, 4, 8, 16, 1 << 30)) -> list:
    """Does the correction grow the further into an episode you get?

    If drift is flat in step index then it is not accumulation and the claim that
    a longer-episode corpus would pay more has no support. If it rises, the
    mechanism is confirmed and the size of the correction here is a property of
    this corpus rather than of the method.
    """
    open_states = openloop["_per_state"]
    closed_states = closed["_per_state"]
    if not np.array_equal(open_states["target_indices"], closed_states["target_indices"]):
        raise ValueError("the two passes visited states in different orders")

    steps = open_states["steps"]
    delta = closed_states["gated_bits"] - open_states["gated_bits"]
    buckets = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (steps >= low) & (steps < high)
        if not selected.any():
            continue
        buckets.append({
            "step_range": f"{low}-{high - 1}" if high < (1 << 30) else f"{low}+",
            "states": int(selected.sum()),
            "mean_drift_bits": float(delta[selected].mean()),
            "median_drift_bits": float(np.median(delta[selected])),
        })
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mask", required=True, help="export_mask.py output")
    parser.add_argument("--lambdas", type=float, nargs="*", default=None,
                        help="defaults to the lambda recorded in the mask")
    parser.add_argument("--all-send", action="store_true",
                        help="null control: keep everything (lambda = 0). The "
                             "closed-loop rate must then equal the open-loop "
                             "rate exactly")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=["gate"],
                        choices=("gate", "random", "anti"),
                        help="budget-matched controls. `random` answers whether "
                             "the gate's choice of positions matters or any "
                             "same-size subset would do")
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with np.load(args.mask, allow_pickle=True) as data:
        utility = np.asarray(data["utility"], np.float64)
        mask_meta = json.loads(str(data["metadata"]))
        default_lam = float(data["lam"])

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
    except Exception:                                            # noqa: BLE001
        rows = np.arange(cache.transitions)
    if not len(rows):
        raise SystemExit(f"no transitions in split {args.split!r}")

    lambdas = [0.0] if args.all_send else (args.lambdas or [default_lam])
    results = []
    for lam, arm in [(l, a) for l in lambdas for a in args.arms]:
        openloop = run_pass(cache, params, variant, config, rows,
                            utility=utility, lam=lam, closed=False, arm=arm,
                            batch_size=args.batch_size, seed=args.seed)
        closed = run_pass(cache, params, variant, config, rows,
                          utility=utility, lam=lam, closed=True, arm=arm,
                          batch_size=args.batch_size, seed=args.seed)
        drift = closed["gated_rate_bits"] - openloop["gated_rate_bits"]
        depth = drift_by_depth(openloop, closed)
        summary = {name: {k: v for k, v in block.items() if k != "_per_state"}
                   for name, block in (("open_loop", openloop), ("closed_loop", closed))}
        row = {
            "lambda": lam,
            "arm": arm,
            **summary,
            "drift_bits": drift,
            "drift_fraction_of_open_saving": (
                drift / max(openloop["full_rate_bits"] - openloop["gated_rate_bits"], 1e-9)),
            "closed_loop_saved_fraction": float(
                1.0 - closed["gated_rate_bits"] / openloop["full_rate_bits"]),
            "closed_loop_compression": (
                float(6144.0 / closed["gated_rate_bits"])
                if closed["gated_rate_bits"] > 0 else float("inf")),
            # With everything dropped the gated rate is zero by construction and
            # the interesting number is instead the closed pass's own
            # `full_rate_bits`: what sending everything costs once the history is
            # pure prior. That is the ceiling on drift for this model and corpus.
            "history_corruption_cost_bits": float(
                closed["full_rate_bits"] - openloop["full_rate_bits"]),
            # The saving is measured against full-send, which by §6.2 has no
            # drift of its own: with m==1 the decoder state *is* the encoder
            # state, so 5182.83 needs no closed-loop correction and is the right
            # denominator. The closed pass's own full_rate_bits is a different
            # quantity -- what sending everything would cost from an already
            # drifted history -- and is not the baseline.
            "drift_by_episode_depth": depth,
        }
        results.append(row)
        print(json.dumps(row, indent=2), flush=True)

        if lam == 0.0:
            # The null control. All-send means the reconstruction is the truth,
            # so nothing the splice writes can differ from what was already
            # there, and the two passes must agree to the last bit.
            gap = abs(closed["gated_rate_bits"] - openloop["gated_rate_bits"])
            if gap > 1e-6:
                raise SystemExit(
                    f"NULL CONTROL FAILED: all-send closed loop {closed['gated_rate_bits']:.9f} "
                    f"!= open loop {openloop['gated_rate_bits']:.9f} (gap {gap:.3e}). "
                    f"Under m==1 the decoder state equals the encoder state by "
                    f"construction (§6.2); a gap means the splice is wired wrong "
                    f"and no gated number from this script is trustworthy.")
            print(f"[closed-loop] null control passed (gap {gap:.2e} bits)", flush=True)

    report = {
        "protocol": "residualmem_closed_loop_rate_v1",
        "cache": str(Path(args.cache).resolve()),
        "cache_manifest_sha256": sha256_file(Path(args.cache) / CACHE_FILES["manifest"]),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "mask": str(Path(args.mask).resolve()),
        "mask_sha256": sha256_file(args.mask),
        "mask_fitted_on": mask_meta.get("fitted_on_question_distribution"),
        "variant": variant,
        "split": args.split,
        "transitions": int(len(rows)),
        "all_send_null_control": bool(args.all_send),
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
