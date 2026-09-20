"""Turn the utility gate from a measurement into a shippable artifact.

``global_mask_curve`` computes the 1024-dimensional utility vector, applies it at
a dozen thresholds, records what each cost, and throws the vector away. Its
output is a table of per-row measurements; nothing on disk carries the gate
itself, so nothing downstream can use it. This writes the vector out.

**What ships is the utility vector and lambda, not a boolean mask.** The decision
is per state -- ``drop j`` when ``|U_j| < lambda * H_j`` -- and ``H_j`` is that
state's posterior entropy, which the decoder computes from ``h`` and ``u``. Only
the 1024 constants and the scalar are shared, so the gate costs **zero mask
bits**. Shipping a boolean mask instead would throw away the per-state part and
still cost nothing, which is strictly worse.

Entropy rather than the code length ``-log2 p(z+_j)``: the code length is a
function of the very symbol being decided about, so the decoder cannot have it.

☠️ **The rate accounting is the subtle part and it is not what the curve files
hold.** Those record one row per *question*, so a state carrying four questions
is counted four times; read that way the same configuration reports 4419.4 bits
against a 5313.9 full-send. The rate is a property of the corpus, not of the
question set: every state is stored once, whether or not anyone asks about it.
Averaged per state over all of them, full-send reproduces the world model's own
``code_bits_per_transition`` and the gate reads 4312.0. :func:`per_state_rate` is
that accounting; use it rather than averaging the curve files.

☠️ **The mask is bound to the question distribution it was fitted on.**
``UTILITY_GATE.md`` §7.4: fitting on generated questions made it *worse* on WMA
questions. The fitting distribution is recorded in the npz metadata so the
limitation travels with the artifact instead of living only in a document.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.utility_gate.groups import NUM_POSITIONS, PROTOCOL

MASK_PROTOCOL = "residualmem_utility_gate_mask_v1"


def utility_vector(label_paths: list[str]) -> tuple[np.ndarray, int]:
    """Per-position mean ``|delta_nll_bits|`` over every label row.

    Magnitude, not signed utility, and that is a correction rather than a
    convenience: signed utility rewards destroying memory, because 42% of
    positions carry a *negative* signed contribution -- the quantized code is
    sometimes misleading -- so ranking by it selects for damage. Magnitude asks
    "does this position move the answer at all", which is what a gate needs.
    """
    positions, utilities = [], []
    for path in label_paths:
        with np.load(path, allow_pickle=True) as data:
            positions.append(np.asarray(data["position"]))
            utilities.append(np.asarray(data["delta_nll_bits"], np.float64))
    position = np.concatenate(positions)
    utility = np.abs(np.concatenate(utilities))

    vector = np.zeros(NUM_POSITIONS, np.float64)
    for index in range(NUM_POSITIONS):
        block = utility[position == index]
        vector[index] = block.mean() if len(block) else 0.0
    return vector, len(utility)


def keep_mask(utility: np.ndarray, entropy: np.ndarray, lam: float) -> np.ndarray:
    """``keep j`` iff ``|U_j| >= lambda * H_j``. Broadcasts over states.

    ``entropy`` is ``(states, 1024)`` or ``(1024,)``; the returned mask matches.
    This is the one definition of the gate -- import it, do not restate it.
    """
    utility = np.asarray(utility, np.float64).reshape(-1)
    entropy = np.asarray(entropy, np.float64)
    if entropy.ndim == 1:
        return utility >= lam * entropy
    return utility[None, :] >= lam * entropy.reshape(len(entropy), -1)


def ev_keep_mask(v: np.ndarray, edges: np.ndarray, rates: np.ndarray,
                 entropy: np.ndarray, lam: float) -> np.ndarray:
    """``keep j`` iff ``e(H_{t,j}) * V_j >= lambda * H_{t,j}`` -- the state-dependent gate.

    ``e`` is the frozen empirical error-rate table over the carrier statistic
    (posterior entropy, 0.25-bit bins) fitted by ``wmaexperiment.ev_gate_wma_fit``;
    ``V_j`` is the per-position mean task damage conditioned on the world-model
    argmax being wrong.  Decomposes the fixed vector exactly:
    ``|U_j| = e_bar_j * V_j``.  Same shape contract as :func:`keep_mask`.
    """
    v = np.asarray(v, np.float64).reshape(-1)
    entropy = np.asarray(entropy, np.float64)
    edges = np.asarray(edges, np.float64)
    rates = np.asarray(rates, np.float64)
    flat = entropy.reshape(len(entropy), -1)
    idx = np.clip(np.digitize(flat.ravel(), edges) - 1, 0, len(rates) - 1)
    e = rates[idx].reshape(flat.shape)
    return e * v >= lam * flat


def per_state_rate(code_bits: np.ndarray, keep: np.ndarray | None = None) -> float:
    """Mean bits per state over the whole corpus, one count per state.

    Not per question. A state is written once regardless of how many questions
    later touch it, and states nobody asks about still have to be stored.
    """
    flat = np.asarray(code_bits, np.float64).reshape(len(code_bits), -1)
    if keep is None:
        return float(flat.sum(1).mean())
    # keep may arrive as (states, 32, 32) from a model output or (states, 1024)
    # from a curve script; flatten both rather than making callers remember.
    flags = np.asarray(keep, bool).reshape(len(flat), -1)
    if flags.shape != flat.shape:
        raise ValueError(
            f"keep {flags.shape} does not match code_bits {flat.shape}")
    return float((flat * flags).sum(1).mean())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", required=True,
                        help="label npz files the utility vector is fitted from")
    parser.add_argument("--lambda", dest="lam", type=float, default=0.0010,
                        help="operating point; 0.0010 saves 16.8%% for 0.103 bits")
    parser.add_argument("--posteriors", default=None,
                        help="posterior dump to self-check the rate against")
    parser.add_argument("--expect-full-rate", type=float, default=None,
                        help="the world model's code_bits_per_transition; the "
                             "full-send rate must reproduce it")
    parser.add_argument("--expect-keep", type=float, default=None,
                        help="published keep fraction, e.g. 0.836")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="absolute bits. Not bitwise: code_bits is stored "
                             "float32, so the corpus mean lands ~1e-5 off a "
                             "float64 reference")
    parser.add_argument("--fitted-on", default="wma-fitcorpus-questions",
                        help="the question distribution the labels came from; "
                             "the mask is bound to it (UTILITY_GATE.md 7.4)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    utility, num_labels = utility_vector(args.labels)
    covered = int((utility > 0).sum())
    print(f"[mask] {num_labels} labels -> {covered}/{NUM_POSITIONS} positions covered",
          flush=True)
    if covered < NUM_POSITIONS:
        # A position with no label gets utility 0 and is therefore dropped at
        # every lambda. That is a silent, permanent decision made by absence of
        # data rather than by measurement, so it must not pass unremarked.
        print(f"[mask] ☠️ {NUM_POSITIONS - covered} positions have no label and "
              f"will be dropped at every lambda", flush=True)

    check: dict[str, float | str] = {}
    if args.posteriors:
        with np.load(args.posteriors, allow_pickle=True) as data:
            code_bits = np.asarray(data["code_bits"], np.float64)
            entropy = np.asarray(data["entropy_bits"], np.float64)
        states = len(code_bits)
        full = per_state_rate(code_bits)
        keep = keep_mask(utility, entropy, args.lam)
        gated = per_state_rate(code_bits, keep)
        check = {
            "states": states,
            "full_rate_bits": full,
            "gated_rate_bits": gated,
            "keep_fraction": float(keep.mean()),
            "saved_fraction": float(1.0 - gated / full),
            "compression_vs_fixed_width": float(6144.0 / gated),
        }
        print(f"[mask] {states} states  full {full:.2f} -> gated {gated:.2f} bits "
              f"({check['saved_fraction']:.2%} saved, keep {keep.mean():.4f}, "
              f"{check['compression_vs_fixed_width']:.3f}x)", flush=True)

        if args.expect_full_rate is not None:
            delta = abs(full - args.expect_full_rate)
            if delta > args.tolerance:
                raise SystemExit(
                    f"full-send rate {full:.6f} does not reproduce the world "
                    f"model's {args.expect_full_rate:.6f} (off by {delta:.2e}). "
                    f"The posteriors and the checkpoint disagree; do not ship "
                    f"this mask.")
            print(f"[mask] full-send reproduces {args.expect_full_rate} "
                  f"(off by {delta:.2e})", flush=True)
        if args.expect_keep is not None and abs(keep.mean() - args.expect_keep) > 1e-3:
            raise SystemExit(
                f"keep fraction {keep.mean():.4f} != published "
                f"{args.expect_keep:.4f}; the label set or lambda has changed")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        utility=utility,
        lam=np.float64(args.lam),
        metadata=json.dumps({
            "protocol": MASK_PROTOCOL,
            "groups_protocol": PROTOCOL,
            "rule": "drop j iff |U_j| < lambda * H_j, H from the WM posterior",
            "lambda": args.lam,
            "mask_bits": 0,
            "why_zero_mask_bits":
                "utility and lambda ship with the model; H_j is recomputed "
                "decoder-side from h and u, so no per-state decision is "
                "transmitted",
            "utility_is": "per-position mean |delta_nll_bits|, magnitude not signed",
            "fitted_on_question_distribution": args.fitted_on,
            "question_binding_warning":
                "the mask is bound to the question distribution above; fitting "
                "it on generated questions measurably worsened it on WMA "
                "questions (UTILITY_GATE.md 7.4)",
            "labels": [
                {"path": str(p), "sha256": _sha256(Path(p))} for p in args.labels],
            "num_labels": num_labels,
            "positions_covered": covered,
            "self_check": check,
        }, ensure_ascii=False),
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
