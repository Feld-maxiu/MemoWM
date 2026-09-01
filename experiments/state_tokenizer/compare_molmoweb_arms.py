"""Compare MolmoWeb's two text-modality arms against WorldMemArena's states.

The question is which arm puts MolmoWeb's Q-Former states closer to the
distribution the resampler will actually meet at evaluation time. All three sets
come from one ``QFormerInstructTokenizer.encode`` call on one checkpoint, so a
difference here is a difference in the observations.

Two things are deliberately *not* reported.

``tail_gt3`` -- ``wma_latent_coverage`` can read it as "mass outside three
training sigmas" only because the pooled ``xbar`` is the output of
``GroupChannelNormalizer`` and is therefore already a z-score in training
coordinates. The Q-Former path has neither PCA nor that normalizer
(``qformer_runtime``'s docstring is explicit), so the same number would have no
such meaning here.

QA-C -- on ``agent/gui/web`` it cannot separate latent representations at all:
the memory pool is 25-28 entries against ``top_k=10``, so every arm lands in
0.5936-0.5985 and all pairwise McNemar tests are ns. Whatever this script
concludes has to come from the representation, not from the answer metric.

Absolute distances are unreadable alone, so WorldMemArena is split in half and
the half-vs-half value is reported as the same-distribution yardstick.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .wma_latent_coverage import nn_distance, rbf_mmd2


def spread(states: np.ndarray) -> dict[str, float]:
    """``train_qformer_joint.spread``, reproduced for states loaded from disk.

    Mean-centred before anything else: the raw cosine of a learned resampler's
    output reads near 1.0 from a shared offset and looks like total collapse
    when it is not.
    """
    flat = states.reshape(len(states), -1).astype(np.float64)
    centred = flat - flat.mean(0, keepdims=True)
    unit = centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-12)
    gram = unit @ unit.T
    upper = gram[np.triu_indices(len(unit), 1)]
    # ``np.linalg.svdvals`` only exists on numpy >= 2.0, and this repo's three
    # environments do not agree on that. The two-argument form is identical and
    # portable.
    singular = np.linalg.svd(centred, compute_uv=False)
    share = singular / max(singular.sum(), 1e-12)
    share = share[share > 0]
    deviation = np.linalg.norm(centred, axis=1).mean()
    return {
        "pairwise_cosine": float(upper.mean()),
        "effective_rank": float(np.exp(-(share * np.log(share)).sum())),
        "mean_to_deviation": float(np.linalg.norm(flat.mean(0)) / max(deviation, 1e-12)),
    }


def _load(path: Path) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=True) as data:
        xbar = np.asarray(data["xbar"], np.float64)
        metadata = json.loads(str(np.asarray(data["metadata"])))
    return xbar, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a", type=Path, required=True)
    parser.add_argument("--arm-b", type=Path, required=True)
    parser.add_argument("--wma", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    arms = {}
    for name, path in (("arm-a", args.arm_a), ("arm-b", args.arm_b), ("wma", args.wma)):
        xbar, metadata = _load(path)
        arms[name] = xbar
        print(f"{name:6s} {xbar.shape}  checkpoint={Path(metadata['checkpoint']).name}")

    checkpoints = {json.loads(str(np.asarray(np.load(p, allow_pickle=True)["metadata"])))["checkpoint"]
                   for p in (args.arm_a, args.arm_b, args.wma)}
    if len(checkpoints) != 1:
        raise SystemExit(f"arms were encoded by different checkpoints: {checkpoints}")

    # One vector per observation. Every Q-Former slot is valid by construction,
    # so this is a plain mean rather than the masked pool the PCA path needs.
    pooled = {k: v.mean(axis=1) for k, v in arms.items()}

    wma = pooled["wma"]
    order = rng.permutation(len(wma))
    half = len(wma) // 2
    left, right = wma[order[:half]], wma[order[half:]]

    report = {
        "yardstick_wma_half_vs_half": {
            "mmd2": rbf_mmd2(left, right, rng=rng)["mmd2"],
            "nn": nn_distance(left, right, rng=rng),
        },
    }
    for name in ("arm-a", "arm-b"):
        report[name] = {
            "mmd2_vs_wma": rbf_mmd2(pooled[name], wma, rng=rng)["mmd2"],
            "nn_to_wma": nn_distance(pooled[name], wma, rng=rng),
            "spread": spread(arms[name]),
        }
    report["wma"] = {"spread": spread(arms["wma"])}

    scale = report["yardstick_wma_half_vs_half"]["mmd2"]
    yard_nn = report["yardstick_wma_half_vs_half"]["nn"]["median"]
    print(f"\n{'':22s}{'MMD^2 vs WMA':>15}{'x yardstick':>13}"
          f"{'NN median':>12}{'x yardstick':>13}")
    print(f"{'同分布参照 (WMA 半vs半)':22s}{scale:>15.5f}{1.0:>13.1f}"
          f"{yard_nn:>12.3f}{1.0:>13.1f}")
    for name in ("arm-a", "arm-b"):
        entry = report[name]
        print(f"{name:22s}{entry['mmd2_vs_wma']:>15.5f}"
              f"{entry['mmd2_vs_wma'] / max(scale, 1e-12):>13.1f}"
              f"{entry['nn_to_wma']['median']:>12.3f}"
              f"{entry['nn_to_wma']['median'] / max(yard_nn, 1e-12):>13.1f}")

    print(f"\n{'':22s}{'pairwise cos':>14}{'eff rank':>11}{'mean/dev':>11}")
    for name in ("arm-a", "arm-b", "wma"):
        s = report[name]["spread"]
        print(f"{name:22s}{s['pairwise_cosine']:>14.4f}"
              f"{s['effective_rank']:>11.1f}{s['mean_to_deviation']:>11.3f}")

    winner = min(("arm-a", "arm-b"), key=lambda k: report[k]["mmd2_vs_wma"])
    ratio = (report["arm-a"]["mmd2_vs_wma"] / max(report["arm-b"]["mmd2_vs_wma"], 1e-12))
    print(f"\n距离 WMA 更近: {winner}   (arm-a / arm-b 的 MMD^2 之比 = {ratio:.2f}x)")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
