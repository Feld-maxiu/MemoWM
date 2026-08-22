"""D0: does WorldMemArena ``xbar`` land inside the training support at all?

Every existing measurement of the cross-domain failure looks at the *output* of
the retrieval head, which cannot separate two very different diagnoses:

  * the tokenizer/PCA already encode WMA observations outside anything the
    training distribution covers -- the head's failure is then a symptom, and
    retraining it is wasted effort; or
  * ``xbar`` is well inside the training support and the head simply does not
    generalize there -- which is fixable by retraining the head alone, at a
    fraction of the cost of redoing the PCA -> A1 -> A2 chain.

This measures the input side only. No teacher, no head, no GPU.

The z-scores need no separate normalizer: ``xbar`` is the output of
``GroupChannelNormalizer``, i.e. already ``(x - mu) / sigma`` with ``mu, sigma``
fitted on the train split alone. So ``xbar`` *is* the z-score in training
coordinates and ``P(|xbar| > 3)`` is directly the mass outside three training
sigmas. The script asserts this by checking in-domain train marginals.

Absolute distribution distances mean nothing on their own, so every WMA number
is reported beside the train-vs-validation value for the same statistic. That
pair is the "what does same-distribution look like" yardstick; without it an MMD
of 0.03 is unreadable. A third reference is available via ``--anchor``: in-domain
states projected through the *invalidated* PCA, which is a known-bad point
(paired cosine 0.3919, 500-way R@1 0.004) produced by
``coordinate_provenance_check.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.slot_layout import GROUP_NAMES, KEY64_LAYOUT

GROUP_SPANS = {}
_start = 0
for _name, _size in zip(GROUP_NAMES, KEY64_LAYOUT):
    GROUP_SPANS[_name] = (_start, _start + _size)
    _start += _size


def masked_pool(xbar: np.ndarray, valid: np.ndarray, span=None) -> np.ndarray:
    """Mean over valid slots, restricted to ``span``. Rows with none are dropped."""
    if span is not None:
        lo, hi = span
        xbar, valid = xbar[:, lo:hi, :], valid[:, lo:hi]
    counts = valid.sum(axis=1)
    keep = counts > 0
    weights = valid[keep].astype(np.float64)[..., None]
    pooled = (xbar[keep].astype(np.float64) * weights).sum(axis=1)
    return pooled / counts[keep][:, None]


def slot_values(xbar: np.ndarray, valid: np.ndarray, span=None) -> np.ndarray:
    """All channel values of all valid slots, flattened."""
    if span is not None:
        lo, hi = span
        xbar, valid = xbar[:, lo:hi, :], valid[:, lo:hi]
    return xbar[valid].astype(np.float64).ravel()


def tail_mass(values: np.ndarray, threshold: float = 3.0) -> float:
    return float(np.mean(np.abs(values) > threshold))


def rbf_mmd2(a: np.ndarray, b: np.ndarray, *, rng, cap: int = 1500) -> dict:
    """Unbiased MMD^2 with an RBF kernel, median-heuristic bandwidth.

    Subsampled to ``cap`` rows per side: the estimator is O(n^2) and the train
    split alone is 5,000 states. The bandwidth is taken from the pooled sample so
    both sides share one kernel.
    """
    if len(a) > cap:
        a = a[rng.choice(len(a), cap, replace=False)]
    if len(b) > cap:
        b = b[rng.choice(len(b), cap, replace=False)]

    def sqdist(x, y):
        return np.maximum(
            (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2.0 * x @ y.T, 0.0
        )

    pooled = np.vstack([a, b])
    d2 = sqdist(pooled, pooled)
    upper = d2[np.triu_indices_from(d2, k=1)]
    sigma2 = float(np.median(upper))
    if sigma2 <= 0:
        sigma2 = 1.0
    k = lambda x, y: np.exp(-sqdist(x, y) / sigma2)
    n, m = len(a), len(b)
    kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
    np.fill_diagonal(kaa, 0.0)
    np.fill_diagonal(kbb, 0.0)
    value = (
        kaa.sum() / (n * (n - 1))
        + kbb.sum() / (m * (m - 1))
        - 2.0 * kab.mean()
    )
    return {"mmd2": float(value), "bandwidth_sq": sigma2, "n_a": n, "n_b": m}


def nn_distance(query: np.ndarray, reference: np.ndarray, *, rng, cap: int = 4000) -> dict:
    """Distance from each query row to its nearest reference row."""
    if len(reference) > cap:
        reference = reference[rng.choice(len(reference), cap, replace=False)]
    best = np.empty(len(query))
    for start in range(0, len(query), 256):
        chunk = query[start:start + 256]
        d2 = (
            (chunk * chunk).sum(1)[:, None]
            + (reference * reference).sum(1)[None, :]
            - 2.0 * chunk @ reference.T
        )
        best[start:start + 256] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return {
        "median": float(np.median(best)),
        "p95": float(np.percentile(best, 95)),
        "mean": float(best.mean()),
        "n_query": int(len(query)),
        "n_reference": int(len(reference)),
    }


def describe(name: str, xbar: np.ndarray, valid: np.ndarray) -> dict:
    out = {"n_states": int(len(xbar)), "valid_slots_mean": float(valid.sum(1).mean())}
    for group, span in GROUP_SPANS.items():
        lo, hi = span
        if hi <= lo:
            continue
        vals = slot_values(xbar, valid, span)
        if vals.size == 0:
            out[group] = {"valid_slot_rows": 0}
            continue
        norms = np.linalg.norm(xbar[:, lo:hi, :].astype(np.float64), axis=-1)
        out[group] = {
            "valid_slot_rows": int(valid[:, lo:hi].sum()),
            "z_mean": float(vals.mean()),
            "z_std": float(vals.std()),
            "z_abs_p99": float(np.percentile(np.abs(vals), 99)),
            "tail_gt3": tail_mass(vals, 3.0),
            "tail_gt5": tail_mass(vals, 5.0),
            "slot_norm_median": float(np.median(norms[valid[:, lo:hi]])),
        }
    allv = slot_values(xbar, valid)
    out["overall"] = {
        "z_mean": float(allv.mean()),
        "z_std": float(allv.std()),
        "tail_gt3": tail_mass(allv, 3.0),
        "tail_gt5": tail_mass(allv, 5.0),
    }
    return out


def load_cache(path: Path):
    cache = np.load(path, allow_pickle=True)
    return cache["xbar"], cache["valid"], cache["split"]


def load_wma(directory: Path, arm: str):
    xs, vs, meta = [], [], []
    for file in sorted(directory.glob("*.npz")):
        z = np.load(file, allow_pickle=False)
        info = json.loads(str(np.asarray(z["metadata"])))
        keys = sorted(k for k in z.files if k.startswith(f"{arm}/xbar/"))
        if not keys:
            continue
        for key in keys:
            index = key.rsplit("/", 1)[1]
            xs.append(z[key])
            vs.append(z[f"{arm}/valid/{index}"])
        for record in info["records"]:
            record["sample_id"] = info["sample_id"]
            meta.append(record)
    if not xs:
        raise FileNotFoundError(f"no arm {arm!r} under {directory}")
    return np.stack(xs), np.stack(vs), meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="bridge cache npz (in-domain reference)")
    parser.add_argument("--wma-dir", required=True, help="directory of per-sample xbar npz")
    parser.add_argument("--arm", default="m11")
    parser.add_argument("--extra-arm", action="append", default=None,
                        help="additional arms to describe, e.g. m11_resampled")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    xbar, valid, split = load_cache(Path(args.cache))
    train = (xbar[split == "train"], valid[split == "train"])
    val = (xbar[split == "validation"], valid[split == "validation"])
    wma = load_wma(Path(args.wma_dir), args.arm)
    print(f"train={len(train[0])} validation={len(val[0])} wma[{args.arm}]={len(wma[0])}")

    report = {
        "protocol": "wma_latent_coverage_v1",
        "arm": args.arm,
        "marginals": {
            "train": describe("train", *train),
            "validation": describe("validation", *val),
            f"wma_{args.arm}": describe("wma", wma[0], wma[1]),
        },
    }
    for extra in args.extra_arm or []:
        other = load_wma(Path(args.wma_dir), extra)
        report["marginals"][f"wma_{extra}"] = describe(extra, other[0], other[1])

    # The normalizer was fitted on train, so train marginals must be ~N(0,1).
    # If they are not, xbar is not what this script assumes and every tail
    # number below is uninterpretable.
    tm = report["marginals"]["train"]["overall"]
    report["assumption_check"] = {
        "train_z_mean": tm["z_mean"],
        "train_z_std": tm["z_std"],
        "train_tail_gt3": tm["tail_gt3"],
        "holds": bool(abs(tm["z_mean"]) < 0.05 and abs(tm["z_std"] - 1.0) < 0.10),
    }

    # Distances, per group and on the whole-state pooled vector. Every WMA entry
    # is paired with train-vs-validation for the same statistic.
    distances = {}
    spaces = {"all": None}
    spaces.update({g: s for g, s in GROUP_SPANS.items() if s[1] > s[0]})
    for space, span in spaces.items():
        tr = masked_pool(*train, span)
        va = masked_pool(*val, span)
        wm = masked_pool(wma[0], wma[1], span)
        distances[space] = {
            "yardstick_train_vs_validation": {
                "mmd": rbf_mmd2(tr, va, rng=rng),
                "nn": nn_distance(va, tr, rng=rng),
            },
            f"wma_{args.arm}_vs_train": {
                "mmd": rbf_mmd2(tr, wm, rng=rng),
                "nn": nn_distance(wm, tr, rng=rng),
            },
        }
    report["distances"] = distances

    print(json.dumps({
        "assumption_check": report["assumption_check"],
        "tail_gt3": {
            k: v["overall"]["tail_gt3"] for k, v in report["marginals"].items()
        },
        "mmd_all": {
            "train_vs_validation": distances["all"]["yardstick_train_vs_validation"]["mmd"]["mmd2"],
            f"wma_vs_train": distances["all"][f"wma_{args.arm}_vs_train"]["mmd"]["mmd2"],
        },
        "nn_median_all": {
            "validation_to_train": distances["all"]["yardstick_train_vs_validation"]["nn"]["median"],
            "wma_to_train": distances["all"][f"wma_{args.arm}_vs_train"]["nn"]["median"],
        },
    }, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
