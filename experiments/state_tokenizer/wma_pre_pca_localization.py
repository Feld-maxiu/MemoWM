"""P3.0: localize the shift *before* PCA, on the raw layer-16 Key64.

Everything measured so far lives on ``xbar`` -- after PCA and after
normalization -- so it cannot separate two diagnoses whose repair costs differ by
an order of magnitude:

  (A) the frozen Qwen/Key64 representation itself does not transfer, or
  (B) Key64 still carries the information and it is the 20k-BrowserGym PCA basis
      that does not span WorldMemArena.

If (B), refitting the projection is the fix and the representation is fine. If
(A), refitting only relieves symptoms. The four measurements below are chosen so
each one discriminates between them.

**1. Out-of-subspace residual.** With ``P`` the 4096x512 components and ``mu``
the fitted mean,

    r_perp = || (H-mu) - (H-mu) P P^T ||^2 / || H-mu ||^2

reported for train / validation / WMA. ``r_perp_WMA >> r_perp_val`` says WMA's
variance largely lives outside the retained subspace -- the signature of (B).
Centering matters: the subspace is defined in centered coordinates, so an
uncentered residual would count the mean offset itself as "out of subspace" and
inflate every number.

**2. Pre-PCA distribution distance.** The same statistics D0 ran on ``xbar``, but
on ``H`` through a fixed random projection (Johnson-Lindenstrauss; exact MMD in
4096 dimensions is neither affordable nor better conditioned). The random basis
is shared across all three splits and seeded, so the comparison is like-for-like.
``H_WMA ~ H_ID`` together with ``xbar_WMA != xbar_ID`` is (B) stated directly.

**3. Inverse-PCA fidelity.** ``H -> PCA -> PCA^-1 -> H_hat``, scored by per-slot
cosine and by R^2. This is the metric closest to the actual memory claim: it asks
how much of WorldMemArena's original Qwen state the 512-d code still carries,
rather than how far apart two distributions are. A tokenizer that keeps a low
rate but drops the content it was supposed to preserve fails here and passes
every distance test.

**4. Effective rank**, per split and per slot group -- if WMA occupies far fewer
directions than the training data, the collapse seen downstream starts here.

In-domain ``H`` is read from the on-disk feature store (the 4096-d
``key64-static-bf16.npy``, hardlinked across both coordinate trees); the WMA side
comes from ``wma_extract_xbar.py --save-key64``.
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


def _decode_bf16(bits: np.ndarray) -> np.ndarray:
    """bf16 bit pattern -> float32, without torch.

    bfloat16 is exactly the top 16 bits of a float32 (1 sign, 8 exponent,
    7 mantissa), so widening and shifting is lossless and needs no framework.
    Keeping this numpy-only lets the analysis run in the jax environment, which
    has no torch, alongside the extraction environment which does.
    """
    widened = np.asarray(bits, np.uint16).astype(np.uint32) << 16
    return widened.view(np.float32)


def load_indomain_key64(features_root: Path, global_indices: list[int]):
    """Raw 4096-d Key64 for the given global indices, from the feature store."""
    locations: dict[int, tuple[Path, int]] = {}
    for worker in sorted(features_root.glob("worker*")):
        path = worker / "record_indices.npy"
        if not path.exists():
            continue
        for local, index in enumerate(np.load(path)):
            locations[int(index)] = (worker, local)
    missing = [i for i in global_indices if i not in locations]
    if missing:
        raise KeyError(f"feature store misses {len(missing)} indices, e.g. {missing[:5]}")
    opened: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    rows, masks = [], []
    for index in global_indices:
        worker, local = locations[index]
        if worker not in opened:
            opened[worker] = (
                np.load(worker / "key64-static-bf16.npy", mmap_mode="r"),
                np.load(worker / "key64-static-valid.npy", mmap_mode="r"),
            )
        bits, valid = opened[worker]
        rows.append(_decode_bf16(np.asarray(bits[local])))
        masks.append(np.asarray(valid[local], bool))
    return np.stack(rows), np.stack(masks)


def load_wma_key64(directory: Path):
    hs, vs = [], []
    for file in sorted(directory.glob("*.npz")):
        with np.load(file, allow_pickle=False) as z:
            keys = sorted(k for k in z.files if k.startswith("key64/"))
            if not keys:
                continue
            for key in keys:
                index = key.split("/", 1)[1]
                hs.append(_decode_bf16(z[key]))
                vs.append(np.asarray(z[f"m11/valid/{index}"], bool))
    if not hs:
        raise FileNotFoundError(
            f"no key64 arrays under {directory}; re-run extraction with --save-key64"
        )
    return np.stack(hs), np.stack(vs)


def valid_rows(h: np.ndarray, valid: np.ndarray, span=None) -> np.ndarray:
    """All valid slot vectors, flattened to (n_slots, 4096)."""
    if span is not None:
        lo, hi = span
        h, valid = h[:, lo:hi, :], valid[:, lo:hi]
    return h[valid].astype(np.float64)


def subspace_projector(components: np.ndarray) -> np.ndarray:
    """Exact orthogonal projector onto the column space of ``components``.

    The stored components are only near-orthonormal -- ``||P^T P - I||_max`` is
    about 2e-4, consistent with float32 storage -- so ``P P^T`` is not exactly a
    projector. ``P (P^T P)^-1 P^T`` is, whatever the conditioning, and costs one
    512x512 solve. The difference is far below the effects being measured, but
    getting it right removes the question rather than leaving a caveat.
    """
    gram = components.T @ components
    return components @ np.linalg.solve(gram, components.T)


def out_of_subspace(rows: np.ndarray, mean: np.ndarray, projector: np.ndarray) -> dict:
    centered = rows - mean
    residual = centered - centered @ projector
    num = float(np.sum(residual * residual))
    den = float(np.sum(centered * centered))
    per_slot = (residual * residual).sum(1) / np.maximum((centered * centered).sum(1), 1e-12)
    return {
        "r_perp": num / max(den, 1e-12),
        "r_perp_median_per_slot": float(np.median(per_slot)),
        "r_perp_p95_per_slot": float(np.percentile(per_slot, 95)),
        "n_slots": int(len(rows)),
    }


def inverse_pca_fidelity(rows: np.ndarray, mean: np.ndarray, projector: np.ndarray) -> dict:
    centered = rows - mean
    reconstructed = centered @ projector + mean
    dot = (rows * reconstructed).sum(1)
    norms = np.linalg.norm(rows, axis=1) * np.linalg.norm(reconstructed, axis=1)
    cos = dot / np.maximum(norms, 1e-12)
    sse = float(np.sum((rows - reconstructed) ** 2))
    sst = float(np.sum((rows - rows.mean(0)) ** 2))
    return {
        "cosine_mean": float(cos.mean()),
        "cosine_median": float(np.median(cos)),
        "cosine_p05": float(np.percentile(cos, 5)),
        "r2": 1.0 - sse / max(sst, 1e-12),
    }


def effective_rank(rows: np.ndarray, cap: int = 4000, rng=None) -> dict:
    """exp(entropy of the normalized spectrum) -- how many directions are in use."""
    if len(rows) > cap:
        rows = rows[rng.choice(len(rows), cap, replace=False)]
    centered = rows - rows.mean(0)
    singular = np.linalg.svd(centered, compute_uv=False)
    power = singular ** 2
    total = power.sum()
    if total <= 0:
        return {"effective_rank": 0.0, "n": int(len(rows))}
    p = power / total
    p = p[p > 0]
    return {
        "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
        "top1_share": float(p[0]),
        "top10_share": float(p[:10].sum()),
        "n": int(len(rows)),
    }


def random_projection(dim_in: int, dim_out: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    basis = rng.normal(size=(dim_in, dim_out)) / np.sqrt(dim_out)
    return basis


def rbf_mmd2(a, b, *, rng, cap=1500) -> float:
    if len(a) > cap:
        a = a[rng.choice(len(a), cap, replace=False)]
    if len(b) > cap:
        b = b[rng.choice(len(b), cap, replace=False)]

    def sqdist(x, y):
        return np.maximum((x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T, 0.0)

    pooled = np.vstack([a, b])
    d2 = sqdist(pooled, pooled)
    sigma2 = float(np.median(d2[np.triu_indices_from(d2, 1)])) or 1.0
    k = lambda x, y: np.exp(-sqdist(x, y) / sigma2)
    n, m = len(a), len(b)
    kaa, kbb = k(a, a), k(b, b)
    np.fill_diagonal(kaa, 0.0)
    np.fill_diagonal(kbb, 0.0)
    return float(kaa.sum() / (n * (n - 1)) + kbb.sum() / (m * (m - 1)) - 2 * k(a, b).mean())


def nn_median(query, reference, *, rng, cap=4000) -> float:
    if len(reference) > cap:
        reference = reference[rng.choice(len(reference), cap, replace=False)]
    best = np.empty(len(query))
    for start in range(0, len(query), 256):
        chunk = query[start:start + 256]
        d2 = ((chunk * chunk).sum(1)[:, None] + (reference * reference).sum(1)[None, :]
              - 2 * chunk @ reference.T)
        best[start:start + 256] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return float(np.median(best))


def describe_pre_pca(rows: np.ndarray, basis: np.ndarray, *, rng) -> dict:
    projected = rows @ basis
    norms = np.linalg.norm(rows, axis=1)
    unit = rows / np.maximum(norms[:, None], 1e-12)
    sample = unit[rng.choice(len(unit), min(len(unit), 600), replace=False)]
    pair = sample @ sample.T
    return {
        "norm_median": float(np.median(norms)),
        "norm_p05": float(np.percentile(norms, 5)),
        "norm_p95": float(np.percentile(norms, 95)),
        "pairwise_cosine_median": float(np.median(pair[np.triu_indices_from(pair, 1)])),
        **effective_rank(rows, rng=rng),
        "_projected": projected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--features", required=True, help="static_features root (4096-d store)")
    parser.add_argument("--pca", required=True)
    parser.add_argument("--wma-dir", required=True)
    parser.add_argument("--train-states", type=int, default=1000)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    with np.load(args.pca, allow_pickle=False) as artifact:
        mean = np.asarray(artifact["mean"], np.float64)
        components = np.asarray(artifact["components"], np.float64)
        evr = artifact["explained_variance_ratio"] if "explained_variance_ratio" in artifact.files else None
    if mean.shape != (4096,) or components.shape != (4096, 512):
        raise ValueError(f"unexpected PCA shapes {mean.shape} / {components.shape}")
    projector = subspace_projector(components)
    expected_train_r_perp = float(1.0 - np.asarray(evr).ravel().sum()) if evr is not None else None

    cache = np.load(args.cache, allow_pickle=True)
    split = cache["split"]
    gidx = cache["global_indices"]
    train_idx = gidx[split == "train"][: args.train_states].tolist()
    val_idx = gidx[split == "validation"].tolist()

    print(f"[pre-pca] loading in-domain Key64: train={len(train_idx)} val={len(val_idx)}")
    features = Path(args.features)
    h_train, v_train = load_indomain_key64(features, train_idx)
    h_val, v_val = load_indomain_key64(features, val_idx)
    print(f"[pre-pca] loading WMA Key64 from {args.wma_dir}")
    h_wma, v_wma = load_wma_key64(Path(args.wma_dir))
    print(f"[pre-pca] shapes train={h_train.shape} val={h_val.shape} wma={h_wma.shape}")

    basis = random_projection(4096, args.projection_dim, args.seed)
    report = {
        "protocol": "wma_pre_pca_localization_v1",
        "counts": {"train_states": len(train_idx), "validation_states": len(val_idx),
                   "wma_observations": int(len(h_wma))},
        "projection_dim": args.projection_dim,
        "by_group": {},
    }

    spans = {"all": None}
    spans.update({g: s for g, s in GROUP_SPANS.items() if s[1] > s[0]})
    for group, span in spans.items():
        rows = {
            "train": valid_rows(h_train, v_train, span),
            "validation": valid_rows(h_val, v_val, span),
            "wma": valid_rows(h_wma, v_wma, span),
        }
        if any(len(r) == 0 for r in rows.values()):
            continue
        entry = {"subspace": {}, "fidelity": {}, "distribution": {}}
        described = {}
        for name, r in rows.items():
            entry["subspace"][name] = out_of_subspace(r, mean, projector)
            entry["fidelity"][name] = inverse_pca_fidelity(r, mean, projector)
            described[name] = describe_pre_pca(r, basis, rng=rng)
        proj = {k: v.pop("_projected") for k, v in described.items()}
        entry["distribution"] = described
        entry["distances"] = {
            "yardstick_train_vs_validation": {
                "mmd2": rbf_mmd2(proj["train"], proj["validation"], rng=rng),
                "nn_median": nn_median(proj["validation"], proj["train"], rng=rng),
            },
            "wma_vs_train": {
                "mmd2": rbf_mmd2(proj["train"], proj["wma"], rng=rng),
                "nn_median": nn_median(proj["wma"], proj["train"], rng=rng),
            },
        }
        report["by_group"][group] = entry
        print(f"[pre-pca] {group}: r_perp train={entry['subspace']['train']['r_perp']:.4f} "
              f"val={entry['subspace']['validation']['r_perp']:.4f} "
              f"wma={entry['subspace']['wma']['r_perp']:.4f}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Free self-check: the PCA records its explained-variance ratio, so the
    # train-side residual must come out at 1 - EVR. If it does not, the loading,
    # the centering or the projector is wrong and no other number here is safe.
    if expected_train_r_perp is not None:
        measured = report["by_group"]["all"]["subspace"]["train"]["r_perp"]
        report["assumption_check"] = {
            "expected_train_r_perp_from_evr": expected_train_r_perp,
            "measured_train_r_perp": measured,
            "holds": bool(abs(measured - expected_train_r_perp) < 0.02),
        }
        print(f"[pre-pca] self-check: train r_perp measured {measured:.4f} "
              f"vs 1-EVR {expected_train_r_perp:.4f} -> "
              f"{'OK' if report['assumption_check']['holds'] else 'MISMATCH'}")

    allg = report["by_group"]["all"]
    print("\n" + "=" * 74)
    print(f"{'':22s}{'train':>16s}{'validation':>16s}{'WMA':>16s}")
    for label, getter in (
        ("r_perp (out-of-sub)", lambda s: allg["subspace"][s]["r_perp"]),
        ("inv-PCA cosine", lambda s: allg["fidelity"][s]["cosine_mean"]),
        ("inv-PCA R^2", lambda s: allg["fidelity"][s]["r2"]),
        ("effective rank", lambda s: allg["distribution"][s]["effective_rank"]),
        ("pairwise cosine", lambda s: allg["distribution"][s]["pairwise_cosine_median"]),
        ("norm median", lambda s: allg["distribution"][s]["norm_median"]),
    ):
        print(f"{label:22s}" + "".join(f"{getter(s):16.4f}"
                                       for s in ("train", "validation", "wma")))
    print("-" * 74)
    d = allg["distances"]
    print(f"{'MMD^2 (pre-PCA)':22s}{'':16s}"
          f"{d['yardstick_train_vs_validation']['mmd2']:16.4f}"
          f"{d['wma_vs_train']['mmd2']:16.4f}")
    print(f"{'NN median (pre-PCA)':22s}{'':16s}"
          f"{d['yardstick_train_vs_validation']['nn_median']:16.4f}"
          f"{d['wma_vs_train']['nn_median']:16.4f}")
    print("=" * 74)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
