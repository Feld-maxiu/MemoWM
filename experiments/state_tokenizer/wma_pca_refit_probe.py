"""Does refitting the PCA basis recover WorldMemArena coverage?

P3.0 established that the frozen Key64 transfers fine -- matched norms, comparable
effective rank and pairwise cosine -- while 42% of WMA's centered variance falls
outside the 512-d subspace fitted on 20k BrowserGym states, against 8% in domain.
That points at the basis rather than the representation, and this probe tests the
implication directly, before anything as expensive as rebuilding the PCA -> A1 ->
A2 chain is committed to.

Three fit corpora, evaluated on held-out data from both domains:

    train_only        the current recipe, refit here so corpus size is controlled
    train_plus_wma    half BrowserGym, half WorldMemArena
    wma_only          WorldMemArena alone

**The WMA split is by sample, not by observation.** Observations inside one
sample come from consecutive rounds of a single session and are heavily
correlated; splitting by observation would leak nearly-identical rows across the
fit/held-out boundary and report a recovery that does not exist.

Corpus size is held equal across variants so the comparison is about composition
rather than data volume, and the eigendecomposition runs on the 4096x4096
covariance rather than an SVD of the row matrix -- same subspace, a fraction of
the memory.

Reported per variant, on each held-out set: ``r_perp`` (variance outside the
subspace) and ``R^2`` (how much of the original Qwen state the code retains).
The in-domain column is the one that decides whether a domain-inclusive basis
costs anything where the pipeline already works.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.wma_pre_pca_localization import (
    _decode_bf16,
    load_indomain_key64,
)


def load_wma_by_sample(directory: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for file in sorted(directory.glob("*.npz")):
        with np.load(file, allow_pickle=False) as z:
            keys = sorted(k for k in z.files if k.startswith("key64/"))
            if not keys:
                continue
            hs = np.stack([_decode_bf16(z[k]) for k in keys])
            vs = np.stack([np.asarray(z[f"m11/valid/{k.split('/', 1)[1]}"], bool) for k in keys])
        out[file.stem] = (hs, vs)
    if not out:
        raise FileNotFoundError(f"no key64 under {directory}; extract with --save-key64")
    return out


def slot_rows(h: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return h[valid].astype(np.float64)


def subsample(rows: np.ndarray, n: int, rng) -> np.ndarray:
    if len(rows) <= n:
        return rows
    return rows[rng.choice(len(rows), n, replace=False)]


def fit_pca(rows: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and top-``dim`` eigenvectors of the covariance."""
    mean = rows.mean(0)
    centered = rows - mean
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:dim]
    return mean, np.ascontiguousarray(vectors[:, order])


def evaluate(rows: np.ndarray, mean: np.ndarray, components: np.ndarray) -> dict:
    centered = rows - mean
    coeffs = centered @ components
    reconstructed = coeffs @ components.T
    residual = centered - reconstructed
    r_perp = float(np.sum(residual ** 2) / max(np.sum(centered ** 2), 1e-12))
    restored = reconstructed + mean
    sse = float(np.sum((rows - restored) ** 2))
    sst = float(np.sum((rows - rows.mean(0)) ** 2))
    dot = (rows * restored).sum(1)
    norms = np.linalg.norm(rows, axis=1) * np.linalg.norm(restored, axis=1)
    return {
        "r_perp": r_perp,
        "r2": 1.0 - sse / max(sst, 1e-12),
        "cosine_mean": float((dot / np.maximum(norms, 1e-12)).mean()),
        "n_slots": int(len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--wma-dir", required=True)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--fit-rows", type=int, default=20000)
    parser.add_argument("--train-states", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cache = np.load(args.cache, allow_pickle=True)
    split, gidx = cache["split"], cache["global_indices"]
    train_idx = gidx[split == "train"][: args.train_states].tolist()
    val_idx = gidx[split == "validation"].tolist()

    features = Path(args.features)
    print(f"[refit] in-domain Key64: train={len(train_idx)} val={len(val_idx)}")
    h_tr, v_tr = load_indomain_key64(features, train_idx)
    h_va, v_va = load_indomain_key64(features, val_idx)
    train_rows = slot_rows(h_tr, v_tr)
    val_rows = slot_rows(h_va, v_va)
    del h_tr, v_tr, h_va, v_va

    print(f"[refit] WMA Key64 from {args.wma_dir}")
    per_sample = load_wma_by_sample(Path(args.wma_dir))
    names = sorted(per_sample)
    half = len(names) // 2
    fit_names, held_names = names[:half], names[half:]
    wma_fit = np.concatenate([slot_rows(*per_sample[n]) for n in fit_names])
    wma_held = np.concatenate([slot_rows(*per_sample[n]) for n in held_names])
    del per_sample
    print(f"[refit] WMA split by sample: fit={len(fit_names)} held-out={len(held_names)}")
    print(f"[refit] slot rows: train={len(train_rows)} val={len(val_rows)} "
          f"wma_fit={len(wma_fit)} wma_held={len(wma_held)}")

    n = args.fit_rows
    corpora = {
        "train_only": [subsample(train_rows, n, rng)],
        "train_plus_wma": [subsample(train_rows, n // 2, rng), subsample(wma_fit, n // 2, rng)],
        "wma_only": [subsample(wma_fit, n, rng)],
    }

    report = {
        "protocol": "wma_pca_refit_probe_v1",
        "dim": args.dim,
        "fit_rows_target": n,
        "wma_fit_samples": fit_names,
        "wma_heldout_samples": held_names,
        "variants": {},
    }
    held_out = {"indomain_validation": val_rows, "wma_heldout": wma_held}

    for variant, parts in corpora.items():
        rows = np.concatenate(parts)
        mean, components = fit_pca(rows, args.dim)
        entry = {"fit_rows": int(len(rows)),
                 "evaluations": {k: evaluate(v, mean, components) for k, v in held_out.items()}}
        entry["evaluations"]["fit_corpus"] = evaluate(rows, mean, components)
        report["variants"][variant] = entry
        e = entry["evaluations"]
        print(f"[refit] {variant:16s} rows={len(rows):6d}  "
              f"ID r_perp={e['indomain_validation']['r_perp']:.4f} R2={e['indomain_validation']['r2']:.4f}  |  "
              f"WMA r_perp={e['wma_heldout']['r_perp']:.4f} R2={e['wma_heldout']['r2']:.4f}",
              flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"{'fit corpus':18s}{'ID r_perp':>12s}{'ID R^2':>10s}{'WMA r_perp':>13s}{'WMA R^2':>10s}")
    print("-" * 78)
    for variant, entry in report["variants"].items():
        e = entry["evaluations"]
        print(f"{variant:18s}"
              f"{e['indomain_validation']['r_perp']:12.4f}{e['indomain_validation']['r2']:10.4f}"
              f"{e['wma_heldout']['r_perp']:13.4f}{e['wma_heldout']['r2']:10.4f}")
    print("=" * 78)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
