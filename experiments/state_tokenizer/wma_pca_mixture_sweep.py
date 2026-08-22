"""Pick the BrowserGym:WorldMemArena mixing ratio for the refitted PCA basis.

``wma_pca_refit_probe.py`` answered a coarser question -- can a domain-inclusive
basis recover WMA coverage at all -- by splitting one WMA directory in half. Its
numbers are cited in the handover as ``wma_pca_refit_probe_v1`` and are left
alone. This script answers the deployment question under a stricter protocol:

* the fit corpus is WMA's **non-web** GUI subcategories, and
* the evaluation set is all 27 ``agent/gui/web`` samples, **never fitted on**.

So this is a cross-task generalization test rather than the probe's
same-distribution interpolation, and the recovery it reports should be read as a
lower bound on what the probe measured.

The headline criterion is ``r_perp`` **on the image slots**, not on all slots.
Image is 32 of the ~51 occupied slots on WMA and carries the bulk of the residual
shift (0.3841 against 0.0827 in domain); the serializer provably cannot touch it,
because image tokens precede the DOM in the sequence and causal attention keeps
their layer-16 states independent of it. An aggregate that mixes in the 3
context slots can move for reasons that have nothing to do with the visual
subspace, which is what actually needs fixing.

Corpus size is held fixed across ratios so the comparison is about composition
and not data volume. ``--subcategory-ablation`` refits at one ratio while
dropping each subcategory in turn, which is how a subcategory whose visual regime
differs from the eval domain (``mobile`` at 2.59 MP against web's 0.92 MP) gets
judged on evidence rather than on a guess.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.slot_layout import GROUP_NAMES, KEY64_LAYOUT
from experiments.state_tokenizer.wma_pre_pca_localization import (
    _decode_bf16,
    load_indomain_key64,
)

GROUP_SPANS = {}
_start = 0
for _name, _size in zip(GROUP_NAMES, KEY64_LAYOUT):
    if _size:
        GROUP_SPANS[_name] = (_start, _start + _size)
    _start += _size

_SAMPLE_SUFFIX = re.compile(r"_\d+$")


def subcategory_of(stem: str) -> str:
    """``webarena_lite_13`` -> ``webarena_lite``."""
    return _SAMPLE_SUFFIX.sub("", stem)


def slot_group_ids() -> np.ndarray:
    """Group index per slot, for the 64-slot layout."""
    ids = np.empty(64, np.int8)
    for index, (name, (lo, hi)) in enumerate(GROUP_SPANS.items()):
        ids[lo:hi] = index
    return ids


GROUP_ORDER = list(GROUP_SPANS)
SLOT_GROUPS = slot_group_ids()


def rows_and_groups(h: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten valid slots to rows, keeping each row's slot group.

    float32 throughout: the fit corpora run to tens of thousands of 4096-d rows
    and float64 would cost 4x the memory for no accuracy that survives the
    covariance eigendecomposition, which is done in float64 anyway.
    """
    rows = h[valid].astype(np.float32, copy=False)
    groups = np.broadcast_to(SLOT_GROUPS, valid.shape)[valid]
    return rows, groups.astype(np.int8, copy=False)


def load_wma_pool(directory: Path, per_sample_cap: int, rng) -> dict:
    """Per-subcategory slot rows, capped per sample so one long sample cannot
    dominate its subcategory."""
    pool: dict[str, list[np.ndarray]] = {}
    counts: dict[str, int] = {}
    for file in sorted(directory.glob("*.npz")):
        with np.load(file, allow_pickle=False) as z:
            keys = sorted(k for k in z.files if k.startswith("key64/"))
            if not keys:
                continue
            hs = np.stack([_decode_bf16(z[k]) for k in keys])
            vs = np.stack([np.asarray(z[f"m11/valid/{k.split('/', 1)[1]}"], bool) for k in keys])
        rows, _ = rows_and_groups(hs, vs)
        if per_sample_cap and len(rows) > per_sample_cap:
            rows = rows[rng.choice(len(rows), per_sample_cap, replace=False)]
        sub = subcategory_of(file.stem)
        pool.setdefault(sub, []).append(rows)
        counts[sub] = counts.get(sub, 0) + 1
    if not pool:
        raise FileNotFoundError(f"no key64 under {directory}; extract with --save-key64")
    return {sub: (np.concatenate(parts), counts[sub]) for sub, parts in pool.items()}


def load_wma_eval(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    rows_parts, group_parts = [], []
    for file in sorted(directory.glob("*.npz")):
        with np.load(file, allow_pickle=False) as z:
            keys = sorted(k for k in z.files if k.startswith("key64/"))
            if not keys:
                continue
            hs = np.stack([_decode_bf16(z[k]) for k in keys])
            vs = np.stack([np.asarray(z[f"m11/valid/{k.split('/', 1)[1]}"], bool) for k in keys])
        rows, groups = rows_and_groups(hs, vs)
        rows_parts.append(rows)
        group_parts.append(groups)
    if not rows_parts:
        raise FileNotFoundError(f"no key64 under {directory}; extract with --save-key64")
    return np.concatenate(rows_parts), np.concatenate(group_parts)


def subsample(rows: np.ndarray, n: int, rng) -> np.ndarray:
    if n <= 0 or len(rows) <= n:
        return rows
    return rows[rng.choice(len(rows), n, replace=False)]


def fit_pca(rows: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and top-``dim`` eigenvectors of the covariance, in float64."""
    rows64 = rows.astype(np.float64, copy=False)
    mean = rows64.mean(0)
    centered = rows64 - mean
    cov = (centered.T @ centered) / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:dim]
    return mean, np.ascontiguousarray(vectors[:, order])


def evaluate(rows: np.ndarray, mean: np.ndarray, components: np.ndarray,
             chunk: int = 8192) -> dict:
    """``r_perp`` and ``R^2``, accumulated in chunks so a 50k-row eval set fits."""
    grand_mean = rows.astype(np.float64).mean(0)
    res_sq = cen_sq = sse = sst = 0.0
    for start in range(0, len(rows), chunk):
        block = rows[start:start + chunk].astype(np.float64)
        centered = block - mean
        reconstructed = (centered @ components) @ components.T
        residual = centered - reconstructed
        res_sq += float(np.sum(residual ** 2))
        cen_sq += float(np.sum(centered ** 2))
        sse += float(np.sum(residual ** 2))
        sst += float(np.sum((block - grand_mean) ** 2))
    return {
        "r_perp": res_sq / max(cen_sq, 1e-12),
        "r2": 1.0 - sse / max(sst, 1e-12),
        "n_slots": int(len(rows)),
    }


def evaluate_by_group(rows: np.ndarray, groups: np.ndarray, mean, components) -> dict:
    out = {"all": evaluate(rows, mean, components)}
    for index, name in enumerate(GROUP_ORDER):
        mask = groups == index
        if mask.any():
            out[name] = evaluate(rows[mask], mean, components)
    return out


def build_corpus(train_rows, wma_rows, fraction: float, total: int, rng) -> np.ndarray:
    n_wma = int(round(total * fraction))
    parts = []
    if total - n_wma > 0:
        parts.append(subsample(train_rows, total - n_wma, rng))
    if n_wma > 0:
        parts.append(subsample(wma_rows, n_wma, rng))
    return np.concatenate(parts)


def _row(label: str, evals: dict) -> str:
    def cell(setname, group, key):
        value = evals[setname].get(group)
        return f"{value[key]:8.4f}" if value else f"{'-':>8s}"
    return (f"{label:22s}"
            f"{cell('indomain_validation','all','r_perp')}{cell('indomain_validation','all','r2')}"
            f"{cell('wma_web','image','r_perp')}{cell('wma_web','all','r_perp')}"
            f"{cell('wma_web','all','r2')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--features", required=True, help="in-domain static_features root")
    parser.add_argument("--fit-dir", required=True, help="WMA non-web fit corpus (key64)")
    parser.add_argument("--wma-eval-dir", required=True, help="WMA web 27, never fitted")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--fit-rows", type=int, default=20000)
    parser.add_argument("--train-states", type=int, default=1200)
    parser.add_argument("--per-sample-cap", type=int, default=400)
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 1.0],
                        help="WMA fraction of the fit corpus")
    parser.add_argument("--subcategory-ablation", type=float, default=None,
                        help="also refit at this ratio dropping each subcategory in turn")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cache = np.load(args.cache, allow_pickle=True)
    split, gidx = cache["split"], cache["global_indices"]
    train_idx = gidx[split == "train"][: args.train_states].tolist()
    val_idx = gidx[split == "validation"].tolist()

    features = Path(args.features)
    print(f"[sweep] in-domain Key64: train={len(train_idx)} val={len(val_idx)}", flush=True)
    h, v = load_indomain_key64(features, train_idx)
    train_rows, _ = rows_and_groups(h, v)
    del h, v
    h, v = load_indomain_key64(features, val_idx)
    val_rows, val_groups = rows_and_groups(h, v)
    del h, v

    print(f"[sweep] WMA fit corpus from {args.fit_dir}", flush=True)
    pool = load_wma_pool(Path(args.fit_dir), args.per_sample_cap, rng)
    for sub, (rows, n) in sorted(pool.items()):
        print(f"    {sub:18s} {n:3d} samples  {len(rows):6d} slot rows", flush=True)
    wma_fit_all = np.concatenate([rows for rows, _ in pool.values()])

    print(f"[sweep] WMA eval (held out) from {args.wma_eval_dir}", flush=True)
    web_rows, web_groups = load_wma_eval(Path(args.wma_eval_dir))
    print(f"[sweep] slot rows: train={len(train_rows)} val={len(val_rows)} "
          f"wma_fit={len(wma_fit_all)} wma_web={len(web_rows)}", flush=True)

    held = {"indomain_validation": (val_rows, val_groups), "wma_web": (web_rows, web_groups)}
    report = {
        "protocol": "wma_pca_mixture_sweep_v1",
        "dim": args.dim,
        "fit_rows_target": args.fit_rows,
        "fit_corpus_subcategories": {s: {"samples": n, "slot_rows": int(len(r))}
                                     for s, (r, n) in sorted(pool.items())},
        "eval_samples": sorted(p.stem for p in Path(args.wma_eval_dir).glob("*.npz")),
        "variants": {},
    }

    def run(label: str, corpus: np.ndarray) -> None:
        mean, components = fit_pca(corpus, args.dim)
        evals = {k: evaluate_by_group(r, g, mean, components) for k, (r, g) in held.items()}
        report["variants"][label] = {"fit_rows": int(len(corpus)), "evaluations": evals}
        print(_row(label, evals), flush=True)

    header = (f"{'variant':22s}{'ID r_perp':>8s}{'ID R2':>8s}"
              f"{'WEB img':>8s}{'WEB all':>8s}{'WEB R2':>8s}")
    print("\n" + header)
    print("-" * len(header))
    for fraction in args.ratios:
        run(f"wma={fraction:.2f}", build_corpus(train_rows, wma_fit_all, fraction,
                                                args.fit_rows, rng))

    if args.subcategory_ablation is not None:
        fraction = args.subcategory_ablation
        print(f"\n[sweep] subcategory ablation at wma={fraction:.2f} "
              f"(each row drops that subcategory from the WMA half)")
        print(header)
        print("-" * len(header))
        for dropped in sorted(pool):
            kept = np.concatenate([rows for sub, (rows, _) in pool.items() if sub != dropped])
            run(f"-{dropped}", build_corpus(train_rows, kept, fraction, args.fit_rows, rng))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nlower r_perp is better. WEB img is the headline: image is 32 of ~51")
    print(f"occupied slots and carries the residual shift the serializer cannot reach.")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
