"""Score serializer variants against a fixed PCA basis.

P3.2 asks whether a pure formatting change moves WorldMemArena's Key64 closer to
the training distribution. The measurement cannot use the retrieval head: it has
collapsed on this domain -- output pairwise cosine 0.82 against 0.34 in domain --
so downstream cosine is nearly constant in its input and says almost nothing
about an upstream change.

What it uses instead is the *current* BrowserGym-only PCA as a fixed ruler. That
is exactly the right use for it here: ``r_perp`` measures how much of ``H`` falls
outside one fixed subspace, so a variant that lowers it is producing states that
sit closer to the space the training corpus spans. The ruler never moves between
variants, and it is deliberately not the basis anyone plans to ship.

Reference values for the v1 serializer, all 27 web samples:
``r_perp`` 0.4223, MMD^2 0.8598, nearest-neighbour median 14.60,
``P(|z|>3)`` 0.01786. Screening runs on a subset, so compare variants against the
v1 row measured on the *same* subset rather than against those numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.wma_pre_pca_localization import (
    inverse_pca_fidelity,
    load_indomain_key64,
    load_wma_key64,
    nn_median,
    out_of_subspace,
    random_projection,
    rbf_mmd2,
    subspace_projector,
    valid_rows,
)


def structural_stats(directory: Path) -> dict:
    """Node/candidate counts, from the serialized rows the extraction stored."""
    chars, obs = [], 0
    for file in sorted(directory.glob("*.npz")):
        with np.load(file, allow_pickle=False) as z:
            meta = json.loads(str(np.asarray(z["metadata"])))
        for record in meta["records"]:
            chars.append(record.get("synthetic_axtree_chars", 0))
            obs += 1
    return {
        "observations": obs,
        "axtree_chars_median": float(np.median(chars)) if chars else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", required=True, help="parent of per-variant dirs")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--pca", required=True, help="the fixed ruler; not refitted here")
    parser.add_argument("--train-states", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    with np.load(args.pca, allow_pickle=False) as artifact:
        mean = np.asarray(artifact["mean"], np.float64)
        components = np.asarray(artifact["components"], np.float64)
    projector = subspace_projector(components)

    cache = np.load(args.cache, allow_pickle=True)
    split, gidx = cache["split"], cache["global_indices"]
    features = Path(args.features)
    h_tr, v_tr = load_indomain_key64(features, gidx[split == "train"][: args.train_states].tolist())
    h_va, v_va = load_indomain_key64(features, gidx[split == "validation"].tolist())
    train_rows = valid_rows(h_tr, v_tr)
    val_rows = valid_rows(h_va, v_va)
    basis = random_projection(4096, 256, args.seed)
    train_proj = train_rows @ basis
    print(f"[screen] in-domain reference: train={len(train_rows)} val={len(val_rows)} slot rows")

    yard = {
        "mmd2": rbf_mmd2(train_proj, val_rows @ basis, rng=rng),
        "nn_median": nn_median(val_rows @ basis, train_proj, rng=rng),
        **out_of_subspace(val_rows, mean, projector),
        **inverse_pca_fidelity(val_rows, mean, projector),
    }

    report = {"protocol": "wma_serializer_screen_v1", "yardstick_validation": yard, "variants": {}}
    for variant in sorted(p for p in Path(args.screen_dir).iterdir() if p.is_dir()):
        try:
            h, v = load_wma_key64(variant)
        except FileNotFoundError:
            print(f"[screen] {variant.name}: no key64, skipped")
            continue
        rows = valid_rows(h, v)
        entry = {
            **structural_stats(variant),
            **out_of_subspace(rows, mean, projector),
            **inverse_pca_fidelity(rows, mean, projector),
            "mmd2": rbf_mmd2(train_proj, rows @ basis, rng=rng),
            "nn_median": nn_median(rows @ basis, train_proj, rng=rng),
            "tail_gt3_xbar": None,
        }
        report["variants"][variant.name] = entry
        print(f"[screen] {variant.name}: r_perp={entry['r_perp']:.4f} "
              f"R2={entry['r2']:.4f} MMD2={entry['mmd2']:.4f} NN={entry['nn_median']:.3f}",
              flush=True)

    print("\n" + "=" * 92)
    print(f"{'variant':26s}{'r_perp':>10s}{'R^2':>9s}{'MMD^2':>10s}{'NN med':>9s}"
          f"{'axtree chars':>14s}{'obs':>7s}")
    print("-" * 92)
    print(f"{'[yardstick] in-domain val':26s}{yard['r_perp']:10.4f}{yard['r2']:9.4f}"
          f"{yard['mmd2']:10.4f}{yard['nn_median']:9.3f}{'':>14s}{'':>7s}")
    for name, e in report["variants"].items():
        print(f"{name:26s}{e['r_perp']:10.4f}{e['r2']:9.4f}{e['mmd2']:10.4f}"
              f"{e['nn_median']:9.3f}{e['axtree_chars_median']:14.0f}{e['observations']:7d}")
    print("=" * 92)
    print("lower r_perp / MMD^2 / NN and higher R^2 are better; the yardstick is what")
    print("same-distribution looks like against this ruler.")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
