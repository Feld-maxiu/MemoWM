"""What does quantisation cost the frozen downstream head?

The C sweep trades reconstruction for rate: R^2 0.9909 / 0.9881 / 0.9767 at
8,224 / 6,176 / 4,128 bits. Whether that matters is not answerable from R^2 --
it depends on whether the discarded variance is the part the downstream reads.

This runs the bridge cache's own ``xbar`` through encode -> decode for each
codebook and writes a cache with the reconstruction in place of the original, so
``head_recall`` can score the frozen retrieval head on it. Nothing is retrained:
the head, the Q-Former and the codebooks are all fixed, and the only thing that
changes between runs is how many bits the state was squeezed through.

The cache and the codebooks come from the same Q-Former checkpoint
(``qformer-K32e-obs0.5.gapbest.pt``, sha c9068916dce44ce0), and the cache's
states are the WorldMemArena fit corpus that supplied the codebook's 10% mix --
so this is in-distribution for the quantiser, not a transfer test.

``rotate``/``unrotate`` and ``encode``/``decode`` are imported rather than
reimplemented: ``qformer_pq`` rotates with ``states @ bases.T``, and a
reimplementation that used ``@ bases`` round-tripped exactly (the basis is
orthonormal) while collapsing quantised reconstruction from R^2 0.991 to 0.106.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .qformer_pq import decode, encode


def reconstruct(states: np.ndarray, artifact: Path) -> tuple[np.ndarray, dict]:
    """Quantise and dequantise, returning the reconstruction and its error."""
    with np.load(artifact, allow_pickle=True) as data:
        mean, scale = np.asarray(data["mean"]), np.asarray(data["scale"])
        centroids = np.asarray(data["centroids"])
        bases = np.asarray(data["rotation_bases"]) if "rotation_bases" in data else None
        order = np.asarray(data["rotation_order"]) if "rotation_order" in data else None
    codes = encode(states, mean, scale, centroids, bases=bases, order=order)
    rebuilt = decode(codes, mean, scale, centroids, bases=bases, order=order)
    centre = states.mean(axis=0, keepdims=True)
    residual = float(np.square(states - rebuilt, dtype=np.float64).sum())
    spread = float(np.square(states - centre, dtype=np.float64).sum())
    return rebuilt, {
        "categories": int(centroids.shape[2]),
        "subspaces": int(centroids.shape[1]),
        "fixed_width_bits": int(
            centroids.shape[0] * centroids.shape[1] * np.log2(centroids.shape[2])
        ),
        "r2": 1.0 - residual / spread,
        "rmse": float(np.sqrt(residual / states.size)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True,
                        help="a build_qformer_bridge_cache artifact")
    parser.add_argument("--artifact", type=Path, action="append", required=True,
                        help="a qformer_pq codebook; repeatable")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=True) as data:
        contents = {name: np.asarray(data[name]) for name in data.files}
    states = np.asarray(contents["xbar"], np.float32)
    print(f"  cache {args.cache.name}: {states.shape}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for artifact in args.artifact:
        rebuilt, stats = reconstruct(states, artifact)
        name = f"{args.cache.stem}-C{stats['categories']}.npz"
        np.savez_compressed(args.output_dir / name,
                            **{**contents, "xbar": rebuilt.astype(np.float32)})
        stats["cache"] = str((args.output_dir / name).resolve())
        stats["artifact"] = str(artifact.resolve())
        report.append(stats)
        print(f"  C={stats['categories']:>4}  {stats['fixed_width_bits']:>5} bit  "
              f"R2 {stats['r2']:.4f}  -> {name}", flush=True)

    (args.output_dir / "reconstruction.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
