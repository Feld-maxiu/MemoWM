"""Coordinate-provenance check for the WMA retrieval failure.

The WMA smoke report records no PCA hash, and the shell history / .tmp logs were
wiped by a session-container rebuild, so we cannot tell from records whether the
smoke ran under the valid task-balanced PCA (f98c517c...) or the invalidated
first-N PCA (6d2df9b5...).

This decides it empirically, in-domain, with no GPU forward and no teacher
encoding: run the SAME validation states through the trained head under both
coordinate systems.  If the old-PCA arm collapses to roughly the WMA smoke's
0.1836 paired cosine, then a coordinate mismatch reproduces the reported
"cross-domain failure" signature and the smoke must be re-run before any
domain-shift attribution work is done.

Read-only.  Writes nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.state_tokenizer.build_instruct_bridge_cache import load_static_features
from residualmem.encoders.normalization import GroupChannelNormalizer
from residualmem.latent.instruct_bridge import (
    RETRIEVAL_BRIDGE_PROTOCOL,
    MaskedAttentionRetrievalHead,
    load_bridge,
)

WMA_PAIRED_COSINE = 0.18355846405029297  # final-checkpoint smoke, 14 observations
INDOMAIN_EXPECTED = 0.8711105            # retrieval-head-fused-observation.json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_cosine(head, xbar, valid, teacher):
    with torch.no_grad():
        pred = head(torch.from_numpy(xbar), torch.from_numpy(valid))
    tea = torch.from_numpy(teacher)
    tea = tea / tea.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return (pred * tea).sum(-1).numpy()


def ranking(head, xbar, valid, teacher):
    """N-way retrieval of the matching teacher row, same protocol as the smoke."""
    with torch.no_grad():
        pred = head(torch.from_numpy(xbar), torch.from_numpy(valid))
    tea = torch.from_numpy(teacher)
    tea = tea / tea.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scores = pred @ tea.T                      # (N, N)
    n = scores.shape[0]
    order = torch.argsort(scores, dim=-1, descending=True)
    rank = (order == torch.arange(n)[:, None]).float().argmax(dim=-1) + 1
    return {
        "recall_at_1": float((rank <= 1).float().mean()),
        "recall_at_5": float((rank <= 5).float().mean()),
        "recall_at_10": float((rank <= 10).float().mean()),
        "mrr": float((1.0 / rank.float()).mean()),
        "n_way": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", required=True, help="bridge output dir (head + cache)")
    parser.add_argument("--reference-tree", required=True, help="tree holding the coordinates the head was trained on")
    parser.add_argument("--mismatch-tree", required=True, help="tree holding the invalidated coordinates")
    parser.add_argument("--output", default=None, help="optional JSON report path")
    args = parser.parse_args()

    bridge = Path(args.bridge)
    ref_tree = Path(args.reference_tree)
    bad_tree = Path(args.mismatch_tree)

    head = MaskedAttentionRetrievalHead()
    artifact = load_bridge(
        bridge / "retrieval-head-fused-observation.pt",
        head,
        expected_protocol=RETRIEVAL_BRIDGE_PROTOCOL,
    )
    head.eval()
    print(f"head loaded: {artifact.metadata.get('representation')!r}")

    cache = np.load(bridge / "train-cache-fused-observation.npz", allow_pickle=True)
    split = cache["split"]
    sel = split == "validation"
    gidx = cache["global_indices"][sel].tolist()
    teacher = cache["teacher_fused_embedding"][sel]
    xbar_cached = cache["xbar"][sel]
    valid_cached = cache["valid"][sel]
    print(f"validation rows: {len(gidx)}")

    results = {"pca_sha256": {}}
    for name, tree in (("reference", ref_tree), ("mismatch", bad_tree)):
        pca = tree / "key64-static-pca.npz"
        results["pca_sha256"][name] = _sha256(pca) if pca.exists() else None

    # Arm A: the cache's own xbar.  Must reproduce the recorded in-domain cosine.
    cos = paired_cosine(head, xbar_cached, valid_cached, teacher)
    results["cached"] = {
        "paired_cosine_mean": float(cos.mean()),
        **ranking(head, xbar_cached, valid_cached, teacher),
    }

    # Arm B: rebuild from the reference tree.  Sanity: must match arm A exactly.
    x_t, valid_ref = load_static_features(ref_tree / "static_features", gidx)
    xbar_ref = GroupChannelNormalizer.from_npz(
        ref_tree / "key64-static-pca-normalization.npz"
    ).normalize(x_t, valid_ref).astype(np.float32)
    cos = paired_cosine(head, xbar_ref, valid_ref, teacher)
    results["rebuilt_reference"] = {
        "paired_cosine_mean": float(cos.mean()),
        "max_abs_diff_vs_cached": float(np.abs(xbar_ref - xbar_cached).max()),
        **ranking(head, xbar_ref, valid_ref, teacher),
    }

    # Arm C: the SAME states under the invalidated coordinates.
    x_t_bad, valid_bad = load_static_features(bad_tree / "static_features", gidx)
    xbar_bad = GroupChannelNormalizer.from_npz(
        bad_tree / "key64-static-pca-normalization.npz"
    ).normalize(x_t_bad, valid_bad).astype(np.float32)
    cos = paired_cosine(head, xbar_bad, valid_bad, teacher)
    results["coordinate_mismatch"] = {
        "paired_cosine_mean": float(cos.mean()),
        **ranking(head, xbar_bad, valid_bad, teacher),
    }

    print(json.dumps(results, indent=2))

    new_cos = results["cached"]["paired_cosine_mean"]
    old_cos = results["coordinate_mismatch"]["paired_cosine_mean"]
    print("\n" + "=" * 68)
    print(f"in-domain, correct coordinates : {new_cos:.4f}  (expected {INDOMAIN_EXPECTED:.4f})")
    print(f"in-domain, mismatched coords   : {old_cos:.4f}")
    print(f"WMA cross-domain smoke         : {WMA_PAIRED_COSINE:.4f}")
    print("=" * 68)
    if abs(new_cos - INDOMAIN_EXPECTED) > 0.01:
        print("!! arm A does not reproduce the recorded in-domain cosine -- fix the")
        print("!! harness before trusting anything else here.")
    elif abs(old_cos - WMA_PAIRED_COSINE) < 0.08:
        print(">> A coordinate swap alone reproduces the WMA failure signature.")
        print(">> The smoke's PCA provenance is unrecorded, so the 0/100 result")
        print(">> CANNOT be attributed to domain shift until the smoke is re-run.")
    else:
        print(">> A coordinate swap does NOT reproduce the WMA signature; the")
        print(">> cross-domain failure is not explained by a PCA mixup.")
        print(">> Arm C is the calibration anchor for D0/D1: this is what")
        print(">> 'in-domain data, wrong coordinates' looks like.")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
