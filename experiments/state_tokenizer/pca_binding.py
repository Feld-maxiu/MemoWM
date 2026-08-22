"""Three-way PCA binding check, with no torch dependency.

``outputs/state_tokenizer/`` holds two trees with identical directory structure
and identical file names::

    v9-instruct/                    key64-static-pca.npz  6d2df9b5...  INVALIDATED
    v9-instruct-pca20k-balanced/    key64-static-pca.npz  f98c517c...  official

The invalidated one is a 2,000-state first-N selection whose every state came
from ``click-button-v1``; the official one is task-balanced over 20,000 states.
Both must stay on disk -- the invalidated tree also holds the ~420 GiB Full-H --
so the trap is permanent, and picking the wrong one raises nothing at all. It
just silently answers with the wrong coordinates: routing in-domain validation
states through the mismatched tree drops the retrieval head's paired cosine from
0.8711 to 0.3919 and 500-way Recall@1 from 0.840 to 0.004 (chance is 0.002).
``coordinate_provenance_check.py`` reproduces those numbers on demand.

Every artifact already records the PCA it was built from -- ``pca transform``
writes ``<prefix>-pca-artifact.sha256`` into each shard, and ``fit_normalization``
stores ``pca_sha256`` inside the normalization npz. Nothing read either one, so
the recorded provenance could not stop a mixup. This module reads them.

This lives apart from ``key64_pca`` and ``feature_store`` for the reason spelled
out in ``slot_layout``: those modules import torch, and the bottlenecks that also
need this check run in a jax-only environment where importing torch fails.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_PREFIX = "key64-static"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_tree_pca_sha(root: str | Path, prefix: str = DEFAULT_PREFIX) -> str:
    """The PCA sha recorded by every shard under ``root``.

    Disagreement between shards means the tree was projected in more than one
    pass under different coordinates, which is unrecoverable -- the feature store
    is a single coordinate system by construction.
    """
    root = Path(root)
    recorded: dict[str, str] = {}
    for path in sorted(root.glob(f"*/{prefix}-pca-artifact.sha256")):
        recorded[path.parent.name] = path.read_text().strip()
    if not recorded:
        raise FileNotFoundError(
            f"no {prefix}-pca-artifact.sha256 under {root}; the tree has not been "
            "PCA-transformed, or the prefix is wrong"
        )
    distinct = set(recorded.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{shard}={sha[:12]}" for shard, sha in sorted(recorded.items()))
        raise RuntimeError(
            f"shards under {root} disagree on their PCA coordinates ({detail}); "
            "the feature store must be a single coordinate system -- re-run "
            "`pca transform` for the whole tree"
        )
    return distinct.pop()


def normalization_pca_sha(path: str | Path) -> str:
    import numpy as np  # deferred: keeps the shard-only path dependency-free

    with np.load(path, allow_pickle=True) as artifact:
        if "pca_sha256" not in artifact.files:
            raise KeyError(
                f"{path} predates PCA provenance recording and cannot be verified; "
                "re-run the `normalization` stage"
            )
        return str(np.asarray(artifact["pca_sha256"]).item())


def verify_binding(
    *,
    features: str | Path | None = None,
    normalization: str | Path | None = None,
    pca: str | Path | None = None,
    prefix: str = DEFAULT_PREFIX,
) -> str:
    """Assert that every supplied artifact was built from the same PCA.

    Returns the agreed sha.  Raises ``RuntimeError`` naming the disagreeing
    artifacts otherwise.  At least two artifacts are needed for the check to say
    anything, and passing only one is treated as a caller error rather than a
    silent pass.
    """
    observed: dict[str, str] = {}
    if features is not None:
        observed[f"features={features}"] = feature_tree_pca_sha(features, prefix)
    if normalization is not None:
        observed[f"normalization={normalization}"] = normalization_pca_sha(normalization)
    if pca is not None:
        observed[f"pca={pca}"] = sha256_file(pca)

    if len(observed) < 2:
        raise ValueError(
            "verify_binding needs at least two artifacts to compare; got "
            f"{len(observed)}"
        )

    if len(set(observed.values())) > 1:
        lines = "\n".join(f"  {sha[:16]}...  {name}" for name, sha in observed.items())
        raise RuntimeError(
            "PCA coordinate mismatch -- these artifacts were not built from the "
            f"same PCA:\n{lines}\n"
            "The two trees under outputs/state_tokenizer/ use identical file "
            "names; check that --features, --normalization and --pca all point "
            "into the same one."
        )
    return next(iter(observed.values()))
