"""Concatenate retrieval-bridge caches so one head can be trained across domains.

The shipped head is fitted on 5,500 BrowserGym rows and collapses on
WorldMemArena -- output pairwise cosine 0.83 against 0.34 in domain, and its
observation rows never enter a top-10. Retraining the same architecture on
WorldMemArena rows alone fixes that (9x chance on held-out web samples), which
locates the fault in the training distribution rather than in the PCA basis or
the head's capacity.

The mechanism is the contrastive term. ``symmetric_infonce`` draws its negatives
from the batch, so with a single-domain cache the objective never once asks the
head to keep a WorldMemArena state apart from a BrowserGym one. Merging the
caches puts both domains in every batch and supplies exactly that constraint.

Each source keeps its own train/validation assignment rather than being
re-split. The in-domain validation rows then stay the same 500 the shipped head
was measured on, so a regression there is comparable against the recorded
0.8711 rather than against a fresh draw.

A ``domain`` column is added so evaluation can be reported per source; it is
extra to the schema ``train_retrieval_bridge.load_cache`` requires and is
ignored by it.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from residualmem.latent.instruct_bridge import (
    BRIDGE_CACHE_PROTOCOL,
    FUSED_OBSERVATION_TEACHER_PROTOCOL,
)

REQUIRED = ("xbar", "valid", "teacher_fused_embedding", "split")


def load(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=True) as data:
        missing = [name for name in REQUIRED if name not in data.files]
        if missing:
            raise ValueError(f"{path.name} is missing {missing}")
        metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        if metadata.get("protocol") != BRIDGE_CACHE_PROTOCOL:
            raise ValueError(f"{path.name} protocol {metadata.get('protocol')!r}")
        if metadata.get("teacher_protocol") != FUSED_OBSERVATION_TEACHER_PROTOCOL:
            raise ValueError(f"{path.name} teacher is not fused observation v1")
        if metadata.get("representation", "xbar") != "xbar":
            raise ValueError(f"{path.name} representation must be xbar")
        arrays = {name: np.asarray(data[name]) for name in REQUIRED}
        arrays["global_indices"] = (
            np.asarray(data["global_indices"]) if "global_indices" in data.files
            else np.full(len(arrays["xbar"]), -1, np.int64)
        )
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", action="append", required=True,
                        help="repeatable; NAME=PATH labels the rows' domain")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    parts, sources = [], []
    for entry in args.cache:
        name, _, path = entry.partition("=")
        if not path:
            raise ValueError(f"--cache expects NAME=PATH, got {entry!r}")
        arrays, metadata = load(Path(path))
        arrays["domain"] = np.full(len(arrays["xbar"]), name, dtype=object)
        parts.append(arrays)
        counts = collections.Counter(arrays["split"].tolist())
        sources.append({"domain": name, "path": path, "rows": int(len(arrays["xbar"])),
                        "split_counts": dict(counts), "source_metadata": metadata})
        print(f"[merge] {name:14s} {len(arrays['xbar']):6d} rows  {dict(counts)}", flush=True)

    merged = {
        key: np.concatenate([part[key] for part in parts])
        for key in ("xbar", "valid", "teacher_fused_embedding", "split", "global_indices")
    }
    merged["domain"] = np.concatenate([part["domain"] for part in parts]).astype(str)

    teacher = merged["teacher_fused_embedding"].astype(np.float32)
    norms = np.linalg.norm(teacher, axis=1)
    if norms.min() < 0.5 or norms.max() > 2.0:
        raise ValueError(
            f"teacher norms are pathological ({norms.min():.4f}..{norms.max():.4f}); "
            "this is not rounding drift"
        )
    # Renormalize even though the drift is small. The two sources disagree
    # systematically -- the BrowserGym cache sits at 0.9964..1.0039 while the
    # WorldMemArena one is exactly 1.0 -- and a norm that correlates with the
    # domain is a shortcut the contrastive term can exploit to separate the two
    # without looking at content at all. Both loss terms assume unit teachers
    # anyway.
    print(f"[merge] teacher norms {norms.min():.6f}..{norms.max():.6f}, "
          f"{int((abs(norms - 1) > 1e-3).sum())} rows off unit by >1e-3; renormalizing")
    teacher = teacher / np.maximum(norms, 1e-12)[:, None]

    metadata = {
        "protocol": BRIDGE_CACHE_PROTOCOL,
        "teacher_protocol": FUSED_OBSERVATION_TEACHER_PROTOCOL,
        "representation": "xbar",
        "merged_from": sources,
        "split_unit": "inherited from each source",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        xbar=merged["xbar"].astype(np.float32),
        valid=merged["valid"].astype(bool),
        teacher_fused_embedding=teacher,
        split=merged["split"],
        global_indices=merged["global_indices"].astype(np.int64),
        domain=merged["domain"],
        metadata=np.asarray(json.dumps(metadata)),
    )
    total = collections.Counter(merged["split"].tolist())
    print(f"\n{len(merged['xbar'])} rows  {dict(total)}")
    for name in sorted(set(merged["domain"].tolist())):
        mask = merged["domain"] == name
        share = mask.sum() / len(mask)
        print(f"   {name:14s} {int(mask.sum()):6d} rows  {share:.1%} of the corpus")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
