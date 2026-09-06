"""Create a deterministic, provenance-carrying subset of QFormer pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = ("sample_id", "record_index", "question", "answer", "question_type", "split")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as source:
        metadata = json.loads(str(np.asarray(source["metadata"])))
        split = np.asarray(source["split"]).astype(str)
        candidates = np.flatnonzero(split == args.split)
        if args.count <= 0 or args.count > len(candidates):
            raise ValueError(
                f"count must be in 1..{len(candidates)} for split {args.split!r}"
            )
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(candidates, size=args.count, replace=False))
        arrays = {name: np.asarray(source[name])[selected] for name in FIELDS}
        for optional in ("sample_weight", "quality_label"):
            if optional in source.files:
                arrays[optional] = np.asarray(source[optional])[selected]

    # Keep the row coordinate of the original component cache.  The mixed
    # merger propagates this instead of silently renumbering the subset.
    arrays["teacher_row"] = selected.astype(np.int64)
    selection_sha256 = hashlib.sha256(selected.astype("<i8").tobytes()).hexdigest()
    subset_metadata = {
        **metadata,
        "dataset": f"{metadata.get('dataset', 'unknown')}-subset",
        "pairs": int(args.count),
        "observations": int(len(set(zip(
            arrays["sample_id"].astype(str), arrays["record_index"].astype(int)
        )))),
        "trajectories": int(len(set(arrays["sample_id"].astype(str)))),
        "subset": {
            "source_pairs_sha256": file_sha256(args.input),
            "selection_sha256": selection_sha256,
            "split": args.split,
            "count": int(args.count),
            "seed": int(args.seed),
        },
        "qa_sha256": hashlib.sha256(
            (str(metadata.get("qa_sha256")) + selection_sha256).encode()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **arrays,
        metadata=np.asarray(json.dumps(subset_metadata, sort_keys=True)),
    )
    report = {
        **subset_metadata,
        "source_rows": int(len(split)),
        "source_split_rows": int(len(candidates)),
        "selected_rows": int(len(selected)),
        "teacher_row_min": int(selected.min()),
        "teacher_row_max": int(selected.max()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
