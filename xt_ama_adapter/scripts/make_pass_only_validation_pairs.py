"""Keep training/test unchanged and restrict QFormer validation to PASS rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
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
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files if name != "metadata"}
        metadata = json.loads(str(np.asarray(source["metadata"])))
    split = arrays["split"].astype(str)
    quality = arrays["quality_label"].astype(str)
    demoted = (split == "validation") & (quality != "PASS")
    holdout_label = "holdout_validation_nonpass"
    width = max(split.dtype.itemsize // np.dtype("U1").itemsize, len(holdout_label))
    revised_split = np.asarray(split, dtype=f"<U{width}")
    revised_split[demoted] = holdout_label
    arrays["split"] = revised_split
    source_hash = sha256(args.input)
    rule = "validation=original_validation_and_quality_PASS;nonpass=holdout"
    metadata = {
        **metadata,
        "qa_sha256": hashlib.sha256(
            (str(metadata.get("qa_sha256")) + source_hash + rule).encode()
        ).hexdigest(),
        "validation_filter": {
            "rule": rule,
            "source_pairs_sha256": source_hash,
            "demoted_rows": int(demoted.sum()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **arrays,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    validation = revised_split == "validation"
    report = {
        "protocol": metadata.get("protocol"),
        "pairs": int(len(split)),
        "split_counts": {
            name: int((revised_split == name).sum())
            for name in sorted(set(revised_split))
        },
        "validation_quality_counts": {
            name: int(((quality == name) & validation).sum())
            for name in sorted(set(quality[validation]))
        },
        "validation_observations": int(len(set(zip(
            arrays["sample_id"][validation].astype(str),
            arrays["record_index"][validation].astype(int),
        )))),
        "validation_trajectories": int(len(set(
            arrays["sample_id"][validation].astype(str)
        ))),
        "validation_filter": metadata["validation_filter"],
        "output_sha256_pending": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
