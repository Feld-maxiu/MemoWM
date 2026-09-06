"""Freeze a QFormer dataset to samples with complete observation-teacher files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output-pairs", type=Path, required=True)
    parser.add_argument("--output-store", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    store_files = {path.stem: path.resolve() for path in args.store.glob("*.npz")}
    teacher_files = {path.stem: path for path in args.teacher_dir.glob("*.npz")}
    complete = []
    for sample_id in sorted(store_files.keys() & teacher_files.keys()):
        try:
            with np.load(teacher_files[sample_id], allow_pickle=False) as teacher:
                teacher_metadata = json.loads(str(np.asarray(teacher["metadata"])))
                if teacher_metadata.get("sample_id") != sample_id:
                    continue
        except Exception:
            continue
        complete.append(sample_id)
    if not complete:
        raise ValueError("no observation-store sample has a teacher cache")

    with np.load(args.pairs, allow_pickle=False) as source:
        metadata = json.loads(str(np.asarray(source["metadata"])))
        sample_ids = np.asarray(source["sample_id"]).astype(str)
        keep = np.isin(sample_ids, np.asarray(complete))
        selected = np.flatnonzero(keep)
        arrays = {
            name: np.asarray(source[name])[selected]
            for name in source.files
            if name != "metadata" and np.asarray(source[name]).shape[:1] == sample_ids.shape
        }
        original_teacher_rows = (
            np.asarray(source["teacher_row"], dtype=np.int64)
            if "teacher_row" in source.files
            else np.arange(len(sample_ids), dtype=np.int64)
        )
        arrays["teacher_row"] = original_teacher_rows[selected]

    if args.output_store.exists() and any(args.output_store.iterdir()):
        raise FileExistsError(f"refusing non-empty output store {args.output_store}")
    args.output_store.mkdir(parents=True, exist_ok=True)
    for sample_id in complete:
        os.symlink(store_files[sample_id], args.output_store / f"{sample_id}.npz")

    split = arrays["split"].astype(str)
    observations = set(zip(
        arrays["sample_id"].astype(str), arrays["record_index"].astype(int)
    ))
    observation_split = {}
    for sample_id, record_index, split_name in zip(
        arrays["sample_id"].astype(str),
        arrays["record_index"].astype(int),
        split,
    ):
        observation_split[(sample_id, int(record_index))] = split_name
    filtered_metadata = {
        **metadata,
        "qa_pairs": int(len(selected)),
        "observations": int(len(observations)),
        "groups": int(len(complete)),
        "observation_split_counts": {
            name: sum(value == name for value in observation_split.values())
            for name in ("train", "validation", "test")
        },
        "teacher_complete_subset": {
            "source_pairs_sha256": sha256(args.pairs),
            "source_samples": int(len(store_files)),
            "included_samples": int(len(complete)),
            "excluded_samples": int(len(store_files) - len(complete)),
            "selection": "teacher_npz_exists_after_integrity_check",
        },
    }
    args.output_pairs.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_pairs,
        **arrays,
        metadata=np.asarray(json.dumps(filtered_metadata, sort_keys=True)),
    )
    report = {
        "protocol": "qformer_teacher_complete_subset_v1",
        "source_rows": int(len(keep)),
        "included_rows": int(len(selected)),
        "excluded_rows": int(len(keep) - len(selected)),
        "source_samples": int(len(store_files)),
        "included_samples": int(len(complete)),
        "excluded_samples": int(len(store_files) - len(complete)),
        "output_pairs": str(args.output_pairs.resolve()),
        "output_store": str(args.output_store.resolve()),
        "metadata": filtered_metadata,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
