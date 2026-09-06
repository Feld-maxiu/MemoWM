"""Merge accepted HumanTrajs and WebChain QFormer pairs without split leakage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


PROTOCOL = "mixed_web_qformer_qa_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in (
            "sample_id", "record_index", "question", "answer", "question_type", "split"
        )}
        arrays["sample_weight"] = (
            np.asarray(data["sample_weight"], dtype=np.float32)
            if "sample_weight" in data.files
            else np.ones(len(arrays["sample_id"]), dtype=np.float32)
        )
        arrays["quality_label"] = (
            np.asarray(data["quality_label"])
            if "quality_label" in data.files
            else np.full(len(arrays["sample_id"]), "VERIFIED")
        )
        # Observation-teacher QA entries are keyed by the row number of the
        # component pairs file used to build that cache.  A deterministic
        # subset therefore carries its original row numbers explicitly.
        arrays["teacher_row"] = (
            np.asarray(data["teacher_row"], dtype=np.int64)
            if "teacher_row" in data.files
            else np.arange(len(arrays["sample_id"]), dtype=np.int64)
        )
        metadata = json.loads(str(np.asarray(data["metadata"])))
    return arrays, metadata


def link_store(source: Path, target: Path) -> int:
    count = 0
    for path in sorted(source.glob("*.npz")):
        destination = target / path.name
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != path.resolve():
                raise ValueError(f"store filename collision: {path.name}")
            continue
        os.symlink(path.resolve(), destination)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humantrajs-pairs", type=Path, required=True)
    parser.add_argument("--webchain-pairs", type=Path, required=True)
    parser.add_argument("--humantrajs-store", type=Path, required=True)
    parser.add_argument("--webchain-store", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    left, left_meta = load(args.humantrajs_pairs)
    right, right_meta = load(args.webchain_pairs)
    if left_meta.get("protocol") != "humantrajs_qformer_qa_v1":
        raise ValueError("unexpected HumanTrajs pairs protocol")
    if right_meta.get("protocol") != "webchain_qformer_qa_v1":
        raise ValueError("unexpected WebChain pairs protocol")
    overlap = set(left["sample_id"].astype(str)) & set(right["sample_id"].astype(str))
    if overlap:
        raise ValueError(f"sample-id collision across sources: {next(iter(overlap))}")

    merged = {name: np.concatenate((left[name], right[name])) for name in left}
    # ``load`` has already resolved each component's teacher coordinate.  Do
    # not renumber it here: subsets must keep the source-cache row number.
    metadata = {
        "protocol": PROTOCOL,
        "dataset": "HumanTrajs+WebChain",
        "component_protocols": [left_meta["protocol"], right_meta["protocol"]],
        "component_pair_sha256": [sha256(args.humantrajs_pairs), sha256(args.webchain_pairs)],
        "official_ama_test_included": False,
        "instruction_included": False,
        "qa_sha256": hashlib.sha256(
            (sha256(args.humantrajs_pairs) + sha256(args.webchain_pairs)).encode()
        ).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            (str(left_meta.get("manifest_sha256")) + str(right_meta.get("manifest_sha256"))).encode()
        ).hexdigest(),
        "pairs": int(len(merged["sample_id"])),
        "source_counts": {
            "HumanTrajs": int(len(left["sample_id"])),
            "WebChain": int(len(right["sample_id"])),
        },
    }
    args.pairs.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.pairs, **merged, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    args.store_dir.mkdir(parents=True, exist_ok=True)
    linked = link_store(args.humantrajs_store, args.store_dir)
    linked += link_store(args.webchain_store, args.store_dir)
    report = {
        **metadata,
        "split_counts": {
            split: int(np.sum(merged["split"].astype(str) == split))
            for split in ("train", "validation", "test")
        },
        "quality_counts": {
            value: int(np.sum(merged["quality_label"].astype(str) == value))
            for value in sorted(set(merged["quality_label"].astype(str)))
        },
        "store_files": len(list(args.store_dir.glob("*.npz"))),
        "new_links": linked,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
