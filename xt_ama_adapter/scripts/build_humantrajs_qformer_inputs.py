"""Build the pair table and observation store consumed by QFormer training."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROTOCOL = "humantrajs_qformer_qa_v1"
STORE_PROTOCOL = "humantrajs_qformer_observation_store_v1"
BASELINE_PROTOCOL = "hashed_text_collapse_monitor_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fixed_text_baseline(text: str, *, slots: int = 32, width: int = 512) -> np.ndarray:
    """A deterministic lexical reference used only by the collapse monitor.

    This is deliberately not a learned target or a training loss. Tokens are
    partitioned by position, feature-hashed into fixed slots, and L2-normalized.
    It gives the trainer a stable observation-separability reference without
    pretending that pseudo Web text came from the legacy WorldMemArena PCA.
    """
    tokens = re.findall(r"[\w]+|[^\w\s]", text.casefold(), flags=re.UNICODE)
    output = np.zeros((slots, width), dtype=np.float32)
    if not tokens:
        return output
    for position, token in enumerate(tokens):
        slot = min(slots - 1, position * slots // len(tokens))
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "little") % width
        sign = 1.0 if digest[8] & 1 else -1.0
        output[slot, index] += sign
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    np.divide(output, np.maximum(norms, 1e-12), out=output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline-slots", type=int, default=32)
    args = parser.parse_args()

    manifest = {}
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                manifest[(row["trajectory_id"], int(row["step_idx"]))] = row

    with args.qa.open(encoding="utf-8") as handle:
        qa_rows = [json.loads(line) for line in handle if line.strip()]
    if not qa_rows:
        raise ValueError("QA input is empty")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for qa in qa_rows:
        if qa.get("qa_input_protocol") != "vl-pseudo-web-observation-v1":
            raise ValueError(f"unsupported QA protocol: {qa.get('qa_input_protocol')!r}")
        if qa.get("verification_status") != "pseudo_web_text_verified":
            raise ValueError("unverified QA reached the QFormer builder")
        if "instruction" in qa:
            raise ValueError("training QA still contains instruction")
        key = (qa["trajectory_id"], int(qa["step_idx"]))
        source = manifest.get(key)
        if source is None:
            raise ValueError(f"QA has no manifest parent: {key}")
        if source["image_sha256"] != qa["image_sha256"]:
            raise ValueError(f"image mismatch for {key}")
        grouped[qa["trajectory_id"]].append(qa)

    args.store_dir.mkdir(parents=True, exist_ok=True)
    record_lookup: dict[tuple[str, int], int] = {}
    observations = 0
    for trajectory_id, rows in sorted(grouped.items()):
        unique = {}
        for row in rows:
            unique.setdefault(int(row["step_idx"]), row)
        records, arrays = [], {}
        for index, (step_idx, row) in enumerate(sorted(unique.items())):
            record_lookup[(trajectory_id, step_idx)] = index
            text = row["text_observation"]
            records.append({
                "trajectory_id": trajectory_id,
                "step_idx": step_idx,
                "split": row["split"],
                "screenshot": row["image_path"],
                "image_ids": [row["image_sha256"]],
                "synthetic_axtree": text,
                "observation_text": text,
                "fused_text": text,
                "observation_protocol": "vl-pseudo-web-observation-v1",
                "instruction_excluded_from_training": True,
            })
            baseline = fixed_text_baseline(text, slots=args.baseline_slots)
            arrays[f"m11/xbar/{index:04d}"] = baseline
            arrays[f"m11/valid/{index:04d}"] = np.any(baseline != 0, axis=1)
            observations += 1
        arrays["metadata"] = np.asarray(json.dumps({
            "protocol": STORE_PROTOCOL,
            "sample_id": trajectory_id,
            "records": records,
            "collapse_monitor_baseline": BASELINE_PROTOCOL,
            "baseline_slots": args.baseline_slots,
        }, ensure_ascii=False))
        target = args.store_dir / f"{trajectory_id}.npz"
        temporary = target.with_suffix(".tmp.npz")
        np.savez(temporary, **arrays)
        temporary.replace(target)

    samples, indices, questions, answers, kinds, splits = [], [], [], [], [], []
    for row in qa_rows:
        samples.append(row["trajectory_id"])
        indices.append(record_lookup[(row["trajectory_id"], int(row["step_idx"]))])
        questions.append(row["question"])
        answers.append(row["answer"])
        kinds.append("humantrajs_local_web")
        splits.append(row["split"])

    metadata = {
        "protocol": PROTOCOL,
        "dataset": "HumanTrajs",
        "observation_protocol": "vl-pseudo-web-observation-v1",
        "store_protocol": STORE_PROTOCOL,
        "collapse_monitor_baseline": BASELINE_PROTOCOL,
        "official_ama_test_included": False,
        "instruction_included": False,
        "qa_sha256": sha256_file(args.qa),
        "manifest_sha256": sha256_file(args.manifest),
        "pairs": len(samples),
        "observations": observations,
        "trajectories": len(grouped),
    }
    args.pairs.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.pairs,
        sample_id=np.asarray(samples),
        record_index=np.asarray(indices, dtype=np.int32),
        question=np.asarray(questions),
        answer=np.asarray(answers),
        question_type=np.asarray(kinds),
        split=np.asarray(splits),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    report = {
        **metadata,
        "pairs_by_split": dict(Counter(splits)),
        "trajectories_by_split": dict(Counter(
            next(iter({row["split"] for row in rows})) for rows in grouped.values())),
        "store_dir": str(args.store_dir.resolve()),
        "pairs_path": str(args.pairs.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
