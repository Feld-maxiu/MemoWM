"""Build text-only observation store and QA references for the manual QFormer trainer."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np


PROTOCOL = "molmoweb_text_qformer_qa_v1"
OBSERVATION_PROTOCOL = "qwen35_9b_visual_transcription_v1"
COLLAPSE_BASELINE_PROTOCOL = "hashed_text_collapse_monitor_v1"


def fixed_text_baseline(text: str, *, slots: int = 32, width: int = 512) -> np.ndarray:
    """Deterministic text-only reference for the report-only collapse monitor.

    MolmoWeb's model input has no screenshot, so the legacy screenshot+AXTree
    PCA coordinates are not a valid comparator.  This is the same positional
    feature-hash reference used by the HumanTrajs/WebChain text-only QFormer
    inputs.  It is never consumed by a loss or by the reader.
    """
    tokens = re.findall(r"[\w]+|[^\w\s]", text.casefold(), flags=re.UNICODE)
    output = np.zeros((slots, width), dtype=np.float32)
    if not tokens:
        return output
    for position, token in enumerate(tokens):
        slot = min(slots - 1, position * slots // len(tokens))
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        feature = int.from_bytes(digest[:8], "little") % width
        output[slot, feature] += 1.0 if digest[8] & 1 else -1.0
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    np.divide(output, np.maximum(norms, 1e-12), out=output)
    return output


def canonical_url(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    return urlunsplit((parsed.scheme.casefold(), parsed.netloc.casefold(),
                       parsed.path.rstrip("/") or "/", "", ""))


def split_for(key: str, validation_percent: int, test_percent: int) -> str:
    bucket = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % 100
    if bucket < test_percent:
        return "test"
    if bucket < test_percent + validation_percent:
        return "validation"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--test-percent", type=int, default=5)
    parser.add_argument("--baseline-slots", type=int, default=32)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()
    if args.validation_percent + args.test_percent >= 100:
        raise ValueError("validation plus test percentage must be below 100")

    rows = []
    for path in args.input:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError("no verified QA rows")
    if any(row.get("observation_protocol") != OBSERVATION_PROTOCOL for row in rows):
        raise ValueError("every row must use the frozen-9B transcription protocol")

    by_observation: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_observation[str(row["observation_id"])].append(row)
    groups: dict[str, list[tuple[str, list[dict]]]] = collections.defaultdict(list)
    for observation_id, qa_rows in sorted(by_observation.items()):
        representative = qa_rows[0]
        key = canonical_url(representative.get("url")) or str(representative["image_sha256"])
        groups[key].append((observation_id, qa_rows))

    args.store.mkdir(parents=True, exist_ok=True)
    args.pairs.parent.mkdir(parents=True, exist_ok=True)
    sample_ids, record_indices, questions, answers, kinds, splits = [], [], [], [], [], []
    observation_count = 0
    group_counts = collections.Counter()
    for group_key, observations in sorted(groups.items()):
        sample_id = "mw-" + hashlib.sha256(group_key.encode()).hexdigest()[:24]
        split = split_for(group_key, args.validation_percent, args.test_percent)
        records, arrays = [], {}
        for record_index, (observation_id, qa_rows) in enumerate(observations):
            representative = qa_rows[0]
            text = str(representative["text_observation"]).strip()
            if not text:
                raise ValueError(f"{observation_id}: empty text observation")
            records.append({
                "observation_id": observation_id,
                "text_observation": text,
                "fused_text": text,
                "observation_protocol": OBSERVATION_PROTOCOL,
                "image_ids": [],
                "source_image_sha256": representative["image_sha256"],
                "canonical_url_group": group_key,
                "split": split,
            })
            baseline = fixed_text_baseline(text, slots=args.baseline_slots)
            arrays[f"m11/xbar/{record_index:04d}"] = baseline
            arrays[f"m11/valid/{record_index:04d}"] = np.any(baseline != 0, axis=1)
            for row in qa_rows:
                sample_ids.append(sample_id)
                record_indices.append(record_index)
                questions.append(str(row["question"]))
                answers.append(str(row["answer"]))
                kinds.append(str(row.get("question_type") or ""))
                splits.append(split)
            observation_count += 1
            group_counts[split] += 1
        arrays["metadata"] = np.asarray(json.dumps({
                "protocol": "molmoweb_text_observation_store_v1",
                "sample_id": sample_id,
                "group_key": group_key,
                "split": split,
                "records": records,
                "collapse_monitor_baseline": COLLAPSE_BASELINE_PROTOCOL,
                "baseline_slots": args.baseline_slots,
                "collapse_monitor_only": True,
            }, ensure_ascii=False, sort_keys=True))
        target = args.store / f"{sample_id}.npz"
        temporary = target.with_suffix(".tmp.npz")
        np.savez(temporary, **arrays)
        temporary.replace(target)

    metadata = {
        "protocol": PROTOCOL,
        "observation_protocol": OBSERVATION_PROTOCOL,
        "input_modality": "text_only",
        "official_ama_test_included": False,
        "instruction_included": False,
        "action_included": False,
        "trajectory_included": False,
        "question_used_to_generate_observation": False,
        "collapse_monitor_baseline": COLLAPSE_BASELINE_PROTOCOL,
        "baseline_slots": args.baseline_slots,
        "collapse_monitor_only": True,
        "session_definition": "canonical_url_fallback_image_sha256",
        "validation_percent": args.validation_percent,
        "test_percent": args.test_percent,
        "seed": args.seed,
        "observations": observation_count,
        "qa_pairs": len(questions),
        "groups": len(groups),
        "observation_split_counts": dict(group_counts),
    }
    np.savez(
        args.pairs,
        sample_id=np.asarray(sample_ids),
        record_index=np.asarray(record_indices, dtype=np.int32),
        question=np.asarray(questions),
        answer=np.asarray(answers),
        question_type=np.asarray(kinds),
        split=np.asarray(splits),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
