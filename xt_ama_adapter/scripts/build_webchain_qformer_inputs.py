"""Build query-independent WebChain AXTree inputs for QFormer fine-tuning."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from xt_ama_adapter.scripts.build_humantrajs_qformer_inputs import fixed_text_baseline


PAIRS_PROTOCOL = "webchain_qformer_qa_v1"
STORE_PROTOCOL = "webchain_qformer_observation_store_v1"
OBSERVATION_PROTOCOL = "webchain-axtree-page-chunk-v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split(uid: str) -> str:
    bucket = int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def ax_lines(node: object, output: list[str]) -> None:
    if not isinstance(node, dict):
        return
    role = clean(node.get("role"))
    name = clean(node.get("name"))
    attrs = node.get("attributes") or {}
    useful = []
    for key in (
        "value", "aria-label", "placeholder", "title", "alt", "href",
        "checked", "selected", "disabled",
    ):
        value = clean(attrs.get(key))
        if value and value.casefold() != name.casefold():
            useful.append(f"{key}={value}")
    if role or name or useful:
        interactive = {
            "button", "link", "textbox", "searchbox", "checkbox", "radio",
            "combobox", "option", "menuitem", "tab", "slider", "spinbutton",
        }
        role_text = f"<{role}> " if role.casefold() in interactive else ""
        output.append(
            role_text + name + (" | " + " | ".join(useful) if useful else "")
        )
    for child in node.get("children") or []:
        ax_lines(child, output)


def axtree_chunks(path: Path, max_chars: int) -> list[str]:
    root = json.loads(path.read_text(encoding="utf-8"))
    lines: list[str] = []
    ax_lines(root, lines)
    deduplicated = []
    seen = set()
    for line in lines:
        key = clean(line).casefold()
        if key and key not in seen:
            seen.add(key)
            deduplicated.append(line)
    if not deduplicated:
        raise ValueError(f"{path}: AXTree produced no visible nodes")
    chunks, current, current_chars = [], [], 0
    for line in deduplicated:
        pieces = [line[start:start + max_chars] for start in range(0, len(line), max_chars)]
        for piece in pieces:
            extra = len(piece) + (1 if current else 0)
            if current and current_chars + extra > max_chars:
                chunks.append("\n".join(current))
                current, current_chars = [], 0
                extra = len(piece)
            current.append(piece)
            current_chars += extra
    if current:
        chunks.append("\n".join(current))
    return chunks


def judge_files(directory: Path) -> list[Path]:
    return (
        sorted(directory.glob("judge-*.accepted.jsonl"))
        + sorted(directory.glob("judge-*.review.jsonl"))
    )


def read_sources(old_dir: Path, new_dir: Path) -> list[dict]:
    rows = []
    for generation, directory in (
        ("legacy", old_dir), ("full_trajectory_v4", new_dir)
    ):
        for path in judge_files(directory):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not line.strip():
                    continue
                row = json.loads(line)
                decision = row.get("judge_final_decision")
                if decision not in {"PASS", "REVIEW"}:
                    raise ValueError(
                        f"{path}:{line_number}: unexpected decision {decision!r}"
                    )
                if not all(clean(row.get(key)) for key in ("uid", "question", "answer")):
                    raise ValueError(f"{path}:{line_number}: missing uid/question/answer")
                row["training_source_generation"] = generation
                rows.append(row)
    return rows


def choose_ax(
    row: dict, full_root: Path, final_root: Path
) -> tuple[Path, int, str]:
    uid = str(row["uid"])
    requested = int(row.get("candidate_answer_step", -1))
    full_dir = full_root / uid
    evidence = clean(row.get("evidence")).casefold()
    steps = sorted(full_dir.glob("step_*_ax.json"))
    # This is gold-step association, not observation filtering: once a step is
    # selected its AXTree is serialized without access to the query or answer.
    # The source candidate_answer_step was inherited from the trajectory and
    # is not always the step from which the generator copied its evidence.
    if evidence:
        for path in steps:
            if evidence in clean(path.read_text(encoding="utf-8")).casefold():
                step = int(path.stem.split("_")[1])
                return path, step, "full_trajectory_evidence_step"
    exact = full_dir / f"step_{requested:04d}_ax.json"
    if requested >= 0 and exact.exists():
        return exact, requested, "full_trajectory_exact_step"
    if steps and requested >= 0:
        parsed = [(int(path.stem.split("_")[1]), path) for path in steps]
        step, path = min(parsed, key=lambda item: (abs(item[0] - requested), item[0]))
        return path, step, "full_trajectory_nearest_step"
    final = final_root / uid / "final_ax.json"
    if final.exists():
        return final, max(requested, 0), "legacy_final_step"
    raise FileNotFoundError(f"{uid}: no local AXTree for candidate step {requested}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-judge-dir", type=Path, required=True)
    parser.add_argument("--new-judge-dir", type=Path, required=True)
    parser.add_argument("--full-evidence-root", type=Path, required=True)
    parser.add_argument("--final-evidence-root", type=Path, required=True)
    parser.add_argument("--qa-output", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--blank-image", type=Path, required=True)
    parser.add_argument("--max-observation-chars", type=int, default=7000)
    args = parser.parse_args()
    if args.max_observation_chars < 1000:
        raise ValueError("max-observation-chars is implausibly small")

    rows = read_sources(args.old_judge_dir, args.new_judge_dir)
    if not rows:
        raise ValueError("no PASS/REVIEW rows found")
    args.blank_image.parent.mkdir(parents=True, exist_ok=True)
    if not args.blank_image.exists():
        Image.new("RGB", (1280, 720), "white").save(args.blank_image)

    observations: dict[tuple[str, int, int], dict] = {}
    chunk_cache: dict[Path, list[str]] = {}
    prepared = []
    failures = []
    for source_order, row in enumerate(rows):
        try:
            path, step, resolution = choose_ax(
                row, args.full_evidence_root, args.final_evidence_root
            )
            if path not in chunk_cache:
                chunk_cache[path] = axtree_chunks(
                    path, args.max_observation_chars
                )
            chunks = chunk_cache[path]
            evidence = clean(row.get("evidence")).casefold()
            matches = [
                index for index, chunk in enumerate(chunks)
                if evidence and evidence in clean(chunk).casefold()
            ]
            chunk_index = matches[0] if matches else 0
            key = (str(row["uid"]), step, chunk_index)
            if key not in observations:
                observations[key] = {
                    "text": chunks[chunk_index],
                    "path": str(path.resolve()),
                    "resolution": resolution,
                    "split": stable_split(str(row["uid"])),
                    "source_step_idx": step,
                    "page_chunk_idx": chunk_index,
                    "page_chunks": len(chunks),
                }
            prepared.append((source_order, row, key))
        except Exception as exc:
            failures.append({"uid": row.get("uid"), "error": str(exc)})
    if failures:
        raise ValueError(
            f"{len(failures)} rows could not be prepared; first={failures[0]}"
        )

    grouped: dict[str, list[tuple[int, int, dict]]] = defaultdict(list)
    for (uid, step, chunk_index), observation in observations.items():
        grouped[uid].append((step, chunk_index, observation))

    args.store_dir.mkdir(parents=True, exist_ok=True)
    record_lookup: dict[tuple[str, int, int], int] = {}
    for uid, values in sorted(grouped.items()):
        records, arrays = [], {}
        for index, (step, chunk_index, observation) in enumerate(sorted(values)):
            record_lookup[(uid, step, chunk_index)] = index
            text = observation["text"]
            records.append({
                "trajectory_id": uid,
                "step_idx": step * 1000 + chunk_index,
                "source_step_idx": step,
                "page_chunk_idx": chunk_index,
                "page_chunks": observation["page_chunks"],
                "split": observation["split"],
                "screenshot": str(args.blank_image.resolve()),
                "image_ids": [],
                "synthetic_axtree": text,
                "observation_text": text,
                "fused_text": text,
                "observation_protocol": OBSERVATION_PROTOCOL,
                "axtree_path": observation["path"],
                "axtree_resolution": observation["resolution"],
                "instruction_excluded_from_training": True,
            })
            baseline = fixed_text_baseline(text)
            arrays[f"m11/xbar/{index:04d}"] = baseline
            arrays[f"m11/valid/{index:04d}"] = np.any(baseline != 0, axis=1)
        arrays["metadata"] = np.asarray(json.dumps({
            "protocol": STORE_PROTOCOL,
            "sample_id": uid,
            "records": records,
            "collapse_monitor_baseline": "hashed_text_collapse_monitor_v1",
            "baseline_slots": 32,
        }, ensure_ascii=False))
        target = args.store_dir / f"{uid}.npz"
        temporary = target.with_suffix(".tmp.npz")
        np.savez(temporary, **arrays)
        temporary.replace(target)

    columns = [[] for _ in range(8)]
    samples, indices, questions, answers, kinds, splits, decisions, weights = columns
    qa_rows = []
    evidence_covered = []
    for source_order, row, key in prepared:
        uid, step, chunk_index = key
        decision = str(row["judge_final_decision"])
        split = stable_split(uid)
        weight = 1.0 if decision == "PASS" else 0.5
        samples.append(uid)
        indices.append(record_lookup[key])
        questions.append(clean(row["question"]))
        answers.append(clean(row["answer"]))
        kinds.append("webchain_grounded_axtree")
        splits.append(split)
        decisions.append(decision)
        weights.append(weight)
        evidence_quote = clean(row.get("evidence"))
        evidence_present = bool(
            evidence_quote
            and evidence_quote.casefold() in observations[key]["text"].casefold()
        )
        evidence_covered.append(evidence_present)
        qa_rows.append({
            "qa_id": hashlib.sha256(
                f"{source_order}\0{uid}\0{questions[-1]}\0{answers[-1]}".encode()
            ).hexdigest(),
            "trajectory_id": uid,
            "step_idx": step,
            "page_chunk_idx": chunk_index,
            "split": split,
            "question": questions[-1],
            "answer": answers[-1],
            "quality_label": decision,
            "sample_weight": weight,
            "evidence_quote": evidence_quote,
            "evidence_quote_in_compact_observation": evidence_present,
            "judge_score": row.get("rubric_computed_score"),
            "observation_protocol": OBSERVATION_PROTOCOL,
            "axtree_path": observations[key]["path"],
            "source_generation": row["training_source_generation"],
            "official_ama_test_included": False,
            "instruction_included": False,
        })

    args.qa_output.parent.mkdir(parents=True, exist_ok=True)
    args.qa_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in qa_rows),
        encoding="utf-8",
    )
    source_pairs = len(samples)
    source_quality = Counter(decisions)
    eligible = np.asarray(evidence_covered, dtype=bool)
    # Rows whose judged quote cannot be recovered from the deterministic
    # question-independent observation are retained in the audit JSONL, but
    # must not contribute QA CE/KL or Bridge supervision.
    filtered_columns = []
    for column in columns:
        filtered_columns.append([value for value, keep in zip(column, eligible) if keep])
    samples, indices, questions, answers, kinds, splits, decisions, weights = filtered_columns
    metadata = {
        "protocol": PAIRS_PROTOCOL,
        "dataset": "WebChain",
        "observation_protocol": OBSERVATION_PROTOCOL,
        "store_protocol": STORE_PROTOCOL,
        "official_ama_test_included": False,
        "instruction_included": False,
        "qa_sha256": file_sha256(args.qa_output),
        "manifest_sha256": file_sha256(args.qa_output),
        "pairs": len(samples),
        "observations": len(observations),
        "trajectories": len(grouped),
        "quality_weighting": {"PASS": 1.0, "REVIEW": 0.5},
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
        quality_label=np.asarray(decisions),
        sample_weight=np.asarray(weights, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    report = {
        **metadata,
        "pairs_by_split": dict(Counter(splits)),
        "pairs_by_quality": dict(Counter(decisions)),
        "source_pairs_before_evidence_gate": source_pairs,
        "source_pairs_by_quality": dict(source_quality),
        "evidence_quote_coverage": float(np.mean(evidence_covered)),
        "evidence_quote_missing": int(len(evidence_covered) - sum(evidence_covered)),
        "unique_trajectory_steps": len(observations),
        "training_referenced_steps": len(set(zip(samples, indices))),
        "duplicate_training_qa_steps": len(samples) - len(set(zip(samples, indices))),
        "store_dir": str(args.store_dir.resolve()),
        "pairs_path": str(args.pairs.resolve()),
        "qa_path": str(args.qa_output.resolve()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
