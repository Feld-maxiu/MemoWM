"""Validate deterministic BrowserGym episode lanes and assemble the v8 corpus.

The legacy collector assigns task ``t`` the episode lattice ``t + 12*k`` and
stops after 8,334 states.  Lane collection partitions only ``k``; it does not
change episode ids, environment seeds, actions, or split assignment.  This
assembler refuses gaps, duplicates, partial step sequences, seed drift, missing
screenshots, or semantic overlap disagreement before taking the requested prefix
per task. Chrome's process-local CDP node ids are canonicalized for comparison;
all AXTree topology and semantic fields still have to agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .common import PILOT_TASKS, split_for_episode, write_json


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_difference(left: Path, right: Path) -> tuple[float, float]:
    """Return changed-pixel fraction and uint8-channel MAE; shape drift is fatal."""
    a = np.asarray(Image.open(left).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(right).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        raise ValueError(f"overlap screenshot shape mismatch: {left}={a.shape}, {right}={b.shape}")
    difference = np.abs(a - b)
    changed_fraction = float(np.any(difference != 0, axis=2).mean())
    return changed_fraction, float(difference.mean())


def _reference_rows(root: Path | None) -> dict[str, tuple[dict, Path]]:
    if root is None:
        return {}
    output = {}
    for path in sorted(root.glob("records-worker*.jsonl")):
        for row in _read_jsonl(path):
            state_id = row["state_id"]
            if state_id in output:
                raise ValueError(f"duplicate reference state_id: {state_id}")
            output[state_id] = (row, root / row["screenshot"])
    return output


def _canonical_axtree_ids(value: object) -> object:
    """Replace volatile CDP ids with node-order ids while preserving topology."""
    if not isinstance(value, dict) or not isinstance(value.get("nodes"), list):
        return value
    nodes = value["nodes"]
    mapping = {
        str(node["nodeId"]): f"n{index}"
        for index, node in enumerate(nodes)
        if isinstance(node, dict) and "nodeId" in node
    }

    def mapped(raw: object) -> object:
        if raw is None:
            return None
        key = str(raw)
        return mapping.get(key, f"external:{key}")

    canonical = []
    for node in nodes:
        if not isinstance(node, dict):
            canonical.append(node)
            continue
        item = {
            key: field for key, field in node.items()
            if key not in {"nodeId", "parentId", "childIds"}
        }
        if "nodeId" in node:
            item["nodeId"] = mapped(node["nodeId"])
        if "parentId" in node:
            item["parentId"] = mapped(node["parentId"])
        if "childIds" in node:
            item["childIds"] = [mapped(child) for child in node["childIds"]]
        canonical.append(item)
    return {**value, "nodes": canonical}


def _records_equivalent(row: dict, reference: dict) -> bool:
    keys = set(row) | set(reference)
    for key in keys - {"axtree_raw"}:
        if row.get(key) != reference.get(key):
            return False
    return _canonical_axtree_ids(row.get("axtree_raw")) == _canonical_axtree_ids(
        reference.get("axtree_raw")
    )


def assemble(args: argparse.Namespace) -> dict:
    root = Path(args.input_dir).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"no lane shards matching {args.pattern!r} under {root}")

    per_task = int(np.ceil(args.target_states / len(PILOT_TASKS)))
    grouped: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for path in paths:
        for row in _read_jsonl(path):
            state_id = row["state_id"]
            if state_id in seen:
                raise ValueError(f"duplicate lane state_id: {state_id}")
            seen.add(state_id)
            grouped[row["task"]].append(row)

    unknown = sorted(set(grouped) - set(PILOT_TASKS))
    missing_tasks = sorted(set(PILOT_TASKS) - set(grouped))
    if unknown or missing_tasks:
        raise ValueError(f"task coverage mismatch: unknown={unknown}, missing={missing_tasks}")

    references = _reference_rows(Path(args.reference_dir).resolve() if args.reference_dir else None)
    overlap_records = overlap_images = overlap_images_exact = overlap_images_tolerated = 0
    max_overlap_image_changed_fraction = max_overlap_image_mae = 0.0
    selected_total = available_total = 0
    task_reports = {}
    output_paths = []

    for task_index, task in enumerate(PILOT_TASKS):
        rows = sorted(grouped[task], key=lambda row: (int(row["episode_index"]), int(row["step"])))
        episodes: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            episode = int(row["episode_index"])
            step = int(row["step"])
            if episode % args.original_stride != task_index:
                raise ValueError(f"episode left canonical lattice: {row['state_id']}")
            expected_seed = args.seed + task_index * 1_000_000 + episode
            if int(row.get("environment_seed", -1)) != expected_seed:
                raise ValueError(f"environment seed drift: {row['state_id']}")
            if row["split"] != split_for_episode(episode):
                raise ValueError(f"split drift: {row['state_id']}")
            expected_episode_id = f"task{task_index:02d}-ep{episode:06d}"
            expected_state_id = f"{expected_episode_id}-t{step:02d}"
            if row["episode_id"] != expected_episode_id or row["state_id"] != expected_state_id:
                raise ValueError(f"canonical id drift: {row['state_id']}")
            image_path = root / row["screenshot"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            episodes[episode].append(row)

        episode_indices = sorted(episodes)
        expected_indices = [
            task_index + args.original_stride * offset
            for offset in range(len(episode_indices))
        ]
        if episode_indices != expected_indices:
            for position, (actual, expected) in enumerate(zip(episode_indices, expected_indices)):
                if actual != expected:
                    raise ValueError(
                        f"episode gap for task {task_index}: position={position}, "
                        f"actual={actual}, expected={expected}"
                    )
            raise ValueError(
                f"episode coverage length mismatch for task {task_index}: "
                f"actual={len(episode_indices)}, expected-prefix={len(expected_indices)}"
            )

        ordered = []
        for episode in episode_indices:
            values = sorted(episodes[episode], key=lambda row: int(row["step"]))
            steps = [int(row["step"]) for row in values]
            if steps != list(range(len(steps))):
                raise ValueError(f"partial/non-contiguous steps in task {task_index} episode {episode}")
            ordered.extend(values)
        available_total += len(ordered)
        if len(ordered) < per_task:
            raise ValueError(
                f"task {task_index} has only {len(ordered)} states; needs {per_task}; "
                "increase every lane's target-episodes equally and resume"
            )
        selected = ordered[:per_task]
        selected_total += len(selected)

        for row in selected:
            reference = references.get(row["state_id"])
            if reference is None:
                continue
            reference_row, reference_image = reference
            if not _records_equivalent(row, reference_row):
                raise ValueError(f"lane/reference record mismatch: {row['state_id']}")
            overlap_records += 1
            if args.verify_overlap_images:
                lane_image = root / row["screenshot"]
                if not reference_image.is_file():
                    raise FileNotFoundError(reference_image)
                if _sha256(lane_image) == _sha256(reference_image):
                    overlap_images_exact += 1
                else:
                    changed_fraction, mae = _image_difference(lane_image, reference_image)
                    max_overlap_image_changed_fraction = max(
                        max_overlap_image_changed_fraction, changed_fraction
                    )
                    max_overlap_image_mae = max(max_overlap_image_mae, mae)
                    if (
                        changed_fraction > args.max_overlap_image_changed_fraction
                        or mae > args.max_overlap_image_mae
                    ):
                        raise ValueError(
                            f"lane/reference screenshot mismatch: {row['state_id']} "
                            f"changed_fraction={changed_fraction:.9f} mae={mae:.9f}"
                        )
                    overlap_images_tolerated += 1
                overlap_images += 1

        destination = output / f"records-worker{task_index:02d}.jsonl"
        with destination.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        output_paths.append(str(destination))
        task_reports[task] = {
            "episodes_available": len(episode_indices),
            "states_available": len(ordered),
            "states_selected": len(selected),
            "last_selected_episode": int(selected[-1]["episode_index"]),
            "last_selected_step": int(selected[-1]["step"]),
        }

    expected_total = per_task * len(PILOT_TASKS)
    if selected_total != expected_total:
        raise AssertionError(f"selected {selected_total}, expected {expected_total}")
    summary = {
        "protocol": "browsergym_deterministic_episode_lanes_v1",
        "input_dir": str(root),
        "output": str(output),
        "pattern": args.pattern,
        "original_stride": args.original_stride,
        "seed": args.seed,
        "target_states_requested": args.target_states,
        "states_per_task": per_task,
        "states_selected": selected_total,
        "states_available": available_total,
        "overlap_records_verified": overlap_records,
        "overlap_images_verified": overlap_images,
        "overlap_images_exact": overlap_images_exact,
        "overlap_images_tolerated": overlap_images_tolerated,
        "max_overlap_image_changed_fraction": max_overlap_image_changed_fraction,
        "max_overlap_image_mae_uint8": max_overlap_image_mae,
        "task_reports": task_reports,
        "output_paths": output_paths,
    }
    write_json(output / "lane-assembly.summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pattern", default="records-lane-*.jsonl")
    parser.add_argument("--target-states", type=int, default=100_000)
    parser.add_argument("--original-stride", type=int, default=len(PILOT_TASKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-dir")
    parser.add_argument("--verify-overlap-images", action="store_true")
    parser.add_argument("--max-overlap-image-changed-fraction", type=float, default=0.0)
    parser.add_argument("--max-overlap-image-mae", type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(assemble(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
