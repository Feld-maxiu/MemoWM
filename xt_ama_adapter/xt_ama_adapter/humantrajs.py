"""Deterministic, leakage-safe preparation helpers for HumanTrajs.

The source export stores the screenshot *before* the action at the same row.
For AMA-style memory records we therefore align ``action_i`` with
``screenshot_{i+1}``, while retaining both paths for audit and transition QA.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


TERMINAL_ACTIONS = {"send_msg_to_user", "answer", "exit", "terminate", "stop"}


def normalized_qa_key(question: str, answer: str) -> tuple[str, str]:
    """Normalize generated text for deterministic exact-pair deduplication."""
    normalize = lambda text: re.sub(r"\s+", " ", str(text).casefold()).strip(" \t\r\n.,!?;:'\"")
    return normalize(question), normalize(answer)


def action_name(action: Any) -> str:
    if not isinstance(action, dict):
        return ""
    value = action.get("action_name")
    if not value and isinstance(action.get("action_output"), dict):
        value = action["action_output"].get("action_name")
    return str(value or "").strip().lower()


def sanitize_action(action: Any) -> dict[str, str]:
    """Keep executable action facts and discard thoughts/final answers."""
    if not isinstance(action, dict):
        return {"action_str": str(action)} if action else {}
    clean: dict[str, str] = {}
    name = action_name(action)
    if name:
        clean["action_name"] = name
    for key in ("action_str", "action_description"):
        value = action.get(key)
        if value is not None and str(value).strip():
            text = str(value).strip()
            # Some click descriptions contain a serialized copy of the whole
            # page. That is neither an action nor safe label-generation input.
            if key == "action_description" and len(text) > 256:
                text = text.split(" with value", 1)[0]
            limit = 512 if key == "action_str" else 256
            clean[key] = text[:limit]
    return clean


def sanitize_observation(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        return {}
    allowed = ("page_index", "url", "open_pages_titles", "open_pages_urls")
    return {key: observation[key] for key in allowed if observation.get(key) is not None}


def stable_split(trajectory_id: str, train: float = 0.8, validation: float = 0.1) -> str:
    """Assign an entire trajectory to a reproducible split."""
    if train < 0 or validation < 0 or train + validation > 1:
        raise ValueError("split fractions must be non-negative and sum to <= 1")
    bucket = int(hashlib.sha256(trajectory_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < train:
        return "train"
    if bucket < train + validation:
        return "validation"
    return "test"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_rows(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trajectory_id"])].append(row)
    for trajectory in grouped.values():
        trajectory.sort(key=lambda row: int(row["step_idx"]))
    return dict(grouped)


def align_trajectories(
    rows: Iterable[dict[str, Any]],
    image_inspector: Callable[[str], dict[str, Any]],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create post-action records and a complete rejection ledger.

    ``image_inspector`` must return ``exists``, ``sha256`` and ``low_information``.
    The last source row cannot be aligned because it has no post-action image.
    """
    prepared: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for trajectory_id, trajectory in group_rows(rows).items():
        split = stable_split(trajectory_id, train_fraction, validation_fraction)
        image_meta = {str(row.get("image_path")): image_inspector(str(row.get("image_path"))) for row in trajectory}
        candidates: list[dict[str, Any]] = []
        for pos, current in enumerate(trajectory):
            step_idx = int(current["step_idx"])
            name = action_name(current.get("action"))
            reason = None
            if name in TERMINAL_ACTIONS:
                reason = "terminal_action"
            elif pos + 1 >= len(trajectory):
                reason = "no_post_action_frame"
            else:
                next_row = trajectory[pos + 1]
                if int(next_row["step_idx"]) != step_idx + 1:
                    reason = "non_contiguous_next_step"
                else:
                    target_meta = image_meta[str(next_row.get("image_path"))]
                    if not target_meta["exists"]:
                        reason = "post_image_missing"
                    elif target_meta["low_information"]:
                        reason = "post_image_low_information"
            if reason:
                rejected.append({
                    "trajectory_id": trajectory_id,
                    "step_idx": step_idx,
                    "action_name": name,
                    "reason": reason,
                })
                continue

            next_row = trajectory[pos + 1]
            before_path = str(current["image_path"])
            post_path = str(next_row["image_path"])
            before_meta = image_meta[before_path]
            post_meta = image_meta[post_path]
            candidates.append({
                "protocol": "humantrajs-post-action-v1",
                "trajectory_id": trajectory_id,
                "step_idx": step_idx,
                "post_frame_step_idx": int(next_row["step_idx"]),
                "previous_memory_step_idx": None,
                "action": sanitize_action(current.get("action")),
                "observation": sanitize_observation(current.get("observation")),
                "before_image_path": before_path,
                "image_path": post_path,
                "before_image_sha256": before_meta.get("sha256"),
                "image_sha256": post_meta["sha256"],
                "transition_eligible": bool(before_meta["exists"] and not before_meta["low_information"]),
                "instruction": current.get("instruction"),
                "instruction_allowed_for_qa_generation": False,
                "split": split,
                "alignment": "action_i_to_screenshot_i_plus_1",
                "equivalent_step_ids": [],
            })

        by_hash: dict[str, list[int]] = defaultdict(list)
        for record in candidates:
            by_hash[record["image_sha256"]].append(record["step_idx"])
        previous: int | None = None
        for record in candidates:
            record["previous_memory_step_idx"] = (
                previous if previous is not None and previous == record["step_idx"] - 1 else None
            )
            equivalents = by_hash[record["image_sha256"]]
            record["equivalent_step_ids"] = equivalents if len(equivalents) > 1 else [record["step_idx"]]
            prepared.append(record)
            previous = record["step_idx"]
    return prepared, rejected


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
