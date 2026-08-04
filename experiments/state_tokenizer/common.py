"""Shared, dependency-light utilities for the state-tokenizer pilot.

The experiment deliberately keeps the representation ladder small:

    Qwen H_t -> Y64 (40 image / 20 DOM / 4 instruction slots)
             -> Y32 (20 image / 10 DOM / 2 instruction slots)
    Y64      -> PCA512

No learned tokenizer is implemented or launched by this package.  If the Y64
gate fails, the pipeline writes the failure report and exits.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


PILOT_TASKS: tuple[str, ...] = (
    "miniwob/click-button-v1",
    "miniwob/click-dialog-2-v1",
    "miniwob/click-checkboxes-v1",
    "miniwob/choose-list-v1",
    "miniwob/click-option-v1",
    "miniwob/enter-text-v1",
    "miniwob/focus-text-v1",
    "miniwob/use-autocomplete-nodelay-v1",
    "miniwob/copy-paste-v1",
    "miniwob/scroll-text-2-v1",
    "miniwob/login-user-v1",
    "miniwob/form-sequence-v1",
)

SLOT_LAYOUTS: dict[int, tuple[int, int, int]] = {
    64: (40, 20, 4),
    32: (20, 10, 2),
}
MODALITIES: tuple[str, ...] = ("image", "dom", "instruction")
STATE_LABELS: tuple[str, ...] = (
    "role_dialog",
    "role_button",
    "role_textbox",
    "role_checkbox",
    "role_radio",
    "role_select_option",
    "any_checked",
    "any_unchecked",
    "any_selected",
    "any_disabled",
    "any_enabled",
)

WORD_RE = re.compile(r"[A-Za-z0-9]+")


def adaptive_average_pool(values: np.ndarray, slots: int) -> tuple[np.ndarray, np.ndarray]:
    """Order-preserving adaptive mean pool with deterministic short-input handling.

    For non-empty input, every output slot has at least one source token.  When
    ``len(values) < slots``, adjacent slots intentionally overlap.  Empty input
    maps to zero slots and a false validity mask.
    """
    values = np.asarray(values)
    if values.ndim != 2:
        raise ValueError(f"expected (tokens, width), got {values.shape}")
    if slots < 1:
        raise ValueError("slots must be positive")
    count, width = values.shape
    if count == 0:
        return np.zeros((slots, width), np.float32), np.zeros((slots,), bool)
    pooled = np.empty((slots, width), np.float32)
    for index in range(slots):
        start = math.floor(index * count / slots)
        stop = math.ceil((index + 1) * count / slots)
        stop = max(start + 1, min(stop, count))
        pooled[index] = np.asarray(values[start:stop], np.float32).mean(axis=0)
    return pooled, np.ones((slots,), bool)


def pool_modalities(
    modalities: Sequence[np.ndarray], total_slots: int
) -> tuple[np.ndarray, np.ndarray]:
    if len(modalities) != len(MODALITIES):
        raise ValueError(f"expected {len(MODALITIES)} modalities")
    layout = SLOT_LAYOUTS[total_slots]
    chunks, masks = zip(
        *(adaptive_average_pool(values, slots) for values, slots in zip(modalities, layout))
    )
    return np.concatenate(chunks, axis=0), np.concatenate(masks, axis=0)


def modality_slices(total_slots: int) -> tuple[slice, slice, slice]:
    image, dom, instruction = SLOT_LAYOUTS[total_slots]
    return (
        slice(0, image),
        slice(image, image + dom),
        slice(image + dom, image + dom + instruction),
    )


def aggregate_mean_max(tokens: np.ndarray, total_slots: int) -> np.ndarray:
    """Fixed probe input: mean and coordinate-wise max for each modality."""
    tokens = np.asarray(tokens, np.float32)
    if tokens.ndim != 2 or tokens.shape[0] != total_slots:
        raise ValueError(f"expected ({total_slots}, width), got {tokens.shape}")
    parts = []
    for section in modality_slices(total_slots):
        part = tokens[section]
        parts.extend((part.mean(axis=0), part.max(axis=0)))
    return np.concatenate(parts, axis=0).astype(np.float32)


def split_for_episode(episode_index: int) -> str:
    bucket = int(episode_index) % 10
    return "train" if bucket < 6 else ("validation" if bucket < 8 else "test")


def normalize_scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(()).item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def compact_dom(dom_elements: Iterable[dict]) -> str:
    """Stable compact serialization of the observation-visible MiniWoB DOM."""
    rows: list[str] = []
    for raw in dom_elements:
        item = {key: normalize_scalar(value) for key, value in raw.items()}
        pieces = [
            f"ref={item.get('ref', 0)}",
            f"parent={item.get('parent', 0)}",
            f"tag={item.get('tag', '')}",
            "box={:.1f},{:.1f},{:.1f},{:.1f}".format(
                float(item.get("left", 0.0)),
                float(item.get("top", 0.0)),
                float(item.get("width", 0.0)),
                float(item.get("height", 0.0)),
            ),
        ]
        for key in ("text", "value", "id", "classes"):
            value = item.get(key)
            if value not in (None, "", False):
                pieces.append(f"{key}={json.dumps(str(value), ensure_ascii=False)}")
        flags = item.get("flags")
        if isinstance(flags, list) and any(flags):
            pieces.append("flags=" + ",".join(str(int(x)) for x in flags))
        rows.append("<" + " ".join(pieces) + "/>")
    return "\n".join(rows)


def derive_probe_labels(observation: dict, sidecar: Sequence[dict]) -> dict:
    elements = [
        {key: normalize_scalar(value) for key, value in element.items()}
        for element in observation.get("dom_elements", ())
    ]
    tags = {str(element.get("tag", "")).lower() for element in elements}
    classes = " ".join(str(element.get("classes", "")).lower() for element in elements)
    ids = " ".join(str(element.get("id", "")).lower() for element in elements)
    roles = {str(element.get("role", "")).lower() for element in sidecar}

    bool_values = [
        element.get("value") for element in elements
        if str(element.get("tag", "")).lower() in {"input_checkbox", "input_radio"}
        and isinstance(element.get("value"), (bool, np.bool_))
    ]
    interactive = [
        element for element in sidecar
        if element.get("interactive") is True
    ]
    labels = {
        "role_dialog": "dialog" in roles or "ui-dialog" in classes or "dialog" in ids,
        "role_button": bool(tags & {"button", "input_button", "input_submit"}),
        "role_textbox": bool(tags & {"input_text", "input_password", "textarea"}),
        "role_checkbox": "input_checkbox" in tags,
        "role_radio": "input_radio" in tags,
        "role_select_option": bool(tags & {"select", "option"}),
        "any_checked": any(value is True for value in bool_values)
            or any(element.get("checked") is True for element in sidecar),
        "any_unchecked": any(value is False for value in bool_values)
            or any(element.get("checked") is False for element in sidecar),
        "any_selected": any(element.get("selected") is True for element in sidecar),
        "any_disabled": any(element.get("disabled") is True for element in interactive),
        "any_enabled": any(element.get("disabled") is False for element in interactive),
    }

    words: set[str] = set()
    for element in elements:
        for key in ("text", "value"):
            value = element.get(key)
            if value not in (None, "", True, False):
                words.update(match.group(0).lower() for match in WORD_RE.finditer(str(value)))
    return {
        "state": {key: bool(labels[key]) for key in STATE_LABELS},
        "visible_words": sorted(words),
    }


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

