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


def read_jsonl(path: str | Path) -> list[dict]:
    """One JSON object per line, split on newlines only.

    ``read_text().splitlines()`` is the obvious spelling and is wrong here.
    ``splitlines`` also breaks on U+2028, U+2029, \x0b, \x0c and U+0085, and
    ``json.dumps(..., ensure_ascii=False)`` -- which every writer in this repo
    uses -- leaves those characters raw inside strings. MolmoWeb shard 15 has 11
    page titles containing U+2028; each was cut in half and the whole caption
    stage died on "Unterminated string". The pilot's two shards happened to
    contain none, so nothing failed until the corpus grew.
    """
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:          # iteration splits on \n alone
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


AXTREE_ROLE_TO_TAG: dict[str, str] = {
    # controls -- mapped onto the DOM tag names so that key_pooling's VALUE_TAGS /
    # STATE_VALUE_TAGS / INTERACTIVE_TEXT_TAGS keep matching without edits
    "checkbox": "input_checkbox",
    "radio": "input_radio",
    "textbox": "input_text",
    "combobox": "select",
    "listbox": "select",
    "option": "option",
    "button": "button",
    "link": "a",
    # containers and text
    "LabelText": "label",
    # StaticText -> "t" keeps detail_candidates' existing "text node whose parent
    # is interactive" fallback alive.  With AXTree the name usually sits on the
    # control itself, so that path becomes a backstop rather than the main route.
    "StaticText": "t",
    "paragraph": "p",
    "dialog": "dialog",
    "alertdialog": "dialog",
    "status": "status",
    "RootWebArea": "root",
}
# Layout-only nodes. Dropping them is most of the 4.1x token saving over compact
# DOM, and none of them can carry an accessible name a probe would need.
AXTREE_SKIP_ROLES: frozenset[str] = frozenset({
    "generic", "none", "InlineTextBox", "LineBreak", "MenuListPopup",
})


def _axtree_value(node: dict, key: str):
    """Unwrap Chrome's ``{"type": ..., "value": ...}`` envelope."""
    field = node.get(key)
    return field.get("value") if isinstance(field, dict) else field


def _axtree_properties(node: dict) -> dict:
    return {
        item["name"]: _axtree_value(item, "value")
        for item in node.get("properties") or ()
        if isinstance(item, dict) and "name" in item
    }


def axtree_rows(axtree_object, sidecar: Sequence[dict] = ()) -> list[dict]:
    """Reduce a BrowserGym AXTree to the attribute rows the wire format carries.

    Split out from ``compact_axtree`` so that the serialization and the probe
    labels are computed from one list of rows. Deriving labels from the same rows
    Qwen is shown makes it impossible for a label to describe something the input
    does not contain -- the failure mode where a probe looks solvable because the
    label came from a richer source than the representation.
    """
    nodes = axtree_object.get("nodes", axtree_object) if isinstance(axtree_object, dict) else axtree_object
    tampered = {
        str(item.get("bid")): bool(item.get("tampered"))
        for item in sidecar or ()
    }

    by_node_id = {node["nodeId"]: node for node in nodes}

    def _named_controls(node: dict) -> set[str]:
        """Accessible names already carried by a kept control near this node."""
        parent = by_node_id.get(node.get("parentId"))
        if parent is None:
            return set()
        names = set()
        if parent.get("browsergym_id") is not None:
            value = _axtree_value(parent, "name")
            if value:
                names.add(str(value).strip())
        for sibling_id in parent.get("childIds") or ():
            sibling = by_node_id.get(sibling_id)
            if sibling is None or sibling is node:
                continue
            if sibling.get("browsergym_id") is None:
                continue
            value = _axtree_value(sibling, "name")
            if value:
                names.add(str(value).strip())
        return names

    def is_kept(node: dict) -> bool:
        role = _axtree_value(node, "role")
        if node.get("ignored") or role in AXTREE_SKIP_ROLES:
            return False
        if node.get("browsergym_id") is not None:
            return True
        # A name with no bid is body text -- StaticText never carries a bid, yet
        # in scroll-text-2 and copy-paste it holds the passage the task is about.
        # But under a labelled control it merely repeats that control's accessible
        # name, and emitting both doubles the detail candidates for every label,
        # which halves the effective slot budget and is exactly the crowding the
        # switch to AXTree was meant to remove.
        name = _axtree_value(node, "name")
        if not name:
            return False
        return str(name).strip() not in _named_controls(node)

    kept = {node["nodeId"] for node in nodes if is_kept(node)}

    def ref_of(node: dict) -> str:
        if _axtree_value(node, "role") == "RootWebArea":
            return "0"
        bid = node.get("browsergym_id")
        # -1 for text nodes, matching compact_dom's convention for the same thing.
        return str(bid) if bid is not None else "-1"

    def nearest_kept_ancestor(node: dict) -> str:
        cursor = node.get("parentId")
        while cursor is not None:
            parent = by_node_id.get(cursor)
            if parent is None:
                break
            if cursor in kept:
                return ref_of(parent)
            cursor = parent.get("parentId")
        return "0"

    # Depth-first from the root, following childIds: the raw ``nodes`` array is
    # not in document order, and both the position slots and Qwen's own reading
    # order depend on it.
    roots = [n for n in nodes if n.get("parentId") is None or n.get("parentId") not in by_node_id]
    ordered: list[dict] = []
    seen: set[str] = set()
    stack = list(reversed(roots))
    while stack:
        node = stack.pop()
        if node["nodeId"] in seen:
            continue
        seen.add(node["nodeId"])
        ordered.append(node)
        stack.extend(
            by_node_id[child]
            for child in reversed(node.get("childIds") or ())
            if child in by_node_id
        )

    rows: list[dict] = []
    for node in ordered:
        if node["nodeId"] not in kept:
            continue
        role = _axtree_value(node, "role")
        properties = _axtree_properties(node)
        ref = ref_of(node)
        tag = AXTREE_ROLE_TO_TAG.get(role, str(role).lower())
        if tag == "input_text" and properties.get("protected"):
            tag = "input_password"

        # One "value" attribute, three sources -- v2_data reads it as a string for
        # textboxes and as "true"/"false" for checkables, exactly as compact_dom did.
        value = None
        if tag in {"input_checkbox", "input_radio"}:
            checked = properties.get("checked")
            if checked is not None:
                value = "true" if checked in (True, "true") else "false"
        elif tag == "option":
            selected = properties.get("selected")
            if selected is not None:
                value = "true" if selected in (True, "true") else "false"
        else:
            raw = _axtree_value(node, "value")
            if raw not in (None, ""):
                value = str(raw)

        name = _axtree_value(node, "name")
        rows.append({
            "ref": ref,
            "parent": nearest_kept_ancestor(node),
            "tag": tag,
            "text": None if name in (None, "") else str(name),
            "value": value,
            # Fixed width of four: v2_data rejects any other length, and reads
            # [0] as focused and [1] as tampered.
            "flags": [
                int(bool(properties.get("focused"))),
                int(tampered.get(ref, False)),
                0,
                int(not [c for c in node.get("childIds") or () if c in kept]),
            ],
            "focusable": bool(properties.get("focusable")),
        })
    return rows


def compact_axtree(axtree_object, sidecar: Sequence[dict] = ()) -> str:
    """Serialize a BrowserGym AXTree into the same wire format as ``compact_dom``.

    The point of reusing the format is that nothing downstream has to change:
    ``key_pooling.parse_dom_spans`` and ``v2_data.parse_compact_dom`` are generic
    line-oriented ``<key=value/>`` parsers, and mapping AXTree roles onto the
    existing tag vocabulary keeps the candidate-selection rules matching too.

    What does change is where the target-to-element binding lives. In compact DOM
    the label text sits on a separate ``t`` node and the control it names has to
    be found through ``parent``; here the accessible name is an attribute of the
    control itself, so the binding is literal adjacency inside one span.

    Geometry is dropped: it is 66.3% of compact DOM's tokens and no probe label
    depends on it.
    """
    lines = []
    for row in axtree_rows(axtree_object, sidecar):
        pieces = [f"ref={row['ref']}", f"parent={row['parent']}", f"tag={row['tag']}"]
        if row["text"] is not None:
            pieces.append("text=" + json.dumps(row["text"], ensure_ascii=False))
        if row["value"] is not None:
            pieces.append("value=" + json.dumps(row["value"], ensure_ascii=False))
        if any(row["flags"]):
            pieces.append("flags=" + ",".join(str(flag) for flag in row["flags"]))
        lines.append("<" + " ".join(pieces) + "/>")
    return "\n".join(lines)


def derive_probe_labels_axtree(rows: Sequence[dict]) -> dict:
    """``derive_probe_labels`` for AXTree rows, computed from the serialized rows.

    ``any_disabled`` is always false here: Chrome only emits a ``disabled``
    property when something is disabled, and none of the twelve pilot tasks ever
    disables a control. That matches the DOM pipeline, where ``button_disabled``
    is already listed in ``v2_data.UNSUPPORTED_LABELS`` for having zero positives.
    """
    tags = {row["tag"] for row in rows}
    checkables = [row for row in rows if row["tag"] in {"input_checkbox", "input_radio"}]
    options = [row for row in rows if row["tag"] == "option"]
    labels = {
        "role_dialog": "dialog" in tags,
        "role_button": bool(tags & {"button", "input_button", "input_submit"}),
        "role_textbox": bool(tags & {"input_text", "input_password", "textarea"}),
        "role_checkbox": "input_checkbox" in tags,
        "role_radio": "input_radio" in tags,
        "role_select_option": bool(tags & {"select", "option"}),
        "any_checked": any(row["value"] == "true" for row in checkables),
        "any_unchecked": any(row["value"] == "false" for row in checkables),
        "any_selected": any(row["value"] == "true" for row in options),
        "any_disabled": False,
        "any_enabled": any(row["focusable"] for row in rows),
    }
    words: set[str] = set()
    for row in rows:
        for key in ("text", "value"):
            if row[key] not in (None, "", "true", "false"):
                words.update(match.group(0).lower() for match in WORD_RE.finditer(row[key]))
    return {
        "state": {key: bool(labels[key]) for key in STATE_LABELS},
        "visible_words": sorted(words),
    }


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

