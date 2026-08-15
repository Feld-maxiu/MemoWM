"""Dependency-light constants and action canonicalisation for the v8 WM line."""
from __future__ import annotations

import dataclasses
import re
from typing import Mapping

import numpy as np


PROTOCOL = "residualmem_v8_discrete_wm_v1"
CACHE_FORMAT_VERSION = 1
MAX_HISTORY = 7
NUM_LATENT_TOKENS = 64
NUM_SUBSPACES = 32
NUM_CATEGORIES = 256
NUM_OBSERVATION_SLOTS = 64
MAX_PAYLOAD_BYTES = 40

SPLITS = ("train", "validation", "test")
SPLIT_IDS = {name: index for index, name in enumerate(SPLITS)}

# Three values occur in the frozen data. The remaining IDs are model-only
# sentinels and can never be emitted by the data canonicaliser.
ACTION_TYPE_IDS = {
    "CLICK": 0,
    "FILL": 1,
    "SELECT_OPTION": 2,
    "PAD": 3,
    "MASK": 4,
    "UNK": 5,
}
ACTION_TYPE_NAMES = tuple(ACTION_TYPE_IDS)

# These are exactly the seven tags in the frozen v8 action log. SELECT_OPTION is
# canonicalised from the logged option bid to the select bid, but ``option`` is
# retained in the vocabulary so the manifest remains auditable against raw data.
TAG_IDS = {
    "button": 0,
    "input_checkbox": 1,
    "input_radio": 2,
    "input_text": 3,
    "label": 4,
    "option": 5,
    "select": 6,
    "UNK": 7,
}
TAG_PAD_ID = 8
TAG_MASK_ID = 9
NUM_MODEL_TAGS = 10

REF_PAD_ID = 64
REF_MASK_ID = 65
REF_UNK_ID = 66
NUM_MODEL_REFS = 67

POLICY_IDS = {"scripted": 0, "random": 1}
POLICY_NAMES = ("scripted", "random")

VARIANTS = (
    "t_only",
    "no_action",
    "structural_action",
    "no_history",
    "full",
)

# Diagnostic-only variants. Deliberately kept out of ``VARIANTS``: statistics.main
# derives its required run set from that tuple, so adding one here would silently
# turn a diagnostic into a preregistration requirement (three more seeds before
# any formal statistic can be produced).
DEV_VARIANTS = (
    "state_only",
    "struct_no_history",
)

_ROW_RE = re.compile(
    r"(?:^|\n)<ref=(?P<ref>-?\d+)\s+parent=(?P<parent>-?\d+)\s+"
    r"tag=(?P<tag>[^\s/>]+)"
)

# Same rows, but keeping the attribute tail so target semantics can be read out.
_NODE_RE = re.compile(
    r"<ref=(-?\d+)\s+parent=(-?\d+)\s+tag=([^\s/>]+)([^>]*)/>"
)
_TEXT_RE = re.compile(r'\stext="([^"]*)"')

# The element's own text; else a child's (checkbox/option wrapped in a label);
# else a sibling input's (autocomplete, where the label is empty and the field
# carries the caption). Measured recoverability across 599 CLICK/label samples:
# 76%. The remaining 24% is login-user, whose label text is absent from the
# serialisation entirely -- label, sibling input and raw AXTree name are all
# empty -- so it cannot be recovered without regenerating the dom.
MAX_TARGET_BYTES = 40


def parse_dom_nodes(compact: str) -> dict[int, tuple[int, str, str]]:
    """``ref -> (parent, tag, attribute tail)``."""
    nodes: dict[int, tuple[int, str, str]] = {}
    for match in _NODE_RE.finditer(compact or ""):
        nodes[int(match.group(1))] = (
            int(match.group(2)), match.group(3), match.group(4)
        )
    return nodes


def _node_text(attributes: str) -> str:
    found = _TEXT_RE.search(attributes or "")
    return found.group(1) if found else ""


def extract_target_text(compact: str, ref: int) -> str:
    """Accessible name of the element acted on -- four-level fallback.

    ``ref`` must already be the ref that was actually acted on (for
    SELECT_OPTION that is the parent ``select``, not the option).
    """
    nodes = parse_dom_nodes(compact)
    if ref not in nodes:
        return ""
    parent, _tag, attributes = nodes[ref]
    own = _node_text(attributes)
    if own:
        return own
    for child_ref, (child_parent, _t, child_attributes) in nodes.items():
        if child_parent == ref and child_ref != ref:
            text = _node_text(child_attributes)
            if text:
                return text
    for sibling_ref, (sibling_parent, tag, sibling_attributes) in nodes.items():
        if sibling_parent == parent and sibling_ref != ref and tag.startswith("input"):
            text = _node_text(sibling_attributes)
            if text:
                return text
    return ""


@dataclasses.dataclass(frozen=True)
class Action:
    """Canonical action that describes what was actually sent to BrowserGym."""

    type_id: int
    tag_id: int
    ref: int
    payload: bytes
    policy_id: int
    # Which element was acted on. Kept separate from payload -- which says what
    # was applied to it -- so G_semantic and G_payload stay separable.
    target: bytes = b""

    @property
    def payload_length(self) -> int:
        return len(self.payload)

    @property
    def target_length(self) -> int:
        return len(self.target)

    def padded_target(self) -> np.ndarray:
        output = np.zeros((MAX_TARGET_BYTES,), np.uint8)
        if self.target:
            output[: len(self.target)] = np.frombuffer(self.target, np.uint8)
        return output

    def padded_payload(self) -> np.ndarray:
        output = np.zeros((MAX_PAYLOAD_BYTES,), np.uint8)
        if self.payload:
            output[: len(self.payload)] = np.frombuffer(self.payload, np.uint8)
        return output


def _axtree_rows(compact: str) -> dict[int, tuple[int, str]]:
    rows: dict[int, tuple[int, str]] = {}
    for match in _ROW_RE.finditer(compact or ""):
        ref = int(match.group("ref"))
        if ref in rows:
            raise ValueError(f"duplicate AXTree ref {ref}")
        rows[ref] = (int(match.group("parent")), match.group("tag"))
    return rows


def canonicalize_action(
    raw: Mapping[str, object], compact_axtree: str = "", *, strict: bool = True
) -> Action:
    """Convert a collector record into the exact action available to the WM.

    The v8 collector logged an option's bid for ``SELECT_OPTION`` even though
    BrowserGym executed the action on its parent ``select``. Recovering that
    parent here prevents the model from receiving a ref that was never acted on.
    """
    if raw is None:
        raise ValueError("a transition action cannot be null")
    kind = str(raw.get("type", "UNK")).upper()
    if kind not in {"CLICK", "FILL", "SELECT_OPTION"}:
        if strict:
            raise ValueError(f"unsupported action type {kind!r}")
        kind = "UNK"

    try:
        ref = int(raw["ref"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid action ref: {raw.get('ref')!r}") from exc
    tag = str(raw.get("tag", "UNK"))

    if kind == "SELECT_OPTION":
        rows = _axtree_rows(compact_axtree)
        if ref not in rows:
            raise ValueError(f"SELECT_OPTION ref {ref} is absent from source AXTree")
        parent, raw_tag = rows[ref]
        if raw_tag != "option":
            raise ValueError(
                f"SELECT_OPTION ref {ref} resolves to {raw_tag!r}, expected 'option'"
            )
        parent_row = rows.get(parent)
        if parent_row is None or parent_row[1] != "select":
            raise ValueError(
                f"SELECT_OPTION ref {ref} has invalid select parent {parent}: {parent_row}"
            )
        ref, tag = parent, "select"

    if not 0 <= ref < 64:
        raise ValueError(f"action ref {ref} is outside the frozen 0..63 bound")
    if tag not in TAG_IDS:
        if strict:
            raise ValueError(f"unsupported action tag {tag!r}")
        tag = "UNK"

    text = raw.get("text")
    payload = b"" if text is None else str(text).encode("utf-8")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"UTF-8 action payload is {len(payload)} bytes; maximum is "
            f"{MAX_PAYLOAD_BYTES} and truncation is forbidden"
        )
    policy = str(raw.get("policy", ""))
    if policy not in POLICY_IDS:
        raise ValueError(f"unsupported collection policy {policy!r}")
    # Extracted after the SELECT_OPTION ref has been resolved to its parent
    # select, so the target describes the control rather than the chosen option.
    target = extract_target_text(compact_axtree, ref).encode("utf-8")
    if len(target) > MAX_TARGET_BYTES:
        target = target[:MAX_TARGET_BYTES]
        while target and (target[-1] & 0xC0) == 0x80:  # keep valid UTF-8
            target = target[:-1]
    return Action(
        type_id=ACTION_TYPE_IDS[kind],
        tag_id=TAG_IDS[tag],
        ref=ref,
        payload=payload,
        policy_id=POLICY_IDS[policy],
        target=target,
    )


def action_side_information_bits(
    action: Action, *, include_payload: bool, include_target: bool = False
) -> int:
    """Fixed audit serialization from the frozen evidence-gate plan.

    ``target`` is billed on the same terms as ``payload``: it is enriched action
    metadata that must be transmitted, not a free oracle.
    """
    structural = 2 + 3 + 6
    total = structural
    if include_target:
        total += 6 + 8 * action.target_length
    if include_payload:
        total += 6 + 8 * action.payload_length
    return total


def validate_variant(variant: str, *, allow_dev: bool = False) -> str:
    variant = str(variant).lower().replace("-", "_")
    permitted = VARIANTS + DEV_VARIANTS if allow_dev else VARIANTS
    if variant not in permitted:
        raise ValueError(f"unknown variant {variant!r}; expected one of {permitted}")
    return variant
