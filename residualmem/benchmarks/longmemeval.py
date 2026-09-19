"""LongMemEval-V2 observation and QA adaptation primitives.

This module deliberately contains no model calls.  It defines the data contract
used by the Q-Former/reader pipeline and the deterministic AXTree anchor view.
Keeping these operations model-free makes the train/eval split auditable and
lets the same serializer be used for WebChain training and LongMemEval
evaluation.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator


LONGMEMEVAL_OBSERVATION_PROTOCOL = "longmemeval-v2-observation-v1"
LONGMEMEVAL_QA_PROTOCOL = "longmemeval-external-qa-v1"
LONGMEMEVAL_ANCHOR_PROTOCOL = "longmemeval-axtree-anchor-v1"

                                                                         
                                                                             
                                                                          
INTERACTIVE_ROLES = frozenset({
    "link", "button", "textbox", "searchbox", "textarea", "menuitem",
    "menuitemcheckbox", "menuitemradio", "tab", "checkbox", "radio",
    "combobox", "listbox", "option", "treeitem", "slider", "spinbutton",
    "switch", "search",
})
TEXT_ROLES = frozenset({"heading", "StaticText", "text"})
STRUCTURAL_NAMED_ROLES = frozenset({
    "RootWebArea", "dialog", "complementary", "contentinfo", "banner",
    "navigation", "main", "article", "form", "alert", "status", "tooltip",
    "menu", "menubar", "toolbar", "tablist", "tree", "rowheader",
    "columnheader", "gridcell", "cell",
})
ANCHOR_ROLES = INTERACTIVE_ROLES | TEXT_ROLES | STRUCTURAL_NAMED_ROLES


@dataclasses.dataclass(frozen=True)
class LongMemEvalObservation:
    """One real LongMemEval/WebChain state, without a benchmark answer."""

    trajectory_id: str
    state_index: int
    step: int
    url: str
    action: str | None
    thought: str | None
    accessibility_tree: str
    screenshot: Path

    def __post_init__(self) -> None:
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be non-empty")
        if self.state_index < 0 or self.step < 0:
            raise ValueError("state_index and step must be non-negative")
        if not self.url.strip():
            raise ValueError("url must be non-empty")
        if not isinstance(self.accessibility_tree, str) or not self.accessibility_tree.strip():
            raise ValueError("accessibility_tree must be non-empty")
        if not self.screenshot.is_file():
            raise FileNotFoundError(self.screenshot)

    @property
    def state_id(self) -> str:
        return f"{self.trajectory_id}:{self.state_index}"

    @property
    def image_sha256(self) -> str:
        digest = hashlib.sha256()
        with self.screenshot.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    def as_record(self, *, anchor_text: str | None = None) -> dict[str, Any]:
        """Return the query-independent record consumed by cache builders."""
        record = {
            "protocol": LONGMEMEVAL_OBSERVATION_PROTOCOL,
            "trajectory_id": self.trajectory_id,
            "state_id": self.state_id,
            "state_index": self.state_index,
            "step": self.step,
            "url": self.url,
            "action": self.action,
            "thought": self.thought,
            "accessibility_tree": canonicalize_axtree(self.accessibility_tree),
            "screenshot": str(self.screenshot.resolve()),
            "screenshot_sha256": self.image_sha256,
        }
        if anchor_text is not None:
            record["anchor_text"] = anchor_text
        return record


def canonicalize_axtree(text: str) -> str:
    """Canonicalize LME's text AXTree without deleting answer-bearing values.

    This is intentionally milder than AMA's medium cleaner.  LME questions
    often ask about exact values and state flags, so false/disabled/selected
    attributes and non-numeric browser IDs remain untouched.  Only line-ending,
    indentation and horizontal whitespace differences are normalized.
    """
    if not isinstance(text, str):
        raise TypeError("AXTree must be a string")
    output: list[str] = []
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not raw.strip():
            continue
        leading = raw[: len(raw) - len(raw.lstrip(" \t"))]
        body = re.sub(r"[ \t]+", " ", raw[len(leading):]).strip()
        output.append(f"{leading.expandtabs(2)}{body}")
    if not output:
        raise ValueError("AXTree contains no non-empty lines")
    return "\n".join(output)


def _split_ax_line(line: str) -> tuple[str | None, str, str, str] | None:
    """Parse an AXTree line, accepting numeric and opaque IDs."""
    stripped = line.lstrip()
    node_id: str | None = None
    if stripped.startswith("["):
        end = stripped.find("]")
        if end <= 1:
            return None
        node_id = stripped[1:end]
        stripped = stripped[end + 1 :].lstrip()
    role_match = re.match(r"([^\s]+)(?:\s+|$)", stripped)
    if role_match is None:
        return None
    role = role_match.group(1)
    rest = stripped[role_match.end() :].lstrip()
    if not rest or rest[0] not in "'\"":
                                                                              
                                                                                 
        return (node_id, role, "", rest)
    quote = rest[0]
    chars: list[str] = []
    escaped = False
    closing = None
    for index, char in enumerate(rest[1:], start=1):
        if escaped:
            chars.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote:
            closing = index
            break
        else:
            chars.append(char)
    if closing is None:
        return None
    return node_id, role, "".join(chars), rest[closing + 1 :].strip()


def iter_anchor_lines(text: str) -> Iterator[str]:
    """Yield deterministic, answer-bearing element lines in document order."""
    for raw in canonicalize_axtree(text).splitlines():
        parsed = _split_ax_line(raw)
        if parsed is None:
            continue
        node_id, role, name, attrs = parsed
        if role not in ANCHOR_ROLES:
            continue
        if role not in INTERACTIVE_ROLES and not name.strip():
            continue
        prefix = f"[{node_id}] " if node_id is not None else ""
        rendered = f"{prefix}{role} '{name}'"
        if attrs:
            rendered += f" {attrs}"
        yield rendered


def extract_anchor(text: str, *, max_bytes: int | None = None) -> dict[str, Any]:
    """Extract a query-independent AXTree side channel.

    Complete lines are selected; an exact value is never cut in the middle.
    When a byte budget is requested, document order is retained and lower
    priority lines are dropped only after the budget is reached.  The unbounded
    view is used for offline audits, while deployment supplies a fixed budget.
    """
    lines = list(iter_anchor_lines(text))
    selected: list[str] = []
    dropped = 0
    used = 0
    for line in lines:
        encoded = (line + "\n").encode("utf-8")
        if max_bytes is not None and selected and used + len(encoded) > max_bytes:
            dropped += 1
            continue
        if max_bytes is not None and not selected and len(encoded) > max_bytes:
                                                                           
                                                                               
            selected.append(line)
            used += len(encoded)
            dropped += len(lines) - 1
            break
        selected.append(line)
        used += len(encoded)
    rendered = "\n".join(selected)
    return {
        "protocol": LONGMEMEVAL_ANCHOR_PROTOCOL,
        "text": rendered,
        "lines": len(selected),
        "lines_total": len(lines),
        "dropped_lines": dropped,
        "bytes": len(rendered.encode("utf-8")),
        "byte_budget": max_bytes,
    }


def normalize_action(action: Any) -> str | None:
    """Make incoming LME actions stable without exposing future state."""
    if action is None:
        return None
    if isinstance(action, str):
        return re.sub(r"\s+", " ", action).strip() or None
    if isinstance(action, dict):
                                                                              
                                                                              
        return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    raise TypeError(f"unsupported action type: {type(action).__name__}")


def read_lme_trajectory(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    """Load and validate one official LME trajectory JSON object."""
    source = Path(path)
    row = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise ValueError(f"{source} must contain a JSON object")
    trajectory_id = row.get("id")
    states = row.get("states")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError(f"{source}: missing trajectory id")
    if not isinstance(states, list) or not states:
        raise ValueError(f"{source}: trajectory has no states")
    base = Path(root) if root is not None else source.parent
    observations: list[LongMemEvalObservation] = []
    for position, state in enumerate(states):
        if not isinstance(state, dict):
            raise ValueError(f"{source}: state {position} is not an object")
        screenshot = Path(str(state.get("screenshot") or ""))
        if not screenshot.is_absolute():
            screenshot = base / screenshot
        observations.append(LongMemEvalObservation(
            trajectory_id=trajectory_id,
            state_index=int(state.get("state_index", position)),
            step=int(state.get("step", position)),
            url=str(state.get("url") or ""),
            action=normalize_action(state.get("action")),
            thought=(str(state["thought"]) if state.get("thought") is not None else None),
            accessibility_tree=canonicalize_axtree(str(state.get("accessibility_tree") or "")),
            screenshot=screenshot.resolve(),
        ))
    return {
        "id": trajectory_id,
        "domain": row.get("domain"),
        "environment": row.get("environment"),
        "goal": row.get("goal"),
        "outcome": row.get("outcome"),
        "start_url": row.get("start_url"),
        "observations": observations,
    }


__all__ = [
    "LONGMEMEVAL_ANCHOR_PROTOCOL",
    "LONGMEMEVAL_OBSERVATION_PROTOCOL",
    "LONGMEMEVAL_QA_PROTOCOL",
    "LongMemEvalObservation",
    "canonicalize_axtree",
    "extract_anchor",
    "iter_anchor_lines",
    "normalize_action",
    "read_lme_trajectory",
]
