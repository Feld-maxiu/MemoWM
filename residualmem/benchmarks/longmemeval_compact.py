"""Compact exact-value view for LongMemEval raw-state retrieval.

The view is the text payload that carries the exact values a compressed latent
state cannot reconstruct.  Five rules build it:

1. keep every interactive control as a compact skeleton (no line numbers, but
   the original node id, role, name and state-bearing attributes);
2. keep literal values, verbatim status/error messages and structural labels;
3. keep per-state State/URL/Page/Action context;
4. keep only the labels that bind to a value, and compress large repeated
   groups (grids/tables) instead of dumping every row;
5. drop exact duplicates inside an observation and inside a slice.

Rule 4 was added after the review that showed rule 2 deleting short labels
such as ``columnheader 'Qty'`` or ``StaticText 'Summary'`` whose strings are
the answer, and after the payload audit that showed ~46% of the kept literal
tokens were grid timestamps and ~28% were bare numbers from large admin grids.
"""
from __future__ import annotations

import json
import collections
import re

from .longmemeval_anchor import CONTROLS, MESSAGE_RE, VALUE_RE, nodes

PROTOCOL = "longmemeval-compact-exact-v2"

_FALSE_VALUES = frozenset(("false", "0", "none", ""))

                                                                               
_BOOLEAN_ATTRS = frozenset(("checked", "selected", "disabled", "expanded"))

_STATE_ATTRS = (
    "value",
    "placeholder",
    "href",
    "checked",
    "selected",
    "disabled",
    "expanded",
)

                                                                            
                                                                
CONTROL_ROLES = frozenset(
    {item.casefold() for item in CONTROLS}
    | {"disclosuretriangle", "switch", "toggle", "treeitem", "slider"}
)

                                                                            
LABEL_ROLES = frozenset(
    ("columnheader", "rowheader", "heading", "label", "caption", "tab", "legend", "term")
)

                                                                         
                                                                            
                                                                               
                           
LABEL_TEXT_ROLES = frozenset(("statictext", "cell", "gridcell"))

                                                                               
                                                      
_GROUP_ROLES = frozenset(("table", "grid", "treegrid", "tree"))
_ROW_ROLE = "row"
_BIG_GROUP_ROWS = 5

                                                                          
                                                                         
_REPEATABLE_VALUE_KINDS = ("timestamp", "bare_short")

_BOILERPLATE_RE = re.compile(
    r"copyright|all rights reserved|privacy (?:and cookie )?policy|cookie "
    r"policy|terms of (?:service|use)|our newsletter|©|®|^skip to (?:main|content)",
    re.I,
)
_FACET_RE = re.compile(
    r"^(?:page|show|display)\s*\d+$|"
    r"^\d+\s*(?:item|items|result|results|product|products|review|reviews|"
    r"comment|comments|vote|votes|star|stars|point|points)$",
    re.I,
)
                                                                             
                                                                             
                                                                              
                                        
_BARE_SMALL_RE = re.compile(r"^\d{1,4}$")
_BARE_LONG_RE = re.compile(r"^\d{5,}$")
_TIME_ONLY_RE = re.compile(
    r"^(?:\w{3,9}\.?\s+\d{1,2},?\s+\d{4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})"
    r"(?:[,\s]+\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?)?$",
    re.I,
)
_SENTENCE_RE = re.compile(r"[.!?]")
_STOP_LABELS = frozenset(("of", "to", "and", "or", "the", "a", "an", "by", "for", "in", "on", "with"))


def _label_like(name: str) -> bool:
    """A short noun phrase: the shape a value's label has, not prose."""
    return (
        len(name.strip()) >= 3
        and name.strip().casefold() not in _STOP_LABELS
        and len(name) <= 48
        and len(name.split()) <= 6
        and not _SENTENCE_RE.search(name)
    )


def _action_text(state: dict) -> str:
    value = state.get("incoming_action_text") or state.get("action")
    if value is None:
        return "<none>"
    if isinstance(value, dict):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return " ".join(str(value).split())


def _page_title(parsed: list[dict]) -> str:
    return next(
        (node["name"] for node in parsed if node["role"] == "RootWebArea"),
        "",
    )


def _node_line(node: dict) -> str:
                                                                            
                                                                           
                                                  
    parts = [f"[{node['id']}]"] if node["id"] else []
    parts.append(node["role"])
    if node["name"]:
        parts.append(f"'{node['name']}'")
    for key in _STATE_ATTRS:
        value = node["attrs"].get(key)
        if not value:
            continue
        if key in _BOOLEAN_ATTRS and str(value).casefold() in _FALSE_VALUES:
            continue
        parts.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
    return " ".join(parts)


def _role(node: dict) -> str:
    return str(node["role"]).casefold()


def _literal(node: dict) -> bool:
    text = " ".join([node["name"], *[str(v) for v in node["attrs"].values()]])
    return bool(VALUE_RE.search(text) or MESSAGE_RE.search(node["name"]))


def _value_kind(node: dict) -> str:
    """Classify a value-bearing node for the rule-4 retention policy."""
    name = node["name"].strip()
    if _TIME_ONLY_RE.match(name):
        return "timestamp"
    if _BARE_SMALL_RE.match(name):
        return "bare_short"
    if _BARE_LONG_RE.match(name):
        return "identifier"
    if _literal(node):
        return "literal"
    return ""


def _annotate(parsed: list[dict]) -> list[dict]:
    """Attach parent, row-group and row-ancestor identity to every node."""
    stack: list[dict] = []
    for position, node in enumerate(parsed):
        depth = node["depth"]
        while stack and stack[-1]["depth"] >= depth:
            stack.pop()
        parent = stack[-1] if stack else None
        group = next(
            (item for item in reversed(stack) if _role(item) in _GROUP_ROLES), None
        )
        row = next(
            (item for item in reversed(stack) if _role(item) == _ROW_ROLE), None
        )
        cell = next(
            (
                item
                for item in reversed(stack)
                if _role(item) in ("cell", "gridcell")
            ),
            None,
        )
        node["_parent"] = parent["_position"] if parent else None
        node["_group"] = group["_position"] if group else None
        node["_row"] = row["_position"] if row else None
        node["_cell"] = cell["_position"] if cell else None
        node["_position"] = position
        stack.append(node)
    return parsed


def _group_row_counts(parsed: list[dict]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for node in parsed:
        if _role(node) != _ROW_ROLE or node["_group"] is None:
            continue
        counts[node["_group"]] = counts.get(node["_group"], 0) + 1
    return counts


def _container_key(parsed: list[dict], node: dict) -> int:
    """The nearest grouping container: a table/grid role, else the tree root.

    Filtered trees drop their container roles, so the fallback walks to the
    top-most ancestor: a page whose AXTree lost its ``table`` line still counts
    its repeated timestamps as one log instead of one per cell.
    """
    if node["_group"] is not None:
        return int(node["_group"])
    position = node["_parent"]
    if position is None:
        return -1
    while parsed[position]["_parent"] is not None:
        position = parsed[position]["_parent"]
    return int(position)


def _kept_positions(parsed: list[dict]) -> set[int]:
    """Rules 1-4: decide which nodes survive, before de-duplication."""
    rows_per_group = _group_row_counts(parsed)
    control_names = {
        node["name"].strip()
        for node in parsed
        if _role(node) in CONTROL_ROLES and node["name"]
    }
                                                                           
                                                                
    repeats: collections.Counter = collections.Counter()
    for node in parsed:
        kind = _value_kind(node)
        if kind in _REPEATABLE_VALUE_KINDS:
            repeats[(_container_key(parsed, node), kind)] += 1

    kept: set[int] = set()
    for node in parsed:
        role = _role(node)
        name = node["name"]
        if _BOILERPLATE_RE.search(name) or _FACET_RE.match(name):
                                                                              
                                  
            if role not in CONTROL_ROLES:
                continue
        group = node["_group"]
        big_group = group is not None and rows_per_group.get(group, 0) > _BIG_GROUP_ROWS
        if role in CONTROL_ROLES or role in LABEL_ROLES:
            kept.add(node["_position"])
            continue
        kind = _value_kind(node)
        if kind:
                                                                             
                                                                               
                                                                             
                                                         
            if kind in _REPEATABLE_VALUE_KINDS:
                bound = node["_row"] is not None or node["_cell"] is not None
                repeated = (
                    repeats[(_container_key(parsed, node), kind)] > _BIG_GROUP_ROWS
                )
                if not bound or repeated:
                    continue
            kept.add(node["_position"])
            continue
        if role in LABEL_TEXT_ROLES and name and _label_like(name):
                                                                              
                                                                               
                                                               
            if name.strip() in control_names:
                continue
            if role == "statictext" or not big_group:
                kept.add(node["_position"])
    return kept


def _dedup(lines: list[str], seen: set[str]) -> list[str]:
    output = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        output.append(line)
    return output


def compact_state_lines(state: dict) -> list[str]:
    """Rules 1-5 for one state, without the rule-3 header lines."""
    tree = str(state.get("axtree") or state.get("synthetic_axtree") or "")
    parsed = _annotate(list(nodes(tree)))
    kept = _kept_positions(parsed)
    lines = [
        _node_line(node)
        for node in parsed
        if node["_position"] in kept and (node["name"] or node["attrs"])
    ]
    return _dedup(lines, set())


def compact_axtree_text(state: dict) -> str:
    """Rules 1-5 for one state as a single text block."""
    return "\n".join(compact_state_lines(state))


def _state_header(state: dict, parsed: list[dict]) -> list[str]:
    lines = [
        f"State {state.get('state_index', '')}",
        f"URL: {state.get('url', '<unknown>')}",
    ]
    title = _page_title(parsed)
    if title:
        lines.append(f"Page: {title}")
    lines.append(f"Action: {_action_text(state)}")
    return lines


def compact_state_text(state: dict) -> str:
    """Rules 1-5: one self-contained, retrievable compact state block."""
    tree = str(state.get("axtree") or state.get("synthetic_axtree") or "")
    parsed = list(nodes(tree))
    lines = _state_header(state, parsed)
    lines.extend(compact_state_lines(state))
    return "\n".join(lines)


def compact_slice_blocks(states: list[dict]) -> list[tuple[dict, list[str]]]:
    """Rules 1-5 for a slice: later states carry only their new lines."""
    seen: set[str] = set()
    blocks = []
    for state in states:
        tree = str(state.get("axtree") or state.get("synthetic_axtree") or "")
        parsed = list(nodes(tree))
        body = _dedup(compact_state_lines(state), seen)
        blocks.append((state, _state_header(state, parsed) + body))
    return blocks


def compact_slice_text(states: list[dict]) -> str:
    """Apply rules 1-5 to a center-state slice used as a retrieval document."""
    return "\n\n".join("\n".join(lines) for _state, lines in compact_slice_blocks(states))


def compact_context_states(states: list[dict]) -> list[dict]:
    """Shallow-copy slice states with the AXTree replaced by the compact view.

    Lines already emitted by an earlier state in the same slice are dropped and
    replaced by an explicit marker, so the reader context keeps one copy of the
    repeated chrome while rule 3 still labels every state.
    """
    output = []
    for state, lines in compact_slice_blocks(states):
        body = [line for line in lines if not line.startswith(("State ", "URL:", "Page:", "Action:"))]
        text = (
            "(no new lines relative to the previous state in this slice)"
            if not body
            else "\n".join(body)
        )
        output.append({**state, "axtree": text, "synthetic_axtree": ""})
    return output
