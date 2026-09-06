"""Deterministic redundancy filtering for AMA-Bench AXTree observation text.

The observations stored in each AMA trajectory turn are serialized browser
accessibility trees (``[id] role 'name' attr: value`` lines with indentation as
nesting). Three measured sources of redundancy motivate this module:

* PUA icon glyphs (U+E000-U+F8FF, icon fonts) and default ``attr: False`` /
  ``orientation: horizontal`` clauses that carry no state.
* Empty-shell leaves -- ``generic`` / ``group`` / empty ``StaticText`` /
  decorative ``image`` nodes with no name, no value and no descendants.
* Names that repeat their nearest non-interactive container ancestor verbatim
  (e.g. a wrapper whose only text is its first descendant's label).

Hard invariant: element ``[id]`` tokens and node names are the answer surface of
this benchmark (open-end answers cite exact ids and verbatim text), so this
cleaner never renumbers, merges or drops a node that could be an answer target,
and never touches names on interactive roles.

Rules are keyed by name (R1..R4) so an arm can be audited and reproduced from a
single protocol string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

AXTREE_CLEAN_PROTOCOL = "ama_axtree_clean_medium_v1"

# Private-use-area glyphs used by icon fonts. AMA AXTree text stores them two
# ways: as the real codepoint (U+E000-U+F8FF) and, far more often, as the
# literal escape text ``\ue604`` produced by the serialization layer. Both are
# the same icon-font artifact and neither carries retrievable state.
_PUA_RE = re.compile(r"[\ue000-\uf8ff]")
_LITERAL_ICON_RE = re.compile(r"\\u[ef][0-9a-fA-F]{3}")
# Collapse runs of horizontal whitespace to one space.
_WS_RE = re.compile(r"[ \t\u00a0]+")
# Default-valued attribute clauses removed by R2. True-valued counterparts are
# always preserved because "expanded: True" etc. are referenced by the QA text.
_DEFAULT_CLAUSE_RE = re.compile(
    r"\b(?:required|orientation|multiselectable|expanded|selected|focused|"
    r"disabled|checked|haspopup)\s*:\s*(?:False|horizontal)\b"
)
# Roles that carry no content and are never answer targets when empty.
_EMPTY_SHELL_ROLES = frozenset({
    "generic", "group", "none", "presentation", "separator",
    "StaticText", "text", "image", "img", "note",
})
# Non-interactive container roles whose verbatim-name copy may be dropped by R4.
_CONTAINER_ROLES = frozenset({
    "generic", "group", "none", "presentation", "tablist",
    "list", "listitem", "row", "grid",
})

# ---- verbatim anchor extraction (latent+anchor memory arm) -----------------
# Roles a QA can cite or click.  Everything else (containers, icon shells,
# decorative images) is excluded so the anchor text stays small enough to
# inject for the retrieved top-k screens without swallowing the prompt budget.
AMA_ANCHOR_EXTRACT_V1 = "ama_anchor_extract_v1"
ANCHOR_INTERACTIVE_ROLES = frozenset({
    "link", "button", "textbox", "searchbox", "textarea", "menuitem",
    "menuitemcheckbox", "menuitemradio", "tab", "checkbox", "radio",
    "combobox", "listbox", "option", "treeitem", "slider", "spinbutton",
    "switch",
})
# Content roles kept only when they carry text (verbatim answer surface).
ANCHOR_TEXT_ROLES = frozenset({"heading", "StaticText"})
# Structural/landmark roles that QA occasionally cites by id.  Kept only when
# they have a non-empty name (their text is otherwise duplicated by children),
# which bounds the token cost: grid rows/cells are deliberately excluded
# because they would explode the anchor on dense tables.
ANCHOR_STRUCT_NAMED_ROLES = frozenset({
    "RootWebArea", "dialog", "complementary", "contentinfo", "banner",
    "navigation", "main", "article", "form", "alert", "status", "tooltip",
    "menu", "menubar", "toolbar", "tablist", "tree", "rowheader",
    "columnheader",
})
ANCHOR_NAMED_ROLES = ANCHOR_TEXT_ROLES | ANCHOR_STRUCT_NAMED_ROLES
ANCHOR_ROLES = ANCHOR_INTERACTIVE_ROLES | ANCHOR_NAMED_ROLES
ANCHOR_STYLE_ELEMENT = "element"
ANCHOR_STYLE_IDS = "ids"

_NODE_LINE = re.compile(r"^(\s*)\[(\d+)\]\s+\S+\s+['\"]")
_NODE_PREFIX = re.compile(r"^\[(\d+)\]\s+")
_TRAILING_WS = re.compile(r"[ \t]+$")


@dataclass
class _Node:
    index: int          # position in the parsed node list (not the AX id)
    line_index: int     # position in the document line list
    depth: int          # nesting depth, measured on expanded leading whitespace
    ax_id: str          # literal [id] token, preserved verbatim
    role: str
    name: str           # cleaned name (may be cleared by R4)
    attrs: str
    leading: str        # original indentation whitespace
    has_children: bool = False


def _clean_ws(value: str) -> str:
    return _WS_RE.sub(" ", value).strip()


def _split_node(line: str) -> tuple[str, str, str, str] | None:
    """Return (id, role, name, attrs) for an AXTree node line, else None.

    Names may be quoted with single or double quotes and may contain escaped
    quotes inside (``"the \\'cloud\\' example"``), so the closing quote is
    found by scanning while honouring backslash escapes.
    """
    stripped = line.lstrip()
    head = _NODE_PREFIX.match(stripped)
    if head is None:
        return None
    ax_id = head.group(1)
    rest = stripped[head.end():]
    role = re.match(r"\S+", rest)
    if role is None:
        return None
    role = role.group(0)
    rest = rest[len(role):].lstrip()
    if not rest or rest[0] not in ("'", '"'):
        return None
    quote = rest[0]
    name_chars: list[str] = []
    cursor = 1
    closed = False
    while cursor < len(rest):
        char = rest[cursor]
        if char == "\\" and cursor + 1 < len(rest):
            name_chars.append(rest[cursor + 1])   # unescape \' \" \\
            cursor += 2
            continue
        if char == quote:
            closed = True
            break
        name_chars.append(char)
        cursor += 1
    if not closed:
        return None                               # multi-line name; leave line alone
    attrs = rest[cursor + 1:].strip()
    return ax_id, role, "".join(name_chars), attrs


def clean_observation(text: str, mode: str = "medium") -> tuple[str, dict]:
    """Filter one AXTree observation string. Returns ``(text, stats)``.

    ``mode="none"`` is the identity; ``mode="medium"`` applies R1..R4. Any
    other mode raises so a typo cannot silently change an arm's input.
    """
    if mode == "none":
        return text, {}
    if mode != "medium":
        raise ValueError(f"unknown axtree clean mode: {mode!r}")

    stats = {
        "protocol": AXTREE_CLEAN_PROTOCOL,
        "mode": mode,
        "chars_before": len(text),
        "icon_glyphs_removed": 0,
        "default_attrs_removed": 0,
        "nodes_before": 0,
        "nodes_after": 0,
        "dropped_shells": 0,
        "collapsed_container_names": 0,
        "saved_chars": 0,
        "chars_after": 0,
    }
    # R1: drop blank lines, strip PUA glyphs (real and literal escapes) and
    # trailing whitespace.
    lines: list[str] = []
    for raw in text.split("\n"):
        glyphs = _PUA_RE.findall(raw) + _LITERAL_ICON_RE.findall(raw)
        if glyphs:
            stats["icon_glyphs_removed"] += len(glyphs)
            raw = _PUA_RE.sub("", raw)
            raw = _LITERAL_ICON_RE.sub("", raw)
        raw = _TRAILING_WS.sub("", raw)
        if raw.strip():
            lines.append(raw)

    # Parse node lines into typed records.
    nodes: list[_Node] = []
    node_by_line: dict[int, _Node] = {}
    for line_index, line in enumerate(lines):
        if _NODE_LINE.match(line) is None:
            continue
        stats["nodes_before"] += 1
        leading, ax_id = _NODE_LINE.match(line).groups()[:2]
        split = _split_node(line)
        if split is None:
            continue                     # unparseable (multi-line name): keep as-is
        _ax_id, role, raw_name, attrs = split
        node = _Node(
            index=len(nodes), line_index=line_index,
            depth=len(leading.expandtabs(4)),
            ax_id=ax_id, role=role,
            name=_clean_ws(raw_name), attrs=attrs,
            leading=leading,
        )
        node_by_line[line_index] = node
        nodes.append(node)

    # has_children: next node line at depth <= this node's depth.
    next_no_deeper = [len(nodes)] * len(nodes)
    stack: list[tuple[int, int]] = []    # (depth, index), monotonically deeper
    for index in range(len(nodes) - 1, -1, -1):
        depth = nodes[index].depth
        while stack and stack[-1][0] > depth:
            stack.pop()
        if stack:
            next_no_deeper[index] = stack[-1][1]
        stack.append((depth, index))
    for node in nodes:
        node.has_children = (
            node.index + 1 < len(nodes)
            and nodes[node.index + 1].depth > node.depth
        )

    # R3: drop empty-shell leaves.
    drop = [False] * len(nodes)
    for node in nodes:
        if (node.role in _EMPTY_SHELL_ROLES and not node.has_children
                and not node.name and not node.attrs):
            drop[node.index] = True

    # R4: a container whose first descendant repeats its name verbatim loses
    # the copy on the container; the descendant keeps id, role and text. This
    # only touches non-interactive container roles (tab/link/etc. are exempt).
    for position, node in enumerate(nodes):
        if drop[node.index] or node.role not in _CONTAINER_ROLES or not node.name:
            continue
        for later in nodes[position + 1: next_no_deeper[position]]:
            if later.depth <= node.depth:
                continue
            if later.name == node.name:
                node.name = ""
                stats["collapsed_container_names"] += 1
            break

    # Rebuild the document.
    out: list[str] = []
    for line_index, line in enumerate(lines):
        node = node_by_line.get(line_index)
        if node is None:
            out.append(line)
            continue
        if drop[node.index]:
            stats["dropped_shells"] += 1
            continue
        attrs = _DEFAULT_CLAUSE_RE.sub("", node.attrs)
        removed = _DEFAULT_CLAUSE_RE.findall(node.attrs)
        if removed:
            stats["default_attrs_removed"] += len(removed)
        attrs = _clean_ws(attrs)
        parts = [f"[{node.ax_id}] {node.role}"]
        if node.name:
            parts.append(f"'{node.name}'")
        else:
            parts.append("''")
        if attrs:
            parts.append(attrs)
        out.append(f"{node.leading}{' '.join(parts)}")

    result = "\n".join(out)
    stats["nodes_after"] = len(_NODE_LINE.findall(result)) if result else 0
    stats["chars_after"] = len(result)
    stats["saved_chars"] = stats["chars_before"] - stats["chars_after"]
    return result, stats


def clean_trajectory(episode: dict, mode: str) -> tuple[list[dict], dict]:
    """Return a trajectory copy with each step's observation cleaned.

    Only the ``observation`` field is rewritten; ``turn_idx``/``action`` and the
    episode-level ``task``/``qa_pairs`` stay untouched, so an adapted step text
    differs solely by the filtered tree. Returns ``(steps, aggregate_audit)``.
    """
    aggregate = {
        "protocol": AXTREE_CLEAN_PROTOCOL,
        "mode": mode,
        "steps": len(episode["trajectory"]),
        "chars_before": 0,
        "saved_chars": 0,
        "dropped_shells": 0,
        "collapsed_container_names": 0,
    }
    steps: list[dict] = []
    for step in episode["trajectory"]:
        if "observation" not in step:
            steps.append(step)
            continue
        cleaned, per_step = clean_observation(str(step["observation"]), mode)
        aggregate["chars_before"] += per_step.get("chars_before", 0)
        aggregate["saved_chars"] += per_step.get("saved_chars", 0)
        aggregate["dropped_shells"] += per_step.get("dropped_shells", 0)
        aggregate["collapsed_container_names"] += per_step.get(
            "collapsed_container_names", 0)
        steps.append({**step, "observation": cleaned})
    return steps, aggregate


def element_lines(text: str) -> list[str]:
    """Verbatim, injectable element rows for one AXTree observation.

    Returns ``[id] role 'name'`` (+ surviving attributes) lines, document
    order.  Interactive elements always keep their id (even when the name was
    an icon and is now empty); content/structural roles (heading, StaticText,
    RootWebArea, dialog, columnheader, ...) are kept only when they carry text.
    Ids are never rewritten or dropped; a node row whose long/multi-line name
    defeats the full parser is still kept as ``[id] role <raw tail>`` so its id
    survives.
    """
    cleaned, _ = clean_observation(text, "medium")
    rows: list[str] = []
    for line in cleaned.split("\n"):
        stripped = line.rstrip()
        split = _split_node(stripped)
        if split is None:
            # Parser fallback: preserve the id even when a long/multi-line name
            # defeats the full parser.  Only genuine ``[id] role ...`` rows
            # match; the whole (already medium-cleaned) line is kept as-is.
            match = _NODE_LINE.match(stripped)
            if match is None:
                continue
            ax_id = match.group(2)
            tail = stripped[match.end():]
            role = tail.split(None, 1)[0]
            name_match = re.search(r"'([^']*)'", tail)
            has_name = bool(name_match and name_match.group(1).strip())
            if role not in ANCHOR_ROLES:
                continue
            if role not in ANCHOR_INTERACTIVE_ROLES and not has_name:
                continue
            rows.append(stripped.strip())
            continue
        ax_id, role, name, attrs = split
        if role not in ANCHOR_ROLES:
            continue
        if role not in ANCHOR_INTERACTIVE_ROLES and not name.strip():
            continue
        core = f"[{ax_id}] {role}"
        if name.strip():
            core += f" '{name}'"
        elif role in ANCHOR_INTERACTIVE_ROLES:
            core += " ''"
        if attrs:
            core += " " + attrs
        rows.append(core)
    return rows


def ids_from(text: str) -> list[str]:
    """``[id]`` tokens of the anchor-eligible rows of one observation."""
    return [row.split(" ", 1)[0] for row in element_lines(text)]
