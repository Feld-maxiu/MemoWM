"""Sparse exact-value anchors with local UI context, independent of any query.

Difficulty is approximated by literal-value structure and external label
frequency. This is not an empirical claim that every selected span is
incompressible. Common controls remain the latent channel's responsibility.
"""
from __future__ import annotations

from collections import Counter
import html
import json
import re

from .longmemeval import _split_ax_line

PROTOCOL = "longmemeval-sparse-contextual-anchor-v3"
COMMON_CONTROLS = frozenset((
    "press", "click", "save", "submit", "cancel", "back", "next", "previous", "close", "ok", "yes", "no",
    "edit", "delete", "remove", "add", "search", "reset", "apply", "clear", "refresh", "reload", "more",
    "home", "menu", "help", "login", "log in", "logout", "log out", "sign in", "sign out", "sign up",
    "continue", "view", "select", "filter", "filters", "sort", "print", "download", "upload", "share",
    "copy", "paste", "cut", "undo", "redo", "settings", "account", "overview", "summary", "details",
    "name", "title", "description", "email", "password", "quantity", "price", "total", "status", "date",
    "required", "optional", "image", "default", "loading", "loading...", "learn more", "read more",
    "add to cart", "add to wish list", "add to wishlist", "add to compare", "view as list", "view as grid",
    "set ascending direction", "set descending direction", "show report", "all websites", "what is this?",
    "product name", "search terms", "advanced search", "privacy and cookie policy", "report all bugs",
    "account activity", "report an issue", "sign up for our newsletter", "find partners & extensions",
))
CONTROLS = frozenset({"button", "link", "tab", "menuitem", "option", "checkbox", "radio", "combobox", "textbox", "searchbox", "spinbutton"})
REGIONS = frozenset({"dialog", "complementary", "navigation", "main", "form", "toolbar", "tablist", "menu", "region", "section", "table", "grid", "group"})
VALUE_RE = re.compile(r"(?:[$€£¥₹]\s*\d|\b\d+(?:[.,:/-]\d+)+\b|\b\d{3,}\b|https?://\S+|\b[^\s@]+@[^\s@]+\.[^\s@]+)")
MESSAGE_RE = re.compile(r"\b(?:error|failed|failure|cannot|couldn't|unable|invalid|warning|successfully|scheduled|confirmation|was updated|added to|not allowed|not available)\b", re.I)
ATTR_RE = re.compile(r"([\w:.-]+)\s*=\s*(?:'((?:\\.|[^'])*)'|\"((?:\\.|[^\"])*)\"|([^\s,]+))")


def clean(value):
    value = re.sub(r"(?:\\u|\bu)[ef][0-9a-fA-F]{3}\b|[\ue000-\uf8ff]", "", str(value))
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def nodes(tree):
    for position, raw in enumerate(tree.splitlines()):
        parsed = _split_ax_line(raw)
        if parsed is None:
            continue
        node_id, role, name, tail = parsed
        attrs = {match.group(1): clean(next(v for v in match.groups()[1:] if v is not None))
                 for match in ATTR_RE.finditer(tail)}
        yield {"position": position, "depth": len(raw) - len(raw.lstrip()),
               "id": node_id, "role": role, "name": clean(name), "attrs": attrs}


def label_profile(trees):
    counts = Counter()
    observations = 0
    for tree in trees:
        counts.update({node["name"].casefold() for node in nodes(tree)
                       if node["role"] in CONTROLS and node["name"]})
        observations += 1
    return {"protocol": PROTOCOL, "external_observations": observations,
            "label_document_frequency": dict(counts), "common_min_observations": max(8, observations // 100)}


def page_title(tree):
    return next((node["name"] for node in nodes(tree) if node["role"] == "RootWebArea"), "")


def annotated_incoming_action(action, previous_tree):
    """Official LME action on state t describes the transition t-1 -> t."""
    if action is None:
        return "<initial state or no recorded incoming action>"
    value = action if isinstance(action, str) else json.dumps(action, ensure_ascii=False, sort_keys=True)
    lookup = {node["id"]: node for node in nodes(previous_tree) if node["id"]}
    match = re.match(r"\s*(\w+)\(\s*['\"]([^'\"]+)['\"]", value)
    if match and match.group(2) in lookup:
        target = lookup[match.group(2)]
                                                                            
                                                                           
        return f"{value} [target in previous state: {target['role']} {json.dumps(target['name'], ensure_ascii=False)}]"
    return value


def extract_sparse_anchor(tree, profile, tokenizer, *, max_tokens=1536):
    if max_tokens < 1:
        raise ValueError("anchor token budget must be positive")
    frequency = profile["label_document_frequency"]
    common_threshold = profile["common_min_observations"]
    stack = []
    heading = ""
    candidates = []
    seen = set()
    row_cells = {}
    table_headers = {}
    list_identities = {}
    for node in nodes(tree):
        depth, role, name, attrs = node["depth"], node["role"], node["name"], node["attrs"]
        while stack and stack[-1]["depth"] >= depth:
            stack.pop()
        parents = list(stack)
        stack.append(node)
        if role == "heading" and name and len(name) <= 160:
            heading = name
        region = next((p["name"] for p in reversed(parents) if p["role"] in REGIONS and p["name"] and len(p["name"]) <= 160), "")
        context = region or heading
        table = next((p["position"] for p in reversed(parents) if p["role"] in {"table", "grid"}), None)
        row = next((p["position"] for p in reversed(parents) if p["role"] == "row"), None)
        list_item = next((p["position"] for p in reversed(parents) if p["role"] == "listitem"), None)
        if list_item is not None:
            if role in {"link", "heading"} and len(name) >= 12 and name.casefold() not in COMMON_CONTROLS:
                list_identities.setdefault(list_item, name)
            identity = list_identities.get(list_item)
            if identity and len(identity) <= 220:
                context = " / ".join(x for x in [context, f"item {identity}"] if x)
        if role == "columnheader" and name:
            table_headers.setdefault(table, []).append(name)
        label = name
        value = attrs.get("value", "")
        is_table_cell = role in {"cell", "gridcell"}
        if is_table_cell and row is not None:
            cells = row_cells.setdefault(row, [])
            column = len(cells)
            cells.append(name)
            headers = table_headers.get(table, [])
            if column < len(headers):
                label, value = headers[column], name
            if cells[0] and len(cells[0]) <= 100:
                context = " / ".join(x for x in [context, f"row {cells[0]}"] if x)
        if not name and not value:
            continue
        if role in {"RootWebArea", "generic", "group", "row", "table", "grid", "image", "img", "separator", "presentation"}:
            continue
        if any(p["name"] == name and p["role"] in CONTROLS for p in parents):
            continue                                                            
        lower = name.casefold().rstrip(".: ")
        common = lower in COMMON_CONTROLS or frequency.get(name.casefold(), 0) >= common_threshold
        reason = None
        priority = 0
        if value and role in {"textbox", "searchbox", "spinbutton", "combobox", "cell", "gridcell"}:
            if VALUE_RE.search(value) or (len(value) >= 3 and value.casefold() not in COMMON_CONTROLS and not value.isdigit()):
                reason, priority = "literal_field_value", 90 if not is_table_cell else 75
        if MESSAGE_RE.search(name) and len(name) >= 18 and role in {"StaticText", "text", "paragraph", "alert", "status", "heading"}:
            reason, priority = "verbatim_message", 95
        elif VALUE_RE.search(name) and not reason:
            reason, priority = "literal_content", 80
        elif role in {"button", "link", "checkbox", "radio"} and not common and 20 <= len(name) <= 180 and len(name.split()) >= 3:
                                                                            
                                                                               
            reason, priority = reason or "uncommon_control_label", max(priority, 15)
        if reason is None:
            continue
        if len(name) > 1600 or len(value) > 1600:
            continue                                                    
        if not context:
            context = "page"
        payload = {"region": context, "role": role, "label": label}
        if value:
            payload["value"] = value
        for key in ("checked", "selected", "disabled", "expanded"):
            if key in attrs:
                payload[key] = attrs[key]
                                                                              
        if role in CONTROLS:
            payload["tree_position"] = node["position"]
        signature = json.dumps({k: v for k, v in payload.items() if k != "tree_position"}, sort_keys=True, ensure_ascii=False)
        if signature in seen:
            continue
        seen.add(signature)
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        candidates.append((priority, node["position"], context, reason, text))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    counts = Counter()
    rare_counts = Counter()
    rare_total = 0
    used = 0
    for item in candidates:
                                                                              
                                                                               
        if counts[item[2]] >= 8:
            continue
        if item[3] == "uncommon_control_label" and (rare_counts[item[2]] >= 2 or rare_total >= 8):
            continue
        cost = len(tokenizer.encode(item[4] + "\n", add_special_tokens=False))
        if used + cost > max_tokens:
            continue
        selected.append(item)
        counts[item[2]] += 1
        if item[3] == "uncommon_control_label":
            rare_counts[item[2]] += 1
            rare_total += 1
        used += cost
    selected.sort(key=lambda item: item[1])
    text = "\n".join(item[4] for item in selected)
    return {"protocol": PROTOCOL, "text": text, "tokens": len(tokenizer.encode(text, add_special_tokens=False)),
            "token_budget": max_tokens, "fields": len(selected), "candidate_fields": len(candidates),
            "dropped_fields": len(candidates) - len(selected), "bytes": len(text.encode()),
            "selection": dict(Counter(item[3] for item in selected)),
            "difficulty_proxy": "external_label_frequency_and_literal_structure",
            "omitted_content_is_not_evidence_of_absence": True}
