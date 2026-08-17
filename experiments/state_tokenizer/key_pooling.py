"""Deterministic key-preserving pooling for the fixed-prompt MiniWoB pilot."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


# Re-exported from slot_layout so torch-free consumers (the jax bottlenecks) can
# read the same constants without importing this module.
from .slot_layout import (  # noqa: F401
    CONTEXT_SLOTS,
    DETAIL_SLOTS,
    IMAGE_SLOTS,
    KEY64_LAYOUT,
    PROMPT_SLOTS,
)
KEY64_PROTOCOL = "fixed_prompt_key64_v1"

TEXTBOX_TAGS = {"input_text", "input_password", "textarea"}
VALUE_TAGS = TEXTBOX_TAGS | {"select", "option"}
STATE_VALUE_TAGS = {"input_checkbox", "input_radio"}
INTERACTIVE_TEXT_TAGS = VALUE_TAGS | STATE_VALUE_TAGS | {
    "button", "input_button", "input_submit", "label", "a",
}
ATTRIBUTE_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_-]*)="
    r"(?P<value>\"(?:\\.|[^\"\\])*\"|[^\s/>]+)"
)


@dataclass(frozen=True)
class AttributeSpan:
    key: str
    value: str
    source: str
    start: int
    stop: int


@dataclass(frozen=True)
class NodeSpan:
    index: int
    start: int
    stop: int
    tag: str
    ref: str
    parent: str
    attributes: tuple[AttributeSpan, ...]


@dataclass(frozen=True)
class DetailCandidate:
    priority: int
    start: int
    stop: int
    node_index: int
    field: str


@dataclass
class Key64Output:
    tokens: torch.Tensor
    modality_ids: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    detail_source: torch.Tensor
    audit: dict


def parse_dom_spans(dom: str) -> list[NodeSpan]:
    """Parse compact DOM rows while preserving exact attribute value spans."""
    nodes = []
    cursor = 0
    for raw in dom.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        line_start = cursor
        line_stop = line_start + len(line)
        cursor += len(raw)
        if not line.strip():
            continue
        stripped = line.strip()
        leading = len(line) - len(line.lstrip())
        if not (stripped.startswith("<") and stripped.endswith("/>")):
            raise ValueError(f"invalid compact DOM row: {stripped[:80]!r}")
        attributes = []
        values = {}
        for match in ATTRIBUTE_RE.finditer(line):
            key = match.group("key")
            raw_value = match.group("value")
            value_start, value_stop = match.span("value")
            if raw_value.startswith('"'):
                try:
                    value = json.loads(raw_value)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON DOM attribute {raw_value!r}") from exc
                value_start += 1
                value_stop -= 1
            else:
                value = raw_value
            span = AttributeSpan(
                key=key,
                value=str(value),
                source=line[value_start:value_stop],
                start=line_start + value_start,
                stop=line_start + value_stop,
            )
            attributes.append(span)
            values[key] = str(value)
        nodes.append(NodeSpan(
            index=len(nodes),
            start=line_start + leading,
            stop=line_stop,
            tag=values.get("tag", "").lower(),
            ref=values.get("ref", ""),
            parent=values.get("parent", ""),
            attributes=tuple(attributes),
        ))
    if not nodes:
        raise ValueError("compact DOM is empty")
    return nodes


def detail_candidates(nodes: Sequence[NodeSpan]) -> list[DetailCandidate]:
    """Return schema-driven candidates without consulting probe labels or values."""
    by_ref = {node.ref: node for node in nodes if node.ref}
    output = []
    for node in nodes:
        parent_tag = by_ref.get(node.parent).tag if node.parent in by_ref else ""
        for attribute in node.attributes:
            if attribute.start >= attribute.stop:
                continue
            if attribute.key == "value" and node.tag in VALUE_TAGS:
                for match in re.finditer(r"\d+", attribute.source):
                    output.append(DetailCandidate(
                        priority=-1,
                        start=attribute.start + match.start(),
                        stop=attribute.start + match.end(),
                        node_index=node.index,
                        field="value_numeric",
                    ))
            priority = None
            if attribute.key == "value" and node.tag in VALUE_TAGS:
                priority = 0
            elif attribute.key == "text" and (
                node.tag in INTERACTIVE_TEXT_TAGS
                or (node.tag == "t" and parent_tag in INTERACTIVE_TEXT_TAGS)
            ):
                priority = 1
            elif attribute.key == "value" and node.tag in STATE_VALUE_TAGS:
                priority = 2
            elif attribute.key == "flags" and any(char != "0" for char in attribute.value if char.isdigit()):
                priority = 2
            elif attribute.key in {"id", "classes"}:
                priority = 3
            if priority is not None:
                output.append(DetailCandidate(
                    priority=priority,
                    start=attribute.start,
                    stop=attribute.stop,
                    node_index=node.index,
                    field=attribute.key,
                ))
    return sorted(output, key=lambda item: (item.priority, item.start, item.stop, item.field))


def _overlap(first: tuple[int, int], second: tuple[int, int]) -> int:
    return max(0, min(first[1], second[1]) - max(first[0], second[0]))


def token_indices_for_span(
    token_offsets: Sequence[tuple[int, int]], start: int, stop: int,
) -> list[int]:
    return [
        index for index, offset in enumerate(token_offsets)
        if offset[1] > offset[0] and _overlap(offset, (start, stop)) > 0
    ]


def select_detail_indices(
    nodes: Sequence[NodeSpan],
    token_offsets: Sequence[tuple[int, int]],
    *,
    slots: int = DETAIL_SLOTS,
) -> tuple[list[int], dict]:
    """Pack complete high-priority spans, then restore original token order."""
    selected: set[int] = set()
    selected_spans = []
    overflow = False
    candidate_rows = []
    for candidate in detail_candidates(nodes):
        indices = token_indices_for_span(token_offsets, candidate.start, candidate.stop)
        fresh = [index for index in indices if index not in selected]
        if not fresh:
            continue
        complete = len(selected) + len(fresh) <= slots
        chosen = fresh if complete else []
        if not complete and candidate.priority <= 0:
            overflow = True
            if not selected and len(fresh) > slots:
                front = (slots + 1) // 2
                back = slots - front
                chosen = fresh[:front] + (fresh[-back:] if back else [])
        if chosen:
            selected.update(chosen)
            selected_spans.append({
                "priority": candidate.priority,
                "node_index": candidate.node_index,
                "field": candidate.field,
                "tokens": len(indices),
                "complete": complete,
            })
        candidate_rows.append({
            "priority": candidate.priority,
            "node_index": candidate.node_index,
            "field": candidate.field,
            "tokens": len(indices),
            "selected_tokens": len(chosen),
            "complete": complete and bool(chosen),
        })
    ordered = sorted(selected)
    return ordered, {
        "candidate_spans": len(candidate_rows),
        "selected_spans": len(selected_spans),
        "selected_tokens": len(ordered),
        "detail_overflow": overflow,
        "candidates": candidate_rows,
        "selected": selected_spans,
    }


def _adaptive_pool_1d(values: torch.Tensor, slots: int) -> torch.Tensor:
    if values.ndim != 2 or not len(values):
        raise ValueError(f"expected non-empty (tokens,width), got {tuple(values.shape)}")
    chunks = []
    for index in range(slots):
        start = math.floor(index * len(values) / slots)
        stop = math.ceil((index + 1) * len(values) / slots)
        stop = max(start + 1, min(stop, len(values)))
        chunks.append(values[start:stop].float().mean(dim=0))
    return torch.stack(chunks)


def pool_image_2d(
    image_hidden: torch.Tensor,
    image_grid_thw: Sequence[int],
    spatial_merge_size: int,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    if len(image_grid_thw) != 3:
        raise ValueError(f"expected image_grid_thw=(T,H,W), got {image_grid_thw}")
    temporal, grid_h, grid_w = (int(value) for value in image_grid_thw)
    if temporal != 1:
        raise ValueError(f"MiniWoB key64 expects one image frame, got T={temporal}")
    if grid_h % spatial_merge_size or grid_w % spatial_merge_size:
        raise ValueError(
            f"grid {(grid_h, grid_w)} is not divisible by merge={spatial_merge_size}"
        )
    merged_h = grid_h // spatial_merge_size
    merged_w = grid_w // spatial_merge_size
    if len(image_hidden) != merged_h * merged_w:
        raise ValueError(
            f"image token/grid mismatch: {len(image_hidden)} != {merged_h}*{merged_w}"
        )
    output_hw = (8, 4) if merged_h >= merged_w else (4, 8)
    grid = image_hidden.reshape(merged_h, merged_w, image_hidden.shape[-1])
    channels_first = grid.permute(2, 0, 1).unsqueeze(0).float()
    pooled = F.adaptive_avg_pool2d(channels_first, output_hw)
    values = pooled[0].permute(1, 2, 0).reshape(IMAGE_SLOTS, image_hidden.shape[-1])
    return values, (merged_h, merged_w), output_hw


def _node_token_rows(
    nodes: Sequence[NodeSpan], token_offsets: Sequence[tuple[int, int]],
) -> list[list[int]]:
    rows = [[] for _ in nodes]
    for token_index, offset in enumerate(token_offsets):
        overlaps = [_overlap(offset, (node.start, node.stop)) for node in nodes]
        maximum = max(overlaps, default=0)
        if maximum:
            rows[overlaps.index(maximum)].append(token_index)
    return rows


def pool_full_nodes(
    dom_hidden: torch.Tensor,
    nodes: Sequence[NodeSpan],
    token_offsets: Sequence[tuple[int, int]],
    *,
    dom: str,
    slots: int = CONTEXT_SLOTS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Pool complete nodes, intentionally retaining tokens also used by detail."""
    node_rows = _node_token_rows(nodes, token_offsets)
    vectors = []
    positions = []
    for rows in node_rows:
        if not rows:
            continue
        index = torch.tensor(rows, dtype=torch.long, device=dom_hidden.device)
        vectors.append(dom_hidden.index_select(0, index).float().mean(dim=0))
        positions.append(float(np.mean([(row + 0.5) / len(dom_hidden) for row in rows])))
    if not vectors:
        raise ValueError("no DOM nodes overlap Qwen DOM token offsets")
    values = torch.stack(vectors)
    source_positions = torch.tensor(positions, dtype=torch.float32, device=dom_hidden.device)
    width = dom_hidden.shape[-1]
    output = torch.zeros((slots, width), dtype=torch.float32, device=dom_hidden.device)
    output_positions = torch.zeros((slots,), dtype=torch.float32, device=dom_hidden.device)
    valid = torch.zeros((slots,), dtype=torch.bool, device=dom_hidden.device)
    if len(values) <= slots:
        output[:len(values)] = values
        output_positions[:len(values)] = source_positions
        valid[:len(values)] = True
    else:
        for index in range(slots):
            start = index * len(values) // slots
            stop = (index + 1) * len(values) // slots
            output[index] = values[start:stop].mean(dim=0)
            output_positions[index] = source_positions[start:stop].mean()
            valid[index] = True
    covered = {index for rows in node_rows for index in rows}
    separators = []
    for index, (start, stop) in enumerate(token_offsets):
        if index in covered:
            continue
        fragment = dom[max(0, start):min(len(dom), stop)]
        if fragment.strip():
            raise ValueError(
                f"non-whitespace DOM token {index} was not assigned to a node: {fragment!r}"
            )
        separators.append(index)
    return output, output_positions, valid, {
        "dom_nodes": len(nodes),
        "context_nodes": len(values),
        "context_valid_slots": int(valid.sum()),
        "separator_dom_tokens": len(separators),
    }


def build_key64(
    image_hidden: torch.Tensor,
    dom_hidden: torch.Tensor,
    prompt_hidden: torch.Tensor,
    *,
    dom: str,
    dom_token_offsets: Sequence[tuple[int, int]],
    image_grid_thw: Sequence[int],
    spatial_merge_size: int,
) -> Key64Output:
    if len(dom_token_offsets) != len(dom_hidden):
        raise ValueError(
            f"DOM offset/hidden mismatch: {len(dom_token_offsets)} != {len(dom_hidden)}"
        )
    nodes = parse_dom_spans(dom)
    detail_indices, detail_audit = select_detail_indices(nodes, dom_token_offsets)
    width = dom_hidden.shape[-1]

    image, merged_hw, output_hw = pool_image_2d(
        image_hidden, image_grid_thw, spatial_merge_size
    )
    detail = torch.zeros((DETAIL_SLOTS, width), dtype=torch.float32, device=dom_hidden.device)
    detail_positions = torch.zeros((DETAIL_SLOTS,), dtype=torch.float32, device=dom_hidden.device)
    detail_valid = torch.zeros((DETAIL_SLOTS,), dtype=torch.bool, device=dom_hidden.device)
    detail_source = torch.full(
        (DETAIL_SLOTS,), -1, dtype=torch.int32, device=dom_hidden.device
    )
    if detail_indices:
        selected = torch.tensor(detail_indices, dtype=torch.long, device=dom_hidden.device)
        count = len(detail_indices)
        detail[:count] = dom_hidden.index_select(0, selected).float()
        detail_positions[:count] = (selected.float() + 0.5) / len(dom_hidden)
        detail_valid[:count] = True
        detail_source[:count] = selected.to(torch.int32)

    context, context_positions, context_valid, context_audit = pool_full_nodes(
        dom_hidden, nodes, dom_token_offsets, dom=dom
    )
    # See build_static_key64: with PROMPT_SLOTS == 0 the band is omitted, because
    # _adaptive_pool_1d stacks an empty list and the position term divides by zero.
    empty = torch.zeros((0, dom_hidden.shape[-1]), dtype=torch.float32,
                        device=dom_hidden.device)
    prompt = _adaptive_pool_1d(prompt_hidden, PROMPT_SLOTS) if PROMPT_SLOTS else empty
    prompt_positions = (
        (torch.arange(PROMPT_SLOTS, device=dom_hidden.device, dtype=torch.float32) + 0.5)
        / PROMPT_SLOTS
        if PROMPT_SLOTS
        else torch.zeros((0,), dtype=torch.float32, device=dom_hidden.device)
    )
    tokens = torch.cat((image, detail, context, prompt), dim=0)
    valid = torch.cat((
        torch.ones((IMAGE_SLOTS,), dtype=torch.bool, device=tokens.device),
        detail_valid,
        context_valid,
        torch.ones((PROMPT_SLOTS,), dtype=torch.bool, device=tokens.device),
    ))
    positions = torch.cat((
        (torch.arange(IMAGE_SLOTS, device=tokens.device, dtype=torch.float32) + 0.5)
        / IMAGE_SLOTS,
        detail_positions,
        context_positions,
        prompt_positions,
    ))
    modality_ids = torch.cat((
        torch.zeros((IMAGE_SLOTS,), dtype=torch.long, device=tokens.device),
        torch.ones((DETAIL_SLOTS + CONTEXT_SLOTS,), dtype=torch.long, device=tokens.device),
        torch.full((PROMPT_SLOTS,), 2, dtype=torch.long, device=tokens.device),
    ))
    if tokens.shape != (64, width):
        raise RuntimeError(f"invalid key64 shape: {tuple(tokens.shape)}")
    return Key64Output(
        tokens=tokens,
        modality_ids=modality_ids,
        positions=positions,
        valid=valid,
        detail_source=detail_source,
        audit={
            "protocol": KEY64_PROTOCOL,
            "layout": list(KEY64_LAYOUT),
            "merged_image_hw": list(merged_hw),
            "pooled_image_hw": list(output_hw),
            **detail_audit,
            **context_audit,
        },
    )
