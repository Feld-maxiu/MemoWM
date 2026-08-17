"""Static semantic selector for the general-purpose key64_v2 representation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from .filler_vocabulary import filler_token_mask
from .key_pooling import (
    CONTEXT_SLOTS,
    DETAIL_SLOTS,
    IMAGE_SLOTS,
    KEY64_LAYOUT,
    PROMPT_SLOTS,
    AttributeSpan,
    NodeSpan,
    _adaptive_pool_1d,
    parse_dom_spans,
    pool_full_nodes,
    pool_image_2d,
    token_indices_for_span,
)


STATIC_KEY64_PROTOCOL = "fixed_prompt_key64_static_v2"
STATIC_KEY64_FILTERED_PROTOCOL = "fixed_prompt_key64_static_v3_filtered"
SHORT_SPAN_MAX_TOKENS = 8
FILLER_MIN_RUN = 3
GROUPS = ("current_state", "actions", "choices", "auxiliary")
GROUP_RANK = {name: index for index, name in enumerate(GROUPS)}
GROUP_SCHEDULE = (
    "current_state", "actions", "current_state",
    "choices", "current_state", "auxiliary",
)
EDITABLE_TAGS = {"input_text", "input_password", "textarea", "select"}
ACTION_TAGS = {"button", "input_submit", "input_button", "a"}
CHOICE_TAGS = {"option", "label"}
CHECKABLE_TAGS = {"input_checkbox", "input_radio"}
AUXILIARY_FIELDS = {"aria-label", "aria_label", "placeholder", "title", "tooltip"}

SLOT_IMAGE_POOLED = 0
SLOT_RAW_LITERAL = 1
SLOT_POOLED_LITERAL = 2
SLOT_NODE_CONTEXT = 3
SLOT_PROMPT_POOLED = 4
SLOT_PADDING = 5


@dataclass(frozen=True)
class StaticCandidate:
    group: str
    start: int
    stop: int
    node_index: int
    field: str
    tag: str


@dataclass(frozen=True)
class MaterializedCandidate:
    candidate: StaticCandidate
    token_indices: tuple[int, ...]


@dataclass
class StaticDetailOutput:
    tokens: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    slot_kind: torch.Tensor
    source_ranges: torch.Tensor
    audit: dict


@dataclass
class StaticKey64Output:
    tokens: torch.Tensor
    modality_ids: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    slot_kind: torch.Tensor
    detail_ranges: torch.Tensor
    audit: dict


def _attribute(node: NodeSpan, key: str) -> AttributeSpan | None:
    return next((item for item in node.attributes if item.key == key), None)


def deduplicate_candidates(candidates: Sequence[StaticCandidate]) -> list[StaticCandidate]:
    """Keep only the highest-priority semantic role for each exact char span."""
    deduplicated: dict[tuple[int, int], StaticCandidate] = {}
    for candidate in candidates:
        key = (candidate.start, candidate.stop)
        current = deduplicated.get(key)
        if current is None or GROUP_RANK[candidate.group] < GROUP_RANK[current.group]:
            deduplicated[key] = candidate
    return sorted(
        deduplicated.values(),
        key=lambda item: (item.start, item.stop, GROUP_RANK[item.group], item.field),
    )


def static_candidates(nodes: Sequence[NodeSpan]) -> list[StaticCandidate]:
    """Collect human-visible literals using only fixed DOM semantics."""
    by_ref = {node.ref: node for node in nodes if node.ref}
    checked_labels = set()
    for node in nodes:
        if node.tag not in CHECKABLE_TAGS:
            continue
        value = _attribute(node, "value")
        parent = by_ref.get(node.parent)
        if (
            value is not None
            and value.value.lower() == "true"
            and parent is not None
            and parent.tag == "label"
        ):
            checked_labels.add(parent.ref)

    output = []
    for node in nodes:
        parent = by_ref.get(node.parent)
        parent_tag = parent.tag if parent is not None else ""
        associated_with_checked = (
            node.ref in checked_labels
            or node.parent in checked_labels
        )
        for attribute in node.attributes:
            if attribute.start >= attribute.stop:
                continue
            group = None
            if attribute.key == "value" and node.tag in EDITABLE_TAGS:
                group = "current_state"
            elif attribute.key == "text" and associated_with_checked:
                group = "current_state"
            elif attribute.key == "text" and node.tag in ACTION_TAGS:
                group = "actions"
            elif (
                attribute.key == "text"
                and node.tag == "t"
                and parent_tag in ACTION_TAGS
            ):
                group = "actions"
            elif attribute.key in {"text", "value"} and node.tag in CHOICE_TAGS:
                group = "choices"
            elif attribute.key == "text" and node.tag in CHECKABLE_TAGS:
                # A checkbox's own label. In compact DOM this text lived on a
                # sibling ``t`` node under the ``label``, so it was caught by the
                # CHOICE_TAGS parent rule below; an accessibility tree puts the
                # accessible name on the control itself, where no rule reached it
                # and the candidate was silently dropped. Same literal, same role
                # for the reader, so it belongs in the same group.
                group = "choices"
            elif (
                attribute.key == "text"
                and node.tag == "t"
                and parent_tag in CHOICE_TAGS
            ):
                group = "choices"
            elif attribute.key in AUXILIARY_FIELDS:
                group = "auxiliary"
            if group is not None:
                output.append(StaticCandidate(
                    group=group,
                    start=attribute.start,
                    stop=attribute.stop,
                    node_index=node.index,
                    field=attribute.key,
                    tag=node.tag,
                ))
    return deduplicate_candidates(output)


def materialize_candidates(
    candidates: Sequence[StaticCandidate],
    token_offsets: Sequence[tuple[int, int]],
) -> list[MaterializedCandidate]:
    output = []
    for candidate in candidates:
        indices = tuple(token_indices_for_span(
            token_offsets, candidate.start, candidate.stop
        ))
        if indices:
            output.append(MaterializedCandidate(candidate, indices))
    return output


def _next_candidate_that_fits(
    queue: Sequence[MaterializedCandidate],
    cursor: int,
    selected: set[int],
    remaining: int,
) -> tuple[MaterializedCandidate | None, tuple[int, ...], int, int]:
    skipped = 0
    while cursor < len(queue):
        item = queue[cursor]
        cursor += 1
        fresh = tuple(index for index in item.token_indices if index not in selected)
        if not fresh:
            continue
        if len(fresh) <= remaining:
            return item, fresh, cursor, skipped
        skipped += 1
    return None, (), cursor, skipped


def _first_long_candidate(
    queues: dict[str, list[MaterializedCandidate]],
) -> MaterializedCandidate | None:
    for group in GROUP_SCHEDULE:
        if queues[group]:
            return queues[group][0]
    return None


def _pool_with_ranges(
    values: torch.Tensor,
    source_indices: Sequence[int],
    slots: int,
) -> tuple[torch.Tensor, list[tuple[int, int]], torch.Tensor]:
    if slots < 1 or len(source_indices) < 1:
        raise ValueError("long-span pooling requires positive tokens and slots")
    slots = min(slots, len(source_indices))
    pooled = _adaptive_pool_1d(values, slots)
    ranges = []
    positions = []
    for index in range(slots):
        start = math.floor(index * len(source_indices) / slots)
        stop = math.ceil((index + 1) * len(source_indices) / slots)
        stop = max(start + 1, min(stop, len(source_indices)))
        rows = source_indices[start:stop]
        ranges.append((min(rows), max(rows) + 1))
        positions.append(sum(row + 0.5 for row in rows) / len(rows))
    return (
        pooled,
        ranges,
        torch.tensor(positions, dtype=torch.float32, device=values.device),
    )


def drop_filler_tokens(
    materialized: Sequence[MaterializedCandidate],
    dom: str,
    token_offsets: Sequence[tuple[int, int]],
    min_run: int = FILLER_MIN_RUN,
) -> list[MaterializedCandidate]:
    """Remove lorem tokens from each candidate, dropping ones left empty.

    A candidate that loses its filler can fall back under ``short_limit`` and so
    earn raw slots instead of being averaged into a pooled long span. That is the
    point: in copy-paste the five-digit value sits at the end of a lorem-padded
    attribute (27 tokens), and stripping the filler leaves 7 -- short enough for
    the value to keep its own slots rather than be smeared across a pooled one.

    Candidates that are entirely filler disappear, which returns their budget to
    real content.
    """
    kept = []
    for item in materialized:
        indices = item.token_indices
        span = (token_offsets[indices[0]][0], token_offsets[indices[-1]][1])
        text = dom[span[0]:span[1]]
        local = [
            (token_offsets[index][0] - span[0], token_offsets[index][1] - span[0])
            for index in indices
        ]
        mask = filler_token_mask(text, local, min_run)
        fresh = tuple(
            index for index, drop in zip(indices, mask) if not drop
        )
        if fresh:
            kept.append(MaterializedCandidate(item.candidate, fresh))
    return kept


def build_static_detail(
    dom_hidden: torch.Tensor,
    nodes: Sequence[NodeSpan],
    token_offsets: Sequence[tuple[int, int]],
    *,
    slots: int = DETAIL_SLOTS,
    short_limit: int = SHORT_SPAN_MAX_TOKENS,
    dom: str | None = None,
    filter_filler: bool = False,
    instruction: str | None = None,
) -> StaticDetailOutput:
    materialized = materialize_candidates(static_candidates(nodes), token_offsets)
    if filter_filler:
        if dom is None:
            raise ValueError("filter_filler needs the DOM text")
        materialized = drop_filler_tokens(materialized, dom, token_offsets)
    short_queues = {
        group: [
            item for item in materialized
            if item.candidate.group == group and len(item.token_indices) <= short_limit
        ]
        for group in GROUPS
    }
    if instruction:
        if dom is None:
            raise ValueError("instruction priority needs the DOM text")
        # Order *within* each group only. The cross-group rotation below is what
        # keeps the state generic -- it guarantees actions and choices are
        # represented whatever the task asks for -- so reordering across groups
        # would trade generality for task fit. Reordering inside a group spends
        # that group's own share on the controls the instruction actually names.
        #
        # Without this the budget is split evenly between targets and
        # distractors: a decoded click-checkboxes state spent 3 of 12 slots on a
        # label the instruction never mentions while dropping 3 that it did.
        for group, queue in short_queues.items():
            short_queues[group] = sorted(
                queue,
                key=lambda item: dom[item.candidate.start:item.candidate.stop] not in instruction,
            )
    long_queues = {
        group: [
            item for item in materialized
            if item.candidate.group == group and len(item.token_indices) > short_limit
        ]
        for group in GROUPS
    }
    cursors = {group: 0 for group in GROUPS}
    selected: set[int] = set()
    entries = []
    skipped_short = 0
    selected_short_spans = []

    while len(selected) < slots:
        progressed = False
        for group in GROUP_SCHEDULE:
            remaining = slots - len(selected)
            item, fresh, cursor, skipped = _next_candidate_that_fits(
                short_queues[group], cursors[group], selected, remaining
            )
            cursors[group] = cursor
            skipped_short += skipped
            if item is None:
                continue
            selected.update(fresh)
            for token_index in fresh:
                entries.append({
                    "position": (token_index + 0.5) / len(dom_hidden),
                    "token": dom_hidden[token_index].float(),
                    "kind": SLOT_RAW_LITERAL,
                    "range": (token_index, token_index + 1),
                })
            selected_short_spans.append({
                "group": item.candidate.group,
                "tag": item.candidate.tag,
                "field": item.candidate.field,
                "tokens": len(fresh),
            })
            progressed = True
            if len(selected) == slots:
                break
        if not progressed:
            break

    remaining = slots - len(entries)
    selected_long = None
    if remaining:
        long_item = _first_long_candidate(long_queues)
        if long_item is not None:
            fresh = tuple(
                index for index in long_item.token_indices if index not in selected
            )
            if fresh:
                if len(fresh) <= remaining:
                    for token_index in fresh:
                        entries.append({
                            "position": (token_index + 0.5) / len(dom_hidden),
                            "token": dom_hidden[token_index].float(),
                            "kind": SLOT_RAW_LITERAL,
                            "range": (token_index, token_index + 1),
                        })
                else:
                    index_tensor = torch.tensor(
                        fresh, dtype=torch.long, device=dom_hidden.device
                    )
                    pooled, ranges, centers = _pool_with_ranges(
                        dom_hidden.index_select(0, index_tensor), fresh, remaining
                    )
                    for row in range(len(pooled)):
                        entries.append({
                            "position": float(centers[row] / len(dom_hidden)),
                            "token": pooled[row],
                            "kind": SLOT_POOLED_LITERAL,
                            "range": ranges[row],
                        })
                selected_long = {
                    "group": long_item.candidate.group,
                    "tag": long_item.candidate.tag,
                    "field": long_item.candidate.field,
                    "source_tokens": len(fresh),
                    "output_slots": len(entries) - len(selected),
                }

    entries.sort(key=lambda item: (item["position"], item["kind"], item["range"]))
    width = dom_hidden.shape[-1]
    tokens = torch.zeros((slots, width), dtype=torch.float32, device=dom_hidden.device)
    positions = torch.zeros((slots,), dtype=torch.float32, device=dom_hidden.device)
    valid = torch.zeros((slots,), dtype=torch.bool, device=dom_hidden.device)
    slot_kind = torch.full(
        (slots,), SLOT_PADDING, dtype=torch.uint8, device=dom_hidden.device
    )
    source_ranges = torch.full(
        (slots, 2), -1, dtype=torch.int32, device=dom_hidden.device
    )
    for index, entry in enumerate(entries[:slots]):
        tokens[index] = entry["token"]
        positions[index] = entry["position"]
        valid[index] = True
        slot_kind[index] = entry["kind"]
        source_ranges[index] = torch.tensor(
            entry["range"], dtype=torch.int32, device=dom_hidden.device
        )
    return StaticDetailOutput(
        tokens=tokens,
        positions=positions,
        valid=valid,
        slot_kind=slot_kind,
        source_ranges=source_ranges,
        audit={
            "candidate_spans": len(materialized),
            "short_candidate_spans": sum(len(items) for items in short_queues.values()),
            "long_candidate_spans": sum(len(items) for items in long_queues.values()),
            "selected_short_spans": selected_short_spans,
            "selected_short_tokens": sum(
                item["tokens"] for item in selected_short_spans
            ),
            "skipped_short_spans": skipped_short,
            "selected_long": selected_long,
            "raw_slots": int((slot_kind == SLOT_RAW_LITERAL).sum()),
            "pooled_long_slots": int((slot_kind == SLOT_POOLED_LITERAL).sum()),
            "valid_detail_slots": int(valid.sum()),
        },
    )


def build_static_key64(
    image_hidden: torch.Tensor,
    dom_hidden: torch.Tensor,
    prompt_hidden: torch.Tensor,
    *,
    dom: str,
    dom_token_offsets: Sequence[tuple[int, int]],
    image_grid_thw: Sequence[int],
    spatial_merge_size: int,
    filter_filler: bool = False,
    instruction: str | None = None,
) -> StaticKey64Output:
    if len(dom_token_offsets) != len(dom_hidden):
        raise ValueError(
            f"DOM offset/hidden mismatch: {len(dom_token_offsets)} != {len(dom_hidden)}"
        )
    nodes = parse_dom_spans(dom)
    detail = build_static_detail(
        dom_hidden, nodes, dom_token_offsets, dom=dom, filter_filler=filter_filler,
        instruction=instruction,
    )
    image, merged_hw, output_hw = pool_image_2d(
        image_hidden, image_grid_thw, spatial_merge_size
    )
    context, context_positions, context_valid, context_audit = pool_full_nodes(
        dom_hidden, nodes, dom_token_offsets, dom=dom
    )
    # PROMPT_SLOTS is 0 once the prompt band is recycled into detail: the prompt
    # hidden states describe the same fixed observation text in every state, so
    # those slots spend part of a fixed 64-slot budget on something with no
    # per-state content. _adaptive_pool_1d cannot produce zero slots (it stacks an
    # empty list) and the position term would divide by zero, so the band is left
    # out entirely rather than pooled to width 0.
    empty = torch.zeros((0, dom_hidden.shape[-1]), dtype=torch.float32,
                        device=dom_hidden.device)
    prompt = _adaptive_pool_1d(prompt_hidden, PROMPT_SLOTS) if PROMPT_SLOTS else empty
    prompt_positions = (
        (torch.arange(PROMPT_SLOTS, device=dom_hidden.device, dtype=torch.float32) + 0.5)
        / PROMPT_SLOTS
        if PROMPT_SLOTS
        else torch.zeros((0,), dtype=torch.float32, device=dom_hidden.device)
    )
    tokens = torch.cat((image, detail.tokens, context, prompt), dim=0)
    valid = torch.cat((
        torch.ones((IMAGE_SLOTS,), dtype=torch.bool, device=tokens.device),
        detail.valid,
        context_valid,
        torch.ones((PROMPT_SLOTS,), dtype=torch.bool, device=tokens.device),
    ))
    positions = torch.cat((
        (torch.arange(IMAGE_SLOTS, device=tokens.device, dtype=torch.float32) + 0.5)
        / IMAGE_SLOTS,
        detail.positions,
        context_positions,
        prompt_positions,
    ))
    modality_ids = torch.cat((
        torch.zeros((IMAGE_SLOTS,), dtype=torch.long, device=tokens.device),
        torch.ones((DETAIL_SLOTS + CONTEXT_SLOTS,), dtype=torch.long, device=tokens.device),
        torch.full((PROMPT_SLOTS,), 2, dtype=torch.long, device=tokens.device),
    ))
    context_kind = torch.where(
        context_valid,
        torch.full_like(context_valid, SLOT_NODE_CONTEXT, dtype=torch.uint8),
        torch.full_like(context_valid, SLOT_PADDING, dtype=torch.uint8),
    )
    slot_kind = torch.cat((
        torch.full((IMAGE_SLOTS,), SLOT_IMAGE_POOLED, dtype=torch.uint8, device=tokens.device),
        detail.slot_kind,
        context_kind,
        torch.full((PROMPT_SLOTS,), SLOT_PROMPT_POOLED, dtype=torch.uint8, device=tokens.device),
    ))
    if tokens.shape != (64, dom_hidden.shape[-1]):
        raise RuntimeError(f"invalid static key64 shape: {tuple(tokens.shape)}")
    return StaticKey64Output(
        tokens=tokens,
        modality_ids=modality_ids,
        positions=positions,
        valid=valid,
        slot_kind=slot_kind,
        detail_ranges=detail.source_ranges,
        audit={
            "protocol": (
                STATIC_KEY64_FILTERED_PROTOCOL if filter_filler
                else STATIC_KEY64_PROTOCOL
            ),
            "filter_filler": filter_filler,
            "layout": list(KEY64_LAYOUT),
            "group_schedule": list(GROUP_SCHEDULE),
            "short_span_max_tokens": SHORT_SPAN_MAX_TOKENS,
            "merged_image_hw": list(merged_hw),
            "pooled_image_hw": list(output_hw),
            **detail.audit,
            **context_audit,
        },
    )
