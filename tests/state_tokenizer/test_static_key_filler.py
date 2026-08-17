"""Filler filtering inside the static detail selector.

The filter changes what the tokenizer preserves, so the risk is asymmetric: a
regression silently degrades every downstream representation, and the deleted
tokens cannot be recovered later. These tests pin the default (off, bit-identical
to the shipped v5 behaviour) and the one behaviour the v6 rebuild is for -- a
lorem-padded value attribute falling back under the raw-slot threshold.
"""
from __future__ import annotations

import pytest
import torch

from experiments.state_tokenizer.key_pooling import parse_dom_spans
from experiments.state_tokenizer.static_key_pooling import (
    SHORT_SPAN_MAX_TOKENS,
    SLOT_POOLED_LITERAL,
    SLOT_RAW_LITERAL,
    build_static_detail,
    drop_filler_tokens,
    materialize_candidates,
    static_candidates,
)

# the copy-paste shape: a single real value at the end of a lorem tail
COPY_PASTE = (
    '<ref=1 parent=0 tag=body box=0.0,0.0,485.0,210.0/>\n'
    '<ref=2 parent=1 tag=textarea box=2.0,57.0,156.0,106.0 '
    'value="Adipiscing enim id diam. Fermentum. Auctor. In vestibulum aenean '
    'tempus ut imperdiet cras orci consequat nunc tellus aliquam hendrerit '
    'pulvinar habitant morbi sed amet elementum vulputate vel state-88347" '
    'id="ta" flags=1,1,0,1/>\n'
    '<ref=3 parent=1 tag=button box=2.0,165.0,95.5,31.0 text="Submit" id="b" flags=0,0,0,1/>\n'
)


def _tokenize(text: str):
    """Character-level offsets: enough to exercise span logic without a model."""
    return [(index, index + 1) for index in range(len(text))]


def _word_offsets(text: str):
    """Whitespace tokenisation, closer to how BPE groups words."""
    offsets, cursor = [], 0
    for piece in text.split(" "):
        if piece:
            start = text.index(piece, cursor)
            offsets.append((start, start + len(piece)))
            cursor = start + len(piece)
    return offsets


def test_filter_is_off_by_default():
    """v5 artefacts must stay reproducible bit for bit."""
    nodes = parse_dom_spans(COPY_PASTE)
    offsets = _word_offsets(COPY_PASTE)
    hidden = torch.zeros((len(offsets), 4))
    plain = build_static_detail(hidden, nodes, offsets)
    same = build_static_detail(hidden, nodes, offsets, dom=COPY_PASTE, filter_filler=False)
    assert torch.equal(plain.source_ranges, same.source_ranges)
    assert torch.equal(plain.slot_kind, same.slot_kind)
    assert torch.equal(plain.valid, same.valid)


def test_filter_requires_the_dom_text():
    nodes = parse_dom_spans(COPY_PASTE)
    offsets = _word_offsets(COPY_PASTE)
    with pytest.raises(ValueError):
        build_static_detail(
            torch.zeros((len(offsets), 4)), nodes, offsets, filter_filler=True
        )


def test_lorem_is_removed_but_the_value_survives():
    nodes = parse_dom_spans(COPY_PASTE)
    offsets = _word_offsets(COPY_PASTE)
    materialized = materialize_candidates(static_candidates(nodes), offsets)
    filtered = drop_filler_tokens(materialized, COPY_PASTE, offsets)

    before = {c.candidate.field: len(c.token_indices) for c in materialized}
    after = {c.candidate.field: len(c.token_indices) for c in filtered}
    assert after["value"] < before["value"], (before, after)

    kept = next(c for c in filtered if c.candidate.field == "value")
    text = " ".join(COPY_PASTE[a:b] for a, b in (offsets[i] for i in kept.token_indices))
    assert "state-88347" in text
    for word in ("Adipiscing", "vestibulum", "Fermentum"):
        assert word not in text, text


def test_the_value_falls_back_under_the_raw_threshold():
    """The whole point: pooled -> raw, so the digits keep their own slots."""
    nodes = parse_dom_spans(COPY_PASTE)
    offsets = _word_offsets(COPY_PASTE)
    materialized = materialize_candidates(static_candidates(nodes), offsets)
    value_before = next(c for c in materialized if c.candidate.field == "value")
    assert len(value_before.token_indices) > SHORT_SPAN_MAX_TOKENS

    filtered = drop_filler_tokens(materialized, COPY_PASTE, offsets)
    value_after = next(c for c in filtered if c.candidate.field == "value")
    assert len(value_after.token_indices) <= SHORT_SPAN_MAX_TOKENS


def test_button_text_is_untouched_by_the_filter():
    nodes = parse_dom_spans(COPY_PASTE)
    offsets = _word_offsets(COPY_PASTE)
    materialized = materialize_candidates(static_candidates(nodes), offsets)
    filtered = drop_filler_tokens(materialized, COPY_PASTE, offsets)
    before = [c for c in materialized if c.candidate.field == "text"]
    after = [c for c in filtered if c.candidate.field == "text"]
    assert len(before) == len(after)
    for left, right in zip(before, after):
        assert left.token_indices == right.token_indices


def test_filtering_shifts_the_value_from_pooled_to_raw_slots():
    """A long span is pooled only when it overflows the remaining budget.

    That is the real trigger: the selector emits raw slots for a long candidate
    that still fits, so the value is only smeared once the lorem pushes it past
    the free slots. The fixture therefore needs a tail long enough to overflow.
    """
    nodes = parse_dom_spans(COPY_PASTE)
    offsets = _word_offsets(COPY_PASTE)
    hidden = torch.zeros((len(offsets), 4))
    plain = build_static_detail(hidden, nodes, offsets)
    filtered = build_static_detail(
        hidden, nodes, offsets, dom=COPY_PASTE, filter_filler=True
    )
    pooled_before = int((plain.slot_kind == SLOT_POOLED_LITERAL).sum())
    pooled_after = int((filtered.slot_kind == SLOT_POOLED_LITERAL).sum())
    raw_after = int((filtered.slot_kind == SLOT_RAW_LITERAL).sum())
    assert pooled_after < pooled_before, (pooled_before, pooled_after)
    assert raw_after > 0
