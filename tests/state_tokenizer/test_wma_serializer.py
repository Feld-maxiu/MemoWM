"""The frozen WorldMemArena synthetic-AXTree serialization.

These rows are the only text the tokenizer sees for a WorldMemArena observation,
and the format is load-bearing in a way that is easy to break while trying to
improve it. Two rules in particular:

* the literal must ride on ``value``, not ``text``. ``static_candidates`` groups
  ``text`` only for ACTION/CHOICE/CHECKABLE tags, so ``tag=textarea text="..."``
  matches no rule and the caption is dropped before slot contention -- silently,
  with no error and no empty output, exactly the failure that once cost 52% of
  targets when the AXTree serializer landed.
* the rows must parse. ``parse_dom_spans`` raises on anything that is not
  ``<...>`` closing with ``/>``, and the pooler needs at least one node.

Torch environment only. ``key_pooling`` and ``static_key_pooling`` are torch
modules -- that dependency split is why ``slot_layout`` exists as a separate
file -- so this cannot run under the jax interpreter.
"""
from __future__ import annotations

from experiments.state_tokenizer.key_pooling import parse_dom_spans
from experiments.state_tokenizer.static_key_pooling import static_candidates
from residualmem.benchmarks.worldmemarena_tokenizer import (
    DEFAULT_AXTREE_STYLE,
    LEGACY_V1_AXTREE_STYLE,
    AxtreeStyle,
    WebObservation,
    observation_from_turn,
    synthetic_axtree,
)


def _obs(**kwargs):
    base = {"screenshot": "shot.png", "user_text": "", "captions": ()}
    base.update(kwargs)
    return WebObservation(**base)


def test_the_frozen_default_is_the_aligned_style():
    # Freezing happens before the PCA is fitted; a silent flip here would make
    # the shipped basis a different coordinate system from the one measured.
    assert DEFAULT_AXTREE_STYLE == AxtreeStyle(
        canonical_rows=True, root_node=True, explicit_no_instruction=True
    )
    assert not DEFAULT_AXTREE_STYLE.split_captions


def test_rows_match_the_training_wire_format():
    dom = synthetic_axtree(_obs(captions=("hello.",)))
    for line in dom.splitlines():
        assert line.startswith("<ref=")
        assert line.endswith("/>")
        assert not line.endswith(" />")   # compact_axtree leaves no space
        assert "<node " not in line
        assert '"' not in line.split("tag=")[0]  # ref/parent are unquoted


def test_the_literal_rides_on_value_not_text():
    # The whole reason the caption survives into the detail slots.
    dom = synthetic_axtree(_obs(captions=("a caption",)))
    candidates = static_candidates(parse_dom_spans(dom))
    assert [c.group for c in candidates].count("current_state") >= 1

    # And the counterfactual, so the reason is recorded rather than assumed.
    dropped = '<ref=1 parent=0 tag=textarea text="a caption"/>'
    assert static_candidates(parse_dom_spans(dropped)) == []


def test_parent_zero_resolves_to_an_emitted_root():
    dom = synthetic_axtree(_obs(captions=("hello.",)))
    nodes = parse_dom_spans(dom)
    refs = {node.ref for node in nodes}
    parents = {node.parent for node in nodes if node.parent}
    assert parents <= refs, "every parent reference must name a node that exists"
    assert nodes[0].tag == "root"


def test_no_instruction_keeps_one_template():
    # On WorldMemArena web the instruction is always absent, so without this the
    # benchmark path always took a different shape than training did.
    without = synthetic_axtree(_obs(captions=("hello.",)))
    with_text = synthetic_axtree(_obs(user_text="do the thing", captions=("hello.",)))
    assert "<no_instruction>" in without
    assert "<no_instruction>" not in with_text
    assert len(without.splitlines()) == len(with_text.splitlines())


def test_every_style_still_parses_and_is_non_empty():
    observations = [
        _obs(),                                   # screenshot only
        _obs(captions=("one.", "two.")),
        _obs(user_text="instruction", captions=("cap",)),
        WebObservation(screenshot=None, user_text="", captions=()),   # empty control
    ]
    styles = [
        DEFAULT_AXTREE_STYLE,
        LEGACY_V1_AXTREE_STYLE,
        AxtreeStyle(canonical_rows=True),
        AxtreeStyle(canonical_rows=True, split_captions=True),
    ]
    for style in styles:
        for observation in observations:
            dom = synthetic_axtree(observation, style)
            assert dom.strip(), "the pooler requires at least one node"
            assert parse_dom_spans(dom), dom


def test_legacy_v1_still_reproduces_the_old_rows():
    dom = synthetic_axtree(_obs(captions=("hello.",)), LEGACY_V1_AXTREE_STYLE)
    assert dom == '<node ref="wma_caption_0" tag="textarea" value="hello." />'


def test_captions_are_not_split_by_default():
    # split_captions measured worse and is off; a caption stays one node.
    dom = synthetic_axtree(_obs(captions=("One. Two. Three.",)))
    textareas = [l for l in dom.splitlines() if "tag=textarea" in l and "no_instruction" not in l]
    assert len(textareas) == 1


def test_assistant_turns_are_still_refused():
    # The leakage guard predates this module and must not regress: assistant
    # text is policy output, not part of the observation.
    class _Turn:
        role = "assistant"
        text = "I can see a button"
        attachments = ()

    try:
        observation_from_turn(_Turn())
    except ValueError:
        pass
    else:
        raise AssertionError("assistant turn was accepted as an observation")
