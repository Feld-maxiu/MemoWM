"""Contract between the AXTree serializer and everything downstream of it.

The v8 switch rests on one claim: an AXTree can be written in ``compact_dom``'s
wire format so that ``key_pooling`` and ``v2_data`` keep working untouched. That
claim is cheap to state and easy to break -- a role that stops mapping onto the
tag vocabulary, a flags list that drifts off four entries, a parent pointing at a
node the skip filter removed. None of those raise at serialization time; they
surface as a silently worse representation several GPU-hours later.

The second claim is the reason for the switch at all: with the accessible name on
the control itself, an instruction target should reach the detail slots as a
``priority == 1`` candidate *directly*, without the "text node whose parent is
interactive" hop that compact DOM needed.

Fixtures are real ``axtree_object`` payloads reduced to the fields the serializer
reads, so a change in what BrowserGym emits shows up here rather than in a rebuild.
"""
from __future__ import annotations

import pytest

from experiments.state_tokenizer.common import (
    axtree_rows,
    compact_axtree,
    derive_probe_labels_axtree,
)
from experiments.state_tokenizer.key_pooling import detail_candidates, parse_dom_spans
from experiments.state_tokenizer.v2_data import derive_v2_targets, parse_compact_dom


def node(node_id, role, *, parent=None, bid=None, name=None, value=None,
         properties=None, children=(), ignored=False):
    payload = {
        "nodeId": node_id,
        "parentId": parent,
        "role": {"type": "role", "value": role},
        "browsergym_id": bid,
        "ignored": ignored,
        "childIds": list(children),
    }
    if name is not None:
        payload["name"] = {"type": "computedString", "value": name}
    if value is not None:
        payload["value"] = {"type": "string", "value": value}
    if properties:
        payload["properties"] = [
            {"name": key, "value": {"type": "token", "value": item}}
            for key, item in properties.items()
        ]
    return payload


# click-checkboxes: two labelled checkboxes and a submit button. "Nb" is the
# instruction target, and in the AXTree it is the checkbox's own accessible name.
CHECKBOXES = {"nodes": [
    node("2", "RootWebArea", name="Click Checkboxes Task", children=["3"],
         properties={"focused": True}),
    node("3", "generic", parent="2", bid="12", children=["17", "20", "27"]),
    node("17", "LabelText", parent="3", bid="17", name="", children=["18", "19"]),
    node("18", "checkbox", parent="17", bid="18", name="11LO",
         properties={"checked": "false", "focusable": True}),
    node("19", "StaticText", parent="17", name="11LO"),
    node("20", "LabelText", parent="3", bid="20", name="", children=["21"]),
    node("21", "checkbox", parent="20", bid="21", name="Nb",
         properties={"checked": "true", "focusable": True}),
    node("27", "button", parent="3", bid="27", name="Submit",
         properties={"focusable": True}),
]}

# enter-text after a fill: the value lives on the textbox node, not in the name.
TEXTBOX = {"nodes": [
    node("2", "RootWebArea", name="Enter Text Task", children=["14", "15"]),
    node("14", "textbox", parent="2", bid="14", name="", value="state-04771",
         properties={"focused": True, "focusable": True}),
    node("15", "button", parent="2", bid="15", name="Submit",
         properties={"focusable": True}),
]}

DIALOG = {"nodes": [
    node("2", "RootWebArea", name="Click Dialog Task", children=["9"]),
    node("9", "dialog", parent="2", bid="9", name="Dialog", children=["10"],
         properties={"modal": True}),
    node("10", "button", parent="9", bid="10", name="Close",
         properties={"focusable": True}),
]}

ALL_FIXTURES = [CHECKBOXES, TEXTBOX, DIALOG]


@pytest.mark.parametrize("axtree", ALL_FIXTURES)
def test_both_parsers_accept_the_serialization(axtree):
    """The wire format is the whole compatibility story; both readers must agree.

    ``parse_dom_spans`` (pooling) and ``parse_compact_dom`` (labels) are separate
    implementations of the same grammar, so passing one proves little.
    """
    text = compact_axtree(axtree)
    spans = parse_dom_spans(text)
    elements = parse_compact_dom(text)
    assert len(spans) == len(elements) == len(text.splitlines())


@pytest.mark.parametrize("axtree", ALL_FIXTURES)
def test_flags_are_always_four_wide(axtree):
    """``v2_data.parse_compact_dom`` raises on any other width."""
    for element in parse_compact_dom(compact_axtree(axtree)):
        assert len(element["flags"]) == 4


@pytest.mark.parametrize("axtree", ALL_FIXTURES)
def test_parent_references_stay_closed(axtree):
    """A parent must name a row that survived the skip filter, or 0/-1.

    ``generic`` wrappers are dropped for the token saving, so every dropped node
    has to be spliced out of the parent chain rather than left dangling.
    """
    rows = axtree_rows(axtree)
    refs = {row["ref"] for row in rows} | {"0", "-1"}
    for row in rows:
        assert row["parent"] in refs, f"dangling parent {row['parent']!r}"


def test_target_reaches_detail_as_a_direct_priority_one_candidate():
    """The point of the switch: no ``t``-node-with-interactive-parent hop.

    In compact DOM "Nb" sat on a separate text node and only became a priority-1
    candidate through its parent's tag. Here it must qualify on the checkbox row
    itself, which is what makes the binding survive into a single raw slot.
    """
    text = compact_axtree(CHECKBOXES)
    nodes = parse_dom_spans(text)
    by_index = {n.index: n for n in nodes}
    direct = {
        text[c.start:c.stop]
        for c in detail_candidates(nodes)
        if c.priority == 1 and c.field == "text"
        and by_index[c.node_index].tag == "input_checkbox"
    }
    assert "Nb" in direct
    assert "11LO" in direct


def test_roles_map_onto_the_existing_tag_vocabulary():
    rows = {row["ref"]: row for row in axtree_rows(CHECKBOXES)}
    assert rows["21"]["tag"] == "input_checkbox"
    assert rows["21"]["text"] == "Nb"
    assert rows["21"]["value"] == "true"      # checked -> the DOM's value convention
    assert rows["18"]["value"] == "false"
    assert rows["27"]["tag"] == "button"
    assert rows["17"]["tag"] == "label"


def test_textbox_value_survives():
    """``value exact`` and ``has_random_value`` both read this field.

    The value is a top-level ``value`` on the node, not part of the accessible
    name -- reading the name instead yields an empty string and silently kills
    the headline reconstruction metric.
    """
    rows = {row["ref"]: row for row in axtree_rows(TEXTBOX)}
    assert rows["14"]["tag"] == "input_text"
    assert rows["14"]["value"] == "state-04771"
    assert rows["14"]["flags"][0] == 1, "focused must land in flags[0]"


def test_body_text_without_a_bid_is_kept():
    """StaticText never carries a bid, but in scroll-text-2 it *is* the content."""
    passage = {"nodes": [
        node("2", "RootWebArea", name="Scroll Text", children=["3"]),
        node("3", "paragraph", parent="2", bid="3", children=["4"]),
        node("4", "StaticText", parent="3", name="Faucibus nibh nisl fermentum."),
    ]}
    rows = axtree_rows(passage)
    texts = [row["text"] for row in rows]
    assert "Faucibus nibh nisl fermentum." in texts
    assert [row for row in rows if row["text"] and "Faucibus" in row["text"]][0]["ref"] == "-1"


def test_tampered_sidecar_lands_in_flags_one():
    """AXTree cannot know MiniWoB's per-episode tampered bit; the sidecar carries it."""
    rows = {r["ref"]: r for r in axtree_rows(CHECKBOXES, [{"bid": "21", "tampered": True}])}
    assert rows["21"]["flags"][1] == 1
    assert rows["18"]["flags"][1] == 0


def test_probe_labels_and_v2_targets_agree_with_the_fixture():
    record = {
        "dom": compact_axtree(CHECKBOXES),
        "instruction": "Select Nb and click Submit.",
        "probe": derive_probe_labels_axtree(axtree_rows(CHECKBOXES)),
    }
    targets = derive_v2_targets(record)
    assert targets["static"]["has_checkbox"] is True
    assert targets["static"]["has_button"] is True
    assert targets["static"]["has_textbox"] is False
    assert targets["dynamic"]["checkbox_0_checked"] is False   # 11LO, document order
    assert targets["dynamic"]["checkbox_1_checked"] is True    # Nb
    assert "nb" in targets["instruction_overlap_words"]


def test_dialog_role_is_recognised_without_class_sniffing():
    """compact DOM had to look for ``ui-dialog`` in ``classes``; AXTree states the role."""
    record = {
        "dom": compact_axtree(DIALOG),
        "instruction": "Close the dialog.",
        "probe": derive_probe_labels_axtree(axtree_rows(DIALOG)),
    }
    assert derive_v2_targets(record)["static"]["has_dialog"] is True


def test_document_order_is_preserved():
    """The raw ``nodes`` array is not in document order; position slots depend on it."""
    refs = [row["ref"] for row in axtree_rows(CHECKBOXES)]
    assert refs.index("17") < refs.index("21") < refs.index("27")


def test_checkbox_label_becomes_a_choices_candidate():
    """The AXTree regression that halved target coverage, pinned.

    compact DOM put a checkbox's label on a sibling ``t`` node under the
    ``label``, which ``static_candidates`` grouped as "choices" via the parent-tag
    rule. An accessibility tree puts the accessible name on the control itself,
    where none of the group rules reached it -- so the candidate was dropped
    before slot allocation ever ran and 52% of instruction targets vanished with
    no error anywhere. The unchecked case is the one that broke: a *checked* box
    was still caught by the associated_with_checked rule, which is why the
    failure looked partial rather than total.
    """
    from experiments.state_tokenizer.static_key_pooling import static_candidates

    text = compact_axtree(CHECKBOXES)
    candidates = {
        text[c.start:c.stop]: c.group for c in static_candidates(parse_dom_spans(text))
    }
    # 11LO is unchecked -- this is the case that was dropped entirely.
    assert candidates.get("11LO") == "choices"
    # Nb is checked, so the earlier associated_with_checked rule claims it for
    # current_state. That is pre-existing v7 behaviour and must not shift.
    assert candidates.get("Nb") == "current_state"
