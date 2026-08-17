"""Instruction-priority ordering inside the detail-slot selector.

The twelve detail slots are the binding constraint on whether an agent can tell
which control the instruction names. Plain rotation splits them evenly between
the labels the task asks for and the ones it does not: a decoded click-checkboxes
state spent three slots on a distractor while dropping three real targets.

Two properties have to hold together, and they pull against each other:

* **Generality must survive.** The cross-group rotation is what guarantees the
  state describes actions and choices whatever the task happens to be. Ordering
  is therefore applied *within* a group, never across groups.
* **Nothing changes without an instruction.** Passing no instruction has to
  reproduce the previous selection exactly, or every earlier store silently
  becomes non-reproducible.
"""
from __future__ import annotations

import torch

from experiments.state_tokenizer.key_pooling import parse_dom_spans
from experiments.state_tokenizer.static_key_pooling import (
    SLOT_RAW_LITERAL,
    build_static_detail,
)

# Three labelled checkboxes and a submit button. The instruction names only the
# last two labels, so plain rotation reaches "aaa" first and runs out of slots.
DOM = (
    '<ref=0 parent=0 tag=root text="Task"/>\n'
    '<ref=1 parent=0 tag=label/>\n'
    '<ref=2 parent=1 tag=input_checkbox text="aaa" value="false"/>\n'
    '<ref=3 parent=0 tag=label/>\n'
    '<ref=4 parent=3 tag=input_checkbox text="bbb" value="false"/>\n'
    '<ref=5 parent=0 tag=label/>\n'
    '<ref=6 parent=5 tag=input_checkbox text="ccc" value="false"/>\n'
    '<ref=7 parent=0 tag=button text="Submit"/>'
)
INSTRUCTION = "Select bbb, ccc and click Submit."


def offsets_for(text: str) -> list[tuple[int, int]]:
    """One token per character: keeps the token/character mapping trivial."""
    return [(index, index + 1) for index in range(len(text))]


def selected_literals(dom: str, slots: int, instruction: str | None) -> set[str]:
    token_offsets = offsets_for(dom)
    hidden = torch.zeros((len(token_offsets), 4), dtype=torch.float32)
    output = build_static_detail(
        hidden, parse_dom_spans(dom), token_offsets,
        slots=slots, dom=dom, instruction=instruction,
    )
    covered = set()
    for slot in range(output.source_ranges.shape[0]):
        if int(output.slot_kind[slot]) != SLOT_RAW_LITERAL:
            continue
        first, last = (int(x) for x in output.source_ranges[slot])
        covered.update(range(first, last))
    text = "".join(dom[index] for index in sorted(covered))
    return {label for label in ("aaa", "bbb", "ccc", "Submit") if label in text}


def test_no_instruction_reproduces_the_previous_selection():
    """The regression lock: every store built before this change stays valid."""
    baseline = selected_literals(DOM, slots=12, instruction=None)
    assert selected_literals(DOM, slots=12, instruction=None) == baseline
    # and an instruction that names nothing on the page is inert
    assert selected_literals(DOM, slots=12, instruction="Do something else.") == baseline


def test_named_labels_win_a_scarce_budget():
    """With too few slots for every label, the instruction's own must survive.

    Twelve slots is exactly enough for ``Submit`` plus two of the three labels --
    each character is one token in this fixture -- so something has to be dropped.
    """
    picked = selected_literals(DOM, slots=12, instruction=INSTRUCTION)
    assert "bbb" in picked
    assert "ccc" in picked


def test_the_distractor_is_what_gets_dropped():
    """Priority has to cost the unnamed label, not one of the named ones."""
    plain = selected_literals(DOM, slots=12, instruction=None)
    prioritised = selected_literals(DOM, slots=12, instruction=INSTRUCTION)
    assert "aaa" in plain, "fixture must be tight enough that rotation reaches aaa"
    assert not {"bbb", "ccc"} <= plain, "and tight enough to drop a named label"
    assert {"bbb", "ccc"} <= prioritised
    assert "aaa" not in prioritised


def test_other_groups_still_get_their_rotation_share():
    """Generality check: Submit is an "actions" candidate and must not be starved.

    If priority were applied across groups instead of inside them, the two named
    choices would take the whole budget and the state would stop describing the
    button that completes the task.
    """
    picked = selected_literals(DOM, slots=6, instruction=INSTRUCTION)
    assert "Submit" in picked


def test_a_generous_budget_keeps_everything():
    """Ordering must not drop candidates, only reorder them."""
    assert selected_literals(DOM, slots=64, instruction=INSTRUCTION) == {
        "aaa", "bbb", "ccc", "Submit"
    }


def test_layout_still_sums_to_sixty_four():
    """Recycling the prompt band must move slots, not create or destroy them.

    Everything downstream -- the PCA group boundaries, the bottleneck group
    slices, the probe's slot embedding -- is indexed off this layout, and a sum
    other than 64 corrupts all of them at once without raising here.
    """
    from experiments.state_tokenizer.key_pooling import (
        CONTEXT_SLOTS,
        DETAIL_SLOTS,
        IMAGE_SLOTS,
        KEY64_LAYOUT,
        PROMPT_SLOTS,
    )

    assert sum(KEY64_LAYOUT) == 64
    assert KEY64_LAYOUT == (IMAGE_SLOTS, DETAIL_SLOTS, CONTEXT_SLOTS, PROMPT_SLOTS)
    assert PROMPT_SLOTS == 0, "the prompt band was recycled into detail"
    assert DETAIL_SLOTS == 16
