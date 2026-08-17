"""Filler filter invariants.

The filter deletes DOM content before it is ever encoded, so a mistake here is
unrecoverable downstream -- no codebook or loss weighting can restore a token
that was dropped at tokenisation. These pin the properties that make deletion
safe, all of them traced to concrete corpus cases that earlier rule designs got
wrong.
"""
from __future__ import annotations

import pytest

from experiments.state_tokenizer.filler_vocabulary import (
    LOREM_WORDS,
    filler_runs,
    is_filler,
)

# every one of these was found in the corpus carrying real meaning
UI_WORDS = (
    "true",       # value="true" -- checkbox/radio state
    "close",      # text="Close" -- the target of click-dialog-2
    "search", "results",   # text= in use-autocomplete
    "no",         # a button label, and a prefix of the lorem word "non"
    "submit", "login", "cancel", "ok", "yes", "next", "previous",
    "username", "password", "state",
)


@pytest.mark.parametrize("word", UI_WORDS)
def test_real_ui_words_are_never_filler(word):
    assert not is_filler(word)


def test_lorem_words_and_truncations_are_filler():
    for word in ("sed", "amet", "eget", "vestibulum", "adipiscing"):
        assert is_filler(word), word
    # the DOM cuts attributes mid-word, leaving prefixes behind
    for word in ("elementu", "pharet", "sollicit", "malesua", "vulp"):
        assert is_filler(word), word


def test_short_prefixes_are_left_alone():
    """Below four characters the prefix rule starts eating real words."""
    for word in ("no", "se", "di", "do", "fa", "cr"):
        assert not is_filler(word), word


def test_a_run_is_required_so_standalone_labels_survive():
    """This is the property that makes a global vocabulary safe to apply."""
    assert filler_runs(["Close"], 3) == [False]
    assert filler_runs(["Search", "Results"], 3) == [False, False]
    # "in" and "at" are lorem words but also ordinary English
    assert filler_runs(["Sign", "in"], 3) == [False, False]
    assert filler_runs(["Look", "at", "this"], 3) == [False, False, False]


def test_the_copy_paste_case_strips_lorem_and_keeps_the_value():
    """The case the whole change exists for: value at the end of a lorem tail."""
    words = "Adipiscing enim id diam Fermentum Auctor In vestibulum aenean state".split()
    keep = filler_runs(words, 3)
    assert all(keep[:-1]), keep      # the lorem run goes
    assert keep[-1] is False         # "state" of state-88347 stays


def test_a_run_shorter_than_the_threshold_is_kept():
    assert filler_runs(["Sign", "in", "at", "Home"], 3) == [False] * 4
    assert filler_runs(["Sign", "in", "at", "sed", "Home"], 3) == [
        False, True, True, True, False,
    ]


def test_min_run_is_validated():
    with pytest.raises(ValueError):
        filler_runs(["sed"], 0)


def test_vocabulary_holds_no_digits_or_punctuation():
    """Digit-bearing tokens must be structurally unfilterable."""
    for word in LOREM_WORDS:
        assert word.isalpha() and word.islower(), word
    for word in ("state-88347", "88347", "34388", "value1"):
        assert not is_filler(word), word


# --------------------------------------------------------------------------- #
# token-level masking
# --------------------------------------------------------------------------- #
def _offsets(text: str, pieces: list[str]):
    """Char offsets for a hand-written tokenisation of ``text``."""
    spans, cursor = [], 0
    for piece in pieces:
        start = text.index(piece, cursor)
        spans.append((start, start + len(piece)))
        cursor = start + len(piece)
    return spans


def test_separators_left_by_dead_words_are_swept():
    """Word removal alone leaves punctuation that keeps the span over budget."""
    from experiments.state_tokenizer.filler_vocabulary import filler_token_mask

    text = "diam. Fermentum. Auctor. state-88347"
    pieces = ["diam", ".", " Fermentum", ".", " Auctor", ".", " state", "-", "88347"]
    mask = filler_token_mask(text, _offsets(text, pieces), 3)
    kept = [piece for piece, drop in zip(pieces, mask) if not drop]
    assert kept == [" state", "-", "88347"]


def test_the_hyphen_of_a_surviving_value_is_kept():
    """One-sided rule: a separator after kept content must not be swept."""
    from experiments.state_tokenizer.filler_vocabulary import filler_token_mask

    text = "state-88347"
    pieces = ["state", "-", "88347"]
    assert filler_token_mask(text, _offsets(text, pieces), 3) == [False, False, False]


def test_a_span_with_no_filler_is_untouched():
    from experiments.state_tokenizer.filler_vocabulary import filler_token_mask

    text = "Search Results"
    pieces = ["Search", " Results"]
    assert filler_token_mask(text, _offsets(text, pieces), 3) == [False, False]


def test_digit_tokens_are_never_masked():
    """The invariant the whole change depends on."""
    from experiments.state_tokenizer.filler_vocabulary import filler_token_mask

    text = "sed amet eget 88347 vitae enim diam"
    pieces = ["sed", " amet", " eget", " 88347", " vitae", " enim", " diam"]
    mask = filler_token_mask(text, _offsets(text, pieces), 3)
    assert mask[3] is False, "a digit-bearing token was masked"
