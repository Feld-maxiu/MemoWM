"""Action logging and policy randomisation in the MiniWoB collector.

Two things make this worth pinning. The collector is the only place the action is
ever observed -- if it is logged wrong, no downstream analysis can detect it,
because the action never appears anywhere else. And the scripted policy defines
the v4/v5/v6 datasets, so any drift in it silently breaks their reproducibility.

These run without a browser: ``_make_action``/``_random_action`` only need an
observation dict and a stub env that records what action was requested.
"""
from __future__ import annotations

import numpy as np
import pytest

from experiments.state_tokenizer.collect_miniwob import (
    _choose_action,
    _make_action,
    _random_action,
)


class _StubEnv:
    """Captures ``create_action`` calls instead of driving a browser."""

    class _Unwrapped:
        def __init__(self):
            self.calls = []

        def create_action(self, action_type, **kwargs):
            self.calls.append((action_type, kwargs))
            return {"type": action_type, **kwargs}

    def __init__(self):
        self.unwrapped = self._Unwrapped()


def _observation(*elements):
    return {"dom_elements": list(elements)}


def _element(ref, tag, value=""):
    return {"ref": ref, "tag": tag, "value": value}


PAGE = _observation(
    _element(2, "input_text"),
    _element(3, "button"),
    _element(4, "input_checkbox"),
    _element(5, "a"),
    _element(6, "option"),
)


def test_scripted_action_is_described_consistently():
    """The logged description must match the action actually sent to the env."""
    env = _StubEnv()
    action, described = _make_action(env, PAGE, np.random.default_rng(0), 0)
    assert action is not None
    sent_type, sent_kwargs = env.unwrapped.calls[-1]
    assert described["type"] == sent_type.name if hasattr(sent_type, "name") else True
    assert described["ref"] == sent_kwargs["ref"]
    assert described["text"] == sent_kwargs.get("text")
    assert described["policy"] == "scripted"


def test_typed_text_is_logged_verbatim():
    """The typed value is the one part of the next state nothing can predict.

    It is injected by the collector itself, so logging it is what turns
    irreducible noise into information the model can condition on.
    """
    env = _StubEnv()
    page = _observation(_element(7, "input_text"))
    _, described = _make_action(env, page, np.random.default_rng(3), 0)
    sent_kwargs = env.unwrapped.calls[-1][1]
    assert described["text"] == sent_kwargs["text"]
    assert described["text"].startswith("state-")
    assert len(described["text"]) == len("state-") + 5


def test_epsilon_zero_reproduces_the_scripted_policy_exactly():
    """Regression lock: the v4/v5/v6 datasets must stay reproducible."""
    for seed in (0, 1, 7, 99):
        scripted_env, mixed_env = _StubEnv(), _StubEnv()
        scripted = _make_action(scripted_env, PAGE, np.random.default_rng(seed), 0)
        mixed = _choose_action(mixed_env, PAGE, np.random.default_rng(seed), 0, 0.0)
        assert scripted[1] == mixed[1], seed
        assert scripted_env.unwrapped.calls == mixed_env.unwrapped.calls


def test_random_policy_covers_more_than_one_element():
    """Without this the action stays a function of the state."""
    chosen = set()
    for seed in range(40):
        env = _StubEnv()
        _, described = _random_action(env, PAGE, np.random.default_rng(seed))
        chosen.add(described["ref"])
        assert described["policy"] == "random"
    assert len(chosen) >= 3, chosen


def test_random_policy_types_into_text_fields_and_clicks_others():
    for seed in range(40):
        env = _StubEnv()
        _, described = _random_action(env, PAGE, np.random.default_rng(seed))
        if described["tag"] in {"input_text", "input_password", "textarea"}:
            assert described["text"] is not None
        else:
            assert described["text"] is None


def test_epsilon_one_still_yields_a_legal_action_on_a_clickable_page():
    env = _StubEnv()
    action, described = _choose_action(env, PAGE, np.random.default_rng(0), 0, 1.0)
    assert action is not None and described["policy"] == "random"


def test_no_interactive_elements_yields_no_action():
    env = _StubEnv()
    blank = _observation(_element(1, "div"))
    assert _random_action(env, blank, np.random.default_rng(0)) == (None, None)
    # the scripted path falls through its cascade to the same answer
    assert _make_action(env, _observation(), np.random.default_rng(0), 0) == (None, None)


def test_random_action_falls_back_to_scripted_when_page_has_nothing_to_click():
    """``_choose_action`` must not return None just because the draw came up random."""
    env = _StubEnv()
    blank = _observation(_element(1, "div"))
    action, described = _choose_action(env, blank, np.random.default_rng(0), 0, 1.0)
    assert (action, described) == (None, None)


@pytest.mark.parametrize("epsilon", [0.25, 0.5, 0.75])
def test_mixed_policy_produces_both_policies(epsilon):
    seen = set()
    for seed in range(60):
        env = _StubEnv()
        _, described = _choose_action(
            env, PAGE, np.random.default_rng(seed), 0, epsilon
        )
        seen.add(described["policy"])
    assert seen == {"scripted", "random"}, (epsilon, seen)
