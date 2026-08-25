"""The hybrid reader's sequence assembly.

``Qwen35LatentReader`` concatenates retrieved memory in *retrieval-rank order*
and that ordering is the only thing separating one state from another:
``rank_embedding`` is ``(64, 4096)`` and broadcasts the same values over every
state, so once the blocks are concatenated their RoPE positions are the sole
rank signal. Reorder the segments and nothing recovers which row was retrieved
first.

The other invariant here is that text rows survive. Keeping only rows that
carry latents empties the context for 84% of WorldMemArena web questions, after
which the reader returns its fallback string and the judge scores an omission.

Torch-only; no model weights are loaded -- the assembly is exercised through a
stub whose embedding table and layer stack are the smallest things that satisfy
the code path.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from residualmem.latent.instruct_bridge import (
    InputSoftTokenConnector,
    MemorySegment,
    Qwen35LatentReader,
    _as_segment,
)

DIM = 4096


class _Tokenizer:
    """Deterministic stand-in: one id per character, offset off the pad id."""

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = [ord(character) % 1000 + 1 for character in text]
        if add_special_tokens:
            ids = [1] + ids
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(value) % 128) for value in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        """A minimal stand-in that keeps the one property the reader relies on.

        The reader splits the rendered string on a sentinel to place the latent
        block where ``{context}`` sits, so a template that mangled or duplicated
        the sentinel would break it silently in production. Wrapping each turn
        in explicit role markers reproduces that structure without pulling a
        real tokenizer into the test.
        """
        assert not tokenize
        body = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return body + ("<|assistant|>" if add_generation_prompt else "")


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer()


class _Model(nn.Module):
    """Records what generate() was handed, so the assembly can be asserted."""

    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(1001, DIM)
        self.seen = {}

    def get_input_embeddings(self):
        return self.embeddings

    def generate(self, *, inputs_embeds, attention_mask, max_new_tokens, do_sample):
        self.seen["inputs_embeds"] = inputs_embeds
        self.seen["attention_mask"] = attention_mask
        return torch.tensor([[104, 105]])


def _reader():
    model = _Model()
    connector = InputSoftTokenConnector()
    return Qwen35LatentReader(model, _Processor(), connector, mode="input"), model


def _state(seed: int, valid_slots: int = 64):
    rng = np.random.default_rng(seed)
    xbar = rng.normal(size=(64, 512)).astype(np.float32)
    valid = np.zeros(64, bool)
    valid[:valid_slots] = True
    return xbar, valid


def test_empty_retrieval_still_falls_back():
    reader, _ = _reader()
    assert reader.answer("q", []) == "Not mentioned in memory."


def test_text_only_retrieval_reaches_the_model():
    """The regression that motivated the hybrid path: text rows must not be
    silently dropped into the fallback."""
    reader, model = _reader()
    answer = reader.answer("q", [MemorySegment(text="a retrieved round")])
    assert answer != "Not mentioned in memory."
    assert "inputs_embeds" in model.seen


def _official_prompt_halves(question: str) -> tuple[str, str]:
    """The rendered official prompt, split where the memory block goes."""
    from residualmem.latent.instruct_bridge import (
        OFFICIAL_ANSWER_SYSTEM_PROMPT,
        OFFICIAL_ANSWER_USER_TEMPLATE,
    )

    sentinel = "\x00RETRIEVED_MEMORY\x00"
    rendered = _Tokenizer().apply_chat_template(
        [
            {"role": "system", "content": OFFICIAL_ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": OFFICIAL_ANSWER_USER_TEMPLATE.format(
                context=sentinel, question=question)},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    head, tail = rendered.split(sentinel)
    return head, tail          # one stub id per character, so len() is the token count


def test_latent_costs_64_positions_and_text_costs_its_tokens():
    reader, model = _reader()
    text = "hello"
    question = "q"
    reader.answer(question, [MemorySegment(latent=_state(0)), MemorySegment(text=text)])

    head, tail = _official_prompt_halves(question)
    # The memory block sits inside the official template, not in front of it:
    # 64 positions for the state regardless of validity, the text's own tokens
    # with no special tokens added per segment, wrapped by the rendered prompt.
    expected = len(head) + 64 + len(text) + len(tail)
    assert model.seen["inputs_embeds"].shape[1] == expected
    assert model.seen["attention_mask"].shape[1] == expected


def test_segments_keep_retrieval_rank_order():
    """Latent-then-text and text-then-latent must not produce the same tensor."""
    reader, model = _reader()
    latent = MemorySegment(latent=_state(7))
    text = MemorySegment(text="zzzz")

    reader.answer("q", [latent, text])
    first = model.seen["inputs_embeds"].clone()
    reader.answer("q", [text, latent])
    second = model.seen["inputs_embeds"].clone()

    assert first.shape == second.shape
    assert not torch.equal(first, second)


def test_masked_slots_do_not_attend():
    reader, model = _reader()
    reader.answer("q", [MemorySegment(latent=_state(1, valid_slots=40))])
    mask = model.seen["attention_mask"][0]
    offset = len(_official_prompt_halves("q")[0])
    # 24 of the 64 latent positions are invalid and must be masked out. The
    # offset is load-bearing: the prompt prefix is all-ones, so indexing from
    # zero would find no zeros and the test would pass on a broken mask.
    assert int((mask[:offset] == 0).sum()) == 0
    assert int((mask[offset:offset + 64] == 0).sum()) == 24


def test_segment_carries_exactly_one_payload():
    for bad in ({}, {"latent": _state(0), "text": "x"}):
        try:
            MemorySegment(**bad)
        except ValueError:
            continue
        raise AssertionError(f"MemorySegment accepted {bad!r}")


def test_bare_tuples_and_strings_are_still_accepted():
    """The pre-hybrid calling shape stays valid so older entry points keep working."""
    assert _as_segment(_state(0)).latent is not None
    assert _as_segment("some text").text == "some text"
    assert _as_segment(MemorySegment(text="x")).text == "x"


def test_the_official_prompt_actually_reaches_the_model():
    """The bug this guards: a home-made prompt instead of the benchmark's.

    Answering ran a twenty-word prompt through a bare tokenizer for four
    evaluation arms. Two of the official rules it dropped map straight onto how
    that arm failed -- rule 7's "*Only if* ... NO information relevant" became
    "If absent", and rule 5 (answer image questions with an image_id) was gone
    entirely -- and the arm posted the highest omission and lowest
    hallucination of any. Nothing in the code said which prompt was in use, so
    this asserts on the assembled ids rather than on the source.
    """
    from residualmem.latent.instruct_bridge import (
        OFFICIAL_ANSWER_SYSTEM_PROMPT,
        OFFICIAL_ANSWER_USER_TEMPLATE,
    )

    reader, model = _reader()
    reader.answer("what colour", [MemorySegment(text="a row")])
    ids = model.seen["inputs_embeds"]
    tokenizer = _Tokenizer()

    def contains(needle: str) -> bool:
        want = tokenizer(needle, add_special_tokens=False)["input_ids"][0]
        # The reader hands over inputs_embeds only, so compare in embedding space.
        wanted = reader.model.get_input_embeddings()(want.unsqueeze(0))[0]
        for start in range(ids.shape[1] - len(want) + 1):
            if torch.allclose(ids[0, start:start + len(want)], wanted):
                return True
        return False

    assert contains("Only if the retrieved memories truly contain NO information")
    assert contains("return the relevant image_id(s) found in the memory captions")
    assert contains("Question: what colour")
    # The template must be the one the drift check compares against.
    assert "{context}" in OFFICIAL_ANSWER_USER_TEMPLATE
    assert OFFICIAL_ANSWER_SYSTEM_PROMPT.count("# Instructions:") == 1


def test_json_answers_are_unwrapped_and_junk_survives():
    from residualmem.latent.instruct_bridge import _parse_official_answer

    assert _parse_official_answer('{"answer": "blue"}') == "blue"
    assert _parse_official_answer('```json\n{"answer": "blue"}\n```') == "blue"
    assert _parse_official_answer('sure: {"answer": "Not mentioned in memory."}') == (
        "Not mentioned in memory."
    )
    # A malformed generation is scored as what the model said, not dropped: an
    # empty string here would be graded an omission the model never committed.
    assert _parse_official_answer("blue, probably") == "blue, probably"
    assert _parse_official_answer('{"nope": 1}') == '{"nope": 1}'
