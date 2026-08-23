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


def test_latent_costs_64_positions_and_text_costs_its_tokens():
    reader, model = _reader()
    text = "hello"
    question = "q"
    reader.answer(question, [MemorySegment(latent=_state(0)), MemorySegment(text=text)])

    tokenizer = _Tokenizer()
    prompt = (
        "Answer the question from the retrieved memory. Keep the answer concise. "
        "If absent, say exactly 'Not mentioned in memory.'.\nQuestion: " + question
    )
    prompt_len = int(tokenizer(prompt, add_special_tokens=True)["input_ids"].shape[1])
    # 64 positions for the state regardless of validity, the text's own tokens
    # with no special tokens added per segment, then the prompt.
    expected = 64 + len(text) + prompt_len
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
    # 24 of the 64 latent positions are invalid and must be masked out.
    assert int((mask[:64] == 0).sum()) == 24


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
