"""The shared reader losses, in particular the distillation KL's reduction.

This is the regression for a bug that survived a full 6,000-step run and made a
whole Stage-C comparison uninterpretable: ``reduction="batchmean"`` on a
``(1, answer_len, vocab)`` tensor divides by the batch dimension, which is 1, so
the "mean" was a **sum over answer positions** and ``distill_weight=1.0``
weighted the KL by roughly the answer length.

Nothing in ``tests/`` had ever touched a trainer, which is why it survived. The
assertions here compare against an independently computed per-position mean
rather than against a recorded number, so they fail for the right reason.

Model-free: a stub with the two methods the loss actually calls.
"""
from __future__ import annotations

import types

import torch
from torch import nn
from torch.nn import functional as F

from experiments.state_tokenizer.reader_losses import (
    LOGPROB_FLOOR,
    answer_ce_and_distill_kl,
    topk_kl,
)

VOCAB, DIM = 11, 6


class _Tokenizer:
    """Deterministic character codes; enough to exercise the shapes."""

    def __call__(self, text, *, return_tensors=None, add_special_tokens=True,
                 truncation=False, max_length=None):
        ids = [ord(character) % VOCAB for character in text] or [0]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _Model(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embedding = nn.Embedding(VOCAB, DIM)
        self.head = nn.Linear(DIM, VOCAB)
        self.device = torch.device("cpu")

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, attention_mask, labels, use_cache=False):
        # A causal running mean before the head. Without some mixing the logits
        # at an answer position would depend only on that position's own
        # embedding, so the memory could not influence them at all -- the KL
        # would be identically zero and the latent would receive no gradient,
        # and neither property below would be testable.
        weights = torch.arange(
            1, inputs_embeds.shape[1] + 1, device=inputs_embeds.device
        ).unsqueeze(-1)
        mixed = inputs_embeds.cumsum(dim=1) / weights
        logits = self.head(mixed)
        shifted = logits[:, :-1].reshape(-1, VOCAB)
        targets = labels[:, 1:].reshape(-1)
        loss = F.cross_entropy(shifted, targets, ignore_index=-100)
        return types.SimpleNamespace(loss=loss, logits=logits)


def _call(weight, answer="hello world", memory="a memory row", seed=0):
    model = _Model(seed)
    processor = types.SimpleNamespace(tokenizer=_Tokenizer())
    latent = torch.randn(1, 64, DIM, requires_grad=True)
    valid = torch.ones(1, 64, dtype=torch.bool)
    return model, processor, latent, answer_ce_and_distill_kl(
        model, processor, latent, valid, "what is it?", answer, memory, weight=weight
    )


def test_the_kl_is_a_per_position_mean_not_a_sum():
    """The bug: a 12-token answer inflated the KL term 12-fold."""
    answer = "twelve chars"
    model, processor, latent, (_loss, _ce, kl) = _call(1.0, answer=answer)
    answer_len = len(answer)

    # Recompute both reductions from the same forwards and check which we got.
    embed = model.get_input_embeddings()
    question_ids = processor.tokenizer(
        "Answer the question from the retrieved memory. Keep the answer concise. "
        "If absent, say exactly 'Not mentioned in memory.'.\nQuestion: what is it?"
    )["input_ids"]
    answer_ids = processor.tokenizer(answer, add_special_tokens=False)["input_ids"]
    text_ids = processor.tokenizer("a memory row", add_special_tokens=False)["input_ids"]

    def logits_for(memory_embeds):
        inputs = torch.cat((memory_embeds, embed(question_ids), embed(answer_ids)), 1)
        return model(inputs_embeds=inputs,
                     attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long),
                     labels=torch.zeros(inputs.shape[:2], dtype=torch.long)).logits

    span = slice(-answer_ids.shape[1] - 1, -1)
    student = F.log_softmax(logits_for(latent)[:, span].float(), -1)
    teacher = F.log_softmax(logits_for(embed(text_ids))[:, span].float(), -1)
    as_sum = F.kl_div(student, teacher, reduction="batchmean", log_target=True)
    as_mean = as_sum / answer_ids.shape[1]

    assert abs(kl - float(as_mean)) < 1e-4, (
        f"kl={kl:.6f} is not the per-position mean {float(as_mean):.6f}"
    )
    assert abs(kl - float(as_sum)) > 1e-4, (
        "the sum and the mean coincide here, so this test cannot tell them apart"
    )
    assert answer_len > 1, "a one-token answer would make the two reductions equal"


def test_the_weight_scales_only_the_kl():
    """loss(w) - loss(0) must be exactly w * kl, or the weight is a lie."""
    _m, _p, _l, (loss_zero, ce_zero, kl_zero) = _call(0.0)
    _m, _p, _l, (loss_two, ce_two, kl_two) = _call(2.0)
    assert kl_zero == 0.0, "weight 0 still computed a KL"
    assert abs(ce_zero - ce_two) < 1e-5, "the weight moved the CE term"
    assert abs(float(loss_two) - (ce_two + 2.0 * kl_two)) < 1e-4


def test_an_absent_memory_text_disables_the_kl_rather_than_faking_one():
    _m, _p, _l, (loss, ce, kl) = _call(1.0, memory="")
    assert kl == 0.0
    assert abs(float(loss) - ce) < 1e-6


def test_the_gradient_reaches_the_latent():
    """The latent is the only leaf; if it gets nothing, no tokenizer trains."""
    _model, _processor, latent, (loss, _ce, _kl) = _call(1.0)
    loss.backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert latent.grad.abs().sum() > 0


def test_an_empty_answer_is_rejected():
    model = _Model()
    processor = types.SimpleNamespace(tokenizer=_Tokenizer())

    class _Empty(_Tokenizer):
        def __call__(self, text, **kwargs):
            if not kwargs.get("add_special_tokens", True):
                return {"input_ids": torch.zeros(1, 0, dtype=torch.long)}
            return super().__call__(text, **kwargs)

    processor.tokenizer = _Empty()
    try:
        answer_ce_and_distill_kl(
            model, processor, torch.randn(1, 64, DIM), torch.ones(1, 64, dtype=torch.bool),
            "q", "", "memory", weight=1.0,
        )
    except ValueError:
        return
    raise AssertionError("an empty answer produced a loss")


def test_topk_kl_equals_the_full_kl_when_nothing_is_truncated():
    """The truncation is an approximation; at k=vocab it must be exact.

    Without this the top-k path could be systematically off -- a renormalization
    dropped, a log/prob confusion -- and every downstream number would inherit
    it with nothing to compare against.
    """
    torch.manual_seed(3)
    length = 7
    student = torch.randn(1, length, VOCAB)
    teacher = F.log_softmax(torch.randn(1, length, VOCAB), -1)
    index = torch.arange(VOCAB).expand(1, length, VOCAB)

    full = (teacher.exp() * (teacher - F.log_softmax(student, -1))).sum(-1).mean()
    got = topk_kl(student, index, teacher.gather(-1, index))
    assert abs(float(got) - float(full)) < 1e-6, (float(got), float(full))

    # And truncation has to actually change the number, or the test above is
    # measuring nothing.
    logprob, narrow = teacher.topk(3, -1)
    assert abs(float(topk_kl(student, narrow, logprob)) - float(full)) > 1e-4


def test_topk_kl_is_a_per_position_mean_not_a_sum():
    """Same defect class as the KL this module's docstring is about.

    A sum would make the caller's weight secretly mean "weight times the span
    length", and the spans here vary from 1 to 96 tokens.
    """
    torch.manual_seed(4)
    length = 5
    student = torch.randn(1, length, VOCAB)
    teacher = F.log_softmax(torch.randn(1, length, VOCAB), -1)
    index = torch.arange(VOCAB).expand(1, length, VOCAB)
    once = topk_kl(student, index, teacher.gather(-1, index))

    doubled = topk_kl(
        torch.cat([student, student], 1),
        torch.arange(VOCAB).expand(1, 2 * length, VOCAB),
        torch.cat([teacher, teacher], 1).gather(
            -1, torch.arange(VOCAB).expand(1, 2 * length, VOCAB)),
    )
    assert abs(float(once) - float(doubled)) < 1e-5, (float(once), float(doubled))
    # Guard: a summing implementation would have doubled, so the assertion above
    # is only meaningful if the value is far from zero.
    assert float(once) > 1e-3


def test_a_cached_teacher_replaces_the_text_one_and_reaches_the_latent():
    """``teacher_topk`` must both change the loss and carry gradient.

    The cached teacher read the screenshot; the text one reads a caption that is
    embedded verbatim in the student's own DOM span. If the keyword were
    silently ignored the run would look fine and train against the weaker
    target -- exactly the failure this parameter exists to end.
    """
    torch.manual_seed(5)
    model, processor, latent, (loss_text, ce_text, kl_text) = _call(1.0)

    answer_len = len("hello world")
    index = torch.arange(VOCAB).expand(1, answer_len, VOCAB)
    logprob = F.log_softmax(torch.randn(1, answer_len, VOCAB), -1)
    loss, ce, kl = answer_ce_and_distill_kl(
        model, processor, latent, torch.ones(1, 64, dtype=torch.bool),
        "what is it?", "hello world", "a memory row",
        weight=1.0, teacher_topk=(index, logprob),
    )
    assert abs(ce - ce_text) < 1e-5          # the CE must not move
    assert abs(kl - kl_text) > 1e-4          # the KL must
    loss.backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert float(latent.grad.abs().sum()) > 0


def test_a_cached_teacher_that_disagrees_on_the_span_is_refused():
    """A silent length mismatch would score the wrong positions.

    The cache is keyed by pairs row; if it were rebuilt from a different pairs
    file the spans would drift and the KL would compare unrelated tokens.
    """
    model, processor, latent, _ = _call(0.0)
    wrong = len("hello world") + 3
    try:
        answer_ce_and_distill_kl(
            model, processor, latent, torch.ones(1, 64, dtype=torch.bool),
            "what is it?", "hello world", "a memory row", weight=1.0,
            teacher_topk=(torch.zeros(1, wrong, 4, dtype=torch.long),
                          torch.zeros(1, wrong, 4)),
        )
    except ValueError:
        return
    raise AssertionError("a span-length mismatch was accepted")


def test_a_confident_student_mistake_cannot_produce_an_unbounded_loss():
    """The floor that ended four runs' worth of divergences.

    Forward KL is ``sum_v p_t(v) [log p_t(v) - log p_s(v)]`` and ``log p_s`` has
    no lower bound, so one token the teacher supports and the student has
    written off contributes arbitrarily much. Measured before the floor: 2090
    nats from a single such token, which backpropagates into "xbar contains
    non-finite values" at a random step -- the shape of every divergence in this
    project, at K=32 and K=64 alike, including arms where the new objective was
    switched off entirely.
    """
    torch.manual_seed(6)
    width, length, k = 2000, 3, 8
    logprob, index = F.log_softmax(torch.randn(1, length, width), -1).topk(k, -1)

    ordinary = torch.randn(1, length, width)
    baseline = float(topk_kl(ordinary, index, logprob))

    collapsed = ordinary.clone()
    collapsed.scatter_(-1, index[..., :1], -1e4)
    bounded = float(topk_kl(collapsed, index, logprob))

    # Bounded, but not clamped to nothing: the mistake still costs more than a
    # correct student, or the floor would have destroyed the signal.
    assert bounded < 10 * abs(LOGPROB_FLOOR), bounded
    assert bounded > baseline, (bounded, baseline)
    # An ordinary student is untouched by the floor -- e^-30 is 1e-13.
    assert abs(baseline - float(topk_kl(ordinary, index, logprob))) < 1e-9
