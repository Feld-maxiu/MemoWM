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

from experiments.state_tokenizer.reader_losses import answer_ce_and_distill_kl

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
