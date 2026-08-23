"""Trunk-state extraction: the autograd invariant and batch collation.

The one thing that makes a learnable resampler possible at all is that ``H_t``
is captured under ``no_grad`` rather than ``inference_mode``. That distinction
is easy to erase in a refactor and produces a failure far away from its cause --
``RuntimeError: Inference tensors cannot be saved for backward``, raised inside
the resampler, on a tensor the resampler did not create.
``test_inference_mode_poisons_the_graph_but_no_grad_does_not`` pins the reason
so the next person does not have to rediscover it.

Model-free; runs under ``tests/run_tests.py``.
"""
from __future__ import annotations

import torch
from torch import nn

from experiments.state_tokenizer.trunk_states import (
    DOM,
    IMAGE,
    INSTRUCTION,
    WRAPPER,
    TrunkStates,
    collate,
    modality_labels,
)


def _states(tokens: int, width: int = 8, seed: int = 0) -> TrunkStates:
    generator = torch.Generator().manual_seed(seed)
    return TrunkStates(
        hidden=torch.randn(tokens, width, generator=generator),
        modality_ids=torch.full((tokens,), DOM, dtype=torch.long),
        positions=torch.linspace(0, 1, tokens),
    )


def test_inference_mode_poisons_the_graph_but_no_grad_does_not():
    """Why ``trunk_states`` cannot reuse ``frozen_v8_runtime.encode``'s context.

    Both contexts stop the trunk's own parameters from accumulating gradient.
    Only ``inference_mode`` also makes the *output* unusable as an input to a
    trainable module -- which is exactly how the resampler consumes it.
    """
    frozen = nn.Linear(8, 8)
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    trainable = nn.Linear(8, 4)
    source = torch.randn(3, 8)

    with torch.inference_mode():
        poisoned = frozen(source)
    try:
        trainable(poisoned).sum().backward()
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "inference_mode output was safe to backprop through -- this test no "
            "longer pins anything and trunk_states' context choice is unjustified"
        )

    with torch.no_grad():
        clean = frozen(source)
    trainable(clean).sum().backward()
    assert trainable.weight.grad is not None
    assert trainable.weight.grad.abs().sum() > 0


def test_modality_labels_cover_every_position():
    """No holes: wrapper tokens get a label rather than being dropped."""
    length = 20
    indices = (
        torch.tensor([2, 3, 4, 5]),       # image
        torch.arange(8, 14),              # dom
        torch.arange(15, 18),             # instruction
    )
    labels = modality_labels(length, indices, torch.device("cpu"))
    assert labels.shape == (length,)
    assert (labels[2:6] == IMAGE).all()
    assert (labels[8:14] == DOM).all()
    assert (labels[15:18] == INSTRUCTION).all()
    # 0,1,6,7,14,18,19 belong to no span and must still be labelled.
    for position in (0, 1, 6, 7, 14, 18, 19):
        assert labels[position] == WRAPPER, position


def test_collate_pads_right_and_marks_the_mask():
    batch = [_states(12, seed=1), _states(7, seed=2), _states(20, seed=3)]
    hidden, modality, positions, mask = collate(batch)
    assert hidden.shape == (3, 20, 8)
    assert modality.shape == positions.shape == mask.shape == (3, 20)
    assert mask.sum(dim=1).tolist() == [12, 7, 20]
    for row, item in enumerate(batch):
        end = len(item)
        assert torch.equal(hidden[row, :end], item.hidden)
        assert torch.equal(positions[row, :end], item.positions)
        assert bool(mask[row, :end].all()) and not bool(mask[row, end:].any())
        assert torch.count_nonzero(hidden[row, end:]) == 0


def test_collate_honours_an_explicit_width():
    hidden, _, _, mask = collate([_states(5)], pad_to=32)
    assert hidden.shape[1] == 32
    assert mask.sum().item() == 5


def test_collate_rejects_an_empty_batch():
    try:
        collate([])
    except ValueError:
        return
    raise AssertionError("an empty batch produced tensors")
