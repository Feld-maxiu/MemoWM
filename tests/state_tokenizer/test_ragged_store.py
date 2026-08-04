from __future__ import annotations

import torch

from experiments.state_tokenizer.ragged_store import normalized_modality_positions, pad_token_batch


def test_normalized_positions_are_modality_local_and_scale_invariant():
    short = normalized_modality_positions(torch.tensor([0, 0, 1, 1, 1, 2]))
    assert torch.allclose(short[:2], torch.tensor([0.25, 0.75]))
    assert torch.allclose(short[2:5], torch.tensor([1 / 6, 0.5, 5 / 6]))
    assert torch.allclose(short[5:], torch.tensor([0.5]))


def test_pad_token_batch_marks_padding_invalid():
    first = (torch.ones((2, 4), dtype=torch.bfloat16), torch.tensor([0, 2]))
    second = (torch.ones((3, 4), dtype=torch.bfloat16), torch.tensor([0, 1, 2]))
    tokens, modalities, positions, valid = pad_token_batch([first, second])
    assert tokens.shape == (2, 3, 4)
    assert valid.tolist() == [[True, True, False], [True, True, True]]
    assert positions[0, 2] == 0
    assert modalities[0, 2] == 0
