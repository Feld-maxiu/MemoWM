from __future__ import annotations

import torch

from experiments.state_tokenizer.slot_reader import FourierPositionEncoding, SlotAwareReader


def test_fourier_positions_have_fixed_width_and_aligned_scale():
    encoding = FourierPositionEncoding(256)
    first = encoding(torch.tensor([[0.25, 0.75]]))
    second = encoding(torch.tensor([[0.25, 0.75]]))
    assert first.shape == (1, 2, 256)
    assert torch.equal(first, second)


def test_padding_does_not_change_reader_output():
    torch.manual_seed(0)
    model = SlotAwareReader(
        8, dynamic=2, static=1, dom_words=2, overlap_words=1, tasks=2, dropout=0.0
    ).eval()
    tokens = torch.randn(1, 2, 8)
    modalities = torch.tensor([[0, 1]])
    positions = torch.tensor([[0.5, 0.5]])
    valid = torch.tensor([[True, True]])
    padded_tokens = torch.cat((tokens, torch.randn(1, 2, 8)), dim=1)
    padded_modalities = torch.tensor([[0, 1, 0, 0]])
    padded_positions = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    padded_valid = torch.tensor([[True, True, False, False]])
    with torch.inference_mode():
        first = model(tokens, modalities, positions, valid)["dynamic"]
        second = model(padded_tokens, padded_modalities, padded_positions, padded_valid)["dynamic"]
    assert torch.allclose(first, second, atol=1e-6)
