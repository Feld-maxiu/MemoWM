from __future__ import annotations

import torch

from experiments.state_tokenizer.ragged_store import (
    FixedRepresentationStore,
    normalized_modality_positions,
    pad_token_batch,
)
from experiments.state_tokenizer.slot_layout import (
    CONTEXT_SLOTS,
    DETAIL_SLOTS,
    IMAGE_SLOTS,
    PROMPT_SLOTS,
)


def test_key64_modality_layout_tracks_frozen_slot_layout():
    expected = (IMAGE_SLOTS, DETAIL_SLOTS + CONTEXT_SLOTS, PROMPT_SLOTS)
    for representation in ("key64", "key64_pca", "key64_static", "key64_static_pca"):
        assert FixedRepresentationStore.SPECS[representation][3] == expected


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
