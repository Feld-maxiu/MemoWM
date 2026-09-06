"""Run the frozen trunk to layer 16 and hand back the *whole* sequence.

``FrozenV9InstructTokenizer.encode`` reduces the captured states to 64 pooled
slots before anything else sees them, and it does so under
``torch.inference_mode``. Both are fine for the frozen pipeline and both block a
learnable resampler:

* the pooled 64x4096 has already thrown away the variable-length structure a
  Q-Former reads;
* inference-mode tensors are permanently poisoned for autograd. They are not
  merely detached -- feeding one into any op that records grad raises
  ``RuntimeError: Inference tensors cannot be saved for backward``, and that
  applies to everything derived from them.

So this is a parallel path, deliberately not a flag on ``encode``: that
dataclass is all-numpy and every eval consumer depends on it.

The trunk stays frozen (``requires_grad_(False)`` in ``_load_model``) and runs
under ``no_grad``; gradients are only ever needed *downstream* of ``H_t``, in
the resampler and the connector.
"""
from __future__ import annotations

import dataclasses

import torch
from PIL import Image

from .extract_qwen import (
    _StopAtLayer,
    modality_indices,
    prepare_inputs,
)
from .fixed_prompt import OBSERVATION_PROMPT

# 3 is not "unknown" -- it is the chat-template wrapper plus the four DOM and
# instruction marker tokens, which modality_indices deliberately excludes from
# all three spans. They are constant across observations, so they carry nothing,
# but dropping them would mean the resampler sees a sequence with holes in it.
IMAGE, DOM, INSTRUCTION, WRAPPER = 0, 1, 2, 3
NUM_MODALITIES = 4


@dataclasses.dataclass(frozen=True)
class TrunkStates:
    """One observation's layer-16 sequence, ready for the resampler."""

    hidden: torch.Tensor        # (tokens, 4096) bfloat16
    modality_ids: torch.Tensor  # (tokens,) int64
    positions: torch.Tensor     # (tokens,) float32 in [0, 1]
    # Truncation audit, populated when the caller opted into ``allow_truncate``.
    truncated: bool = False
    kept_dom_chars: int | None = None
    full_dom_chars: int | None = None

    def __len__(self) -> int:
        return int(self.hidden.shape[0])


def modality_labels(length: int, indices, device) -> torch.Tensor:
    """Per-token modality over the full sequence, wrapper tokens included."""
    labels = torch.full((length,), WRAPPER, dtype=torch.long, device=device)
    for value, index in zip((IMAGE, DOM, INSTRUCTION), indices):
        labels[index.to(device)] = value
    return labels


def trunk_states(
    processor,
    model,
    image: Image.Image,
    dom: str,
    *,
    layer: int = 16,
    max_length: int = 8192,
    instruction: str = OBSERVATION_PROMPT,
    device: torch.device | None = None,
    allow_truncate: bool = False,
) -> TrunkStates:
    """``(image, dom)`` -> the layer-16 sequence, unpooled.

    Mirrors ``frozen_v8_runtime.encode`` up to the pooling call, including the
    ``"instruct"`` prompt mode and the refusal to proceed on truncation -- a
    silently shortened DOM would change ``H_t`` without changing anything the
    caller can see.  Callers that would rather keep going on over-long DOMs
    (AMA episodes reach 43k characters) pass ``allow_truncate=True`` and get
    the truncation facts back on the returned ``TrunkStates`` for audit.
    """
    inputs, truncated, kept_dom_chars, _original = prepare_inputs(
        processor, image, dom, instruction, max_length, "instruct"
    )
    if truncated and not allow_truncate:
        raise ValueError(
            f"observation exceeded max length: kept {kept_dom_chars}/{len(dom)} DOM chars"
        )
    kept = int(kept_dom_chars) if truncated else len(dom)
    indices = modality_indices(processor, model, inputs["input_ids"])
    target = device if device is not None else model.device
    device_inputs = inputs.to(target)

    layers = model.model.language_model.layers
    if not 1 <= layer <= len(layers):
        raise ValueError(f"layer must be in [1, {len(layers)}], got {layer}")

    capture: dict[str, torch.Tensor] = {}

    def hook(_module, _args, value):
        capture["hidden"] = value[0] if isinstance(value, tuple) else value
        raise _StopAtLayer

    handle = layers[layer - 1].register_forward_hook(hook)
    try:
        # no_grad, not inference_mode: see the module docstring.
        with torch.no_grad():
            try:
                model.model(**device_inputs, use_cache=False, output_hidden_states=False)
            except _StopAtLayer:
                pass
    finally:
        handle.remove()

    hidden = capture.get("hidden")
    if hidden is None:
        raise RuntimeError(f"the layer-{layer} hook did not fire")
    hidden = hidden[0]
    length = int(hidden.shape[0])
    positions = torch.arange(length, dtype=torch.float32, device=target)
    return TrunkStates(
        hidden=hidden,
        modality_ids=modality_labels(length, indices, target),
        positions=positions / max(length - 1, 1),
        truncated=bool(truncated),
        kept_dom_chars=kept,
        full_dom_chars=len(dom),
    )


def collate(batch: list[TrunkStates], pad_to: int | None = None):
    """Right-pad a batch to a common length and build the key mask.

    Returns ``(hidden, modality_ids, positions, key_mask)``. Padded positions are
    zero in every tensor *and* False in the mask; the resampler zeroes them again
    so that "padding cannot reach the output" is a property of the module rather
    than of this function.
    """
    if not batch:
        raise ValueError("cannot collate an empty batch")
    width = batch[0].hidden.shape[-1]
    device = batch[0].hidden.device
    longest = max(len(item) for item in batch)
    total = max(longest, pad_to or 0)

    hidden = torch.zeros((len(batch), total, width), dtype=batch[0].hidden.dtype, device=device)
    modality = torch.full((len(batch), total), WRAPPER, dtype=torch.long, device=device)
    positions = torch.zeros((len(batch), total), dtype=torch.float32, device=device)
    mask = torch.zeros((len(batch), total), dtype=torch.bool, device=device)
    for row, item in enumerate(batch):
        end = len(item)
        hidden[row, :end] = item.hidden
        modality[row, :end] = item.modality_ids
        positions[row, :end] = item.positions
        mask[row, :end] = True
    return hidden, modality, positions, mask
