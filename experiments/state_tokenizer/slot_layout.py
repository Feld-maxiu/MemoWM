"""The 64-slot layout, with no third-party dependencies.

This lives apart from ``key_pooling`` because the layout is needed on both sides
of a dependency split that the pipeline genuinely has: pooling is torch-only and
runs in the extraction environment, while the bottlenecks are jax-only and run in
another. Importing the constant from ``key_pooling`` drags torch into the jax
environment and fails at import time.

Keeping it here also removes the failure that motivated the split. The layout was
previously repeated as a literal in four places -- ``fit_normalization``,
``rebuild_static_key64``'s manifest, ``a0`` and ``a1`` -- and when the prompt band
was recycled into detail, every one of them kept reporting and slicing the old
(32, 12, 16, 4). Nothing raised: the normalizer simply computed each group's
statistics over the wrong slots.
"""
from __future__ import annotations

IMAGE_SLOTS = 32

# 16, not 12: the four prompt slots were recycled into detail. The detail budget
# is the binding constraint on whether an instruction's target labels survive --
# click-checkboxes names up to five, each 1-5 tokens, and twelve slots left ~19%
# of them out even with instruction-priority ordering.
DETAIL_SLOTS = 16

CONTEXT_SLOTS = 16

# The prompt band held the pooled hidden states of one fixed, task-independent
# observation prompt -- byte-identical across every state. Cross-state cosine
# similarity ran at median 0.214 against detail's 0.002, two orders of magnitude
# apart, which is what "carries almost no per-state information" looks like.
PROMPT_SLOTS = 0

KEY64_LAYOUT = (IMAGE_SLOTS, DETAIL_SLOTS, CONTEXT_SLOTS, PROMPT_SLOTS)
GROUP_NAMES = ("image", "detail", "context", "prompt")

assert sum(KEY64_LAYOUT) == 64, KEY64_LAYOUT
assert len(KEY64_LAYOUT) == len(GROUP_NAMES)
