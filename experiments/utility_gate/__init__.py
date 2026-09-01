"""Rate-utility send gating over the 1024 discrete code positions.

The world model supplies a categorical distribution per position; the gate
decides which positions are worth their bits, judged by how much the reader's
answer distribution moves when a position is replaced by the model's estimate.

Only dependency-free names are re-exported here. The stages themselves live in
separate modules because they straddle the repo's hard environment split: the
posterior dump runs under the jax venv, the counterfactual labelling under the
torch env, and the two exchange npz files and nothing else.
"""

from .groups import (
    GRANULARITIES,
    NUM_POSITIONS,
    NUM_SLOTS,
    NUM_SUBSPACES,
    PROTOCOL,
    group_of,
    position_of,
    raw_mask_bits,
    slot_of,
    subspace_of,
)

__all__ = [
    "GRANULARITIES",
    "NUM_POSITIONS",
    "NUM_SLOTS",
    "NUM_SUBSPACES",
    "PROTOCOL",
    "group_of",
    "position_of",
    "raw_mask_bits",
    "slot_of",
    "subspace_of",
]
