"""Episode-ordered views over a transition cache, built without rebuilding it.

The windowed model samples random transitions, each carrying a pre-baked window
of the last ``max_history`` observations. A recurrent state cannot be trained
that way: ``h_t`` depends on ``h_{t-1}``, so the batch has to be ordered in time.

Rebuilding the cache for that turns out to be unnecessary. Sorting the existing
transitions by ``(episode_id, step)`` makes them adjacent -- transition *n*'s
target is transition *n+1*'s current state, verified over the pilot cache -- so a
whole trajectory is

    [history_indices[n, -1] for n in episode] + [target_indices[last]]

and the action taken at each of those states is ``action_*[n, -1]``. Everything a
recurrent model needs is already on disk under a different indexing.

Episodes are cut into fixed-length chunks so a batch is rectangular. A chunk
boundary is a genuine loss of context -- ``h`` restarts -- so ``chunk_length``
should exceed the bulk of the length distribution. On MolmoWeb the mean is 17.5
states and the p90 is 32, so 64 leaves only the tail split, and the fraction of
transitions that lose their prefix is reported rather than assumed.
"""
from __future__ import annotations

import numpy as np

from .cache import FrozenCache


#: Per-step action columns, taken from the last entry of each transition's window.
ACTION_COLUMNS = (
    "action_types", "action_payloads", "action_lengths",
    "action_tags", "action_refs", "action_x", "action_y",
    "action_dx", "action_dy", "action_has_coord", "action_has_delta",
)


class EpisodeView:
    """Chunks of consecutive steps, and the transition each step corresponds to."""

    def __init__(self, cache: FrozenCache, *, chunk_length: int = 64):
        self.cache = cache
        self.chunk_length = int(chunk_length)
        transitions = cache.transitions
        episodes = np.asarray(transitions["episode_ids"], np.int64)
        steps = np.asarray(transitions["steps"], np.int64)
        order = np.lexsort((steps, episodes))

        self._order = order
        self._episodes = episodes[order]
        self._current = np.asarray(transitions["history_indices"], np.int64)[order, -1]
        self._target = np.asarray(transitions["target_indices"], np.int64)[order]
        self._actions = {
            name: np.asarray(transitions[name])[order, -1]
            for name in ACTION_COLUMNS if name in transitions
        }
        self._split = np.asarray(transitions["split_ids"], np.uint8)[order]
        self._task = np.asarray(transitions["task_ids"], np.uint8)[order]

        boundaries = np.flatnonzero(np.diff(self._episodes)) + 1
        self._segments = np.split(np.arange(len(order)), boundaries)

    def chunks_for(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Chunk every episode that ``rows`` touches, keeping only rows in it.

        ``loss_mask`` marks the steps whose bits are wanted; ``valid`` marks the
        steps that exist at all. They differ because a chunk may span an episode
        whose other steps belong to a different split -- those still run, so the
        recurrent state is correct, but they are not scored.
        """
        wanted = np.zeros(len(self._order), np.bool_)
        position = np.empty(len(self._order), np.int64)
        position[self._order] = np.arange(len(self._order))
        wanted[position[np.asarray(rows, np.int64)]] = True

        length = self.chunk_length
        states, transitions_in, valid, loss = [], [], [], []
        for segment in self._segments:
            if not wanted[segment].any():
                continue
            for start in range(0, len(segment), length):
                block = segment[start:start + length]
                pad = length - len(block)
                states.append(np.concatenate([
                    self._current[block], np.zeros(pad, np.int64)
                ]))
                transitions_in.append(np.concatenate([
                    self._order[block], np.full(pad, -1, np.int64)
                ]))
                valid.append(np.concatenate([
                    np.ones(len(block), np.bool_), np.zeros(pad, np.bool_)
                ]))
                loss.append(np.concatenate([
                    wanted[block], np.zeros(pad, np.bool_)
                ]))

        if not states:
            raise ValueError("no episode contains any of the requested rows")
        packed = {
            "state_indices": np.stack(states),
            "transition_indices": np.stack(transitions_in),
            "valid": np.stack(valid),
            "loss_mask": np.stack(loss),
        }
        # Only chunks that score something are worth running.
        keep = packed["loss_mask"].any(axis=1)
        return {name: value[keep] for name, value in packed.items()}

    def materialise(self, chunk: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Codes and actions for one batch of chunks, padded steps zeroed."""
        states = chunk["state_indices"]
        valid = chunk["valid"]
        rows = np.maximum(chunk["transition_indices"], 0)
        codes = np.asarray(self.cache.codes[states], np.uint8)
        batch = {
            "codes": codes,
            "code_valid": np.asarray(self.cache.valid[states], np.bool_) & valid[..., None],
            "valid": valid,
            "loss_mask": chunk["loss_mask"],
            "transition_indices": chunk["transition_indices"],
            "task_ids": self._task[rows],
        }
        targets = self._target[rows]
        batch["target_codes"] = np.asarray(self.cache.codes[targets], np.uint8)
        batch["target_valid"] = np.asarray(self.cache.valid[targets], np.bool_)
        for name, values in self._actions.items():
            batch[name] = values[rows]
        return batch

    def coverage(self) -> dict:
        """How much context the chunking costs, in the units that matter."""
        lengths = np.array([len(s) for s in self._segments])
        split = np.maximum(np.ceil(lengths / self.chunk_length).astype(np.int64), 1)
        lost = int((lengths - np.minimum(lengths, self.chunk_length))[split > 1].sum())
        return {
            "episodes": int(len(lengths)),
            "transitions": int(lengths.sum()),
            "chunk_length": self.chunk_length,
            "episodes_split": int((split > 1).sum()),
            "transitions_after_a_boundary": lost,
            "fraction_after_a_boundary": float(lost / max(lengths.sum(), 1)),
        }
