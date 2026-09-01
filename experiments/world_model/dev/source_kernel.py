"""Dense reconstruction of the source-conditioned Markov kernel.

``baselines.py`` evaluates the source baseline transition-by-transition and
never persists its count tables, so any kernel-level diagnostic has to refit
them. This module reproduces that arithmetic exactly -- including the
leave-source-out renormalisation of the copy prior, which makes the copy
"prior" a genuine kernel ``copy_p(c'|c)`` rather than a function of
``c' == c`` alone.

``baselines.py`` is deliberately not modified; only its constants are imported,
and ``tests/world_model/test_dev_source_kernel.py`` asserts that the kernel
built here reproduces the frozen ``source_code_bits`` to 1e-9.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..baselines import ALPHA, SOURCE_PRIOR_CONCENTRATION, cache_positions
from ..cache import FrozenCache
from ..schema import NUM_CATEGORIES


@dataclasses.dataclass(frozen=True)
class TaskTables:
    """Per-task fit statistics, sufficient to rebuild any position's kernel."""

    task_id: int
    n_fit: int
    fit_source: np.ndarray      # (n_fit, POSITIONS) uint8
    fit_target: np.ndarray      # (n_fit, POSITIONS) uint8
    p_keep: np.ndarray          # (POSITIONS,) float64
    change_count: np.ndarray    # (POSITIONS,) int64
    changed_destination: np.ndarray  # (POSITIONS, NUM_CATEGORIES) int64


def fit_task(cache: FrozenCache, task_id: int, fit_rows: np.ndarray) -> TaskTables:
    """Refit one task's copy tables from the frozen cache (train rows only)."""
    POSITIONS = cache_positions(cache)  # noqa: N806 -- geometry from the cache, not the v8 default
    fit_rows = np.asarray(fit_rows, np.int64)
    task_ids = cache.transitions["task_ids"][fit_rows]
    rows = fit_rows[task_ids == task_id]
    if len(rows) < 1:
        raise ValueError(f"task {task_id} has no fit transitions")

    source_ix = cache.transitions["history_indices"][rows, -1]
    target_ix = cache.transitions["target_indices"][rows]
    if np.any(source_ix < 0):
        raise ValueError("a transition's current source state cannot be padding")
    source = np.asarray(cache.codes[source_ix]).reshape(len(rows), POSITIONS)
    target = np.asarray(cache.codes[target_ix]).reshape(len(rows), POSITIONS)

    n_fit = len(rows)
    same = source == target
    keep_count = same.sum(axis=0, dtype=np.int64)
    change_count = n_fit - keep_count
    offsets = np.arange(POSITIONS, dtype=np.int64) * NUM_CATEGORIES
    changed_destination = np.bincount(
        (target.astype(np.int64) + offsets)[~same],
        minlength=POSITIONS * NUM_CATEGORIES,
    ).reshape(POSITIONS, NUM_CATEGORIES)
    p_keep = (keep_count + ALPHA) / (n_fit + 2 * ALPHA)
    return TaskTables(
        task_id=int(task_id), n_fit=n_fit, fit_source=source, fit_target=target,
        p_keep=p_keep, change_count=change_count,
        changed_destination=changed_destination,
    )


def copy_kernel(tables: TaskTables, positions: np.ndarray) -> np.ndarray:
    """Dense ``copy_p(c'|c)`` for the given positions -> (P, 256, 256) float64."""
    positions = np.asarray(positions, np.int64)
    destination = tables.changed_destination[positions].astype(np.float64) + ALPHA
    # Leave-source-out denominator: the source category's own mass is removed,
    # which is what makes the change distribution normalise over c' != c.
    total = (
        tables.change_count[positions].astype(np.float64)
        + NUM_CATEGORIES * ALPHA
    )
    denominator = total[:, None] - destination                    # (P, 256) indexed by c
    kernel = destination[:, None, :] / denominator[:, :, None]    # (P, c, c')
    p_keep = tables.p_keep[positions]
    kernel *= (1.0 - p_keep)[:, None, None]
    diagonal = np.arange(NUM_CATEGORIES)
    kernel[:, diagonal, diagonal] = p_keep[:, None]
    return kernel


def pair_counts(tables: TaskTables, positions: np.ndarray) -> np.ndarray:
    """Dense train pair counts ``n_p(c, c')`` -> (P, 256, 256) int64."""
    positions = np.asarray(positions, np.int64)
    source = tables.fit_source[:, positions].astype(np.int64)
    target = tables.fit_target[:, positions].astype(np.int64)
    local = np.arange(len(positions), dtype=np.int64) * (
        NUM_CATEGORIES * NUM_CATEGORIES
    )
    keys = (local + source * NUM_CATEGORIES + target).ravel()
    return np.bincount(
        keys, minlength=len(positions) * NUM_CATEGORIES * NUM_CATEGORIES
    ).reshape(len(positions), NUM_CATEGORIES, NUM_CATEGORIES)


def source_kernel(tables: TaskTables, positions: np.ndarray) -> np.ndarray:
    """Dense ``K_p(c'|c)`` -> (P, 256, 256) float64, rows summing to 1."""
    counts = pair_counts(tables, positions).astype(np.float64)
    copy = copy_kernel(tables, positions)
    source_n = counts.sum(axis=-1)
    return (counts + SOURCE_PRIOR_CONCENTRATION * copy) / (
        source_n[:, :, None] + SOURCE_PRIOR_CONCENTRATION
    )


def source_code_bits(
    cache: FrozenCache,
    fit_rows: np.ndarray,
    eval_rows: np.ndarray,
    *,
    block: int = 128,
) -> np.ndarray:
    """Per-transition source codelength, rebuilt through the dense kernel.

    This exists so the dense kernel used by the residual diagnostic can be
    checked against the frozen baseline artifact bit-for-bit.
    """
    POSITIONS = cache_positions(cache)  # noqa: N806
    eval_rows = np.asarray(eval_rows, np.int64)
    fit_rows = np.asarray(fit_rows, np.int64)
    bits = np.zeros((len(eval_rows),), np.float64)
    eval_tasks = cache.transitions["task_ids"][eval_rows]
    for task_id in sorted(set(eval_tasks.tolist())):
        tables = fit_task(cache, task_id, fit_rows)
        local = np.flatnonzero(eval_tasks == task_id)
        rows = eval_rows[local]
        source = np.asarray(
            cache.codes[cache.transitions["history_indices"][rows, -1]]
        ).reshape(len(rows), POSITIONS)
        target = np.asarray(
            cache.codes[cache.transitions["target_indices"][rows]]
        ).reshape(len(rows), POSITIONS)
        accumulated = np.zeros((len(rows),), np.float64)
        for start in range(0, POSITIONS, block):
            positions = np.arange(start, min(start + block, POSITIONS))
            kernel = source_kernel(tables, positions)
            probability = kernel[
                np.arange(len(positions))[None, :],
                source[:, positions].astype(np.int64),
                target[:, positions].astype(np.int64),
            ]
            accumulated += -np.log2(probability).sum(axis=1, dtype=np.float64)
        bits[local] = accumulated
    return bits
