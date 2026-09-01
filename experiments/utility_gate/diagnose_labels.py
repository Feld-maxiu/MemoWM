"""Read a label table and answer the questions that decide whether to go on.

The labelling pass measures what omitting a code costs. These are the checks on
whether that measurement can support a write-time gate at all, stated before the
numbers were seen and reported whether or not they pass.

**D3 -- the variance decomposition, and the reason the rest matters.** A gate
decides at write time, before any question exists, so it can at best predict
``E_q[U]``. Splitting the variance of U three ways --

    between states / between positions within a state / between questions of the
    same (state, position)

-- bounds what is achievable. If most of it sits in the last term, the future
question is what determines utility, no feature of the code can recover it, and
the ceiling for any head is the per-state mean. That is a stopping condition, not
a tuning problem, which is why the labelling samples positions per state rather
than per row: the last term needs the same (state, position) seen under more than
one question, and independent per-row sampling would give it almost no pairs.

**D4 -- is utility just rate?** If the two rank positions identically there is
nothing for a utility head to add over a surprise gate, and the whole hypothesis
is refuted cheaply.

**D6 -- is it just position identity?** A gate that reduces to a fixed mask is a
much weaker result than one that reads the state. Reported as the share of
variance a slot- or subspace-only predictor explains.

**D7 -- do KL and the answer NLL agree?** They are computed from the same logits
and measure different things: KL is the whole distribution's movement, the NLL
delta is the gold token's. If they rank positions differently, KL is not
measuring anything task-directed and must not be used alone.

Also reported: whether the summed delta carries an answer-length scale, since
that decides whether the sum or the per-token mean is the quantity commensurate
with the code bits.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denominator = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denominator) if denominator else float("nan")


def variance_components(state: np.ndarray, position: np.ndarray,
                        value: np.ndarray) -> dict:
    """Split var(U) into state / position-within-state / question-within-cell.

    Only cells seen under more than one question contribute to the last term, so
    the count of such cells is reported alongside -- a decomposition estimated
    from a handful of pairs is not evidence of anything.
    """
    keys = state.astype(np.int64) * 100000 + position.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    keys, values, states = keys[order], value[order], state[order]
    boundaries = np.flatnonzero(np.diff(keys)) + 1
    cells = np.split(np.arange(len(keys)), boundaries)

    cell_means, cell_states, within_cell, repeated = [], [], [], 0
    for cell in cells:
        block = values[cell]
        cell_means.append(block.mean())
        cell_states.append(states[cell[0]])
        if len(block) > 1:
            within_cell.append(block.var())
            repeated += 1
    cell_means = np.asarray(cell_means)
    cell_states = np.asarray(cell_states)

    state_means, position_within = [], []
    for one in np.unique(cell_states):
        block = cell_means[cell_states == one]
        state_means.append(block.mean())
        if len(block) > 1:
            position_within.append(block.var())

    between_states = float(np.var(state_means)) if len(state_means) > 1 else 0.0
    between_positions = float(np.mean(position_within)) if position_within else 0.0
    between_questions = float(np.mean(within_cell)) if within_cell else 0.0
    total = between_states + between_positions + between_questions
    share = (lambda v: float(v / total) if total > 0 else float("nan"))
    return {
        "between_states": between_states,
        "between_positions_within_state": between_positions,
        "between_questions_within_cell": between_questions,
        "share_between_states": share(between_states),
        "share_between_positions": share(between_positions),
        "share_between_questions": share(between_questions),
        "cells": int(len(cells)),
        "cells_with_repeats": int(repeated),
        "states": int(len(state_means)),
    }


def grouped_r2(group: np.ndarray, value: np.ndarray) -> float:
    """Share of var(value) explained by the group mean alone."""
    total = float(value.var())
    if total <= 0:
        return float("nan")
    predicted = np.zeros_like(value, dtype=np.float64)
    for one in np.unique(group):
        mask = group == one
        predicted[mask] = value[mask].mean()
    return float(1.0 - ((value - predicted) ** 2).mean() / total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--noise-floor-bits", type=float, default=5.34e-5,
                        help="measured float32 batch-geometry floor; thresholds "
                             "are stated as multiples of it")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    parts: dict[str, list] = {}
    metadata = None
    for path in args.labels:
        with np.load(path, allow_pickle=True) as data:
            metadata = json.loads(str(np.asarray(data["metadata"])))
            for name in data.files:
                if name in ("metadata", "state_ids"):
                    continue
                parts.setdefault(name, []).append(np.asarray(data[name]))
    table = {name: np.concatenate(blocks) for name, blocks in parts.items()}

    delta = table["delta_nll_bits"].astype(np.float64)
    kl = table["kl_bits"].astype(np.float64)
    state = table["state_row"]
    position = table["position"]
    slot = position // 32
    subspace = position % 32
    floor = args.noise_floor_bits
    nonzero = delta != 0

    answer_len = np.repeat(
        table["row_answer_len"],
        [int((table["pair_row"] == row).sum()) for row in table["row_pair_row"]],
    ) if len(table.get("row_pair_row", [])) else None

    report = {
        "metadata": metadata,
        "labels": int(len(delta)),
        "rows": int(len(table.get("row_pair_row", []))),
        "states": int(len(np.unique(state))),
        "D1_signal": {
            "abs_median_bits": float(np.median(np.abs(delta))),
            "abs_median_nonzero_bits": float(np.median(np.abs(delta[nonzero])))
            if nonzero.any() else 0.0,
            "multiple_of_floor": float(np.median(np.abs(delta[nonzero])) / floor)
            if nonzero.any() else 0.0,
            "nonzero_fraction": float(nonzero.mean()),
            "negative_fraction": float((delta < 0).mean()),
            "p90_bits": float(np.percentile(np.abs(delta), 90)),
            "max_bits": float(np.abs(delta).max()),
        },
        "D3_variance_delta_nll": variance_components(state, position, delta),
        "D6_identity_only_r2": {
            "slot": grouped_r2(slot, delta),
            "subspace": grouped_r2(subspace, delta),
            "position": grouped_r2(position, delta),
        },
        "D7_kl_vs_nll": {
            "spearman_all": spearman(kl, np.abs(delta)),
            "spearman_nonzero": spearman(kl[nonzero], np.abs(delta[nonzero])),
            "kl_median_multiple_of_floor": float(np.median(kl[nonzero]) / floor)
            if nonzero.any() else 0.0,
            "argmax_changed_fraction": float(table["argmax_changed"].mean()),
        },
    }

    rate = table.get("rate_bits")
    if rate is not None and np.isfinite(rate).any():
        finite = np.isfinite(rate)
        report["D4_utility_vs_rate"] = {
            "spearman": spearman(rate[finite], np.abs(delta[finite])),
            "pearson": float(np.corrcoef(rate[finite], np.abs(delta[finite]))[0, 1]),
            "labels_with_rate": int(finite.sum()),
        }
    else:
        report["D4_utility_vs_rate"] = "no rate: labelled without world-model posteriors"

    if answer_len is not None and len(answer_len) == len(delta):
        report["answer_length_scale"] = {
            "corr_abs_delta_vs_len": float(
                np.corrcoef(np.abs(delta), answer_len)[0, 1]),
            "cancellation_ratio_median": float(np.median(
                table["abs_delta_nll_bits"] / np.maximum(np.abs(delta), 1e-12)))
            if "abs_delta_nll_bits" in table else None,
            "effective_tokens_median": float(np.median(table["effective_tokens"]))
            if "effective_tokens" in table else None,
            "answer_len_median": float(np.median(answer_len)),
        }

    verdict = {
        "D1_signal_over_floor": report["D1_signal"]["multiple_of_floor"] >= 100,
        "D1_nonzero_fraction": report["D1_signal"]["nonzero_fraction"] >= 0.5,
        "D3_position_share": (
            report["D3_variance_delta_nll"]["share_between_positions"] >= 0.20),
        "D7_kl_agrees": report["D7_kl_vs_nll"]["spearman_nonzero"] >= 0.7,
    }
    if isinstance(report["D4_utility_vs_rate"], dict):
        verdict["D4_not_collinear_with_rate"] = (
            abs(report["D4_utility_vs_rate"]["spearman"]) < 0.8)
    report["verdict"] = verdict
    report["verdict"]["all_passed"] = all(verdict.values())

    print(json.dumps(report, indent=2, sort_keys=False, default=float))
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(
            json.dumps(report, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
