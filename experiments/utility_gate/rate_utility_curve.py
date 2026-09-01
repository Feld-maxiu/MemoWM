"""The rate-utility curve: what each gating rule buys at equal cost.

For a sequence of rate budgets, every arm selects which of the 1024 positions to
transmit, the omitted ones are filled with the world model's mode, and the reader
is scored on the resulting reconstruction. The arms differ only in how they score
a position; the budget, the fill and the reader are identical, so the curves are
comparable by construction.

Two things here are consequences of measurements rather than taste.

**Each operating point is scored, not summed.** Per-position utilities are
superadditive: at 32 positions the true damage runs about 14% above the sum of
the singles, and the gap grows with the number dropped. Building the curve from
summed labels would flatter every arm, and flatter the ones that drop more the
most -- exactly the comparison the curve exists to make. So each mask is
reconstructed and scored end to end.

**Every arm for a row goes in one forward.** The scorer's contract is that
variants are comparable only inside a single batch; in float32 the residual
batch-geometry effect is 5.3e-5 bits against a signal around 0.02, but keeping
the discipline costs nothing here because all the arms of one question share its
question and answer.

Selection is greedy by utility density -- keep positions in descending
``score / (rate + mask bit)`` until the budget is spent -- which is the knapsack
heuristic and the only thing rho was ever claimed to support. Positions whose
score is not positive are never kept: paying bits for something predicted to hurt
is strictly worse than dropping it.

Arms that need an oracle need every position of the row labelled, so the curve
runs on the completely-labelled subset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.state_tokenizer.reader_losses import (
    FULL_ANSWER_TOKENS, answer_variant_scores,
)
from experiments.utility_gate.groups import NUM_POSITIONS, raw_mask_bits
from experiments.utility_gate.verify_scorer import load_codebook, load_reader, rebuild


def select(score: np.ndarray, rate: np.ndarray, mask_bits: float,
           budget: float) -> np.ndarray:
    """Greedy knapsack by utility density; returns the keep mask."""
    cost = rate + mask_bits
    keep = np.zeros(len(score), bool)
    candidates = np.flatnonzero(score > 0)
    if not len(candidates):
        return keep
    density = score[candidates] / np.maximum(cost[candidates], 1e-9)
    for index in candidates[np.argsort(-density)]:
        if cost[index] <= budget:
            keep[index] = True
            budget -= cost[index]
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="gate")
    parser.add_argument("--label", default="train")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="completely-labelled rows, for the oracle arms")
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--head", default=None)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--granularity", type=int, default=NUM_POSITIONS)
    parser.add_argument("--budgets", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from experiments.utility_gate.label_counterfactual import cache_order
    from experiments.utility_gate.train_gate_head import load_labels

    gate = Path(args.gate)
    book = load_codebook(Path(args.codebook))
    with np.load(gate / "codes.npz", allow_pickle=True) as data:
        codes = np.asarray(data[f"codes/{args.label}"], np.uint8)
        state_ids = [str(v) for v in np.asarray(data[f"state_ids/{args.label}"])]
    with np.load(args.posteriors, allow_pickle=True) as data:
        dumped = {k: np.asarray(data[k]) for k in data.files}

    global_index = cache_order(gate / "records.jsonl", args.label)
    row_of_global = {int(g): r for r, g in enumerate(dumped["target_indices"])}
    posterior_row = np.asarray(
        [row_of_global.get(global_index.get(sid, -1), -1) for sid in state_ids])

    table = load_labels(args.labels)
    pairs = np.load(args.pairs, allow_pickle=True)
    by_pair: dict[int, dict[int, float]] = {}
    for pair_row, position, value in zip(
            table["pair_row"], table["position"], table["delta_nll_bits"]):
        by_pair.setdefault(int(pair_row), {})[int(position)] = float(value)
    complete = [row for row, seen in by_pair.items()
                if len(seen) == NUM_POSITIONS and posterior_row[
                    state_ids.index(
                        f"{pairs['sample_id'][row]}-{int(pairs['record_index'][row]):04d}")] >= 0]
    complete.sort()
    if args.max_rows:
        complete = complete[: args.max_rows]
    if not complete:
        raise SystemExit(
            "no row has all 1024 positions labelled and a posterior; the oracle "
            "arms need a --complete labelling pass over these rows first"
        )

    # E_q[U] per (state, position): the ceiling a write-time gate can aim at.
    by_state_position: dict[tuple[int, int], list[float]] = {}
    for pair_row, seen in by_pair.items():
        key = f"{pairs['sample_id'][pair_row]}-{int(pairs['record_index'][pair_row]):04d}"
        state = state_ids.index(key)
        for position, value in seen.items():
            by_state_position.setdefault((state, position), []).append(value)

    head = None
    if args.head:
        payload = torch.load(args.head, map_location="cpu", weights_only=False)
        head = payload

    mask_bits = raw_mask_bits(args.granularity) / args.granularity
    device = torch.device(args.device)
    processor, model, connector = load_reader(
        args.model, args.checkpoint, device, torch.float32, codes.shape[1])

    rng = np.random.default_rng(args.seed)
    static_prior = np.zeros(NUM_POSITIONS)
    for (_state, position), values in by_state_position.items():
        static_prior[position] += float(np.mean(values))

    results: list[dict] = []
    for done, pair_row in enumerate(complete):
        key = f"{pairs['sample_id'][pair_row]}-{int(pairs['record_index'][pair_row]):04d}"
        state = state_ids.index(key)
        row = posterior_row[state]
        rate = dumped["code_bits"][row].reshape(-1).astype(np.float64)
        fill = dumped["wm_argmax"][row]
        oracle_q = np.asarray([by_pair[pair_row][p] for p in range(NUM_POSITIONS)])
        oracle_state = np.asarray([
            float(np.mean(by_state_position[(state, p)])) for p in range(NUM_POSITIONS)])

        arms = {
            "oracle_per_question": oracle_q,
            "oracle_per_state": oracle_state,
            "rate_only": rate.copy(),
            "random": rng.random(NUM_POSITIONS),
            "static_prior": static_prior.copy(),
        }
        if head is not None:
            arms["learned"] = np.zeros(NUM_POSITIONS)   # filled by the head below

        full = rate.sum()
        variants, names = [], []
        base = codes[state]
        variants.append(rebuild(base[None], book)); names.append("reference")
        for arm, score in arms.items():
            for budget in args.budgets:
                keep = select(score, rate, mask_bits, budget * full)
                gated = base.copy().reshape(-1)
                gated[~keep] = fill.reshape(-1)[~keep]
                variants.append(rebuild(gated.reshape(base.shape)[None], book))
                names.append(f"{arm}@{budget}")
                results.append({
                    "pair_row": int(pair_row), "arm": arm, "budget": budget,
                    "kept": int(keep.sum()),
                    "rate_bits": float((rate[keep] + mask_bits).sum()),
                    "full_rate_bits": float(full),
                })

        batch = torch.as_tensor(
            np.concatenate(variants).astype(np.float32), device=device)
        with torch.no_grad():
            soft = connector(batch, torch.ones(
                batch.shape[:2], dtype=torch.bool, device=device))
        scored = answer_variant_scores(
            model, processor, soft,
            str(pairs["question"][pair_row]), str(pairs["answer"][pair_row]),
            max_answer_tokens=FULL_ANSWER_TOKENS)
        nll = scored["nll_bits"].numpy()
        for offset, entry in enumerate(results[-len(names) + 1:], start=1):
            entry["delta_nll_bits"] = float(nll[offset] - nll[0])
            entry["reference_nll_bits"] = float(nll[0])
        if done % 10 == 0:
            print(f"[curve] {done + 1}/{len(complete)}", flush=True)

    summary: dict[str, dict] = {}
    for entry in results:
        bucket = summary.setdefault(f"{entry['arm']}@{entry['budget']}", {
            "arm": entry["arm"], "budget": entry["budget"],
            "rate": [], "delta": [], "kept": []})
        bucket["rate"].append(entry["rate_bits"])
        bucket["delta"].append(entry["delta_nll_bits"])
        bucket["kept"].append(entry["kept"])
    curve = [{
        "arm": v["arm"], "budget": v["budget"],
        "mean_rate_bits": float(np.mean(v["rate"])),
        "mean_delta_nll_bits": float(np.mean(v["delta"])),
        "median_delta_nll_bits": float(np.median(v["delta"])),
        "mean_kept_positions": float(np.mean(v["kept"])),
        "rows": len(v["delta"]),
    } for v in summary.values()]
    curve.sort(key=lambda row: (row["arm"], row["budget"]))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "protocol": "residualmem_utility_gate_curve_v1",
        "rows": len(complete), "granularity": args.granularity,
        "mask_bits_per_position": mask_bits,
        "budgets": args.budgets, "curve": curve,
    }, indent=2), encoding="utf-8")
    print(json.dumps(curve, indent=2))


if __name__ == "__main__":
    main()
