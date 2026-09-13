"""AMA open-loop lambda curve and control arms.

Port of ``global_mask_curve.py`` to the AMA stack. The question it answers is
the same one the WebWorld report asks: a single fixed mask over the 1024 code
positions, dropped when ``|U_j| < lambda * H_j`` -- what does it cost in bits and
what does it cost the reader's answer?

Two axes, deliberately measured over different sets (UTILITY_GATE.md section 0):

* **rate** over every validation state the world model can predict. Storage
  happens once per state and does not care whether anybody asks about it.
* **quality** only over the states that carry a question, which is the only
  place it can be measured.

Both traps the report warns about are avoided by construction: the rate is not
weighted by question count, and it is not restricted to states that have
questions.

Control arms are matched per state, not per corpus: every arm keeps exactly as
many positions as the gate did for that state, so the comparison is at equal
budget on the same row.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.utility_gate.groups import NUM_POSITIONS
from experiments.utility_gate.label_counterfactual_ama import (
    answer_variant_scores_ama, load_ama_reader, load_codes, load_pairs,
    load_posteriors, load_xbar, place_bridge, split_of_state,
)
from experiments.utility_gate.verify_scorer import load_codebook, rebuild

ARMS = ("random", "displacement", "rate")


def load_labels(paths: list[Path]) -> dict:
    parts = [np.load(p, allow_pickle=True) for p in paths]
    table = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files
             if k != "metadata"}
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq", default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    parser.add_argument("--codebook",
                        default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    parser.add_argument("--records", default="outputs/wm_train/ama-v1/records.jsonl")
    parser.add_argument("--states", default="outputs/wm_train/ama-v1/states.jsonl")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="fit labels from label_counterfactual_ama.py (train)")
    parser.add_argument("--xbar-shards", nargs="*", default=None)
    parser.add_argument("--posteriors", required=True, help="validation posteriors")
    parser.add_argument("--pairs", required=True, help="evaluation pairs")
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--reader-model", default="/data1/models/Qwen3-32B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto",
                        help="'auto' shards the float32 reader across the visible "
                             "GPUs")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--enable-thinking", type=int, default=1)
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0005, 0.00075, 0.0010, 0.00125, 0.0015, 0.0020])
    parser.add_argument("--max-rows", type=int, default=200,
                        help="question rows scored with the reader")
    parser.add_argument("--max-answer-tokens", type=int, default=104)
    parser.add_argument("--reference-rate", type=float, default=None,
                        help="full-send bits/transition to print alongside, e.g. "
                             "the value final_eval.py reports for this checkpoint")
    parser.add_argument("--fixed-width-bits", type=float, default=6144.0)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    book = load_codebook(Path(args.codebook))
    codes, state_ids = load_codes(Path(args.pq))
    by_id = {sid: i for i, sid in enumerate(state_ids)}
    split, shared = split_of_state(Path(args.states), Path(args.records))
    row_of, dumped = load_posteriors(Path(args.posteriors), Path(args.records))
    xbar = load_xbar([Path(p) for p in args.xbar_shards]) if args.xbar_shards else {}

    # ---------------------------------------------------------------- fit |U|
    table = load_labels([Path(p) for p in args.labels])
    fit_position = table["position"]
    fit_utility = table["delta_nll_bits"].astype(np.float64)
    fit_abs = np.abs(fit_utility)

    magnitude = np.zeros(NUM_POSITIONS)
    signed = np.zeros(NUM_POSITIONS)
    for p in range(NUM_POSITIONS):
        block = fit_abs[fit_position == p]
        magnitude[p] = block.mean() if len(block) else 0.0
        signed_block = fit_utility[fit_position == p]
        signed[p] = signed_block.mean() if len(signed_block) else 0.0
    print(f"[curve] |U| fitted from {len(fit_abs)} labels covering "
          f"{len(np.unique(fit_position))} positions", flush=True)

    # ------------------------------------------------- state-level quantities
    # Rate and displacement are averaged over the validation states that have a
    # posterior, not over the labelled rows: they need no reader and every state
    # pays its storage bill.
    val_states = [sid for sid, row in by_id.items()
                  if split.get(sid) == "validation" and sid not in shared
                  and sid in row_of]
    val_states.sort()
    slot = np.arange(NUM_POSITIONS) // 32
    sub = np.arange(NUM_POSITIONS) % 32
    val_index = np.asarray([by_id[s] for s in val_states])
    val_posterior = np.asarray([row_of[s] for s in val_states])
    with np.load(args.codebook, allow_pickle=True) as data:
        centroids = np.asarray(data["centroids"], np.float32)

    true_code = codes[val_index][:, slot, sub]                       # (S, 1024)
    fill_code = dumped["wm_argmax"][val_posterior][:, slot, sub]      # (S, 1024)
    rate_bits = dumped["code_bits"][val_posterior][:, slot, sub].astype(np.float64)
    entropy = dumped["entropy_bits"][val_posterior][:, slot, sub].astype(np.float64)
    displacement = np.linalg.norm(
        centroids[slot, sub, true_code] - centroids[slot, sub, fill_code],
        axis=-1).mean(0).astype(np.float64)
    mean_rate = rate_bits.mean(0)
    print(f"[curve] rate/displacement averaged over {len(val_states)} validation "
          f"states; full-send {rate_bits.sum(1).mean():.2f} bits/state", flush=True)

    rng = np.random.default_rng(args.seed)
    scores = {
        "magnitude": magnitude,
        "displacement": displacement,
        "rate": mean_rate,
        "random": rng.random(NUM_POSITIONS),
    }
    del signed  # kept only for the diagnostic below; the criterion is |U|
    diagnostic = {
        "signed_negative_fraction": float((fit_utility < 0).mean()),
        "magnitude_spearman_rate": float(np.corrcoef(
            np.argsort(np.argsort(magnitude)),
            np.argsort(np.argsort(mean_rate)))[0, 1]),
    }

    # --------------------------------------------------------- corpus rate axis
    full_send_bits = float(rate_bits.sum(1).mean())
    corpus = []
    for lam in args.lambdas:
        keep = magnitude[None, :] >= lam * entropy
        kept = keep.sum(1)
        corpus.append({
            "lambda": float(lam),
            "keep_fraction": float(kept.mean() / NUM_POSITIONS),
            "rate_bits": float(rate_bits[keep].sum() / len(val_states)),
            "full_send_bits": full_send_bits,
        })
    for entry in corpus:
        entry["saved_fraction"] = 1.0 - entry["rate_bits"] / entry["full_send_bits"]
        entry["compression_ratio"] = args.fixed_width_bits / entry["rate_bits"]

    # ------------------------------------------------------- quality axis
    pairs = load_pairs(Path(args.pairs))
    usable = []
    for pair_row, pair in enumerate(pairs):
        sid = str(pair["state_id"])
        if sid not in by_id or split.get(sid) != "validation":
            continue
        if sid in shared or sid not in row_of:
            continue
        usable.append(pair_row)
    chooser = np.random.default_rng(args.seed + 2)
    usable = list(chooser.permutation(usable))[: args.max_rows]
    print(f"[curve] {len(usable)} evaluation rows scored with the reader", flush=True)

    device = torch.device(args.device)
    tokenizer, model = load_ama_reader(
        args.reader_model, args.dtype, args.device, args.device_map)
    bridge, _metadata = place_bridge(Path(args.bridge), model)

    records = []
    started = time.time()
    for done, pair_row in enumerate(usable):
        pair = pairs[pair_row]
        sid = str(pair["state_id"])
        state = by_id[sid]
        posterior = row_of[sid]
        question, answer = str(pair["question"]), str(pair["answer"])

        state_entropy = dumped["entropy_bits"][posterior].reshape(-1).astype(np.float64)
        state_rate = dumped["code_bits"][posterior].reshape(-1).astype(np.float64)
        fill = dumped["wm_argmax"][posterior].reshape(-1)
        base = codes[state].reshape(-1)

        variants, names, meta = [rebuild(codes[state][None], book)], ["reference"], []
        for lam in args.lambdas:
            keep = magnitude >= lam * state_entropy
            gated = base.copy()
            gated[~keep] = fill[~keep]
            variants.append(rebuild(gated.reshape(codes[state].shape)[None], book))
            names.append(f"gate@{lam}")
            meta.append({"arm": "gate", "lambda": float(lam),
                         "kept": int(keep.sum()),
                         "rate_bits": float(state_rate[keep].sum())})
            order = {arm: np.argsort(-scores[arm]) for arm in ARMS}
            for arm, ranked in order.items():
                keep_arm = np.zeros(NUM_POSITIONS, bool)
                keep_arm[ranked[: int(keep.sum())]] = True
                arm_codes = base.copy()
                arm_codes[~keep_arm] = fill[~keep_arm]
                variants.append(rebuild(arm_codes.reshape(codes[state].shape)[None], book))
                names.append(f"{arm}@{lam}")
                meta.append({"arm": arm, "lambda": float(lam),
                             "kept": int(keep_arm.sum()),
                             "rate_bits": float(state_rate[keep_arm].sum())})
        if sid in xbar:
            variants.append(xbar[sid][None])
            names.append("raw_xbar")

        batch = np.concatenate(variants).astype(np.float32)
        with torch.no_grad():
            soft, _mask = bridge(
                torch.as_tensor(batch, device=device)[:, None],
                torch.ones((batch.shape[0], 1, batch.shape[1]),
                           dtype=torch.bool, device=device))
            scored = answer_variant_scores_ama(
                model, tokenizer, soft, question, answer,
                enable_thinking=bool(args.enable_thinking),
                max_answer_tokens=args.max_answer_tokens)
        nll = scored["nll_bits"].numpy()
        for offset, (name, entry) in enumerate(zip(names[1:], meta), start=1):
            records.append({
                "pair_row": int(pair_row), "state_id": sid, "variant": name,
                "arm": entry["arm"], "lambda": entry["lambda"],
                "kept": entry["kept"],
                "rate_bits": entry["rate_bits"],
                "full_rate_bits": float(state_rate.sum()),
                "delta_nll_bits": float(nll[offset] - nll[0]),
                "reference_nll_bits": float(nll[0]),
            })
        if done % 10 == 0:
            elapsed = time.time() - started
            print(f"[curve] {done + 1}/{len(usable)}  "
                  f"{elapsed / max(done + 1, 1):.2f}s/row", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{
        key: np.asarray([r[key] for r in records]) for key in records[0]})

    # ------------------------------------------------------------- aggregation
    def rows_for(arm: str, lam: float) -> list[dict]:
        return [r for r in records if r["arm"] == arm and r["lambda"] == lam]

    report = {"corpus_rate": corpus, "quality": [], "diagnostic": diagnostic,
              "full_send_bits_per_state": full_send_bits,
              "states_with_posterior": len(val_states),
              "rows_scored": len(usable),
              "fixed_width_bits": args.fixed_width_bits,
              "reference_rate": args.reference_rate}
    for lam in args.lambdas:
        gate = rows_for("gate", lam)
        abs_gate = np.abs([r["delta_nll_bits"] for r in gate])
        entry = {
            "lambda": float(lam),
            "rows": len(gate),
            "kept": float(np.mean([r["kept"] for r in gate])),
            "keep_fraction": float(np.mean([r["kept"] for r in gate]) / NUM_POSITIONS),
            "rate_bits": float(np.mean([r["rate_bits"] for r in gate])),
            "full_rate_bits": float(np.mean([r["full_rate_bits"] for r in gate])),
            "abs_delta_nll_bits": float(abs_gate.mean()),
            "abs_delta_nll_median": float(np.median(abs_gate)),
            "arms": {},
        }
        for arm in ARMS:
            other = rows_for(arm, lam)
            abs_other = np.abs([r["delta_nll_bits"] for r in other])
            diff = abs_gate - abs_other
            n = len(diff)
            t = float(diff.mean() / (diff.std(ddof=1) / math.sqrt(n))) if n > 1 and diff.std(ddof=1) > 0 else float("nan")
            entry["arms"][arm] = {
                "abs_delta_nll_bits": float(abs_other.mean()),
                "paired_t": t,
                "mean_gap_bits": float(diff.mean()),
                "budget_matched": bool(np.all([a["kept"] == b["kept"]
                                               for a, b in zip(gate, other)])),
            }
        report["quality"].append(entry)

    (out.with_suffix(".report.json")).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["quality"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
