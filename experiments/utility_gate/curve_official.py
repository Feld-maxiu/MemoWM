"""AMA open-loop lambda curve on the official questions, with control arms.

Companion to ``global_mask_curve_ama.py``, rebuilt for the official-question
pipeline. Two axes, measured over different sets on purpose (UTILITY_GATE.md
section 0):

* **rate** over every validation state the world model can predict. Storage
  happens once per state and does not care whether anyone asks about it, so the
  number is an unweighted per-state mean over all of them.
* **quality** only over the states that carry an official question, which is the
  only place the reader can be scored.

Control arms are matched per state, not per corpus: every arm keeps exactly as
many positions as the gate kept for that same row, so the comparison is at equal
budget on the same forward.

|U| is fitted on train rows and applied to validation rows. ``--prompt`` selects
the regime: ``latent-only`` isolates the codes (the report's utility
definition), ``matched`` is the deployed prompt with the retrieved step's text
row.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.state_tokenizer.common import iter_jsonl
from experiments.utility_gate.groups import NUM_POSITIONS, slot_of, subspace_of
from experiments.utility_gate.label_counterfactual_ama import (
    load_ama_reader, load_codes, load_posteriors, place_bridge, split_of_state,
)
from experiments.utility_gate.label_official import (
    build_deployed_prompt, build_prompt, score_variants, step_texts_for,
)
from experiments.utility_gate.verify_scorer import load_codebook, rebuild

ARMS = ("random", "displacement", "rate")


def fit_magnitude(paths):
    """Mean |U_j| and the supporting diagnostics, from the train labels."""
    parts = [np.load(p, allow_pickle=True) for p in paths]
    position = np.concatenate([p["position"] for p in parts])
    delta = np.concatenate([p["delta_nll_bits"] for p in parts])
    signed = np.zeros(NUM_POSITIONS)
    magnitude = np.zeros(NUM_POSITIONS)
    counts = np.zeros(NUM_POSITIONS)
    for j in range(NUM_POSITIONS):
        block = delta[position == j]
        counts[j] = block.size
        if block.size:
            magnitude[j] = np.abs(block).mean()
            signed[j] = block.mean()
    return magnitude, signed, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-labels", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS),
                        choices=list(ARMS) + ["none"],
                        help="control arms to score alongside the gate "
                             "(default: all; 'none' skips them)")
    parser.add_argument("--utility-override", default=None,
                        help="npz with a 'utility' vector to use as the gate "
                             "score (two-stage envelope) instead of fitting "
                             "magnitude from --fit-labels")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--pq", default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    parser.add_argument("--codebook",
                        default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--records", default="outputs/wm_train/ama-v1/records.jsonl")
    parser.add_argument("--states", default="outputs/wm_train/ama-v1/states.jsonl")
    parser.add_argument("--bridge", default=(
        "outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms.pt"))
    parser.add_argument("--reader-model", default="/data1/models/Qwen3-32B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--anchor-k", type=int, default=8)
    parser.add_argument("--anchor-style", default="element")
    parser.add_argument("--cache-dir", default="outputs/utility_gate_ama/ama-cache")
    parser.add_argument("--prompt", choices=("matched", "latent-only",
                                             "latent+anchor"),
                        default="latent-only")
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0005, 0.001, 0.002, 0.004, 0.008, 0.02])
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--max-prompt-tokens", type=int, default=0,
                        help="skip rows whose assembled prompt exceeds this; fp32 "
                             "attention is quadratic, so the handful of rows with "
                             "~30k-token step texts cannot be scored")
    parser.add_argument("--positions-per-forward", type=int, default=6)
    parser.add_argument("--max-answer-tokens", type=int, default=104)
    parser.add_argument("--max-model-len", type=int, default=32000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--fixed-width-bits", type=float, default=6144.0)
    parser.add_argument("--reference-rate", type=float, default=3773.95)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    book = load_codebook(Path(args.codebook))
    codes, state_ids = load_codes(Path(args.pq))
    by_id = {sid: i for i, sid in enumerate(state_ids)}
    split, shared = split_of_state(Path(args.states), Path(args.records))
    row_of, dumped = load_posteriors(Path(args.posteriors), Path(args.records))

    if args.utility_override:
        # two-stage envelope: use the exported max(|U1|, U_cond) vector as the
        # gate score instead of the stage-1 train-label magnitude
        magnitude = np.load(args.utility_override,
                            allow_pickle=True)["utility"].astype(np.float64)
        signed = np.zeros(NUM_POSITIONS)
        counts = np.zeros(NUM_POSITIONS, np.int64)
        print(f"[curve] utility override: {args.utility_override} "
              f"(two-stage envelope)", flush=True)
    else:
        magnitude, signed, counts = fit_magnitude(
            [Path(p) for p in args.fit_labels])
    print(f"[curve] |U| fitted from {int(counts.sum())} labels; positions covered "
          f"{int((counts > 0).sum())}/1024; mean |U| {magnitude.mean():.5f} bits; "
          f"signed-negative fraction {float((signed < 0).mean()):.3f}", flush=True)

    # ------------------------------------------------------------ rate axis
    slot = np.arange(NUM_POSITIONS) // 32
    sub = np.arange(NUM_POSITIONS) % 32
    val_states = sorted(sid for sid, row in by_id.items()
                        if split.get(sid) == "validation" and sid not in shared
                        and sid in row_of)
    val_index = np.asarray([by_id[s] for s in val_states])
    val_posterior = np.asarray([row_of[s] for s in val_states])
    with np.load(args.codebook, allow_pickle=True) as data:
        centroids = np.asarray(data["centroids"], np.float32)
    true_code = codes[val_index][:, slot, sub]
    fill_code = dumped["wm_argmax"][val_posterior][:, slot, sub]
    rate_bits = dumped["code_bits"][val_posterior][:, slot, sub].astype(np.float64)
    entropy = dumped["entropy_bits"][val_posterior][:, slot, sub].astype(np.float64)
    displacement = np.linalg.norm(
        centroids[slot, sub, true_code] - centroids[slot, sub, fill_code],
        axis=-1).mean(0).astype(np.float64)
    mean_rate = rate_bits.mean(0)
    full_send_bits = float(rate_bits.sum(1).mean())
    print(f"[curve] rate axis over {len(val_states)} validation states; full-send "
          f"{full_send_bits:.2f} bits/state", flush=True)

    rng = np.random.default_rng(args.seed)
    scores = {"magnitude": magnitude, "displacement": displacement,
              "rate": mean_rate, "random": rng.random(NUM_POSITIONS)}

    corpus = []
    for lam in args.lambdas:
        keep = magnitude[None, :] >= lam * entropy
        kept = keep.sum(1)
        entry = {"lambda": float(lam),
                 "keep_fraction": float(kept.mean() / NUM_POSITIONS),
                 "rate_bits": float(rate_bits[keep].sum() / len(val_states)),
                 "full_send_bits": full_send_bits}
        entry["saved_fraction"] = 1.0 - entry["rate_bits"] / full_send_bits
        entry["compression_ratio"] = args.fixed_width_bits / entry["rate_bits"]
        corpus.append(entry)

    # --------------------------------------------------------- quality axis
    pairs = list(iter_jsonl(Path(args.pairs)))
    usable = []
    for index, pair in enumerate(pairs):
        state_id = str(pair["state_id"])
        if pair.get("split") != "validation" or state_id in shared:
            continue
        state_row = by_id.get(state_id)
        if state_row is None or state_id not in row_of:
            continue
        usable.append((index, state_row, row_of[state_id]))
    usable = usable[: args.max_rows]

    tokenizer, model = load_ama_reader(
        args.reader_model, args.dtype, args.device, args.device_map)
    bridge, metadata = place_bridge(Path(args.bridge), model)
    from xt_ama_adapter.qwen32_bridge import Qwen32LatentReader
    reader = Qwen32LatentReader(model, tokenizer, bridge,
                                enable_thinking=bool(metadata.get("enable_thinking")))
    print(f"[curve] scoring {len(usable)} evaluation rows; prompt={args.prompt} "
          f"dtype={args.dtype}", flush=True)

    arms_run = [] if "none" in args.arms else [a for a in args.arms if a != "none"]
    if not arms_run:
        print("[curve] control arms disabled (--arms none); scoring "
              "reference + gate only", flush=True)
    records = []
    skipped = {"prompt_too_long": 0, "out_of_memory": 0}
    for done, (index, state_row, posterior) in enumerate(usable):
        pair = pairs[index]
        answer_ids = tokenizer.encode(
            str(pair["answer"]), add_special_tokens=False)[: args.max_answer_tokens]
        if not answer_ids:
            continue
        base = codes[state_row].reshape(-1)
        fill = dumped["wm_argmax"][posterior].reshape(-1)
        state_entropy = dumped["entropy_bits"][posterior].reshape(-1).astype(np.float64)
        state_rate = dumped["code_bits"][posterior].reshape(-1).astype(np.float64)

        variants, meta = [base.copy()], [{"arm": "reference", "lambda": None,
                                          "kept": NUM_POSITIONS,
                                          "rate_bits": float(state_rate.sum())}]
        for lam in args.lambdas:
            keep = magnitude >= lam * state_entropy
            gated = base.copy()
            gated[~keep] = fill[~keep]
            variants.append(gated)
            meta.append({"arm": "gate", "lambda": float(lam), "kept": int(keep.sum()),
                         "rate_bits": float(state_rate[keep].sum())})
            for arm in arms_run:
                order = np.argsort(-scores[arm])
                keep_arm = np.zeros(NUM_POSITIONS, bool)
                keep_arm[order[: int(keep.sum())]] = True
                arm_codes = base.copy()
                arm_codes[~keep_arm] = fill[~keep_arm]
                variants.append(arm_codes)
                meta.append({"arm": arm, "lambda": float(lam),
                             "kept": int(keep_arm.sum()),
                             "rate_bits": float(state_rate[keep_arm].sum())})

        states = rebuild(np.stack([v.reshape(codes[state_row].shape)
                                   for v in variants]), book)
        valid = np.ones(states.shape[1], dtype=bool)
        if args.prompt == "latent+anchor":
            from experiments.state_tokenizer.run_ama_web_latent import (
                build_anchor_block,
            )
            entries = [entry for entry in pair["topk"][: args.anchor_k]
                       if entry.get("has_code_row")]
            texts = step_texts_for(args.cache_dir, str(pair["episode_id"]))
            anchor, _stats = build_anchor_block(
                [(int(entry["step_index"]), texts[int(entry["position"])])
                 for entry in entries], style=args.anchor_style)
            others = [rebuild(codes[by_id[str(entry["state_id"])]][None], book)[0]
                      for entry in entries[1:]]
            scored_states = np.stack(
                [states] + [np.repeat(other[None], len(states), axis=0)
                            for other in others], axis=1)
            inputs, _audit, prefix_len = build_prompt(
                reader, tokenizer, str(pair["question"]), [states[0]] + others,
                anchor, enable_thinking=bool(metadata.get("enable_thinking")),
                max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens)
        else:
            scored_states = states
            inputs, _audit, prefix_len = build_deployed_prompt(
                reader, tokenizer, str(pair["question"]), str(pair["step_text"]),
                states[0], valid,
                enable_thinking=bool(metadata.get("enable_thinking")),
                max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens,
                latent_only=(args.prompt == "latent-only"))

        prompt_tokens = int(inputs["inputs_embeds"].shape[1]) + len(answer_ids)
        if args.max_prompt_tokens and prompt_tokens > args.max_prompt_tokens:
            skipped["prompt_too_long"] += 1
            print(f"[curve] skip row {index}: prompt {prompt_tokens} tokens", flush=True)
            continue

        nll = np.full(len(variants), np.nan)
        try:
            for start in range(0, len(variants), args.positions_per_forward):
                stop = min(start + args.positions_per_forward, len(variants))
                scored = score_variants(
                    model, bridge, inputs["inputs_embeds"], inputs["attention_mask"],
                    prefix_len, answer_ids, scored_states[start:stop])
                nll[start:stop] = scored["nll_bits"]
        except torch.OutOfMemoryError:
            skipped["out_of_memory"] += 1
            torch.cuda.empty_cache()
            print(f"[curve] skip row {index}: OOM at prompt {prompt_tokens} tokens",
                  flush=True)
            continue

        for offset, entry in enumerate(meta[1:], start=1):
            records.append({
                "pair_index": int(index), "arm": entry["arm"],
                "lambda": entry["lambda"], "kept": entry["kept"],
                "rate_bits": entry["rate_bits"],
                "full_rate_bits": float(state_rate.sum()),
                "delta_nll_bits": float(nll[offset] - nll[0]),
                "abs_delta_nll_bits": float(abs(nll[offset] - nll[0])),
                "reference_nll_bits": float(nll[0]),
            })
        if done % 5 == 0:
            print(f"[curve] {done + 1}/{len(usable)}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(records[0])
    np.savez_compressed(out, **{k: np.asarray([r[k] for r in records]) for k in keys})

    def rows_for(arm, lam):
        return [r for r in records if r["arm"] == arm and r["lambda"] == lam]

    report = {"corpus_rate": corpus, "quality": [], "prompt": args.prompt,
              "skipped": skipped, "max_prompt_tokens": args.max_prompt_tokens,
              "fit_mean_abs_u_bits": float(magnitude.mean()),
              "full_send_bits_per_state": full_send_bits,
              "val_states_with_posterior": len(val_states),
              "rows_scored": len(usable),
              "reference_rate_bits_per_transition": args.reference_rate,
              "fixed_width_bits": args.fixed_width_bits}
    for lam in args.lambdas:
        gate = rows_for("gate", lam)
        abs_gate = np.asarray([r["abs_delta_nll_bits"] for r in gate])
        entry = {"lambda": float(lam), "rows": len(gate),
                 "kept_mean": float(np.mean([r["kept"] for r in gate])),
                 "keep_fraction": float(np.mean([r["kept"] for r in gate])
                                        / NUM_POSITIONS),
                 "rate_bits": float(np.mean([r["rate_bits"] for r in gate])),
                 "full_rate_bits": float(np.mean([r["full_rate_bits"] for r in gate])),
                 "abs_delta_nll_bits": float(abs_gate.mean()),
                 "abs_delta_nll_median": float(np.median(abs_gate)),
                 "arms": {}}
        for arm in ARMS:
            other = np.asarray([r["abs_delta_nll_bits"] for r in rows_for(arm, lam)])
            diff = abs_gate - other
            n = diff.size
            std = diff.std(ddof=1) if n > 1 else 0.0
            entry["arms"][arm] = {
                "abs_delta_nll_bits": float(other.mean()) if n else None,
                "paired_t": float(diff.mean() / (std / math.sqrt(n))) if std > 0 else None,
                "mean_gap_bits": float(diff.mean()) if n else None,
                "budget_matched": bool(np.all(
                    [a["kept"] == b["kept"]
                     for a, b in zip(gate, rows_for(arm, lam))]))}
        report["quality"].append(entry)

    (out.with_suffix(".report.json")).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["quality"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
