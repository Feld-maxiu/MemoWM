"""Label each code position with what omitting it costs the reader.

For one question and one observation, the reference memory is the fully
transmitted state and each variant replaces a single position with the estimate a
decoder would fall back to. Scoring all of them in one forward gives, per
position, the KL the reader's gold-answer distribution moves and the change in
the gold answer's own NLL.

Three choices here are load-bearing.

**Positions are sampled per state, not per row.** A head learns
``f(position features) -> utility``, so it does not need every position of every
row labelled -- but the variance decomposition does need the *same* (state,
position) labelled under more than one question. Sampling independently per row
would give two questions of the same state an expected overlap of 32/1024, about
3%, and the "same state, different question" term -- the one that bounds what any
write-time gate can achieve -- would have almost no paired data to estimate from.

**Only states the world model can predict are labelled.** ``cache_web`` emits a
transition per non-initial state, so episode-initial observations have no
posterior and therefore no principled fill. They are skipped rather than filled
some other way, because mixing two fills inside one label table would make the
utilities incomparable. The count is reported.

**The reference is the quantised state, not the raw xbar.** Both sides of every
difference are then in the same space and the quantiser's own error cancels. The
raw row is still scored, as its own variant, because the gap between it and the
reference is what the quantiser costs in answer NLL -- which the report lists as
unmeasured.

Runs under the torch interpreter. Shardable with ``--shard-index/--shard-count``;
each shard writes its own npz and they concatenate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.state_tokenizer.qformer_pq import decode
from experiments.state_tokenizer.reader_losses import (
    FULL_ANSWER_TOKENS, answer_variant_scores,
)
from experiments.utility_gate.groups import (
    NUM_POSITIONS, NUM_SLOTS, NUM_SUBSPACES, slot_of, subspace_of,
)
from experiments.utility_gate.verify_scorer import load_codebook, load_reader, rebuild


def cache_order(records_path: Path, split: str) -> dict[str, int]:
    """state_id -> global_index, reproducing ``cache_web``'s ordering.

    ``cache_web`` sorts the records by ``(episode_id, step)`` and enumerates; the
    cache keeps only that order, never the ids. Anything joining a per-transition
    quantity back to a state has to redo the sort rather than guess.
    """
    records = [json.loads(line) for line in
               records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if r["split"] == split]
    records.sort(key=lambda r: (r["episode_id"], int(r["step"])))
    return {str(r["state_id"]): index for index, r in enumerate(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="gate", help="the gate cache directory")
    parser.add_argument("--label", default="train")
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--codebook-coarse", default=None)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--posteriors", default=None,
                        help="dump_wm_posteriors output. Without it the fill is "
                             "the corpus mode and no rate is recorded -- usable "
                             "for a pilot, not for the rate-utility curve")
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--positions", type=int, default=32,
                        help="positions sampled per state, shared by its questions")
    parser.add_argument("--slots", type=int, default=0,
                        help="instead of scattering positions, sample this many "
                             "whole slots per state and label all 32 subspaces of "
                             "each. Scattered positions cannot be aggregated back "
                             "into a slot -- 32 of 1024 leaves about one per slot "
                             "-- and the per-position utility is too noisy to "
                             "predict on its own (split-half reliability 0.09), so "
                             "measuring whether a slot-level gate has a target at "
                             "all needs the subspaces of one slot together.")
    parser.add_argument("--complete", action="store_true",
                        help="label all 1024 positions instead of sampling; for "
                             "the oracle curve and the additivity check")
    parser.add_argument("--chunk", type=int, default=34,
                        help="ablations per forward when --complete. The batch "
                             "geometry stays fixed across chunks; float32 makes "
                             "that safe (measured 1.5e-4 bits)")
    parser.add_argument("--additivity", action="store_true",
                        help="also ablate nested subsets of the sampled "
                             "positions, so U(set) can be compared against the "
                             "sum of its singles (D5)")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gate = Path(args.gate)
    book = load_codebook(Path(args.codebook))
    with np.load(gate / "codes.npz", allow_pickle=True) as data:
        codes = np.asarray(data[f"codes/{args.label}"], np.uint8)
        state_ids = [str(v) for v in np.asarray(data[f"state_ids/{args.label}"])]
    with np.load(gate / "states.npz", allow_pickle=True) as data:
        xbar = np.asarray(data["xbar"], np.float32)
        if [str(v) for v in np.asarray(data["state_ids"])] != state_ids:
            raise SystemExit("states.npz and codes.npz disagree on row order")

    coarse_codes = coarse_book = None
    coarse_path = gate / "codes-coarse.npz"
    if args.codebook_coarse and coarse_path.exists():
        coarse_book = load_codebook(Path(args.codebook_coarse))
        with np.load(coarse_path, allow_pickle=True) as data:
            coarse_codes = np.asarray(data[f"codes/{args.label}"], np.uint8)

    by_id = {sid: i for i, sid in enumerate(state_ids)}
    fill = np.zeros_like(codes)
    rate = np.full(codes.shape, np.nan, np.float32)
    has_posterior = np.zeros(len(codes), bool)
    if args.posteriors:
        with np.load(args.posteriors, allow_pickle=True) as data:
            wm_argmax = np.asarray(data["wm_argmax"], np.uint8)
            code_bits = np.asarray(data["code_bits"], np.float32)
            target_indices = np.asarray(data["target_indices"], np.int64)
        global_index = cache_order(gate / "records.jsonl", args.label)
        row_of_global = {int(g): r for r, g in enumerate(target_indices)}
        for sid, row in by_id.items():
            posterior = row_of_global.get(global_index.get(sid, -1), None)
            if posterior is None:
                continue
            fill[row] = wm_argmax[posterior]
            rate[row] = code_bits[posterior]
            has_posterior[row] = True
        source = "world_model_argmax"
    else:
        # The corpus mode: what an unconditional codec would fall back to. Carries
        # no state-specific information, so it destroys more than the world
        # model's mode would and the |U| it gives is an upper bound.
        for slot in range(codes.shape[1]):
            for subspace in range(codes.shape[2]):
                fill[:, slot, subspace] = np.bincount(
                    codes[:, slot, subspace], minlength=256).argmax()
        has_posterior[:] = True
        source = "corpus_mode"

    pairs = np.load(args.pairs, allow_pickle=True)
    keys = [f"{s}-{int(r):04d}" for s, r in zip(pairs["sample_id"], pairs["record_index"])]
    usable = [i for i, key in enumerate(keys)
              if key in by_id and has_posterior[by_id[key]]]
    skipped = len(keys) - len(usable)
    rng = np.random.default_rng(args.seed)

    # One sample of positions per state, shared by every question about it.
    per_state: dict[int, np.ndarray] = {}
    for row in sorted({by_id[keys[i]] for i in usable}):
        if args.complete:
            per_state[row] = np.arange(NUM_POSITIONS)
        elif args.slots:
            picked = rng.permutation(NUM_SLOTS)[: args.slots]
            per_state[row] = np.sort(np.concatenate(
                [np.arange(int(slot) * NUM_SUBSPACES,
                           (int(slot) + 1) * NUM_SUBSPACES) for slot in picked]))
        else:
            per_state[row] = np.sort(
                rng.permutation(NUM_POSITIONS)[: args.positions])

    rows = usable[args.shard_index::args.shard_count]
    # ☠️ Subsample whole states, never loose rows. Two failure modes bracket
    # this, and both were hit:
    #   * a prefix of ``usable`` follows the pairs file, which is ordered by
    #     sample -- 400 rows turned out to be 9 samples of 154, enough to fit a
    #     head and nowhere near enough to hold one out;
    #   * a random sample of *rows* fixes the coverage but scatters, so a state
    #     rarely gets two of its questions labelled and the "same cell, different
    #     question" variance -- the term that bounds what any write-time gate can
    #     achieve -- has almost no pairs to be estimated from.
    # Taking every question of a randomly chosen state gives both.
    if args.max_rows:
        by_state: dict[int, list[int]] = {}
        for row in rows:
            by_state.setdefault(by_id[keys[row]], []).append(row)
        chooser = np.random.default_rng(args.seed + 1)
        chosen, budget = [], args.max_rows
        for state in chooser.permutation(sorted(by_state)):
            block = by_state[int(state)]
            if len(chosen) + len(block) > budget and chosen:
                continue
            chosen.extend(block)
            if len(chosen) >= budget:
                break
        rows = sorted(chosen)
    covered = len({keys[r].rsplit("-", 1)[0] for r in rows})
    states_covered = len({by_id[keys[r]] for r in rows})
    print(f"[label] {len(rows)} rows of {len(usable)} usable "
          f"({skipped} skipped: no state or no posterior), fill={source}, "
          f"covering {states_covered} states in {covered} samples", flush=True)

    subset_sizes: list[int] = []
    if args.additivity and not args.complete:
        size = 2
        while size <= args.positions:
            subset_sizes.append(size)
            size *= 2
    elif args.additivity:
        raise SystemExit("--additivity needs the sampled mode, not --complete")

    device = torch.device(args.device)
    processor, model, connector = load_reader(
        args.model, args.checkpoint, device, torch.float32, codes.shape[1])

    out: dict[str, list] = {k: [] for k in (
        "pair_row", "state_row", "position", "kl_bits", "delta_nll_bits",
        "abs_delta_nll_bits", "effective_tokens", "top1_token_share",
        "argmax_changed", "rate_bits")}
    per_row: dict[str, list] = {k: [] for k in (
        "row_pair_row", "row_state_row", "row_reference_nll_bits",
        "row_raw_nll_bits", "row_coarse_nll_bits", "row_answer_len",
        "row_question_len")}
    subsets: dict[str, list] = {k: [] for k in (
        "subset_pair_row", "subset_size", "subset_delta_nll_bits",
        "subset_sum_of_singles_bits", "subset_kl_bits")}

    started = time.time()
    for done, pair_row in enumerate(rows):
        state = by_id[keys[pair_row]]
        positions = per_state[state]
        question = str(pairs["question"][pair_row])
        answer = str(pairs["answer"][pair_row])

        reference_nll = raw_nll = coarse_nll = float("nan")
        # Every forward carries exactly ``chunk + 1`` rows. The first one spends
        # two of its slots on the per-row extras, so it takes fewer positions --
        # otherwise chunk 0 would be wider than the rest and the batch geometry
        # would vary within a single row, which is the one thing the scorer's
        # contract asks callers not to do.
        extras = 1 + (1 if coarse_codes is not None else 0)
        blocks, cursor = [], 0
        while cursor < len(positions):
            width = args.chunk - (extras if not blocks else 0)
            blocks.append(positions[cursor:cursor + width])
            cursor += width
        for start, block in enumerate(blocks):
            stacked = np.repeat(codes[state][None], len(block) + 1, axis=0)
            for offset, position in enumerate(block, start=1):
                slot, subspace = slot_of(int(position)), subspace_of(int(position))
                stacked[offset, slot, subspace] = fill[state, slot, subspace]
            variants = [rebuild(stacked, book)]
            names = ["reference"] + [f"position:{int(p)}" for p in block]
            # The extras ride along in the first chunk only: they are per-row
            # quantities, and scoring them there keeps every comparison inside a
            # single forward with the reference.
            if start == 0:
                variants.append(xbar[state][None])
                names.append("raw_xbar")
                if coarse_codes is not None:
                    variants.append(rebuild(coarse_codes[state][None], coarse_book))
                    names.append("coarse_quantised")
                # Nested subsets, for the additivity check. The report warns that
                # per-group utilities are not guaranteed to add (§6.3), which
                # makes rho a ranking signal rather than an optimality claim.
                # Nesting rather than sampling means the sum of the singles over
                # a subset is directly the prediction being tested.
                if subset_sizes:
                    for size in subset_sizes:
                        chosen = block[:size]
                        multi = codes[state].copy()
                        for position in chosen:
                            s, m = slot_of(int(position)), subspace_of(int(position))
                            multi[s, m] = fill[state, s, m]
                        variants.append(rebuild(multi[None], book))
                        names.append(f"subset:{size}")
            batch = np.concatenate(variants).astype(np.float32)
            latents = torch.as_tensor(batch, device=device, dtype=torch.float32)
            with torch.no_grad():
                soft = connector(latents, torch.ones(
                    latents.shape[:2], dtype=torch.bool, device=device))
            scored = answer_variant_scores(
                model, processor, soft, question, answer,
                max_answer_tokens=FULL_ANSWER_TOKENS)

            nll = scored["nll_bits"].numpy()
            kl = scored["kl_bits"].numpy()
            token_nll = scored["token_nll_bits"].numpy()
            argmax = scored["teacher_forced_argmax"].numpy()
            reference_nll = float(nll[0])
            if start == 0:
                raw_nll = float(nll[names.index("raw_xbar")])
                if "coarse_quantised" in names:
                    coarse_nll = float(nll[names.index("coarse_quantised")])
                per_row["row_pair_row"].append(pair_row)
                per_row["row_state_row"].append(state)
                per_row["row_reference_nll_bits"].append(reference_nll)
                per_row["row_raw_nll_bits"].append(raw_nll)
                per_row["row_coarse_nll_bits"].append(coarse_nll)
                per_row["row_answer_len"].append(scored["answer_len"])
                per_row["row_question_len"].append(scored["question_len"])

            for offset, position in enumerate(block, start=1):
                slot, subspace = slot_of(int(position)), subspace_of(int(position))
                # How concentrated is the disturbance across answer tokens? The
                # participation ratio is ~1 when a single token absorbs it and
                # ~answer_len when it is spread evenly. If it is localized, the
                # summed delta does not scale with answer length and the sum is
                # the quantity commensurate with the code bits; if it is diffuse,
                # the sum carries a length scale and the per-token mean would be
                # the fairer label. Measured rather than assumed.
                spread = np.abs(token_nll[offset] - token_nll[0])
                total = float(spread.sum())
                square = float((spread ** 2).sum())
                out["pair_row"].append(pair_row)
                out["state_row"].append(state)
                out["position"].append(int(position))
                out["kl_bits"].append(float(kl[offset]))
                out["delta_nll_bits"].append(float(nll[offset] - nll[0]))
                out["abs_delta_nll_bits"].append(total)
                out["effective_tokens"].append(
                    float(total ** 2 / square) if square > 0 else 0.0)
                out["top1_token_share"].append(
                    float(spread.max() / total) if total > 0 else 0.0)
                out["argmax_changed"].append(
                    bool((argmax[offset] != argmax[0]).any()))
                out["rate_bits"].append(float(rate[state, slot, subspace]))

            if start == 0 and subset_sizes:
                singles = {int(p): float(nll[o] - nll[0])
                           for o, p in enumerate(block, start=1)}
                for size in subset_sizes:
                    row = names.index(f"subset:{size}")
                    subsets["subset_pair_row"].append(pair_row)
                    subsets["subset_size"].append(size)
                    subsets["subset_delta_nll_bits"].append(float(nll[row] - nll[0]))
                    subsets["subset_sum_of_singles_bits"].append(
                        float(sum(singles[int(p)] for p in block[:size])))
                    subsets["subset_kl_bits"].append(float(kl[row]))

        if done % 25 == 0:
            rate_per_row = (time.time() - started) / max(done + 1, 1)
            print(f"[label] {done + 1}/{len(rows)}  {rate_per_row:.2f}s/row  "
                  f"eta {(len(rows) - done - 1) * rate_per_row / 60:.1f} min",
                  flush=True)

    dumped = {k: np.asarray(v) for k, v in out.items()}
    dumped.update({k: np.asarray(v) for k, v in per_row.items()})
    dumped.update({k: np.asarray(v) for k, v in subsets.items()})
    dumped["state_ids"] = np.asarray(state_ids, dtype=object)
    dumped["metadata"] = json.dumps({
        "protocol": "residualmem_utility_gate_labels_v1",
        "fill": source, "positions_per_state": int(args.positions),
        "slots_per_state": int(args.slots),
        "complete": bool(args.complete), "seed": args.seed,
        "shard": [args.shard_index, args.shard_count],
        "rows": len(rows), "usable": len(usable), "skipped": skipped,
        "codebook": str(Path(args.codebook).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "posteriors": args.posteriors,
        "max_answer_tokens": FULL_ANSWER_TOKENS,
        "dtype": "float32",
    }, ensure_ascii=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dumped)
    delta = dumped["delta_nll_bits"]
    kl = dumped["kl_bits"]
    print(json.dumps({
        "rows": len(rows), "labels": int(len(delta)),
        "delta_nll_bits": {"mean": float(delta.mean()),
                           "abs_median": float(np.median(np.abs(delta))),
                           "nonzero": float((delta != 0).mean())},
        "kl_bits": {"mean": float(kl.mean()), "median": float(np.median(kl)),
                    "nonzero": float((kl != 0).mean())},
        "argmax_changed": float(dumped["argmax_changed"].mean()),
        "effective_tokens": {
            "median": float(np.median(dumped["effective_tokens"])),
            "mean": float(dumped["effective_tokens"].mean()),
            "answer_len_median": float(np.median(dumped["row_answer_len"])),
        },
        "top1_token_share_median": float(np.median(dumped["top1_token_share"])),
        "seconds_per_row": round((time.time() - started) / max(len(rows), 1), 3),
    }, indent=2))


if __name__ == "__main__":
    main()
