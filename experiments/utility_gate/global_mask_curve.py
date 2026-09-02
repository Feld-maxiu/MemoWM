"""End-to-end check on a global send mask: what does dropping those bits cost?

A global mask is one fixed vector over the 1024 code positions, shared by every
state. That makes it far weaker than a per-state gate, and far cheaper in two
ways that matter: it needs no per-state prediction, and -- because it ships with
the model as a schema constant -- **it costs zero mask bits**. A per-position
gate that has to transmit its decisions spends 1024 bits, about a fifth of the
code rate, before saving anything.

The scores are fitted on one set of samples and applied to another. Fitting and
evaluating on the same rows would select on noise: ranking 1024 positions by a
quantity measured with ~84 observations each and then reporting the top of that
ranking manufactures a gain that does not transfer. The measured split-half
transfer of the signed per-position mean is about zero, so that arm is expected
to do no better than random -- it is here precisely as the control that shows
what selection on noise looks like when it is done honestly.

Arms, all ranking the 1024 positions and keeping the top fraction:

* ``magnitude`` -- mean |U| over the fit samples. Transfers at 0.88.
* ``signed``    -- mean U. Transfers at ~0. The utility criterion, which the
                   data says is not estimable globally.
* ``insample``  -- mean |U| fitted on the evaluation rows themselves. Not a
                   usable method; the gap to ``magnitude`` is the size of the
                   selection-on-noise effect.
* ``displacement`` -- mean centroid distance. Needs no labels at all, only the
                   codebook, and predicts |U| at Spearman 0.79. If it matches
                   ``magnitude`` then the reader was never needed.
* ``rate``      -- mean code length. The surprise criterion.
* ``random``    -- fixed permutation.

Every arm and budget for one question is scored in a single forward, so the
comparison is free of the batch-geometry effect the scorer's contract warns
about.
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
from experiments.utility_gate.groups import NUM_POSITIONS
from experiments.utility_gate.label_counterfactual import cache_order
from experiments.utility_gate.train_gate_head import load_labels
from experiments.utility_gate.verify_scorer import load_codebook, load_reader, rebuild


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="gate",
                        help="where the states being SCORED live")
    parser.add_argument("--label", default="train")
    parser.add_argument("--records-split", default=None,
                        help="split name inside records.jsonl; defaults to --label")
    parser.add_argument("--fit-gate", default=None,
                        help="where the LABELLED states live, if different from "
                             "--gate. When they differ no sample hold-out is "
                             "needed: the evaluation corpus is out of domain by "
                             "construction")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--fit-pairs", default=None,
                        help="pairs file the labels index into, if not --pairs")
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--keep", type=float, nargs="+",
                        default=[0.70, 0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--pairs-split", default=None,
                        help="evaluate only on this split of the pairs file")
    parser.add_argument("--fit-pairs-split", default=None,
                        help="fit the mask only from labels whose question is in "
                             "this split. Separating the two by question is what "
                             "keeps the Q-Former's own CE_gold targets out of the "
                             "evaluation: on its training questions the latent "
                             "carries 26 bits of state-specific signal, on held-out "
                             "questions of the same corpus 11.4")
    parser.add_argument("--lambdas", type=float, nargs="*", default=None,
                        help="thresholds for |U| >= lambda * entropy")
    parser.add_argument("--max-rows", type=int, default=200)
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
    with np.load(args.posteriors, allow_pickle=True) as data:
        dumped = {k: np.asarray(data[k]) for k in data.files}
    with np.load(args.codebook, allow_pickle=True) as data:
        centroids = np.asarray(data["centroids"], np.float32)

    global_index = cache_order(gate / "records.jsonl",
                               args.records_split or args.label)
    row_of_global = {int(g): r for r, g in enumerate(dumped["target_indices"])}
    posterior_row = np.asarray(
        [row_of_global.get(global_index.get(sid, -1), -1) for sid in state_ids])
    by_id = {sid: i for i, sid in enumerate(state_ids)}
    sample_of = np.asarray([sid.rsplit("-", 1)[0] for sid in state_ids])

    fit_gate = Path(args.fit_gate) if args.fit_gate else gate
    external = fit_gate != gate
    rng = np.random.default_rng(args.seed)
    if external:
        # The scored corpus never contributed a label, so every row of it is
        # held out already and splitting it again would only shrink the sample.
        held = set(np.unique(sample_of))
        with np.load(fit_gate / "codes.npz", allow_pickle=True) as data:
            fit_state_ids = [str(v) for v in np.asarray(data["state_ids/train"])]
        fit_sample_of = np.asarray([s.rsplit("-", 1)[0] for s in fit_state_ids])
    elif args.pairs_split:
        # The pairs file already splits by sample, and its split is the one that
        # matters: those samples' questions were never CE_gold targets. Using it
        # as the hold-out makes the evaluation disjoint from the mask fit in both
        # question and sample, and keeps all of it rather than intersecting with
        # a second random split.
        eval_pairs = np.load(args.pairs, allow_pickle=True)
        eval_split = np.asarray([str(v) for v in eval_pairs["split"]])
        held = {f"{s}" for s, v in zip(eval_pairs["sample_id"], eval_split)
                if v == args.pairs_split}
        fit_sample_of = sample_of
    else:
        samples = np.unique(sample_of)
        held = set(rng.permutation(samples)[: max(1, len(samples) // 4)])
        fit_sample_of = sample_of

    fit_pairs = np.load(args.fit_pairs or args.pairs, allow_pickle=True)
    fit_split = ([str(v) for v in fit_pairs["split"]]
                 if "split" in fit_pairs.files else None)

    table = load_labels(args.labels)
    label_sample = np.asarray([fit_sample_of[s] for s in table["state_row"]])
    fit = np.ones(len(label_sample), bool) if external else ~np.isin(
        label_sample, list(held))
    if args.fit_pairs_split and fit_split is not None:
        by_question = np.asarray(
            [fit_split[int(r)] == args.fit_pairs_split for r in table["pair_row"]])
        fit &= by_question
    print(f"[curve] mask fitted from {int(fit.sum())} of {len(fit)} labels",
          flush=True)
    position = table["position"][fit]
    utility = table["delta_nll_bits"][fit].astype(np.float64)

    def per_position(values, reducer):
        out = np.zeros(NUM_POSITIONS)
        for p in range(NUM_POSITIONS):
            block = values[position == p]
            out[p] = reducer(block) if len(block) else 0.0
        return out

    # Displacement and rate need no labels: they are averaged over the states of
    # the fit samples, straight from the codebook and the posterior dump.
    fit_states = [i for i, sid in enumerate(state_ids)
                  if (external or sample_of[i] not in held) and posterior_row[i] >= 0]
    fit_states = np.asarray(fit_states[:2000])
    rows = posterior_row[fit_states]
    slot = np.arange(NUM_POSITIONS) // 32
    sub = np.arange(NUM_POSITIONS) % 32
    true_code = codes[fit_states][:, slot, sub]
    fill_code = dumped["wm_argmax"][rows][:, slot, sub]
    displacement = np.linalg.norm(
        centroids[slot, sub, true_code] - centroids[slot, sub, fill_code],
        axis=-1).mean(0).astype(np.float64)
    mean_rate = dumped["code_bits"][rows][:, slot, sub].mean(0).astype(np.float64)

    magnitude = per_position(np.abs(utility), np.mean)
    scores = {
        "magnitude": magnitude,
        # The criterion the objective actually names: utility per bit. Ranking by
        # |U| alone ignores that positions differ in what they cost -- p1 to p99
        # of the code length spans 1.3 to 8.8 bits -- so two positions of equal
        # impact are not equally worth keeping.
        "density": magnitude / np.maximum(mean_rate, 1e-6),
        "displacement": displacement,
        "rate": mean_rate,
        "random": rng.random(NUM_POSITIONS),
    }
    all_position = table["position"]
    all_utility = np.abs(table["delta_nll_bits"].astype(np.float64))
    scores["insample"] = np.asarray([
        all_utility[all_position == p].mean() if (all_position == p).any() else 0.0
        for p in range(NUM_POSITIONS)])

    masks = {}
    for arm, score in scores.items():
        order = np.argsort(-score)
        for keep in args.keep:
            mask = np.zeros(NUM_POSITIONS, bool)
            mask[order[: int(round(keep * NUM_POSITIONS))]] = True
            masks[f"{arm}@{keep}"] = mask

    # The Lagrangian form, kept per state rather than per corpus: drop j when
    # |U_j| < lambda * H_j, with |U| a constant shipped with the model and H the
    # posterior entropy at this state.
    #
    # The entropy is the part that has to be decoder-side. The obvious choice --
    # the actual code length -log2 p(z+_j) -- is not available to the decoder,
    # because it is a function of the very symbol being decided about. Entropy is
    # what the decoder can compute from h and u alone, and it is the expected
    # cost of the position, so the trade it expresses is the intended one and it
    # costs no mask bits.
    lambdas = args.lambdas or []

    pairs = np.load(args.pairs, allow_pickle=True)
    keys = [f"{s}-{int(r):04d}" for s, r in zip(pairs["sample_id"], pairs["record_index"])]
    split = ([str(v) for v in pairs["split"]] if "split" in pairs.files
             else [""] * len(keys))
    usable = [i for i, key in enumerate(keys)
              if key in by_id and sample_of[by_id[key]] in held
              and posterior_row[by_id[key]] >= 0
              and (args.pairs_split is None or split[i] == args.pairs_split)]
    chooser = np.random.default_rng(args.seed + 2)
    usable = list(chooser.permutation(usable))[args.shard_index::args.shard_count]
    usable = usable[: args.max_rows]
    print(f"[curve] {len(usable)} held-out rows, {len(masks)} arm-budget pairs",
          flush=True)

    device = torch.device(args.device)
    processor, model, connector = load_reader(
        args.model, args.checkpoint, device, torch.float32, codes.shape[1])

    records = []
    for done, pair_row in enumerate(usable):
        state = by_id[keys[pair_row]]
        row = posterior_row[state]
        rate = dumped["code_bits"][row].reshape(-1).astype(np.float64)
        fill = dumped["wm_argmax"][row].reshape(-1)
        base = codes[state].reshape(-1)

        entropy = dumped["entropy_bits"][row].reshape(-1).astype(np.float64)
        state_masks = dict(masks)
        for lam in lambdas:
            state_masks[f"threshold@{lam}"] = scores["magnitude"] >= lam * entropy

        variants = [rebuild(codes[state][None], book)]
        names = ["reference"]
        for name, mask in state_masks.items():
            gated = base.copy()
            gated[~mask] = fill[~mask]
            variants.append(rebuild(gated.reshape(codes[state].shape)[None], book))
            names.append(name)
        variants.append(xbar[state][None])
        names.append("raw_xbar")

        batch = torch.as_tensor(
            np.concatenate(variants).astype(np.float32), device=device)
        with torch.no_grad():
            soft = connector(batch, torch.ones(
                batch.shape[:2], dtype=torch.bool, device=device))
        scored = answer_variant_scores(
            model, processor, soft, str(pairs["question"][pair_row]),
            str(pairs["answer"][pair_row]), max_answer_tokens=FULL_ANSWER_TOKENS)
        nll = scored["nll_bits"].numpy()
        for offset, name in enumerate(names[1:], start=1):
            if name == "raw_xbar":
                continue
            mask = state_masks[name]
            records.append({
                "pair_row": int(pair_row), "arm": name.split("@")[0],
                "keep": float(name.split("@")[1]),
                "kept_positions": int(mask.sum()),
                "rate_bits": float(rate[mask].sum()),
                "full_rate_bits": float(rate.sum()),
                "delta_nll_bits": float(nll[offset] - nll[0]),
            })
        if done % 20 == 0:
            print(f"[curve] {done + 1}/{len(usable)}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{
        key: np.asarray([r[key] for r in records])
        for key in records[0]}, metadata=json.dumps({
            "protocol": "residualmem_utility_gate_global_mask_v1",
            "held_out_samples": len(held), "rows": len(usable),
            "keep": args.keep, "labels": list(args.labels),
            "mask_bits": 0, "note": "global mask is a schema constant",
        }))
    print(f"wrote {args.output}  ({len(records)} rows)")


if __name__ == "__main__":
    main()
