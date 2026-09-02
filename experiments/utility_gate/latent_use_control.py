"""Does the latent carry anything for this question, or is the reader ignoring it?

Every utility number in this work is a difference between two memories, and a
difference is only meaningful if the memory was being used. The control is the
one the repo already uses for screen fidelity: score a question against its own
observation's latent, and against a different observation's, and compare.

If the two are the same, the reader is not reading. A perturbation study on top
of that measures the reader's sensitivity to noise in a channel it ignores, and
every arm of a gating curve becomes a lottery -- which is what an inverted,
non-monotonic curve looks like from the outside.

Reported per corpus so a domain shift shows up as what it is. The matched and
mismatched conditions ride in the same forward, so the comparison is free of the
batch-geometry effect.
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
from experiments.utility_gate.verify_scorer import load_codebook, load_reader, rebuild


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--label", default="train")
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--pairs-split", default=None,
                        help="restrict to this split of the pairs file. The fit "
                             "corpus questions were the Q-Former's own CE_gold "
                             "targets, so a gap measured over all of them cannot "
                             "separate representation from memorisation")
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    gate = Path(args.gate)
    book = load_codebook(Path(args.codebook))
    with np.load(gate / "codes.npz", allow_pickle=True) as data:
        codes = np.asarray(data[f"codes/{args.label}"], np.uint8)
        state_ids = [str(v) for v in np.asarray(data[f"state_ids/{args.label}"])]
    with np.load(gate / "states.npz", allow_pickle=True) as data:
        xbar = np.asarray(data["xbar"], np.float32)
    by_id = {sid: i for i, sid in enumerate(state_ids)}

    pairs = np.load(args.pairs, allow_pickle=True)
    keys = [f"{s}-{int(r):04d}" for s, r in zip(pairs["sample_id"], pairs["record_index"])]
    split = ([str(v) for v in pairs["split"]] if "split" in pairs.files
             else [""] * len(keys))
    usable = [i for i, key in enumerate(keys)
              if key in by_id and (args.pairs_split is None
                                   or split[i] == args.pairs_split)]
    if not usable:
        raise SystemExit(f"no rows with split == {args.pairs_split!r}")
    rng = np.random.default_rng(args.seed)
    chosen = rng.permutation(usable)[: args.rows]

    device = torch.device(args.device)
    processor, model, connector = load_reader(
        args.model, args.checkpoint, device, torch.float32, codes.shape[1])

    matched, mismatched, empty = [], [], []
    for done, pair_row in enumerate(chosen):
        state = by_id[keys[int(pair_row)]]
        other = state
        while other == state:
            other = int(rng.integers(len(state_ids)))
        variants = np.concatenate([
            rebuild(codes[state][None], book),
            rebuild(codes[other][None], book),
            np.zeros((1, codes.shape[1], xbar.shape[-1]), np.float32),
        ]).astype(np.float32)
        latents = torch.as_tensor(variants, device=device, dtype=torch.float32)
        with torch.no_grad():
            soft = connector(latents, torch.ones(
                latents.shape[:2], dtype=torch.bool, device=device))
        scored = answer_variant_scores(
            model, processor, soft, str(pairs["question"][int(pair_row)]),
            str(pairs["answer"][int(pair_row)]), max_answer_tokens=FULL_ANSWER_TOKENS)
        nll = scored["nll_bits"].numpy()
        matched.append(float(nll[0]))
        mismatched.append(float(nll[1]))
        empty.append(float(nll[2]))
        if done % 25 == 0:
            print(f"[control] {done + 1}/{len(chosen)}", flush=True)

    matched = np.asarray(matched); mismatched = np.asarray(mismatched)
    empty = np.asarray(empty)
    report = {
        "gate": str(gate), "pairs": str(args.pairs),
        "pairs_split": args.pairs_split, "rows": int(len(matched)),
        "matched_nll_bits": float(matched.mean()),
        "mismatched_nll_bits": float(mismatched.mean()),
        "zeroed_nll_bits": float(empty.mean()),
        # The quantity that matters: how many bits worse the answer gets when the
        # memory belongs to a different observation. Zero means the reader is not
        # using it, and every utility measured on this corpus is noise.
        "gap_mismatched_minus_matched": float((mismatched - matched).mean()),
        "gap_median": float(np.median(mismatched - matched)),
        "matched_beats_mismatched": float((matched < mismatched).mean()),
        "gap_zeroed_minus_matched": float((empty - matched).mean()),
        "zeroed_beats_matched": float((empty < matched).mean()),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
