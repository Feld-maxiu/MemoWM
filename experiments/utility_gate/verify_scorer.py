"""Preflight for the counterfactual scorer: does the measurement hold up at all.

Four questions, none of which need the world model, all of which decide whether
the labelling pass is worth running.

**Is the pipeline wired correctly?** ``--null`` substitutes each position with
its own true code. Every variant is then bit-identical to the reference, so the
KL must be exactly zero and the NLL must match to the last bit. Anything else
means a position index, a decode or a batch row is crossed, and no amount of
downstream care would recover it. This is the cheapest test that can fail.

**Does the arithmetic survive the dtype?** A second-hand measurement claimed the
single-slot signal is ~0.85 nats while bfloat16 injects ~0.64 nats of batch-order
error, with bf16 and fp32 disagreeing on the *sign* of the mean delta. If that
holds, bf16 labels would be noise. ``--dtype`` scores the same variants under
each so the comparison is first-hand.

**How big is the signal, really?** The provenance of that 0.85 was never
established -- most likely it perturbed with random vectors rather than a
decoder's estimate, which would make it meaningless as a threshold. Here the fill
is the corpus mode: the estimate an unconditional codec would use. It is the
world-model-free stand-in, and because a mode carries no state-specific
information it is close to the most destructive fill available, so the |U| it
produces is an upper bound on what the world model's mode will give.

**Does it fit?** fp32 weights are 36 GB and the answer-position logits are 3.65 GB
a copy, so peak memory is measured rather than asserted.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.state_tokenizer.qformer_pq import decode, encode
from experiments.state_tokenizer.reader_losses import answer_variant_scores
from experiments.utility_gate.groups import NUM_POSITIONS, slot_of, subspace_of

LN2 = math.log(2.0)


def load_codebook(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        return {
            "mean": np.asarray(data["mean"]),
            "scale": np.asarray(data["scale"]),
            "centroids": np.asarray(data["centroids"]),
            "bases": np.asarray(data["rotation_bases"]) if "rotation_bases" in data else None,
            "order": np.asarray(data["rotation_order"]) if "rotation_order" in data else None,
        }


def rebuild(codes: np.ndarray, book: dict) -> np.ndarray:
    return decode(codes, book["mean"], book["scale"], book["centroids"],
                  bases=book["bases"], order=book["order"])


def load_reader(model_path: str, checkpoint: str, device: torch.device,
                dtype: torch.dtype, slots: int):
    """The 9B and the connector, without the Q-Former or the retrieval head.

    ``xbar`` is precomputed, so the resampler and the trunk are dead weight here;
    only the slot-wise projection into the reader's embedding space is needed.
    Loading the connector directly also lets the dtype be chosen, which
    ``QFormerInstructTokenizer`` does not expose.
    """
    from experiments.state_tokenizer.extract_qwen import _load_model
    from residualmem.latent.instruct_bridge import InputSoftTokenConnector

    processor, model = _load_model(model_path, device, False, dtype=dtype)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    prefix = "connector."
    weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not weights:
        raise SystemExit(f"{checkpoint} carries no connector weights")
    connector = InputSoftTokenConnector(slots=slots)
    connector.load_state_dict(weights, strict=True)
    # The connector stays float32 whatever the reader is. It is 77 MB against the
    # reader's 35 GB, so the memory is irrelevant, and keeping the projection
    # exact means a dtype sweep varies only the thing being swept --
    # ``answer_variant_scores`` casts the soft tokens to the reader's dtype at the
    # boundary.
    return processor, model, connector.to(device=device, dtype=torch.float32).eval()


def build_variants(xbar: np.ndarray, codes: np.ndarray, fill: np.ndarray,
                   positions: list[int], book: dict,
                   coarse: tuple[np.ndarray, dict] | None) -> tuple[np.ndarray, list[str]]:
    """Reference, one substitution per position, raw xbar, and the coarse arm.

    The reference is ``decode(codes)`` rather than the raw xbar on purpose: both
    sides of every difference are then in the quantised space, so quantisation
    error cancels and what is left is the cost of the omission itself. The raw
    row is carried anyway, as its own variant, because the gap between it and the
    reference is the quantiser's own price in answer NLL -- a number the report
    lists as unmeasured.
    """
    stacked = np.repeat(codes[None], len(positions) + 1, axis=0)
    for row, position in enumerate(positions, start=1):
        slot, subspace = slot_of(position), subspace_of(position)
        stacked[row, slot, subspace] = fill[slot, subspace]
    variants = [rebuild(stacked, book)]
    names = ["reference"] + [f"position:{p}" for p in positions]

    variants.append(xbar[None].astype(np.float32))
    names.append("raw_xbar")
    if coarse is not None:
        coarse_row, coarse_book = coarse
        variants.append(rebuild(coarse_row[None], coarse_book))
        names.append("coarse_quantised")
    return np.concatenate(variants).astype(np.float32), names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", default="gate/states.npz")
    parser.add_argument("--codes", default="gate/codes.npz")
    parser.add_argument("--codes-coarse", default="gate/codes-coarse.npz")
    parser.add_argument("--label", default="train")
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--codebook-coarse", default=None)
    parser.add_argument("--pairs", required=True, help="qformer-qa-pairs.npz")
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--checkpoint", required=True, help="Q-Former checkpoint")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "both"),
                        default="both")
    parser.add_argument("--questions", type=int, default=4)
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--null", action="store_true",
                        help="fill each position with its own true code; every "
                             "variant then equals the reference and U must be 0")
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    book = load_codebook(Path(args.codebook))
    with np.load(args.codes, allow_pickle=True) as data:
        codes = np.asarray(data[f"codes/{args.label}"], np.uint8)
        state_ids = [str(v) for v in np.asarray(data[f"state_ids/{args.label}"])]
    with np.load(args.states, allow_pickle=True) as data:
        xbar = np.asarray(data["xbar"], np.float32)
        assert [str(v) for v in np.asarray(data["state_ids"])] == state_ids

    coarse_codes = coarse_book = None
    if args.codebook_coarse and Path(args.codes_coarse).exists():
        coarse_book = load_codebook(Path(args.codebook_coarse))
        with np.load(args.codes_coarse, allow_pickle=True) as data:
            coarse_codes = np.asarray(data[f"codes/{args.label}"], np.uint8)

    # The corpus mode per position: the estimate an unconditional codec falls
    # back to. Stands in for the world model's mode, which is not available yet.
    modes = np.zeros(codes.shape[1:], np.uint8)
    for slot in range(codes.shape[1]):
        for subspace in range(codes.shape[2]):
            modes[slot, subspace] = np.bincount(
                codes[:, slot, subspace], minlength=256).argmax()

    pairs = np.load(args.pairs, allow_pickle=True)
    index = {sid: i for i, sid in enumerate(state_ids)}
    keys = [f"{s}-{int(r):04d}" for s, r in zip(pairs["sample_id"], pairs["record_index"])]
    usable = [i for i, key in enumerate(keys) if key in index]
    rng = np.random.default_rng(args.seed)
    chosen = rng.permutation(usable)[: args.questions]
    positions = sorted(rng.permutation(NUM_POSITIONS)[: args.positions].tolist())

    dtypes = ([torch.float32, torch.bfloat16] if args.dtype == "both"
              else [getattr(torch, args.dtype)])
    device = torch.device(args.device)
    report: dict = {"positions": positions, "null": args.null,
                    "questions": [], "by_dtype": {}}

    def coarse_for(state: int):
        return None if coarse_codes is None else (coarse_codes[state], coarse_book)

    for dtype in dtypes:
        # reset_peak_memory_stats needs a live context; the first allocation is
        # what creates one.
        torch.zeros(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        processor, model, connector = load_reader(
            args.model, args.checkpoint, device, dtype, codes.shape[1])
        weights_gb = torch.cuda.max_memory_allocated(device) / 2**30

        rows, elapsed = [], 0.0
        for pair_row in chosen:
            state = index[keys[int(pair_row)]]
            fill = codes[state] if args.null else modes
            variants, names = build_variants(
                xbar[state], codes[state], fill, positions, book, coarse_for(state))
            latents = torch.as_tensor(variants, device=device, dtype=torch.float32)
            with torch.no_grad():
                soft = connector(
                    latents, torch.ones(latents.shape[:2], dtype=torch.bool, device=device))
            started = time.time()
            scored = answer_variant_scores(
                model, processor, soft,
                str(pairs["question"][int(pair_row)]),
                str(pairs["answer"][int(pair_row)]))
            elapsed += time.time() - started
            rows.append({"pair_row": int(pair_row), "state_id": state_ids[state],
                         "names": names, "scored": scored})

        peak_gb = torch.cuda.max_memory_allocated(device) / 2**30
        name = str(dtype).replace("torch.", "")
        ablations = slice(1, 1 + len(positions))
        delta = np.concatenate([
            (r["scored"]["nll_bits"][ablations] - r["scored"]["nll_bits"][0]).numpy()
            for r in rows])
        kl = np.concatenate([r["scored"]["kl_bits"][ablations].numpy() for r in rows])
        quantiser_cost = np.array([
            float(r["scored"]["nll_bits"][0] - r["scored"]["nll_bits"][names.index("raw_xbar")])
            for r in rows])

        summary = {
            "weights_gb": round(weights_gb, 2), "peak_gb": round(peak_gb, 2),
            "seconds_per_question": round(elapsed / max(len(rows), 1), 3),
            "variants": len(names),
            "delta_nll_bits": {
                "mean": float(delta.mean()), "median": float(np.median(delta)),
                "abs_median": float(np.median(np.abs(delta))),
                "max_abs": float(np.abs(delta).max()),
                "nonzero_fraction": float((delta != 0).mean()),
            },
            "kl_bits": {
                "mean": float(kl.mean()), "median": float(np.median(kl)),
                "max": float(kl.max()), "min": float(kl.min()),
                "nonzero_fraction": float((kl != 0).mean()),
            },
            "quantiser_cost_bits": {
                "mean": float(quantiser_cost.mean()),
                "values": [round(float(v), 3) for v in quantiser_cost],
            },
        }
        if args.null:
            summary["null_control"] = {
                "max_abs_delta_nll_bits": float(np.abs(delta).max()),
                "max_kl_bits": float(kl.max()),
                "passed": bool(np.abs(delta).max() == 0.0 and kl.max() == 0.0),
            }
        report["by_dtype"][name] = summary

        # Batch invariance: the same variant scored alone against inside the
        # full batch. Mixing batch sizes is only safe if this is negligible
        # against the signal above.
        single = rows[0]["scored"]["nll_bits"][0]
        state = index[keys[int(chosen[0])]]
        variants, _ = build_variants(
            xbar[state], codes[state],
            codes[state] if args.null else modes, positions, book,
            coarse_for(state))
        alone = torch.as_tensor(variants[:1], device=device, dtype=torch.float32)
        with torch.no_grad():
            soft = connector(
                alone, torch.ones(alone.shape[:2], dtype=torch.bool, device=device))
        solo = answer_variant_scores(
            model, processor, soft,
            str(pairs["question"][int(chosen[0])]),
            str(pairs["answer"][int(chosen[0])]))
        summary["batch_invariance_bits"] = float(abs(solo["nll_bits"][0] - single))

        del model, connector
        torch.cuda.empty_cache()

    print(json.dumps(report["by_dtype"], indent=2, sort_keys=True))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(report["by_dtype"], indent=2, sort_keys=True), encoding="utf-8")

    if args.null:
        for name, summary in report["by_dtype"].items():
            control = summary["null_control"]
            verdict = "PASS" if control["passed"] else "FAIL"
            print(f"\n[{name}] null control {verdict}: "
                  f"max |dNLL| {control['max_abs_delta_nll_bits']:.3e} bits, "
                  f"max KL {control['max_kl_bits']:.3e} bits")


if __name__ == "__main__":
    main()
