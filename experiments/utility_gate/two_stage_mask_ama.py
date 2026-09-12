"""Two-stage utility gate ported to the AMA stack (32B reader, v2 codebook).

Mirrors ``two_stage_mask.py`` (WMA) stage for stage; only the reader I/O and
the artefact formats differ:

  * stage-1 utility ``|U_j|`` is fitted from the official-question labels
    (``label_official.py`` output, latent+anchor caliber, fp32 protocol);
  * the second pass scores each first-pass dropped position restored alone
    against the joint-drop baseline, with the deployed reader (Qwen3-32B +
    ``qwen32-input-k32-rms-top10`` bridge, sha-locked) and the same fixed-
    geometry batch + null control + reference-drift checks;
  * the exported envelope ``max(|U|, U_cond)`` keeps the released interface:
    a fixed 1024-dim vector plus the frozen lambda, no per-observation mask.

Domain-independent pieces (``restore_variants``, ``combine_utilities``,
``select_states``) are imported from the WMA implementation and shared.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.utility_gate.two_stage_mask import (
    combine_utilities, restore_variants, select_states, sha256, write_json,
)
from experiments.utility_gate.export_mask import keep_mask, per_state_rate

PROTOCOL = "residualmem_two_stage_utility_ama_v1"
MASK_PROTOCOL = "residualmem_utility_gate_mask_ama_official_v1"


def read_mask(path: Path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if meta.get("protocol") != MASK_PROTOCOL:
            raise ValueError(f"stage-1 mask protocol mismatch: {meta.get('protocol')}")
        utility = np.asarray(data["utility"], np.float64)
        lam = float(data["lam"])
    if utility.shape != (1024,) or not np.isfinite(utility).all() or (utility < 0).any():
        raise ValueError("invalid stage-1 utility vector")
    return utility, lam, meta


def fit_mask(args):
    """Stage-1 |U_j| from the official-question labels, as a mask artefact.

    ``utility[j]`` is the mean absolute single-position counterfactual effect
    over the fit rows. Positions without any label are imputed with the mean
    of the measured positions and explicitly listed -- they are few, and the
    envelope rule treats them the same as any low-support estimate.
    """
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.labels, allow_pickle=False) as data:
        position = np.asarray(data["position"], np.int64)
        delta = np.asarray(data["delta_nll_bits"], np.float64)
    magnitude = np.zeros(1024)
    counts = np.zeros(1024, np.int64)
    for j in range(1024):
        block = delta[position == j]
        counts[j] = block.size
        if block.size:
            magnitude[j] = np.abs(block).mean()
    measured = counts >= args.minimum_labels
    if not measured.any():
        raise ValueError("no measured positions in the label file")
    imputed = np.flatnonzero(~measured).tolist()
    utility = magnitude.copy()
    utility[~measured] = magnitude[measured].mean()
    meta = {
        "protocol": MASK_PROTOCOL, "stage": "stage1-fit",
        "lambda": args.lambda_value, "estimator": "mean |delta_nll_bits| per position",
        "labels": str(Path(args.labels).resolve()), "labels_sha256": sha256(args.labels),
        "labels_count": int(counts.sum()), "measured_positions": int(measured.sum()),
        "imputed_positions": imputed,
        "imputation": "mean |U| of measured positions",
        "min_labels_per_position": args.minimum_labels,
        "utility_magnitude_bits": float(utility.mean()),
        "prompt": "latent+anchor (deployed caliber)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, utility=utility, lam=args.lambda_value,
                        magnitude=magnitude, label_counts=counts, measured=measured,
                        metadata=np.asarray(json.dumps(meta, ensure_ascii=False)))
    write_json(args.output.with_suffix(".json"), meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "imputed_positions"},
                     indent=2), flush=True)


def load_inputs(args):
    from experiments.utility_gate.label_official import (
        load_codes, load_posteriors, split_of_state,
    )
    original, lam, mask_meta = read_mask(args.mask)
    codes, state_ids = load_codes(Path(args.pq))
    by_id = {sid: i for i, sid in enumerate(state_ids)}
    row_of, dumped = load_posteriors(Path(args.posteriors), Path(args.records))
    split, shared = split_of_state(Path(args.states), Path(args.records))
    return original, lam, mask_meta, codes, by_id, dumped, row_of, split, shared


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    from experiments.state_tokenizer.common import iter_jsonl
    original, lam, mask_meta, codes, by_id, dumped, row_of, split, shared = load_inputs(args)
    pairs = list(iter_jsonl(Path(args.pairs)))
    eligible = []
    pair_state = {}
    for index, pair in enumerate(pairs):
        if pair.get("split") != "train":
            continue
        state_id = str(pair["state_id"])
        state = by_id.get(state_id, -1)
        posterior = row_of.get(state_id, -1)
        if state < 0 or posterior < 0:
            continue
        if not str(pair["question"]).strip() or not str(pair["answer"]).strip():
            continue
        pair_state[index] = (state, posterior)
        eligible.append(index)
    states = np.unique([pair_state[i][0] for i in eligible])
    if not len(states):
        raise SystemExit("no eligible train pairs with posteriors")
    entropy = dumped["entropy_bits"]
    state_posterior = {}
    for index in eligible:
        state_posterior.setdefault(pair_state[index][0], pair_state[index][1])
    post_rows = np.asarray([state_posterior[int(s)] for s in states])
    drop = ~keep_mask(original, entropy[post_rows].reshape(len(states), -1), lam)
    # sample balancing key: the episode that first touches each state (full
    # episode id -- the prefix is the constant "ama" and would not balance)
    episode_of_state = {}
    for index in eligible:
        state = pair_state[index][0]
        if state not in episode_of_state:
            episode_of_state[state] = str(pairs[index]["episode_id"])
    chosen = select_states(states, drop,
                           [episode_of_state[int(s)] for s in states],
                           args.max_states, args.seed)
    chosen_set = set(int(s) for s in chosen)
    eligible_in = [int(i) for i in eligible if pair_state[i][0] in chosen_set]
    if args.max_questions_per_state:
        # One question per state keeps full position coverage at a fraction of
        # the cost: the pilot's first job is covering candidate positions, and
        # per-position support is reported honestly either way (low-support
        # positions fall back to the stage-1 utility, never to zero).
        rng_rows = np.random.default_rng(args.seed + 11)
        rows_by_state = {}
        for i in eligible_in:
            rows_by_state.setdefault(pair_state[i][0], []).append(i)
        rows = []
        for state in sorted(rows_by_state):
            pool = rows_by_state[state]
            order = rng_rows.permutation(len(pool))
            rows.extend(pool[int(k)] for k in
                        sorted(order[:args.max_questions_per_state]))
    else:
        rows = eligible_in
    chosen_post = np.asarray([state_posterior[int(s)] for s in chosen])
    chosen_drop = ~keep_mask(original, entropy[chosen_post].reshape(len(chosen), -1), lam)
    files = {key: str(Path(value).resolve()) for key, value in {
        "mask": args.mask, "pq": args.pq, "codebook": args.codebook,
        "posteriors": args.posteriors, "pairs": args.pairs,
        "records": args.records, "states": args.states, "bridge": args.bridge,
    }.items()}
    output = {
        "protocol": PROTOCOL, "stage": "pilot", "files": files,
        "sha256": {key: sha256(path) for key, path in files.items()},
        "lambda": lam, "seed": args.seed, "pair_split": "train",
        "prompt": "latent+anchor", "anchor_k": args.anchor_k,
        "anchor_style": args.anchor_style, "cache_dir": str(Path(args.cache_dir).resolve()),
        "reader_model": args.reader_model,
        "max_model_len": args.max_model_len, "max_new_tokens": args.max_new_tokens,
        "max_answer_tokens": args.max_answer_tokens,
        "pair_rows": rows, "state_rows": [int(s) for s in chosen],
        "pair_state": {str(i): int(pair_state[i][0]) for i in rows},
        "pair_posterior": {str(i): int(pair_state[i][1]) for i in rows},
        "state_posterior": {str(int(s)): int(p) for s, p in zip(chosen, chosen_post)},
        "observations": len(chosen), "questions": len(rows),
        "samples": len({episode_of_state[int(s)] for s in chosen}),
        "eligible_candidate_positions": int(drop.any(0).sum()),
        "selected_candidate_positions": int(chosen_drop.any(0).sum()),
        "expected_labels": int(sum(
            (~keep_mask(original, entropy[pair_state[i][1]].reshape(-1), lam)).sum()
            for i in rows)),
        "stage1_fit_question_distribution": mask_meta.get("prompt"),
        "background": "stage-1 gate with frozen all-send-history WM posterior; not closed loop",
        "decision": "rescue only; lambda fixed; effective utility = max(original, conditional mean)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, output)
    print(json.dumps({key: value for key, value in output.items()
                      if key not in {"pair_rows", "pair_state", "pair_posterior",
                                     "state_rows", "state_posterior"}}, indent=2), flush=True)


def label(args):
    import torch
    from experiments.state_tokenizer.run_ama_web_latent import build_anchor_block
    from experiments.utility_gate.label_official import (
        build_prompt, efficient_attention, load_ama_reader, place_bridge,
        score_variants, step_texts_for,
    )
    from experiments.utility_gate.verify_scorer import load_codebook, rebuild

    plan = json.loads(args.plan.read_text())
    if plan["protocol"] != PROTOCOL or not 0 <= args.rank < args.world_size:
        raise ValueError("invalid plan or shard")
    for key, path in plan["files"].items():
        if sha256(path) != plan["sha256"][key]:
            raise ValueError(f"input changed: {key}")
    original, lam, _ = read_mask(Path(plan["files"]["mask"]))
    with np.load(plan["files"]["pq"], allow_pickle=True) as data:
        codes = np.asarray(data["codes/all"], np.uint8)
        id_row = {str(v): i for i, v in enumerate(np.asarray(data["state_ids/all"]))}
    with np.load(plan["files"]["posteriors"], allow_pickle=False) as data:
        entropy = np.asarray(data["entropy_bits"])
        fill = np.asarray(data["wm_argmax"])
    from experiments.state_tokenizer.common import iter_jsonl
    pairs = list(iter_jsonl(Path(plan["files"]["pairs"])))
    book = load_codebook(Path(plan["files"]["codebook"]))

    tokenizer, model = load_ama_reader(
        args.reader_model, "float32", args.device, args.device_map)
    bridge, metadata = place_bridge(Path(plan["files"]["bridge"]), model)
    enable_thinking = bool(metadata.get("enable_thinking"))
    from xt_ama_adapter.qwen32_bridge import Qwen32LatentReader
    reader = Qwen32LatentReader(model, tokenizer, bridge,
                                enable_thinking=enable_thinking)

    rows = plan["pair_rows"][args.rank::args.world_size]
    if args.limit is not None:
        rows = rows[: args.limit]
    output = Path(args.output) / f"rank-{args.rank}"
    output.mkdir(parents=True, exist_ok=True)
    config = {"protocol": PROTOCOL, "plan_sha256": sha256(args.plan),
              "model": str(Path(args.reader_model).resolve()),
              "bridge": plan["files"]["bridge"], "rank": args.rank,
              "world_size": args.world_size, "chunk": args.chunk,
              "batch_size": args.chunk + 3, "dtype": "float32",
              "prompt": plan["prompt"], "anchor_k": plan["anchor_k"],
              "anchor_style": plan["anchor_style"], "pair_rows": rows}
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("refusing to mix labeling configurations")
    write_json(config_path, config)

    started = time.monotonic()
    scored_rows = 0
    for done, pair_row in enumerate(rows, 1):
        target = output / f"pair-{pair_row:06d}.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as cached:
                if str(cached["plan_sha256"]) != config["plan_sha256"] \
                        or int(cached["pair_row"]) != pair_row:
                    raise ValueError("invalid resumed label row")
            continue
        pair = pairs[pair_row]
        state = plan["pair_state"][str(pair_row)]
        posterior = plan["pair_posterior"][str(pair_row)]
        truth = codes[state]
        keep = keep_mask(original, entropy[posterior].reshape(-1), lam).reshape(truth.shape)
        positions = np.flatnonzero(~keep.reshape(-1))
        fill_codes = fill[posterior]
        effective = truth.reshape(-1)[positions] != fill_codes.reshape(-1)[positions]
        active = positions[effective]
        gains = np.zeros(len(positions), np.float64)
        position_index = {int(j): i for i, j in enumerate(positions)}
        base_nll = full_nll = None
        row_null_max = 0.0
        row_started = time.monotonic()

        answer_ids = tokenizer.encode(str(pair["answer"]), add_special_tokens=False)[: plan["max_answer_tokens"]]
        if not answer_ids:
            raise ValueError(f"pair row {pair_row} has an empty answer")

        # fixed anchor memories for this pair (deployment caliber latent+anchor)
        entries = [entry for entry in pair.get("topk", [])[: plan["anchor_k"]]
                   if entry.get("has_code_row")]
        texts = step_texts_for(plan["cache_dir"], str(pair["episode_id"]))
        anchor, anchor_stats = build_anchor_block(
            [(int(entry["step_index"]), texts[int(entry["position"])])
             for entry in entries], style=plan["anchor_style"])
        others = [rebuild(codes[id_row[str(entry["state_id"])]][None], book)[0]
                  for entry in entries[1:]]

        # the prompt is identical for every variant of this pair: build it once
        # on the baseline restore (all dropped codes at the WM fill)
        base_codes = restore_variants(truth, fill_codes, keep,
                                      np.empty(0, np.int64))[0]
        base_latent = rebuild(base_codes[None], book)[0]
        inputs, audit, prefix_len = build_prompt(
            reader, tokenizer, str(pair["question"]),
            [base_latent] + others, anchor, enable_thinking=enable_thinking,
            max_model_len=plan["max_model_len"],
            max_new_tokens=plan["max_new_tokens"])
        # per-row adaptive chunk: keep tokens-per-forward roughly constant so
        # long prompts shrink the batch instead of OOM-ing (logits and KV both
        # scale with batch x seq; rows 0-2 stay base/truth/null regardless)
        seq_len = int(inputs["inputs_embeds"].shape[1]) + len(answer_ids)
        if args.max_prompt_tokens and seq_len > args.max_prompt_tokens:
            # long-anchor rows collapse to chunk 1 and cost 10-30x a normal
            # row for the same +1 label per position: skip them (the export
            # reports skipped rows as unmeasured, which fall back to stage-1)
            print(json.dumps({"rank": args.rank, "skipped": int(pair_row),
                              "prompt_tokens": seq_len}), flush=True)
            continue
        row_chunk = max(1, min(args.chunk, args.chunk_tokens // seq_len))

        offset = 0
        row_skipped = False
        while offset < max(len(active), 1):
            block = active[offset:offset + row_chunk]
            restored = restore_variants(truth, fill_codes, keep, block)
            batch_codes = np.repeat(restored[:1], len(block) + 3, axis=0)
            batch_codes[1] = truth
            # row 2 is an identical reference/null control in every forward
            batch_codes[3:3 + len(block)] = restored[1:]
            decoded = rebuild(batch_codes, book)
            scored_states = np.stack(
                [decoded] + [np.repeat(other[None], len(decoded), axis=0)
                             for other in others], axis=1)
            try:
                scored = score_variants(
                    model, bridge, inputs["inputs_embeds"], inputs["attention_mask"],
                    prefix_len, answer_ids, scored_states)
            except torch.OutOfMemoryError:
                # transient activation spike: shrink the batch, free the cache
                # and retry the same block; a row that OOMs even at chunk 1 is
                # skipped rather than killing the rank
                torch.cuda.empty_cache()
                if row_chunk <= 1:
                    print(json.dumps({"rank": args.rank, "skipped_oom": int(pair_row),
                                      "prompt_tokens": seq_len}), flush=True)
                    row_skipped = True
                    break
                row_chunk = max(1, row_chunk // 2)
                continue
            nll = scored["nll_bits"].astype(np.float64)
            if not np.isfinite(nll).all():
                raise ValueError("non-finite reader NLL")
            null_error = abs(nll[2] - nll[0])
            row_null_max = max(row_null_max, null_error)
            if null_error > 1e-4:
                raise ValueError(f"null-control error too large: {null_error}")
            if base_nll is not None and max(abs(base_nll - nll[0]),
                                            abs(full_nll - nll[1])) > 1e-3:
                raise ValueError("reference drift across equal-geometry chunks")
            base_nll, full_nll = float(nll[0]), float(nll[1])
            for index, position in enumerate(block, 3):
                gains[position_index[int(position)]] = nll[0] - nll[index]
            del decoded, scored_states, scored
            offset += len(block) if len(block) else 1

        if row_skipped:
            continue
        temporary = target.with_suffix(".partial.npz")
        np.savez_compressed(temporary, pair_row=pair_row, state_row=state,
                            position=positions, restore_gain_bits=gains,
                            conditional_utility=np.abs(gains),
                            original_utility=original[positions],
                            entropy_bits=entropy[posterior].reshape(-1)[positions],
                            effective=effective, baseline_nll_bits=base_nll,
                            full_nll_bits=full_nll, null_max_bits=row_null_max,
                            prompt_tokens=int(inputs["inputs_embeds"].shape[1]),
                            answer_len=len(answer_ids),
                            truncated=bool(audit.get("truncated")),
                            plan_sha256=np.asarray(config["plan_sha256"]))
        temporary.replace(target)
        scored_rows += 1
        print(json.dumps({"rank": args.rank, "done": done, "total": len(rows),
                          "pair_row": pair_row, "positions": len(positions),
                          "effective_positions": len(active),
                          "row_seconds": round(time.monotonic() - row_started, 2),
                          "average_seconds": round((time.monotonic() - started) / scored_rows, 2),
                          "null_max_bits": row_null_max}), flush=True)
    write_json(output / "complete.json",
               {"protocol": PROTOCOL, "rows": len(rows), "config": config})


def model_tokenizer_prompt(_tokenizer):
    """Placeholder kept out of the call path; build_prompt takes ``reader``."""
    return None


def export(args):
    plan = json.loads(args.plan.read_text())
    plan_hash = sha256(args.plan)
    if args.output.exists():
        raise FileExistsError(args.output)
    original, lam, old_meta = read_mask(Path(plan["files"]["mask"]))
    if sha256(plan["files"]["mask"]) != plan["sha256"]["mask"]:
        raise ValueError("stage-1 mask changed")
    totals = np.zeros(1024, np.float64)
    counts = np.zeros(1024, np.int64)
    state_sets = [set() for _ in range(1024)]
    seen = set()
    protocols = set()
    for config_path in sorted(Path(args.labels).glob("rank-*/config.json")):
        config = json.loads(config_path.read_text())
        protocols.add((config["plan_sha256"], config["model"], config["bridge"],
                       config["chunk"], config["dtype"], config["batch_size"],
                       config["prompt"], config["anchor_k"], config["anchor_style"]))
    if len(protocols) != 1 or next(iter(protocols))[0] != plan_hash:
        raise ValueError("missing or mixed labeling configurations")
    for path in sorted(Path(args.labels).glob("rank-*/pair-*.npz")):
        if path.name.endswith(".partial.npz"):
            continue
        with np.load(path, allow_pickle=False) as data:
            row = int(data["pair_row"])
            if row in seen or row not in plan["pair_rows"] \
                    or str(data["plan_sha256"]) != plan_hash:
                raise ValueError(f"invalid or duplicated label row: {path}")
            state = int(data["state_row"])
            if state != plan["pair_state"][str(row)]:
                raise ValueError("label state mismatch")
            positions = np.asarray(data["position"], np.int64)
            values = np.asarray(data["conditional_utility"], np.float64)
            if len(positions) != len(np.unique(positions)) or not np.isfinite(values).all() \
                    or (values < 0).any():
                raise ValueError("invalid labels")
            np.add.at(totals, positions, values)
            np.add.at(counts, positions, 1)
            for j in positions:
                state_sets[int(j)].add(state)
            seen.add(row)
    if seen != set(plan["pair_rows"]) or int(counts.sum()) != plan["expected_labels"]:
        if not args.allow_partial and len(seen) < len(plan["pair_rows"]):
            raise ValueError(f"incomplete labels: {len(seen)}/{len(plan['pair_rows'])} rows, "
                         f"{counts.sum()}/{plan['expected_labels']} labels")
    effective, conditional, measured = combine_utilities(
        original, totals, counts, minimum_labels=args.minimum_labels)
    state_counts = np.asarray([len(s) for s in state_sets], np.int64)
    reports = {}
    for name, path in [("fit_corpus", Path(plan["files"]["posteriors"])),
                       ("ama_val", Path(args.val_posteriors))]:
        with np.load(path, allow_pickle=False) as data:
            entropy, bits = data["entropy_bits"], data["code_bits"]
        flat_entropy = entropy.reshape(len(entropy), -1)
        flat_bits = bits.reshape(len(bits), -1)
        before = keep_mask(original, flat_entropy, lam)
        after = keep_mask(effective, flat_entropy, lam)
        assert np.all(after | ~before)
        rescued = after & ~before
        both_drop = (~before & ~after).sum()
        reports[name] = {
            "observations": len(before), "posterior_sha256": sha256(path),
            "comparison": "same frozen all-send-history posteriors, not a closed-loop rate",
            "initial_rate_bits": per_state_rate(flat_bits, before),
            "upgraded_rate_bits": per_state_rate(flat_bits, after),
            "initial_keep_fraction": float(before.mean()),
            "upgraded_keep_fraction": float(after.mean()),
            "mask_agreement": float((before == after).mean()),
            "changed_positions_mean": float(rescued.sum(1).mean()),
            "rescued_fraction_of_initial_drops":
                float(rescued.sum() / max((~before).sum(), 1)),
            "dropped_set_jaccard": float(both_drop / max((~before | ~after).sum(), 1)),
            "unmeasured_candidate_positions":
                np.flatnonzero((~before).any(0) & ~measured).tolist(),
            "low_support_candidate_positions":
                np.flatnonzero((~before).any(0) & ((counts < 16) | (state_counts < 4))).tolist(),
            "newly_dropped_positions": int((before & ~after).sum()),
        }
    meta = {
        "protocol": MASK_PROTOCOL, "estimator_protocol": PROTOCOL, "stage": "pilot",
        "lambda": lam, "mask_bits": 0, "plan": str(args.plan.resolve()),
        "plan_sha256": plan_hash,
        "background_mask_sha256": plan["sha256"]["mask"],
        "background_lambda": lam,
        "rule": "keep iff max(original_utility, conditional_utility) >= lambda * entropy",
        "utility_is": "rescue-only envelope of original and mean absolute "
                      "single-restoration NLL effect",
        "fitted_on_question_distribution": old_meta.get("prompt"),
        "pair_split": "train", "questions": len(seen),
        "observations": plan["observations"], "labels": int(counts.sum()),
        "measured_positions": int(measured.sum()),
        "increased_utility_positions": int((effective > original).sum()),
        "minimum_labels": args.minimum_labels,
        "insufficient_support_policy": "unchanged original utility, explicitly unverified",
        "gate_lambda_must_match_background": True,
        "same_posterior_comparison": reports,
        "not_validated": ["closed-loop upgraded-policy rate",
                          "upgraded-policy QA quality"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, utility=effective, lam=lam,
                        original_utility=original, conditional_utility=conditional,
                        label_counts=counts, state_counts=state_counts,
                        measured=measured,
                        metadata=np.asarray(json.dumps(meta, ensure_ascii=False)))
    write_json(args.output.with_suffix(".json"), meta)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fit-mask")
    p.add_argument("--labels", required=True)
    p.add_argument("--lambda-value", dest="lambda_value", type=float, required=True)
    p.add_argument("--minimum-labels", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--mask", required=True)
    p.add_argument("--pq", default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    p.add_argument("--codebook", default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    p.add_argument("--posteriors", default="outputs/utility_gate_ama/posteriors-train.npz")
    p.add_argument("--pairs", default="outputs/utility_gate_ama/pairs-official.jsonl")
    p.add_argument("--records", default="outputs/wm_train/ama-v1/records.jsonl")
    p.add_argument("--states", default="outputs/wm_train/ama-v1/states.jsonl")
    p.add_argument("--cache-dir", default="outputs/utility_gate_ama/ama-cache")
    p.add_argument("--bridge",
                   default="outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-top10.pt")
    p.add_argument("--reader-model", default="/data1/models/Qwen3-32B")
    p.add_argument("--anchor-k", type=int, default=8)
    p.add_argument("--anchor-style", default="element")
    p.add_argument("--max-states", type=int, default=96)
    p.add_argument("--max-questions-per-state", type=int, default=0,
                   help="0 = all eligible questions per state")
    p.add_argument("--seed", type=int, default=35)
    p.add_argument("--max-model-len", type=int, default=32000)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-answer-tokens", type=int, default=104)
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("label")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--reader-model", default="/data1/models/Qwen3-32B")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--chunk", type=int, default=14)
    p.add_argument("--chunk-tokens", type=int, default=8000,
                   help="per-forward token budget: chunk shrinks on long prompts")
    p.add_argument("--max-prompt-tokens", type=int, default=0,
                   help="skip rows whose built prompt exceeds this (0 = off)")

    p = sub.add_parser("export")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--allow-partial", action="store_true",
                   help="export with the rows completed so far; unmeasured "
                        "positions keep their stage-1 utility and are listed")
    p.add_argument("--val-posteriors",
                   default="outputs/utility_gate_ama/posteriors-val.npz")
    p.add_argument("--minimum-labels", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    {"fit-mask": fit_mask, "prepare": prepare, "label": label, "export": export}[args.command](args)


if __name__ == "__main__":
    main()
