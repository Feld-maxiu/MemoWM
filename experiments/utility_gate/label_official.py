"""Counterfactual utility labels on the official AMA questions.

Same definition as the WebWorld gate (UTILITY_GATE.md section 2.1):

    U_j = [ L_ans(q, a | x_hat(-j)) - L_ans(q, a | x_hat(full)) ] / ln 2

with three AMA-specific choices, each forced by how this system actually runs.

**The prompt is the deployed one.** Every published AMA number was produced with
memory_mode="matched" and top_k=1: the reader sees the retrieved step's latent
row *and* that step's raw text row. Measuring on the latent alone would ask a
question the system never asks, so the text row is included and prompt
truncation is delegated to the adapter's own build_inputs_segments.

**The reference is the world model's reconstruction.** The gate governs the
discrete codes, so the intact memory is Decode(codes), not the cached float32
state. Both sides are quantisation-space quantities and the quantiser's error
cancels in the difference. Verified directly: decoding costs the reader -3.7
bits against the cached latent, i.e. nothing.

**Every variant rides in one forward.** In fp32 the answer NLL still shifts by
~2e-4 bits when the batch shape changes, and that is the order of the
single-position signal, so the reference and its ablations must share a batch.
The latent block sits at a fixed offset (immediately after the prefix) and is the
only span that differs between rows; text, suffix and answer are identical
across the batch by construction.

Runs under the torch venv; fp32 needs --device-map auto across >= 2 GPUs.
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.state_tokenizer.common import iter_jsonl
from experiments.utility_gate.groups import NUM_POSITIONS, slot_of, subspace_of
from experiments.utility_gate.label_counterfactual_ama import (
    latent_position_ids, load_ama_reader, load_codes, load_posteriors,
    place_bridge, split_of_state,
)
from experiments.utility_gate.verify_scorer import load_codebook, rebuild

LN2 = math.log(2.0)


@contextlib.contextmanager
def efficient_attention():
    """Force the memory-efficient SDPA kernel for fp32 forwards.

    transformers hands SDPA a 4D mask, which sends torch to the math backend:
    at a 5.3k-token ``latent+anchor`` prompt that materialises the full
    B x heads x L x L score tensor (21 GB at B=2, ~50 GB at B=5). The efficient
    kernel handles the same call in 1.1 GB, which is the difference between
    scoring the anchor regime and not scoring it at all.
    """
    import os
    if not os.environ.get("RESIDUALMEM_FORCE_EFFICIENT_ATTN"):
        yield
        return
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:                      # pragma: no cover - older torch
        yield
        return
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        yield


def build_prompt(reader, tokenizer, question, memory_states, text_block,
                 *, enable_thinking, max_model_len, max_new_tokens):
    """Assemble the prompt from k latent rows plus one optional text block.

    This mirrors the deployed ``latent+anchor`` construction: the latent rows
    are the retrieved screens in rank order, and the text block is the element
    anchor built from those same screens.
    """
    from xt_ama_adapter.qwen32_bridge import render_ama_openend_parts

    segments = []
    for state in memory_states:
        array = np.asarray(state, dtype=np.float32)
        segments.append(("latent", (array, np.ones(array.shape[0], dtype=bool))))
    if text_block:
        segments.append(("text", text_block))
    inputs, audit = reader.build_inputs_segments(
        question, segments, max_model_len=max_model_len,
        max_new_tokens=max_new_tokens)
    prefix, _suffix, _ = render_ama_openend_parts(
        tokenizer, question, enable_thinking=bool(enable_thinking))
    prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False))
    return inputs, audit, prefix_len


def build_deployed_prompt(reader, tokenizer, question, step_text, state, valid,
                          *, enable_thinking, max_model_len, max_new_tokens,
                          latent_only=False):
    """The adapter's own prompt assembly, so truncation matches the deployment.

    latent_only drops the retrieved step's text row, which is the isolation the
    report's utility definition assumes: the gate governs the codes, so the
    measurement asks what the codes alone carry. The deployed prompt also serves
    the raw text, and that arm is measured separately.
    """
    from xt_ama_adapter.qwen32_bridge import render_ama_openend_parts

    segments = [("latent", (state, valid))]
    if not latent_only:
        segments.append(("text", step_text))
    inputs, audit = reader.build_inputs_segments(
        question, segments,
        max_model_len=max_model_len, max_new_tokens=max_new_tokens)
    prefix, _suffix, _ = render_ama_openend_parts(
        tokenizer, question, enable_thinking=bool(enable_thinking))
    prefix_len = len(tokenizer.encode(prefix, add_special_tokens=False))
    return inputs, audit, prefix_len


def score_variants(model, bridge, embeds, mask, prefix_len, answer_ids,
                   states, *, chunk=8):
    """Teacher-forced NLL (bits) and KL-from-reference for each memory variant.

    states is (V, slots, input_dim) with the reference at row 0. Only the latent
    span differs between rows; every other token is shared.
    """
    device = embeds.device
    states = np.asarray(states)
    if states.ndim == 3:                      # (V, slots, dim) -- one memory row
        states = states[:, None]
    variants, memories, slots = (int(value) for value in states.shape[:3])
    span = memories * slots
    if prefix_len + span > embeds.shape[1]:
        raise ValueError("latent span falls outside the assembled prompt")

    soft, _span_mask = bridge(
        torch.as_tensor(states, dtype=torch.float32, device=device),
        torch.ones((variants, memories, slots), dtype=torch.bool, device=device))
    if soft.shape[1] != span:
        raise ValueError(f"bridge returned {soft.shape[1]} tokens for {span} slots")
    soft = soft.to(embeds.dtype)

    prompted = embeds.expand(variants, -1, -1).clone()
    prompted[:, prefix_len:prefix_len + span, :] = soft

    embed = model.get_input_embeddings()
    answer_len = len(answer_ids)
    answer_embed = embed(
        torch.tensor([answer_ids], dtype=torch.long, device=device)
    ).expand(variants, -1, -1)
    inputs = torch.cat((prompted, answer_embed), 1)
    attention = torch.cat((
        mask.expand(variants, -1),
        torch.ones((variants, answer_len), dtype=mask.dtype, device=device)), 1)

    kwargs = {"use_cache": False}
    parameters = inspect.signature(model.forward).parameters
    if "position_ids" in parameters:
        kwargs["position_ids"] = latent_position_ids(attention)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = answer_len + 1

    with torch.no_grad(), efficient_attention():
        out = model(inputs_embeds=inputs, attention_mask=attention, **kwargs)
        logits = out.logits[:, -answer_len - 1:-1, :]
        reduce_device = logits.device
        target = torch.tensor([answer_ids] * variants, dtype=torch.long,
                              device=reduce_device)
        nll = torch.zeros(variants, dtype=torch.float32, device=reduce_device)
        token_nll = torch.empty((variants, answer_len), dtype=torch.float32,
                                device=reduce_device)
        kl = torch.zeros(variants, dtype=torch.float32, device=reduce_device)
        for start in range(0, answer_len, chunk):
            stop = min(start + chunk, answer_len)
            logprob = torch.nn.functional.log_softmax(logits[:, start:stop].float(), -1)
            gathered = logprob.gather(-1, target[:, start:stop].unsqueeze(-1)).squeeze(-1)
            nll -= gathered.sum(-1)
            token_nll[:, start:stop] = -gathered
            anchor = logprob[0:1]
            kl += (anchor.exp() * (anchor - logprob)).sum(-1).sum(-1)

    return {"nll_bits": (nll / LN2).cpu().numpy(),
            "token_nll_bits": (token_nll / LN2).cpu().numpy(),
            "kl_bits": (kl / LN2).cpu().numpy()}


def score_text_only(reader, model, tokenizer, question, step_text, answer_ids,
                    *, enable_thinking, max_model_len, max_new_tokens):
    """Gold-answer NLL with the latent row removed: the filter's other arm."""
    inputs, _audit = reader.build_inputs_segments(
        question, [("text", step_text)],
        max_model_len=max_model_len, max_new_tokens=max_new_tokens)
    embeds = inputs["inputs_embeds"]
    mask = inputs["attention_mask"]
    device = embeds.device
    answer_embed = model.get_input_embeddings()(
        torch.tensor([answer_ids], dtype=torch.long, device=device))
    full = torch.cat((embeds, answer_embed), 1)
    attention = torch.cat((mask, torch.ones((1, len(answer_ids)),
                                            dtype=mask.dtype, device=device)), 1)
    kwargs = {"use_cache": False}
    parameters = inspect.signature(model.forward).parameters
    if "position_ids" in parameters:
        kwargs["position_ids"] = latent_position_ids(attention)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = len(answer_ids) + 1
    with torch.no_grad(), efficient_attention():
        out = model(inputs_embeds=full, attention_mask=attention, **kwargs)
        logits = out.logits[:, -len(answer_ids) - 1:-1, :].float()
        logprob = torch.nn.functional.log_softmax(logits, -1)
        target = torch.tensor([answer_ids], dtype=torch.long, device=logits.device)
        gathered = logprob.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(-gathered.sum() / LN2)


_EPISODE_TEXTS: dict[str, list[str]] = {}


def step_texts_for(cache_dir, episode_id: str) -> list[str]:
    """Step texts of one episode, read from its cache and kept for that episode.

    Only one episode is held at a time: the pair file is grouped by episode, so
    a single-entry cache covers the whole pass without holding 208 caches in RAM.
    """
    if not _EPISODE_TEXTS:
        pass
    index = int(str(episode_id).split(":")[1])
    payload = torch.load(Path(cache_dir) / f"episode-{index:06d}.pt",
                         map_location="cpu", weights_only=False)
    return [str(text) for text in payload["step_texts"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--pq", default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    parser.add_argument("--codebook",
                        default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--records", default="outputs/wm_train/ama-v1/records.jsonl")
    parser.add_argument("--states", default="outputs/wm_train/ama-v1/states.jsonl")
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--bridge", default=(
        "outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms.pt"))
    parser.add_argument("--reader-model", default="/data1/models/Qwen3-32B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--positions-per-forward", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-states", type=int, default=0)
    parser.add_argument("--max-answer-tokens", type=int, default=104)
    parser.add_argument("--max-model-len", type=int, default=32000)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--anchor-k", type=int, default=8,
                        help="retrieved screens injected as latent rows in the "
                             "latent+anchor prompt; the anchor block is built "
                             "from the same screens")
    parser.add_argument("--anchor-style", default="element")
    parser.add_argument("--cache-dir",
                        default="outputs/utility_gate_ama/ama-cache",
                        help="episode caches, read lazily for the anchor text")
    parser.add_argument("--prompt", choices=("matched", "latent-only",
                                             "latent+anchor"),
                        default="matched",
                        help="matched is the deployed prompt; latent-only strips "
                             "the retrieved step text row so the measurement "
                             "isolates what the codes carry")
    parser.add_argument("--controls", action="store_true")
    parser.add_argument("--verify-splice", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    book = load_codebook(Path(args.codebook))
    codes, state_ids = load_codes(Path(args.pq))
    by_id = {sid: i for i, sid in enumerate(state_ids)}
    split, shared = split_of_state(Path(args.states), Path(args.records))
    row_of, dumped = load_posteriors(Path(args.posteriors), Path(args.records))

    pairs = list(iter_jsonl(Path(args.pairs)))
    usable = []
    for index, pair in enumerate(pairs):
        state_id = str(pair["state_id"])
        if pair.get("split") != args.split:
            continue
        if args.split == "validation" and state_id in shared:
            continue
        state_row = by_id.get(state_id)
        if state_row is None or state_id not in row_of:
            continue
        usable.append((index, state_row, row_of[state_id]))
    if args.max_states:
        seen = {}
        for item in usable:
            seen.setdefault(item[1], item[2])
        chosen = set(list(sorted(seen))[: args.max_states])
        usable = [item for item in usable if item[1] in chosen]
    if args.max_rows:
        usable = usable[: args.max_rows]
    usable = usable[args.shard_index::args.shard_count]
    print(f"[official] {len(usable)} rows for split={args.split} "
          f"({len(pairs)} pairs total)", flush=True)

    tokenizer, model = load_ama_reader(
        args.reader_model, args.dtype, args.device, args.device_map)
    bridge, metadata = place_bridge(Path(args.bridge), model)
    enable_thinking = bool(metadata.get("enable_thinking"))
    from xt_ama_adapter.qwen32_bridge import Qwen32LatentReader
    reader = Qwen32LatentReader(model, tokenizer, bridge,
                                enable_thinking=enable_thinking)
    print(f"[official] reader ready; bridge={Path(args.bridge).name} "
          f"enable_thinking={enable_thinking} dtype={args.dtype}", flush=True)

    rng = np.random.default_rng(args.seed)
    per_state = {}
    for index, state_row, _posterior in usable:
        if state_row not in per_state:
            per_state[state_row] = np.sort(
                rng.permutation(NUM_POSITIONS)[: args.positions])

    label_out = {key: [] for key in (
        "pair_index", "state_row", "position", "kl_bits", "delta_nll_bits",
        "abs_delta_nll_bits", "rate_bits", "fill_code", "true_code")}
    row_out = {key: [] for key in (
        "row_pair_index", "row_state_row", "row_reference_nll_bits",
        "row_answer_len", "row_text_only_nll_bits", "row_all_fill_nll_bits",
        "row_prompt_tokens", "row_text_truncated")}

    started = time.time()
    for done, (index, state_row, posterior) in enumerate(usable):
        pair = pairs[index]
        question = str(pair["question"])
        answer = str(pair["answer"])
        step_text = str(pair["step_text"])
        answer_ids = tokenizer.encode(
            answer, add_special_tokens=False)[: args.max_answer_tokens]
        if not answer_ids:
            continue

        true_codes = codes[state_row]
        fill_codes = dumped["wm_argmax"][posterior]
        rate_flat = dumped["code_bits"][posterior].reshape(-1)
        positions = per_state[state_row]
        states = np.repeat(true_codes[None], args.positions + 1, axis=0)
        for offset, position in enumerate(positions, start=1):
            slot, subspace = slot_of(int(position)), subspace_of(int(position))
            states[offset, slot, subspace] = fill_codes[slot, subspace]
        decoded = rebuild(states, book)
        valid = np.ones(true_codes.shape[0], dtype=bool)

        if args.prompt == "latent+anchor":
            from experiments.state_tokenizer.run_ama_web_latent import (
                build_anchor_block,
            )
            entries = [entry for entry in pair["topk"][: args.anchor_k]
                       if entry.get("has_code_row")]
            texts = step_texts_for(args.cache_dir, str(pair["episode_id"]))
            anchor, anchor_stats = build_anchor_block(
                [(int(entry["step_index"]), texts[int(entry["position"])])
                 for entry in entries], style=args.anchor_style)
            others = [rebuild(codes[by_id[str(entry["state_id"])]][None], book)[0]
                      for entry in entries[1:]]
            scored_states = np.stack(
                [decoded] + [np.repeat(other[None], len(decoded), axis=0)
                             for other in others], axis=1)
            inputs, audit, prefix_len = build_prompt(
                reader, tokenizer, question, [decoded[0]] + others, anchor,
                enable_thinking=enable_thinking,
                max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens)
        else:
            scored_states = decoded
            inputs, audit, prefix_len = build_deployed_prompt(
                reader, tokenizer, question, step_text, decoded[0], valid,
                enable_thinking=enable_thinking, max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens,
                latent_only=(args.prompt == "latent-only"))

        if args.verify_splice and done == 0:
            probe, _probe_audit, probe_prefix = build_deployed_prompt(
                reader, tokenizer, question, step_text, decoded[1], valid,
                enable_thinking=enable_thinking, max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens,
                latent_only=(args.prompt == "latent-only"))
            span = slice(prefix_len, prefix_len + true_codes.shape[0])
            delta = float((probe["inputs_embeds"][:, span]
                           - inputs["inputs_embeds"][:, span]).abs().max())
            print(f"[official] splice check: span changes by {delta:.6f} "
                  f"(must be > 0); prefix_len={prefix_len} "
                  f"same_offset={probe_prefix == prefix_len}", flush=True)

        nll = np.full(args.positions + 1, np.nan)
        kl = np.full(args.positions + 1, np.nan)
        for start in range(0, args.positions + 1, args.positions_per_forward + 1):
            stop = min(start + args.positions_per_forward + 1, args.positions + 1)
            scored = score_variants(
                model, bridge, inputs["inputs_embeds"], inputs["attention_mask"],
                prefix_len, answer_ids, scored_states[start:stop])
            nll[start:stop] = scored["nll_bits"]
            kl[start:stop] = scored["kl_bits"]

        row_out["row_pair_index"].append(index)
        row_out["row_state_row"].append(state_row)
        row_out["row_reference_nll_bits"].append(float(nll[0]))
        row_out["row_answer_len"].append(len(answer_ids))
        row_out["row_prompt_tokens"].append(int(inputs["inputs_embeds"].shape[1]))
        row_out["row_text_truncated"].append(bool(audit.get("truncated")))
        row_out["row_text_only_nll_bits"].append(float("nan"))
        row_out["row_all_fill_nll_bits"].append(float("nan"))

        if args.controls and done == 0:
            text_only = score_text_only(
                reader, model, tokenizer, question, step_text, answer_ids,
                enable_thinking=enable_thinking, max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens)
            worst = true_codes.copy()
            worst[...] = fill_codes
            worst_state = rebuild(worst[None], book)
            # ``worst_state`` is the gated row fully replaced by the world
            # model's mode; the other rows stay at their true codes, exactly as
            # the per-position ablations leave them.
            worst_block = scored_states[0:1].copy()
            worst_block[:, 0] = worst_state[0]
            worst_scored = score_variants(
                model, bridge, inputs["inputs_embeds"], inputs["attention_mask"],
                prefix_len, answer_ids,
                np.concatenate([scored_states[0:1], worst_block]))
            row_out["row_text_only_nll_bits"][-1] = text_only
            row_out["row_all_fill_nll_bits"][-1] = float(worst_scored["nll_bits"][1])
            print(f"[official] controls: reference {nll[0]:.4f} | text-only "
                  f"{text_only:.4f} (latent gain {text_only - nll[0]:+.4f}) | "
                  f"all-fill {worst_scored['nll_bits'][1]:.4f} "
                  f"({worst_scored['nll_bits'][1] - nll[0]:+.4f})", flush=True)

        for offset, position in enumerate(positions, start=1):
            slot, subspace = slot_of(int(position)), subspace_of(int(position))
            label_out["pair_index"].append(index)
            label_out["state_row"].append(state_row)
            label_out["position"].append(int(position))
            label_out["kl_bits"].append(float(kl[offset]))
            label_out["delta_nll_bits"].append(float(nll[offset] - nll[0]))
            label_out["abs_delta_nll_bits"].append(float(abs(nll[offset] - nll[0])))
            label_out["rate_bits"].append(float(rate_flat[position]))
            label_out["fill_code"].append(int(fill_codes[slot, subspace]))
            label_out["true_code"].append(int(true_codes[slot, subspace]))

        if done % 5 == 0:
            elapsed = time.time() - started
            print(f"[official] {done + 1}/{len(usable)} "
                  f"{elapsed / max(done + 1, 1):.2f}s/row eta "
                  f"{(len(usable) - done - 1) * elapsed / max(done + 1, 1) / 60:.1f} min",
                  flush=True)

    arrays = {key: np.asarray(value)
              for key, value in {**label_out, **row_out}.items()}
    arrays["metadata"] = json.dumps({
        "protocol": "residualmem_utility_gate_labels_ama_official_v1",
        "split": args.split, "rows": len(row_out["row_pair_index"]),
        "positions_per_row": args.positions,
        "positions_per_forward": args.positions_per_forward,
        "prompt": ("deployed matched (latent row + retrieved step text row)"
                   if args.prompt == "matched"
                   else "latent row only (codes isolated)"),
        "reference": "Decode(opq codes)",
        "fill": "world model argmax",
        "bridge": str(Path(args.bridge).resolve()),
        "reader_model": args.reader_model, "dtype": args.dtype,
        "enable_thinking": enable_thinking,
        "max_model_len": args.max_model_len, "max_new_tokens": args.max_new_tokens,
        "max_answer_tokens": args.max_answer_tokens, "seed": args.seed,
        "pairs": str(Path(args.pairs).resolve()),
        "shard": [args.shard_index, args.shard_count],
    }, ensure_ascii=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)

    delta = arrays["delta_nll_bits"]
    controls = arrays["row_text_only_nll_bits"]
    summary = {
        "rows": len(arrays["row_reference_nll_bits"]),
        "labels": int(delta.size),
        "position_coverage": int(len(np.unique(arrays["position"]))),
        "mean_abs_delta_nll_bits": float(np.abs(delta).mean()) if delta.size else None,
        "median_abs_delta_nll_bits": float(np.median(np.abs(delta))) if delta.size else None,
        "max_abs_delta_nll_bits": float(np.abs(delta).max()) if delta.size else None,
        "mean_kl_bits": float(arrays["kl_bits"].mean()) if delta.size else None,
        "reference_nll_bits_mean": float(arrays["row_reference_nll_bits"].mean()),
        "seconds_per_row": round((time.time() - started) / max(len(usable), 1), 3),
        "latent_gain_bits_mean": (
            float(np.nanmean(controls - arrays["row_reference_nll_bits"]))
            if not np.all(np.isnan(controls)) else None),
    }
    (output.with_suffix(".summary.json")).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
