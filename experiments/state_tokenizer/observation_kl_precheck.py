"""Does ``KL_obs`` measure content, or does it measure position?

The proposed observation-distillation term compares a teacher that reads the
raw observation against a student that reads a 16-slot latent, at the positions
of a continuation the teacher generated itself::

    teacher:  [image ~880, DOM ~167, probe, continuation]   continuation at ~1100
    student:  [latent 16,            probe, continuation]   continuation at ~50

That is a twenty-fold position offset. The existing ``L_distill`` has the same
asymmetry at five-fold (89 text tokens against 16 slots) and works, but nothing
in this repo has run it at this scale, and under RoPE the same tokens at wildly
different positions produce different logits for reasons that have nothing to
do with what is in memory. If the term is dominated by that, thirteen hours of
training would optimize an artifact.

The discriminative check: score each teacher against its *own* student and
against a student built from a **different observation**. A loss that measures
content separates them; a loss that measures position does not.

Reported alongside: per-position KL. If the teacher opens every continuation
with the same stock phrase, the first positions carry no observation signal and
their KL is unrelated to the latent -- visible here, invisible in the mean.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from residualmem.latent.qformer_runtime import QFormerInstructTokenizer

from .extract_qwen import prepare_inputs
from .trunk_states import collate, trunk_states

# All English: the evaluation is English, mode 2's questions and gold answers
# are English, and keeping the reader side monolingual removes a variable. The
# encoder-side Chinese OBSERVATION_PROMPT is untouched -- it is baked into every
# existing artifact and serves a different purpose.
PROBES = {
    "P1": ("Faithfully describe the current screen state: visible text, input values, "
           "control types, selected/focused/enabled states, and spatial relations."),
    "P2": "List the interactive controls visible on this screen and their states.",
    "P3": "Describe the layout of this screen from top to bottom.",
    "P4": "What text is currently visible on this screen?",     # held out from training
}


def teacher_continuation(processor, model, image, dom, probe, *, max_new_tokens, device):
    """Greedy continuation from the raw observation, plus the prompt inputs.

    ``enable_thinking=False`` is load-bearing. ``prepare_inputs(..., "instruct")``
    renders a prompt ending in ``<|im_start|>assistant\\n<think>\\n``, and greedy
    decoding from that spends the whole budget on a reasoning preamble -- the
    failure mode already documented at ``connector_fidelity.py:201`` and
    ``instruct_bridge.py:361``.
    """
    inputs, truncated, kept, _full = prepare_inputs(
        processor, image, dom, probe, 8192, "instruct", enable_thinking=False
    )
    if truncated:
        raise ValueError(f"observation exceeded max length: kept {kept}/{len(dom)} DOM chars")
    inputs = inputs.to(device)
    prompt_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    # The input_ids path returns prompt+new; the inputs_embeds path returns only
    # new. Mixing the two conventions silently prepends the whole prompt.
    return inputs, generated[0, prompt_len:].unsqueeze(0)


def logits_over(model, embed, prefix_embeds, prefix_mask, probe_ids, continuation_ids,
                *, pad_prefix_to: int = 0):
    """Logits at the continuation positions, whatever the prefix is.

    ``pad_prefix_to`` left-pads the prefix with masked-out positions so the
    continuation lands at the same absolute index it does on the teacher side.
    That is the control for the term's structural risk: the student carries 16
    slots and the teacher ~1100 real tokens, so under RoPE the same continuation
    sits twenty times further along on one side than the other, and a KL that
    merely reads that offset would look like a content signal.
    """
    dtype = embed.weight.dtype
    pieces = [prefix_embeds.to(dtype), embed(probe_ids), embed(continuation_ids)]
    masks = [prefix_mask, torch.ones_like(probe_ids), torch.ones_like(continuation_ids)]
    if pad_prefix_to > prefix_embeds.shape[1]:
        gap = pad_prefix_to - prefix_embeds.shape[1]
        pad = torch.zeros((1, gap, prefix_embeds.shape[2]),
                          dtype=dtype, device=prefix_embeds.device)
        pad_mask = torch.zeros((1, gap), dtype=prefix_mask.dtype,
                               device=prefix_mask.device)
        pieces.insert(0, pad)
        masks.insert(0, pad_mask)
    with torch.inference_mode():
        out = model(
            inputs_embeds=torch.cat(pieces, 1),
            attention_mask=torch.cat(masks, 1).to(prefix_mask.dtype),
            use_cache=False,
        )
    length = int(continuation_ids.shape[1])
    return out.logits[:, -length - 1 : -1].float()


def continuation_overlap(continuations, tokenizer):
    """How much of the target is boilerplate rather than observation content.

    The gap between matched and mismatched KL is an average over every position
    of the teacher's continuation. If most of those positions are a house style
    the model uses for any screen -- "The screen shows a web page with..." --
    then the content-bearing positions are a small minority and the mean is
    diluted by the rest. A small gap would then say nothing about how much the
    latent encodes; it would say the target is mostly boilerplate.

    Measured two ways, both over token ids so no tokenizer round-trip is
    involved: the fraction of each continuation's tokens that also appear in a
    *different observation's* continuation, and the length of the common prefix
    between consecutive pairs.
    """
    ids = [c[0].tolist() for c in continuations]
    shared, prefixes = [], []
    for i, own in enumerate(ids):
        other = ids[(i + 1) % len(ids)]
        shared.append(len(set(own) & set(other)) / max(len(set(own)), 1))
        common = 0
        for a, b in zip(own, other):
            if a != b:
                break
            common += 1
        prefixes.append(common)
    distinct = [len(set(c)) / max(len(c), 1) for c in ids]
    return {
        "median_token_overlap_with_other_observation": statistics.median(shared),
        "median_distinct_token_ratio": statistics.median(distinct),
        "median_common_prefix_tokens": statistics.median(prefixes),
        "max_common_prefix_tokens": max(prefixes),
        "median_length": statistics.median(len(c) for c in ids),
    }


def per_position_kl(student_logits, teacher_logits):
    """KL(teacher || student) at each position, in nats. No reduction."""
    student = F.log_softmax(student_logits, -1)
    teacher = F.log_softmax(teacher_logits, -1)
    return (teacher.exp() * (teacher - student)).sum(-1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbar-dir", required=True)
    parser.add_argument("--checkpoint", help="a joint Q-Former checkpoint")
    parser.add_argument("--pooling-connector",
                        help="score the fixed pooling's 64 slots instead of a "
                             "resampler. Its xbar is already cached in the "
                             "extraction, so only the connector runs. Neither "
                             "representation was ever trained on this objective, "
                             "so a larger gap here isolates slot budget from "
                             "everything else")
    parser.add_argument("--model", required=True)
    parser.add_argument("--queries", type=int, default=16,
                        help="query count of the --checkpoint being scored")
    parser.add_argument("--probe", default="P1", choices=sorted(PROBES))
    parser.add_argument("--observations", type=int, default=64)
    parser.add_argument("--pad-student", action="store_true",
                        help="left-pad the student prefix to the teacher's length, "
                             "so the continuation sits at the same absolute "
                             "position on both sides")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output")
    args = parser.parse_args()

    if bool(args.checkpoint) == bool(args.pooling_connector):
        parser.error("pass exactly one of --checkpoint or --pooling-connector")
    if args.pooling_connector:
        from residualmem.latent.instruct_bridge import (
            READER_BRIDGE_PROTOCOL, InputSoftTokenConnector, load_bridge,
        )

        from .extract_qwen import _load_model
        device = torch.device(args.device)
        processor, model = _load_model(args.model, device, False)
        connector = InputSoftTokenConnector()
        load_bridge(args.pooling_connector, connector,
                    expected_protocol=READER_BRIDGE_PROTOCOL)
        connector.to(device).eval()
        tokenizer = None
    else:
        tokenizer = QFormerInstructTokenizer(
            model_path=args.model, checkpoint=args.checkpoint,
            queries=args.queries, device=args.device,
        )
        processor, model = tokenizer.processor, tokenizer.model
        device = tokenizer.device
        connector = None
    embed = model.get_input_embeddings()
    probe = PROBES[args.probe]
    probe_ids = processor.tokenizer(
        probe, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)

    rng = np.random.default_rng(args.seed)
    samples = sorted(Path(args.xbar_dir).glob("*.npz"))
    picked = []
    for path in samples:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata"])))
        for index, record in enumerate(metadata.get("records") or []):
            if record.get("synthetic_axtree"):
                picked.append((path.stem, index, record))
    order = rng.permutation(len(picked))[: args.observations]
    picked = [picked[int(i)] for i in order]
    print(f"[precheck] probe {args.probe}  {len(picked)} observations", flush=True)

    latents, teachers, continuations, prefix_lengths = [], [], [], []
    for position, (sample_id, index, record) in enumerate(picked):
        with Image.open(record["screenshot"]) as handle:
            image = handle.convert("RGB")
        dom = record["synthetic_axtree"]
        inputs, continuation = teacher_continuation(
            processor, model, image, dom, probe,
            max_new_tokens=args.max_new_tokens, device=device,
        )
        if continuation.shape[1] < 8:
            print(f"[precheck] {sample_id}[{index}] continuation only "
                  f"{continuation.shape[1]} tokens, skipped", flush=True)
            continue
        teacher_logits = logits_over(
            model, embed,
            embed(inputs["input_ids"]), inputs["attention_mask"],
            probe_ids, continuation,
        )
        if connector is not None:
            # The pooling's xbar is already on disk from the extraction; the
            # trunk never has to run again for this side.
            with np.load(Path(args.xbar_dir) / f"{sample_id}.npz",
                         allow_pickle=False) as data:
                pooled = np.asarray(data[f"m11/xbar/{index:04d}"], np.float32)
            xbar = torch.as_tensor(pooled[None], device=device)
            valid = torch.ones(xbar.shape[:-1], dtype=torch.bool, device=device)
            with torch.no_grad():
                soft = connector(xbar, valid)
        else:
            states = trunk_states(processor, model, image, dom, device=device)
            with torch.no_grad():
                soft, _xbar, valid = tokenizer.reader(*collate([states]))
        latents.append((soft, valid))
        teachers.append(teacher_logits)
        continuations.append(continuation)
        prefix_lengths.append(int(inputs["input_ids"].shape[1]))
        if position % 8 == 0:
            print(f"[precheck] {position + 1}/{len(picked)}", flush=True)

    matched, mismatched, position_curve = [], [], []
    for i, (soft, valid) in enumerate(latents):
        pad_to = prefix_lengths[i] if args.pad_student else 0
        own = logits_over(model, embed, soft, valid.to(torch.long),
                          probe_ids, continuations[i], pad_prefix_to=pad_to)
        kl = per_position_kl(own, teachers[i])
        matched.append(float(kl.mean()))
        position_curve.append(kl.float().cpu().numpy())
        # A different observation's latent, against this teacher and this
        # continuation. Everything except the memory content is held fixed.
        other = (i + 1) % len(latents)
        swapped = logits_over(model, embed, latents[other][0],
                              latents[other][1].to(torch.long),
                              probe_ids, continuations[i], pad_prefix_to=pad_to)
        mismatched.append(float(per_position_kl(swapped, teachers[i]).mean()))

    width = min(len(c) for c in position_curve)
    curve = np.stack([c[:width] for c in position_curve]).mean(0)
    report = {
        "probe": args.probe,
        "pad_student": bool(args.pad_student),
        "observations": len(matched),
        "matched_kl_mean": statistics.mean(matched),
        "mismatched_kl_mean": statistics.mean(mismatched),
        "gap": statistics.mean(mismatched) - statistics.mean(matched),
        "matched_beats_mismatched": sum(
            1 for a, b in zip(matched, mismatched) if a < b
        ) / max(len(matched), 1),
        "per_position_kl": curve.tolist(),
        "continuations": continuation_overlap(continuations, processor.tokenizer),
        "samples": [
            processor.tokenizer.decode(c[0], skip_special_tokens=True)
            for c in continuations[:6]
        ],
        "representation": ("pooling-64" if args.pooling_connector else "qformer"),
        "slots": int(latents[0][0].shape[1]) if latents else 0,
        "checkpoint": str(Path(args.checkpoint or args.pooling_connector).resolve()),
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))

    print(f"\n[{args.probe}]  {report['representation']} ({report['slots']} slots)  "
          f"{report['observations']} observations"
          f"{'  (student padded to teacher position)' if args.pad_student else ''}")
    print(f"  matched     KL {report['matched_kl_mean']:.4f}")
    print(f"  mismatched  KL {report['mismatched_kl_mean']:.4f}")
    print(f"  gap            {report['gap']:+.4f}   "
          f"matched lower on {report['matched_beats_mismatched']:.1%} of observations")
    print("  per-position KL (matched): " +
          " ".join(f"{v:.2f}" for v in curve[:16]) + (" ..." if width > 16 else ""))
    c = report["continuations"]
    print(f"\n  target text: median {c['median_length']:.0f} tokens, "
          f"{c['median_distinct_token_ratio']:.1%} distinct")
    print(f"    token overlap with ANOTHER observation's continuation: "
          f"{c['median_token_overlap_with_other_observation']:.1%}   "
          f"<- high means the target is boilerplate, not content")
    print(f"    common prefix with it: median {c['median_common_prefix_tokens']:.0f} tokens, "
          f"max {c['max_common_prefix_tokens']}")
    for n, text in enumerate(report["samples"][:3]):
        print(f"    [{n}] {text[:150]!r}")
    if report["gap"] <= 0:
        print("\n  VERDICT: the term does not separate content. Do not train on it.")
    elif report["matched_beats_mismatched"] < 0.9:
        print("\n  VERDICT: separates on average but not per observation. Investigate first.")
    else:
        print("\n  VERDICT: separates content. Safe to train.")


if __name__ == "__main__":
    main()
