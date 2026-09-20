"""Precompute a teacher that actually read the screen.

Both distillation terms need a teacher trajectory. Until now the only one was
``fused_text`` -- roughly 89 tokens of caption, no image -- and that caption is
embedded *verbatim* in the AXTree the student already receives
(``<ref=2 tag=textarea value="This is a Windows 10 desktop..."/>``). The
teacher's entire input was a subset of the student's, so the term was teaching
the latent to reproduce text it already held. The only thing the student has
that such a teacher does not is the screenshot, and the KL said nothing about
it.

This builds the teacher that does: ``[screenshot, AXTree, probe]`` through the
frozen model, greedily continued, with the per-position distribution over that
continuation stored.

**Why precompute.** That teacher forward is ~1172 tokens through a vision
tower. Running it every micro-batch would make training three to four times
slower. But the model is frozen, decoding is greedy and the inputs are fixed, so
the result is deterministic and can be computed once.

``train_reader_qa.py:19-22`` rejects exactly this: "cheaper than the storage and
simpler than a top-k approximation." That is correct for its teacher, which
reads 89 tokens of text. It is not correct for one that reads the screenshot.
The premise changed, not the reasoning.

**Why top-k rather than full logits.** The vocabulary is 248,320, so a full
cache is 48 MB per span -- 0.2 TB in total. Top-128 is 98 KB per span, 1.7 GB in
total. Truncating at the teacher's top-k is principled for this KL direction
rather than merely cheap: every term is weighted by a teacher probability, so
the dropped tail is the smallest-weighted part and the residual mass bounds the
error. This records that mass and refuses to write a cache whose median falls
below ``--min-coverage``.

**Why ``enable_thinking=False``.** ``prepare_inputs(..., "instruct")`` renders a
prompt ending in ``<|im_start|>assistant\\n<think>\\n``, and greedy decoding from
there spends the whole budget narrating an approach rather than describing the
screen -- the failure already documented at ``connector_fidelity.py:201`` and
``instruct_bridge.py:361``. The encoder side is untouched: it never generates,
so it never needed the flag, and every existing artifact stays byte-identical.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from residualmem.latent.instruct_bridge import OBSERVATION_TEACHER_PROTOCOL

from .extract_qwen import _load_model, prepare_inputs
from .observation_kl_precheck import PROBES
from .reader_losses import INSTRUCTION
from .train_qformer_joint import PAIRS_PROTOCOL


def teacher_topk(model, inputs, span_ids, *, k: int):
    """Per-position top-k over ``span_ids``, teacher-forced after ``inputs``.

    Returns ``(index, logprob, coverage)``. ``coverage`` is the probability mass
    the top-k retains at each position -- the error bound on the truncation.
    """
    embed = model.get_input_embeddings()
    dtype = embed.weight.dtype
    pieces = torch.cat((embed(inputs["input_ids"]).to(dtype), embed(span_ids).to(dtype)), 1)
    attention = torch.cat((inputs["attention_mask"], torch.ones_like(span_ids)), 1)
    extra = {n: inputs[n] for n in ("pixel_values", "image_grid_thw") if n in inputs}
    with torch.inference_mode():
        out = model(inputs_embeds=pieces, attention_mask=attention,
                    use_cache=False, **extra)
    length = int(span_ids.shape[1])
    logits = out.logits[:, -length - 1 : -1].float()
    logprob = F.log_softmax(logits, -1)
    top_logprob, top_index = logprob.topk(k, dim=-1)
    coverage = top_logprob.exp().sum(-1)
    return top_index[0], top_logprob[0], coverage[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbar-dir", required=True,
                        help="extraction carrying synthetic_axtree and records")
    parser.add_argument("--pairs", required=True, help="build_qformer_qa_pairs output")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--min-coverage", type=float, default=0.99,
                        help="refuse to write if the median retained mass falls "
                             "below this; the truncation error would be larger "
                             "than the signal it is meant to carry")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error(f"rank {args.rank} outside world size {args.world_size}")

    pairs = np.load(args.pairs, allow_pickle=False)
    metadata = json.loads(str(np.asarray(pairs["metadata"])))
    if metadata.get("protocol") != PAIRS_PROTOCOL:
        raise ValueError(f"pairs protocol {metadata.get('protocol')!r}")
    by_observation: dict[tuple[str, int], list[int]] = {}
    sample_ids = pairs["sample_id"].astype(str)
    record_indices = pairs["record_index"].astype(int)
    for row in range(len(sample_ids)):
        by_observation.setdefault((sample_ids[row], int(record_indices[row])), []).append(row)

    device = torch.device(args.device)
    processor, model = _load_model(args.model, device, False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Strided sharding, one process per GPU: the idiom this repo already uses
    # (extract_qwen.py:270). There is no torchrun anywhere here.
    # WMA extraction stores trajectory files flat, while the standalone
    # LongMemEval store keeps them under ``trajectories/``.  Accept both
    # layouts so the observation teacher cannot silently see zero samples when
    # called on the v4 WebChain store (the flat adapter remains optional).
    samples = sorted(Path(args.xbar_dir).glob("*.npz"))
    if not samples:
        samples = sorted((Path(args.xbar_dir) / "trajectories").glob("*.npz"))
    mine = samples[args.rank :: args.world_size]
    print(f"[teacher] rank {args.rank}/{args.world_size}: {len(mine)}/{len(samples)} samples, "
          f"probes {sorted(PROBES)}, top-{args.topk}", flush=True)

    done = skipped = 0
    coverages: list[float] = []
    started = time.time()
    for path in mine:
        target = output / f"{path.stem}.npz"
        if args.resume and target.exists():
            skipped += 1
            continue
        with np.load(path, allow_pickle=False) as data:
            records = json.loads(str(np.asarray(data["metadata"]))).get("records") or []
        arrays: dict[str, np.ndarray] = {}
        entries = []
        for index, record in enumerate(records):
            if not record.get("synthetic_axtree"):
                raise ValueError(f"{path.stem}[{index}] has no synthetic_axtree")
            with Image.open(record["screenshot"]) as handle:
                image = handle.convert("RGB")
            dom = record["synthetic_axtree"]

            for name in sorted(PROBES):
                inputs, truncated, kept, _full = prepare_inputs(
                    processor, image, dom, PROBES[name], 8192, "instruct",
                    enable_thinking=False,
                )
                if truncated:
                    raise ValueError(
                        f"{path.stem}[{index}] exceeded max length: kept {kept}/{len(dom)}"
                    )
                inputs = inputs.to(device)
                prompt_len = int(inputs["input_ids"].shape[1])
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False,
                                               max_new_tokens=args.max_new_tokens)
                # The input_ids path returns prompt+new; the inputs_embeds path
                # returns only new. Conflating the two silently prepends the
                # entire prompt as if the teacher had said it.
                continuation = generated[:, prompt_len:]
                if int(continuation.shape[1]) == 0:
                    raise ValueError(f"{path.stem}[{index}] {name}: empty continuation")
                idx, lp, cov = teacher_topk(model, inputs, continuation, k=args.topk)
                key = f"{index:04d}/{name}"
                arrays[f"obs/{key}/ids"] = continuation[0].cpu().numpy().astype(np.int32)
                arrays[f"obs/{key}/index"] = idx.cpu().numpy().astype(np.int32)
                arrays[f"obs/{key}/logprob"] = lp.cpu().numpy().astype(np.float16)
                coverages.extend(cov.cpu().numpy().tolist())
                entries.append({"record_index": index, "probe": name,
                                "length": int(continuation.shape[1])})

            # Mode 2: the same observation teacher, scored on each gold answer.
            for row in by_observation.get((path.stem, index), []):
                answer_ids = processor.tokenizer(
                    str(pairs["answer"][row]), return_tensors="pt",
                    add_special_tokens=False, truncation=True,
                    max_length=args.max_answer_tokens,
                )["input_ids"].to(device)
                if int(answer_ids.shape[1]) == 0:
                    continue
                question = INSTRUCTION + str(pairs["question"][row])
                question_ids = processor.tokenizer(
                    question, return_tensors="pt", add_special_tokens=True
                )["input_ids"].to(device)
                span = torch.cat((question_ids, answer_ids), 1)
                base, _t, _k, _f = prepare_inputs(
                    processor, image, dom, PROBES["P1"], 8192, "instruct",
                    enable_thinking=False,
                )
                idx, lp, _cov = teacher_topk(model, base.to(device), span, k=args.topk)
                keep = int(answer_ids.shape[1])
                arrays[f"qa/{row:05d}/index"] = idx[-keep:].cpu().numpy().astype(np.int32)
                arrays[f"qa/{row:05d}/logprob"] = lp[-keep:].cpu().numpy().astype(np.float16)

        np.savez(target, metadata=np.asarray(json.dumps({
            "protocol": OBSERVATION_TEACHER_PROTOCOL,
            "sample_id": path.stem, "topk": args.topk,
            "max_new_tokens": args.max_new_tokens,
            "probes": {n: PROBES[n] for n in sorted(PROBES)},
            "entries": entries,
        })), **arrays)
        done += 1
        rate = done / max(time.time() - started, 1e-9)
        print(f"[teacher] {path.stem}: {len(records)} observations  "
              f"({done}/{len(mine) - skipped}, {rate * 3600:.0f}/h)", flush=True)

    if coverages:
        median = statistics.median(coverages)
        print(f"\n[teacher] rank {args.rank}: top-{args.topk} retained mass "
              f"median {median:.6f}  min {min(coverages):.6f}  "
              f"over {len(coverages)} positions", flush=True)
        if median < args.min_coverage:
            raise ValueError(
                f"median retained mass {median:.6f} is below --min-coverage "
                f"{args.min_coverage}: the truncation error would be larger than "
                "the signal this cache is meant to carry"
            )
    print(f"[teacher] rank {args.rank} done={done} skipped={skipped}")


if __name__ == "__main__":
    main()
