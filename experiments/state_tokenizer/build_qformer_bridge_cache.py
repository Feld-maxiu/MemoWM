"""Run a trained Q-Former over the corpus and emit an A1 training cache.

The resampler's states live in a coordinate system nothing else has been fitted
to, so the retrieval head has to be retrained on them. ``L_sem`` already trains
a ``MaskedAttentionRetrievalHead`` inside the joint checkpoint, but only against
a cosine to the teacher -- there is no contrastive term and, at micro-batch 1,
no batch to draw negatives from. Measured, that projection points the right way
and barely discriminates: 0.82 cosine to its own teacher against 0.65 to
everyone else's, and R@1 of 0.045 where the shipped head reaches 0.84.

So this writes the cache ``train_retrieval_bridge`` consumes, which trains with
``symmetric_infonce`` over a real batch. Same schema as
``wma_build_bridge_cache``, same by-sample split discipline, so the two heads
are trained under the same protocol and their numbers are comparable.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from residualmem.latent.instruct_bridge import (
    BRIDGE_CACHE_PROTOCOL,
    FUSED_OBSERVATION_TEACHER_PROTOCOL,
)
from residualmem.latent.qformer_runtime import QFormerInstructTokenizer

from .trunk_states import collate, trunk_states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbar-dir", required=True,
                        help="extraction carrying synthetic_axtree and records")
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--checkpoint", required=True, help="a joint Q-Former checkpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--qformer-layers", type=int, default=4)
    parser.add_argument("--self-attention", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()

    tokenizer = QFormerInstructTokenizer(
        model_path=args.model, checkpoint=args.checkpoint, queries=args.queries,
        layers=args.qformer_layers, self_attention=args.self_attention,
        device=args.device,
    )
    processor, model, reader = tokenizer.processor, tokenizer.model, tokenizer.reader

    samples = sorted(Path(args.xbar_dir).glob("*.npz"))
    stems = [p.stem for p in samples]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(stems))
    held = {stems[i] for i in order[: max(1, int(round(len(stems) * args.validation_fraction)))]}

    xbars, valids, teachers, splits, sample_ids, texts = [], [], [], [], [], []
    for path in samples:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata"])))
        records = metadata.get("records") or []
        with np.load(Path(args.teacher_dir) / f"{path.stem}.npz", allow_pickle=False) as data:
            protocol = str(np.asarray(data["protocol"]))
            teacher = np.asarray(data["teacher"], np.float32)
        if protocol != FUSED_OBSERVATION_TEACHER_PROTOCOL:
            raise ValueError(f"{path.stem}: teacher protocol {protocol!r}")
        if len(teacher) != len(records):
            raise ValueError(
                f"{path.stem}: {len(records)} records against {len(teacher)} teacher rows"
            )
        for index, record in enumerate(records):
            if not record.get("synthetic_axtree"):
                raise ValueError(f"{path.stem}[{index}] has no synthetic_axtree")
            with Image.open(record["screenshot"]) as handle:
                image = handle.convert("RGB")
            states = trunk_states(processor, model, image, record["synthetic_axtree"],
                                  layer=tokenizer.layer, device=tokenizer.device)
            with torch.no_grad():
                xbar, valid = reader.encode(*collate([states]))
            xbars.append(xbar[0].float().cpu().numpy())
            valids.append(valid[0].cpu().numpy())
            teachers.append(teacher[index])
            splits.append("validation" if path.stem in held else "train")
            sample_ids.append(path.stem)
            texts.append(str(record.get("fused_text", "")))
        print(f"[cache] {path.stem}: {len(records)} states", flush=True)

    embeddings = np.stack(teachers).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings / np.maximum(norms, 1e-12)[:, None]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        xbar=np.stack(xbars).astype(np.float32),
        valid=np.stack(valids).astype(bool),
        teacher_fused_embedding=embeddings,
        split=np.asarray(splits),
        sample_id=np.asarray(sample_ids),
        target_text=np.asarray(texts),
        metadata=np.asarray(json.dumps({
            "protocol": BRIDGE_CACHE_PROTOCOL,
            "teacher_protocol": FUSED_OBSERVATION_TEACHER_PROTOCOL,
            "representation": "xbar",
            "source": "qformer",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "queries": int(np.stack(xbars).shape[1]),
            "split_unit": "sample",
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
        })),
    )
    counts = collections.Counter(splits)
    print(f"\n{len(xbars)} states over {len(stems)} samples  {dict(counts)}")
    print(f"teacher norms renormalized from {norms.min():.6f}..{norms.max():.6f}")
    print(f"wrote {args.output} ({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
