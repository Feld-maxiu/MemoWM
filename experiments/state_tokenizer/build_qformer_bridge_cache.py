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
import hashlib
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
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="override the checkpoint's own qk_norm setting; "
                             "unset means read it from the checkpoint metadata. "
                             "QK-norm adds no parameters, so getting this wrong "
                             "loads cleanly and encodes with the wrong forward")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1,
                        help="shard the store across N workers by sample stem")
    parser.add_argument("--text-only", action="store_true",
                        help="encode with a blank white image instead of reading "
                             "record screenshots, matching the AMA runtime path")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must satisfy 0 <= rank < world-size")

    tokenizer = QFormerInstructTokenizer(
        model_path=args.model, checkpoint=args.checkpoint, queries=args.queries,
        layers=args.qformer_layers, self_attention=args.self_attention,
        qk_norm=args.qk_norm, device=args.device,
    )
    processor, model, reader = tokenizer.processor, tokenizer.model, tokenizer.reader

    samples = sorted(Path(args.xbar_dir).glob("*.npz"))
    if args.world_size > 1:
        samples = [
            path for position, path in enumerate(samples)
            if position % args.world_size == args.rank
        ]
    stems = [p.stem for p in samples]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(stems))
    held = {stems[i] for i in order[: max(1, int(round(len(stems) * args.validation_fraction)))]}

    if args.rank == 0:
        print(f"[cache] {len(stems)} samples on rank {args.rank}/{args.world_size}", flush=True)

    xbars, valids, teachers, splits, sample_ids, step_indices, texts = (
        [], [], [], [], [], [], []
    )
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
            if args.text_only:
                image = Image.new("RGB", (1280, 720), "white")
            else:
                with Image.open(record["screenshot"]) as handle:
                    image = handle.convert("RGB")
            states = trunk_states(processor, model, image, record["synthetic_axtree"],
                                  layer=tokenizer.layer, device=tokenizer.device)
            with torch.no_grad():
                xbar, valid = reader.encode(*collate([states]))
            xbars.append(xbar[0].float().cpu().numpy())
            valids.append(valid[0].cpu().numpy())
            teachers.append(teacher[index])
            # Prefer the record's own split when the extraction carried one (the
            # HumanTrajs manifest marks train/validation/test per trajectory);
            # fall back to the held-out-sample discipline otherwise.
            record_split = str(record.get("split", ""))
            if record_split in {"train", "validation", "test"}:
                splits.append(record_split)
            else:
                splits.append("validation" if path.stem in held else "train")
            sample_ids.append(path.stem)
            # step_index: the record's position inside its trajectory, which is
            # how downstream (_unique_states / merge) orders and deduplicates.
            step = record.get("step_idx", record.get("step_index", index))
            step_indices.append(int(step))
            texts.append(str(record.get("fused_text", "")))
        print(f"[cache] {path.stem}: {len(records)} states", flush=True)

    embeddings = np.stack(teachers).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    embeddings = embeddings / np.maximum(norms, 1e-12)[:, None]

    # Bind the cache to the exact Q-Former that produced it.  frozen.pt's own
    # file hash is the artifact hash the retrieval head stores, and the
    # checkpoint metadata carries the model-state hash.
    checkpoint_path = str(Path(args.checkpoint).resolve())
    artifact_sha256 = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    qformer_state_sha256 = str(tokenizer.metadata.get("qformer_sha256", ""))
    resolved_qk_norm = bool(
        args.qk_norm if args.qk_norm is not None
        else tokenizer.metadata.get("qk_norm", False)
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        xbar=np.stack(xbars).astype(np.float32),
        valid=np.stack(valids).astype(bool),
        teacher_fused_embedding=embeddings,
        split=np.asarray(splits),
        sample_id=np.asarray(sample_ids),
        step_index=np.asarray(step_indices, dtype=np.int64),
        target_text=np.asarray(texts),
        metadata=np.asarray(json.dumps({
            "protocol": BRIDGE_CACHE_PROTOCOL,
            "teacher_protocol": FUSED_OBSERVATION_TEACHER_PROTOCOL,
            "representation": "xbar",
            "source": "qformer",
            "checkpoint": checkpoint_path,
            "queries": int(np.stack(xbars).shape[1]),
            "qformer_artifact_sha256": artifact_sha256,
            "qformer_state_sha256": qformer_state_sha256,
            "qk_norm": resolved_qk_norm,
            "self_attention": bool(args.self_attention),
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
