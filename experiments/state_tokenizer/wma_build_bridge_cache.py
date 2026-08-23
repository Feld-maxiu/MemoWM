"""Assemble a retrieval-bridge cache from WorldMemArena extractions.

``build_instruct_bridge_cache.py`` builds the in-domain cache by walking the
BrowserGym records and the 4096-d feature store. The WorldMemArena side already
has both halves on disk from separate passes -- ``wma_extract_xbar.py`` wrote the
``xbar``/``valid`` per sample and ``wma_encode_teacher.py`` wrote the matching
fused-observation teacher rows -- so this only has to pair and stack them into
the format ``train_retrieval_bridge.load_cache`` accepts.

**The split is by sample, not by observation.** Observations inside one sample
come from consecutive rounds of a single session and are heavily correlated;
splitting by observation would put near-identical rows on both sides of the
train/validation boundary and report a fit that does not exist. This is the same
discipline the PCA probes use.

The pairing is positional -- row *i* of ``m11/xbar/NNNN`` goes with row *i* of
``teacher`` -- which is safe only because both passes iterate the same adapter
over the same sessions in the same order. The count is asserted per sample
rather than assumed, because a silent off-by-one here would train the head
against the wrong targets and still converge to something plausible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from residualmem.latent.instruct_bridge import (
    BRIDGE_CACHE_PROTOCOL,
    FUSED_OBSERVATION_TEACHER_PROTOCOL,
)


def load_sample(xbar_path: Path, teacher_path: Path, arm: str):
    with np.load(xbar_path, allow_pickle=False) as z:
        keys = sorted(k for k in z.files if k.startswith(f"{arm}/xbar/"))
        if not keys:
            raise ValueError(f"{xbar_path.name} has no {arm} arm")
        xbar = np.stack([np.asarray(z[k], np.float32) for k in keys])
        valid = np.stack(
            [np.asarray(z[f"{arm}/valid/{k.rsplit('/', 1)[1]}"], bool) for k in keys]
        )
        metadata = json.loads(str(np.asarray(z["metadata"])))
    # The reader trainer reconstructs this text from the latent. It is the same
    # string the teacher embedded, so it is the WorldMemArena analogue of
    # browsergym_teacher_text and the two domains train on a matched target.
    # Note it is a re-serialization of the same content, not the synthetic
    # AXTree the tokenizer actually consumed -- only that one's length survives.
    records = metadata.get("records") or []
    if len(records) != len(xbar):
        raise ValueError(
            f"{xbar_path.stem}: {len(records)} records against {len(xbar)} states; "
            "the metadata is not positionally aligned with the arrays"
        )
    target_text = np.asarray([str(record.get("fused_text", "")) for record in records])
    with np.load(teacher_path, allow_pickle=False) as z:
        teacher = np.asarray(z["teacher"], np.float32)
        protocol = str(np.asarray(z["protocol"]))
    if protocol != FUSED_OBSERVATION_TEACHER_PROTOCOL:
        raise ValueError(f"{teacher_path.name} teacher protocol {protocol!r}")
    if len(teacher) != len(xbar):
        raise ValueError(
            f"{xbar_path.stem}: {len(xbar)} xbar rows against {len(teacher)} teacher rows; "
            "the two passes disagree on the observation set"
        )
    return xbar, valid, teacher, target_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbar-dir", required=True)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", default="m11")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()

    xbar_dir, teacher_dir = Path(args.xbar_dir), Path(args.teacher_dir)
    stems = sorted(p.stem for p in xbar_dir.glob("*.npz"))
    if not stems:
        raise FileNotFoundError(f"no npz under {xbar_dir}")
    missing = [s for s in stems if not (teacher_dir / f"{s}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"no teacher for {len(missing)} samples, e.g. {missing[:3]}")

    # Deterministic sample-level split. Held-out *samples*, not rows.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(stems))
    n_val = max(1, int(round(len(stems) * args.validation_fraction)))
    val_stems = {stems[i] for i in order[:n_val]}

    xbars, valids, teachers, texts, splits, sample_ids = [], [], [], [], [], []
    for stem in stems:
        x, v, t, target = load_sample(
            xbar_dir / f"{stem}.npz", teacher_dir / f"{stem}.npz", args.arm
        )
        xbars.append(x)
        valids.append(v)
        teachers.append(t)
        texts.append(target)
        tag = "validation" if stem in val_stems else "train"
        splits.extend([tag] * len(x))
        sample_ids.extend([stem] * len(x))
        print(f"[cache] {stem:22s} {len(x):4d} rows -> {tag}", flush=True)

    xbar = np.concatenate(xbars)
    valid = np.concatenate(valids)
    teacher = np.concatenate(teachers)
    target_text = np.concatenate(texts)
    empty = int((target_text == "").sum())
    if empty:
        print(f"[cache] {empty} rows have no fused_text; the reader trainer will "
              f"reconstruct an empty string for them")
    split = np.asarray(splits)
    norms = np.linalg.norm(teacher, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        # The trainer's InfoNCE and cosine terms both assume unit teachers; the
        # in-domain builder normalizes before writing, so match it here.
        teacher = teacher / np.maximum(norms, 1e-12)[:, None]
        print(f"[cache] re-normalized teacher rows (was {norms.min():.4f}..{norms.max():.4f})")

    metadata = {
        "protocol": BRIDGE_CACHE_PROTOCOL,
        "teacher_protocol": FUSED_OBSERVATION_TEACHER_PROTOCOL,
        "representation": "xbar",
        "source": "worldmemarena",
        "xbar_dir": str(xbar_dir),
        "teacher_dir": str(teacher_dir),
        "arm": args.arm,
        "seed": args.seed,
        "split_unit": "sample",
        "samples_total": len(stems),
        "samples_validation": sorted(val_stems),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        xbar=xbar,
        valid=valid,
        teacher_fused_embedding=teacher,
        target_text=target_text,
        split=split,
        sample_id=np.asarray(sample_ids),
        metadata=np.asarray(json.dumps(metadata)),
    )
    n_train = int((split == "train").sum())
    print(f"\n{len(stems)} samples -> {len(xbar)} rows "
          f"({n_train} train / {len(xbar) - n_train} validation), "
          f"{len(val_stems)} held-out samples")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
