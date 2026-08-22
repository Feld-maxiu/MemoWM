"""Teacher embeddings for the WorldMemArena observations extracted by ``wma_extract_xbar``.

The attribution metrics compare the retrieval head's output against the official
Qwen3-VL document embedding of the *same* observation. That teacher is the fixed
point of the whole comparison, so it is encoded exactly as the WorldMemArena
Raw-Fused baseline encodes its ``fused_observation`` rows: one joint document
request carrying the screenshot and the fused observation text together, unit
normalized, 4096-d.

The text is read back from the extraction npz rather than recomputed here. Both
repositories define a ``fused_observation_text`` and they are separate
implementations; they agree on every WorldMemArena web observation today only
because the user text is always empty there, which is a coincidence of this
dataset rather than a guarantee. Reusing the stored string removes the question.

Writes one npz per sample, so a killed run resumes rather than restarts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TEACHER_PROTOCOL = "qwen3_vl_fused_observation_v1"


def load_teacher(model_path: str, device: str):
    """Load the encoder once.

    ``encode_teacher_shard.encode_payloads`` constructs a SentenceTransformer per
    call, which is right for its one-shot sharded use and wrong here: this loops
    over 27 samples and would otherwise pay a 16 GB load for each.
    """
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_path,
        device=device,
        trust_remote_code=True,
        model_kwargs={"dtype": torch.bfloat16},
    )


def encode(model, payloads: list[dict[str, str]], batch_size: int) -> np.ndarray:
    output = model.encode(
        payloads,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    output = np.asarray(output, dtype=np.float32)
    if output.shape != (len(payloads), 4096):
        raise ValueError(f"teacher dimension must be 4096, got {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("teacher output contains non-finite values")
    return output


def load_records(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(np.asarray(z["metadata"])))


def payloads_for(meta: dict) -> list[dict[str, str]]:
    """Mirror ``FusedObservationDocument.encoder_input``: text and/or image."""
    rows: list[dict[str, str]] = []
    for record in meta["records"]:
        payload: dict[str, str] = {}
        text = str(record.get("fused_text", "")).strip()
        if text:
            payload["text"] = text
        if record.get("screenshot"):
            payload["image"] = record["screenshot"]
        if not payload:
            raise ValueError(
                f"{meta['sample_id']}: observation with neither text nor image; "
                "the extraction and the benchmark disagree on what an "
                "observation row is"
            )
        rows.append(payload)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbar-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    xbar_dir = Path(args.xbar_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(xbar_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no extraction npz under {xbar_dir}")
    print(f"[teacher] {len(files)} samples")

    done = skipped = total = 0
    model = None
    for file in files:
        target = out_dir / file.name
        meta = load_records(file)
        if args.resume and target.exists():
            with np.load(target, allow_pickle=False) as existing:
                if len(existing["teacher"]) == len(meta["records"]):
                    skipped += 1
                    total += len(meta["records"])
                    continue
            print(f"[teacher] {file.stem}: stale cache, re-encoding")
        rows = payloads_for(meta)
        if model is None:  # deferred so a fully-resumed run never loads it
            print(f"[teacher] loading {args.model} on {args.device}", flush=True)
            model = load_teacher(args.model, args.device)
        vectors = encode(model, rows, args.batch_size)
        np.savez(
            target,
            teacher=vectors,
            protocol=np.asarray(TEACHER_PROTOCOL),
            sample_id=np.asarray(meta["sample_id"]),
            image_ids=np.asarray(json.dumps(
                [r.get("image_ids", []) for r in meta["records"]]
            )),
        )
        done += 1
        total += len(rows)
        print(f"[teacher] {file.stem}: {len(rows)} observations -> {target.name}", flush=True)
    print(f"[teacher] done={done} skipped={skipped} observations={total}")


if __name__ == "__main__":
    main()
