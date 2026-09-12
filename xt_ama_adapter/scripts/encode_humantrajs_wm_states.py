"""Encode humantrajs pseudo-web store states into QFormer xbar shards.

This is the AMA-consistent encode stage of the WM data supply loop: each store
state's pseudo-web AXTree text is passed through the *same* frozen
``QFormerXTDocumentEncoder`` used by the AMA evaluation cache stage, so the
world model is trained and later queried in the exact latent space the
benchmark answers were produced in.

Output shards follow the ``qformer_pq`` contract (keys ``xbar`` / ``state_ids``
/ ``metadata`` with ``metadata.checkpoint``), so a shard set can be fed straight
to ``experiments.state_tokenizer.qformer_pq``.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np


PROTOCOL = "residualmem_humantrajs_wm_qformer_encode_v1"
_CHARS = 4000


def _load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _state_observations(store: Path) -> dict[str, dict]:
    """state_id -> store metadata record, keyed by the image content hash."""
    table: dict[str, dict] = {}
    for npz_path in sorted(store.glob("*.npz")):
        with np.load(npz_path, allow_pickle=True) as data:
            metadata = json.loads(str(data["metadata"]))
        for record in metadata["records"]:
            image_ids = record.get("image_ids") or ()
            if not image_ids:
                continue
            table[str(image_ids[0])] = record
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True,
                        help="records.jsonl from build_humantrajs_wm_records; "
                             "defines exactly which state_ids to encode")
    parser.add_argument("--residualmem-root", type=Path, required=True)
    parser.add_argument("--qwen35-model", type=Path, required=True)
    parser.add_argument("--qformer", type=Path, required=True)
    parser.add_argument("--retrieval-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    records = _load_records(args.records)
    wanted: dict[str, tuple[str, int, str]] = {}  # state_id -> (episode_id, step, split)
    for record in records:
        wanted.setdefault(record["state_id"], (
            record["episode_id"], int(record["step"]), str(record["split"])
        ))
    observations = _state_observations(args.store)
    missing = [state_id for state_id in wanted if state_id not in observations]
    if missing:
        raise SystemExit(f"{len(missing)} state_ids have no store record, e.g. {missing[:3]}")

    state_ids = sorted(wanted)
    if args.shard_count > 1:
        state_ids = [sid for i, sid in enumerate(state_ids)
                     if i % args.shard_count == args.shard_index]
    logging.info("encoding %d unique states (shard %d/%d)", len(state_ids),
                 args.shard_index, args.shard_count)

    from xt_ama_adapter.adapter import AMAXTObservation
    from xt_ama_adapter.runtime import QFormerRuntimeConfig, QFormerXTDocumentEncoder
    from xt_ama_adapter.runtime import validate_artifact_pair

    qformer_path = Path(args.qformer).resolve()
    config = QFormerRuntimeConfig(
        residualmem_root=str(Path(args.residualmem_root).resolve()),
        qwen35_model_path=str(Path(args.qwen35_model).resolve()),
        qformer_checkpoint=str(qformer_path),
        retrieval_head_checkpoint=str(Path(args.retrieval_head).resolve()),
        # Not used by the document encoder; kept for the shared config object.
        query_model_path=str(qformer_path.parent),
        device=args.device,
        queries=32,
        qk_norm=True,
        self_attention=False,
        layer=16,
        max_length=args.max_length,
        allow_truncate=True,
    )
    validate_artifact_pair(config)
    encoder = QFormerXTDocumentEncoder(config)

    states: list[np.ndarray] = []
    batch: list[AMAXTObservation] = []
    encoded_ids: list[str] = []
    started = time.time()
    for position, state_id in enumerate(state_ids, 1):
        record = observations[state_id]
        text = str(record.get("synthetic_axtree") or record.get("fused_text") or "")
        episode_id, step, _split = wanted[state_id]
        batch.append(AMAXTObservation(
            episode_id=episode_id,
            step_index=step,
            task=str(record.get("trajectory_id") or ""),
            action="",
            observation="",
            user_text="",
            captions=(text,),
        ))
        if len(batch) == args.batch_size or position == len(state_ids):
            _, latents, _flags = encoder.encode_with_latents(batch)
            for value, _valid in latents:
                states.append(value.astype(np.float32))
            encoded_ids.extend(state_ids[position - len(batch):position])
            batch = []
        if position % 100 == 0:
            rate = position / max(time.time() - started, 1e-9)
            logging.info("%d/%d  %.2f states/s  eta %.1f min",
                         position, len(state_ids), rate,
                         (len(state_ids) - position) / max(rate, 1e-9) / 60)

    xbar = np.stack(states)
    metadata = {
        "protocol": PROTOCOL,
        "checkpoint": str(qformer_path),
        "queries": 32,
        "count": len(encoded_ids),
        "source": "humantrajs_pseudo_web",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        xbar=xbar,
        # Fixed-width unicode keeps the shard readable by numpy<2 (object
        # arrays pickled under numpy 2 are not).
        state_ids=np.asarray(encoded_ids),
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    logging.info("wrote %s  xbar %s", args.output, xbar.shape)


if __name__ == "__main__":
    main()
