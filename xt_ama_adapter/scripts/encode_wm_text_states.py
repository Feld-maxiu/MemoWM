"""Encode a text state index into QFormer xbar shards (AMA-consistent).

Reads ``states.jsonl`` lines ``{state_id, split, text}`` and runs each state's
page text through the *same* frozen ``QFormerXTDocumentEncoder`` used by the
AMA evaluation cache stage, producing shards that satisfy the ``qformer_pq``
contract (``xbar`` / ``state_ids`` / ``metadata.checkpoint``).

The script is shard-parallel: run one process per GPU with
``--shard-index k --shard-count N``.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np


PROTOCOL = "residualmem_text_wm_qformer_encode_v1"


def _load_states(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append((row["state_id"], row["text"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states-jsonl", type=Path, required=True)
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

    states = _load_states(args.states_jsonl)
    if args.shard_count > 1:
        states = [s for i, s in enumerate(states)
                  if i % args.shard_count == args.shard_index]
    if not states:
        raise SystemExit(f"no states assigned to shard {args.shard_index}")
    logging.info("encoding %d states (shard %d/%d)", len(states),
                 args.shard_index, args.shard_count)

    from xt_ama_adapter.adapter import AMAXTObservation
    from xt_ama_adapter.runtime import (
        QFormerRuntimeConfig,
        QFormerXTDocumentEncoder,
        validate_artifact_pair,
    )

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

    state_ids = [state_id for state_id, _ in states]
    encoded_ids: list[str] = []
    batch: list[AMAXTObservation] = []
    arrays: list[np.ndarray] = []
    started = time.time()
    for position, (state_id, text) in enumerate(states, 1):
        batch.append(AMAXTObservation(
            episode_id=state_id,
            step_index=0,
            task="",
            action="",
            observation="",
            user_text="",
            captions=(text,),
        ))
        if len(batch) == args.batch_size or position == len(states):
            _, latents, _flags = encoder.encode_with_latents(batch)
            arrays.extend(value.astype(np.float32) for value, _valid in latents)
            encoded_ids.extend(state_ids[position - len(batch):position])
            batch = []
        if position % 100 == 0:
            rate = position / max(time.time() - started, 1e-9)
            logging.info("%d/%d  %.2f states/s  eta %.1f min",
                         position, len(states), rate,
                         (len(states) - position) / max(rate, 1e-9) / 60)

    if not arrays or len(arrays) != len(encoded_ids):
        raise SystemExit("encode produced no/partial states")
    xbar = np.stack(arrays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        xbar=xbar,
        # Fixed-width unicode keeps the shard readable by numpy<2.
        state_ids=np.asarray(encoded_ids),
        metadata=json.dumps({
            "protocol": PROTOCOL,
            "checkpoint": str(qformer_path),
            "queries": 32,
            "count": len(encoded_ids),
            "source": "text_states",
        }, ensure_ascii=False),
    )
    logging.info("wrote %s  xbar %s", args.output, xbar.shape)


if __name__ == "__main__":
    main()
