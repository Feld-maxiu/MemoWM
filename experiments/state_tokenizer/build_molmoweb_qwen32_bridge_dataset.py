"""Bind verified MolmoWeb QA to its matched frozen-QFormer state for bridge training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from residualmem.latent.instruct_bridge import (
    BRIDGE_CACHE_PROTOCOL,
    RETRIEVAL_BRIDGE_PROTOCOL,
)
from xt_ama_adapter.qwen32_bridge import file_sha256
from .train_qwen32_bridge import MOLMOWEB_DATA_PROTOCOL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as raw:
        cache = {name: np.asarray(raw[name]) for name in (
            "xbar", "valid", "sample_id", "step_index", "target_text", "split"
        )}
        cache_meta = json.loads(str(np.asarray(raw["metadata"]).item()))
    if cache_meta.get("protocol") != BRIDGE_CACHE_PROTOCOL:
        raise ValueError("unexpected QFormer bridge-cache protocol")
    if cache_meta.get("queries") != 32 or cache_meta.get("qk_norm") is not True:
        raise ValueError("bridge cache must be K32 with qk_norm=true")
    if cache["xbar"].shape[1:] != (32, 512):
        raise ValueError(f"unexpected xbar shape {cache['xbar'].shape}")

    head_payload = torch.load(args.head, map_location="cpu", weights_only=True)
    if head_payload.get("protocol") != RETRIEVAL_BRIDGE_PROTOCOL:
        raise ValueError("unexpected retrieval-head protocol")
    head_meta = dict(head_payload.get("metadata") or {})
    if head_meta.get("qformer_artifact_sha256") != cache_meta.get(
        "qformer_artifact_sha256"
    ):
        raise ValueError("retrieval head is not bound to this QFormer artifact")

    with np.load(args.pairs, allow_pickle=False) as raw:
        pair = {name: np.asarray(raw[name]) for name in (
            "sample_id", "record_index", "question", "answer", "split"
        )}
        pair_meta = json.loads(str(np.asarray(raw["metadata"]).item()))
    if pair_meta.get("protocol") != "molmoweb_text_qformer_qa_v1":
        raise ValueError("unexpected MolmoWeb pairs protocol")

    state_index = {}
    for index, (sample_id, step) in enumerate(zip(
        cache["sample_id"].astype(str), cache["step_index"].astype(int)
    )):
        key = (sample_id, int(step))
        if key in state_index:
            raise ValueError(f"duplicate cached state {key}")
        state_index[key] = index

    selected = []
    for row, (sample_id, record_index) in enumerate(zip(
        pair["sample_id"].astype(str), pair["record_index"].astype(int)
    )):
        key = (sample_id, int(record_index))
        if key not in state_index:
            raise ValueError(f"QA row {row} has no state {key}")
        selected.append(state_index[key])
    chosen = np.asarray(selected, dtype=np.int64)
    if not np.array_equal(cache["split"][chosen].astype(str), pair["split"].astype(str)):
        raise ValueError("QA and state splits differ")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        xbar=cache["xbar"][chosen, None].astype(np.float32),
        valid=cache["valid"][chosen, None].astype(bool),
        trajectory_id=pair["sample_id"].astype(str),
        question=pair["question"].astype(str),
        answer=pair["answer"].astype(str),
        oracle_text=cache["target_text"][chosen].astype(str),
        split=pair["split"].astype(str),
        metadata=np.asarray(json.dumps({
            "protocol": MOLMOWEB_DATA_PROTOCOL,
            "top_k": 1,
            "slots": 32,
            "state_dimension": 512,
            "selection": "matched_single_observation_no_retrieval",
            "official_ama_test_included": False,
            "qformer_artifact_sha256": cache_meta["qformer_artifact_sha256"],
            "retrieval_head_artifact_sha256": file_sha256(args.head),
            "qformer_cache_sha256": file_sha256(args.cache),
            "qformer_pairs_sha256": file_sha256(args.pairs),
            "rows": len(chosen),
        }, sort_keys=True)),
    )
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(chosen)},
                     indent=2))


if __name__ == "__main__":
    main()
