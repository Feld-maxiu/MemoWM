"""Quantize encoded xbar states with an *existing* PQ codebook.

Unlike ``qformer_pq`` this never fits anything: it loads the saved artifacts
(``mean`` / ``scale`` / ``centroids`` / ``rotation_bases`` / ``rotation_order``)
and reproduces the exact inference path of ``qformer_pq.encode`` so the new
states land in the same code space as the training cache.  The output npz
satisfies the ``cache_web.load_codes`` contract (``codes/<split>`` keyed by
``state_ids/<split>``) and carries ``centroids`` so downstream alphabet
inference works.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.qformer_pq import (
    decode,
    encode,
    reconstruction_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", action="append", required=True,
                        help="encoded .npz shard (xbar/state_ids); repeatable")
    parser.add_argument("--pq", type=Path, required=True,
                        help="fitted pq artifact (e.g. pq-h32-c64.npz)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-name", default="all",
                        help="split key under codes/ and state_ids/")
    parser.add_argument("--probe-count", type=int, default=512,
                        help="states used for the round-trip reconstruction check")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    with np.load(args.pq, allow_pickle=True) as data:
        mean = np.asarray(data["mean"], np.float32)
        scale = np.asarray(data["scale"], np.float32)
        centroids = np.asarray(data["centroids"], np.float32)
        bases = np.asarray(data["rotation_bases"], np.float32)
        order = np.asarray(data["rotation_order"], np.int64)
        pq_report = json.loads(str(data["report"])) if "report" in data.files else {}
    logging.info("codebook %s: categories=%d subspaces=%d slots=%d",
                 args.pq.name, centroids.shape[2], centroids.shape[1],
                 centroids.shape[0])

    codes_parts: list[np.ndarray] = []
    ids_parts: list[np.ndarray] = []
    probe_states: list[np.ndarray] = []
    probe_codes: list[np.ndarray] = []
    for path in args.states:
        with np.load(path, allow_pickle=True) as data:
            xbar = np.asarray(data["xbar"], np.float32)
            state_ids = [str(v) for v in np.asarray(data["state_ids"])]
        if len(xbar) != len(state_ids):
            raise SystemExit(f"{path}: {len(xbar)} xbar rows vs {len(state_ids)} ids")
        logging.info("encoding %d states from %s", len(xbar), Path(path).name)
        shard_codes = encode(xbar, mean, scale, centroids, bases=bases, order=order)
        codes_parts.append(shard_codes)
        ids_parts.append(np.asarray(state_ids))
        take = min(args.probe_count - len(probe_states), len(xbar))
        if take > 0:
            probe_states.extend(xbar[:take])
            probe_codes.extend(shard_codes[:take])

    codes = np.concatenate(codes_parts).astype(np.uint8)
    state_ids = np.concatenate(ids_parts)
    if len({*state_ids.tolist()}) != len(state_ids):
        raise SystemExit("duplicate state_ids across shards")

    # Round-trip check on raw xbar: decode(encode(x)) must reproduce the
    # corpus-level reconstruction quality the codebook was fitted for.
    probe = np.stack(probe_states[:args.probe_count])
    rebuilt = decode(
        np.stack(probe_codes[:args.probe_count]).astype(np.uint8),
        mean, scale, centroids, bases=bases, order=order,
    )
    metrics = reconstruction_metrics(probe, rebuilt)
    logging.info("round-trip on %d probe states: mse=%.6f r2=%.5f",
                 metrics["states"], metrics["mse"], metrics["r2"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **{
            f"codes/{args.split_name}": codes,
            f"state_ids/{args.split_name}": state_ids,
            "centroids": centroids,
            "mean": mean,
            "scale": scale,
            "metadata": json.dumps({
                "protocol": "residualmem_pq_requantize_v1",
                "source_pq": str(args.pq.resolve()),
                "pq_protocol": pq_report.get("protocol"),
                "states": int(len(codes)),
                "reconstruction_probe": metrics,
            }, ensure_ascii=False),
        },
    )
    logging.info("wrote %s  codes %s", args.output, codes.shape)


if __name__ == "__main__":
    main()
