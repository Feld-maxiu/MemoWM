"""Fit and apply the conditional Y64 -> PCA512 experiment."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from .common import iter_jsonl, write_json
from .feature_store import decode_bf16_bits, discover_shards, encode_bf16_bits, iter_y64_batches


def require_y64_gate(metrics_dir: str | Path) -> dict:
    path = Path(metrics_dir) / "gate64.json"
    if not path.exists():
        raise RuntimeError("missing gate64.json; PCA is forbidden before the Y64 gate")
    gate = json.loads(path.read_text())
    if not gate.get("passed"):
        raise RuntimeError("Y64 gate failed; PCA is forbidden and the pipeline must stop")
    return gate


def fit(args: argparse.Namespace) -> dict:
    require_y64_gate(args.metrics)
    records = list(iter_jsonl(args.records))
    train_indices = [
        index for index, record in enumerate(records) if record["split"] == "train"
    ][:args.fit_states]
    if len(train_indices) < args.fit_states:
        raise ValueError(f"requested {args.fit_states} PCA states, found {len(train_indices)}")
    selected = set(train_indices)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    rows = len(train_indices) * 64
    matrix = torch.empty((rows, 4096), dtype=torch.float32, device=device)
    cursor = 0
    for _, batch in iter_y64_batches(
        args.features, batch_states=args.load_batch_states,
        selected_indices=selected, device=device,
    ):
        flat = batch.reshape(-1, 4096)
        matrix[cursor:cursor + len(flat)] = flat
        cursor += len(flat)
    if cursor != rows:
        raise RuntimeError(f"loaded {cursor} PCA rows, expected {rows}")
    torch.manual_seed(args.seed)
    started = time.time()
    mean = matrix.mean(dim=0)
    total_variance = matrix.var(dim=0, unbiased=True).sum()
    u, singular, components = torch.pca_lowrank(
        matrix, q=args.oversample, center=True, niter=args.niter
    )
    components = components[:, :args.components].contiguous()
    singular = singular[:args.components].contiguous()
    explained = singular.square() / max(rows - 1, 1)
    explained_ratio = explained.sum() / total_variance
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        mean=mean.cpu().numpy().astype(np.float32),
        components=components.cpu().numpy().astype(np.float32),
        singular_values=singular.cpu().numpy().astype(np.float32),
        explained_variance=explained.cpu().numpy().astype(np.float32),
        explained_variance_ratio=np.asarray(float(explained_ratio), np.float32),
        fit_record_indices=np.asarray(train_indices, np.int64),
    )
    summary = {
        "fit_states": len(train_indices),
        "fit_slot_rows": rows,
        "input_dim": 4096,
        "components": args.components,
        "oversample": args.oversample,
        "niter": args.niter,
        "seed": args.seed,
        "explained_variance_ratio": float(explained_ratio),
        "elapsed_seconds": time.time() - started,
        "artifact": str(output.resolve()),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    del matrix, u, components, singular
    torch.cuda.empty_cache()
    return summary


def transform(args: argparse.Namespace) -> dict:
    require_y64_gate(args.metrics)
    artifact = np.load(args.pca)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    mean = torch.from_numpy(np.asarray(artifact["mean"], np.float32)).to(device)
    components = torch.from_numpy(np.asarray(artifact["components"], np.float32)).to(device)
    if components.shape != (4096, 512):
        raise ValueError(f"unexpected PCA shape: {tuple(components.shape)}")
    shards = discover_shards(args.features)
    assigned = [shard for index, shard in enumerate(shards) if index % args.world_size == args.rank]
    transformed = 0
    started = time.time()
    for shard in assigned:
        source = np.load(shard.path / "y64-bf16.npy", mmap_mode="r")
        destination_path = shard.path / "x64-pca-bf16.npy"
        done_path = shard.path / "pca-done.npy"
        if args.resume and destination_path.exists() and done_path.exists():
            destination = np.load(destination_path, mmap_mode="r+")
            done = np.load(done_path, mmap_mode="r+")
        else:
            destination = np.lib.format.open_memmap(
                destination_path, mode="w+", dtype=np.uint16,
                shape=(len(source), 64, 512),
            )
            done = np.lib.format.open_memmap(
                done_path, mode="w+", dtype=np.bool_, shape=(len(source),)
            )
            done[:] = False
            done.flush()
        for start in range(0, len(source), args.batch_states):
            stop = min(start + args.batch_states, len(source))
            if bool(np.asarray(done[start:stop]).all()):
                continue
            values = decode_bf16_bits(source[start:stop], device=device)
            with torch.inference_mode():
                projected = torch.matmul(values - mean, components)
            destination[start:stop] = encode_bf16_bits(projected)
            done[start:stop] = True
            transformed += stop - start
            if transformed % args.flush_every < args.batch_states:
                destination.flush()
                done.flush()
        destination.flush()
        done.flush()
    summary = {
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned_shards": [str(shard.path) for shard in assigned],
        "transformed_states": transformed,
        "elapsed_seconds": time.time() - started,
        "device": str(device),
    }
    write_json(Path(args.features) / f"pca-transform-rank{args.rank:02d}.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--records", required=True)
    fit_parser.add_argument("--features", required=True)
    fit_parser.add_argument("--metrics", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--device", default="cuda:0")
    fit_parser.add_argument("--fit-states", type=int, default=5000)
    fit_parser.add_argument("--components", type=int, default=512)
    fit_parser.add_argument("--oversample", type=int, default=576)
    fit_parser.add_argument("--niter", type=int, default=3)
    fit_parser.add_argument("--seed", type=int, default=0)
    fit_parser.add_argument("--load-batch-states", type=int, default=8)

    transform_parser = subparsers.add_parser("transform")
    transform_parser.add_argument("--features", required=True)
    transform_parser.add_argument("--metrics", required=True)
    transform_parser.add_argument("--pca", required=True)
    transform_parser.add_argument("--device", default="cuda:0")
    transform_parser.add_argument("--rank", type=int, default=0)
    transform_parser.add_argument("--world-size", type=int, default=1)
    transform_parser.add_argument("--batch-states", type=int, default=16)
    transform_parser.add_argument("--flush-every", type=int, default=128)
    transform_parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    result = fit(args) if args.command == "fit" else transform(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
