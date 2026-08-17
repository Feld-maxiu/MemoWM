"""Fit and apply a shared PCA512 transform to valid key64 slots."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import iter_jsonl, sha256_file, write_json
from .extract_full_h import _open_array
from .feature_store import decode_bf16_bits, discover_shards, encode_bf16_bits
from .key_pooling import KEY64_LAYOUT, KEY64_PROTOCOL
from .static_key_pooling import STATIC_KEY64_PROTOCOL


PREFIX_PROTOCOLS = {
    "key64": KEY64_PROTOCOL,
    "key64-static": STATIC_KEY64_PROTOCOL,
}


def _prefix(args: argparse.Namespace) -> str:
    return getattr(args, "prefix", "key64")


def valid_rows(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError(f"invalid key64 values/mask: {tuple(values.shape)} {tuple(valid.shape)}")
    return values.reshape(-1, values.shape[-1])[valid.reshape(-1)]


def fit(args: argparse.Namespace) -> dict:
    if args.components != 512:
        raise ValueError("key64 PCA artifacts require exactly 512 components")
    if args.load_batch_states < 1 or args.fit_states < 1:
        raise ValueError("fit-states and load-batch-states must be positive")
    train_indices = []
    for record in iter_jsonl(args.records):
        if record["split"] == "train":
            train_indices.append(int(record["global_index"]))
            if len(train_indices) == args.fit_states:
                break
    if len(train_indices) < args.fit_states:
        raise ValueError(f"requested {args.fit_states} PCA states, found {len(train_indices)}")
    if len(set(train_indices)) != len(train_indices):
        raise ValueError("duplicate train global indices in PCA fit selection")
    selected = set(train_indices)
    prefix = _prefix(args)
    protocol = PREFIX_PROTOCOLS[prefix]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    shards = discover_shards(args.features)
    selected_rows = []
    found_indices = []
    total_rows = 0
    group_rows = np.zeros((4,), np.int64)
    boundaries = np.cumsum((0, *KEY64_LAYOUT))
    for shard in shards:
        local_rows = np.asarray([
            local for local, global_index in enumerate(shard.indices)
            if int(global_index) in selected
        ], np.int64)
        masks = np.load(shard.path / f"{prefix}-valid.npy", mmap_mode="r")
        if len(masks) != len(shard.indices) or masks.shape[1:] != (64,):
            raise ValueError(f"invalid key64 mask shape in {shard.path}: {masks.shape}")
        selected_mask = np.asarray(masks[local_rows], np.bool_)
        total_rows += int(selected_mask.sum())
        for group in range(4):
            group_rows[group] += int(
                selected_mask[:, boundaries[group]:boundaries[group + 1]].sum()
            )
        found_indices.extend(int(shard.indices[row]) for row in local_rows)
        selected_rows.append((shard, local_rows))
    if set(found_indices) != selected or len(found_indices) != len(selected):
        missing = sorted(selected - set(found_indices))
        raise ValueError(f"PCA feature store misses selected train states: {missing[:20]}")
    if total_rows < 512:
        raise ValueError(f"PCA512 requires at least 512 valid rows, found {total_rows}")
    matrix = torch.empty((total_rows, 4096), dtype=torch.float32, device=device)
    cursor = 0
    for shard, local_rows in selected_rows:
        source = np.load(shard.path / f"{prefix}-bf16.npy", mmap_mode="r")
        masks = np.load(shard.path / f"{prefix}-valid.npy", mmap_mode="r")
        for start in range(0, len(local_rows), args.load_batch_states):
            rows = local_rows[start:start + args.load_batch_states]
            if not len(rows):
                continue
            values = decode_bf16_bits(source[rows], device=device)
            valid = torch.from_numpy(np.asarray(masks[rows], np.bool_).copy()).to(device)
            batch = valid_rows(values, valid)
            matrix[cursor:cursor + len(batch)] = batch
            cursor += len(batch)
    if cursor != total_rows:
        raise RuntimeError(f"loaded {cursor} valid PCA rows, expected {total_rows}")
    q = min(args.oversample, matrix.shape[0], matrix.shape[1])
    if q < args.components:
        raise ValueError(f"PCA rank q={q} is smaller than components={args.components}")
    torch.manual_seed(args.seed)
    started = time.time()
    mean = matrix.mean(dim=0)
    total_variance = matrix.var(dim=0, unbiased=True).sum()
    matrix.sub_(mean)
    _, singular, components = torch.pca_lowrank(
        matrix, q=q, center=False, niter=args.niter
    )
    components = components[:, :args.components].contiguous()
    singular = singular[:args.components].contiguous()
    explained = singular.square() / max(len(matrix) - 1, 1)
    explained_ratio = explained.sum() / total_variance
    output = Path(args.output)
    if output.suffix != ".npz":
        output = Path(str(output) + ".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        mean=mean.cpu().numpy().astype(np.float32),
        components=components.cpu().numpy().astype(np.float32),
        singular_values=singular.cpu().numpy().astype(np.float32),
        explained_variance=explained.cpu().numpy().astype(np.float32),
        explained_variance_ratio=np.asarray(float(explained_ratio), np.float32),
        fit_record_indices=np.asarray(train_indices, np.int64),
        layout=np.asarray(KEY64_LAYOUT, np.int32),
        group_valid_rows=group_rows,
        prefix=np.asarray(prefix),
        protocol=np.asarray(protocol),
    )
    summary = {
        "protocol": protocol,
        "fit_states": len(train_indices),
        "fit_valid_slot_rows": len(matrix),
        "group_valid_rows": group_rows.tolist(),
        "layout": list(KEY64_LAYOUT),
        "prefix": prefix,
        "input_dim": 4096,
        "components": args.components,
        "oversample": q,
        "requested_oversample": args.oversample,
        "niter": args.niter,
        "seed": args.seed,
        "explained_variance_ratio": float(explained_ratio),
        "elapsed_seconds": time.time() - started,
        "artifact": str(output.resolve()),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return summary


def transform(args: argparse.Namespace) -> dict:
    artifact = np.load(args.pca)
    prefix = _prefix(args)
    protocol = PREFIX_PROTOCOLS[prefix]
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    mean = torch.from_numpy(np.asarray(artifact["mean"], np.float32)).to(device)
    components = torch.from_numpy(np.asarray(artifact["components"], np.float32)).to(device)
    if components.shape != (4096, 512):
        raise ValueError(f"unexpected PCA shape: {tuple(components.shape)}")
    if tuple(np.asarray(artifact["layout"], np.int32)) != KEY64_LAYOUT:
        raise ValueError("PCA artifact key64 layout mismatch")
    artifact_prefix = str(
        np.asarray(artifact["prefix"]).item() if "prefix" in artifact.files else "key64"
    )
    if artifact_prefix != prefix:
        raise ValueError(f"PCA prefix mismatch: {artifact_prefix} != {prefix}")
    artifact_protocol = str(
        np.asarray(artifact["protocol"]).item()
        if "protocol" in artifact.files else KEY64_PROTOCOL
    )
    if artifact_protocol != protocol:
        raise ValueError(f"PCA protocol mismatch: {artifact_protocol} != {protocol}")
    artifact.close()
    pca_sha256 = sha256_file(args.pca)
    if args.batch_states < 1 or args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("batch-states/world-size must be positive and rank must be valid")
    shards = discover_shards(args.features)
    assigned = [shard for index, shard in enumerate(shards) if index % args.world_size == args.rank]
    transformed = 0
    started = time.time()
    for shard in assigned:
        source = np.load(shard.path / f"{prefix}-bf16.npy", mmap_mode="r")
        masks = np.load(shard.path / f"{prefix}-valid.npy", mmap_mode="r")
        destination_path = shard.path / f"{prefix}-pca-bf16.npy"
        done_path = shard.path / f"{prefix}-pca-done.npy"
        hash_path = shard.path / f"{prefix}-pca-artifact.sha256"
        existing = [destination_path.exists(), done_path.exists(), hash_path.exists()]
        if args.resume and any(existing):
            if not all(existing):
                raise RuntimeError(f"incomplete PCA resume bundle in {shard.path}")
            if hash_path.read_text().strip() != pca_sha256:
                raise RuntimeError(
                    f"PCA artifact changed for {shard.path}; rerun with --no-resume"
                )
            destination = _open_array(
                destination_path, np.uint16, (len(source), 64, 512), True
            )
            done = _open_array(done_path, np.bool_, (len(source),), True)
        else:
            destination = _open_array(
                destination_path, np.uint16, (len(source), 64, 512), False
            )
            done = _open_array(done_path, np.bool_, (len(source),), False)
            done[:] = False
            done.flush()
            hash_path.write_text(pca_sha256 + "\n")
        mean_projection = torch.matmul(mean, components)
        for start in range(0, len(source), args.batch_states):
            stop = min(start + args.batch_states, len(source))
            pending = np.flatnonzero(~np.asarray(done[start:stop], np.bool_)) + start
            if not len(pending):
                continue
            values = decode_bf16_bits(source[pending], device=device)
            valid = torch.from_numpy(
                np.asarray(masks[pending], np.bool_).copy()
            ).to(device)
            with torch.inference_mode():
                projected = torch.zeros(
                    (len(pending), 64, 512), dtype=torch.float32, device=device
                )
                projected_flat = projected.reshape(-1, 512)
                values_flat = values.reshape(-1, 4096)
                valid_flat = valid.reshape(-1)
                projected_flat[valid_flat] = (
                    torch.matmul(values_flat[valid_flat], components) - mean_projection
                )
            destination[pending] = encode_bf16_bits(projected)
            destination.flush()
            done[pending] = True
            done.flush()
            transformed += len(pending)
    summary = {
        "protocol": protocol,
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned_shards": [str(shard.path) for shard in assigned],
        "transformed_states": transformed,
        "pca_sha256": pca_sha256,
        "elapsed_seconds": time.time() - started,
    }
    write_json(
        Path(args.features) / f"{prefix}-pca-transform-rank{args.rank:02d}.json",
        summary,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--records", required=True)
    fit_parser.add_argument("--features", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--device", default="cuda:0")
    fit_parser.add_argument("--fit-states", type=int, default=2000)
    fit_parser.add_argument("--components", type=int, choices=(512,), default=512)
    fit_parser.add_argument("--oversample", type=int, default=576)
    fit_parser.add_argument("--niter", type=int, default=3)
    fit_parser.add_argument("--seed", type=int, default=0)
    fit_parser.add_argument("--load-batch-states", type=int, default=8)
    fit_parser.add_argument("--prefix", choices=tuple(PREFIX_PROTOCOLS), default="key64")
    transform_parser = subparsers.add_parser("transform")
    transform_parser.add_argument("--features", required=True)
    transform_parser.add_argument("--pca", required=True)
    transform_parser.add_argument("--device", default="cuda:0")
    transform_parser.add_argument("--rank", type=int, default=0)
    transform_parser.add_argument("--world-size", type=int, default=1)
    transform_parser.add_argument("--batch-states", type=int, default=16)
    transform_parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    transform_parser.add_argument("--prefix", choices=tuple(PREFIX_PROTOCOLS), default="key64")
    args = parser.parse_args()
    result = fit(args) if args.command == "fit" else transform(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
