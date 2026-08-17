"""D-data: does codebook quality depend on how many states K-means sees?

Everything is held fixed except the number of train states the codebook is fitted
on: one frozen A1 encoder, one validation split, one PCA, one normalizer, no
decoder training. The only moving part is the K-means sample count.

This isolates the question left open by A2 v1, where C=1024 drove train
quantization error to 0.032 but left validation at 0.215 -- classic codebook
overfitting on 2,000 states. If coverage is the binding constraint, the larger
fits should close that gap.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from residualmem.encoders.normalization import GroupChannelNormalizer
from residualmem.world_model import categorical_bottleneck as Q

from .a1_continuous_bottleneck import FeatureStore, build_split, select_split_rows
from .a2_categorical_bottleneck import collect_latents, kmeans_codebook, load_a1_warm_start
from .common import iter_jsonl, write_json


def quantization_error(params, config, latents, batch: int = 500) -> float:
    error = total = 0.0
    for start in range(0, latents.shape[0], batch):
        block = latents[start:start + batch]
        quantized, _ = Q.quantize(params, block, config)
        error += float(jnp.sum(jnp.square(block - quantized)))
        total += float(jnp.sum(jnp.square(block)))
    return error / max(total, 1e-30)


def run(args: argparse.Namespace) -> dict:
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    records = list(iter_jsonl(args.records))
    normalizer = GroupChannelNormalizer.from_npz(args.normalization)
    store = FeatureStore(args.features)
    device = jax.devices(args.platform)[args.device_index]

    base = Q.CategoricalBottleneckConfig(num_e_tokens=args.num_e_tokens, e_dim=args.e_dim,
                                         num_heads=args.num_heads, ffn_hidden=args.ffn_hidden)
    params = Q.initialize_params(base, args.seed)
    params.update(load_a1_warm_start(args.a1_checkpoint, base))
    params = jax.device_put(params, device)

    splits = {
        name: build_split(name, select_split_rows(records, name), records, store,
                          normalizer, device)
        for name in ("train", "validation")
    }
    latents = {
        name: collect_latents(params, split, base, args.init_batch_size)
        for name, split in splits.items()
    }
    train_states = latents["train"].shape[0]
    counts = sorted({min(int(value), train_states) for value in args.fit_states})

    rows = []
    for subspaces, categories in zip(args.num_subspaces, args.num_categories):
        dim = args.e_dim // subspaces
        config = Q.CategoricalBottleneckConfig(
            num_e_tokens=args.num_e_tokens, e_dim=args.e_dim, num_heads=args.num_heads,
            ffn_hidden=args.ffn_hidden, num_subspaces=subspaces,
            num_categories=categories, temperature=1.0,
        )
        for count in counts:
            # Deterministic prefix of the train split; the split order is itself
            # deterministic, so the smaller fit is a strict subset of the larger.
            sample = latents["train"][:count]
            points = sample.reshape(count, args.num_e_tokens, subspaces, dim)
            points = jnp.transpose(points, (1, 2, 0, 3)).reshape(
                args.num_e_tokens * subspaces, count, dim
            )
            codebook, stats = kmeans_codebook(points, categories, args.seed,
                                              args.kmeans_iterations)
            fitted = {**params, Q.CODEBOOK: codebook.reshape(
                args.num_e_tokens, subspaces, categories, dim)}
            row = {
                "num_subspaces": subspaces,
                "num_categories": categories,
                "subspace_dim": dim,
                "code_bits_per_state": config.code_bits,
                "fit_states": count,
                "samples_per_codeword": count / categories,
                "train_error": quantization_error(fitted, config, latents["train"]),
                "validation_error": quantization_error(fitted, config, latents["validation"]),
                "empty_clusters": stats["empty_clusters"],
                "singleton_clusters": stats["singleton_clusters"],
                "occupancy_median": stats["occupancy_median"],
            }
            row["generalization_gap"] = row["validation_error"] - row["train_error"]
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    result = {
        "protocol": "d_data_codebook_coverage_v1",
        "a1_checkpoint": str(Path(args.a1_checkpoint).resolve()),
        "records": str(Path(args.records).resolve()),
        "normalization_hash": normalizer.hash_bytes.hex(),
        "pca_sha256": normalizer.pca_sha256,
        "train_states": train_states,
        "validation_states": int(latents["validation"].shape[0]),
        "held_fixed": ["a1_encoder", "pca", "normalizer", "validation_split",
                       "decoder (never trained)"],
        "varied": ["kmeans_fit_states"],
        "numerics": {"matmul_precision": args.matmul_precision},
        "rows": rows,
    }
    write_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--a1-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-e-tokens", type=int, default=64)
    parser.add_argument("--e-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--ffn-hidden", type=int, default=1024)
    parser.add_argument("--num-subspaces", type=int, nargs="+", default=[8, 8])
    parser.add_argument("--num-categories", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--fit-states", type=int, nargs="+", default=[2000, 3500, 7013])
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--init-batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--matmul-precision", default="highest")
    args = parser.parse_args()
    if len(args.num_subspaces) != len(args.num_categories):
        raise ValueError("--num-subspaces and --num-categories must pair up")
    result = run(args)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "configurations": len(result["rows"])}, indent=2))


if __name__ == "__main__":
    main()
