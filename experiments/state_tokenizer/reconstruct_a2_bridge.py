"""JAX-side helper for reconstructing bridge-cache Xbar through frozen A2."""
from __future__ import annotations

import argparse
import json

import jax.numpy as jnp
import numpy as np

from residualmem.world_model.categorical_bottleneck import (
    CategoricalBottleneckConfig,
    reconstruct,
)
from .a2_categorical_bottleneck import load_checkpoint, read_checkpoint_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        xbar = np.asarray(data["xbar"], np.float32)
        valid = np.asarray(data["valid"], np.bool_)
    stored = read_checkpoint_config(args.checkpoint)
    stored["group_sizes"] = tuple(stored["group_sizes"])
    config = CategoricalBottleneckConfig(**stored)
    params = load_checkpoint(args.checkpoint, config)
    chunks = []
    for start in range(0, len(xbar), args.batch_size):
        stop = min(start + args.batch_size, len(xbar))
        chunks.append(np.asarray(reconstruct(
            params, jnp.asarray(xbar[start:stop]), jnp.asarray(valid[start:stop]), config
        ), np.float32))
    output = np.concatenate(chunks)
    np.save(args.output, output)
    print(json.dumps({"states": len(output), "shape": list(output.shape)}))


if __name__ == "__main__":
    main()
