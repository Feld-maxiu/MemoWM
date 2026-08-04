"""Validate Full-H ragged storage against the existing deterministic pools."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .common import iter_jsonl
from .extract_qwen import adaptive_pool_torch
from .feature_store import encode_bf16_bits
from .ragged_store import FixedRepresentationStore, RaggedFullHStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--full-h", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--device", default="cpu",
        help="Pool on the original extraction device for a bit-exact reduction check.",
    )
    args = parser.parse_args()
    records = list(iter_jsonl(args.records))
    if args.limit is not None:
        records = records[:args.limit]
    full = RaggedFullHStore(args.full_h, len(records))
    fixed64 = FixedRepresentationStore(args.features, "y64")
    fixed32 = FixedRepresentationStore(args.features, "y32")
    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(len(records), size=min(args.samples, len(records)), replace=False))
    maximum = {"y64": 0.0, "y32": 0.0}
    for row in rows:
        tokens, modalities = full.get(int(row))
        tokens = tokens.to(args.device)
        modalities = modalities.to(args.device)
        parts = [tokens[modalities == modality] for modality in range(3)]
        for name, layout, store in (
            ("y64", (40, 20, 4), fixed64),
            ("y32", (20, 10, 2), fixed32),
        ):
            pooled = torch.cat([
                adaptive_pool_torch(part, slots) for part, slots in zip(parts, layout)
            ])
            existing, _ = store.get(records[int(row)]["global_index"])
            pooled_bf16 = pooled.to(torch.bfloat16)
            delta = float((pooled_bf16.float() - existing.to(pooled.device).float()).abs().max())
            maximum[name] = max(maximum[name], delta)
            if not np.array_equal(encode_bf16_bits(pooled_bf16), encode_bf16_bits(existing)):
                raise AssertionError(f"{name} BF16 mismatch at subset row {row}, max_abs={delta}")
    print(json.dumps({"samples": len(rows), "max_abs_error": maximum, "passed": True}, indent=2))


if __name__ == "__main__":
    main()
