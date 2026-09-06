"""Merge trajectory-sharded Q-Former caches into one deterministic artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from residualmem.latent.instruct_bridge import BRIDGE_CACHE_PROTOCOL


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    shard_paths = [Path(path).resolve() for path in args.shards]
    columns = ("xbar", "valid", "teacher_fused_embedding", "split",
               "sample_id", "step_index", "target_text")
    parts = {name: [] for name in columns}
    metadata_rows = []
    for path in shard_paths:
        with np.load(path, allow_pickle=False) as data:
            missing = (set(columns) | {"metadata"}) - set(data.files)
            if missing:
                raise ValueError(f"{path}: missing {sorted(missing)}")
            metadata = json.loads(str(np.asarray(data["metadata"]).item()))
            if metadata.get("protocol") != BRIDGE_CACHE_PROTOCOL:
                raise ValueError(f"{path}: cache protocol mismatch")
            metadata_rows.append(metadata)
            for name in columns:
                parts[name].append(np.asarray(data[name]))

    world_sizes = {int(row.get("shard_world_size", -1)) for row in metadata_rows}
    ranks = {int(row.get("shard_rank", -1)) for row in metadata_rows}
    if len(world_sizes) != 1:
        raise ValueError(f"shards disagree on world size: {world_sizes}")
    world_size = world_sizes.pop()
    if world_size != len(shard_paths) or ranks != set(range(world_size)):
        raise ValueError(f"need exactly ranks 0..{world_size - 1}, got {sorted(ranks)}")
    ignored = {"shard_rank"}
    reference = {key: value for key, value in metadata_rows[0].items() if key not in ignored}
    for index, metadata in enumerate(metadata_rows[1:], 1):
        comparable = {key: value for key, value in metadata.items() if key not in ignored}
        if comparable != reference:
            differing = sorted(key for key in set(reference) | set(comparable)
                               if reference.get(key) != comparable.get(key))
            raise ValueError(f"shard {index} metadata differs at {differing}")

    merged = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    sample_ids = merged["sample_id"].astype(str)
    steps = merged["step_index"].astype(np.int64)
    order = np.lexsort((steps, sample_ids))
    merged = {name: value[order] for name, value in merged.items()}
    keys = list(zip(merged["sample_id"].astype(str),
                    merged["step_index"].astype(np.int64)))
    if len(keys) != len(set(keys)):
        raise ValueError("merged cache contains duplicate trajectory-step keys")

    final_metadata = {
        **reference,
        "shard_rank": None,
        "merged_shards": [
            {"rank": int(metadata["shard_rank"]), "path": str(path),
             "sha256": _sha256(path)}
            for path, metadata in sorted(zip(shard_paths, metadata_rows),
                                         key=lambda pair: int(pair[1]["shard_rank"]))
        ],
        "rows": len(keys),
        "sort_order": "sample_id_then_step_index",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **merged,
             metadata=np.asarray(json.dumps(final_metadata, sort_keys=True)))
    print(json.dumps({"output": str(output.resolve()), "rows": len(keys),
                      "samples": len(set(merged["sample_id"].astype(str))),
                      "sha256": _sha256(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
