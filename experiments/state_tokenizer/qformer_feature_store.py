"""Present Q-Former states in the layout A1's ``FeatureStore`` already reads.

A1 was written for the PCA path: it memory-maps ``worker*/`` shards of BF16
PCA output and applies a ``GroupChannelNormalizer`` online. The Q-Former path
produces the 512-wide state directly, in four flat ``.npz`` files. Rather than
teach A1 a second input format, this writes the states into the format it
already has, so the only change needed on its side is the group layout.

Split assignment, and why ``test`` means something different here:

    train        MolmoWeb train split         (selection: no)
    validation   MolmoWeb validation split    (selection: yes)
    test         **WorldMemArena web**        (reported via --evaluate-test)

MolmoWeb's own test split is *excluded from the output entirely* so it cannot
leak through a mislabelled slot. A1 only has three split names and the target
domain is what we want reported, so ``test`` carries WorldMemArena web here.
The manifest states this in full; reading the slot name alone will mislead.

One consequence to carry into any write-up: selecting A1's step on MolmoWeb
validation while reporting WorldMemArena web is clean, but the caller may also
choose to look at the reported web number when deciding anything downstream,
and at that point the web result is transductively adapted rather than fully
held out. It does not violate the rule that web's QA / answers / memory_points /
evidence never enter training -- A1 sees none of those, only observation states.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import ml_dtypes
import numpy as np


PROTOCOL = "residualmem_qformer_feature_store_v1"



def _jsonl_lines(path):
    """Newline-split only. See common.read_jsonl: `splitlines` also breaks on
    U+2028, which 11 MolmoWeb page titles contain, cutting records in half."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield line


def _load_states(paths: list[Path]) -> tuple[np.ndarray, list[str], set[str]]:
    states: list[np.ndarray] = []
    ids: list[str] = []
    checkpoints: set[str] = set()
    for path in sorted(paths):
        with np.load(path, allow_pickle=True) as data:
            states.append(np.asarray(data["xbar"], np.float32))
            ids.extend(str(v) for v in np.asarray(data["state_ids"]))
            checkpoints.add(json.loads(str(np.asarray(data["metadata"])))["checkpoint"])
    return np.concatenate(states), ids, checkpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molmoweb-states", type=Path, action="append", required=True)
    parser.add_argument("--molmoweb-records", type=Path, action="append", required=True)
    parser.add_argument("--wma-states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-states", type=int, default=8000)
    args = parser.parse_args()

    molmo_paths = [p for pattern in args.molmoweb_states for p in map(Path, glob.glob(str(pattern)))]
    molmo_x, molmo_ids, checkpoints = _load_states(molmo_paths)
    wma_x, wma_ids, wma_checkpoints = _load_states([args.wma_states])
    checkpoints |= wma_checkpoints
    if len(checkpoints) != 1:
        raise SystemExit(f"states came from different checkpoints: {checkpoints}")

    records: dict[str, dict] = {}
    for path in args.molmoweb_records:
        for line in _jsonl_lines(path):
            if line.strip():
                row = json.loads(line)
                records[row["state_id"]] = row

    rows: list[dict] = []
    keep_index: list[int] = []
    dropped_test = 0
    for position, state_id in enumerate(molmo_ids):
        record = records.get(state_id)
        if record is None:
            raise SystemExit(f"no record for encoded state {state_id}")
        if record["split"] == "test":
            # MolmoWeb's own test split never enters this store.
            dropped_test += 1
            continue
        keep_index.append(position)
        rows.append({**record, "source": "molmoweb"})

    molmo_x = molmo_x[keep_index]
    for position, state_id in enumerate(wma_ids):
        sample = state_id.rsplit("-", 1)[0]
        rows.append({
            "state_id": state_id,
            "task": "wma_web",
            "episode_id": sample,
            "step": int(state_id.rsplit("-", 1)[1]),
            # See the module docstring: this slot carries the target domain.
            "split": "test",
            "source": "worldmemarena_web",
        })

    states = np.concatenate([molmo_x, wma_x])
    if len(states) != len(rows):
        raise SystemExit(f"{len(states)} states against {len(rows)} records")

    for index, row in enumerate(rows):
        row["global_index"] = index

    args.output.mkdir(parents=True, exist_ok=True)
    features = args.output / "features"
    features.mkdir(exist_ok=True)

    count = len(states)
    shards = max(1, (count + args.shard_states - 1) // args.shard_states)
    for shard in range(shards):
        lo = shard * args.shard_states
        hi = min(count, lo + args.shard_states)
        worker = features / f"worker{shard:03d}"
        worker.mkdir(exist_ok=True)
        # The loader does ``asarray(bits, uint16).view(bfloat16).astype(float32)``,
        # so what goes on disk is the BF16 bit pattern, not a float array.
        bits = states[lo:hi].astype(ml_dtypes.bfloat16).view(np.uint16)
        np.save(worker / "key64-static-pca-bf16.npy", bits)
        # Every Q-Former query is valid by construction -- unlike the pooled path,
        # which left 13 of its 16 context slots empty on every WMA observation.
        np.save(worker / "key64-static-valid.npy",
                np.ones((hi - lo, states.shape[1]), np.bool_))
        np.save(worker / "record_indices.npy",
                np.arange(lo, hi, dtype=np.int64))
        np.save(worker / "done.npy", np.ones((hi - lo,), np.bool_))
        np.save(worker / "key64-static-pca-done.npy", np.ones((hi - lo,), np.bool_))

    manifest_path = args.output / "records.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    manifest = {
        "protocol": PROTOCOL,
        "checkpoint": next(iter(checkpoints)),
        "num_states": count,
        "num_slots": int(states.shape[1]),
        "slot_dim": int(states.shape[2]),
        "shards": shards,
        "split_counts": counts,
        "split_semantics": {
            "train": "MolmoWeb train",
            "validation": "MolmoWeb validation -- drives A1 selection",
            "test": "WorldMemArena web -- reported via --evaluate-test, NOT MolmoWeb test",
        },
        "molmoweb_test_states_excluded": dropped_test,
        "records_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
