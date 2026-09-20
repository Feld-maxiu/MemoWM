"""Materialize the exact closed-loop ResidualMem state consumed by WMA QA.

The rate evaluator already implements the decoder-visible recurrence: each
utility-dropped code is replaced by the world-model mode, and that reconstructed
code is fed into every later prediction in the episode.  This command reuses
that implementation, OPQ-decodes the final code table, and writes a query-free
lookup artifact keyed by the official WMA image IDs.

Initial states have no causal prior.  They follow the codec rule and transmit
all OPQ codes; consequently they still undergo OPQ reconstruction, but no WM
fill.  The output contains no raw Q-Former state and cannot silently fall back
to it during QA.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np

from experiments.state_tokenizer.common import sha256_file
from experiments.state_tokenizer.qformer_pq import decode
from experiments.utility_gate.closed_loop_rate import run_pass
from experiments.world_model.cache import CACHE_FILES, FrozenCache
from experiments.world_model.config import load_config
from experiments.world_model.train import load_checkpoint


PROTOCOL = "residualmem_wma_closed_loop_reconstruction_v1"


def _load_codebook(path: Path) -> dict[str, np.ndarray | None]:
    # Keep this reader local and NumPy-only: the similarly named retrieval-cache
    # helper lives in a Torch-side module, while this command deliberately runs
    # in the isolated JAX interpreter.
    with np.load(path, allow_pickle=True) as data:
        required = {"mean", "scale", "centroids"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"codebook missing {sorted(missing)}")
        return {
            "mean": np.asarray(data["mean"], dtype=np.float32),
            "scale": np.asarray(data["scale"], dtype=np.float32),
            "centroids": np.asarray(data["centroids"], dtype=np.float32),
            "bases": (
                np.asarray(data["rotation_bases"], dtype=np.float32)
                if "rotation_bases" in data.files else None
            ),
            "order": (
                np.asarray(data["rotation_order"], dtype=np.int64)
                if "rotation_order" in data.files else None
            ),
        }


def _records(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _lookup_table(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    lookup: dict[str, int] = {}
    for row, record in enumerate(records):
        keys = [str(record["state_id"])]
        keys.extend(str(value) for value in record.get("image_ids", ()) if value)
        for key in keys:
            previous = lookup.setdefault(key, row)
            if previous != row:
                raise ValueError(
                    f"lookup key {key!r} names both row {previous} and row {row}"
                )
    keys = np.asarray(sorted(lookup), dtype=np.str_)
    rows = np.asarray([lookup[str(key)] for key in keys], dtype=np.int64)
    return keys, rows


def _r2(reference: np.ndarray, rebuilt: np.ndarray) -> float:
    mean = reference.mean(axis=0, keepdims=True)
    total = np.square(reference - mean, dtype=np.float64).sum()
    error = np.square(reference - rebuilt, dtype=np.float64).sum()
    return float(1.0 - error / max(total, 1e-30))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--states", required=True, help="Q-Former states used to build the cache")
    parser.add_argument("--records", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cache = FrozenCache(args.cache, verify_hashes=True)
    records = _records(Path(args.records))
    if len(records) != len(cache.codes):
        raise ValueError(f"{len(records)} records != {len(cache.codes)} cached states")
    state_ids = np.asarray([str(row["state_id"]) for row in records], dtype=np.str_)
    if len(state_ids) != len(set(state_ids.tolist())):
        raise ValueError("records contain duplicate state IDs")

    with np.load(args.states, allow_pickle=True) as source:
        source_ids = np.asarray(source["state_ids"], dtype=np.str_)
        source_xbar = np.asarray(source["xbar"], dtype=np.float32)
    if not np.array_equal(source_ids, state_ids):
        raise ValueError("states and records use different state order")
    if source_xbar.shape != (len(records), cache.codes.shape[1], 512):
        raise ValueError(f"unexpected source xbar shape {source_xbar.shape}")

    with np.load(args.mask, allow_pickle=False) as mask:
        if "v" in mask.files:                                    # e·V artifact
            ev = {"v": np.asarray(mask["v"], dtype=np.float64),
                  "edges": np.asarray(mask["h_edges"], dtype=np.float64),
                  "rates": np.asarray(mask["h_rates"], dtype=np.float64)}
            utility = np.asarray(mask["u"], dtype=np.float64)
            arm = "ev"
        else:
            ev = None
            utility = np.asarray(mask["utility"], dtype=np.float64)
            arm = "gate"
        lam = float(np.asarray(mask["lam"]).item())

    config = load_config(args.config, num_tasks=len(cache.task_names))
    cache.max_history = config.model.max_history
    jax.config.update("jax_default_matmul_precision", config.training.matmul_precision)
    device = jax.devices(args.platform)[args.device_index]
    checkpoint = load_checkpoint(args.checkpoint)
    variant = str(checkpoint["metadata"]["variant"])
    params = jax.device_put(checkpoint["params"], device)
    rows = np.asarray(cache.indices_for_split(args.split))
    if not len(rows):
        raise ValueError(f"split {args.split!r} contains no transitions")

    closed = run_pass(
        cache, params, variant, config, rows,
        utility=utility, lam=lam, closed=True, arm=arm, ev=ev,
        batch_size=args.batch_size, collect_reconstructions=True,
    )
    reconstructed = closed["_reconstructed_codes"]
    target_indices = np.asarray(cache.transitions["target_indices"])[rows]
    expected_targets = set(int(value) for value in target_indices)
    if set(reconstructed) != expected_targets:
        raise ValueError(
            f"closed loop returned {len(reconstructed)} targets, expected "
            f"{len(expected_targets)}"
        )

    final_codes = np.asarray(cache.codes, dtype=np.uint8).copy()
    for index, value in reconstructed.items():
        final_codes[int(index)] = np.asarray(value, dtype=np.uint8)
    initial = np.ones(len(final_codes), dtype=np.bool_)
    initial[target_indices] = False
    expected_initial = len(cache.episode_names)
    if int(initial.sum()) != expected_initial:
        raise ValueError(
            f"found {int(initial.sum())} initial states, expected {expected_initial}"
        )

    book = _load_codebook(Path(args.codebook))
    rebuilt = decode(
        final_codes, book["mean"], book["scale"], book["centroids"],
        bases=book["bases"], order=book["order"],
    ).astype(np.float32)
    valid = np.asarray(cache.valid, dtype=np.bool_)
    rebuilt *= valid[..., None]
    lookup_keys, lookup_rows = _lookup_table(records)

    metadata = {
        "protocol": PROTOCOL,
        "states": int(len(final_codes)),
        "transitions": int(len(rows)),
        "initial_all_send_states": int(initial.sum()),
        "lambda": lam,
        "closed_loop_gated_rate_bits": float(closed["gated_rate_bits"]),
        "closed_loop_full_rate_from_drifted_history_bits": float(closed["full_rate_bits"]),
        "keep_fraction": float(closed["keep_fraction"]),
        "opq_and_closed_loop_r2": _r2(source_xbar, rebuilt),
        "lookup": "official image_id or state_id; no query input",
        "initial_state_rule": "full OPQ codes; no causal WM prior",
        "dropped_position_rule": "WM argmax from decoder-visible closed-loop history",
        "cache_manifest_sha256": sha256_file(Path(args.cache) / CACHE_FILES["manifest"]),
        "states_sha256": sha256_file(args.states),
        "records_sha256": sha256_file(args.records),
        "codebook_sha256": sha256_file(args.codebook),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "mask_sha256": sha256_file(args.mask),
    }
    payload = {
        "state_ids": state_ids,
        "lookup_keys": lookup_keys,
        "lookup_rows": lookup_rows,
        "gated_xbar": rebuilt,
        "valid": valid,
        "reconstructed_codes": final_codes,
        "initial_all_send": initial,
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, output)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
