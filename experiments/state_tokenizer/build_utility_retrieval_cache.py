"""Add full/gated OPQ reconstructions to an existing retrieval-head cache.

The base cache carries raw Q-Former states and fused-observation retrieval
teachers.  This script joins those rows to the frozen OPQ codes and world-model
posterior, then creates the two views used by ``train_retrieval_bridge``:

* ``full_recon_xbar``: OPQ decode of every transmitted code; and
* ``gated_recon_xbar``: dropped positions filled with the WM argmax, then decoded.

Initial states have no causal WM posterior.  The codec sends them in full, so
their gated view is exactly their full view.  No Q-Former forward is needed and
the base cache is never overwritten.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.qformer_pq import decode
from experiments.utility_gate.export_mask import MASK_PROTOCOL, keep_mask
from experiments.utility_gate.label_counterfactual import cache_order
from residualmem.latent.instruct_bridge import BRIDGE_CACHE_PROTOCOL


POSTERIOR_PROTOCOL = "residualmem_utility_gate_posterior_v1"


def _state_ids(cache: dict[str, np.ndarray]) -> list[str]:
    if "state_id" in cache:
        values = [str(value) for value in np.asarray(cache["state_id"])]
    else:
        if "sample_id" not in cache:
            raise ValueError(
                "base cache has neither state_id nor sample_id; rows cannot be "
                "joined to utility artifacts")
        # Legacy Q-Former caches were emitted sample-by-sample and omitted the
        # record index.  Reconstruct the exact ID convention used by gate/states.
        seen: collections.defaultdict[str, int] = collections.defaultdict(int)
        values = []
        for raw_sample in np.asarray(cache["sample_id"]):
            sample = str(raw_sample)
            values.append(f"{sample}-{seen[sample]:04d}")
            seen[sample] += 1
    if len(values) != len(set(values)):
        raise ValueError("retrieval cache carries duplicate state IDs")
    return values


def _load_codebook(path: Path) -> dict[str, np.ndarray | None]:
    with np.load(path, allow_pickle=True) as data:
        required = {"mean", "scale", "centroids"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"codebook missing {sorted(missing)}")
        return {
            "mean": np.asarray(data["mean"], np.float32),
            "scale": np.asarray(data["scale"], np.float32),
            "centroids": np.asarray(data["centroids"], np.float32),
            "bases": (np.asarray(data["rotation_bases"], np.float32)
                      if "rotation_bases" in data.files else None),
            "order": (np.asarray(data["rotation_order"], np.int64)
                      if "rotation_order" in data.files else None),
        }


def build_views(
    state_ids: list[str], codes: np.ndarray, code_state_ids: list[str],
    global_index: dict[str, int], posterior: dict[str, np.ndarray],
    utility: np.ndarray, lam: float, book: dict[str, np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Build the two frozen-codec views and return accounting diagnostics."""
    code_row = {state_id: row for row, state_id in enumerate(code_state_ids)}
    if len(code_row) != len(code_state_ids):
        raise ValueError("codes artifact carries duplicate state IDs")
    missing = [state_id for state_id in state_ids if state_id not in code_row]
    if missing:
        raise ValueError(
            f"{len(missing)} retrieval states have no code; first is {missing[0]!r}")

    full_codes = np.stack([codes[code_row[state_id]] for state_id in state_ids])
    gated_codes = full_codes.copy()
    target_indices = np.asarray(posterior["target_indices"], np.int64)
    if len(target_indices) != len(set(target_indices.tolist())):
        raise ValueError("posterior dump carries duplicate target indices")
    posterior_row = {int(index): row for row, index in enumerate(target_indices)}

    entropy = np.asarray(posterior["entropy_bits"], np.float64)
    wm_argmax = np.asarray(posterior["wm_argmax"], np.uint8)
    target_codes = np.asarray(posterior["target_codes"], np.uint8)
    posterior_states = 0
    keep_total = 0
    position_total = 0
    for cache_row, state_id in enumerate(state_ids):
        if state_id not in global_index:
            raise ValueError(f"state {state_id!r} is absent from the records manifest")
        row = posterior_row.get(int(global_index[state_id]))
        if row is None:
            # No predecessor means no WM prediction.  The codec's causal rule
            # is all-send, not a corpus-mode or zero fill.
            continue
        if not np.array_equal(target_codes[row], full_codes[cache_row]):
            raise ValueError(
                f"posterior/code mismatch for {state_id!r}; artifacts use "
                "different state order or codebooks")
        keep = keep_mask(utility, entropy[row].reshape(-1), lam).reshape(
            full_codes.shape[1:])
        gated_codes[cache_row] = np.where(
            keep, full_codes[cache_row], wm_argmax[row]).astype(np.uint8)
        posterior_states += 1
        keep_total += int(keep.sum())
        position_total += int(keep.size)

    full = decode(
        full_codes, book["mean"], book["scale"], book["centroids"],
        bases=book["bases"], order=book["order"]).astype(np.float32)
    gated = decode(
        gated_codes, book["mean"], book["scale"], book["centroids"],
        bases=book["bases"], order=book["order"]).astype(np.float32)
    return full, gated, {
        "states": len(state_ids),
        "posterior_states": posterior_states,
        "all_send_states": len(state_ids) - posterior_states,
        "keep_fraction_with_posterior": (
            float(keep_total / position_total) if position_total else 1.0),
        "changed_code_fraction": float((gated_codes != full_codes).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="base retrieval cache")
    parser.add_argument("--codes", required=True, help="Q-Former OPQ codes npz")
    parser.add_argument("--codebook", required=True, help="frozen OPQ codebook npz")
    parser.add_argument("--posteriors", required=True, help="WM posterior dump npz")
    parser.add_argument("--mask", required=True, help="exported utility vector/lambda npz")
    parser.add_argument("--records", required=True, help="world-model records jsonl")
    parser.add_argument("--label", default="train", help="codes/records split label")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=False) as data:
        base = {name: np.asarray(data[name]) for name in data.files}
    metadata = json.loads(str(np.asarray(base["metadata"]).item()))
    if metadata.get("protocol") != BRIDGE_CACHE_PROTOCOL:
        raise ValueError("base retrieval cache protocol mismatch")
    if metadata.get("representation", "xbar") != "xbar":
        raise ValueError("utility views must be built from a representation=xbar cache")
    state_ids = _state_ids(base)

    with np.load(args.codes, allow_pickle=True) as data:
        codes = np.asarray(data[f"codes/{args.label}"], np.uint8)
        code_state_ids = [
            str(value) for value in np.asarray(data[f"state_ids/{args.label}"])]
    with np.load(args.posteriors, allow_pickle=False) as data:
        required = {"target_indices", "target_codes", "entropy_bits", "wm_argmax"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"posterior dump missing {sorted(missing)}")
        posterior = {name: np.asarray(data[name]) for name in required}
    posterior_report_path = Path(args.posteriors).with_suffix(".json")
    if posterior_report_path.exists():
        posterior_report = json.loads(posterior_report_path.read_text(encoding="utf-8"))
        if posterior_report.get("protocol") != POSTERIOR_PROTOCOL:
            raise ValueError("posterior report protocol mismatch")

    with np.load(args.mask, allow_pickle=False) as data:
        utility = np.asarray(data["utility"], np.float64)
        lam = float(np.asarray(data["lam"]).item())
        mask_metadata = json.loads(str(np.asarray(data["metadata"]).item()))
    if mask_metadata.get("protocol") != MASK_PROTOCOL:
        raise ValueError("utility mask protocol mismatch")

    full, gated, diagnostics = build_views(
        state_ids, codes, code_state_ids, cache_order(Path(args.records), args.label),
        posterior, utility, lam, _load_codebook(Path(args.codebook)))
    if full.shape != np.asarray(base["xbar"]).shape:
        raise ValueError(
            f"reconstruction shape {full.shape} != cache xbar {base['xbar'].shape}")

    metadata.update({
        "representation": "utility_gated",
        "base_cache": str(Path(args.cache).resolve()),
        "utility_views": {
            "codes": str(Path(args.codes).resolve()),
            "codebook": str(Path(args.codebook).resolve()),
            "posteriors": str(Path(args.posteriors).resolve()),
            "mask": str(Path(args.mask).resolve()),
            "records": str(Path(args.records).resolve()),
            "label": args.label,
            "lambda": lam,
            "full_view": "OPQ decode of true codes",
            "gated_view": "OPQ decode after utility-dropped codes use WM argmax",
            "initial_state_rule": "all-send when no causal WM posterior exists",
            **diagnostics,
        },
    })
    payload = dict(base)
    payload.update({
        "state_id": np.asarray(state_ids),
        "full_recon_xbar": full,
        "gated_recon_xbar": gated,
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    })
    output = Path(args.output)
    if output.resolve() == Path(args.cache).resolve():
        raise ValueError("refusing to overwrite the base retrieval cache")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    print(f"wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
