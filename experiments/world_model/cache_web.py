from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.common import sha256_file, write_json

from .schema import MAX_HISTORY, MAX_PAYLOAD_BYTES, SPLIT_IDS
from .schema_web import (
    ACTION_TYPE_IDS,
    NUM_ACTION_TYPES,
    WEB_CACHE_PROTOCOL as PROTOCOL,
    WebAction,
    action_side_information_bits,
)


CACHE_FORMAT_VERSION = 1

CACHE_FILES = {
    "codes": "codes.npy",
    "valid": "valid.npy",
    "global_indices": "global_indices.npy",
    "transitions": "transitions.npz",
    "manifest": "manifest.json",
}



def _jsonl_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield line


def _action_from_record(payload: dict) -> WebAction:
    return WebAction(
        type_id=int(payload["type_id"]),
        x=float(payload["x"]), y=float(payload["y"]),
        dx=float(payload["dx"]), dy=float(payload["dy"]),
        payload=str(payload["payload"]).encode("utf-8")[:MAX_PAYLOAD_BYTES],
        has_coord=bool(payload["has_coord"]),
        has_delta=bool(payload["has_delta"]),
        source_id=int(payload.get("source_id", 0)),
        merged=int(payload.get("merged", 1)),
    )


def load_categories(paths: Path | list[Path]) -> int:
    sizes = set()
    for path in ([paths] if isinstance(paths, Path) else paths):
        with np.load(path, allow_pickle=True) as data:
            if "centroids" in data.files:
                sizes.add(int(np.asarray(data["centroids"]).shape[2]))
    if len(sizes) > 1:
        raise SystemExit(f"artifacts disagree about the alphabet size: {sorted(sizes)}")
    if not sizes:
        raise SystemExit("no artifact carries centroids; cannot infer the alphabet")
    return sizes.pop()


def load_codes(paths: Path | list[Path]) -> dict[str, np.ndarray]:
    table: dict[str, np.ndarray] = {}
    for path in ([paths] if isinstance(paths, Path) else paths):
        with np.load(path, allow_pickle=True) as data:
            splits = [k.split("/", 1)[1] for k in data.files if k.startswith("codes/")]
            for split in splits:
                codes = np.asarray(data[f"codes/{split}"], np.uint8)
                ids = [str(v) for v in np.asarray(data[f"state_ids/{split}"])]
                if len(codes) != len(ids):
                    raise SystemExit(f"{split}: {len(codes)} codes against {len(ids)} ids")
                for state_id, row in zip(ids, codes):
                    if state_id in table and not np.array_equal(table[state_id], row):
                        raise SystemExit(
                            f"{state_id} has different codes in two artifacts"
                        )
                    table[state_id] = row
    return table


def load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        for line in _jsonl_lines(path):
            if line.strip():
                records.append(json.loads(line))
    return records


def _check_width(name: str, values: np.ndarray, dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    if values.size and (values.min() < info.min or values.max() > info.max):
        raise SystemExit(
            f"{name}: values span [{values.min()}, {values.max()}], "
            f"outside {dtype.__name__} [{info.min}, {info.max}]"
        )
    return values.astype(dtype)


def build_transitions(records: list[dict], output: Path, *,
                      max_history: int = MAX_HISTORY) -> dict:
    episodes: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        episodes[record["episode_id"]].append(record)

    task_names = sorted({r["task"] for r in records})
    task_index = {name: i for i, name in enumerate(task_names)}
    episode_names = sorted(episodes)
    episode_index = {name: i for i, name in enumerate(episode_names)}

    columns: dict[str, list] = {k: [] for k in (
        "history_indices", "target_indices", "task_ids", "episode_ids", "steps",
        "split_ids", "action_types", "action_payloads", "action_lengths",
        "action_x", "action_y", "action_dx", "action_dy",
        "action_has_coord", "action_has_delta",
        "structural_action_bits", "full_action_bits",
    )}
    counts: Counter[str] = Counter()
    type_histogram: Counter[str] = Counter()

    for name in episode_names:
        values = sorted(episodes[name], key=lambda item: int(item["step"]))
        steps = [int(item["step"]) for item in values]
        if steps != list(range(len(values))):
            raise SystemExit(
                f"episode {name} has non-consecutive steps: {steps[:8]}..."
            )
        if len({item["split"] for item in values}) != 1:
            raise SystemExit(f"episode {name} spans splits")
        if len(values) < 2:
            continue

        actions = [_action_from_record(item["action"]) for item in values[:-1]]
        for action in actions:
            type_histogram[action.type_name] += 1

        for position in range(len(values) - 1):
            current, target = values[position], values[position + 1]
            start = max(0, position - max_history + 1)
            prefix = values[start:position + 1]
            prefix_actions = actions[start:position + 1]
            pad = max_history - len(prefix)

            columns["history_indices"].append(
                [-1] * pad + [int(item["global_index"]) for item in prefix]
            )
            padded: list[WebAction | None] = [None] * pad + prefix_actions
            rows: dict[str, list] = {k: [] for k in (
                "action_types", "action_payloads", "action_lengths",
                "action_x", "action_y", "action_dx", "action_dy",
                "action_has_coord", "action_has_delta")}
            for action in padded:
                if action is None:
                    rows["action_types"].append(ACTION_TYPE_IDS["PAD"])
                    rows["action_payloads"].append(np.zeros((MAX_PAYLOAD_BYTES,), np.uint8))
                    rows["action_lengths"].append(0)
                    for key in ("action_x", "action_y", "action_dx", "action_dy"):
                        rows[key].append(0.0)
                    rows["action_has_coord"].append(False)
                    rows["action_has_delta"].append(False)
                    continue
                rows["action_types"].append(action.type_id)
                rows["action_payloads"].append(action.padded_payload())
                rows["action_lengths"].append(action.payload_length)
                rows["action_x"].append(action.x)
                rows["action_y"].append(action.y)
                rows["action_dx"].append(action.dx)
                rows["action_dy"].append(action.dy)
                rows["action_has_coord"].append(action.has_coord)
                rows["action_has_delta"].append(action.has_delta)
            for key, value in rows.items():
                columns[key].append(value)

            columns["target_indices"].append(int(target["global_index"]))
            columns["task_ids"].append(task_index[current["task"]])
            columns["episode_ids"].append(episode_index[name])
            columns["steps"].append(int(current["step"]))
            columns["split_ids"].append(SPLIT_IDS[current["split"]])
            current_action = prefix_actions[-1]
            columns["structural_action_bits"].append(
                action_side_information_bits(current_action, include_payload=False)
            )
            columns["full_action_bits"].append(
                action_side_information_bits(current_action, include_payload=True)
            )
            counts[current["split"]] += 1

    arrays = {
        "history_indices": np.asarray(columns["history_indices"], np.int64),
        "target_indices": np.asarray(columns["target_indices"], np.int64),
        "task_ids": _check_width("task_ids", np.asarray(columns["task_ids"]), np.uint8),
        "episode_ids": _check_width("episode_ids", np.asarray(columns["episode_ids"]), np.int32),
        "steps": _check_width("steps", np.asarray(columns["steps"]), np.uint16),
        "split_ids": _check_width("split_ids", np.asarray(columns["split_ids"]), np.uint8),
        "action_types": _check_width("action_types", np.asarray(columns["action_types"]), np.uint8),
        "action_payloads": np.asarray(columns["action_payloads"], np.uint8),
        "action_lengths": _check_width("action_lengths", np.asarray(columns["action_lengths"]), np.uint8),
        "action_x": np.asarray(columns["action_x"], np.float32),
        "action_y": np.asarray(columns["action_y"], np.float32),
        "action_dx": np.asarray(columns["action_dx"], np.float32),
        "action_dy": np.asarray(columns["action_dy"], np.float32),
        "action_has_coord": np.asarray(columns["action_has_coord"], np.bool_),
        "action_has_delta": np.asarray(columns["action_has_delta"], np.bool_),
        "structural_action_bits": _check_width(
            "structural_action_bits", np.asarray(columns["structural_action_bits"]), np.uint16),
        "full_action_bits": _check_width(
            "full_action_bits", np.asarray(columns["full_action_bits"]), np.uint16),
    }

    total = len(arrays["target_indices"])
    for name, shape in (
        ("history_indices", (total, max_history)),
        ("action_payloads", (total, max_history, MAX_PAYLOAD_BYTES)),
        ("action_types", (total, max_history)),
        ("action_x", (total, max_history)),
    ):
        if arrays[name].shape != shape:
            raise SystemExit(f"{name}: expected {shape}, got {arrays[name].shape}")
    if int(arrays["action_types"].max()) >= NUM_ACTION_TYPES:
        raise SystemExit("an action type id exceeds the taxonomy")

    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, output)
    delta = np.abs(arrays["action_dy"][arrays["action_has_delta"]])
    coordinate = arrays["action_x"][arrays["action_has_coord"]]
    return {
        "max_history": int(max_history),
        "transitions": total,
        "transition_counts": dict(counts),
        "episodes": len(episode_names),
        "tasks": task_names,
        "episode_names": episode_names,
        "action_type_histogram": dict(type_histogram.most_common()),
        "mean_full_action_bits": float(arrays["full_action_bits"].mean()),
        "mean_structural_action_bits": float(arrays["structural_action_bits"].mean()),
        "delta_diagnostics": {
            "count": int(delta.size),
            "p50": float(np.percentile(delta, 50)) if delta.size else 0.0,
            "p99_9": float(np.percentile(delta, 99.9)) if delta.size else 0.0,
            "max": float(delta.max()) if delta.size else 0.0,
            "over_50_viewports": int((delta > 50).sum()),
        },
        "coordinate_diagnostics": {
            "count": int(coordinate.size),
            "at_bound": int((coordinate >= 1.0).sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codes", type=Path, action="append", required=True,
                        help="a qformer_pq artifact; repeatable")
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-history", type=int, default=MAX_HISTORY,
                        help="history window baked into the arrays. A model may "
                             "use fewer -- FrozenCache keeps the last k -- so "
                             "build once at the widest value to be compared. "
                             "MolmoWeb averages 17.5 states per trajectory and "
                             "32 is its p90.")
    parser.add_argument("--include-splits", nargs="+", default=["train", "validation"],
                        help="test stays out unless asked for")
    args = parser.parse_args()

    records = [r for r in load_records(args.records) if r["split"] in args.include_splits]
    records.sort(key=lambda r: (r["episode_id"], int(r["step"])))
    codes = load_codes(args.codes)

    missing = [r["state_id"] for r in records if r["state_id"] not in codes]
    if missing:
        raise SystemExit(f"{len(missing)} records have no code, e.g. {missing[:3]}")

    for index, record in enumerate(records):
        record["global_index"] = index

    categories = load_categories(args.codes)
    slots, subspaces = codes[records[0]["state_id"]].shape
    stacked = np.stack([codes[r["state_id"]] for r in records]).astype(np.uint8)
    valid = np.ones((len(records), slots), np.bool_)

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / CACHE_FILES["codes"], stacked)
    np.save(args.output / CACHE_FILES["valid"], valid)
    np.save(args.output / CACHE_FILES["global_indices"],
            np.arange(len(records), dtype=np.int64))

    summary = build_transitions(records, args.output / CACHE_FILES["transitions"],
                                max_history=args.max_history)

    state_counts: Counter[str] = Counter(r["split"] for r in records)
    manifest = {
        "protocol": PROTOCOL,
        "format_version": CACHE_FORMAT_VERSION,
        "codes": [str(p.resolve()) for p in args.codes],
        "num_latent_tokens": int(slots),
        "num_subspaces": int(subspaces),
        "num_categories": categories,
        "layout": [int(slots)],
        "state_counts": dict(state_counts),
        "fixed_width_bits": int(slots * subspaces * math.log2(categories)),
        **summary,
        "artifact_sha256": {
            name: sha256_file(args.output / filename)
            for name, filename in CACHE_FILES.items() if name != "manifest"
        },
    }
    write_json(args.output / CACHE_FILES["manifest"], manifest)

    identity = len(records) - summary["episodes"]
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ("episode_names", "artifact_sha256")},
                     ensure_ascii=False, indent=2))
    print(f"\nstates - episodes = {len(records)} - {summary['episodes']} = {identity}"
          f"  against {summary['transitions']} transitions"
          f"  {'OK' if identity == summary['transitions'] else '<-- MISMATCH'}")


if __name__ == "__main__":
    main()
