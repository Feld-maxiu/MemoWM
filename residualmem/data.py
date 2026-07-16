from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import CanonicalState, StateSchema, Trajectory


def save_trajectory(path: str | Path, trajectory: Trajectory, schema: StateSchema) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for state in trajectory.states:
        schema.validate(state)
    metadata = {
        "schema_id": schema.schema_id,
        "schema_hash": schema.hash_hex,
        "episode_id": trajectory.episode_id,
        "metadata": dict(trajectory.metadata),
    }
    np.savez_compressed(
        path,
        states_json=np.asarray(json.dumps(
            [list(state.values) for state in trajectory.states],
            ensure_ascii=False,
            separators=(",", ":"),
        )),
        actions=np.asarray(trajectory.actions, dtype=np.int32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def load_trajectory(path: str | Path, schema: StateSchema) -> Trajectory:
    with np.load(Path(path), allow_pickle=False) as data:
        metadata: dict[str, Any] = json.loads(str(data["metadata"]))
        if metadata["schema_hash"] != schema.hash_hex:
            raise ValueError(
                f"schema hash mismatch: {metadata['schema_hash']} != {schema.hash_hex}")
        if "states_json" in data:
            rows = json.loads(str(data["states_json"]))
            states = tuple(schema.make_state(row) for row in rows)
        else:
            # Backward compatibility with the first numeric-only prototype.
            states = tuple(schema.make_state(row.tolist()) for row in data["states"])
        actions = tuple(int(x) for x in data["actions"])
    return Trajectory(
        states, actions, metadata.get("episode_id", ""), metadata.get("metadata", {}))


def save_manifest(path: str | Path, records: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    return path
