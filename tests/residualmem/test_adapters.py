from __future__ import annotations

import json

from residualmem.adapters.crafter import CrafterOracleAdapter, load_emembench
from residualmem.schemas import (
    CRAFTER_ACHIEVEMENTS,
    CRAFTER_INVENTORY,
    crafter_schema,
)


def _info():
    return {
        "player_pos": [7, 11],
        "inventory": {
            name: (9 if name in {"health", "food", "drink", "energy"} else 0)
            for name in CRAFTER_INVENTORY
        },
        "achievements": {name: 0 for name in CRAFTER_ACHIEVEMENTS},
    }


def test_crafter_oracle_has_stable_40_field_schema():
    schema = crafter_schema("exact")
    adapter = CrafterOracleAdapter(schema)
    state = adapter.encode(_info())

    assert len(schema.fields) == 40
    assert schema.as_dict(state)["pos_x"] == 7
    assert schema.as_dict(state)["pos_y"] == 11
    assert len(adapter.hash_bytes) == 32


def test_emembench_jsonl_import(tmp_path):
    record = {"action_id": 4, "seed": 3, "info": _info()}
    path = tmp_path / "episode.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    trajectory = load_emembench(path, crafter_schema("exact"))

    assert trajectory.actions == (4,)
    assert len(trajectory.states) == 2
    assert trajectory.metadata["seed"] == 3
