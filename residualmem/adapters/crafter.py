from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..schemas import CRAFTER_ACHIEVEMENTS, CRAFTER_INVENTORY
from ..types import CanonicalState, StateSchema, Trajectory


INITIAL_INVENTORY = {
    "health": 9, "food": 9, "drink": 9, "energy": 9,
    **{name: 0 for name in CRAFTER_INVENTORY[4:]},
}


class CrafterOracleAdapter:
    adapter_id = "crafter-oracle-v1"

    def __init__(self, schema: StateSchema):
        self.schema = schema

    @property
    def hash_bytes(self) -> bytes:
        return hashlib.sha256(self.adapter_id.encode("utf-8")).digest()

    def initial_state(self) -> CanonicalState:
        info = {
            "player_pos": [32, 32],
            "inventory": INITIAL_INVENTORY,
            "achievements": {name: 0 for name in CRAFTER_ACHIEVEMENTS},
        }
        return self.encode(info)

    def encode(self, observation: dict[str, Any]) -> CanonicalState:
        info = observation.get("info", observation)
        try:
            pos = info["player_pos"]
            inventory = info["inventory"]
            achievements = info["achievements"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Crafter oracle state requires player_pos, inventory, and achievements") from exc
        if len(pos) < 2:
            raise ValueError("player_pos must contain x and y")
        values: dict[str, Any] = {"pos_x": int(pos[0]), "pos_y": int(pos[1])}
        for name in CRAFTER_INVENTORY:
            if name not in inventory:
                raise ValueError(f"missing Crafter inventory field: {name}")
            values[f"inventory/{name}"] = int(inventory[name])
        for name in CRAFTER_ACHIEVEMENTS:
            if name not in achievements:
                raise ValueError(f"missing Crafter achievement field: {name}")
            values[f"achievement/{name}"] = bool(int(achievements[name]) > 0)
        return self.schema.make_state(values)

    def render(self, state: CanonicalState) -> list[dict[str, Any]]:
        return [{"type": "canonical_state", "state": self.schema.as_dict(state)}]


def _snapshot_env(env: Any) -> dict[str, Any]:
    player = getattr(env, "_player", None)
    if player is None:
        raise RuntimeError("crafter==1.8.3 private player state is unavailable")
    return {
        "player_pos": np.asarray(player.pos).tolist(),
        "inventory": dict(player.inventory),
        "achievements": dict(player.achievements),
    }


def collect_crafter(
    schema: StateSchema,
    seed: int,
    steps: int,
    policy: str = "random",
) -> Trajectory:
    if policy != "random":
        raise ValueError("only the deterministic seeded random sanity policy is built in")
    try:
        import crafter
    except ImportError as exc:
        raise RuntimeError("install crafter==1.8.3 to collect trajectories") from exc
    env = crafter.Env(seed=seed, reward=True, length=steps)
    adapter = CrafterOracleAdapter(schema)
    rng = np.random.default_rng(seed)
    env.reset()
    states = [adapter.encode(_snapshot_env(env))]
    actions: list[int] = []
    try:
        for _ in range(steps):
            action = int(rng.integers(0, env.action_space.n))
            result = env.step(action)
            _, _, done, info = result[:4]
            actions.append(action)
            states.append(adapter.encode(info))
            if done:
                break
    finally:
        close = getattr(env, "close", None)
        if close:
            close()
    return Trajectory(
        tuple(states), tuple(actions), f"crafter-seed{seed}",
        {"source": "crafter", "seed": seed, "policy": policy})


def load_emembench(path: str | Path, schema: StateSchema) -> Trajectory:
    path = Path(path)
    adapter = CrafterOracleAdapter(schema)
    states = [adapter.initial_state()]
    actions: list[int] = []
    seed: int | None = None
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                action = int(record["action_id"])
                info = record["info"]
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid EMemBench record at line {lineno}") from exc
            actions.append(action)
            states.append(adapter.encode(info))
            if seed is None and "seed" in record:
                seed = int(record["seed"])
    return Trajectory(
        tuple(states), tuple(actions), path.stem,
        {"source": "emembench", "path": str(path), "seed": seed})
