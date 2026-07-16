from __future__ import annotations

from collections.abc import Mapping

from .types import FieldSpec, StateSchema


CRAFTER_INVENTORY = (
    "health", "food", "drink", "energy", "sapling", "wood", "stone",
    "coal", "iron", "diamond", "wood_pickaxe", "stone_pickaxe",
    "iron_pickaxe", "wood_sword", "stone_sword", "iron_sword",
)

CRAFTER_ACHIEVEMENTS = (
    "collect_coal", "collect_diamond", "collect_drink", "collect_iron",
    "collect_sapling", "collect_stone", "collect_wood", "defeat_skeleton",
    "defeat_zombie", "eat_cow", "eat_plant", "make_iron_pickaxe",
    "make_iron_sword", "make_stone_pickaxe", "make_stone_sword",
    "make_wood_pickaxe", "make_wood_sword", "place_furnace", "place_plant",
    "place_stone", "place_table", "wake_up",
)


def crafter_schema(
    profile: str = "exact",
    task_weights: Mapping[str, float] | None = None,
) -> StateSchema:
    if profile not in {"exact", "rd_uniform", "rd_task"}:
        raise ValueError(f"unknown Crafter schema profile: {profile}")
    task_weights = dict(task_weights or {})
    policy = "must" if profile == "exact" else "weighted"

    def weight(name: str) -> float:
        return 1.0 if profile != "rd_task" else float(task_weights.get(name, 0.05))

    fields = [
        FieldSpec("pos_x", "integer", policy, weight("pos_x"), num_values=64),
        FieldSpec("pos_y", "integer", policy, weight("pos_y"), num_values=64),
    ]
    fields += [
        FieldSpec(
            f"inventory/{name}", "integer", policy,
            weight(f"inventory/{name}"), num_values=10)
        for name in CRAFTER_INVENTORY
    ]
    fields += [
        FieldSpec(
            f"achievement/{name}", "bool", policy,
            weight(f"achievement/{name}"))
        for name in CRAFTER_ACHIEVEMENTS
    ]
    return StateSchema(f"crafter-v1-{profile}", tuple(fields))
