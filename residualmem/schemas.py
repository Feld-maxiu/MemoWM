from __future__ import annotations

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


def crafter_schema() -> StateSchema:
    fields = [
        FieldSpec("pos_x", "integer", num_values=64),
        FieldSpec("pos_y", "integer", num_values=64),
    ]
    fields += [
        FieldSpec(f"inventory/{name}", "integer", num_values=10)
        for name in CRAFTER_INVENTORY
    ]
    fields += [
        FieldSpec(f"achievement/{name}", "bool")
        for name in CRAFTER_ACHIEVEMENTS
    ]
    return StateSchema("crafter-v3-exact", tuple(fields))
