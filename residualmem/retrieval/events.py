from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from ..types import StateSchema, Trajectory


CRAFTER_ACTIONS = (
    "noop",
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "do",
    "sleep",
    "place_stone",
    "place_table",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
)


@dataclasses.dataclass(frozen=True)
class SegmentEvents:
    segment_id: int
    start_step: int
    end_step: int
    actions: tuple[str, ...]
    changed_fields: tuple[str, ...]
    literals: tuple[str, ...]
    entities: tuple[str, ...]

    def document(self) -> str:
        return "\n".join(
            (
                f"time_range: {self.start_step} {self.end_step}",
                f"actions: {' '.join(self.actions)}",
                f"changed_fields: {' '.join(self.changed_fields)}",
                f"literals: {' '.join(self.literals)}",
                f"entities: {' '.join(self.entities)}",
            )
        )


def extract_segment_events(
    trajectory: Trajectory,
    schema: StateSchema,
    segment_length: int = 64,
    action_names: tuple[str, ...] = CRAFTER_ACTIONS,
) -> tuple[SegmentEvents, ...]:
    if segment_length < 1:
        raise ValueError("segment_length must be positive")
    for state in trajectory.states:
        schema.validate(state)
    starts = list(range(0, len(trajectory.actions), segment_length)) or [0]
    output = []
    for segment_id, start in enumerate(starts):
        stop = min(start + segment_length, len(trajectory.actions))
        states = trajectory.states[start: stop + 1]
        actions = trajectory.actions[start:stop]
        if any(action < 0 or action >= len(action_names) for action in actions):
            raise ValueError("trajectory contains an unknown action id")
        action_values = {action_names[action] for action in actions}
        changed = {
            spec.name
            for before, after in zip(states, states[1:], strict=False)
            for spec, old, new in zip(
                schema.fields, before.values, after.values, strict=True
            )
            if old != new
        }
        literals = {
            str(value)
            for state in states
            for spec, value in zip(schema.fields, state.values, strict=True)
            if spec.field_type == "literal"
        }
        entities = _entities(states, schema, changed, action_values, literals)
        output.append(
            SegmentEvents(
                segment_id=segment_id,
                start_step=start,
                end_step=stop,
                actions=tuple(sorted(action_values)),
                changed_fields=tuple(sorted(changed)),
                literals=tuple(sorted(literals)),
                entities=tuple(sorted(entities)),
            )
        )
    return tuple(output)


def export_index_documents(
    events: tuple[SegmentEvents, ...], path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(
                json.dumps(
                    {
                        "segment_id": event.segment_id,
                        "start_step": event.start_step,
                        "end_step": event.end_step,
                        "document": event.document(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return path


def _entities(states, schema, changed, actions, literals) -> set[str]:
    entities = set(literals)
    for action in actions:
        if action.startswith("place_"):
            entities.add(action.removeprefix("place_"))
        elif action.startswith("make_"):
            entities.add(action.removeprefix("make_"))
    for index, spec in enumerate(schema.fields):
        if spec.name.startswith("inventory/"):
            name = spec.name.split("/", 1)[1]
            if spec.name in changed or any(int(state[index]) > 0 for state in states):
                entities.add(name)
        elif spec.name.startswith("achievement/"):
            name = spec.name.split("/", 1)[1]
            if spec.name in changed or any(bool(state[index]) for state in states):
                parts = name.split("_", 1)
                entities.add(parts[1] if len(parts) == 2 else name)
    return entities
