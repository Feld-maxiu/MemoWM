from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal


FieldType = Literal["bool", "categorical", "integer", "literal", "vq"]
FieldPolicy = Literal["must", "weighted"]


@dataclasses.dataclass(frozen=True)
class FieldSpec:
    name: str
    field_type: FieldType
    policy: FieldPolicy = "must"
    weight: float = 1.0
    num_values: int | None = None
    scale: float = 1.0
    optional: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("field name must be non-empty")
        if self.field_type not in {"bool", "categorical", "integer", "literal", "vq"}:
            raise ValueError(f"unsupported field type: {self.field_type}")
        if self.policy not in {"must", "weighted"}:
            raise ValueError(f"unsupported field policy: {self.policy}")
        if self.weight < 0:
            raise ValueError("field weight must be non-negative")
        if self.scale <= 0:
            raise ValueError("field scale must be positive")
        if self.field_type in {"categorical", "vq"} and not self.num_values:
            raise ValueError(f"{self.field_type} fields require num_values")
        if self.num_values is not None and self.num_values < 2:
            raise ValueError("num_values must be at least 2")


@dataclasses.dataclass(frozen=True)
class CanonicalState:
    values: tuple[Any, ...]

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> Any:
        return self.values[index]


@dataclasses.dataclass(frozen=True)
class StateSchema:
    schema_id: str
    fields: tuple[FieldSpec, ...]

    def __post_init__(self) -> None:
        if not self.schema_id:
            raise ValueError("schema_id must be non-empty")
        names = [field.name for field in self.fields]
        if not names or len(names) != len(set(names)):
            raise ValueError("schema fields must be non-empty and uniquely named")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    @property
    def hash_bytes(self) -> bytes:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).digest()

    @property
    def hash_hex(self) -> str:
        return self.hash_bytes.hex()

    def canonical_json(self) -> str:
        value = {
            "schema_id": self.schema_id,
            "fields": [dataclasses.asdict(field) for field in self.fields],
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def make_state(self, values: Mapping[str, Any] | Sequence[Any]) -> CanonicalState:
        if isinstance(values, Mapping):
            missing = set(self.names) - set(values)
            extra = set(values) - set(self.names)
            if missing or extra:
                raise ValueError(f"state keys mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
            ordered = tuple(values[name] for name in self.names)
        else:
            ordered = tuple(values)
        state = CanonicalState(ordered)
        self.validate(state)
        return state

    def as_dict(self, state: CanonicalState) -> dict[str, Any]:
        self.validate(state)
        return dict(zip(self.names, state.values))

    def validate(self, state: CanonicalState) -> None:
        if len(state) != len(self.fields):
            raise ValueError(f"expected {len(self.fields)} fields, got {len(state)}")
        for spec, value in zip(self.fields, state.values):
            if value is None:
                if not spec.optional:
                    raise ValueError(f"field {spec.name} is not optional")
                continue
            if spec.field_type == "bool":
                if not isinstance(value, (bool, int)):
                    raise TypeError(f"field {spec.name} must be bool")
                if int(value) not in (0, 1):
                    raise ValueError(f"field {spec.name} must be 0 or 1")
            if spec.field_type in {"categorical", "integer", "vq"}:
                if not isinstance(value, int) or isinstance(value, bool):
                    raise TypeError(f"field {spec.name} must be int")
            if spec.field_type == "literal" and not isinstance(value, str):
                raise TypeError(f"field {spec.name} must be str")
            if spec.num_values is not None and not 0 <= int(value) < spec.num_values:
                raise ValueError(
                    f"field {spec.name}={value} outside [0, {spec.num_values})")


@dataclasses.dataclass(frozen=True)
class Prediction:
    default_state: CanonicalState
    distributions: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Trajectory:
    states: tuple[CanonicalState, ...]
    actions: tuple[int, ...]
    episode_id: str = ""
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("trajectory must contain at least one state")
        if len(self.actions) != len(self.states) - 1:
            raise ValueError(
                f"trajectory needs one action per transition: "
                f"states={len(self.states)}, actions={len(self.actions)}")


class Predictor:
    """Deterministic decoder-visible next-state predictor."""

    predictor_id = "predictor"

    def reset(self) -> None:
        raise NotImplementedError

    def predict_next(
        self, reconstructed: CanonicalState, action: int, dt: int = 1
    ) -> Prediction:
        raise NotImplementedError

    @property
    def hash_bytes(self) -> bytes:
        return hashlib.sha256(self.predictor_id.encode("utf-8")).digest()
