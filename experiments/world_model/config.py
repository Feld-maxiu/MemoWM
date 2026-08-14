"""YAML configuration loading and hard protocol validation for the v8 WM."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import yaml

from .model import ModelConfig
from .schema import MAX_HISTORY, MAX_PAYLOAD_BYTES, PROTOCOL, validate_variant


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    warmup_steps: int = 1_000
    max_steps: int = 20_000
    min_steps: int = 5_000
    eval_every: int = 500
    patience_steps: int = 2_500
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    matmul_precision: str = "highest"

    def __post_init__(self):
        if min(
            self.batch_size, self.max_steps, self.min_steps, self.eval_every,
            self.patience_steps, self.gradient_clip,
        ) <= 0:
            raise ValueError("training counts and gradient clip must be positive")
        if self.warmup_steps < 0 or self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must lie in [0,max_steps)")
        if self.min_steps > self.max_steps:
            raise ValueError("min_steps cannot exceed max_steps")
        if self.patience_steps % self.eval_every:
            raise ValueError("patience_steps must be divisible by eval_every")
        if self.matmul_precision != "highest":
            raise ValueError("formal v8 WM runs require highest matmul precision")

    @property
    def patience_evals(self) -> int:
        return self.patience_steps // self.eval_every


@dataclasses.dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 32
    bootstrap_replicates: int = 10_000
    rollout_horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)

    def __post_init__(self):
        object.__setattr__(self, "rollout_horizons", tuple(self.rollout_horizons))
        if self.batch_size < 1 or self.bootstrap_replicates < 1:
            raise ValueError("evaluation sizes must be positive")
        if not self.rollout_horizons or min(self.rollout_horizons) < 1:
            raise ValueError("rollout horizons must be positive")


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    protocol: str
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    action: dict


def load_config(path: str | Path, *, num_tasks: int | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("protocol") != PROTOCOL:
        raise ValueError(f"config must declare protocol: {PROTOCOL}")
    model_values = dict(raw.get("model") or {})
    if num_tasks is not None:
        configured = model_values.get("num_tasks", num_tasks)
        if int(configured) != int(num_tasks):
            raise ValueError(
                f"config num_tasks={configured} disagrees with cache num_tasks={num_tasks}"
            )
        model_values["num_tasks"] = int(num_tasks)
    model = ModelConfig(**model_values)
    training = TrainingConfig(**(raw.get("training") or {}))
    evaluation_values = dict(raw.get("evaluation") or {})
    evaluation = EvaluationConfig(**evaluation_values)
    action = dict(raw.get("action") or {})
    expected_action = {
        "exclude_policy": True,
        "max_ref": 63,
        "payload_bytes": MAX_PAYLOAD_BYTES,
        "select_option_uses_parent": True,
    }
    for key, expected in expected_action.items():
        if action.get(key) != expected:
            raise ValueError(f"action.{key} must be {expected!r}")
    if model.max_history != MAX_HISTORY:
        raise ValueError(f"formal v8 config requires max_history={MAX_HISTORY}")
    return ExperimentConfig(
        protocol=raw["protocol"], model=model, training=training,
        evaluation=evaluation, action=action,
    )


def resolved_dict(
    config: ExperimentConfig, *, variant: str, seed: int, allow_dev: bool = False
) -> dict:
    return {
        "protocol": config.protocol,
        "variant": validate_variant(variant, allow_dev=allow_dev),
        "seed": int(seed),
        "model": dataclasses.asdict(config.model),
        "training": dataclasses.asdict(config.training),
        "evaluation": dataclasses.asdict(config.evaluation),
        "action": config.action,
    }
