"""Canonical AMA-Bench trajectory -> x_t observation adapter.

The x_t runtime was built around ``WebObservation`` objects: a static user
instruction plus the current observation represented as caption-like text. AMA
episodes instead store a list of dictionaries containing ``turn_idx``,
``action`` and ``observation``. This module is the explicit, deterministic
bridge between the two formats.

Important invariants:

* A record is constructed from one trajectory step only. A QA question is never
  accepted as input, so document representations stay query-independent.
* The original action and observation are preserved in the rendered evidence
  text. Retrieval can therefore return the exact AMA step text to the shared QA
  reader.
* No x_t model is loaded here. The GPU/runtime layer can consume
  ``worldmem_payload()`` later without making this package depend on PyTorch,
  Qwen, residual-mem, or AMA-Bench.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Tuple


AMA_XT_OBSERVATION_PROTOCOL = "ama_xt_web_observation_v1"


def _clean(value: Any) -> str:
    """Convert nullable AMA fields to stable text without rewriting content."""
    return "" if value is None else str(value).strip()


@dataclass(frozen=True)
class AMAXTObservation:
    """One AMA trajectory step in the x_t encoder's neutral input format.

    ``user_text`` and ``captions`` intentionally mirror residual-mem's
    ``WebObservation`` fields. They are exposed as a plain mapping so this
    package stays usable from a lightweight evaluation environment.
    """

    episode_id: str
    step_index: int
    task: str
    action: str
    observation: str
    user_text: str
    captions: Tuple[str, ...]
    protocol: str = AMA_XT_OBSERVATION_PROTOCOL

    @property
    def step_text(self) -> str:
        """Exact evidence text that must be returned to the AMA Reader."""
        return self.captions[0]

    def worldmem_payload(self) -> Dict[str, Any]:
        """Return fields compatible with ``residualmem.WebObservation``.

        AMA has no screenshot attachment. The x_t runtime receives ``None`` and
        can use its existing blank-image path. ``image_ids`` is deliberately
        empty rather than fabricated.
        """
        return {
            "screenshot": None,
            "user_text": self.user_text,
            "captions": self.captions,
            "image_ids": (),
        }

    def trace_record(self) -> Dict[str, Any]:
        """Small serializable record for retrieval traces and reproducibility."""
        return {
            "protocol": self.protocol,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "action": self.action,
            "observation": self.observation,
            "step_text": self.step_text,
        }


def adapt_ama_step(
    step: Mapping[str, Any],
    *,
    episode_id: Any = "",
    task: Any = "",
    fallback_step_index: int = 0,
) -> AMAXTObservation:
    """Adapt one AMA step without using future steps or the QA question.

    Args:
        step: An AMA trajectory dictionary with ``turn_idx``, ``action`` and
            ``observation`` fields.
        episode_id: Identifier kept for traceability, not encoded as evidence.
        task: Episode-level task supplied as the static instruction channel.
        fallback_step_index: Position to use when an input omits ``turn_idx``.
    """
    if not isinstance(step, Mapping):
        raise TypeError("AMA trajectory step must be a mapping")

    raw_index = step.get("turn_idx", fallback_step_index)
    try:
        step_index = int(raw_index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid AMA turn_idx: {raw_index!r}") from exc

    action = _clean(step.get("action"))
    observation = _clean(step.get("observation"))
    if not action and not observation:
        raise ValueError("AMA trajectory step has neither action nor observation")

    # The title and labels are part of the frozen wire format. They make action
    # and observation distinguishable while preserving the original values.
    step_text = "\n".join(
        (
            f"AMA Step {step_index}:",
            f"Action: {action}",
            "Observation:",
            observation,
        )
    )
    task_text = _clean(task)
    user_text = f"AMA task:\n{task_text}" if task_text else "AMA task: <unspecified>"

    return AMAXTObservation(
        episode_id=_clean(episode_id),
        step_index=step_index,
        task=task_text,
        action=action,
        observation=observation,
        user_text=user_text,
        captions=(step_text,),
    )


def adapt_ama_trajectory(
    trajectory: Iterable[Mapping[str, Any]],
    *,
    episode_id: Any = "",
    task: Any = "",
) -> Tuple[AMAXTObservation, ...]:
    """Adapt an AMA trajectory and reject ambiguous duplicate step indices."""
    records = tuple(
        adapt_ama_step(
            step,
            episode_id=episode_id,
            task=task,
            fallback_step_index=position,
        )
        for position, step in enumerate(trajectory)
    )
    indices = [record.step_index for record in records]
    if len(indices) != len(set(indices)):
        raise ValueError("AMA trajectory contains duplicate turn_idx values")
    return records
