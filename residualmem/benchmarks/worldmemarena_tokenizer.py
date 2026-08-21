"""Leakage-safe WorldMemArena web observations for the frozen tokenizer.

The state tokenizer consumes the *current observation*.  An assistant turn is
the policy output (plan/action), not part of that observation, and is therefore
never accepted by this module.  WorldMemArena captions are represented as a
small synthetic accessibility-tree row so the frozen AXTree selector can be
reused without training on benchmark examples.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WMA_OBSERVATION_PROTOCOL = "worldmemarena_web_observation_v1"


@dataclass(frozen=True)
class WebObservation:
    screenshot: str | None
    user_text: str
    captions: tuple[str, ...]
    image_ids: tuple[str, ...] = ()

    @property
    def has_observation(self) -> bool:
        return bool(self.screenshot or self.user_text.strip() or any(x.strip() for x in self.captions))


_INLINE_ATTACHMENT = re.compile(
    r"(?:\n\s*)*image:\s*\n(?:image_id:[^\n]*\n)?image_caption:.*?(?=(?:\n\s*image:\s*\n)|\Z)",
    flags=re.DOTALL,
)


def genuine_user_text(text: str) -> str:
    """Remove the loader-inlined copy of attachment metadata/captions."""
    return _INLINE_ATTACHMENT.sub("", str(text or "")).strip()


def observation_from_turn(turn) -> WebObservation:
    """Extract screenshot + genuine user text + captions from a user turn."""
    if getattr(turn, "role", None) != "user":
        raise ValueError("tokenizer observations must come from user turns; assistant is action output")
    paths: list[str] = []
    captions: list[str] = []
    image_ids: list[str] = []
    for attachment in getattr(turn, "attachments", ()):
        caption = str(getattr(attachment, "caption", "") or "").strip()
        if caption:
            captions.append(caption)
        path = getattr(attachment, "file_path", None) or getattr(attachment, "image_path", None)
        if path:
            paths.append(str(path))
        image_id = getattr(attachment, "image_id", None)
        if image_id:
            image_ids.append(str(image_id))
    return WebObservation(
        screenshot=paths[0] if paths else None,
        user_text=genuine_user_text(getattr(turn, "text", "")),
        captions=tuple(captions),
        image_ids=tuple(image_ids),
    )


def fused_observation_text(user_text: str, captions: Iterable[str]) -> str:
    """Canonical text paired with a screenshot by the VL teacher/baseline."""
    blocks: list[str] = []
    clean_user = genuine_user_text(user_text)
    if clean_user:
        blocks.append(f"Current user observation:\n{clean_user}")
    clean_captions = [str(value).strip() for value in captions if str(value).strip()]
    if clean_captions:
        rendered = "\n".join(f"[{index}] {value}" for index, value in enumerate(clean_captions))
        blocks.append(f"Image captions:\n{rendered}")
    return "\n\n".join(blocks)


def synthetic_axtree(observation: WebObservation) -> str:
    """Map WMA text fields to valid compact AXTree rows without a learned adapter."""
    rows: list[str] = []
    if observation.user_text.strip():
        value = json.dumps(observation.user_text.strip(), ensure_ascii=False)
        rows.append(f'<node ref="wma_user" tag="textarea" value={value} />')
    for index, caption in enumerate(observation.captions):
        if caption.strip():
            value = json.dumps(caption.strip(), ensure_ascii=False)
            rows.append(f'<node ref="wma_caption_{index}" tag="textarea" value={value} />')
    # The frozen AXTree pooler requires at least one node.  Screenshot-only
    # rounds get a semantically empty structural row; image slots still carry
    # all observation content and this row is not marked as a detail candidate.
    if not rows:
        rows.append('<node ref="wma_root" tag="document" />')
    return "\n".join(rows)


def validate_observation(observation: WebObservation) -> None:
    if observation.screenshot and not Path(observation.screenshot).is_file():
        raise FileNotFoundError(observation.screenshot)
    if not observation.has_observation:
        raise ValueError("round has no screenshot, user text, or caption")

