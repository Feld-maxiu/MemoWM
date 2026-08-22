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


@dataclass(frozen=True)
class AxtreeStyle:
    """Which serialization rules the synthetic AXTree follows.

    The v1 rows were valid input but not the *training* wire format, and the gap
    is larger than it looks. ``compact_axtree`` emits

        <ref=13 parent=0 tag=button text="no" flags=0,0,0,1/>

    while v1 emitted

        <node ref="wma_caption_0" tag="textarea" value="..." />

    which differs in five ways at once: a ``node`` prefix that is not an
    attribute, quoted non-numeric refs, no ``parent``, no ``flags``, and a space
    before the close. Measured on a real web_01 observation, the downstream
    effect is not cosmetic -- a training state parses to 14 nodes and 7
    candidates spread over ``choices``/``actions``, while the synthetic row gives
    1 node and 1 candidate, all of it ``current_state``, holding a 365-character
    prose caption where training holds 4-6 character UI literals. The weighted
    round-robin in ``static_candidates`` then has nothing to rotate into its
    ``actions``/``choices``/``auxiliary`` turns.

    Each flag is separable so its contribution can be attributed on its own.
    Defaults reproduce v1 exactly, so nothing changes until asked.
    """

    canonical_rows: bool = False
    """Emit ``<ref=N parent=N tag=X .../>`` byte-for-byte like ``compact_axtree``."""

    root_node: bool = False
    """Emit the ``<ref=0 parent=0 tag=root/>`` row every training AXTree starts
    with. Without it the synthetic rows' ``parent=0`` dangles -- no node with
    ref 0 exists -- which is a structural difference from training on top of a
    reference that resolves to nothing. The row carries no text: training's root
    holds the task title, and WorldMemArena observations do not supply one, so
    inventing a title would be fabricating content rather than aligning format."""

    split_captions: bool = False
    """One node per caption sentence, approaching the training span statistics.
    Measured slightly *worse* than leaving captions whole (r_perp 0.4304 against
    0.4205), so it is off by default and kept only for reruns."""

    explicit_no_instruction: bool = False
    """Emit a fixed marker row when there is no user text, instead of branching
    to a different structural template. On WorldMemArena web the instruction is
    always absent, so v1 always took the other branch while training always took
    this one."""


DEFAULT_AXTREE_STYLE = AxtreeStyle(
    canonical_rows=True, root_node=True, explicit_no_instruction=True
)
"""The frozen serialization, decided 2026-08-22.

Screened against the BrowserGym-only PCA as a fixed ruler, on 299 web
observations, this is best or tied on every slot-set-robust metric and worse on
none: r_perp 0.4121 against v1's 0.4205, nearest-neighbour median 13.675 against
13.879. The margin is small next to the gap that needs closing -- the in-domain
yardstick is r_perp 0.0816 and NN 6.131 -- which is the finding, not a
disappointment: *formatting is not what breaks WorldMemArena*. Five variants
moved r_perp by at most 2%, so the shift is in what the observations contain,
not how they are written, and only refitting the basis addresses it.

``split_captions`` is excluded: it measured slightly worse (r_perp 0.4304) than
leaving captions whole.

Changing this changes H16, so any PCA fitted under a different style is a
different coordinate system. Freeze before fitting, not after.
"""

LEGACY_V1_AXTREE_STYLE = AxtreeStyle()
"""The pre-2026-08-22 serialization, kept to reproduce numbers recorded under it."""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _row(style: AxtreeStyle, ref, parent, tag: str, *, text=None, value=None) -> str:
    if not style.canonical_rows:
        attributes = f'ref="{ref}" tag="{tag}"'
        if value is not None:
            attributes += " value=" + json.dumps(value, ensure_ascii=False)
        if text is not None:
            attributes += " text=" + json.dumps(text, ensure_ascii=False)
        return f"<node {attributes} />"
    pieces = [f"ref={ref}", f"parent={parent}", f"tag={tag}"]
    if text is not None:
        pieces.append("text=" + json.dumps(text, ensure_ascii=False))
    if value is not None:
        pieces.append("value=" + json.dumps(value, ensure_ascii=False))
    # focused, tampered, reserved, is_leaf -- the shape compact_axtree writes for
    # an untouched leaf, which every synthetic node is.
    pieces.append("flags=0,0,0,1")
    return "<" + " ".join(pieces) + "/>"


def synthetic_axtree(
    observation: WebObservation, style: AxtreeStyle = DEFAULT_AXTREE_STYLE
) -> str:
    """Map WMA text fields to valid compact AXTree rows without a learned adapter.

    One thing here deliberately does *not* match training. Most training rows
    carry their literal under ``text``; these carry it under ``value``, and that
    has to stay. ``static_candidates`` groups ``text`` only for ACTION/CHOICE/
    CHECKABLE tags, so ``tag=textarea text="..."`` matches no rule and the
    caption is dropped before slot contention -- zero candidates, verified. It is
    ``value`` on an EDITABLE tag that routes it to ``current_state``. Aligning
    this particular attribute to training would silently delete the text.
    """
    rows: list[str] = []
    ref = 1

    def emit(tag: str, *, text=None, value=None) -> None:
        nonlocal ref
        rows.append(_row(style, ref if style.canonical_rows else f"wma_{ref}",
                         0, tag, text=text, value=value))
        ref += 1

    user_text = observation.user_text.strip()
    if user_text:
        if style.canonical_rows:
            emit("textarea", value=user_text)
        else:
            rows.append(f'<node ref="wma_user" tag="textarea" '
                        f'value={json.dumps(user_text, ensure_ascii=False)} />')
            ref += 1
    elif style.explicit_no_instruction:
        # One structural template for both cases. Branching to a different shape
        # when the instruction is absent is itself a distribution difference, and
        # on WorldMemArena web the instruction is *always* absent.
        emit("textarea", value="<no_instruction>")

    for index, caption in enumerate(observation.captions):
        caption = caption.strip()
        if not caption:
            continue
        if style.canonical_rows:
            parts = (
                [p for p in _SENTENCE_SPLIT.split(caption) if p.strip()]
                if style.split_captions else [caption]
            )
            for part in parts:
                emit("textarea", value=part.strip())
        else:
            rows.append(f'<node ref="wma_caption_{index}" tag="textarea" '
                        f'value={json.dumps(caption, ensure_ascii=False)} />')
            ref += 1

    # The frozen AXTree pooler requires at least one node.  Screenshot-only
    # rounds get a semantically empty structural row; image slots still carry
    # all observation content and this row is not marked as a detail candidate.
    if not rows:
        if style.canonical_rows:
            rows.append(_row(style, 0, 0, "document"))
        else:
            rows.append('<node ref="wma_root" tag="document" />')
    elif style.root_node:
        rows.insert(0, _row(style, 0, 0, "root"))
    return "\n".join(rows)


def validate_observation(observation: WebObservation) -> None:
    if observation.screenshot and not Path(observation.screenshot).is_file():
        raise FileNotFoundError(observation.screenshot)
    if not observation.has_observation:
        raise ValueError("round has no screenshot, user text, or caption")

