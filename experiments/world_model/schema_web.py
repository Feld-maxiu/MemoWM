"""Unified pixel-coordinate action schema for MolmoWeb and WorldMemArena.

The v8 line canonicalised BrowserGym actions as ``(type, tag, ref, payload)``
with ``ref`` an AXTree element id in ``0..63``.  That space does not exist
outside MiniWoB: both corpora targeted here are *coordinate* agents.

  MolmoWeb-HumanTrajs   ``mouse_click(x=539.6, y=65.8, button='left')``
  WorldMemArena web     ``{"action": "left_click", "coordinate": [903, 189]}``
                        ``pyautogui.click(760, 19)``

So ``ref``/``tag`` are dropped and replaced by a normalised viewport point.
Normalisation is what makes the two comparable at all -- MolmoWeb screenshots
are 746x720, 943x576, 1280x763, ...; WorldMemArena web is 1280x720.  A raw
pixel pair means different things in each, a fraction of the viewport does not.

Two properties are deliberately preserved from ``schema.py``:

* the payload stays a length-prefixed UTF-8 byte string capped at
  ``MAX_PAYLOAD_BYTES``, so the existing byte-GRU embedding is reusable.  It
  was measured worth +526.93 bits and is the single largest action channel.
* ``action_side_information_bits`` keeps the same shape, so rate accounting
  remains auditable against the v8 numbers.

Unlike ``schema.py`` this module also emits *sequences*.  WorldMemArena has 126
adjacent observation pairs separated by two or more actions (distribution
``{1: 803, 2: 66, 3: 24, 4: 14, >=5: 22}``); collapsing those to a single
action would silently drop conditioning information.

Dependency-light on purpose (stdlib + numpy), following ``slot_layout.py``:
the torch data pipeline and the jax world model both import it.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import re
from typing import Iterable, Mapping, Sequence

import numpy as np


PROTOCOL = "residualmem_web_unified_action_v1"
# The cache protocol lives here rather than in `cache_web` so `cache.py` can
# accept it without importing the builder, and so the string exists in exactly
# one place.
WEB_CACHE_PROTOCOL = "residualmem_web_discrete_wm_v1"

MAX_PAYLOAD_BYTES = 40

# One taxonomy over both corpora.  Types neither corpus emits are absent; PAD /
# MASK / UNK are model-only sentinels and can never come out of a parser.
ACTION_TYPE_IDS = {
    "CLICK": 0,
    "DOUBLE_CLICK": 1,
    "TRIPLE_CLICK": 2,
    "RIGHT_CLICK": 3,
    "DRAG": 4,
    "MOVE": 5,        # hover_at / mouse_move / pyautogui.moveTo
    "TYPE": 6,
    "KEY": 7,
    "SCROLL": 8,
    "WAIT": 9,        # noop(wait_ms) / wait / pyautogui.sleep
    "GOTO": 10,       # MolmoWeb only; WMA navigates by clicking the URL bar
    "SCREENSHOT": 11,  # WMA only; an explicit no-op observation request
    "TAB": 12,        # new_tab / tab_focus
    "ANSWER": 13,     # send_msg_to_user; terminal, no state change
    "PAD": 14,
    "MASK": 15,
    "UNK": 16,
}
ACTION_TYPE_NAMES = tuple(ACTION_TYPE_IDS)
NUM_ACTION_TYPES = len(ACTION_TYPE_IDS)

# Which types legitimately carry a viewport point.  Used to validate parsers and
# to decide whether the coordinate channel is billed for a given action.
COORD_TYPES = frozenset({
    "CLICK", "DOUBLE_CLICK", "TRIPLE_CLICK", "RIGHT_CLICK",
    "DRAG", "MOVE",
})
# SCROLL may or may not carry an anchor point (``scroll`` vs ``scroll_at``).
DELTA_TYPES = frozenset({"SCROLL", "DRAG"})

SOURCE_IDS = {"molmoweb": 0, "wma": 1}


@dataclasses.dataclass(frozen=True)
class WebAction:
    """One canonical action, viewport-normalised.

    ``x``/``y`` are fractions of viewport width/height in ``[0, 1]``.
    ``dx``/``dy`` are fractions too but are **not** bounded: a merged run of
    scrolls routinely covers several viewport heights, and clamping would erase
    the difference between a one-screen and a five-screen scroll.  Use
    ``signed_log_delta`` where a bounded value is needed.  All four are only
    meaningful when the matching ``has_*`` flag is set; otherwise they are
    exactly zero so a downstream embedding cannot read a spurious value.
    """

    type_id: int
    x: float = 0.0
    y: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    payload: bytes = b""
    has_coord: bool = False
    has_delta: bool = False
    source_id: int = 0
    # How many raw actions were folded into this one (scroll merging).
    merged: int = 1

    @property
    def type_name(self) -> str:
        return ACTION_TYPE_NAMES[self.type_id]

    @property
    def payload_length(self) -> int:
        return len(self.payload)

    def padded_payload(self) -> np.ndarray:
        output = np.zeros((MAX_PAYLOAD_BYTES,), np.uint8)
        if self.payload:
            output[: len(self.payload)] = np.frombuffer(self.payload, np.uint8)
        return output

    def __post_init__(self) -> None:
        if not 0 <= self.type_id < NUM_ACTION_TYPES:
            raise ValueError(f"action type id {self.type_id} out of range")
        if len(self.payload) > MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"payload is {len(self.payload)} bytes; maximum is "
                f"{MAX_PAYLOAD_BYTES} and truncation must happen in the parser"
            )
        if not self.has_coord and (self.x or self.y):
            raise ValueError("coordinate set without has_coord")
        if not self.has_delta and (self.dx or self.dy):
            raise ValueError("delta set without has_delta")


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _clip11(value: float) -> float:
    return float(min(1.0, max(-1.0, value)))


def _truncate_utf8(text: str, limit: int = MAX_PAYLOAD_BYTES) -> bytes:
    payload = str(text).encode("utf-8")
    if len(payload) <= limit:
        return payload
    payload = payload[:limit]
    while payload and (payload[-1] & 0xC0) == 0x80:  # keep valid UTF-8
        payload = payload[:-1]
    return payload


# --------------------------------------------------------------------------
# MolmoWeb-HumanTrajs
# --------------------------------------------------------------------------

_MOLMO_TYPE = {
    "goto": "GOTO",
    "mouse_click": "CLICK",
    "click": "CLICK",
    "mouse_drag_and_drop": "DRAG",
    "scroll": "SCROLL",
    "scroll_at": "SCROLL",
    "hover_at": "MOVE",
    "keyboard_type": "TYPE",
    "keyboard_press": "KEY",
    "go_back": "KEY",
    "new_tab": "TAB",
    "tab_focus": "TAB",
    "noop": "WAIT",
    "send_msg_to_user": "ANSWER",
}


def parse_molmoweb_action(step: Mapping[str, object]) -> WebAction:
    """Canonicalise one MolmoWeb trajectory step.

    ``step`` is a value of the decoded ``trajectory`` dict; it carries
    ``image_w``/``image_h`` for that step, which is what the coordinates are
    relative to.  Resolutions vary per trajectory, so the per-step values must
    be used rather than a corpus-wide constant.
    """
    width = float(step.get("image_w") or 0) or 1.0
    height = float(step.get("image_h") or 0) or 1.0
    action = step.get("action") or {}
    output = (action.get("action_output") or {}) if isinstance(action, Mapping) else {}
    name = str(output.get("action_name") or "").strip()
    params = output.get("action") or {}
    if not isinstance(params, Mapping):
        params = {}

    kind = _MOLMO_TYPE.get(name, "UNK")
    x = y = dx = dy = 0.0
    has_coord = has_delta = False
    payload = b""

    if "x" in params and "y" in params:
        x = _clip01(float(params["x"]) / width)
        y = _clip01(float(params["y"]) / height)
        has_coord = True
    if "delta_x" in params or "delta_y" in params:
        dx = float(params.get("delta_x") or 0.0) / width
        dy = float(params.get("delta_y") or 0.0) / height
        has_delta = True

    if kind == "GOTO":
        payload = _truncate_utf8(params.get("url") or "")
    elif kind == "TYPE":
        payload = _truncate_utf8(params.get("text") or "")
    elif kind == "KEY":
        payload = _truncate_utf8(params.get("key") or params.get("text") or ("go_back" if name == "go_back" else ""))
    elif kind == "TAB":
        payload = _truncate_utf8(str(params.get("index", "")) if name == "tab_focus" else "new")
    elif kind == "WAIT":
        payload = _truncate_utf8(str(params.get("wait_ms", "")))
    elif kind == "ANSWER":
        payload = _truncate_utf8(params.get("msg") or params.get("message") or "")

    if kind not in COORD_TYPES and kind != "SCROLL":
        has_coord, x, y = False, 0.0, 0.0
    if kind not in DELTA_TYPES:
        has_delta, dx, dy = False, 0.0, 0.0

    return WebAction(
        type_id=ACTION_TYPE_IDS[kind], x=x, y=y, dx=dx, dy=dy,
        payload=payload, has_coord=has_coord, has_delta=has_delta,
        source_id=SOURCE_IDS["molmoweb"],
    )


# --------------------------------------------------------------------------
# WorldMemArena web -- two syntaxes in the same corpus
# --------------------------------------------------------------------------

_WMA_TOOL_TYPE = {
    "left_click": "CLICK",
    "double_click": "DOUBLE_CLICK",
    "triple_click": "TRIPLE_CLICK",
    "right_click": "RIGHT_CLICK",
    "middle_click": "CLICK",
    "left_click_drag": "DRAG",
    "mouse_move": "MOVE",
    "type": "TYPE",
    "key": "KEY",
    "scroll": "SCROLL",
    "wait": "WAIT",
    "screenshot": "SCREENSHOT",
    "cursor_position": "SCREENSHOT",
}

_PYAUTOGUI_TYPE = {
    "click": "CLICK",
    "doubleClick": "DOUBLE_CLICK",
    "tripleClick": "TRIPLE_CLICK",
    "rightClick": "RIGHT_CLICK",
    "middleClick": "CLICK",
    "dragTo": "DRAG",
    "moveTo": "MOVE",
    "typewrite": "TYPE",
    "write": "TYPE",
    "press": "KEY",
    "hotkey": "KEY",
    "keyDown": "KEY",
    "keyUp": "KEY",
    "scroll": "SCROLL",
    "hscroll": "SCROLL",
    "sleep": "WAIT",
    "screenshot": "SCREENSHOT",
}

# ``Action: {...}`` blocks. Non-greedy from the first brace of an object that
# contains an "input" key, so trailing prose after the JSON does not swallow it.
_TOOL_JSON = re.compile(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', re.DOTALL)
_PY_CALL = re.compile(r'pyautogui\.(\w+)\s*\(([^)]*)\)')
# A scroll amount in the Anthropic tool schema is a click count, not pixels.
_SCROLL_CLICK_PIXELS = 100.0

# ...but the pyautogui-style traces do not all agree with that. Measured over
# the 105 WorldMemArena scroll actions, the first argument of `pyautogui.scroll`
# arrives in two incompatible units, and the corpus separates cleanly:
#
#     word_docs    34 actions   |amount|  10 .. 30        wheel clicks
#     image_edit   22 actions   |amount| 449 .. 1097      pixels
#     excel        15 actions   |amount| 491 .. 491       pixels
#     file_mgmt    34 actions   |amount| 1022 .. 1491     pixels
#
# Reading the large ones as clicks multiplies them by 100 and yields scrolls of
# up to 207 viewport heights, against MolmoWeb's p99 of 2.8 -- a 100x unit error
# in a channel MolmoWeb expresses as viewport fractions. Dividing the three
# desktop subcategories by 100 puts WorldMemArena back on MolmoWeb's scale
# (p99 2.750 against 2.814).
#
# The threshold is magnitude, not subcategory name, because the name is not
# available here and the gap is 15x wide with nothing in it. 50 clicks is
# 5,000 px -- about seven screens in a single action -- so anything above it was
# never a click count.
_SCROLL_CLICK_CEILING = 50.0

# --------------------------------------------------------------------------
# Two further syntaxes, neither of which names a pixel.
#
# ``webarena_lite`` speaks the WebArena action space (``click [3]``,
# ``type [43] [text] [0]``) and ``mobile`` speaks AppAgent's (``tap(4)``,
# ``swipe(1, up, medium)``). Both address an element by index, so they carry an
# action *type* but no point -- they arrive with ``has_coord=False`` and land in
# the coordinate channel's reserved "no point here" bin.
#
# Without these, 1,542 of the 4,558 encoded WorldMemArena states (33.8%) parse
# to zero actions and drop out of any cache built from them -- including all
# 1,073 of webarena_lite, which is the only web-domain data outside the
# evaluation split. ``css`` stays unparseable on purpose: its assistant turns
# are free-form reasoning with no action structure to recover.
# --------------------------------------------------------------------------

_WEBARENA_TYPE = {
    "click": "CLICK",
    "type": "TYPE",
    "hover": "MOVE",
    "scroll": "SCROLL",
    "go_back": "KEY",
    "go_forward": "KEY",
    "goto": "GOTO",
    "new_tab": "TAB",
    "tab_focus": "TAB",
    "close_tab": "TAB",
    "stop": "ANSWER",
    "press": "KEY",
}
_APPAGENT_TYPE = {
    "tap": "CLICK",
    "long_press": "RIGHT_CLICK",
    "text": "TYPE",
    "swipe": "SCROLL",
    "launch": "GOTO",
    "back": "KEY",
    "home": "KEY",
}
# ``click [3]``  /  ``type [43] [Primo Endurance Tank] [0]``  /  ``go_back``
_WEBARENA_CALL = re.compile(
    r'^\s*(\w+)((?:\s*\[[^\]]*\])*)\s*$', re.MULTILINE
)
# ``tap(4)``  /  ``swipe(1, up, medium)``  /  ``launch("bluecoins")``
_APPAGENT_CALL = re.compile(r'^\s*(\w+)\s*\(([^)]*)\)\s*$', re.MULTILINE)
_BRACKET = re.compile(r'\[([^\]]*)\]')


def _py_args(raw: str) -> tuple[list[object], dict[str, object]]:
    """Best-effort literal parse of a pyautogui call's arguments."""
    args: list[object] = []
    kwargs: dict[str, object] = {}
    try:
        node = ast.parse(f"f({raw})", mode="eval").body
    except SyntaxError:
        return args, kwargs
    for item in getattr(node, "args", ()):
        try:
            args.append(ast.literal_eval(item))
        except (ValueError, SyntaxError):
            args.append(None)
    for item in getattr(node, "keywords", ()):
        try:
            kwargs[item.arg] = ast.literal_eval(item.value)
        except (ValueError, SyntaxError):
            kwargs[item.arg] = None
    return args, kwargs


def parse_wma_actions(
    content: str, *, width: float = 1280.0, height: float = 720.0
) -> list[WebAction]:
    """Canonicalise every action in one WorldMemArena assistant turn.

    Returns a list because a turn may contain several ``pyautogui`` calls.  An
    unrecognised turn returns ``[]`` rather than an ``UNK`` action, so callers
    can distinguish "no action here" from "an action I could not read".
    """
    text = str(content or "")
    actions: list[WebAction] = []

    for match in _TOOL_JSON.finditer(text):
        try:
            payload_json = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        params = payload_json.get("input")
        if not isinstance(params, Mapping):
            continue
        name = str(params.get("action") or "")
        kind = _WMA_TOOL_TYPE.get(name)
        if kind is None:
            continue
        actions.append(_wma_tool_action(kind, params, width, height))

    if actions:
        return actions

    for name, raw in _PY_CALL.findall(text):
        kind = _PYAUTOGUI_TYPE.get(name)
        if kind is None:
            continue
        args, kwargs = _py_args(raw)
        actions.append(_wma_py_action(kind, name, args, kwargs, width, height))
    if actions:
        return actions

    return _parse_element_indexed(text)


def _parse_element_indexed(text: str) -> list[WebAction]:
    """WebArena and AppAgent syntax: an element index, never a pixel.

    Tried only after the two coordinate syntaxes have found nothing, because
    both patterns are anchored to a whole line and would otherwise match stray
    prose inside a computer-use turn's commentary.
    """
    actions: list[WebAction] = []
    for verb, brackets in _WEBARENA_CALL.findall(text):
        kind = _WEBARENA_TYPE.get(verb.lower())
        if kind is None:
            continue
        fields = _BRACKET.findall(brackets)
        # click [3] -> the ref; type [43] [text] [0] -> ref, then the text.
        payload = fields[1] if kind == "TYPE" and len(fields) > 1 else (
            fields[0] if fields else ""
        )
        actions.append(WebAction(
            type_id=ACTION_TYPE_IDS[kind],
            x=0.0, y=0.0, dx=0.0, dy=0.0,
            payload=payload.encode("utf-8")[:MAX_PAYLOAD_BYTES],
            has_coord=False, has_delta=False, source_id=SOURCE_IDS["wma"],
        ))
    if actions:
        return actions

    for verb, raw in _APPAGENT_CALL.findall(text):
        kind = _APPAGENT_TYPE.get(verb.lower())
        if kind is None:
            continue
        actions.append(WebAction(
            type_id=ACTION_TYPE_IDS[kind],
            x=0.0, y=0.0, dx=0.0, dy=0.0,
            payload=raw.strip().strip('"\'').encode("utf-8")[:MAX_PAYLOAD_BYTES],
            has_coord=False, has_delta=False, source_id=SOURCE_IDS["wma"],
        ))
    return actions


def _wma_tool_action(
    kind: str, params: Mapping[str, object], width: float, height: float
) -> WebAction:
    x = y = dx = dy = 0.0
    has_coord = has_delta = False
    payload = b""

    coordinate = params.get("coordinate")
    if isinstance(coordinate, Sequence) and not isinstance(coordinate, (str, bytes)) and len(coordinate) >= 2:
        x = _clip01(float(coordinate[0]) / width)
        y = _clip01(float(coordinate[1]) / height)
        has_coord = True

    if kind == "SCROLL":
        amount = float(params.get("scroll_amount") or 0.0) * _SCROLL_CLICK_PIXELS
        direction = str(params.get("scroll_direction") or "down").lower()
        if direction in ("up", "down"):
            dy = (-amount if direction == "up" else amount) / height
        else:
            dx = (-amount if direction == "left" else amount) / width
        has_delta = True
    elif kind in ("TYPE", "KEY"):
        payload = _truncate_utf8(params.get("text") or "")
    elif kind == "WAIT":
        payload = _truncate_utf8(str(params.get("duration", "")))

    if kind not in COORD_TYPES and kind != "SCROLL":
        has_coord, x, y = False, 0.0, 0.0
    return WebAction(
        type_id=ACTION_TYPE_IDS[kind], x=x, y=y, dx=dx, dy=dy,
        payload=payload, has_coord=has_coord, has_delta=has_delta,
        source_id=SOURCE_IDS["wma"],
    )


def _wma_py_action(
    kind: str, name: str, args: list[object], kwargs: dict[str, object],
    width: float, height: float,
) -> WebAction:
    x = y = dx = dy = 0.0
    has_coord = has_delta = False
    payload = b""

    def _num(index: int, key: str):
        if key in kwargs and isinstance(kwargs[key], (int, float)):
            return float(kwargs[key])
        if len(args) > index and isinstance(args[index], (int, float)):
            return float(args[index])
        return None

    if kind in COORD_TYPES:
        px, py = _num(0, "x"), _num(1, "y")
        if px is not None and py is not None:
            x, y = _clip01(px / width), _clip01(py / height)
            has_coord = True
    elif kind == "SCROLL":
        amount = _num(0, "clicks")
        if amount is not None:
            # pyautogui scrolls up for positive clicks; screen y grows downward.
            # See _SCROLL_CLICK_CEILING: the traces disagree about whether this
            # argument counts clicks or pixels, and the magnitude is the only
            # thing here that can tell them apart.
            pixels = amount if abs(amount) > _SCROLL_CLICK_CEILING else amount * _SCROLL_CLICK_PIXELS
            dy = -pixels / height
        px, py = _num(1, "x"), _num(2, "y")
        if px is not None and py is not None:
            x, y = _clip01(px / width), _clip01(py / height)
            has_coord = True
        has_delta = True

    if kind in ("TYPE", "KEY"):
        if name == "hotkey":
            payload = _truncate_utf8("+".join(str(a) for a in args if isinstance(a, str)))
        else:
            literal = next((a for a in args if isinstance(a, str)), "")
            payload = _truncate_utf8(literal)
    elif kind == "WAIT":
        seconds = _num(0, "secs")
        payload = _truncate_utf8("" if seconds is None else str(seconds))

    return WebAction(
        type_id=ACTION_TYPE_IDS[kind], x=x, y=y, dx=dx, dy=dy,
        payload=payload, has_coord=has_coord, has_delta=has_delta,
        source_id=SOURCE_IDS["wma"],
    )


# --------------------------------------------------------------------------
# Corpus shaping
# --------------------------------------------------------------------------

def merge_consecutive_scrolls(
    actions: Iterable[WebAction], *, same_axis_only: bool = True
) -> list[WebAction]:
    """Fold runs of scrolls into one action with the summed delta.

    Human demonstrations recorded through a browser extension emit one step per
    scroll *event*, so MolmoWeb is 40.6% scroll against WorldMemArena's 9.3%.
    Merging brings the action mixture toward the evaluation domain without
    discarding the transitions, which is what plain subsampling would do.

    ``same_axis_only`` keeps a vertical run separate from a horizontal one; a
    direction reversal is also a boundary, since a scroll down followed by a
    scroll up is a real observation-changing round trip, not one movement.
    """
    scroll = ACTION_TYPE_IDS["SCROLL"]
    merged: list[WebAction] = []
    for action in actions:
        if action.type_id != scroll or not merged or merged[-1].type_id != scroll:
            merged.append(action)
            continue
        previous = merged[-1]
        if same_axis_only:
            same_axis = (abs(previous.dx) > abs(previous.dy)) == (abs(action.dx) > abs(action.dy))
            same_sign = (previous.dx + previous.dy) * (action.dx + action.dy) > 0
            if not (same_axis and same_sign):
                merged.append(action)
                continue
        # Deliberately *not* clipped. A run of 11 downward scrolls on the Apple
        # homepage sums to 5.1 viewport heights; clamping it to 1.0 would make
        # "scrolled one screen" and "scrolled five screens" the same action and
        # hand the world model an unpredictable observation. Callers that need a
        # bounded value should compress with ``signed_log_delta`` rather than
        # truncate.
        merged[-1] = dataclasses.replace(
            previous,
            dx=float(previous.dx + action.dx),
            dy=float(previous.dy + action.dy),
            merged=previous.merged + action.merged,
        )
    return merged


def signed_log_delta(value: float) -> float:
    """Bounded, monotone, sign-preserving compression of a scroll delta.

    ``merge_consecutive_scrolls`` may return several viewport heights. This maps
    that onto a range an embedding can use without losing the ordering that
    distinguishes a one-screen scroll from a five-screen one.
    """
    return float(np.sign(value) * np.log1p(abs(value)))


def action_side_information_bits(
    action: WebAction, *, include_payload: bool, include_coordinate: bool = True,
    coordinate_bits: int = 10,
) -> int:
    """Fixed audit serialization, same shape as the v8 accounting.

    ``coordinate_bits`` per axis is the quantisation actually used downstream;
    10 bits is ~1/1024 of the viewport, finer than a click target needs and
    finer than either corpus records.  Coordinates are billed exactly like the
    payload: they are transmitted side information, not a free oracle.
    """
    total = 5  # action type, 17 values
    if include_coordinate and action.has_coord:
        total += 2 * coordinate_bits
    if include_coordinate and action.has_delta:
        total += 2 * coordinate_bits
    if include_payload:
        total += 6 + 8 * action.payload_length
    return total


PAD_ACTION = WebAction(type_id=ACTION_TYPE_IDS["PAD"])
MASK_ACTION = WebAction(type_id=ACTION_TYPE_IDS["MASK"])
