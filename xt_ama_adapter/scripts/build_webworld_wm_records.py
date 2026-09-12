"""Build WM records + state text index from a Qwen/WebWorldData slice.

Each WebWorld row is a real-a11y web trajectory stored as alternating
``human`` / ``gpt`` messages:

* ``human[2i]``      carries the *observed* page state ``S_i`` (the first
  message only, after ``Initial Page State:``) plus the action that leads to
  ``S_{i+1}`` (``First Action:`` for i=0, ``Action:`` afterwards).
* ``gpt[2i+1]``      is the *observed* page state ``S_{i+1}`` after that action.

So one row with ``g`` gpt messages yields ``g+1`` consecutive states and ``g``
single-action transitions, matching the adjacency contract of
``cache_web.build_transitions``.  Page states are browser accessibility trees
(``RootWebArea ... [id] tag 'name'``), i.e. the same body format AMA-Bench WEB
episodes are serialised in.

Actions are function-style element references, e.g. ``click('604')``; they are
canonicalised into ``schema_web`` ``WebAction`` records with ``has_coord=False``
and the element id as payload.

Outputs:
* ``records.jsonl``  cache_web-compatible records (one per state).
* ``states.jsonl``   unique ``{state_id, split, text}`` index for encode.
* ``report.json``    parse statistics (rows/episodes/transitions/actions).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from experiments.world_model.schema_web import (
    ACTION_TYPE_IDS,
    ACTION_TYPE_NAMES,
)

PROTOCOL = "webworld_wm_records_v1"
_ROW_OFFSET = 0
MAX_PAYLOAD_BYTES = 40
_HASH = hashlib.sha256

# WebWorld action verbs -> schema_web taxonomy. Anything unmapped is kept as
# UNK (reported), never silently dropped.
_WEBWORLD_TYPE = {
    "click": "CLICK",
    "double_click": "DOUBLE_CLICK",
    "right_click": "RIGHT_CLICK",
    "hover": "MOVE",
    "mouse_move": "MOVE",
    "move": "MOVE",
    "type": "TYPE",
    "keyboard_type": "TYPE",
    "press": "KEY",
    "key": "KEY",
    "keyboard_press": "KEY",
    "scroll": "SCROLL",
    "scroll_at": "SCROLL",
    "wait": "WAIT",
    "noop": "WAIT",
    "goto": "GOTO",
    "new_tab": "TAB",
    "tab_focus": "TAB",
    "screenshot": "SCREENSHOT",
    "answer": "ANSWER",
    "send_msg_to_user": "ANSWER",
}

_CALL_RE = re.compile(r"^([A-Za-z_]+)\s*\((.*)\)\s*$")


def _truncate_utf8(text: str) -> str:
    payload = str(text).encode("utf-8")
    if len(payload) <= MAX_PAYLOAD_BYTES:
        return text
    payload = payload[:MAX_PAYLOAD_BYTES]
    while payload and (payload[-1] & 0xC0) == 0x80:
        payload = payload[:-1]
    return payload.decode("utf-8", errors="replace")


def _parse_action(action_str: str) -> tuple[dict | None, str]:
    """Return (cache_web action dict, taxonomy kind) or (None, reason)."""
    match = _CALL_RE.match(action_str.strip())
    if not match:
        return None, "NO_CALL_SYNTAX"
    name = match.group(1).lower()
    raw = match.group(2).strip()
    payload = raw
    if len(payload) >= 2 and payload[0] == payload[-1] and payload[0] in "'\"":
        payload = payload[1:-1]
    kind = _WEBWORLD_TYPE.get(name)
    if kind is None:
        return None, f"UNKNOWN_{name}"
    return {
        "type_id": int(ACTION_TYPE_IDS[kind]),
        "x": 0.0, "y": 0.0, "dx": 0.0, "dy": 0.0,
        "payload": _truncate_utf8(payload),
        "has_coord": False,
        "has_delta": False,
        "source_id": 0,
        "merged": 1,
    }, kind


def _split_from_row(row: dict) -> str:
    digest = _HASH(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return "validation" if int(digest[:8], 16) % 10 == 9 else "train"


def _extract_action(human: str) -> str | None:
    """Action text between the trailing 'Action:' marker and 'Next Page State'.

    The action body itself contains nested single quotes (``click('487')``),
    so naive quote-matching cannot be used; the message carries exactly one
    action and always terminates with ``Next Page State``.
    """
    index = human.rfind("Action:")
    if index == -1:
        return None
    tail = human[index + len("Action:"):]
    if "Next Page State" in tail:
        tail = tail.split("Next Page State", 1)[0]
    return tail.strip().strip("'\"").strip()


def _parse_row(row_index: int, row: dict) -> tuple[list[dict], list[str]] | None:
    """Return (records, state texts) or None when the row is unusable."""
    messages = row.get("conversations")
    if not isinstance(messages, list) or not messages:
        return None
    human_values = [str(m["value"]) for m in messages if m.get("from") == "human"]
    gpt_values = [str(m["value"]) for m in messages if m.get("from") == "gpt"]
    if not gpt_values or len(human_values) != len(gpt_values):
        return None
    first = human_values[0]
    if "Initial Page State:" not in first or "First Action:" not in first:
        return None
    state0 = first.split("Initial Page State:", 1)[1].split("First Action:", 1)[0].strip()
    if not state0:
        return None

    texts = [state0]
    actions: list[str] = []
    for index, human in enumerate(human_values):
        action_raw = _extract_action(human)
        if action_raw is None:
            return None
        actions.append(action_raw)
        texts.append(gpt_values[index].strip())
    if len(texts) < 2:
        return None

    split = _split_from_row(row)
    episode_id = f"webworld:{row_index + _ROW_OFFSET:07d}"
    records: list[dict] = []
    for step, text in enumerate(texts):
        state_id = _HASH(text.encode("utf-8")).hexdigest()
        item = {
            "protocol": PROTOCOL,
            "trajectory_id": episode_id,
            "image_number": step,
            "state_id": state_id,
            "episode_id": episode_id,
            "step": step,
            "split": split,
            "task": "webworld",
        }
        if step < len(texts) - 1:
            action, kind = _parse_action(actions[step])
            if action is None:
                return None
            item["action"] = action
            item["action_name"] = kind
        records.append(item)
    return records, texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--row-offset", type=int, default=0,
                        help="added to each row's line index when naming episodes; "
                             "use distinct offsets when parsing byte-range chunks of "
                             "the same source file so episode ids never collide")
    parser.add_argument("--include-splits", nargs="+",
                        default=["train", "validation"])
    args = parser.parse_args()

    include = set(args.include_splits)
    global _ROW_OFFSET
    _ROW_OFFSET = int(args.row_offset)
    records: list[dict] = []
    state_meta: dict[str, tuple[str, str]] = {}   # state_id -> (split, text)
    counters: Counter[str] = Counter()
    action_types: Counter[str] = Counter()
    skipped_actions: Counter[str] = Counter()

    # errors="replace": byte-range chunks of the source file can begin inside a
    # multi-byte utf-8 character; the mangled first line then fails json.loads
    # and is counted as a partial tail line instead of crashing the run.
    with args.slice.open(encoding="utf-8", errors="replace") as handle:
        for row_index, line in enumerate(handle):
            if args.row_limit is not None and row_index >= args.row_limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counters["partial_tail_line"] += 1
                continue
            parsed = _parse_row(row_index, row)
            if parsed is None:
                counters["rows_skipped"] += 1
                continue
            row_records, texts = parsed
            split = row_records[0]["split"]
            if split not in include:
                counters["rows_excluded_split"] += 1
                continue
            counters["rows_used"] += 1
            counters["transitions"] += len(row_records) - 1
            for item, text in zip(row_records, texts):
                if "action" in item:
                    action_types[item.get("action_name", "?")] += 1
                state_meta.setdefault(item["state_id"], (split, text))
            records.extend(row_records)

    if not records:
        raise SystemExit("no usable rows parsed; check the slice path and format")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records.sort(key=lambda item: (item["episode_id"], int(item["step"])))
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with (args.output_dir / "states.jsonl").open("w", encoding="utf-8") as handle:
        for state_id, (split, text) in sorted(state_meta.items()):
            handle.write(json.dumps(
                {"state_id": state_id, "split": split, "text": text},
                ensure_ascii=False) + "\n")

    episode_ids = {item["episode_id"] for item in records}
    report = {
        "protocol": PROTOCOL,
        "slice": str(args.slice.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "include_splits": sorted(include),
        "rows_used": counters["rows_used"],
        "rows_skipped": counters["rows_skipped"],
        "rows_excluded_split": counters["rows_excluded_split"],
        "partial_tail_line": counters["partial_tail_line"],
        "episodes": len(episode_ids),
        "states": len(records),
        "unique_states": len(state_meta),
        "transitions": counters["transitions"],
        "action_type_histogram": dict(action_types.most_common()),
        "skipped_action_reasons": dict(skipped_actions),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
