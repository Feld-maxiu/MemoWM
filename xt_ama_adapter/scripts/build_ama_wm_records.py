"""Build WM records + state text index from AMA-Bench trajectories.

Each AMA-Bench row is one episode with a ``trajectory`` of turns
``{turn_idx, action, observation}``.  Consecutive turns chain into
state-action transitions

    (observation_t, action_{t+1}, observation_{t+1})   for t = 0..n-2

which matches the adjacency contract of ``cache_web.build_transitions``.
Turn 0's own action has no preceding observation and is dropped (counted in
the report).  Under either reading of the corpus (observation before or after
its turn's action) the chain is well defined; only the interpretation of
``observation_0`` shifts.

Actions are free text spanning twelve heterogeneous task families (arrow keys,
grid swaps, shell commands, webarena bracket style, gaia tool calls, ...).
They are canonicalised into ``schema_web`` ``WebAction`` records with
``has_coord=False``; the rule table lives in ``_map_action`` and anything
unmapped falls through to TYPE/UNK (reported), never silently dropped.

All records carry ``task="ama"``: the discrete WM bills task naming per
episode, and a single-task eval cache keeps the model's task embedding
(adapted from the single-task webworld cache) consistent.  The per-episode
``task_type`` is preserved on each record for later slicing.

Outputs (same contract as ``build_webworld_wm_records``):
* ``records.jsonl``  cache_web-compatible records (one per state).
* ``states.jsonl``   unique ``{state_id, split, text}`` index for encode.
* ``report.json``    parse statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from experiments.world_model.schema_web import ACTION_TYPE_IDS

PROTOCOL = "ama_wm_records_v1"
MAX_PAYLOAD_BYTES = 40
_HASH = hashlib.sha256
EMPTY_STATE_TEXT = "(empty observation)"

_CALL_RE = re.compile(r"^([A-Za-z_]\w*)\s*\((.*)\)\s*$")

# Web verbs -> schema_web taxonomy (shared with build_webworld_wm_records).
_WEB_TYPE = {
    "click": "CLICK", "double_click": "DOUBLE_CLICK",
    "middle_click": "CLICK", "right_click": "RIGHT_CLICK",
    "hover": "MOVE", "mouse_move": "MOVE", "move": "MOVE",
    "type": "TYPE", "keyboard_type": "TYPE",
    "press": "KEY", "key": "KEY", "keyboard_press": "KEY",
    "scroll": "SCROLL", "scroll_at": "SCROLL",
    "goto": "GOTO", "go_to": "GOTO", "navigate": "GOTO",
    "new_tab": "TAB", "tab_focus": "TAB", "tab_switch": "TAB",
    "screenshot": "SCREENSHOT",
    "answer": "ANSWER", "send_msg_to_user": "ANSWER",
}
_KEYS = {
    "left", "right", "up", "down", "w", "a", "s", "d", "space", "enter",
    "backspace", "delete", "home", "end", "pageup", "pagedown", "esc",
    "escape", "tab", "insert",
    # NetHack / minihack compass moves.
    "north", "south", "east", "west",
    "northeast", "northwest", "southeast", "southwest",
}
_WAIT_TOKENS = {"idle", "noop", "no_op", "none", "null", "sleep", "wait", ""}
_ANSWER_TOKENS = {"stop", "done", "finish", "terminate", "submit"}
_BOOKKEEPING_CALLS = {"mark_step", "noop", "wait", "sleep"}


def _truncate_utf8(text: str) -> str:
    payload = str(text).encode("utf-8")
    if len(payload) <= MAX_PAYLOAD_BYTES:
        return text
    payload = payload[:MAX_PAYLOAD_BYTES]
    while payload and (payload[-1] & 0xC0) == 0x80:
        payload = payload[:-1]
    return payload.decode("utf-8", errors="replace")


def _action(kind: str, payload: str) -> tuple[dict, str]:
    return {
        "type_id": int(ACTION_TYPE_IDS[kind]),
        "x": 0.0, "y": 0.0, "dx": 0.0, "dy": 0.0,
        "payload": _truncate_utf8(payload),
        "has_coord": False,
        "has_delta": False,
        "source_id": 1,
        "merged": 1,
    }, kind


def _strip_quoted(raw: str) -> str:
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1]
    return raw.strip()


def _bracket_body(rest: str) -> str:
    """Payload of webarena-style ``verb [arg] more`` / ``verb (arg)`` forms."""
    rest = rest.strip()
    match = re.match(r"^[\[\(](.*?)[\]\)](?:\s+(.*))?$", rest)
    if match:
        head, tail = match.group(1).strip(), (match.group(2) or "").strip()
        return f"{head} {tail}".strip()
    return rest


def _map_action(action_str: str) -> tuple[dict | None, str]:
    """Return (cache_web action dict, taxonomy kind) or (None, reason)."""
    text = str(action_str).strip()
    low = text.lower()
    tokens = text.split()

    if low in _WAIT_TOKENS:
        return _action("WAIT", "")
    if len(tokens) == 1:
        first = tokens[0].strip(",.;:").lower()
        if first in _WAIT_TOKENS:
            return _action("WAIT", "")
        if first in _ANSWER_TOKENS:
            return _action("ANSWER", text)
        if first in _KEYS:
            return _action("KEY", text)
        if first == "look":
            return _action("SCREENSHOT", text)

    call = _CALL_RE.match(text)
    if call:
        name = call.group(1).lower()
        raw = _strip_quoted(call.group(2))
        kind = _WEB_TYPE.get(name)
        if kind is not None:
            return _action(kind, raw)
        if name in _BOOKKEEPING_CALLS:
            return _action("WAIT", raw)
        # gaia tool calls, execute_code, and any other function-style verb:
        # the model still sees the call head through the payload channel.
        return _action("TYPE", text)

    if ":" in text:
        head, _, rest = text.partition(":")
        name = head.strip().lower()
        if name in _ANSWER_TOKENS:
            return _action("ANSWER", rest)
        if name in {"think", "reason"}:
            return _action("WAIT", rest)
        if name in _WEB_TYPE:
            return _action(_WEB_TYPE[name], rest)
        # swebench / spider2 shells and editors: execute_bash: ..., SQL, ...
        return _action("TYPE", rest if rest.strip() else text)

    verb = tokens[0].strip(",.;:").lower()
    rest = text[len(tokens[0]):].strip()
    if verb in _ANSWER_TOKENS:
        return _action("ANSWER", rest or text)
    if verb in _WEB_TYPE:
        kind = _WEB_TYPE[verb]
        payload = _bracket_body(rest) if kind in {
            "CLICK", "DOUBLE_CLICK", "RIGHT_CLICK", "MOVE",
            "TYPE", "KEY", "SCROLL", "GOTO", "TAB",
        } else rest
        return _action(kind, payload)
    # alfworld / crafter free-text commands ("go to shelf 1", "Move left").
    return _action("TYPE", text)


def _split_from_row(row: dict) -> str:
    digest = _HASH(json.dumps(row, ensure_ascii=False, sort_keys=True)
                   .encode("utf-8")).hexdigest()
    return "validation" if int(digest[:8], 16) % 10 == 9 else "train"


def _parse_row(row: dict) -> tuple[list[dict], list[str]] | None:
    """Return (records, state texts) or None when the episode is unusable."""
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        return None
    turns = []
    for turn in trajectory:
        if not isinstance(turn, dict):
            return None
        text = str(turn.get("observation", "") or "").strip()
        turns.append((str(turn.get("action", "") or ""), text or EMPTY_STATE_TEXT))

    split = _split_from_row(row)
    episode_id = f"ama:{int(row['episode_id']):06d}"
    task_type = str(row.get("task_type", "unknown"))
    records: list[dict] = []
    for step, (_action_text, state_text) in enumerate(turns):
        state_id = _HASH(state_text.encode("utf-8")).hexdigest()
        item = {
            "protocol": PROTOCOL,
            "trajectory_id": episode_id,
            "image_number": step,
            "state_id": state_id,
            "episode_id": episode_id,
            "step": step,
            "split": split,
            "task": "ama",
            "task_type": task_type,
        }
        if step < len(turns) - 1:
            action, kind = _map_action(turns[step + 1][0])
            if action is None:
                return None
            item["action"] = action
            item["action_name"] = kind
        records.append(item)
    return records, [text for _action_text, text in turns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-splits", nargs="+",
                        default=["train", "validation"])
    args = parser.parse_args()

    include = set(args.include_splits)
    records: list[dict] = []
    state_meta: dict[str, tuple[str, str]] = {}   # state_id -> (split, text)
    counters: Counter[str] = Counter()
    action_types: Counter[str] = Counter()
    per_task: Counter[str] = Counter()

    with args.dataset.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counters["episodes_seen"] += 1
            parsed = _parse_row(row)
            if parsed is None:
                counters["episodes_skipped"] += 1
                continue
            row_records, texts = parsed
            split = row_records[0]["split"]
            if split not in include:
                counters["episodes_excluded_split"] += 1
                continue
            counters["episodes_used"] += 1
            counters["turns"] += len(row_records)
            # Turn 0's action has no preceding observation: dropped by design.
            counters["first_actions_dropped"] += 1
            counters["transitions"] += len(row_records) - 1
            per_task[row_records[0]["task_type"]] += 1
            for item, text in zip(row_records, texts):
                if "action" in item:
                    action_types[item.get("action_name", "?")] += 1
                state_meta.setdefault(item["state_id"], (split, text))
            records.extend(row_records)

    if not records:
        raise SystemExit("no usable episodes parsed; check the dataset path")

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

    report = {
        "protocol": PROTOCOL,
        "dataset": str(args.dataset.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "include_splits": sorted(include),
        "episodes_seen": counters["episodes_seen"],
        "episodes_used": counters["episodes_used"],
        "episodes_skipped": counters["episodes_skipped"],
        "episodes_excluded_split": counters["episodes_excluded_split"],
        "states": len(records),
        "unique_states": len(state_meta),
        "transitions": counters["transitions"],
        "first_actions_dropped": counters["first_actions_dropped"],
        "episodes_by_task_type": dict(per_task.most_common()),
        "action_type_histogram": dict(action_types.most_common()),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
