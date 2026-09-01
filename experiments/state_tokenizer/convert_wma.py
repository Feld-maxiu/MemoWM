from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from experiments.state_tokenizer.common import write_json

from ..world_model.schema_web import parse_wma_actions

TRAIN_SUBCATEGORIES = (
    "excel", "file_mgmt", "image_edit", "mobile", "webarena_lite", "word_docs",
)
EVAL_SUBCATEGORY = "web"


def observations_and_actions(payload: dict) -> list[dict]:
    sample_id = payload["sample_id"]
    rows: list[dict] = []
    index = 0
    for session in payload.get("sessions", []):
        pending: dict | None = None
        for turn in session.get("dialogue", []):
            role = turn.get("role")
            if role == "user" and (turn.get("attachments") or []):
                if pending is not None:
                    rows.append(pending)
                attachments = turn["attachments"]
                pending = {
                    "state_id": f"{sample_id}-{index:04d}",
                    "session_id": session.get("session_id", ""),
                    "captions": [
                        str(a.get("caption") or "").strip()
                        for a in attachments
                        if str(a.get("caption") or "").strip()
                    ],
                    "image_ids": [str(a["image_id"]) for a in attachments
                                  if a.get("image_id")],
                    "turns": [],
                }
                index += 1
            elif role == "assistant" and pending is not None:
                pending["turns"].append(str(turn.get("content") or ""))
        if pending is not None:
            rows.append(pending)
    return rows


def convert_sample(path: Path, subcategory: str, split: str) -> tuple[list[dict], Counter]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = observations_and_actions(payload)
    counts: Counter[str] = Counter()

    parsed: list[tuple[dict, object | None, int]] = []
    for row in rows:
        actions = []
        for content in row["turns"]:
            try:
                actions.extend(parse_wma_actions(content))
            except Exception:                            # noqa: BLE001
                counts["parse_error"] += 1
        parsed.append((row, actions[0] if actions else None, len(row["turns"])))

    records: list[dict] = []
    segment = 0
    step = 0
    for position, (row, action, turn_count) in enumerate(parsed):
        last = position == len(parsed) - 1
        usable = last or (action is not None and turn_count == 1)
        if not last and action is None:
            counts["cut_no_action"] += 1
        elif not last and turn_count > 1:
            counts["cut_multi_action"] += 1
        base = f"{path.stem}/{row['session_id']}" if row["session_id"] else path.stem
        records.append({
            "state_id": row["state_id"],
            "episode_id": f"{base}#{segment}",
            "step": step,
            "task": "wma",
            "subcategory": subcategory,
            "split": split,
            "captions": row["captions"],
            "image_ids": row["image_ids"],
            "action": {
                "type_id": action.type_id if action is not None else 0,
                "x": float(action.x) if action is not None else 0.0,
                "y": float(action.y) if action is not None else 0.0,
                "dx": float(action.dx) if action is not None else 0.0,
                "dy": float(action.dy) if action is not None else 0.0,
                "payload": action.payload.decode("utf-8", "replace") if action is not None else "",
                "has_coord": bool(action.has_coord) if action is not None else False,
                "has_delta": bool(action.has_delta) if action is not None else False,
                "source_id": 1,
                "merged": 1,
            },
        })
        counts["states"] += 1
        if usable:
            step += 1
        else:
            segment += 1
            step = 0
    return records, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="the agent/gui directory")
    parser.add_argument("--subcategory", action="append", default=None,
                        help=f"repeatable; defaults to {list(TRAIN_SUBCATEGORIES)}")
    parser.add_argument("--split", default="train",
                        help="the split label written into every record")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    wanted = args.subcategory or list(TRAIN_SUBCATEGORIES)
    if EVAL_SUBCATEGORY in wanted and not args.subcategory:
        raise SystemExit("refusing to include the evaluation split by default")

    records: list[dict] = []
    per_subcategory: dict[str, dict] = {}
    for subcategory in wanted:
        directory = args.root / subcategory
        if not directory.is_dir():
            raise SystemExit(f"no such subcategory: {directory}")
        totals: Counter[str] = Counter()
        for path in sorted(directory.glob("*.json")):
            rows, counts = convert_sample(path, subcategory, args.split)
            records.extend(rows)
            totals.update(counts)
        per_subcategory[subcategory] = dict(totals)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    episodes = {r["episode_id"] for r in records}
    summary = {
        "records": len(records),
        "episodes": len(episodes),
        "transitions": len(records) - len(episodes),
        "subcategories": per_subcategory,
        "split": args.split,
    }
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
