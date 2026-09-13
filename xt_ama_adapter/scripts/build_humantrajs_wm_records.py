"""Build cache_web-compatible WM records from the humantrajs pseudo-web store.

Data contract (verified offline):

* ``<store>/*.npz`` holds one trajectory per file.  Its metadata ``records``
  list is the QA-filtered observation set for that trajectory; each record
  carries ``trajectory_id``, ``step_idx``, ``split`` and a ``screenshot`` whose
  basename is ``..._step_<m>.png``.
* ``<steps>.jsonl`` (``humantrajs-semantic-steps.jsonl``) indexes the *same*
  image files: jsonl step ``m`` has image ``..._step_<m>.png``.  Alignment is
  therefore done on the image file number ``m``, never on ``step_idx`` (the
  store's ``step_idx`` is one less than the jsonl/file step).
* Under the corpus convention an action whose result is image ``step_{m+1}``
  is stored at jsonl step ``m+1``, so the action leaving state ``m`` is the
  jsonl step ``m+1``.
* A usable WM transition only exists between *adjacent* kept images; a QA gap
  means intermediate actions are unknown.  Each maximal run of adjacent images
  inside one trajectory and one split becomes one WM episode (reindexed to
  step 0..N-1), matching ``cache_web.build_transitions``'s requirement that
  episode steps are exactly ``range(len(values))``.

Actions are canonicalised with ``experiments.world_model.schema_web``
(``parse_molmoweb_action``), which turns each step's pixel coordinate into a
viewport fraction using that step's own image size (resolutions vary per
step: 1280x778, 1280x807, ...).
"""
from __future__ import annotations

import argparse
import functools
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.world_model.schema_web import (
    WebAction,
    parse_molmoweb_action,
)

PROTOCOL = "humantrajs_wm_records_v1"
_STEP_FILE_RE = re.compile(r"_step_(\d+)\.png$")


@functools.lru_cache(maxsize=16384)
def _image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _step_number(screenshot: str) -> int | None:
    match = _STEP_FILE_RE.search(screenshot.replace("\\", "/"))
    return int(match.group(1)) if match else None


def _serialize_action(action: WebAction) -> dict:
    # ``parse_molmoweb_action`` truncates payloads to MAX_PAYLOAD_BYTES and can
    # leave an incomplete UTF-8 sequence at the cut; ``errors="replace"`` keeps
    # the serialisation lossless-deterministic for the cache round trip.
    return {
        "type_id": int(action.type_id),
        "x": float(action.x),
        "y": float(action.y),
        "dx": float(action.dx),
        "dy": float(action.dy),
        "payload": action.payload.decode("utf-8", errors="replace"),
        "has_coord": bool(action.has_coord),
        "has_delta": bool(action.has_delta),
        "source_id": int(action.source_id),
        "merged": int(action.merged),
    }


def _domain(row: dict) -> str:
    observation = row.get("observation") or {}
    url = str(observation.get("url") or "")
    if not url:
        urls = observation.get("open_pages_urls") or []
        url = str(urls[0]) if urls else ""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).netloc
        return host if host else "unknown"
    except Exception:
        return "unknown"


def _load_steps(path: Path) -> dict[str, dict[int, dict]]:
    table: dict[str, dict[int, dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            trajectory_id = str(row["trajectory_id"])
            table.setdefault(trajectory_id, {})[int(row["step_idx"])] = row
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-splits", nargs="+",
                        default=["train", "validation"])
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    steps = _load_steps(args.steps)
    include = set(args.include_splits)

    records: list[dict] = []
    episodes = 0
    states_used = 0
    state_total = 0
    skipped = Counter()
    type_histogram: Counter[str] = Counter()
    splits_used: Counter[str] = Counter()

    for npz_path in sorted(args.store.glob("*.npz")):
        with np.load(npz_path, allow_pickle=True) as data:
            metadata = json.loads(str(data["metadata"]))
        trajectory_id = str(metadata["sample_id"])
        trajectory_steps = steps.get(trajectory_id, {})

        # One entry per kept observation: (image file number, store record).
        entries: list[tuple[int, dict]] = []
        for record in metadata["records"]:
            number = _step_number(str(record.get("screenshot") or ""))
            if number is None:
                skipped["no_image_number"] += 1
                continue
            entries.append((number, record))
        entries.sort()
        state_total += len(entries)

        # Maximal runs of adjacent images, cut at split boundaries and at
        # missing jsonl steps (the leaving action would be unknown).
        runs: list[list[tuple[int, dict]]] = []
        for number, record in entries:
            split = str(record.get("split") or "")
            if split not in include:
                continue
            if not runs or number != runs[-1][-1][0] + 1 or split != runs[-1][-1][1].get("split"):
                runs.append([])
            runs[-1].append((number, record))
        for run in runs:
            if len(run) < 2:
                skipped["single_state_runs"] += 1
                continue
            if run[0][0] + len(run) - 1 != run[-1][0]:
                raise AssertionError("run is not contiguous")
            leaving = {number: trajectory_steps.get(number + 1) for number, _ in run}
            if any(row is None for row in leaving.values()):
                skipped["missing_leaving_action"] += 1
                continue
            episode_id = f"{trajectory_id}:run{episodes}"
            run_split = str(run[0][1].get("split") or "")
            for index, (number, record) in enumerate(run):
                row = trajectory_steps[number]
                state_id = str(record["image_ids"][0]) if record.get("image_ids") else f"{trajectory_id}:{number:04d}"
                item = {
                    "protocol": PROTOCOL,
                    "trajectory_id": trajectory_id,
                    "image_number": number,
                    "state_id": state_id,
                    "episode_id": episode_id,
                    "step": index,
                    "split": run_split,
                    "task": _domain(row),
                }
                if index < len(run) - 1:
                    action_row = leaving[number]
                    action_row = dict(action_row)
                    action_row["image_w"], action_row["image_h"] = _image_size(
                        str(action_row["image_path"])
                    )
                    action = parse_molmoweb_action(action_row)
                    type_histogram[action.type_name] += 1
                    item["action"] = _serialize_action(action)
                records.append(item)
            episodes += 1
            states_used += len(run)
            splits_used[run_split] += 1

    records.sort(key=lambda item: (item["episode_id"], int(item["step"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    per_episode = Counter(item["episode_id"] for item in records)
    transitions = sum(count - 1 for count in per_episode.values())
    report = {
        "protocol": PROTOCOL,
        "store": str(args.store.resolve()),
        "steps": str(args.steps.resolve()),
        "output": str(args.output.resolve()),
        "include_splits": sorted(include),
        "store_states": state_total,
        "wm_states": states_used,
        "episodes": episodes,
        "transitions": transitions,
        "episodes_by_split": dict(splits_used),
        "action_type_histogram": dict(type_histogram.most_common()),
        "skipped": dict(skipped),
        "task_examples": sorted({item["task"] for item in records})[:10],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
