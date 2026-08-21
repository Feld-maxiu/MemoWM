"""Resume semantics for the BrowserGym collector.

A hundred-thousand-state collection runs for hours, and the process owning it can
disappear without warning -- that is how the first v8 attempt ended at 25k, with
every worker killed at once and no manifest written. Resume therefore has to be
correct on the first try, because the thing it protects is the data already on
disk: a wrong ``mode`` truncates the shard it was meant to save.

Two properties matter and neither is visible from a successful run:

* the surviving records are preserved exactly, and
* the last episode of each task is *dropped*, because episodes are flushed as a
  unit and a kill mid-write leaves a short one that would otherwise look complete.
"""
from __future__ import annotations

import json
from pathlib import Path

from experiments.state_tokenizer.collect_browsergym import _resume_state

NUM_WORKERS = 12
WORKER = 0


def record(task: str, episode: int, step: int) -> dict:
    return {
        "state_id": f"{task}-ep{episode:06d}-t{step:02d}",
        "task": task,
        "episode_id": f"{task}-ep{episode:06d}",
        "episode_index": episode,
        "step": step,
        "split": "train",
    }


def write(path: Path, records, trailing_garbage: str = "") -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item) + "\n")
        if trailing_garbage:
            handle.write(trailing_garbage)


def test_missing_shard_starts_from_scratch(tmp_path):
    progress, surviving = _resume_state(tmp_path / "absent.jsonl", NUM_WORKERS)
    assert progress == {}
    assert surviving == []


def test_last_episode_is_dropped_and_redone(tmp_path):
    """Episode 12 may have been cut short mid-flush, so it is not trusted."""
    path = tmp_path / "records-worker00.jsonl"
    write(path, [
        record("miniwob/click-button-v1", 0, 0),
        record("miniwob/click-button-v1", 0, 1),
        record("miniwob/click-button-v1", 12, 0),
    ])
    progress, surviving = _resume_state(path, NUM_WORKERS)
    assert [item["episode_index"] for item in surviving] == [0, 0]
    entry = progress["miniwob/click-button-v1"]
    assert entry["states"] == 2
    assert entry["next_episode"] == 12, "the dropped episode must be collected again"


def test_next_episode_keeps_the_worker_stride(tmp_path):
    """Episode indices are worker_id + k * num_workers; resume must stay on that lattice."""
    path = tmp_path / "records-worker00.jsonl"
    write(path, [
        record("miniwob/click-button-v1", 0, 0),
        record("miniwob/click-button-v1", 12, 0),
        record("miniwob/click-button-v1", 24, 0),
    ])
    progress, _ = _resume_state(path, NUM_WORKERS)
    next_index = progress["miniwob/click-button-v1"]["next_episode"]
    assert next_index == 24
    assert next_index % NUM_WORKERS == WORKER


def test_lane_resume_keeps_custom_stride_and_counts_complete_episodes(tmp_path):
    path = tmp_path / "records-lane.jsonl"
    write(path, [
        record("miniwob/click-button-v1", 0, 0),
        record("miniwob/click-button-v1", 0, 1),
        record("miniwob/click-button-v1", 96, 0),
        record("miniwob/click-button-v1", 192, 0),
    ])
    progress, surviving = _resume_state(path, 96)
    assert [item["episode_index"] for item in surviving] == [0, 0, 96]
    entry = progress["miniwob/click-button-v1"]
    assert entry["states"] == 3
    assert entry["episodes"] == 2
    assert entry["next_episode"] == 192


def test_progress_is_tracked_per_task(tmp_path):
    """A worker walks its tasks in order; each carries its own count and cursor."""
    path = tmp_path / "records-worker00.jsonl"
    write(path, [
        record("miniwob/click-button-v1", 0, 0),
        record("miniwob/click-button-v1", 12, 0),
        record("miniwob/click-checkboxes-v1", 0, 0),
        record("miniwob/click-checkboxes-v1", 0, 1),
    ])
    progress, surviving = _resume_state(path, NUM_WORKERS)
    assert progress["miniwob/click-button-v1"]["states"] == 1
    assert progress["miniwob/click-button-v1"]["next_episode"] == 12
    # the second task has only one episode, so everything in it is re-collected
    assert progress["miniwob/click-checkboxes-v1"]["states"] == 0
    assert progress["miniwob/click-checkboxes-v1"]["next_episode"] == 0
    assert all(item["task"] == "miniwob/click-button-v1" for item in surviving)


def test_truncated_final_line_is_ignored(tmp_path):
    """A kill can land mid-line; that line must not abort the resume."""
    path = tmp_path / "records-worker00.jsonl"
    write(path, [
        record("miniwob/click-button-v1", 0, 0),
        record("miniwob/click-button-v1", 12, 0),
    ], trailing_garbage='{"state_id": "miniwob/click-but')
    progress, surviving = _resume_state(path, NUM_WORKERS)
    assert [item["episode_index"] for item in surviving] == [0]
    assert progress["miniwob/click-button-v1"]["next_episode"] == 12
