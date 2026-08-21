from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from experiments.state_tokenizer.assemble_browsergym_lanes import (
    _image_difference,
    _records_equivalent,
    assemble,
)
from experiments.state_tokenizer.common import PILOT_TASKS, split_for_episode


def _row(task_index: int, episode: int, step: int = 0) -> dict:
    episode_id = f"task{task_index:02d}-ep{episode:06d}"
    return {
        "state_id": f"{episode_id}-t{step:02d}",
        "task": PILOT_TASKS[task_index],
        "episode_id": episode_id,
        "episode_index": episode,
        "step": step,
        "split": split_for_episode(episode),
        "environment_seed": task_index * 1_000_000 + episode,
        "screenshot": f"images/{episode_id}-t{step:02d}.png",
    }


def _args(input_dir, output):
    return SimpleNamespace(
        input_dir=str(input_dir), output=str(output), pattern="records-lane-*.jsonl",
        target_states=12, original_stride=12, seed=0, reference_dir=None,
        verify_overlap_images=False, max_overlap_image_changed_fraction=0.0,
        max_overlap_image_mae=0.0,
    )


def test_assembler_selects_one_canonical_state_per_task(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "images").mkdir(parents=True)
    for task_index in range(12):
        row = _row(task_index, task_index)
        (source / row["screenshot"]).write_bytes(b"png")
        (source / f"records-lane-{task_index:02d}.jsonl").write_text(
            json.dumps(row) + "\n"
        )
    result = assemble(_args(source, output))
    assert result["states_selected"] == 12
    assert len(list(output.glob("records-worker*.jsonl"))) == 12


def test_assembler_rejects_a_gap_in_the_original_episode_lattice(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "images").mkdir(parents=True)
    for task_index in range(12):
        episode = task_index + (24 if task_index == 0 else 0)
        row = _row(task_index, episode)
        (source / row["screenshot"]).write_bytes(b"png")
        (source / f"records-lane-{task_index:02d}.jsonl").write_text(
            json.dumps(row) + "\n"
        )
    with pytest.raises(ValueError, match="episode gap"):
        assemble(_args(source, output))


def test_overlap_comparison_canonicalizes_only_cdp_node_ids():
    left = {
        "state_id": "same",
        "dom": "same semantic serialization",
        "axtree_raw": {"nodes": [
            {"nodeId": "2", "childIds": ["9"], "role": {"value": "root"}},
            {"nodeId": "9", "parentId": "2", "name": {"value": "Submit"}},
        ]},
    }
    right = {
        "state_id": "same",
        "dom": "same semantic serialization",
        "axtree_raw": {"nodes": [
            {"nodeId": "31", "childIds": ["47"], "role": {"value": "root"}},
            {"nodeId": "47", "parentId": "31", "name": {"value": "Submit"}},
        ]},
    }
    assert _records_equivalent(left, right)
    right["axtree_raw"]["nodes"][1]["name"]["value"] = "Cancel"
    assert not _records_equivalent(left, right)


def test_image_difference_counts_pixels_and_channel_mae(tmp_path):
    left = np.zeros((2, 2, 3), dtype=np.uint8)
    right = left.copy()
    right[0, 0] = (3, 6, 9)
    left_path, right_path = tmp_path / "left.png", tmp_path / "right.png"
    Image.fromarray(left).save(left_path)
    Image.fromarray(right).save(right_path)
    changed, mae = _image_difference(left_path, right_path)
    assert changed == pytest.approx(0.25)
    assert mae == pytest.approx(18 / 12)
