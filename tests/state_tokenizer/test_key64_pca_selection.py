from __future__ import annotations

import pytest

from experiments.state_tokenizer.key64_pca import select_fit_indices


def _records():
    rows = []
    index = 0
    for task, count in (("task-c", 5), ("task-a", 5), ("task-b", 5)):
        for _ in range(count):
            rows.append({"global_index": index, "split": "train", "task": task})
            index += 1
    rows.append({"global_index": index, "split": "validation", "task": "task-a"})
    return rows


def test_first_selection_preserves_legacy_manifest_order():
    indices, counts = select_fit_indices(_records(), 4, "first")
    assert indices == [0, 1, 2, 3]
    assert counts == {"task-c": 4}


def test_task_balanced_selection_is_deterministic_and_nearly_equal():
    indices, counts = select_fit_indices(_records(), 10, "task_balanced")
    assert indices == [5, 10, 0, 6, 11, 1, 7, 12, 2, 8]
    assert counts == {"task-a": 4, "task-b": 3, "task-c": 3}
    again, again_counts = select_fit_indices(_records(), 10, "task_balanced")
    assert again == indices
    assert again_counts == counts


def test_selection_rejects_insufficient_or_duplicate_train_rows():
    with pytest.raises(ValueError, match="requested 20"):
        select_fit_indices(_records(), 20, "task_balanced")
    duplicate = _records() + [
        {"global_index": 0, "split": "train", "task": "task-z"}
    ]
    with pytest.raises(ValueError, match="duplicate train global index"):
        select_fit_indices(duplicate, 4, "task_balanced")
