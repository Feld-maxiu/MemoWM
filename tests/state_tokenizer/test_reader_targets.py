"""The reader trainer's target-text resolution.

Targets used to come only from a ``global_index`` lookup into the BrowserGym
jsonl. WorldMemArena rows have no index into that file and a merged cache fills
theirs with ``-1``, so any cross-domain corpus raised before a single step ran --
which is why the reader has never been trained on anything but BrowserGym.
The cache now carries the text itself, and these lock that it survives the
merge, in both domains, with each side's own format.

Numpy-only; no model weights and no torch are needed to check the plumbing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.merge_bridge_caches import TARGET_TEXT, load

BRIDGE_CACHE_PROTOCOL = "qwen35_instruct_bridge_cache_v2"
FUSED_OBSERVATION_TEACHER_PROTOCOL = "qwen3_vl_fused_observation_v1"


def _write(path: Path, rows: int, *, target_text=None, global_indices=None):
    arrays = {
        "xbar": np.zeros((rows, 64, 512), np.float32),
        "valid": np.ones((rows, 64), bool),
        "teacher_fused_embedding": np.tile(
            np.eye(1, 4096, dtype=np.float32), (rows, 1)
        ),
        "split": np.asarray(["train"] * rows),
        "metadata": np.asarray(json.dumps({
            "protocol": BRIDGE_CACHE_PROTOCOL,
            "teacher_protocol": FUSED_OBSERVATION_TEACHER_PROTOCOL,
            "representation": "xbar",
        })),
    }
    if target_text is not None:
        arrays[TARGET_TEXT] = np.asarray(target_text)
    if global_indices is not None:
        arrays["global_indices"] = np.asarray(global_indices, np.int64)
    np.savez(path, **arrays)
    return path


def test_a_cache_without_the_column_reads_as_blank(tmp_path):
    """Older caches must load rather than raise -- the merger fills them."""
    path = _write(tmp_path / "old.npz", 3, global_indices=[7, 8, 9])
    arrays, _ = load(path)
    assert list(arrays[TARGET_TEXT]) == ["", "", ""]
    assert list(arrays["global_indices"]) == [7, 8, 9]


def test_a_cache_with_the_column_keeps_it(tmp_path):
    path = _write(tmp_path / "new.npz", 2, target_text=["alpha", "beta"])
    arrays, _ = load(path)
    assert list(arrays[TARGET_TEXT]) == ["alpha", "beta"]
    # Rows with no index at all get the -1 sentinel, which is what makes the
    # global_index path unusable for them.
    assert list(arrays["global_indices"]) == [-1, -1]


def test_the_real_union_cache_has_text_for_both_domains():
    """Skips when the artifact is absent; asserts hard when it is there."""
    path = Path(
        "outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/"
        "union-cache-fused-observation.npz"
    )
    if not path.is_file():
        return
    with np.load(path, allow_pickle=True) as cache:
        assert TARGET_TEXT in cache.files, "the union cache predates the reader work"
        text = np.asarray(cache[TARGET_TEXT]).astype(str)
        domain = np.asarray(cache["domain"]).astype(str)
    assert (text != "").all(), f"{int((text == '').sum())} rows would reconstruct nothing"
    for name in ("browsergym", "wma_nonweb"):
        rows = text[domain == name]
        assert len(rows), f"{name} contributed no rows"
        assert (rows != "").all()
    # Each domain keeps its own serialization; they are matched in role, not
    # byte-for-byte -- BrowserGym renders an instruction plus AXTree, WMA renders
    # the observation text and captions the teacher embedded.
    assert any(row.startswith("Task instruction:") for row in text[domain == "browsergym"])
    assert any(
        row.startswith(("Current user observation:", "Image captions:"))
        for row in text[domain == "wma_nonweb"]
    )
