"""Shared plumbing for the dev diagnostics: paths, joins, and stamped output."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.common import write_json

from ..cache import FrozenCache
from . import DEV_PROTOCOL

DEV_ROOT = Path("outputs/world_model/v8_dev")
FORMAL_ROOT = Path("outputs/world_model/v8")

# The working tree lives under a directory that an external process periodically
# rolls back to an older snapshot, deleting everything created since. Result
# JSONs are tens of kilobytes and are not recoverable once lost -- unlike code,
# which git can restore -- so every dev artifact is mirrored outside that tree
# as it is written. Set RESIDUALMEM_SNAPSHOT_DIR to relocate or "" to disable.
SNAPSHOT_ROOT = Path(
    os.environ.get(
        "RESIDUALMEM_SNAPSHOT_DIR",
        "/mnt/data/users/luzheng/workspace/ResidualMem_rescue_20260814/results",
    )
)

# statistics._align asserts pairing on exactly these four keys.
PAIR_KEYS = ("transition_indices", "task_ids", "episode_ids", "policies")


def dev_path(*parts: str) -> Path:
    """Resolve a path under the dev tree, refusing to touch the formal tree."""
    path = DEV_ROOT.joinpath(*parts)
    formal = FORMAL_ROOT.resolve()
    resolved = path.resolve()
    if resolved == formal or formal in resolved.parents:
        raise ValueError(f"{path} is inside the formal tree {FORMAL_ROOT}")
    return path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as handle:
        return {name: handle[name] for name in handle.files}


def assert_paired(reference: dict, candidate: dict, label: str) -> None:
    """Row-for-row identity on the join keys, mirroring statistics._align."""
    for key in PAIR_KEYS:
        if key not in reference or key not in candidate:
            raise KeyError(f"{label}: missing pairing key {key!r}")
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"{label}: {key} does not match the reference artifact")


def attach_steps(cache: FrozenCache, artifact: dict, label: str) -> np.ndarray:
    """Supply per-transition source steps.

    The baseline npz carries no ``steps`` array, so it has to come from the
    cache. Positional alignment alone is not trusted: when the artifact does
    carry ``steps`` the cache-derived values must agree exactly.
    """
    rows = np.asarray(artifact["transition_indices"], np.int64)
    steps = np.asarray(cache.transitions["steps"][rows])
    if "steps" in artifact and not np.array_equal(steps, np.asarray(artifact["steps"])):
        raise ValueError(f"{label}: cache steps disagree with the artifact's steps")
    return steps


def assert_validation_only(cache: FrozenCache, artifact: dict, label: str) -> None:
    rows = np.asarray(artifact["transition_indices"], np.int64)
    allowed = set(cache.indices_for_split("validation").tolist())
    if not set(rows.tolist()) <= allowed:
        raise ValueError(f"{label}: artifact contains non-validation transitions")


def snapshot(path: str | Path) -> Path | None:
    """Mirror one artifact outside the roll-backable tree.

    Returns the snapshot path, or None if snapshotting is disabled or failed.
    A failure here must never lose the primary write, so it is reported rather
    than raised.
    """
    path = Path(path)
    if not str(SNAPSHOT_ROOT) or not path.exists():
        return None
    try:
        relative = path.resolve().relative_to(DEV_ROOT.resolve())
    except ValueError:
        relative = Path(path.name)
    target = SNAPSHOT_ROOT / relative
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    except OSError as error:  # pragma: no cover - depends on the filesystem
        print(f"WARNING: could not snapshot {path} -> {target}: {error}")
        return None
    return target


def write_dev_json(path: str | Path, payload: dict) -> None:
    """Write a diagnostic report, stamped so it can never read as formal.

    The artifact is immediately mirrored outside the working tree; see
    ``SNAPSHOT_ROOT``.
    """
    stamped = {
        "protocol": DEV_PROTOCOL,
        "formal": False,
        "test_used": False,
        "interpretation_guard": (
            "diagnostic artifact; not a preregistered gate and not admissible "
            "to statistics.py"
        ),
        **payload,
    }
    path = Path(path)
    write_json(path, stamped)
    mirrored = snapshot(path)
    if mirrored is not None:
        print(f"snapshot: {mirrored}")
