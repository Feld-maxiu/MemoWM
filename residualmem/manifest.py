"""The one place that says which artifact is *the* artifact.

Six components were built separately and each left a directory of near-identical
files behind -- ten Q-Former arms, six retrieval heads, three connectors, and two
codebooks that share a filename. Which of them constitutes the system was
recorded only in prose, in a Chinese handbook, in a section header. Picking the
wrong one does not raise; it returns a slightly different number.

This module makes that choice machine-checked. ``configs/system.lock.yaml`` names
one file per role together with its sha256, and :func:`resolve` refuses to hand
back a path whose bytes disagree.

The codebook is why this exists rather than a naming convention. There are two
fits of ``opq-shared-mix10-M32-C64.npz``, one under ``pq/`` and one under
``pq-full/``, differing by an average of 0.40 per centroid. Only the ``pq-full/``
fit is bit-identical to the decoder embedded in the world model's training data.
Substituting the other one produces in-range ``(32, 32) uint8`` codes, a cache
that builds, and a model that scores it -- with roughly 972 bits of NLL error and
nothing anywhere raising. Deleting the wrong file is not a fix either: the
``pq/`` tree also holds the C=16 codebook that the falsifiability test in
``UTILITY_GATE.md`` §4.4 needs. A hash check is the only thing that separates
them.

Because that failure mode is *specific*, so is the error. The lock records the
hashes of known-wrong candidates under ``known_wrong``; when the bytes on disk
match one of those, :class:`ArtifactMismatch` says which fit you picked up and
which one you wanted, instead of printing two hex strings and leaving you to
guess.

Deliberately importable from both environments -- stdlib plus ``yaml``, no torch,
no jax, no numpy -- because the orchestrator has to verify the same lock from a
jax subprocess and a torch subprocess.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCK = REPO_ROOT / "configs" / "system.lock.yaml"


class ArtifactMismatch(RuntimeError):
    """A locked artifact is missing, or its bytes are not the locked bytes."""


def sha256_file(path: Path, *, chunk: int = 8 << 20) -> str:
    """Hash a file. Every time, with no memoisation.

    An earlier version cached on ``(path, size, mtime_ns)``, which is the usual
    build-system heuristic and is wrong here. Measured on this filesystem: two
    consecutive writes to the same path receive *identical* ``st_mtime_ns``
    (1788340007351385140 both times), so a same-length in-place rewrite -- what a
    rerun of a training script does to a checkpoint -- keeps the cache key and
    the stale hash is returned. A cache that can vouch for bytes it has not read
    defeats the only thing this module is for.

    The saving was not worth it in any case: this filesystem hashes at roughly
    600 MB/s, so verifying the whole lock costs about three seconds against a
    pipeline measured in hours.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_lock(path: Path | str | None = None) -> dict[str, Any]:
    lock_path = Path(path) if path else DEFAULT_LOCK
    if not lock_path.exists():
        raise ArtifactMismatch(f"no lock file at {lock_path}")
    with lock_path.open(encoding="utf-8") as handle:
        lock = yaml.safe_load(handle)
    lock["_path"] = str(lock_path)
    return lock


def _expand(template: str, lock: dict) -> Path:
    """Substitute ``${DATA}`` and ``${REPO}``.

    ``DATA`` lives outside the repository and is mounted at three interchangeable
    paths on this machine, so it is a variable rather than a checked-in constant.
    ``RESIDUALMEM_DATA`` overrides it for a machine that mounts it elsewhere.
    """
    data = os.environ.get("RESIDUALMEM_DATA") or lock.get("data_root", "")
    expanded = template.replace("${DATA}", data).replace("${REPO}", str(REPO_ROOT))
    candidate = Path(expanded)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def resolve(role: str, *, lock: dict | None = None, verify: bool = True) -> Path:
    """Return the locked path for ``role``, refusing to return the wrong bytes.

    ``verify=False`` exists for callers that only need the path (printing a
    provenance record, say). It is never the right choice before a measurement.
    """
    lock = lock if lock is not None else load_lock()
    artifacts = lock.get("artifacts") or {}
    if role not in artifacts:
        raise ArtifactMismatch(
            f"role {role!r} is not in {lock.get('_path')}; "
            f"known roles: {', '.join(sorted(artifacts))}")
    entry = artifacts[role]
    path = _expand(str(entry["path"]), lock)

    if not path.exists():
        note = entry.get("produced_by")
        hint = f" -- it is produced by {note}" if note else ""
        raise ArtifactMismatch(f"{role}: {path} does not exist{hint}")
    if not verify or entry.get("sha256") in (None, "", "null"):
        return path

    actual = sha256_file(path)
    expected = str(entry["sha256"])
    if actual == expected:
        return path

    for wrong in entry.get("known_wrong") or []:
        if actual == str(wrong.get("sha256")):
            raise ArtifactMismatch(
                f"{role}: {path}\n"
                f"  these bytes are a KNOWN-WRONG artifact: {wrong.get('what')}\n"
                f"  {wrong.get('why_it_matters', '')}\n"
                f"  expected sha256 {expected}, got {actual}")
    raise ArtifactMismatch(
        f"{role}: {path}\n"
        f"  expected sha256 {expected}\n"
        f"  actual   sha256 {actual}\n"
        f"  the locked artifact has been replaced or corrupted; do not measure "
        f"with it until this is explained")


def verify_all(lock: dict | None = None, *, skip_pending: bool = True) -> dict[str, str]:
    """Check every locked artifact at once and report, rather than dying on the first.

    A run that is going to fail on a missing codebook should say so before it
    spends four hours encoding. ``skip_pending`` passes over artifacts flagged as
    not-yet-produced, so the lock can name an output before it exists.
    """
    lock = lock if lock is not None else load_lock()
    report: dict[str, str] = {}
    for role, entry in (lock.get("artifacts") or {}).items():
        if skip_pending and entry.get("pending"):
            report[role] = "pending"
            continue
        try:
            resolve(role, lock=lock)
            report[role] = "ok"
        except ArtifactMismatch as error:
            report[role] = f"FAIL: {error}"
    return report


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default=None)
    parser.add_argument("--role", default=None, help="resolve one role and print it")
    parser.add_argument("--rehash", action="store_true",
                        help="print the sha256 actually on disk for every role, "
                             "in lock-file form; use when minting a new lock")
    args = parser.parse_args()

    lock = load_lock(args.lock)
    if args.role:
        print(resolve(args.role, lock=lock))
        return
    if args.rehash:
        for role, entry in (lock.get("artifacts") or {}).items():
            path = _expand(str(entry["path"]), lock)
            marker = sha256_file(path) if path.exists() else "MISSING"
            print(f"{role}: {marker}")
        return

    report = verify_all(lock)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(any(v.startswith("FAIL") for v in report.values()))


if __name__ == "__main__":
    main()
