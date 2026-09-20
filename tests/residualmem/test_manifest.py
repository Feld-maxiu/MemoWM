"""The lock is only worth having if it actually stops the wrong bytes.

The failure this guards against is silent by construction: substituting the
``pq/`` fit of the codebook for the ``pq-full/`` one produces valid ``(32, 32)
uint8`` codes, a cache that builds, and a model that scores it, with roughly 972
bits of NLL error and no exception anywhere. A manifest that merely *records* the
right hash without refusing the wrong one would be decoration. So the load-
bearing test here is the negative one.

Imports neither torch nor jax on purpose: the orchestrator verifies the same lock
from both environments, and a test that only passes under one of them would not
notice the module growing a dependency that breaks the other.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from residualmem import manifest

# The second k-means fit of the same configuration -- the artifact this whole
# mechanism exists to reject. Recorded in the lock under `known_wrong`.
WRONG_CODEBOOK_SHA = "658268350c8ca4c3fa49c67da12484c9c41aa64c4e30b3837ee988d1d1fe4f8a"


@pytest.fixture(scope="module")
def lock() -> dict:
    return manifest.load_lock()


def test_every_locked_artifact_is_present_and_unmodified(lock):
    report = manifest.verify_all(lock)
    bad = {role: status for role, status in report.items() if status.startswith("FAIL")}
    assert not bad, f"locked artifacts disagree with disk: {bad}"


def test_wrong_codebook_is_rejected_by_name(lock):
    """The negative test. Point the lock at ``pq/`` and demand a specific refusal.

    Skips rather than fails if the wrong codebook is not on this machine -- the
    check is about the mechanism, and a machine without the trap file cannot
    exercise it.
    """
    wrong = manifest._expand("${DATA}/pq/opq-shared-mix10-M32-C64.npz", lock)
    if not wrong.exists():
        pytest.skip(f"the known-wrong codebook is not on this machine ({wrong})")

    tampered = copy.deepcopy(lock)
    tampered["artifacts"]["codebook"]["path"] = str(wrong)

    with pytest.raises(manifest.ArtifactMismatch) as caught:
        manifest.resolve("codebook", lock=tampered)

    message = str(caught.value)
    assert "KNOWN-WRONG" in message, message
    assert "972" in message, "the error should quantify what it costs, not just differ"
    assert WRONG_CODEBOOK_SHA in message, message


def test_unknown_mismatch_still_refuses(lock, tmp_path: Path):
    """A file that is neither the locked artifact nor a catalogued wrong one."""
    decoy = tmp_path / "opq-shared-mix10-M32-C64.npz"
    decoy.write_bytes(b"not a codebook")

    tampered = copy.deepcopy(lock)
    tampered["artifacts"]["codebook"]["path"] = str(decoy)

    with pytest.raises(manifest.ArtifactMismatch) as caught:
        manifest.resolve("codebook", lock=tampered)
    assert "do not measure" in str(caught.value)


def test_missing_artifact_names_its_producer(lock, tmp_path: Path):
    tampered = copy.deepcopy(lock)
    tampered["artifacts"]["utility_mask"]["path"] = str(tmp_path / "absent.npz")

    with pytest.raises(manifest.ArtifactMismatch) as caught:
        manifest.resolve("utility_mask", lock=tampered)
    assert "export_mask.py" in str(caught.value)


def test_pending_artifacts_do_not_fail_a_full_verify(lock, tmp_path: Path):
    """A lock may name an output before it exists; that must not block a run.

    Synthesised rather than pointed at whichever artifact happens to be unbuilt
    today -- the mechanism is what is under test, and every real artifact is
    expected to become non-pending eventually.
    """
    tampered = copy.deepcopy(lock)
    tampered["artifacts"]["not_built_yet"] = {
        "path": str(tmp_path / "absent.npz"),
        "sha256": None,
        "pending": True,
        "produced_by": "some_future_script.py",
    }
    report = manifest.verify_all(tampered)
    assert report["not_built_yet"] == "pending"
    assert not any(status.startswith("FAIL") for status in report.values())

    # ...but only because it was flagged. Unflagged, the same entry must fail.
    strict = manifest.verify_all(tampered, skip_pending=False)
    assert strict["not_built_yet"].startswith("FAIL")


def test_training_only_artifacts_are_optional_at_runtime(lock, tmp_path: Path):
    tampered = copy.deepcopy(lock)
    training = tampered["artifacts"]["retrieval_cache"]
    training["path"] = str(tmp_path / "not-shipped.npz")

    runtime = manifest.verify_all(tampered)
    assert "retrieval_cache" not in runtime

    full = manifest.verify_all(tampered, include_training=True)
    assert full["retrieval_cache"].startswith("FAIL")


def test_unknown_role_lists_what_is_available(lock):
    with pytest.raises(manifest.ArtifactMismatch) as caught:
        manifest.resolve("reader_head", lock=lock)
    assert "qformer" in str(caught.value)


def test_hash_cache_keys_on_mtime(tmp_path: Path):
    """A file replaced between resolves must be re-hashed, not trusted.

    Same-length content on purpose: a cache keyed only on size would pass a
    shorter test and still return stale bytes for an in-place rewrite, which is
    exactly what a rerun of a training script does to a checkpoint.
    """
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"first!")
    first = manifest.sha256_file(path)
    path.write_bytes(b"second")
    assert path.stat().st_size == 6, "the two writes must be the same length"
    assert manifest.sha256_file(path) != first
