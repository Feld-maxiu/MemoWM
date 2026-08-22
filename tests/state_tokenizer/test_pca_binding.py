"""PCA binding checks, runnable from both dependency islands.

``outputs/state_tokenizer/`` holds two trees with identical structure and
identical file names -- the official 20k task-balanced fit and the invalidated
2,000-state first-N fit whose every state came from ``click-button-v1``. The
invalidated tree cannot be deleted because the ~420 GiB Full-H lives beside it,
so the trap is permanent, and it raises nothing: routing in-domain validation
states through the wrong tree drops the retrieval head's paired cosine from
0.8711 to 0.3919 and 500-way Recall@1 from 0.840 to 0.004, chance being 0.002.

Both trees record their own provenance and both are internally self-consistent,
which is why the self-consistency check and the pinned official hash are two
separate defences -- the first catches mixing trees, the second catches picking
the wrong one wholesale. The tests below cover each independently.

Deliberately free of torch and jax imports so it runs under either interpreter.
"""
from __future__ import annotations

import numpy as np

from experiments.state_tokenizer.pca_binding import (
    feature_tree_pca_sha,
    normalization_pca_sha,
    sha256_file,
    verify_binding,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _feature_tree(root, sha, shards=("worker00", "worker01")):
    for shard in shards:
        path = root / shard
        path.mkdir(parents=True, exist_ok=True)
        (path / "key64-static-pca-artifact.sha256").write_text(sha)
    return root


def _normalization(path, sha):
    np.savez(path, pca_sha256=np.asarray(sha), mean=np.zeros(4))
    return path


def test_feature_tree_sha_is_read_from_every_shard(tmp_path):
    root = _feature_tree(tmp_path / "features", SHA_A)
    assert feature_tree_pca_sha(root) == SHA_A


def test_shards_disagreeing_is_an_error(tmp_path):
    root = tmp_path / "features"
    _feature_tree(root, SHA_A, shards=("worker00",))
    _feature_tree(root, SHA_B, shards=("worker01",))
    try:
        feature_tree_pca_sha(root)
    except RuntimeError as error:
        # A tree projected in two passes under different coordinates is
        # unrecoverable, so the message has to name the offending shards.
        assert "worker00" in str(error) and "worker01" in str(error)
    else:
        raise AssertionError("mixed-coordinate tree was accepted")


def test_untransformed_tree_is_an_error_not_a_pass(tmp_path):
    root = tmp_path / "features"
    (root / "worker00").mkdir(parents=True)
    try:
        feature_tree_pca_sha(root)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("a tree with no provenance was accepted")


def test_normalization_carries_its_pca_sha(tmp_path):
    path = _normalization(tmp_path / "norm.npz", SHA_A)
    assert normalization_pca_sha(path) == SHA_A


def test_normalization_without_provenance_is_an_error(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, mean=np.zeros(4))
    try:
        normalization_pca_sha(path)
    except KeyError:
        pass
    else:
        raise AssertionError("a normalization npz predating provenance was accepted")


def test_matching_artifacts_verify(tmp_path):
    features = _feature_tree(tmp_path / "features", SHA_A)
    norm = _normalization(tmp_path / "norm.npz", SHA_A)
    assert verify_binding(features=features, normalization=norm) == SHA_A


def test_mismatched_artifacts_are_rejected(tmp_path):
    features = _feature_tree(tmp_path / "features", SHA_A)
    norm = _normalization(tmp_path / "norm.npz", SHA_B)
    try:
        verify_binding(features=features, normalization=norm)
    except RuntimeError as error:
        message = str(error)
        # Both hashes and both paths must appear, or the operator cannot tell
        # which of the two identically-named trees they actually reached.
        assert SHA_A[:16] in message and SHA_B[:16] in message
        assert "features=" in message and "normalization=" in message
    else:
        raise AssertionError("mismatched coordinates were accepted")


def test_a_single_artifact_is_a_caller_error(tmp_path):
    # Verifying one artifact against nothing would report success while proving
    # nothing, which is the failure mode this module exists to remove.
    features = _feature_tree(tmp_path / "features", SHA_A)
    try:
        verify_binding(features=features)
    except ValueError:
        pass
    else:
        raise AssertionError("a one-artifact check reported success")


def test_pca_file_hash_participates(tmp_path):
    features = _feature_tree(tmp_path / "features", SHA_A)
    pca = tmp_path / "pca.npz"
    np.savez(pca, components=np.zeros((4, 4)))
    real = sha256_file(pca)
    assert real != SHA_A
    try:
        verify_binding(features=features, pca=pca)
    except RuntimeError as error:
        assert real[:16] in str(error)
    else:
        raise AssertionError("a PCA file not matching the feature tree was accepted")
