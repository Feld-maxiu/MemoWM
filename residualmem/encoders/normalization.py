"""Fixed train-only normalization for slot-structured PCA state targets."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..latent.types import StateTokens


class GroupChannelNormalizer:
    """Invertible group×channel normalization with explicit padding semantics."""

    def __init__(
        self,
        mean: np.ndarray,
        scale: np.ndarray,
        layout: tuple[int, ...],
        group_names: tuple[str, ...],
        *,
        artifact_id: str,
        pca_sha256: str = "",
    ):
        mean = np.asarray(mean, np.float32)
        scale = np.asarray(scale, np.float32)
        if mean.ndim != 2 or scale.shape != mean.shape:
            raise ValueError(f"mean/scale must be matching (groups,channels): {mean.shape} {scale.shape}")
        if len(layout) != len(group_names) or len(layout) != len(mean):
            raise ValueError("layout, group names, and statistics must have equal group counts")
        # A group may legitimately be empty -- the prompt band is width 0 once
        # recycled into detail -- but a negative width or an empty layout is a
        # corrupt artifact. np.repeat below handles a zero count correctly.
        if any(size < 0 for size in layout) or sum(layout) < 1:
            raise ValueError(f"normalization layout must be non-negative and non-empty: {layout}")
        if not np.isfinite(mean).all():
            raise ValueError("normalization means must be finite")
        if not np.isfinite(scale).all() or not bool((scale > 0).all()):
            raise ValueError("normalization scales must be finite and positive")
        self.mean = mean
        self.scale = scale
        self.layout = tuple(int(size) for size in layout)
        self.group_names = tuple(group_names)
        self.artifact_id = str(artifact_id)
        self.pca_sha256 = str(pca_sha256)
        self.num_tokens = sum(self.layout)
        self.token_dim = int(mean.shape[1])
        group_index = np.repeat(np.arange(len(self.layout)), self.layout)
        self.slot_mean = self.mean[group_index]
        self.slot_scale = self.scale[group_index]

    @classmethod
    def from_npz(cls, path: str | Path, *, expected_pca_sha256: str | None = None):
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            pca_sha256 = str(np.asarray(data["pca_sha256"]).item())
            if expected_pca_sha256 is not None and pca_sha256 != expected_pca_sha256:
                raise ValueError(
                    f"normalizer PCA hash mismatch: {pca_sha256} != {expected_pca_sha256}"
                )
            return cls(
                mean=data["mean"],
                scale=data["scale"],
                layout=tuple(int(value) for value in data["layout"]),
                group_names=tuple(str(value) for value in data["group_names"]),
                artifact_id=str(path.resolve()),
                pca_sha256=pca_sha256,
            )

    def normalize(self, x_t: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Return xbar_t; invalid slots remain exactly zero after normalization."""
        x_t = np.asarray(x_t, np.float32)
        valid_mask = np.asarray(valid_mask, np.bool_)
        if x_t.shape[-2:] != (self.num_tokens, self.token_dim):
            raise ValueError(
                f"x_t must end in {(self.num_tokens, self.token_dim)}; got {x_t.shape}"
            )
        if valid_mask.shape != x_t.shape[:-1]:
            raise ValueError(f"valid mask {valid_mask.shape} does not match x_t {x_t.shape}")
        xbar_t = (x_t - self.slot_mean) / self.slot_scale
        xbar_t = xbar_t * valid_mask[..., None]
        return xbar_t.astype(np.float32)

    def denormalize(self, xbar_t: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        xbar_t = np.asarray(xbar_t, np.float32)
        valid_mask = np.asarray(valid_mask, np.bool_)
        if xbar_t.shape[-2:] != (self.num_tokens, self.token_dim):
            raise ValueError(
                f"xbar_t must end in {(self.num_tokens, self.token_dim)}; got {xbar_t.shape}"
            )
        if valid_mask.shape != xbar_t.shape[:-1]:
            raise ValueError(f"valid mask {valid_mask.shape} does not match xbar_t {xbar_t.shape}")
        x_t = xbar_t * self.slot_scale + self.slot_mean
        x_t = x_t * valid_mask[..., None]
        return x_t.astype(np.float32)

    def normalize_state(self, x_t: StateTokens) -> StateTokens:
        xbar = self.normalize(x_t.tokens, x_t.valid)
        return StateTokens(xbar, x_t.subspaces, x_t.valid.copy())

    @property
    def hash_bytes(self) -> bytes:
        digest = hashlib.sha256()
        digest.update(b"group-channel-normalizer-v1\0")
        digest.update(np.asarray(self.layout, np.int32).tobytes())
        digest.update("\0".join(self.group_names).encode("utf-8"))
        digest.update(self.mean.tobytes(order="C"))
        digest.update(self.scale.tobytes(order="C"))
        digest.update(self.pca_sha256.encode("ascii"))
        return digest.digest()


class NormalizedObservationEncoder:
    """ObservationEncoder wrapper whose output is xbar_t, never z_t."""

    def __init__(self, encoder, normalizer: GroupChannelNormalizer):
        if encoder.spec.num_tokens != normalizer.num_tokens:
            raise ValueError("encoder token count does not match normalizer")
        if encoder.spec.token_dim != normalizer.token_dim:
            raise ValueError("encoder token width does not match normalizer")
        self.encoder = encoder
        self.normalizer = normalizer
        self.spec = encoder.spec
        self.domain = encoder.domain
        self.encoder_id = f"normalized[{encoder.encoder_id}]"

    def reset(self) -> None:
        self.encoder.reset()

    def encode(self, observation: Any) -> StateTokens:
        x_t = self.encoder.encode(observation)
        return self.normalizer.normalize_state(x_t)

    @property
    def hash_bytes(self) -> bytes:
        return hashlib.sha256(
            self.encoder.hash_bytes + self.normalizer.hash_bytes
        ).digest()
