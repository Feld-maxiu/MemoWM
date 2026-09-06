"""Train only the K32 512->5120 soft-token bridge through frozen Qwen3-32B.

The reader is model-parallel over exactly the CUDA devices exposed by the
launcher.  Formal runs must use ``CUDA_VISIBLE_DEVICES=0,1,2``.  Validation is
controlled: matched, trajectory-shuffled, zero-latent, and question-only all
use the same frozen reader and answer labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from xt_ama_adapter.qwen32_bridge import (
    Qwen32InputSoftTokenBridge,
    Qwen32RMSCalibratedInputSoftTokenBridge,
    Qwen32ScaledInputSoftTokenBridge,
    file_sha256,
    latent_position_ids,
    load_qwen32_bridge,
    render_ama_openend_parts,
    save_qwen32_bridge,
)
from .reader_losses import topk_kl


DATA_PROTOCOL = "humantrajs_qwen32_retrieved_latents_v1"
WEBCHAIN_DATA_PROTOCOL = "webchain_qwen32_retrieved_latents_v1"
MOLMOWEB_DATA_PROTOCOL = "molmoweb_qwen32_matched_latents_v1"
QUESTION_HIDDEN_PROTOCOL = "qwen32_question_only_hidden_v1"


def _reader_artifact_sha256(model_dir: Path) -> str:
    """Hash config/tokenizer and every weight shard, including relative names."""
    files = sorted(
        path for path in model_dir.iterdir()
        if path.is_file() and (
            path.suffix in {".json", ".safetensors"}
            or path.name in {"merges.txt", "vocab.json"}
        )
    )
    if not files or not any(path.suffix == ".safetensors" for path in files):
        raise ValueError(f"{model_dir} has no safetensors reader artifact")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        required = {"xbar", "valid", "trajectory_id", "question", "answer",
                    "split", "metadata"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"bridge dataset lacks {sorted(missing)}")
        arrays = {name: np.asarray(data[name]) for name in required - {"metadata"}}
        if "retrieval_gold_hit" in data.files:
            arrays["retrieval_gold_hit"] = np.asarray(
                data["retrieval_gold_hit"], dtype=bool
            )
        metadata = json.loads(str(np.asarray(data["metadata"]).item()))
    if metadata.get("protocol") not in {
        DATA_PROTOCOL, WEBCHAIN_DATA_PROTOCOL, MOLMOWEB_DATA_PROTOCOL
    }:
        raise ValueError("bridge dataset protocol mismatch")
    if metadata.get("official_ama_test_included") is not False:
        raise ValueError("official AMA test provenance is not explicitly excluded")
    top_k = int(metadata.get("top_k", -1))
    if arrays["xbar"].shape[1:] != (top_k, 32, 512):
        raise ValueError(f"unexpected xbar shape {arrays['xbar'].shape}")
    if arrays["valid"].shape != arrays["xbar"].shape[:-1]:
        raise ValueError("valid mask shape mismatch")
    if not np.isfinite(arrays["xbar"]).all():
        raise ValueError("xbar contains non-finite values")
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"bridge dataset columns have lengths {sorted(lengths)}")
    split = arrays["split"].astype(str)
    trajectory = arrays["trajectory_id"].astype(str)
    owners: dict[str, str] = {}
    for group, name in zip(trajectory, split):
        previous = owners.setdefault(group, name)
        if previous != name:
            raise ValueError(f"trajectory {group} appears in {previous} and {name}")
    if set(split) - {"train", "validation", "test"}:
        raise ValueError("unexpected split value")
    if not np.any(split == "train") or not np.any(split == "validation"):
        raise ValueError("non-empty train and validation splits are required")
    return arrays, metadata


def _select_split_indices(arrays: dict[str, np.ndarray], split_name: str, *,
                          retrieval_hit_only: bool, limit: int,
                          seed: int) -> np.ndarray:
    mask = arrays["split"].astype(str) == split_name
    if retrieval_hit_only:
        if "retrieval_gold_hit" not in arrays:
            raise ValueError("retrieval-hit selection needs retrieval_gold_hit")
        mask &= arrays["retrieval_gold_hit"].astype(bool)
    indices = np.flatnonzero(mask)
    if limit > 0 and len(indices) > limit:
        indices = np.sort(np.random.default_rng(seed).choice(
            indices, size=limit, replace=False
        ))
    if not len(indices):
        raise ValueError(f"selection produced no {split_name} rows")
    return indices


def _load_teacher_cache(directory: Path, *, dataset_sha256: str,
                        reader_sha256: str, prompt_hash: str,
                        enable_thinking: bool, rows: int, train_indices: np.ndarray,
                        qformer_sha256: str, retrieval_head_sha256: str,
                        hidden_layers: tuple[int, ...] = ()):
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"teacher cache has no {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = manifest.get("protocol")
    allowed_protocols = {"qwen32_oracle_text_answer_topk_v1",
                         "qwen32_oracle_text_answer_topk_hidden_v2"}
    if protocol not in allowed_protocols:
        raise ValueError(f"unsupported teacher-cache protocol {protocol!r}")
    if hidden_layers and protocol != "qwen32_oracle_text_answer_topk_hidden_v2":
        raise ValueError("hidden distillation requires a v2 hidden teacher cache")
    required = {
        "status": "complete",
        "dataset_sha256": dataset_sha256,
        "reader_model_sha256": reader_sha256,
        "prompt_sha256": prompt_hash,
        "enable_thinking": bool(enable_thinking),
        "rows": rows,
        "cached_split": "train",
        "official_ama_test_included": False,
        "qformer_artifact_sha256": qformer_sha256,
        "retrieval_head_artifact_sha256": retrieval_head_sha256,
        "coverage_pass": True,
    }
    for name, expected in required.items():
        if manifest.get(name) != expected:
            raise ValueError(
                f"teacher-cache manifest {name}={manifest.get(name)!r}; "
                f"expected {expected!r}"
            )
    cached_indices = set(map(int, manifest.get("cached_indices", ())))
    if int(manifest.get("cached_rows", -1)) != len(cached_indices):
        raise ValueError("teacher-cache cached_rows/cached_indices mismatch")
    missing_indices = set(map(int, train_indices)) - cached_indices
    if missing_indices:
        raise ValueError(
            f"teacher-cache lacks {len(missing_indices)} selected train rows"
        )
    if hidden_layers:
        cached_layers = tuple(map(int, manifest.get("hidden_layers", ())))
        if cached_layers != hidden_layers:
            raise ValueError(
                f"teacher-cache hidden layers {cached_layers}; expected {hidden_layers}"
            )
        if int(manifest.get("hidden_width", -1)) != 5120:
            raise ValueError("teacher-cache hidden width is not 5120")
    cached = {}
    for raw_index in train_indices:
        index = int(raw_index)
        path = directory / f"{index:06d}.npz"
        with np.load(path, allow_pickle=False) as data:
            if int(np.asarray(data["row_index"])) != index:
                raise ValueError(f"{path}: row index mismatch")
            topk_index = np.asarray(data["topk_index"], dtype=np.int64)
            topk_logprob = np.asarray(data["topk_logprob"], dtype=np.float32)
            answer_tokens = int(np.asarray(data["answer_tokens"]))
        if topk_index.shape != topk_logprob.shape or topk_index.shape[0] != answer_tokens:
            raise ValueError(f"{path}: malformed answer top-k tensors")
        row = {
            "topk_index": torch.from_numpy(topk_index),
            "topk_logprob": torch.from_numpy(topk_logprob),
        }
        if hidden_layers:
            with np.load(path, allow_pickle=False) as data:
                hidden_anchor = np.asarray(data["hidden_anchor"], dtype=np.float32)
            if hidden_anchor.shape != (len(hidden_layers), 5120):
                raise ValueError(f"{path}: malformed hidden anchor {hidden_anchor.shape}")
            row["hidden_anchor"] = torch.from_numpy(hidden_anchor)
        cached[index] = row
    return cached, manifest, file_sha256(manifest_path)


def _load_question_hidden_cache(directory: Path, *, dataset_sha256: str,
                                reader_sha256: str, prompt_hash: str,
                                enable_thinking: bool, rows: int,
                                train_indices: np.ndarray,
                                qformer_sha256: str,
                                retrieval_head_sha256: str,
                                hidden_layers: tuple[int, ...]):
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"question-hidden cache has no {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "protocol": QUESTION_HIDDEN_PROTOCOL,
        "status": "complete",
        "dataset_sha256": dataset_sha256,
        "reader_model_sha256": reader_sha256,
        "prompt_sha256": prompt_hash,
        "enable_thinking": bool(enable_thinking),
        "rows": rows,
        "cached_split": "train",
        "official_ama_test_included": False,
        "qformer_artifact_sha256": qformer_sha256,
        "retrieval_head_artifact_sha256": retrieval_head_sha256,
        "hidden_layers": list(hidden_layers),
        "hidden_width": 5120,
        "hidden_anchor_position": "last_prompt_token_before_answer",
    }
    for name, expected in required.items():
        if manifest.get(name) != expected:
            raise ValueError(
                f"question-hidden manifest {name}={manifest.get(name)!r}; "
                f"expected {expected!r}"
            )
    cached_indices = set(map(int, manifest.get("cached_indices", ())))
    if int(manifest.get("cached_rows", -1)) != len(cached_indices):
        raise ValueError("question-hidden cached_rows/cached_indices mismatch")
    missing_indices = set(map(int, train_indices)) - cached_indices
    if missing_indices:
        raise ValueError(
            f"question-hidden cache lacks {len(missing_indices)} selected train rows"
        )
    cached = {}
    for raw_index in train_indices:
        index = int(raw_index)
        path = directory / f"{index:06d}.npz"
        with np.load(path, allow_pickle=False) as data:
            if int(np.asarray(data["row_index"])) != index:
                raise ValueError(f"{path}: row index mismatch")
            hidden_anchor = np.asarray(data["hidden_anchor"], dtype=np.float32)
        if hidden_anchor.shape != (len(hidden_layers), 5120):
            raise ValueError(f"{path}: malformed hidden anchor {hidden_anchor.shape}")
        cached[index] = torch.from_numpy(hidden_anchor)
    return cached, manifest, file_sha256(manifest_path)


def hidden_delta_whitening(teacher_cache: dict[int, dict[str, torch.Tensor]],
                           question_hidden_cache: dict[int, torch.Tensor],
                           train_indices: np.ndarray, *, floor_ratio: float = 0.10):
    """Layerwise diagonal whitening statistics for oracle-minus-question deltas."""
    if not 0 < floor_ratio <= 1:
        raise ValueError("floor_ratio must be in (0, 1]")
    delta = torch.stack([
        teacher_cache[int(index)]["hidden_anchor"].float()
        - question_hidden_cache[int(index)].float()
        for index in train_indices
    ])
    mean = delta.mean(0)
    std = delta.std(0, unbiased=False)
    floor = std.median(-1).values[:, None] * floor_ratio
    scale = std.clamp_min(floor)
    if not torch.isfinite(mean).all() or not torch.isfinite(scale).all():
        raise ValueError("non-finite hidden-delta whitening statistics")
    if not bool((scale > 0).all()):
        raise ValueError("hidden-delta whitening scale is not positive")
    return mean, scale, floor.squeeze(-1)


def paired_ce_margin(matched_ce: torch.Tensor, shuffled_ce: torch.Tensor,
                     *, margin: float) -> torch.Tensor:
    """Per-example ranking hinge; averaging is deliberately after the hinge."""
    if matched_ce.shape != shuffled_ce.shape or matched_ce.ndim != 1:
        raise ValueError("matched/shuffled CE must be aligned one-dimensional rows")
    if margin <= 0:
        raise ValueError("margin must be positive")
    return F.relu(float(margin) + matched_ce - shuffled_ce).mean()


def _decoder_layers(model):
    """Return the frozen decoder block sequence without model-family guessing."""
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(getattr(model, "model", None), "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise ValueError("cannot locate decoder layers for hidden-state distillation")


class HiddenAnchorCapture:
    """Capture selected question-boundary states through lightweight hooks."""

    def __init__(self, model, layer_numbers: tuple[int, ...], *,
                 detach_to_cpu: bool = False) -> None:
        layers = _decoder_layers(model)
        if not layer_numbers or min(layer_numbers) < 1 or max(layer_numbers) > len(layers):
            raise ValueError(
                f"hidden layers {layer_numbers} are outside 1..{len(layers)}"
            )
        self.layer_numbers = tuple(layer_numbers)
        self.detach_to_cpu = bool(detach_to_cpu)
        self.positions: list[int] | None = None
        self.captured: dict[int, torch.Tensor] = {}
        self.handles = [
            layers[number - 1].register_forward_hook(self._hook(number))
            for number in self.layer_numbers
        ]

    def _hook(self, number: int):
        def capture(_module, _inputs, output):
            if self.positions is None:
                return
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            positions = torch.as_tensor(self.positions, device=hidden.device)
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            selected = hidden[rows, positions]
            if self.detach_to_cpu:
                selected = selected.detach().to(device="cpu", dtype=torch.float16)
            self.captured[number] = selected
        return capture

    def begin(self, positions: list[int]) -> None:
        self.positions = list(map(int, positions))
        self.captured.clear()

    def stacked(self, *, device: torch.device | None = None) -> torch.Tensor:
        missing = set(self.layer_numbers) - set(self.captured)
        if missing:
            raise RuntimeError(f"hidden hooks did not capture layers {sorted(missing)}")
        values = [self.captured[number] for number in self.layer_numbers]
        if device is None:
            device = values[0].device
        return torch.stack([value.to(device) for value in values], dim=1)

    def clear(self) -> None:
        self.positions = None
        self.captured.clear()

    def close(self) -> None:
        self.clear()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class SemanticReconstructionDecoder(torch.nn.Module):
    """Training-only low-capacity decoder from bridge content back to xbar."""

    def __init__(self, *, dropout: float) -> None:
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("semantic decoder dropout must be in [0, 1)")
        self.norm = torch.nn.LayerNorm(5120, elementwise_affine=False)
        self.dropout = torch.nn.Dropout(dropout)
        self.projection = torch.nn.Linear(5120, 512)

    def forward(self, content: torch.Tensor, *, noise_ratio: float) -> torch.Tensor:
        values = self.norm(content.float())
        if self.training and noise_ratio > 0:
            rms = values.square().mean(-1, keepdim=True).sqrt().detach()
            values = values + torch.randn_like(values) * rms * noise_ratio
        return self.projection(self.dropout(values))


def semantic_reconstruction_loss(decoder: SemanticReconstructionDecoder,
                                 content: torch.Tensor, xbar: torch.Tensor,
                                 valid: torch.Tensor, *, noise_ratio: float):
    target = F.layer_norm(xbar.float(), (xbar.shape[-1],))
    target = target.flatten(1, 2)
    prediction = decoder(content, noise_ratio=noise_ratio)
    mask = valid.flatten(1, 2).bool()
    prediction, target = prediction[mask], target[mask]
    cosine = F.cosine_similarity(prediction, target, dim=-1)
    smooth_l1 = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
    loss = (1.0 - cosine + 0.25 * smooth_l1).mean()
    return loss, cosine.mean(), smooth_l1.mean()


def semantic_relation_loss(content: torch.Tensor, xbar: torch.Tensor,
                           valid: torch.Tensor) -> torch.Tensor:
    x = F.normalize(xbar.float().flatten(1, 2), dim=-1)
    y = F.normalize(content.float(), dim=-1)
    mask = valid.flatten(1, 2).bool()
    intra_losses = []
    pooled_x, pooled_y = [], []
    for row in range(x.shape[0]):
        row_mask = mask[row]
        xr, yr = x[row, row_mask], y[row, row_mask]
        if xr.shape[0] > 1:
            pair_mask = ~torch.eye(xr.shape[0], dtype=torch.bool, device=xr.device)
            intra_losses.append(F.smooth_l1_loss(
                (yr @ yr.T)[pair_mask], (xr @ xr.T)[pair_mask]
            ))
        pooled_x.append(F.normalize(xr.mean(0), dim=0))
        pooled_y.append(F.normalize(yr.mean(0), dim=0))
    intra = torch.stack(intra_losses).mean() if intra_losses else y.new_zeros(())
    if len(pooled_x) < 2:
        return intra
    px, py = torch.stack(pooled_x), torch.stack(pooled_y)
    pair_mask = ~torch.eye(len(pooled_x), dtype=torch.bool, device=px.device)
    inter = F.smooth_l1_loss((py @ py.T)[pair_mask], (px @ px.T)[pair_mask])
    return 0.5 * (intra + inter)


def _visible_gpu_guard(expected: str) -> None:
    actual = os.environ.get("CUDA_VISIBLE_DEVICES")
    normalized = ",".join(part.strip() for part in (actual or "").split(",") if part.strip())
    if normalized != expected:
        raise RuntimeError(
            f"formal bridge training requires CUDA_VISIBLE_DEVICES={expected}; got {actual!r}"
        )
    expected_count = len(expected.split(","))
    if torch.cuda.device_count() != expected_count:
        raise RuntimeError(
            f"expected {expected_count} visible GPUs, torch sees {torch.cuda.device_count()}"
        )


def _token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _row_parts(tokenizer, question: str, answer: str, enable_thinking: bool):
    prefix, suffix, _ = render_ama_openend_parts(
        tokenizer, question, enable_thinking=enable_thinking
    )
    prefix_ids = _token_ids(tokenizer, prefix)
    suffix_ids = _token_ids(tokenizer, suffix)
    answer_ids = _token_ids(tokenizer, answer)
    eos = tokenizer.eos_token_id
    if eos is not None and (not answer_ids or answer_ids[-1] != eos):
        answer_ids.append(int(eos))
    if not answer_ids:
        raise ValueError("answer tokenized to an empty sequence")
    return prefix_ids, suffix_ids, answer_ids


def _reader_input(model, tokenizer, bridge, xbar: torch.Tensor,
                  valid: torch.Tensor, question: str, answer: str,
                  enable_thinking: bool, mode: str):
    prefix_ids, suffix_ids, answer_ids = _row_parts(
        tokenizer, question, answer, enable_thinking
    )
    embedding = model.get_input_embeddings()
    first_device = embedding.weight.device
    reader_dtype = embedding.weight.dtype

    pre = torch.tensor([prefix_ids], dtype=torch.long, device=first_device)
    tail_ids = suffix_ids + answer_ids
    tail = torch.tensor([tail_ids], dtype=torch.long, device=first_device)
    pre_embeds = embedding(pre)
    tail_embeds = embedding(tail)

    if mode == "question_only":
        latent = pre_embeds[:, :0]
        latent_mask = torch.empty((1, 0), dtype=torch.bool, device=first_device)
    else:
        latent, latent_mask = bridge(
            xbar.to(first_device, dtype=torch.float32), valid.to(first_device)
        )
        latent = latent.to(reader_dtype)
        if mode == "zero":
            latent = torch.zeros_like(latent)
        elif mode != "matched":
            raise ValueError(f"unsupported reader-input mode {mode!r}")

    inputs_embeds = torch.cat((pre_embeds, latent, tail_embeds), dim=1)
    attention_mask = torch.cat((
        torch.ones((1, len(prefix_ids)), dtype=torch.bool, device=first_device),
        latent_mask,
        torch.ones((1, len(tail_ids)), dtype=torch.bool, device=first_device),
    ), dim=1)
    labels = torch.full(attention_mask.shape, -100, dtype=torch.long, device=first_device)
    labels[:, -len(answer_ids):] = torch.tensor(
        answer_ids, dtype=torch.long, device=first_device
    )
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": latent_position_ids(attention_mask),
        "labels": labels,
        "use_cache": False,
    }, len(answer_ids)


def _reader_input_batch(model, tokenizer, bridge, arrays,
                        indices: list[int], enable_thinking: bool,
                        source_indices: list[int] | None = None):
    """Pad several reader inputs while preserving per-row answer locations."""
    if source_indices is None:
        source_indices = indices
    if len(source_indices) != len(indices):
        raise ValueError("source indices must align with question indices")
    rows = []
    answer_lengths = []
    sequence_lengths = []
    for index, source in zip(indices, source_indices):
        kwargs, answer_tokens = _reader_input(
            model, tokenizer, bridge,
            torch.as_tensor(arrays["xbar"][source:source + 1]),
            torch.as_tensor(arrays["valid"][source:source + 1]),
            str(arrays["question"][index]), str(arrays["answer"][index]),
            enable_thinking, "matched",
        )
        rows.append(kwargs)
        answer_lengths.append(answer_tokens)
        sequence_lengths.append(int(kwargs["attention_mask"].shape[1]))

    max_length = max(sequence_lengths)
    inputs_embeds = torch.cat([
        F.pad(row["inputs_embeds"], (0, 0, 0, max_length - length))
        for row, length in zip(rows, sequence_lengths)
    ], dim=0)
    attention_mask = torch.cat([
        F.pad(row["attention_mask"], (0, max_length - length), value=False)
        for row, length in zip(rows, sequence_lengths)
    ], dim=0)
    labels = torch.cat([
        F.pad(row["labels"], (0, max_length - length), value=-100)
        for row, length in zip(rows, sequence_lengths)
    ], dim=0)
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": latent_position_ids(attention_mask),
        "labels": labels,
        "use_cache": False,
    }, answer_lengths, sequence_lengths


def _hard_negative_partners(indices: np.ndarray,
                            arrays: dict[str, np.ndarray]) -> dict[int, int]:
    """Nearest pooled QFormer state from a different trajectory."""
    xbar = arrays["xbar"][indices].astype(np.float32)
    valid = arrays["valid"][indices].astype(np.float32)
    pooled = (xbar * valid[..., None]).sum(axis=(1, 2))
    pooled /= np.maximum(valid.sum(axis=(1, 2), keepdims=False)[:, None], 1.0)
    pooled /= np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8)
    similarity = pooled @ pooled.T
    trajectory = arrays["trajectory_id"][indices].astype(str)
    similarity[trajectory[:, None] == trajectory[None, :]] = -np.inf
    if np.any(~np.isfinite(similarity.max(axis=1))):
        raise ValueError("hard-negative selection needs multiple trajectories")
    chosen = similarity.argmax(axis=1)
    return {int(index): int(indices[position])
            for index, position in zip(indices, chosen)}


def _shuffle_partner(indices: np.ndarray, trajectories: np.ndarray) -> dict[int, int]:
    """Deterministic circular derangement across different trajectories."""
    ordered = list(map(int, indices))
    result = {}
    for offset, index in enumerate(ordered):
        for delta in range(1, len(ordered)):
            candidate = ordered[(offset + delta) % len(ordered)]
            if trajectories[candidate] != trajectories[index]:
                result[index] = candidate
                break
        else:
            raise ValueError("validation needs at least two trajectories")
    return result


def _evaluate(model, tokenizer, bridge, arrays, indices: np.ndarray,
              enable_thinking: bool, limit: int) -> dict[str, float]:
    if limit > 0:
        indices = indices[:limit]
    partner = _shuffle_partner(indices, arrays["trajectory_id"].astype(str))
    totals = {name: 0.0 for name in ("matched", "shuffled", "zero", "question_only")}
    tokens = 0
    model.eval()
    bridge.eval()
    with torch.inference_mode():
        for raw_index in indices:
            index = int(raw_index)
            question = str(arrays["question"][index])
            answer = str(arrays["answer"][index])
            for mode in totals:
                source = partner[index] if mode == "shuffled" else index
                call_mode = "matched" if mode == "shuffled" else mode
                kwargs, answer_tokens = _reader_input(
                    model, tokenizer, bridge,
                    torch.as_tensor(arrays["xbar"][source:source + 1]),
                    torch.as_tensor(arrays["valid"][source:source + 1]),
                    question, answer, enable_thinking, call_mode,
                )
                output = model(**kwargs)
                totals[mode] += float(output.loss) * answer_tokens
                if mode == "matched":
                    tokens += answer_tokens
    return {f"{name}_answer_ce": value / max(tokens, 1)
            for name, value in totals.items()}


def _evaluate_semantics(bridge, decoder: SemanticReconstructionDecoder,
                        arrays, indices: np.ndarray, *, batch_size: int = 16):
    """Report auxiliary validation metrics without involving the Reader."""
    bridge.eval()
    decoder.eval()
    cosine_total = mse_total = relation_total = 0.0
    valid_total = batches = 0
    device = next(bridge.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            xbar = torch.as_tensor(
                arrays["xbar"][selected], dtype=torch.float32, device=device
            )
            valid = torch.as_tensor(
                arrays["valid"][selected], dtype=torch.bool, device=device
            )
            _, _, content = bridge(xbar, valid, return_content=True)
            _loss, cosine, mse = semantic_reconstruction_loss(
                decoder, content, xbar, valid, noise_ratio=0.0
            )
            count = int(valid.sum())
            cosine_total += float(cosine) * count
            mse_total += float(mse) * count
            valid_total += count
            relation_total += float(semantic_relation_loss(content, xbar, valid))
            batches += 1
    return {
        "semantic_reconstruction_cosine": cosine_total / max(valid_total, 1),
        "semantic_reconstruction_smooth_l1": mse_total / max(valid_total, 1),
        "semantic_relation_loss": relation_total / max(batches, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-cache")
    parser.add_argument("--question-hidden-cache")
    parser.add_argument("--init-bridge")
    parser.add_argument("--allow-init-prompt-mismatch", action="store_true")
    parser.add_argument("--allow-init-data-mismatch", action="store_true")
    parser.add_argument("--allow-init-coordinate-mismatch", action="store_true")
    parser.add_argument("--max-memory-gib", type=int, default=34)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument(
        "--step-offset", type=int, default=0,
        help=("completed optimizer steps represented by --init-bridge; advances "
              "the deterministic data stream and uses absolute step numbers in logs"),
    )
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-limit", type=int, default=0,
                        help="0 evaluates the complete validation split")
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--train-retrieval-hit-only", action="store_true")
    parser.add_argument("--eval-retrieval-hit-only", action="store_true")
    parser.add_argument("--patience-evals", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--distill-weight", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--initial-output-scale", type=float, default=1.0)
    parser.add_argument("--min-output-scale", type=float, default=0.005)
    parser.add_argument("--max-output-scale", type=float, default=0.05)
    parser.add_argument("--scale-regularization", type=float, default=0.01)
    parser.add_argument("--rms-calibrate-to-reader", action="store_true")
    parser.add_argument("--shuffle-margin-weight", type=float, default=0.0)
    parser.add_argument("--shuffle-margin", type=float, default=0.15)
    parser.add_argument("--hidden-distill-weight", type=float, default=0.0)
    parser.add_argument("--hidden-delta-whiten", action="store_true")
    parser.add_argument("--hidden-whiten-floor-ratio", type=float, default=0.10)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[16, 32, 48])
    parser.add_argument("--semantic-reconstruction-weight", type=float, default=0.0)
    parser.add_argument("--semantic-relation-weight", type=float, default=0.0)
    parser.add_argument("--semantic-warmup-steps", type=int, default=200)
    parser.add_argument("--semantic-decoder-dropout", type=float, default=0.10)
    parser.add_argument("--semantic-noise-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()
    if args.accumulate < 1 or args.micro_batch_size < 1 or args.max_steps < 1:
        raise ValueError("accumulate, micro-batch-size and max-steps must be positive")
    if args.step_offset < 0:
        raise ValueError("step-offset cannot be negative")
    if args.step_offset and not args.init_bridge:
        raise ValueError("positive step-offset requires --init-bridge")
    if args.distill_weight < 0:
        raise ValueError("distill-weight cannot be negative")
    scaled_bridge = args.initial_output_scale != 1.0
    if args.rms_calibrate_to_reader and scaled_bridge:
        raise ValueError(
            "fixed/learned output scaling and hard Reader RMS calibration are exclusive"
        )
    if scaled_bridge and not (
        0 < args.min_output_scale < args.initial_output_scale < args.max_output_scale
    ):
        raise ValueError("scaled bridge requires 0 < min < initial < max")
    if args.scale_regularization < 0:
        raise ValueError("scale-regularization cannot be negative")
    if args.shuffle_margin_weight < 0 or args.shuffle_margin <= 0:
        raise ValueError("shuffle margin weight must be nonnegative and margin positive")
    semantic_weights = (
        args.hidden_distill_weight,
        args.semantic_reconstruction_weight,
        args.semantic_relation_weight,
    )
    if any(weight < 0 for weight in semantic_weights):
        raise ValueError("semantic loss weights cannot be negative")
    if args.semantic_warmup_steps < 0:
        raise ValueError("semantic warmup steps cannot be negative")
    if not 0 <= args.semantic_decoder_dropout < 1:
        raise ValueError("semantic decoder dropout must be in [0, 1)")
    if args.semantic_noise_ratio < 0:
        raise ValueError("semantic noise ratio cannot be negative")
    hidden_layers = tuple(map(int, args.hidden_layers))
    if len(set(hidden_layers)) != len(hidden_layers):
        raise ValueError("hidden layers must be unique")
    if args.hidden_distill_weight > 0 and not hidden_layers:
        raise ValueError("hidden distillation requires hidden layers")
    if args.distill_weight > 0 and not args.teacher_cache:
        raise ValueError("positive distill-weight requires --teacher-cache")
    if args.hidden_distill_weight > 0 and not args.teacher_cache:
        raise ValueError("positive hidden-distill-weight requires --teacher-cache")
    if args.hidden_delta_whiten and args.hidden_distill_weight <= 0:
        raise ValueError("hidden-delta-whiten requires positive hidden-distill-weight")
    if args.hidden_delta_whiten and not args.question_hidden_cache:
        raise ValueError("hidden-delta-whiten requires --question-hidden-cache")
    if not 0 < args.hidden_whiten_floor_ratio <= 1:
        raise ValueError("hidden-whiten-floor-ratio must be in (0, 1]")
    _visible_gpu_guard("0,1,2")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset_path, model_path = Path(args.dataset), Path(args.model)
    arrays, metadata = _load_dataset(dataset_path)
    split = arrays["split"].astype(str)
    train_indices = _select_split_indices(
        arrays, "train", retrieval_hit_only=args.train_retrieval_hit_only,
        limit=args.train_limit, seed=args.seed,
    )
    validation_indices = _select_split_indices(
        arrays, "validation", retrieval_hit_only=args.eval_retrieval_hit_only,
        limit=0, seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _, _, template_prompt_hash = render_ama_openend_parts(
        tokenizer, "{QUESTION}", enable_thinking=args.enable_thinking
    )
    # Do the 62-GB provenance read before reserving any GPU memory.  Keeping a
    # loaded 32B model idle while hashing its own shards would waste the scarce
    # three-card window.
    print("[bridge] hashing the complete Qwen3-32B reader artifact", flush=True)
    reader_hash = _reader_artifact_sha256(model_path)
    data_hash = file_sha256(dataset_path)
    teacher_cache = None
    teacher_manifest_hash = None
    question_hidden_cache = None
    question_hidden_manifest_hash = None
    hidden_whiten_mean = None
    hidden_whiten_scale = None
    hidden_whiten_floor = None
    if args.distill_weight > 0 or args.hidden_distill_weight > 0:
        teacher_cache, _teacher_manifest, teacher_manifest_hash = _load_teacher_cache(
            Path(args.teacher_cache), dataset_sha256=data_hash,
            reader_sha256=reader_hash, prompt_hash=template_prompt_hash,
            enable_thinking=args.enable_thinking, rows=len(arrays["question"]),
            train_indices=train_indices,
            qformer_sha256=metadata["qformer_artifact_sha256"],
            retrieval_head_sha256=metadata["retrieval_head_artifact_sha256"],
            hidden_layers=(hidden_layers if args.hidden_distill_weight > 0 else ()),
        )
    if args.hidden_delta_whiten:
        (question_hidden_cache, _question_hidden_manifest,
         question_hidden_manifest_hash) = _load_question_hidden_cache(
            Path(args.question_hidden_cache), dataset_sha256=data_hash,
            reader_sha256=reader_hash, prompt_hash=template_prompt_hash,
            enable_thinking=args.enable_thinking, rows=len(arrays["question"]),
            train_indices=train_indices,
            qformer_sha256=metadata["qformer_artifact_sha256"],
            retrieval_head_sha256=metadata["retrieval_head_artifact_sha256"],
            hidden_layers=hidden_layers,
        )
        hidden_whiten_mean, hidden_whiten_scale, hidden_whiten_floor = (
            hidden_delta_whitening(
                teacher_cache, question_hidden_cache, train_indices,
                floor_ratio=args.hidden_whiten_floor_ratio,
            )
        )
    max_memory = {
        index: f"{args.max_memory_gib}GiB" for index in range(torch.cuda.device_count())
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="balanced",
        max_memory=max_memory,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("reader freeze failed")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Seed before constructing the only trainable module.  Seeding after this
    # point makes nominally identical bridge runs start from different weights.
    torch.manual_seed(args.seed)
    bridge_kwargs = {"max_memory_ranks": int(metadata["top_k"])}
    if args.init_bridge:
        bridge, init_metadata = load_qwen32_bridge(
            args.init_bridge,
            expected_qformer_sha256=(
                None if args.allow_init_coordinate_mismatch
                else metadata["qformer_artifact_sha256"]
            ),
            expected_retrieval_head_sha256=(
                None if args.allow_init_coordinate_mismatch
                else metadata["retrieval_head_artifact_sha256"]
            ),
        )
        if init_metadata["reader_model_sha256"] != reader_hash:
            raise ValueError("initial bridge reader hash mismatch")
        if (init_metadata["data_manifest_sha256"] != data_hash
                and not args.allow_init_data_mismatch):
            raise ValueError("initial bridge dataset hash mismatch")
        if (init_metadata["prompt_sha256"] != template_prompt_hash
                and not args.allow_init_prompt_mismatch):
            raise ValueError("initial bridge prompt hash mismatch")
        if args.rms_calibrate_to_reader:
            embedding = model.get_input_embeddings().weight
            total, count = 0.0, 0
            with torch.no_grad():
                for start in range(0, embedding.shape[0], 4096):
                    block = embedding[start:start + 4096].float()
                    total += float(block.square().sum().cpu())
                    count += block.numel()
            target_rms = math.sqrt(total / count)
            if isinstance(bridge, Qwen32RMSCalibratedInputSoftTokenBridge):
                if not math.isclose(
                    float(bridge.target_rms), target_rms, rel_tol=0, abs_tol=1e-8
                ):
                    raise ValueError("initial RMS Bridge target differs from Reader RMS")
            else:
                converted = Qwen32RMSCalibratedInputSoftTokenBridge(
                    target_rms=target_rms, **bridge_kwargs
                )
                source_state = bridge.state_dict()
                result = converted.load_state_dict(
                    {key: value for key, value in source_state.items()
                     if key != "output_scale_logit"},
                    strict=False,
                )
                if set(result.missing_keys) != {"target_rms"} or result.unexpected_keys:
                    raise ValueError(
                        f"cannot transfer initial Bridge into RMS-calibrated v3: {result}"
                    )
                bridge = converted
        scaled_bridge = isinstance(bridge, Qwen32ScaledInputSoftTokenBridge)
    elif args.rms_calibrate_to_reader:
        embedding = model.get_input_embeddings().weight
        total, count = 0.0, 0
        with torch.no_grad():
            for start in range(0, embedding.shape[0], 4096):
                block = embedding[start:start + 4096].float()
                total += float(block.square().sum().cpu())
                count += block.numel()
        bridge = Qwen32RMSCalibratedInputSoftTokenBridge(
            target_rms=math.sqrt(total / count), **bridge_kwargs
        )
        scaled_bridge = False
    elif scaled_bridge:
        bridge = Qwen32ScaledInputSoftTokenBridge(
            initial_output_scale=args.initial_output_scale,
            min_output_scale=args.min_output_scale,
            max_output_scale=args.max_output_scale,
            **bridge_kwargs,
        )
    else:
        bridge = Qwen32InputSoftTokenBridge(**bridge_kwargs)
    bridge = bridge.to(model.get_input_embeddings().weight.device)
    semantic_decoder = None
    if args.semantic_reconstruction_weight > 0:
        semantic_decoder = SemanticReconstructionDecoder(
            dropout=args.semantic_decoder_dropout
        ).to(next(bridge.parameters()).device)
    trainable_parameters = list(bridge.parameters())
    if semantic_decoder is not None:
        trainable_parameters.extend(semantic_decoder.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in trainable_parameters}
    if optimized != expected:
        raise RuntimeError("optimizer parameter set does not match bridge + semantic decoder")
    hidden_capture = (
        HiddenAnchorCapture(model, hidden_layers)
        if args.hidden_distill_weight > 0 else None
    )

    rng = np.random.default_rng(args.seed)
    best, stale, history = math.inf, 0, []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    order = rng.permutation(train_indices)
    hard_negative = (_hard_negative_partners(train_indices, arrays)
                     if args.shuffle_margin_weight > 0 else None)
    objective = ("answer_ce+oracle_text_answer_kl"
                 if args.distill_weight > 0 else "answer_ce")
    if scaled_bridge:
        objective += "+bounded_scale_regularization"
    if isinstance(bridge, Qwen32RMSCalibratedInputSoftTokenBridge):
        objective += "+fixed_reader_embedding_rms"
    if args.shuffle_margin_weight > 0:
        objective += "+hard_negative_ce_margin"
    if args.hidden_distill_weight > 0:
        objective += (
            "+reader_hidden_delta_whitened_cosine"
            if args.hidden_delta_whiten else "+reader_hidden_cosine"
        )
    if args.semantic_reconstruction_weight > 0:
        objective += "+xbar_reconstruction"
    if args.semantic_relation_weight > 0:
        objective += "+xbar_relation_geometry"
    cursor = 0
    # Continue the deterministic sample stream represented by an initialized
    # weight-only checkpoint.  The optimizer state is intentionally fresh, but
    # previously consumed examples must not be replayed from the start.
    consumed_rows = (
        args.step_offset * args.accumulate * args.micro_batch_size
    )
    for _ in range(consumed_rows):
        if cursor >= len(order):
            order = rng.permutation(train_indices)
            cursor = 0
        cursor += 1
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.max_steps + 1):
        global_step = args.step_offset + step
        # Qwen3 has zero attention dropout, so train mode does not introduce a
        # stochastic reader.  It does activate Transformers' gradient
        # checkpointing path, which is required to fit the frozen 64-layer
        # backward graph on three cards.
        model.train()
        bridge.train()
        if semantic_decoder is not None:
            semantic_decoder.train()
        running_ce = 0.0
        running_kl = 0.0
        running_hidden = 0.0
        running_reconstruction = 0.0
        running_relation = 0.0
        running_reconstruction_cosine = 0.0
        running_shuffled_ce = 0.0
        running_margin_loss = 0.0
        for _micro in range(args.accumulate):
            batch_indices = []
            for _ in range(args.micro_batch_size):
                if cursor >= len(order):
                    order = rng.permutation(train_indices)
                    cursor = 0
                batch_indices.append(int(order[cursor]))
                cursor += 1
            kwargs, answer_lengths, sequence_lengths = _reader_input_batch(
                model, tokenizer, bridge, arrays, batch_indices,
                args.enable_thinking,
            )
            if hidden_capture is not None:
                hidden_capture.begin([
                    length - answer_tokens - 1
                    for length, answer_tokens in zip(sequence_lengths, answer_lengths)
                ])
            output = model(**kwargs)
            row_ce = []
            row_kl = []
            for row, (index, answer_tokens, sequence_length) in enumerate(zip(
                    batch_indices, answer_lengths, sequence_lengths)):
                answer_start = sequence_length - answer_tokens
                answer_logits = output.logits[
                    row, answer_start - 1:sequence_length - 1
                ]
                answer_labels = kwargs["labels"][row, answer_start:sequence_length]
                row_ce.append(F.cross_entropy(
                    answer_logits.float(), answer_labels, reduction="mean"
                ))
                if teacher_cache is not None:
                    topk_index = teacher_cache[index]["topk_index"]
                    topk_logprob = teacher_cache[index]["topk_logprob"]
                    if int(topk_index.shape[0]) != answer_tokens:
                        raise ValueError(
                            f"teacher-cache answer length mismatch at row {index}"
                        )
                    row_kl.append(topk_kl(
                        answer_logits[None], topk_index[None], topk_logprob[None]
                    ))
            ce = torch.stack(row_ce).mean()
            if teacher_cache is not None:
                kl = torch.stack(row_kl).mean()
            else:
                kl = ce.new_zeros(())
            loss = ce + args.distill_weight * kl
            if hidden_capture is not None:
                student_hidden = hidden_capture.stacked(
                    device=next(bridge.parameters()).device
                ).float()
                teacher_hidden = torch.stack([
                    teacher_cache[index]["hidden_anchor"] for index in batch_indices
                ]).to(student_hidden.device, dtype=torch.float32)
                if args.hidden_delta_whiten:
                    question_hidden = torch.stack([
                        question_hidden_cache[index] for index in batch_indices
                    ]).to(student_hidden.device, dtype=torch.float32)
                    whiten_mean = hidden_whiten_mean.to(student_hidden.device)
                    whiten_scale = hidden_whiten_scale.to(student_hidden.device)
                    student_hidden = (
                        student_hidden - question_hidden - whiten_mean
                    ) / whiten_scale
                    teacher_hidden = (
                        teacher_hidden - question_hidden - whiten_mean
                    ) / whiten_scale
                hidden_loss = (
                    1.0 - F.cosine_similarity(
                        student_hidden, teacher_hidden, dim=-1
                    )
                ).mean()
            else:
                hidden_loss = ce.new_zeros(())

            if (semantic_decoder is not None
                    or args.semantic_relation_weight > 0):
                first_device = next(bridge.parameters()).device
                semantic_xbar = torch.as_tensor(
                    arrays["xbar"][batch_indices], dtype=torch.float32,
                    device=first_device,
                )
                semantic_valid = torch.as_tensor(
                    arrays["valid"][batch_indices], dtype=torch.bool,
                    device=first_device,
                )
                _, _, semantic_content = bridge(
                    semantic_xbar, semantic_valid, return_content=True
                )
            if semantic_decoder is not None:
                reconstruction_loss, reconstruction_cosine, _reconstruction_mse = (
                    semantic_reconstruction_loss(
                        semantic_decoder, semantic_content, semantic_xbar,
                        semantic_valid, noise_ratio=args.semantic_noise_ratio,
                    )
                )
            else:
                reconstruction_loss = ce.new_zeros(())
                reconstruction_cosine = ce.new_zeros(())
            if args.semantic_relation_weight > 0:
                relation_loss = semantic_relation_loss(
                    semantic_content, semantic_xbar, semantic_valid
                )
            else:
                relation_loss = ce.new_zeros(())
            semantic_factor = (
                1.0 if args.semantic_warmup_steps == 0
                else min(1.0, global_step / args.semantic_warmup_steps)
            )
            loss = loss + semantic_factor * (
                args.hidden_distill_weight * hidden_loss
                + args.semantic_reconstruction_weight * reconstruction_loss
                + args.semantic_relation_weight * relation_loss
            )
            if hard_negative is not None:
                negative_sources = [hard_negative[index] for index in batch_indices]
                negative_kwargs, negative_answer_lengths, negative_sequence_lengths = (
                    _reader_input_batch(
                        model, tokenizer, bridge, arrays, batch_indices,
                        args.enable_thinking, source_indices=negative_sources,
                    )
                )
                negative_output = model(**negative_kwargs)
                negative_ce_rows = []
                for row, (answer_tokens, sequence_length) in enumerate(zip(
                        negative_answer_lengths, negative_sequence_lengths)):
                    answer_start = sequence_length - answer_tokens
                    negative_ce_rows.append(F.cross_entropy(
                        negative_output.logits[
                            row, answer_start - 1:sequence_length - 1
                        ].float(),
                        negative_kwargs["labels"][row, answer_start:sequence_length],
                        reduction="mean",
                    ))
                shuffled_ce_rows = torch.stack(negative_ce_rows)
                shuffled_ce = shuffled_ce_rows.mean()
                margin_loss = paired_ce_margin(
                    torch.stack(row_ce), shuffled_ce_rows,
                    margin=args.shuffle_margin,
                )
                loss = loss + args.shuffle_margin_weight * margin_loss
            else:
                shuffled_ce = ce.new_zeros(())
                margin_loss = ce.new_zeros(())
            if isinstance(bridge, Qwen32ScaledInputSoftTokenBridge):
                relative_scale = bridge.output_scale() / bridge.initial_output_scale
                scale_penalty = relative_scale.log().square()
                loss = loss + args.scale_regularization * scale_penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite bridge loss at step {global_step}"
                )
            (loss / args.accumulate).backward()
            if hidden_capture is not None:
                hidden_capture.clear()
            running_ce += float(ce.detach()) / args.accumulate
            running_kl += float(kl.detach()) / args.accumulate
            running_hidden += float(hidden_loss.detach()) / args.accumulate
            running_reconstruction += float(
                reconstruction_loss.detach()
            ) / args.accumulate
            running_relation += float(relation_loss.detach()) / args.accumulate
            running_reconstruction_cosine += float(
                reconstruction_cosine.detach()
            ) / args.accumulate
            running_shuffled_ce += float(shuffled_ce.detach()) / args.accumulate
            running_margin_loss += float(margin_loss.detach()) / args.accumulate
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            trainable_parameters, args.clip_norm
        ))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(
                f"non-finite bridge gradient at step {global_step}"
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = _evaluate(
                model, tokenizer, bridge, arrays, validation_indices,
                args.enable_thinking, args.eval_limit,
            )
            semantic_metrics = (
                _evaluate_semantics(
                    bridge, semantic_decoder, arrays, validation_indices
                ) if semantic_decoder is not None else {}
            )
            record = {"step": global_step, "train_answer_ce": running_ce,
                      "train_distill_kl": running_kl,
                      "train_hidden_distill": running_hidden,
                      "train_semantic_reconstruction": running_reconstruction,
                      "train_semantic_reconstruction_cosine":
                          running_reconstruction_cosine,
                      "train_semantic_relation": running_relation,
                      "semantic_weight_factor": semantic_factor,
                      "train_hard_negative_ce": running_shuffled_ce,
                      "train_margin_loss": running_margin_loss,
                      "grad_norm": grad_norm, **metrics, **semantic_metrics}
            if isinstance(bridge, Qwen32ScaledInputSoftTokenBridge):
                record["bridge_output_scale"] = float(
                    bridge.output_scale().detach().cpu()
                )
                record["output_norm_weight_rms"] = float(
                    bridge.output_norm.weight.detach().float().square().mean().sqrt().cpu()
                )
            history.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            matched = metrics["matched_answer_ce"]
            if matched < best:
                best, stale = matched, 0
                save_qwen32_bridge(
                    output_path, bridge,
                    qformer_artifact_sha256=metadata["qformer_artifact_sha256"],
                    retrieval_head_artifact_sha256=metadata[
                        "retrieval_head_artifact_sha256"
                    ],
                    reader_model_sha256=reader_hash,
                    data_manifest_sha256=data_hash,
                    prompt_sha256_value=template_prompt_hash,
                    best_step=global_step,
                    validation={**metrics, **semantic_metrics},
                    enable_thinking=bool(args.enable_thinking),
                    dataset_protocol=metadata["protocol"],
                    train_split="train",
                    official_ama_test_included=False,
                    seed=args.seed,
                    objective=objective,
                    distill_weight=args.distill_weight,
                    micro_batch_size=args.micro_batch_size,
                    accumulate=args.accumulate,
                    effective_batch_size=args.micro_batch_size * args.accumulate,
                    step_offset=args.step_offset,
                    optimizer_state_resumed=False,
                    gradient_checkpointing=bool(args.gradient_checkpointing),
                    learning_rate=args.learning_rate,
                    clip_norm=args.clip_norm,
                    scale_regularization=args.scale_regularization,
                    teacher_cache_manifest_sha256=teacher_manifest_hash,
                    initialized_from=(file_sha256(args.init_bridge)
                                      if args.init_bridge else None),
                    shuffle_margin_weight=args.shuffle_margin_weight,
                    shuffle_margin=args.shuffle_margin,
                    hidden_distill_weight=args.hidden_distill_weight,
                    hidden_layers=list(hidden_layers),
                    hidden_delta_whiten=bool(args.hidden_delta_whiten),
                    hidden_whiten_floor_ratio=args.hidden_whiten_floor_ratio,
                    hidden_whiten_floor=(
                        hidden_whiten_floor.tolist()
                        if hidden_whiten_floor is not None else None
                    ),
                    question_hidden_cache_manifest_sha256=
                        question_hidden_manifest_hash,
                    semantic_reconstruction_weight=
                        args.semantic_reconstruction_weight,
                    semantic_relation_weight=args.semantic_relation_weight,
                    semantic_warmup_steps=args.semantic_warmup_steps,
                    semantic_decoder_dropout=args.semantic_decoder_dropout,
                    semantic_noise_ratio=args.semantic_noise_ratio,
                    semantic_decoder_saved=False,
                )
            else:
                stale += 1
                if stale >= args.patience_evals:
                    break

    if hidden_capture is not None:
        hidden_capture.close()

    report = {
        "protocol": bridge.protocol,
        "dataset": str(dataset_path.resolve()),
        "reader": str(model_path.resolve()),
        "reader_model_sha256": reader_hash,
        "data_manifest_sha256": data_hash,
        "prompt_sha256": template_prompt_hash,
        "enable_thinking": bool(args.enable_thinking),
        "objective": objective,
        "distill_weight": args.distill_weight,
        "hidden_distill_weight": args.hidden_distill_weight,
        "hidden_layers": list(hidden_layers),
        "hidden_delta_whiten": bool(args.hidden_delta_whiten),
        "hidden_whiten_floor_ratio": args.hidden_whiten_floor_ratio,
        "hidden_whiten_floor": (
            hidden_whiten_floor.tolist()
            if hidden_whiten_floor is not None else None
        ),
        "question_hidden_cache": (
            str(Path(args.question_hidden_cache).resolve())
            if args.question_hidden_cache else None
        ),
        "question_hidden_cache_manifest_sha256":
            question_hidden_manifest_hash,
        "semantic_reconstruction_weight": args.semantic_reconstruction_weight,
        "semantic_relation_weight": args.semantic_relation_weight,
        "semantic_warmup_steps": args.semantic_warmup_steps,
        "semantic_decoder_dropout": args.semantic_decoder_dropout,
        "semantic_noise_ratio": args.semantic_noise_ratio,
        "semantic_decoder_saved": False,
        "micro_batch_size": args.micro_batch_size,
        "accumulate": args.accumulate,
        "effective_batch_size": args.micro_batch_size * args.accumulate,
        "step_offset": args.step_offset,
        "optimizer_state_resumed": False,
        "train_rows": int(len(train_indices)),
        "train_retrieval_hit_only": bool(args.train_retrieval_hit_only),
        "eval_retrieval_hit_only": bool(args.eval_retrieval_hit_only),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "learning_rate": args.learning_rate,
        "clip_norm": args.clip_norm,
        "initial_output_scale": (
            bridge.initial_output_scale
            if isinstance(bridge, Qwen32ScaledInputSoftTokenBridge) else None
        ),
        "learned_output_scale": (
            float(bridge.output_scale().detach().cpu()) if scaled_bridge else None
        ),
        "scale_regularization": args.scale_regularization,
        "target_reader_embedding_rms": (
            float(bridge.target_rms.detach().cpu())
            if isinstance(bridge, Qwen32RMSCalibratedInputSoftTokenBridge)
            else None
        ),
        "shuffle_margin_weight": args.shuffle_margin_weight,
        "shuffle_margin": args.shuffle_margin,
        "teacher_cache": (str(Path(args.teacher_cache).resolve())
                          if args.teacher_cache else None),
        "teacher_cache_manifest_sha256": teacher_manifest_hash,
        "initialized_from": (str(Path(args.init_bridge).resolve())
                             if args.init_bridge else None),
        "best_validation_answer_ce": best,
        "history": history,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "history"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
