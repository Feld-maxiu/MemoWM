"""Diagnose whether a frozen Qwen3-32B Reader uses frozen Bridge residuals.

This is a validation-only, no-training experiment.  It measures:

1. the alignment between the matched->shuffled latent displacement and the
   Reader answer-CE gradient;
2. answer CE after amplifying the matched latent residual around the global
   validation mean; and
3. answer CE after exchanging components parallel/orthogonal to the global
   common Bridge-output direction between matched and shuffled rows.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from xt_ama_adapter.qwen32_bridge import (
    file_sha256,
    latent_position_ids,
    load_qwen32_bridge,
)
from .train_qwen32_bridge import (
    _load_dataset,
    _reader_artifact_sha256,
    _row_parts,
    _select_split_indices,
    _shuffle_partner,
)


PROTOCOL = "qwen32_reader_sensitivity_validation_v1"


def _visible_gpu_guard(expected: str) -> None:
    actual = os.environ.get("CUDA_VISIBLE_DEVICES")
    normalized = ",".join(
        part.strip() for part in (actual or "").split(",") if part.strip()
    )
    if normalized != expected:
        raise RuntimeError(
            f"diagnostic requires CUDA_VISIBLE_DEVICES={expected}; got {actual!r}"
        )
    if torch.cuda.device_count() != len(expected.split(",")):
        raise RuntimeError("visible CUDA device count does not match the launcher")


def _trajectory_stratified_sample(arrays: dict[str, np.ndarray],
                                  validation: np.ndarray, *, size: int,
                                  seed: int) -> np.ndarray:
    """Select one row from each of ``size`` distinct validation trajectories."""
    trajectories = arrays["trajectory_id"].astype(str)
    groups: dict[str, list[int]] = {}
    for raw_index in validation:
        index = int(raw_index)
        groups.setdefault(trajectories[index], []).append(index)
    if size > len(groups):
        raise ValueError(
            f"requested {size} trajectories but validation contains {len(groups)}"
        )
    rng = np.random.default_rng(seed)
    chosen_groups = rng.choice(np.asarray(sorted(groups)), size=size, replace=False)
    chosen = [int(rng.choice(groups[str(group)])) for group in chosen_groups]
    return np.asarray(sorted(chosen), dtype=np.int64)


def _rms_calibrate(values: torch.Tensor, target_rms: float) -> torch.Tensor:
    inverse = values.float().square().mean(-1, keepdim=True).clamp_min(1e-12).rsqrt()
    return values * inverse.to(values.dtype) * float(target_rms)


def _common_residual(values: torch.Tensor,
                     common_direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split each latent token into global-mean-parallel and orthogonal parts."""
    direction = F.normalize(common_direction.float(), dim=-1).to(values.device)
    source = values.float()
    common = (source * direction).sum(-1, keepdim=True) * direction
    return common, source - common


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("summary requires non-empty finite values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _bootstrap_mean_ci(values: list[float], *, seed: int,
                       draws: int = 2000) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(sampled, 0.025)),
        "ci95_high": float(np.quantile(sampled, 0.975)),
    }


def _build_batch(model, tokenizer, latents: torch.Tensor,
                 questions: list[str], answers: list[str],
                 enable_thinking: bool) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    embedding = model.get_input_embeddings()
    device, dtype = embedding.weight.device, embedding.weight.dtype
    if latents.ndim != 3 or latents.shape[0] != len(questions):
        raise ValueError("latents and text rows do not align")
    rows, labels_rows, lengths = [], [], []
    answer_lengths = []
    for row, (question, answer) in enumerate(zip(questions, answers)):
        prefix_ids, suffix_ids, answer_ids = _row_parts(
            tokenizer, question, answer, enable_thinking
        )
        pre = torch.tensor([prefix_ids], dtype=torch.long, device=device)
        tail_ids = suffix_ids + answer_ids
        tail = torch.tensor([tail_ids], dtype=torch.long, device=device)
        latent = latents[row:row + 1].to(device=device, dtype=dtype)
        inputs = torch.cat((embedding(pre), latent, embedding(tail)), dim=1)
        labels = torch.full(
            (1, inputs.shape[1]), -100, dtype=torch.long, device=device
        )
        labels[:, -len(answer_ids):] = torch.tensor(
            answer_ids, dtype=torch.long, device=device
        )
        rows.append(inputs)
        labels_rows.append(labels)
        lengths.append(inputs.shape[1])
        answer_lengths.append(len(answer_ids))
    maximum = max(lengths)
    inputs_embeds = torch.cat([
        F.pad(row, (0, 0, 0, maximum - length))
        for row, length in zip(rows, lengths)
    ])
    labels = torch.cat([
        F.pad(row, (0, maximum - length), value=-100)
        for row, length in zip(labels_rows, lengths)
    ])
    attention_mask = labels.new_zeros(labels.shape, dtype=torch.bool)
    for row, length in enumerate(lengths):
        attention_mask[row, :length] = True
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "position_ids": latent_position_ids(attention_mask),
        "labels": labels,
        "use_cache": False,
    }, torch.tensor(answer_lengths, dtype=torch.long, device=device)


def _per_row_answer_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1), reduction="none", ignore_index=-100,
    ).view_as(shift_labels)
    mask = shift_labels.ne(-100)
    return (losses * mask).sum(1) / mask.sum(1).clamp_min(1)


def _reader_ce(model, tokenizer, latents: torch.Tensor,
               questions: list[str], answers: list[str],
               enable_thinking: bool) -> torch.Tensor:
    kwargs, _ = _build_batch(
        model, tokenizer, latents, questions, answers, enable_thinking
    )
    output = model(**kwargs)
    return _per_row_answer_ce(output.logits, kwargs["labels"])


def _bridge_outputs(bridge, arrays: dict[str, np.ndarray], indices: np.ndarray,
                    *, batch_size: int) -> torch.Tensor:
    device = next(bridge.parameters()).device
    batches = []
    bridge.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            values, mask = bridge(
                torch.as_tensor(
                    arrays["xbar"][selected], dtype=torch.float32, device=device
                ),
                torch.as_tensor(
                    arrays["valid"][selected], dtype=torch.bool, device=device
                ),
            )
            if not bool(mask.all()):
                raise ValueError("v1 diagnostic requires all latent slots to be valid")
            batches.append(values.detach().float().cpu())
    return torch.cat(batches)


def _aggregate(rows: list[dict[str, Any]], alphas: tuple[float, ...],
               *, seed: int) -> dict[str, Any]:
    def column(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    direction_abs = column("gradient_abs_cosine")
    random_mean = column("random_abs_cosine_mean")
    random_p95 = column("random_abs_cosine_p95")
    matched = column("ce_matched")
    shuffled = column("ce_shuffled")
    actual_delta = [s - m for s, m in zip(shuffled, matched)]
    answer_tokens = np.asarray(
        [int(row["answer_tokens"]) for row in rows], dtype=np.float64
    )
    token_weighted_matched = float(
        np.average(np.asarray(matched), weights=answer_tokens)
    )
    token_weighted_shuffled = float(
        np.average(np.asarray(shuffled), weights=answer_tokens)
    )
    result: dict[str, Any] = {
        "rows": len(rows),
        "gradient_direction": {
            "signed_cosine_toward_shuffled": _summary(
                column("gradient_signed_cosine_toward_shuffled")
            ),
            "absolute_cosine_R": _summary(direction_abs),
            "random_absolute_cosine_mean": _summary(random_mean),
            "random_absolute_cosine_p95": _summary(random_p95),
            "R_over_random_mean": _summary([
                value / max(baseline, 1e-12)
                for value, baseline in zip(direction_abs, random_mean)
            ]),
            "fraction_R_above_random_p95": float(np.mean([
                value > baseline for value, baseline in zip(direction_abs, random_p95)
            ])),
            "fraction_gradient_predicts_shuffled_worse": float(np.mean([
                row["gradient_signed_cosine_toward_shuffled"] > 0 for row in rows
            ])),
            "actual_shuffled_minus_matched_ce": {
                **_summary(actual_delta),
                "bootstrap": _bootstrap_mean_ci(actual_delta, seed=seed + 1),
            },
            "first_order_shuffled_minus_matched_ce": _summary(
                column("gradient_dot_matched_to_shuffled")
            ),
            "finite_difference_directional_derivative": _summary(
                column("finite_difference_directional_derivative")
            ),
            "finite_difference_minus_autograd": _summary([
                row["finite_difference_directional_derivative"]
                - row["gradient_dot_matched_to_shuffled"] for row in rows
            ]),
        },
        "residual_amplification": {},
        "common_residual_exchange": {},
    }
    for alpha in alphas:
        key = f"ce_alpha_{alpha:g}"
        values = column(key)
        differences = [value - base for value, base in zip(values, matched)]
        result["residual_amplification"][f"alpha_{alpha:g}"] = {
            "answer_ce": _summary(values),
            "minus_alpha_1_ce": {
                **_summary(differences),
                "bootstrap": _bootstrap_mean_ci(
                    differences, seed=seed + 10 + int(alpha * 10)
                ),
            },
            "fraction_better_than_alpha_1": float(
                np.mean(np.asarray(differences) < 0)
            ),
        }

    exchange_names = (
        "ce_matched_common_shuffled_residual",
        "ce_shuffled_common_matched_residual",
    )
    for offset, name in enumerate(exchange_names):
        values = column(name)
        versus_matched = [value - base for value, base in zip(values, matched)]
        versus_shuffled = [value - base for value, base in zip(values, shuffled)]
        result["common_residual_exchange"][name.removeprefix("ce_")] = {
            "answer_ce": _summary(values),
            "minus_matched_ce": {
                **_summary(versus_matched),
                "bootstrap": _bootstrap_mean_ci(
                    versus_matched, seed=seed + 30 + offset
                ),
            },
            "minus_shuffled_ce": _summary(versus_shuffled),
        }
    result["common_residual_exchange"]["matched_answer_ce"] = _summary(matched)
    result["common_residual_exchange"]["shuffled_answer_ce"] = _summary(shuffled)
    result["common_residual_exchange"]["matched_common_energy_fraction"] = (
        _summary(column("matched_common_energy_fraction"))
    )
    result["token_weighted_control"] = {
        "answer_tokens": int(answer_tokens.sum()),
        "matched_answer_ce": token_weighted_matched,
        "shuffled_answer_ce": token_weighted_shuffled,
        "shuffled_minus_matched_ce": (
            token_weighted_shuffled - token_weighted_matched
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--random-directions", type=int, default=32)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--max-memory-gib", type=int, default=34)
    parser.add_argument("--finite-difference-epsilon", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.sample_size < 2 or args.batch_size < 1 or args.random_directions < 1:
        raise ValueError("sample-size, batch-size, and random-directions are invalid")
    if not 0 < args.finite_difference_epsilon <= 0.01:
        raise ValueError("finite-difference-epsilon must be in (0, 0.01]")
    alphas = tuple(float(value) for value in args.alphas)
    if 1.0 not in alphas or any(value < 0 for value in alphas):
        raise ValueError("alphas must be nonnegative and include 1")
    _visible_gpu_guard("0,1,2")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    dataset_path = Path(args.dataset)
    bridge_path = Path(args.bridge)
    model_path = Path(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays, metadata = _load_dataset(dataset_path)
    validation = _select_split_indices(
        arrays, "validation", retrieval_hit_only=False, limit=0, seed=args.seed
    )
    selected = _trajectory_stratified_sample(
        arrays, validation, size=args.sample_size, seed=args.seed,
    )
    if len(selected) != args.sample_size:
        raise ValueError("validation split is smaller than sample-size")
    partner = _shuffle_partner(selected, arrays["trajectory_id"].astype(str))
    partner_indices = np.asarray([partner[int(index)] for index in selected])

    bridge, bridge_metadata = load_qwen32_bridge(
        bridge_path,
        expected_qformer_sha256=metadata["qformer_artifact_sha256"],
        expected_retrieval_head_sha256=metadata["retrieval_head_artifact_sha256"],
    )
    dataset_hash = file_sha256(dataset_path)
    if bridge_metadata["data_manifest_sha256"] != dataset_hash:
        raise ValueError("Bridge and diagnostic dataset hashes differ")
    print("[diagnostic] hashing complete frozen Reader artifact", flush=True)
    reader_hash = _reader_artifact_sha256(model_path)
    if bridge_metadata["reader_model_sha256"] != reader_hash:
        raise ValueError("Bridge and Reader hashes differ")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    max_memory = {
        index: f"{args.max_memory_gib}GiB"
        for index in range(torch.cuda.device_count())
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="balanced",
        max_memory=max_memory, low_cpu_mem_usage=True, trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    first_device = model.get_input_embeddings().weight.device
    bridge = bridge.to(first_device).eval()
    if not hasattr(bridge, "target_rms"):
        raise ValueError("diagnostic expects the RMS-calibrated v3 Bridge")
    target_rms = float(bridge.target_rms.detach().cpu())

    print("[diagnostic] computing the global validation common direction", flush=True)
    validation_outputs = _bridge_outputs(
        bridge, arrays, validation, batch_size=max(args.batch_size, 16)
    )
    # Preserve the 32-token positional structure: alpha=0 should be the
    # validation-mean soft prompt at each slot, not one vector repeated in all
    # slots.  The exchange experiment still uses the single dominant global
    # mean direction, averaged across those slot-specific centroids.
    common_mean = validation_outputs.mean(dim=0)
    common_direction = F.normalize(common_mean.mean(dim=0), dim=0)
    mean_pair_cosine = float(F.cosine_similarity(
        validation_outputs[:min(512, len(validation_outputs))].mean(1),
        validation_outputs[-min(512, len(validation_outputs)):].flip(0).mean(1),
        dim=-1,
    ).mean())
    del validation_outputs

    matched_outputs = _bridge_outputs(
        bridge, arrays, selected, batch_size=args.batch_size
    )
    shuffled_outputs = _bridge_outputs(
        bridge, arrays, partner_indices, batch_size=args.batch_size
    )
    rows: list[dict[str, Any]] = []
    rng = torch.Generator(device="cpu").manual_seed(args.seed + 1000)
    print(
        f"[diagnostic] evaluating {len(selected)} validation rows in "
        f"batches of {args.batch_size}", flush=True,
    )
    for start in range(0, len(selected), args.batch_size):
        stop = min(start + args.batch_size, len(selected))
        batch_indices = selected[start:stop]
        batch_partners = partner_indices[start:stop]
        questions = [str(arrays["question"][index]) for index in batch_indices]
        answers = [str(arrays["answer"][index]) for index in batch_indices]
        matched_cpu = matched_outputs[start:stop]
        shuffled_cpu = shuffled_outputs[start:stop]

        # One backward pass yields a separate input-gradient vector per row.
        matched_leaf = matched_cpu.to(
            first_device, dtype=model.get_input_embeddings().weight.dtype
        ).detach().requires_grad_(True)
        kwargs, _ = _build_batch(
            model, tokenizer, matched_leaf, questions, answers,
            args.enable_thinking,
        )
        output = model(**kwargs)
        matched_ce = _per_row_answer_ce(output.logits, kwargs["labels"])
        matched_ce.sum().backward()
        gradient = matched_leaf.grad.detach().float().cpu()
        matched_values = matched_ce.detach().float().cpu()
        del output, kwargs, matched_leaf, matched_ce

        arms: dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            arms["shuffled"] = shuffled_cpu
            arms["finite_difference_plus"] = matched_cpu + (
                args.finite_difference_epsilon
                * (shuffled_cpu - matched_cpu)
            )
            arms["finite_difference_minus"] = matched_cpu - (
                args.finite_difference_epsilon
                * (shuffled_cpu - matched_cpu)
            )
            for alpha in alphas:
                if alpha == 1.0:
                    continue
                amplified = common_mean[None] + alpha * (
                    matched_cpu - common_mean[None]
                )
                arms[f"alpha_{alpha:g}"] = _rms_calibrate(
                    amplified, target_rms
                )
            matched_common, matched_residual = _common_residual(
                matched_cpu, common_direction
            )
            shuffled_common, shuffled_residual = _common_residual(
                shuffled_cpu, common_direction
            )
            arms["matched_common_shuffled_residual"] = _rms_calibrate(
                matched_common + shuffled_residual, target_rms
            )
            arms["shuffled_common_matched_residual"] = _rms_calibrate(
                shuffled_common + matched_residual, target_rms
            )

        ce_by_arm: dict[str, torch.Tensor] = {}
        with torch.inference_mode():
            for name, values in arms.items():
                ce_by_arm[name] = _reader_ce(
                    model, tokenizer, values.to(first_device), questions, answers,
                    args.enable_thinking,
                ).detach().float().cpu()

        for local, (index, source) in enumerate(zip(batch_indices, batch_partners)):
            grad = gradient[local].reshape(-1)
            displacement = (shuffled_cpu[local] - matched_cpu[local]).reshape(-1)
            grad_norm = float(grad.norm())
            displacement_norm = float(displacement.norm())
            denominator = max(grad_norm * displacement_norm, 1e-20)
            dot = float(torch.dot(grad, displacement))
            signed = dot / denominator
            random_values = []
            for _ in range(args.random_directions):
                random_direction = torch.randn(
                    displacement.shape, generator=rng, dtype=torch.float32
                )
                random_values.append(float(
                    torch.dot(grad, random_direction).abs()
                    / max(grad_norm * float(random_direction.norm()), 1e-20)
                ))
            random_array = np.asarray(random_values)
            row: dict[str, Any] = {
                "row_index": int(index),
                "shuffled_source_index": int(source),
                "trajectory_id": str(arrays["trajectory_id"][index]),
                "shuffled_trajectory_id": str(arrays["trajectory_id"][source]),
                "answer_tokens": int(_row_parts(
                    tokenizer, questions[local], answers[local],
                    args.enable_thinking,
                )[2].__len__()),
                "ce_matched": float(matched_values[local]),
                "ce_shuffled": float(ce_by_arm["shuffled"][local]),
                "gradient_norm": grad_norm,
                "matched_to_shuffled_norm": displacement_norm,
                "gradient_dot_matched_to_shuffled": dot,
                "gradient_signed_cosine_toward_shuffled": signed,
                "gradient_abs_cosine": abs(signed),
                "finite_difference_directional_derivative": float(
                    (
                        ce_by_arm["finite_difference_plus"][local]
                        - ce_by_arm["finite_difference_minus"][local]
                    ) / (2 * args.finite_difference_epsilon)
                ),
                "matched_common_energy_fraction": float(
                    matched_common[local].square().sum()
                    / matched_cpu[local].float().square().sum().clamp_min(1e-20)
                ),
                "random_abs_cosine_mean": float(random_array.mean()),
                "random_abs_cosine_p95": float(np.quantile(random_array, 0.95)),
                "random_abs_cosines": random_values,
            }
            for alpha in alphas:
                row[f"ce_alpha_{alpha:g}"] = (
                    float(matched_values[local]) if alpha == 1.0
                    else float(ce_by_arm[f"alpha_{alpha:g}"][local])
                )
            for name in (
                "matched_common_shuffled_residual",
                "shuffled_common_matched_residual",
            ):
                row[f"ce_{name}"] = float(ce_by_arm[name][local])
            rows.append(row)

        partial = {
            "protocol": PROTOCOL,
            "status": "running",
            "completed_rows": len(rows),
            "requested_rows": len(selected),
            "rows": rows,
        }
        _atomic_json(output_path.with_suffix(".partial.json"), partial)
        print(
            f"[diagnostic] completed {len(rows)}/{len(selected)} rows "
            f"elapsed={time.time() - started:.1f}s", flush=True,
        )
        torch.cuda.empty_cache()

    aggregate = _aggregate(rows, alphas, seed=args.seed)
    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "scope": {
            "split": "validation",
            "official_ama_test_included": False,
            "validation_population_rows": int(len(validation)),
            "sample_rows": int(len(selected)),
            "sample_trajectories": int(len(set(
                arrays["trajectory_id"][selected].astype(str)
            ))),
            "sampling": "one_random_row_per_distinct_random_validation_trajectory",
            "sample_seed": int(args.seed),
            "batch_size": int(args.batch_size),
            "random_directions_per_row": int(args.random_directions),
            "finite_difference_epsilon": float(args.finite_difference_epsilon),
            "alphas": list(alphas),
        },
        "definitions": {
            "gradient_direction": (
                "cosine between answer-CE gradient at matched latent and the "
                "matched-to-shuffled latent displacement"
            ),
            "residual_amplification": (
                "RMSCalibrate(slotwise_validation_mean + alpha * "
                "(matched - slotwise_validation_mean))"
            ),
            "common_residual_exchange": (
                "exchange components parallel and orthogonal to the normalized "
                "global validation Bridge-output mean direction"
            ),
        },
        "provenance": {
            "dataset": str(dataset_path.resolve()),
            "dataset_sha256": dataset_hash,
            "bridge": str(bridge_path.resolve()),
            "bridge_sha256": file_sha256(bridge_path),
            "reader": str(model_path.resolve()),
            "reader_model_sha256": reader_hash,
            "qformer_artifact_sha256": metadata["qformer_artifact_sha256"],
            "retrieval_head_artifact_sha256": metadata[
                "retrieval_head_artifact_sha256"
            ],
        },
        "bridge_output_geometry": {
            "target_reader_embedding_rms": target_rms,
            "slotwise_common_mean_norm_mean": float(
                common_mean.norm(dim=-1).mean()
            ),
            "paired_pooled_output_cosine_mean": mean_pair_cosine,
        },
        "aggregate": aggregate,
        "rows": rows,
    }
    _atomic_json(output_path, result)
    partial_path = output_path.with_suffix(".partial.json")
    if partial_path.exists():
        partial_path.unlink()
    output_path.with_suffix(".complete").touch()
    print(json.dumps({
        "status": "complete",
        "output": str(output_path),
        "elapsed_seconds": result["elapsed_seconds"],
        "aggregate": aggregate,
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
