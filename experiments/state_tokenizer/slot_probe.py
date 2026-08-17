"""Train and evaluate v2 slot-aware probes and leakage baselines."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .common import iter_jsonl, write_json
from .evaluate_probes import average_precision
from .ragged_store import (
    FixedRepresentationStore,
    RaggedFullHStore,
    pad_token_batch,
)
from .slot_reader import MetadataReader, SlotAwareReader
from .v2_data import DYNAMIC_LABELS, STATIC_LABELS


REPRESENTATIONS = (
    "full_h", "instruction_only", "y64", "y32", "x64", "key64", "key64_pca",
    "key64_static", "key64_static_pca", "task_only", "task_step",
)
OBJECTIVES = ("full", "value_only", "dynamic_state_only")
DYNAMIC_STATE_LABELS = (
    "checkbox_0_checked", "checkbox_1_checked", "checkbox_2_checked",
    "textbox_0_nonempty", "textbox_1_nonempty", "textbox_2_nonempty",
    "textbox_0_focused", "textbox_1_focused", "textbox_2_focused",
)
DYNAMIC_STATE_INDICES = tuple(DYNAMIC_LABELS.index(label) for label in DYNAMIC_STATE_LABELS)


@dataclass
class TargetBundle:
    dynamic: np.ndarray
    dynamic_mask: np.ndarray
    static: np.ndarray
    digits: np.ndarray
    dom_words: np.ndarray
    overlap_words: np.ndarray
    task: np.ndarray
    step: np.ndarray
    global_indices: np.ndarray
    splits: dict[str, np.ndarray]
    task_names: list[str]
    dom_vocab: list[str]
    overlap_vocab: list[str]
    eligible_dynamic: list[str]


def build_targets(records: list[dict], summary: dict) -> TargetBundle:
    tasks = sorted({record["task"] for record in records})
    task_index = {task: index for index, task in enumerate(tasks)}
    dom_vocab = list(summary["dom_only_vocab"])
    overlap_vocab = list(summary["instruction_overlap_vocab"])
    dom_index = {word: index for index, word in enumerate(dom_vocab)}
    overlap_index = {word: index for index, word in enumerate(overlap_vocab)}
    dynamic = np.asarray([
        [int(record["v2"]["dynamic"][label]) for label in DYNAMIC_LABELS]
        for record in records
    ], np.int64)
    dynamic_mask = np.asarray([
        [int(record["v2"]["dynamic_mask"][label]) for label in DYNAMIC_LABELS]
        for record in records
    ], np.bool_)
    static = np.asarray([
        [int(record["v2"]["static"][label]) for label in STATIC_LABELS]
        for record in records
    ], np.int64)
    digits = np.asarray([record["v2"]["random_digits"] for record in records], np.int64)
    dom_words = np.zeros((len(records), len(dom_vocab)), np.int64)
    overlap_words = np.zeros((len(records), len(overlap_vocab)), np.int64)
    for row, record in enumerate(records):
        for word in record["v2"]["dom_only_words"]:
            if word in dom_index:
                dom_words[row, dom_index[word]] = 1
        for word in record["v2"]["instruction_overlap_words"]:
            if word in overlap_index:
                overlap_words[row, overlap_index[word]] = 1
    splits = {
        split: np.asarray([i for i, record in enumerate(records) if record["split"] == split], np.int64)
        for split in ("train", "validation", "test")
    }
    return TargetBundle(
        dynamic=dynamic,
        dynamic_mask=dynamic_mask,
        static=static,
        digits=digits,
        dom_words=dom_words,
        overlap_words=overlap_words,
        task=np.asarray([task_index[record["task"]] for record in records], np.int64),
        step=np.asarray([min(int(record["step"]), 3) for record in records], np.int64),
        global_indices=np.asarray([record["global_index"] for record in records], np.int64),
        splits=splits,
        task_names=tasks,
        dom_vocab=dom_vocab,
        overlap_vocab=overlap_vocab,
        eligible_dynamic=[
            label for label in DYNAMIC_LABELS
            if summary["dynamic_coverage"][label]["eligible"]
        ],
    )


class TokenProvider:
    def __init__(
        self, representation: str, records: list[dict], *, features: str, full_h: str | None
    ):
        self.representation = representation
        self.records = records
        if representation in {"full_h", "instruction_only"}:
            if not full_h:
                raise ValueError("--full-h is required")
            self.store = RaggedFullHStore(full_h, len(records))
            lengths = self.store.lengths()
            if representation == "instruction_only":
                lengths = self.store.modality_lengths()[:, 2]
            self.length_values = lengths
            self.input_dim = 4096
            self.max_examples = 8
            self.max_tokens = 4096
        else:
            self.store = FixedRepresentationStore(features, representation)
            self.length_values = np.full((len(records),), self.store.slots, np.int32)
            self.input_dim = self.store.width
            self.max_examples = 64
            self.max_tokens = 8192

    def get(self, row: int):
        if self.representation in {"full_h", "instruction_only"}:
            if self.representation == "instruction_only":
                return self.store.get_modality(row, 2)
            return self.store.get(row)
        return self.store.get(int(self.records[row]["global_index"]))

    def batches(self, rows: Sequence[int], rng: np.random.Generator | None = None):
        order = np.asarray(rows, np.int64).copy()
        if rng is not None:
            rng.shuffle(order)
        batch: list[int] = []
        maximum = 0
        for row in order:
            length = int(self.length_values[int(row)])
            new_maximum = max(maximum, length)
            if batch and (
                len(batch) >= self.max_examples
                or new_maximum * (len(batch) + 1) > self.max_tokens
            ):
                yield np.asarray(batch, np.int64)
                batch, maximum = [], 0
            batch.append(int(row))
            maximum = max(maximum, length)
        if batch:
            yield np.asarray(batch, np.int64)

    def collate(self, rows: Sequence[int], device: torch.device):
        values = pad_token_batch([self.get(int(row)) for row in rows])
        return tuple(value.to(device, non_blocking=True) for value in values)


def metadata_features(targets: TargetBundle, representation: str) -> np.ndarray:
    task = np.eye(len(targets.task_names), dtype=np.float32)[targets.task]
    if representation == "task_only":
        return task
    if representation == "task_step":
        step = np.eye(4, dtype=np.float32)[targets.step]
        return np.concatenate((task, step), axis=1)
    raise ValueError(representation)


def _pos_weight(values: np.ndarray, masks: np.ndarray | None = None) -> torch.Tensor:
    if values.shape[1] == 0:
        return torch.empty((0,), dtype=torch.float32)
    mask = np.ones_like(values, dtype=bool) if masks is None else masks.astype(bool)
    positives = (values * mask).sum(axis=0)
    total = mask.sum(axis=0)
    negatives = total - positives
    return torch.from_numpy(np.clip(negatives / np.maximum(positives, 1), 1, 20).astype(np.float32))


def loss_weights(targets: TargetBundle, train_rows: np.ndarray, device: torch.device):
    return {
        "dynamic": _pos_weight(
            targets.dynamic[train_rows], targets.dynamic_mask[train_rows]
        ).to(device),
        "static": _pos_weight(targets.static[train_rows]).to(device),
        "dom_words": _pos_weight(targets.dom_words[train_rows]).to(device),
        "overlap_words": _pos_weight(targets.overlap_words[train_rows]).to(device),
    }


def _masked_bce(logits, values, mask, pos_weight):
    loss = F.binary_cross_entropy_with_logits(
        logits, values, pos_weight=pos_weight, reduction="none"
    )
    mask = mask.to(loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def compute_loss(
    output: dict[str, torch.Tensor], targets: TargetBundle, rows: np.ndarray,
    weights: dict[str, torch.Tensor], device: torch.device, objective: str = "full",
) -> tuple[torch.Tensor, dict[str, float]]:
    digits = torch.from_numpy(targets.digits[rows]).to(device)
    valid_digits = digits >= 0
    if objective == "value_only":
        if not bool(valid_digits.all()):
            raise ValueError("value_only batches must contain only valid five-digit targets")
        digit_loss = F.cross_entropy(output["digits"].reshape(-1, 10), digits.reshape(-1))
        return digit_loss, {"digits": float(digit_loss.detach())}
    if objective == "dynamic_state_only":
        indices = list(DYNAMIC_STATE_INDICES)
        dynamic = torch.from_numpy(targets.dynamic[rows][:, indices]).to(device).float()
        dynamic_mask = torch.from_numpy(targets.dynamic_mask[rows][:, indices]).to(device)
        dynamic_loss = _masked_bce(
            output["dynamic"][:, indices], dynamic, dynamic_mask, weights["dynamic"][indices]
        )
        return dynamic_loss, {"dynamic_state": float(dynamic_loss.detach())}
    if objective != "full":
        raise ValueError(objective)
    dynamic = torch.from_numpy(targets.dynamic[rows]).to(device).float()
    dynamic_mask = torch.from_numpy(targets.dynamic_mask[rows]).to(device)
    static = torch.from_numpy(targets.static[rows]).to(device).float()
    task = torch.from_numpy(targets.task[rows]).to(device)
    losses = {
        "dynamic": _masked_bce(output["dynamic"], dynamic, dynamic_mask, weights["dynamic"]),
        "static": _masked_bce(
            output["static"], static, torch.ones_like(static, dtype=torch.bool), weights["static"]
        ),
        "task": F.cross_entropy(output["task"], task),
    }
    if bool(valid_digits.any()):
        digit_logits = output["digits"].reshape(-1, 10)
        digit_targets = digits.reshape(-1)
        losses["digits"] = F.cross_entropy(
            digit_logits[valid_digits.reshape(-1)], digit_targets[valid_digits.reshape(-1)]
        )
    if targets.dom_words.shape[1]:
        dom = torch.from_numpy(targets.dom_words[rows]).to(device).float()
        losses["dom_words"] = _masked_bce(
            output["dom_words"], dom, torch.ones_like(dom, dtype=torch.bool), weights["dom_words"]
        )
    if targets.overlap_words.shape[1]:
        overlap = torch.from_numpy(targets.overlap_words[rows]).to(device).float()
        losses["overlap_words"] = _masked_bce(
            output["overlap_words"], overlap, torch.ones_like(overlap, dtype=torch.bool),
            weights["overlap_words"],
        )
    coefficients = {
        "dynamic": 1.0, "digits": 1.0, "dom_words": 0.25,
        "overlap_words": 0.1, "static": 0.1, "task": 0.1,
    }
    total = sum(coefficients[name] * loss for name, loss in losses.items())
    return total, {name: float(value.detach()) for name, value in losses.items()}


def _evaluate_binary(
    labels: np.ndarray, scores: np.ndarray, masks: np.ndarray, names: list[str],
    task_ids: np.ndarray | None = None,
) -> dict:
    per_label = {}
    macro = []
    for column, name in enumerate(names):
        selected = masks[:, column].astype(bool)
        values = labels[selected, column]
        predictions = scores[selected, column]
        ap = average_precision(values, predictions) if len(values) else float("nan")
        task_aps = {}
        task_prevalences = {}
        if task_ids is not None:
            for task in np.unique(task_ids[selected]):
                task_rows = selected & (task_ids == task)
                task_values = labels[task_rows, column]
                if int(task_values.sum()) >= 5 and int((1 - task_values).sum()) >= 5:
                    task_aps[str(int(task))] = average_precision(task_values, scores[task_rows, column])
                    task_prevalences[str(int(task))] = float(task_values.mean())
        task_macro = float(np.mean(list(task_aps.values()))) if task_aps else float("nan")
        task_prevalence = (
            float(np.mean(list(task_prevalences.values())))
            if task_prevalences else float("nan")
        )
        per_label[name] = {
            "average_precision": ap,
            "task_macro_average_precision": task_macro,
            "task_average_precision": task_aps,
            "task_prevalence": task_prevalence,
            "per_task_prevalence": task_prevalences,
            "prevalence": float(values.mean()) if len(values) else float("nan"),
            "examples": int(len(values)),
            "positives": int(values.sum()),
            "negatives": int(len(values) - values.sum()),
        }
        if np.isfinite(task_macro):
            macro.append(task_macro)
    return {
        "task_macro_average_precision": float(np.mean(macro)) if macro else float("nan"),
        "per_label": per_label,
    }


def evaluate_predictions(
    targets: TargetBundle, rows: np.ndarray, predictions: dict[str, np.ndarray]
) -> dict:
    dynamic = _evaluate_binary(
        targets.dynamic[rows], predictions["dynamic"], targets.dynamic_mask[rows],
        list(DYNAMIC_LABELS), targets.task[rows],
    )
    eligible_values = [
        dynamic["per_label"][label]["task_macro_average_precision"]
        for label in targets.eligible_dynamic
        if np.isfinite(dynamic["per_label"][label]["task_macro_average_precision"])
    ]
    dynamic["eligible_labels"] = targets.eligible_dynamic
    dynamic["eligible_task_macro_average_precision"] = (
        float(np.mean(eligible_values)) if eligible_values else float("nan")
    )
    state_eligible_labels = [
        label for label in targets.eligible_dynamic if label in DYNAMIC_STATE_LABELS
    ]
    state_values = [
        dynamic["per_label"][label]["task_macro_average_precision"]
        for label in state_eligible_labels
        if np.isfinite(dynamic["per_label"][label]["task_macro_average_precision"])
    ]
    dynamic["state_labels"] = list(DYNAMIC_STATE_LABELS)
    dynamic["state_eligible_labels"] = state_eligible_labels
    dynamic["state_eligible_task_macro_average_precision"] = (
        float(np.mean(state_values)) if state_values else float("nan")
    )
    static = _evaluate_binary(
        targets.static[rows], predictions["static"],
        np.ones_like(targets.static[rows], dtype=bool), list(STATIC_LABELS),
    )
    value_rows = np.all(targets.digits[rows] >= 0, axis=1)
    digit_labels = targets.digits[rows][value_rows]
    digit_scores = predictions["digits"][value_rows]
    positions = []
    for position in range(5):
        labels = digit_labels[:, position]
        scores = digit_scores[:, position]
        accuracy = float((scores.argmax(axis=1) == labels).mean()) if len(labels) else float("nan")
        per_digit = {
            str(digit): average_precision((labels == digit).astype(np.int64), scores[:, digit])
            for digit in range(10) if int((labels == digit).sum()) > 0
        }
        positions.append({
            "position": position,
            "accuracy": accuracy,
            "macro_one_vs_rest_average_precision": float(np.nanmean(list(per_digit.values()))),
            "per_digit_average_precision": per_digit,
        })
    value = {
        "examples": int(len(digit_labels)),
        "position_accuracy": [item["accuracy"] for item in positions],
        "macro_position_accuracy": float(np.nanmean([item["accuracy"] for item in positions])),
        "exact_value_accuracy": (
            float(np.all(digit_scores.argmax(axis=2) == digit_labels, axis=1).mean())
            if len(digit_labels) else float("nan")
        ),
        "positions": positions,
    }
    result = {
        "dynamic": dynamic,
        "value": value,
        "static": static,
        "task": {"accuracy": float((predictions["task"].argmax(axis=1) == targets.task[rows]).mean())},
    }
    for key, values, names in (
        ("dom_words", targets.dom_words, targets.dom_vocab),
        ("overlap_words", targets.overlap_words, targets.overlap_vocab),
    ):
        if len(names):
            family = _evaluate_binary(
                values[rows], predictions[key], np.ones_like(values[rows], dtype=bool), names
            )
            aps = [item["average_precision"] for item in family["per_label"].values()]
            family["macro_average_precision"] = float(np.nanmean(aps))
            result[key] = family
    return result


def validation_score(metrics: dict, objective: str = "full") -> float:
    if objective == "value_only":
        return float(metrics["value"]["macro_position_accuracy"])
    if objective == "dynamic_state_only":
        return float(metrics["dynamic"]["state_eligible_task_macro_average_precision"])
    values = [
        metrics["dynamic"]["eligible_task_macro_average_precision"],
        metrics["value"]["macro_position_accuracy"],
    ]
    return float(np.nanmean(values))


def _empty_predictions(targets: TargetBundle, count: int) -> dict[str, np.ndarray]:
    return {
        "dynamic": np.empty((count, len(DYNAMIC_LABELS)), np.float32),
        "static": np.empty((count, len(STATIC_LABELS)), np.float32),
        "digits": np.empty((count, 5, 10), np.float32),
        "task": np.empty((count, len(targets.task_names)), np.float32),
        "dom_words": np.empty((count, len(targets.dom_vocab)), np.float32),
        "overlap_words": np.empty((count, len(targets.overlap_vocab)), np.float32),
    }


def predict(
    model, representation: str, provider: TokenProvider | None, metadata: np.ndarray | None,
    rows: np.ndarray, targets: TargetBundle, device: torch.device,
) -> dict[str, np.ndarray]:
    destination = _empty_predictions(targets, len(rows))
    model.eval()
    cursor = 0
    with torch.inference_mode():
        if provider is not None:
            batches = provider.batches(rows)
        else:
            batches = (rows[start:start + 256] for start in range(0, len(rows), 256))
        for batch_rows in batches:
            if provider is not None:
                inputs = provider.collate(batch_rows, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(*inputs)
            else:
                features = torch.from_numpy(metadata[batch_rows]).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(features)
            size = len(batch_rows)
            for key in destination:
                destination[key][cursor:cursor + size] = output[key].float().cpu().numpy()
            cursor += size
    if cursor != len(rows):
        raise RuntimeError(f"prediction row mismatch: {cursor} != {len(rows)}")
    return destination


def train(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = list(iter_jsonl(args.records))
    summary = json.loads(Path(args.records).with_suffix(".summary.json").read_text())
    targets = build_targets(records, summary)
    train_rows = targets.splits["train"]
    validation_rows = targets.splits["validation"]
    if args.objective == "value_only":
        valid_values = np.all(targets.digits >= 0, axis=1)
        train_rows = train_rows[valid_values[train_rows]]
        validation_rows = validation_rows[valid_values[validation_rows]]
        if not len(train_rows) or not len(validation_rows):
            raise ValueError("value_only objective has an empty train or validation split")
    if args.max_train:
        train_rows = train_rows[:args.max_train]
    if args.overfit:
        validation_rows = train_rows
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    provider = None
    metadata = None
    head_kwargs = {
        "dynamic": len(DYNAMIC_LABELS), "static": len(STATIC_LABELS),
        "dom_words": len(targets.dom_vocab), "overlap_words": len(targets.overlap_vocab),
        "tasks": len(targets.task_names),
    }
    if args.representation in {"task_only", "task_step"}:
        metadata = metadata_features(targets, args.representation)
        model = MetadataReader(metadata.shape[1], **head_kwargs).to(device)
        max_examples = 256
    else:
        provider = TokenProvider(
            args.representation, records, features=args.features, full_h=args.full_h
        )
        model = SlotAwareReader(provider.input_dim, **head_kwargs).to(device)
        max_examples = provider.max_examples
    weights = loss_weights(targets, train_rows, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_score = -float("inf")
    best_state = None
    stale = 0
    history = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        losses = []
        family_sums: dict[str, list[float]] = {}
        if provider is not None:
            batches = provider.batches(train_rows, rng)
        else:
            order = rng.permutation(train_rows)
            batches = (order[start:start + max_examples] for start in range(0, len(order), max_examples))
        for batch_rows in batches:
            if provider is not None:
                inputs = provider.collate(batch_rows, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(*inputs)
                    loss, details = compute_loss(
                        output, targets, batch_rows, weights, device, args.objective
                    )
            else:
                values = torch.from_numpy(metadata[batch_rows]).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(values)
                    loss, details = compute_loss(
                        output, targets, batch_rows, weights, device, args.objective
                    )
            (loss * (len(batch_rows) / args.effective_batch_size)).backward()
            accumulated += len(batch_rows)
            losses.append(float(loss.detach()))
            for key, value in details.items():
                family_sums.setdefault(key, []).append(value)
            if accumulated >= args.effective_batch_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated = 0
        if accumulated:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        val_predictions = predict(
            model, args.representation, provider, metadata, validation_rows, targets, device
        )
        val_metrics = evaluate_predictions(targets, validation_rows, val_predictions)
        score = validation_score(val_metrics, args.objective)
        item = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "validation_score": score,
            "validation_dynamic_ap": val_metrics["dynamic"]["eligible_task_macro_average_precision"],
            "validation_dynamic_state_ap": val_metrics["dynamic"]["state_eligible_task_macro_average_precision"],
            "validation_value_accuracy": val_metrics["value"]["macro_position_accuracy"],
            "family_losses": {key: float(np.mean(values)) for key, values in family_sums.items()},
        }
        history.append(item)
        logging.info("%s seed=%d %s", args.representation, args.seed, item)
        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("probe produced no checkpoint")
    model.load_state_dict(best_state)
    evaluation_rows = train_rows if args.overfit else targets.splits["test"]
    if args.objective == "value_only" and not args.overfit:
        evaluation_rows = evaluation_rows[np.all(targets.digits[evaluation_rows] >= 0, axis=1)]
    predictions = predict(
        model, args.representation, provider, metadata, evaluation_rows, targets, device
    )
    metrics = evaluate_predictions(targets, evaluation_rows, predictions)
    result = {
        "protocol": (
            "slot_aware_v3_fixed_prompt_dynamic"
            if args.objective == "dynamic_state_only"
            else (
                (
                    "slot_aware_v3_fixed_prompt_value_only"
                    if summary.get("instruction_protocol") == "fixed_task_independent_observation_v1"
                    else "slot_aware_v2_value_only"
                )
                if args.objective == "value_only" else "slot_aware_v2"
            )
        ),
        "representation": args.representation,
        "seed": args.seed,
        "best_validation_score": best_score,
        "epochs_ran": len(history),
        "history": history,
        "test": metrics,
        "config": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "effective_batch_size": args.effective_batch_size,
            "max_train": args.max_train,
            "overfit": args.overfit,
            "reader_hidden": 256,
            "reader_queries": 1,
            "reader_cross_attention_layers": 1,
            "reader_attention_heads": 4,
            "position_contract": (
                "explicit source-aware positions with valid mask"
                if args.representation.startswith("key64")
                else "(i+0.5)/N_modality"
            ),
            "cpu_threads": args.cpu_threads,
            "objective": args.objective,
            "train_examples": int(len(train_rows)),
            "validation_examples": int(len(validation_rows)),
            "test_examples": int(len(evaluation_rows)),
        },
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.representation}-seed{args.seed}"
    write_json(output / f"result-{stem}.json", result)
    torch.save({"state_dict": best_state, "config": result["config"]}, output / f"checkpoint-{stem}.pt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--full-h")
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--objective", choices=OBJECTIVES, default="full")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--effective-batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    result = train(args)
    print(json.dumps({
        "representation": result["representation"], "seed": result["seed"],
        "epochs_ran": result["epochs_ran"], "best_validation_score": result["best_validation_score"],
        "dynamic_ap": result["test"]["dynamic"]["eligible_task_macro_average_precision"],
        "dynamic_state_ap": result["test"]["dynamic"]["state_eligible_task_macro_average_precision"],
        "value_accuracy": result["test"]["value"]["macro_position_accuracy"],
        "exact_value_accuracy": result["test"]["value"]["exact_value_accuracy"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
