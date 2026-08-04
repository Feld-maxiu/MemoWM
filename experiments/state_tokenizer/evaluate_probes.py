"""Train matched linear probes and enforce the Y64 stop/go gate."""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import STATE_LABELS, iter_jsonl, write_json
from .feature_store import materialize_probe_features


def average_precision(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(targets, np.int64)
    positives = int(targets.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores, np.float64), kind="stable")
    ranked = targets[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / positives)


def _feature_stats(features: np.ndarray, indices: np.ndarray, batch_size: int = 256):
    width = features.shape[1]
    total = np.zeros((width,), np.float64)
    square = np.zeros((width,), np.float64)
    count = 0
    for start in range(0, len(indices), batch_size):
        batch = np.asarray(features[indices[start:start + batch_size]], np.float32)
        total += batch.sum(axis=0, dtype=np.float64)
        square += np.square(batch, dtype=np.float32).sum(axis=0, dtype=np.float64)
        count += len(batch)
    mean = total / max(count, 1)
    variance = np.maximum(square / max(count, 1) - mean * mean, 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _predict(model, features, indices, mean, std, device, batch_size=512):
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            values = torch.from_numpy(
                np.asarray(features[batch_indices], np.float32)
            ).to(device)
            values = (values - mean) / std
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs.append(model(values).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    split_indices: dict[str, np.ndarray],
    *,
    kind: str,
    device: torch.device,
    seed: int,
    epochs: int,
    patience: int,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_idx = split_indices["train"]
    val_idx = split_indices["validation"]
    test_idx = split_indices["test"]
    mean_np, std_np = _feature_stats(features, train_idx)
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    if kind == "multiclass":
        output_dim = int(labels.max()) + 1
        if output_dim < 2 or int(labels.min()) < 0:
            raise ValueError(
                f"invalid multiclass target range: [{int(labels.min())}, {int(labels.max())}]"
            )
    else:
        output_dim = int(labels.shape[1])
    model = nn.Linear(features.shape[1], output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if kind == "multiclass":
        loss_fn = nn.CrossEntropyLoss()
    elif kind == "multilabel":
        positives = labels[train_idx].sum(axis=0)
        negatives = len(train_idx) - positives
        pos_weight = np.clip(negatives / np.maximum(positives, 1), 1.0, 20.0)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.from_numpy(pos_weight.astype(np.float32)).to(device)
        )
    else:
        raise ValueError(kind)

    best_score = -float("inf")
    best_state = None
    stale = 0
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order), batch_size):
            batch_idx = order[start:start + batch_size]
            values = torch.from_numpy(np.asarray(features[batch_idx], np.float32)).to(device)
            values = (values - mean) / std
            target_np = labels[batch_idx]
            if kind == "multiclass":
                target = torch.from_numpy(target_np[:, 0].astype(np.int64)).to(device)
            else:
                target = torch.from_numpy(target_np.astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(values)
                loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_logits = _predict(model, features, val_idx, mean, std, device)
        if kind == "multiclass":
            score = float((val_logits.argmax(axis=1) == labels[val_idx, 0]).mean())
        else:
            aps = [
                average_precision(labels[val_idx, column], val_logits[:, column])
                for column in range(output_dim)
            ]
            score = float(np.nanmean(aps))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_score": score})
        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("probe did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_logits = _predict(model, features, test_idx, mean, std, device)
    details = {
        "best_validation_score": best_score,
        "epochs_ran": len(history),
        "history": history,
    }
    return test_logits, details


def build_targets(records: list[dict]):
    tasks = sorted({record["task"] for record in records})
    task_to_index = {task: index for index, task in enumerate(tasks)}
    task = np.asarray([[task_to_index[record["task"]]] for record in records], np.int64)
    state = np.asarray([
        [int(record["probe"]["state"].get(label, False)) for label in STATE_LABELS]
        for record in records
    ], np.int64)
    train_words = Counter()
    for record in records:
        if record["split"] == "train":
            train_words.update(set(record["probe"]["visible_words"]))
    vocab = [word for word, _ in sorted(train_words.items(), key=lambda item: (-item[1], item[0]))[:128]]
    word_to_index = {word: index for index, word in enumerate(vocab)}
    bow = np.zeros((len(records), len(vocab)), np.int64)
    for row, record in enumerate(records):
        for word in set(record["probe"]["visible_words"]):
            if word in word_to_index:
                bow[row, word_to_index[word]] = 1
    splits = {
        name: np.asarray([index for index, record in enumerate(records) if record["split"] == name], np.int64)
        for name in ("train", "validation", "test")
    }
    if any(len(indices) == 0 for indices in splits.values()):
        raise ValueError(f"empty split: { {key: len(value) for key, value in splits.items()} }")
    return {
        "task": task,
        "state": state,
        "bow": bow,
        "task_names": tasks,
        "state_names": list(STATE_LABELS),
        "bow_names": vocab,
        "splits": splits,
    }


def evaluate_representation(args, records, targets, representation: str) -> dict:
    output = Path(args.output)
    result_path = output / f"probe-{representation}.json"
    if args.reuse and result_path.exists():
        return json.loads(result_path.read_text())
    probe_input = materialize_probe_features(
        args.features,
        representation,
        len(records),
        output / "probe-inputs" / f"{representation}-fp16.npy",
    )
    features = np.load(probe_input, mmap_mode="r")
    device = torch.device(args.device)
    result = {"representation": representation, "families": {}}
    family_specs = (
        ("task", "multiclass", targets["task_names"]),
        ("state", "multilabel", targets["state_names"]),
        ("bow", "multilabel", targets["bow_names"]),
    )
    for offset, (family, kind, names) in enumerate(family_specs):
        logits, train_details = train_probe(
            features,
            targets[family],
            targets["splits"],
            kind=kind,
            device=device,
            seed=args.seed + offset,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )
        test_labels = targets[family][targets["splits"]["test"]]
        if kind == "multiclass":
            accuracy = float((logits.argmax(axis=1) == test_labels[:, 0]).mean())
            family_result = {"accuracy": accuracy, "training": train_details}
        else:
            per_label = {}
            aps = []
            for column, name in enumerate(names):
                values = test_labels[:, column]
                ap = average_precision(values, logits[:, column])
                prevalence = float(values.mean())
                positives = int(values.sum())
                negatives = int(len(values) - positives)
                per_label[name] = {
                    "average_precision": ap,
                    "prevalence": prevalence,
                    "positives": positives,
                    "negatives": negatives,
                }
                aps.append(ap)
            family_result = {
                "macro_average_precision": float(np.nanmean(aps)),
                "per_label": per_label,
                "training": train_details,
            }
        result["families"][family] = family_result
        logging.info("%s/%s: %s", representation, family, {
            key: value for key, value in family_result.items() if key != "training" and key != "per_label"
        })
    write_json(result_path, result)
    return result


def family_retention(reference: dict, candidate: dict, family: str, min_examples: int) -> dict:
    ratios = {}
    for name, ref_values in reference["families"][family]["per_label"].items():
        cand_values = candidate["families"][family]["per_label"][name]
        baseline = float(ref_values["prevalence"])
        denominator = float(ref_values["average_precision"]) - baseline
        if (
            ref_values["positives"] < min_examples
            or ref_values["negatives"] < min_examples
            or not np.isfinite(denominator)
            or denominator < 0.10
        ):
            continue
        ratios[name] = (float(cand_values["average_precision"]) - baseline) / denominator
    return {
        "eligible_labels": sorted(ratios),
        "per_label": ratios,
        "retention": float(np.mean(list(ratios.values()))) if ratios else float("nan"),
    }


def make_gate(reference: dict, candidate: dict, reference_name: str, candidate_name: str) -> dict:
    state = family_retention(reference, candidate, "state", min_examples=50)
    text = family_retention(reference, candidate, "bow", min_examples=20)
    state_retention = state["retention"]
    text_retention = text["retention"]
    mean_retention = float(np.nanmean([state_retention, text_retention]))
    task_drop = (
        float(reference["families"]["task"]["accuracy"])
        - float(candidate["families"]["task"]["accuracy"])
    )
    passed = bool(
        np.isfinite(state_retention)
        and np.isfinite(text_retention)
        and state_retention >= 0.80
        and text_retention >= 0.80
        and mean_retention >= 0.85
        and task_drop <= 0.05
    )
    return {
        "reference": reference_name,
        "candidate": candidate_name,
        "passed": passed,
        "thresholds": {
            "state_retention_min": 0.80,
            "text_retention_min": 0.80,
            "mean_retention_min": 0.85,
            "task_accuracy_drop_max": 0.05,
        },
        "state": state,
        "text": text,
        "mean_retention": mean_retention,
        "task_accuracy_drop": task_drop,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("probe_h", "probe_y64", "finalize_gate64", "gate64", "y32", "pca64"),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    records = list(iter_jsonl(args.records))
    targets = build_targets(records)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "probe-targets.json", {
        "task_names": targets["task_names"],
        "state_names": targets["state_names"],
        "bow_names": targets["bow_names"],
        "split_counts": {key: len(value) for key, value in targets["splits"].items()},
    })

    if args.mode in {"probe_h", "probe_y64"}:
        representation = args.mode.removeprefix("probe_")
        result = evaluate_representation(args, records, targets, representation)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.mode == "finalize_gate64":
        h_path = output / "probe-h.json"
        y64_path = output / "probe-y64.json"
        if not h_path.exists() or not y64_path.exists():
            raise RuntimeError("both probe-h.json and probe-y64.json are required")
        reference = json.loads(h_path.read_text())
        candidate = json.loads(y64_path.read_text())
        gate = make_gate(reference, candidate, "h", "y64")
        gate["on_failure"] = "STOP_AND_REPORT_NO_LEARNED_QUERIES"
        gate_path = output / "gate64.json"
    elif args.mode == "gate64":
        reference = evaluate_representation(args, records, targets, "h")
        candidate = evaluate_representation(args, records, targets, "y64")
        gate = make_gate(reference, candidate, "h", "y64")
        gate["on_failure"] = "STOP_AND_REPORT_NO_LEARNED_QUERIES"
        gate_path = output / "gate64.json"
    else:
        gate64_path = output / "gate64.json"
        if not gate64_path.exists() or not json.loads(gate64_path.read_text()).get("passed"):
            raise RuntimeError("Y64 gate has not passed; stopping before downstream experiments")
        reference = json.loads((output / "probe-y64.json").read_text())
        representation = "y32" if args.mode == "y32" else "pca64"
        candidate = evaluate_representation(args, records, targets, representation)
        gate = make_gate(reference, candidate, "y64", representation)
        gate_path = output / f"gate-{representation}.json"
    write_json(gate_path, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))
    if args.mode in {"gate64", "finalize_gate64"} and not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
