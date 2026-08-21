"""Train the shared Xbar/A2 retrieval head against fused Qwen3-VL teachers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from residualmem.latent.instruct_bridge import (
    BRIDGE_CACHE_PROTOCOL,
    FUSED_OBSERVATION_TEACHER_PROTOCOL,
    RETRIEVAL_BRIDGE_PROTOCOL,
    MaskedAttentionRetrievalHead,
    save_bridge,
)


def load_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        if "teacher_embedding" in data.files:
            raise ValueError("stale screenshot-only teacher cache is forbidden")
        required = {"xbar", "valid", "teacher_fused_embedding", "split", "metadata"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"bridge cache missing {sorted(missing)}")
        metadata = json.loads(str(np.asarray(data["metadata"]).item()))
        if metadata.get("protocol") != BRIDGE_CACHE_PROTOCOL:
            raise ValueError("bridge cache protocol mismatch")
        if metadata.get("teacher_protocol") != FUSED_OBSERVATION_TEACHER_PROTOCOL:
            raise ValueError("bridge teacher is not fused observation v1")
        representation = metadata.get(
            "representation", "both" if "a2_xbar" in data.files else "xbar"
        )
        if representation not in {"xbar", "both"}:
            raise ValueError(f"unsupported cache representation {representation!r}")
        if representation == "both" and "a2_xbar" not in data.files:
            raise ValueError("representation=both cache has no a2_xbar")
        if "a2_xbar" in data.files:
            required.add("a2_xbar")
        output = {name: np.asarray(data[name]) for name in required if name != "metadata"}
        metadata["representation"] = representation
        output["metadata"] = metadata
    if output["xbar"].shape[1:] != (64, 512):
        raise ValueError(f"unexpected xbar shape {output['xbar'].shape}")
    if output["teacher_fused_embedding"].shape[1:] != (4096,):
        raise ValueError("teacher embedding must be 4096-D")
    return output


def symmetric_infonce(student: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = student @ teacher.T / temperature
    labels = torch.arange(len(student), device=student.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def batch_loss(head, xbar, valid, teacher, temperature, a2=None):
    x_vector = head(xbar, valid)
    if a2 is None:
        contrastive = symmetric_infonce(x_vector, teacher, temperature)
        cosine = (1.0 - (x_vector * teacher).sum(-1)).mean()
        return contrastive + cosine, {
            "contrastive": contrastive,
            "cosine": cosine,
            "xbar_cosine": (x_vector * teacher).sum(-1).mean(),
        }
    a2_vector = head(a2, valid)
    contrastive = 0.5 * (
        symmetric_infonce(x_vector, teacher, temperature)
        + symmetric_infonce(a2_vector, teacher, temperature)
    )
    cosine = 0.5 * (
        (1.0 - (x_vector * teacher).sum(-1)).mean()
        + (1.0 - (a2_vector * teacher).sum(-1)).mean()
    )
    consistency = (1.0 - (x_vector * a2_vector).sum(-1)).mean()
    return contrastive + cosine + 0.1 * consistency, {
        "contrastive": contrastive,
        "cosine": cosine,
        "consistency": consistency,
        "xbar_cosine": (x_vector * teacher).sum(-1).mean(),
        "a2_cosine": (a2_vector * teacher).sum(-1).mean(),
    }


def evaluate(head, tensors, indices, batch_size, temperature):
    totals: dict[str, float] = {}
    count = 0
    head.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            index = indices[start:start + batch_size]
            loss, metrics = batch_loss(
                head, tensors["xbar"][index], tensors["valid"][index],
                tensors["teacher"][index], temperature,
                tensors.get("a2_xbar", None)[index] if "a2_xbar" in tensors else None,
            )
            values = {"loss": loss, **metrics}
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value) * len(index)
            count += len(index)
    return {name: value / max(count, 1) for name, value in totals.items()}


def train(args: argparse.Namespace) -> dict:
    cache = load_cache(args.cache)
    cache_representation = str(cache["metadata"]["representation"])
    representation = cache_representation if args.representation == "auto" else args.representation
    if representation == "both" and "a2_xbar" not in cache:
        raise ValueError("representation=both requested but cache has no a2_xbar")
    device = torch.device(args.device)
    tensors = {
        "xbar": torch.as_tensor(cache["xbar"], dtype=torch.float32, device=device),
        "valid": torch.as_tensor(cache["valid"], dtype=torch.bool, device=device),
        "teacher": F.normalize(torch.as_tensor(
            cache["teacher_fused_embedding"], dtype=torch.float32, device=device
        ), dim=-1),
    }
    if representation == "both":
        tensors["a2_xbar"] = torch.as_tensor(
            cache["a2_xbar"], dtype=torch.float32, device=device
        )
    split = cache["split"].astype(str)
    train_idx = np.flatnonzero(split == "train")
    val_idx = np.flatnonzero(split == "validation")
    if not len(train_idx) or not len(val_idx):
        raise ValueError("cache needs nonempty train and validation splits")
    torch.manual_seed(args.seed)
    head = MaskedAttentionRetrievalHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    best = math.inf
    best_step = 0
    stale = 0
    history = []
    for step in range(1, args.max_steps + 1):
        index = rng.choice(train_idx, size=min(args.batch_size, len(train_idx)), replace=False)
        index_t = torch.as_tensor(index, device=device)
        head.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = batch_loss(
            head, tensors["xbar"][index_t], tensors["valid"][index_t],
            tensors["teacher"][index_t], args.temperature,
            tensors.get("a2_xbar", None)[index_t] if "a2_xbar" in tensors else None,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), args.clip_norm)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.max_steps:
            metrics = evaluate(head, tensors, torch.as_tensor(val_idx, device=device), args.eval_batch_size, args.temperature)
            history.append({"step": step, **metrics})
            print(json.dumps(history[-1], sort_keys=True), flush=True)
            if metrics["loss"] < best:
                best = metrics["loss"]
                best_step = step
                stale = 0
                save_bridge(
                    args.output, head, protocol=RETRIEVAL_BRIDGE_PROTOCOL,
                    cache_protocol=BRIDGE_CACHE_PROTOCOL,
                    teacher_protocol=FUSED_OBSERVATION_TEACHER_PROTOCOL,
                    representation=representation,
                    cache=str(Path(args.cache).resolve()),
                    temperature=args.temperature,
                    best_step=step, validation=metrics,
                )
            else:
                stale += 1
                if stale >= args.patience_evals:
                    break
    report = {
        "protocol": RETRIEVAL_BRIDGE_PROTOCOL,
        "representation": representation,
        "cache": str(Path(args.cache).resolve()),
        "output": str(Path(args.output).resolve()),
        "best_step": best_step,
        "best_validation_loss": best,
        "history": history,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--representation", choices=("auto", "xbar", "both"), default="auto")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience-evals", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
