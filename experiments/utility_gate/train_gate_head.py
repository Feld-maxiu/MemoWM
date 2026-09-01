"""Fit the head that predicts a position's utility from what the encoder knows.

At write time there is no question yet, so the most a gate can aim at is
``E_q[U]``. The labels give ``U`` per (state, position, question); this
aggregates them to a cell mean and fits a small regressor to it.

The ceiling is not 1. The variance decomposition puts roughly half of ``var(U)``
in "which question gets asked", which no feature of the code can carry, so a
per-label R² above ~0.47 would mean something had leaked. Held-out R² is
therefore reported against the *cell means*, which is the quantity the gate
actually needs, alongside the per-label figure for comparison.

Features are deliberately all encoder-side -- everything here is available at
write time:

* the slot's own 512-d latent, projected down;
* slot and subspace identity, as embeddings. The diagnostics show identity alone
  explains ~1% of the variance, so these are a weak prior, not the answer;
* the centroid displacement the substitution would cause. The rotation is
  orthogonal per slot, so the norm of the centroid difference is the norm of the
  perturbation in standardised space -- computable from the codebook alone,
  without decoding anything;
* when a posterior dump is passed, the world model's rate, entropy, top-1
  log-probability, margin, and whether its mode already equals the true code.
  That last flag is worth its own feature: where it fires the utility is exactly
  zero, since the substitution is the identity.

Splits are by sample, never by row: two questions about the same observation
share a state, and putting one in train and the other in validation would let the
model memorise the cell rather than predict it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def load_labels(paths: list[str]) -> dict:
    parts: dict[str, list] = {}
    metadata = None
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            metadata = json.loads(str(np.asarray(data["metadata"])))
            for name in data.files:
                if name in ("metadata", "state_ids"):
                    continue
                parts.setdefault(name, []).append(np.asarray(data[name]))
            if "state_ids" in data.files:
                parts.setdefault("_state_ids", [np.asarray(data["state_ids"])])
    table = {name: np.concatenate(blocks) for name, blocks in parts.items()
             if not name.startswith("_")}
    table["state_ids"] = parts["_state_ids"][0]
    table["metadata"] = metadata
    return table


def cell_means(state: np.ndarray, position: np.ndarray, value: np.ndarray):
    """Average the questions away: (state, position) -> E_q[U], and the count."""
    keys = state.astype(np.int64) * 100000 + position.astype(np.int64)
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    totals = np.zeros(len(unique), np.float64)
    np.add.at(totals, inverse, value)
    first = np.zeros(len(unique), np.int64)
    first[inverse[::-1]] = np.arange(len(inverse))[::-1]
    return (state[first], position[first], totals / counts, counts)


class GateHead(nn.Module):
    def __init__(self, latent_dim: int, extra: int, slots: int, subspaces: int,
                 hidden: int = 128, embed: int = 16):
        super().__init__()
        self.project = nn.Linear(latent_dim, 64)
        self.slot = nn.Embedding(slots, embed)
        self.subspace = nn.Embedding(subspaces, embed)
        self.mlp = nn.Sequential(
            nn.Linear(64 + 2 * embed + extra, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )

    def forward(self, latent, slot, subspace, extra):
        parts = [self.project(latent), self.slot(slot), self.subspace(subspace)]
        if extra.shape[-1]:
            parts.append(extra)
        return self.mlp(torch.cat(parts, -1)).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--gate", default="gate")
    parser.add_argument("--label", default="train")
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--posteriors", default=None)
    parser.add_argument("--target", choices=("delta_nll_bits", "kl_bits"),
                        default="delta_nll_bits")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    table = load_labels(args.labels)
    gate = Path(args.gate)
    with np.load(gate / "states.npz", allow_pickle=True) as data:
        xbar = np.asarray(data["xbar"], np.float32)
    with np.load(args.codebook, allow_pickle=True) as data:
        centroids = np.asarray(data["centroids"], np.float32)
    with np.load(gate / "codes.npz", allow_pickle=True) as data:
        codes = np.asarray(data[f"codes/{args.label}"], np.uint8)

    fill = np.zeros_like(codes)
    posterior_features = None
    if args.posteriors:
        from experiments.utility_gate.label_counterfactual import cache_order
        with np.load(args.posteriors, allow_pickle=True) as data:
            dumped = {k: np.asarray(data[k]) for k in data.files}
        state_ids = [str(v) for v in table["state_ids"]]
        global_index = cache_order(gate / "records.jsonl", args.label)
        row_of_global = {int(g): r for r, g in enumerate(dumped["target_indices"])}
        rows = np.full(len(codes), -1, np.int64)
        for index, sid in enumerate(state_ids):
            rows[index] = row_of_global.get(global_index.get(sid, -1), -1)
        posterior_features = (dumped, rows)
        for index, row in enumerate(rows):
            if row >= 0:
                fill[index] = dumped["wm_argmax"][row]
    else:
        for slot in range(codes.shape[1]):
            for subspace in range(codes.shape[2]):
                fill[:, slot, subspace] = np.bincount(
                    codes[:, slot, subspace], minlength=256).argmax()

    state, position, target, counts = cell_means(
        table["state_row"], table["position"], table[args.target].astype(np.float64))
    slot, subspace = position // 32, position % 32

    true_code = codes[state, slot, subspace]
    fill_code = fill[state, slot, subspace]
    displacement = np.linalg.norm(
        centroids[slot, subspace, true_code] - centroids[slot, subspace, fill_code],
        axis=-1).astype(np.float32)

    extras = [displacement[:, None], (true_code == fill_code)[:, None].astype(np.float32)]
    names = ["centroid_displacement", "fill_is_identity"]
    if posterior_features is not None:
        dumped, rows = posterior_features
        row = rows[state]
        for key, label in (("code_bits", "rate_bits"), ("entropy_bits", "entropy"),
                           ("top1_logprob_bits", "top1"), ("margin_bits", "margin")):
            values = dumped[key][row, slot, subspace].astype(np.float32)
            extras.append(values[:, None])
            names.append(label)
    extra = np.concatenate(extras, axis=-1)

    sample_of = np.asarray([str(s).rsplit("-", 1)[0] for s in table["state_ids"]])
    samples = np.unique(sample_of)
    rng = np.random.default_rng(args.seed)
    held = set(rng.permutation(samples)[: max(1, int(len(samples) * args.validation_fraction))])
    is_validation = np.asarray([sample_of[s] in held for s in state])

    device = torch.device("cpu")
    latent = torch.as_tensor(xbar[state, slot], device=device)
    features = {
        "latent": latent,
        "slot": torch.as_tensor(slot.astype(np.int64), device=device),
        "subspace": torch.as_tensor(subspace.astype(np.int64), device=device),
        "extra": torch.as_tensor(extra, device=device),
    }
    y = torch.as_tensor(target.astype(np.float32), device=device)
    weight = torch.as_tensor(counts.astype(np.float32), device=device)
    train = torch.as_tensor(~is_validation, device=device)
    valid = torch.as_tensor(is_validation, device=device)

    centre = features["extra"][train].mean(0)
    spread = features["extra"][train].std(0).clamp_min(1e-6)
    features["extra"] = (features["extra"] - centre) / spread
    latent_centre = latent[train].mean(0)
    latent_spread = latent[train].std(0).clamp_min(1e-6)
    features["latent"] = (latent - latent_centre) / latent_spread

    torch.manual_seed(args.seed)
    head = GateHead(latent.shape[-1], extra.shape[-1], 32, 32).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)

    def evaluate(mask):
        head.eval()
        with torch.no_grad():
            predicted = head(features["latent"][mask], features["slot"][mask],
                             features["subspace"][mask], features["extra"][mask])
        actual = y[mask]
        residual = ((actual - predicted) ** 2).mean()
        total = actual.var(unbiased=False)
        return float(1 - residual / total), predicted, actual

    best = (-1e9, 0)
    for epoch in range(args.epochs):
        head.train()
        optimiser.zero_grad()
        predicted = head(features["latent"][train], features["slot"][train],
                         features["subspace"][train], features["extra"][train])
        loss = (weight[train] * (predicted - y[train]) ** 2).sum() / weight[train].sum()
        loss.backward()
        optimiser.step()
        if epoch % 10 == 9 or epoch == args.epochs - 1:
            score, _, _ = evaluate(valid)
            if score > best[0]:
                best = (score, epoch)

    train_r2, _, _ = evaluate(train)
    valid_r2, predicted, actual = evaluate(valid)
    order = np.argsort(np.argsort(predicted.numpy())).astype(np.float64)
    truth = np.argsort(np.argsort(actual.numpy())).astype(np.float64)
    order -= order.mean(); truth -= truth.mean()
    spearman = float((order * truth).sum() /
                     np.sqrt((order ** 2).sum() * (truth ** 2).sum()))

    report = {
        "target": args.target,
        "cells": int(len(target)), "states": int(len(np.unique(state))),
        "samples": int(len(samples)), "held_out_samples": int(len(held)),
        "features": ["slot_latent(512)", "slot_id", "subspace_id", *names],
        "questions_per_cell_mean": float(counts.mean()),
        "train_r2": train_r2, "validation_r2": valid_r2,
        "best_validation_r2": best[0], "best_epoch": best[1],
        "validation_spearman": spearman,
        "posteriors": args.posteriors,
        "labels": list(args.labels),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"protocol": "residualmem_utility_gate_head_v1",
                    "state_dict": head.state_dict(), "report": report,
                    "normalisation": {"extra_centre": centre, "extra_spread": spread,
                                      "latent_centre": latent_centre,
                                      "latent_spread": latent_spread}},
                   args.output)


if __name__ == "__main__":
    main()
