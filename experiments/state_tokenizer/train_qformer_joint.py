"""Train the Q-Former tokenizer and the reader connector together.

The objective is the same two terms ``train_reader_qa`` uses -- answer CE plus
distillation KL against the frozen generator reading the observation text -- so
the only variable between the two runs is where the latent comes from. That is
the comparison Stage C measures.

**No anchor losses.** Report Eq. (9a) proposes a mixed frozen target
(semantic / OCR / visual / state subspaces) so the latent cannot drift. Three
reasons it is not here:

* The frozen observation prompt already asks for all four -- "preserve visible
  text, input values, control types, selected/focused/enabled state and spatial
  relations". In the pooled pipeline that request reaches nothing: the prompt is
  last in the sequence, causal attention hides it from the image and DOM
  positions, and its own band is ``PROMPT_SLOTS = 0``. Cross-attention has no
  such barrier, so the prompt starts working the moment the resampler does.
* Report 5.4's collapse scenario is a tokenizer trained *with the world model*,
  which is rewarded for making residuals small. Both terms here reward keeping
  information instead.
* A text-reconstruction anchor is precisely the objective that made the first
  connector answer and then narrate the observation.

What replaces them is a monitor, not a loss: every eval reports pairwise cosine
and effective rank against the *fixed-pooling* xbar of the same observations. If
the learned states are less distinguishable than the pooling they replace, the
run has failed and the specific anchor for the failing metric goes back in.

Micro-batch is 1 with gradient accumulation rather than a padded batch. The
student's memory is 64 soft tokens and the teacher's is variable-length text, so
a real batch means aligning an answer span across two different paddings -- the
same class of bug as the KL reduction this file's loss was just fixed for. The
accumulated version reuses the single-sample forward unchanged and costs ~7%.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from residualmem.latent.instruct_bridge import InputSoftTokenConnector, save_bridge
from residualmem.latent.qformer import (
    QFORMER_PROTOCOL,
    QFormerStateReader,
    StateQFormer,
    qformer_hash,
)

from .extract_qwen import _load_model
from .reader_losses import answer_ce_and_distill_kl
from .trunk_states import NUM_MODALITIES, collate, trunk_states

PAIRS_PROTOCOL = "wma_qformer_qa_v1"


class ObservationStore:
    """``(sample_id, index)`` -> the inputs the trunk saw, plus the pooled xbar."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._records: dict[str, list[dict]] = {}
        self._pooled: dict[str, np.ndarray] = {}

    def _load(self, sample_id: str) -> list[dict]:
        if sample_id not in self._records:
            path = self._directory / f"{sample_id}.npz"
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(np.asarray(data["metadata"])))
            self._records[sample_id] = metadata.get("records") or []
        return self._records[sample_id]

    def observation(self, sample_id: str, index: int) -> dict:
        record = self._load(sample_id)[index]
        if not record.get("synthetic_axtree"):
            raise ValueError(
                f"{sample_id}[{index}] has no synthetic_axtree; re-run the extraction"
            )
        return record

    def pooled_xbar(self, sample_id: str, index: int) -> np.ndarray:
        """The fixed pooling's own output for this observation -- the baseline."""
        key = f"{sample_id}/{index}"
        if key not in self._pooled:
            with np.load(self._directory / f"{sample_id}.npz", allow_pickle=False) as data:
                self._pooled[key] = np.asarray(data[f"m11/xbar/{index:04d}"], np.float32)
        return self._pooled[key]


def spread(states: np.ndarray) -> dict[str, float]:
    """Report 5.4's monitors: how distinguishable are these states from each other.

    Every quantity is computed **after removing the mean across observations**.
    Without that, the cosine measures a shared offset rather than collapse and
    the two representations are not comparable: the pooled xbar is near
    zero-mean by construction (frozen group/channel normalization) while a
    learned resampler's output is not. Measured at step 400 the raw cosine read
    1.0000 for the Q-Former against 0.5739 for the pooling, which looks like
    total collapse; centred, the same states read 0.0389 against -0.0225, i.e.
    close to orthogonal in both. ``mean_to_deviation`` is what the raw cosine
    was actually reporting, kept as its own number -- 251.5 against 1.20 says
    the informative part is 0.4% of the learned state's magnitude, which is a
    real pathology but a different one from collapse.
    """
    flat = states.reshape(len(states), -1).astype(np.float64)
    centred = flat - flat.mean(0, keepdims=True)
    unit = centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), 1e-12)
    gram = unit @ unit.T
    upper = gram[np.triu_indices(len(unit), 1)]
    singular = np.linalg.svdvals(centred)
    share = singular / max(singular.sum(), 1e-12)
    share = share[share > 0]
    deviation = np.linalg.norm(centred, axis=1).mean()
    return {
        "pairwise_cosine": float(upper.mean()),
        "effective_rank": float(np.exp(-(share * np.log(share)).sum())),
        "mean_to_deviation": float(np.linalg.norm(flat.mean(0)) / max(deviation, 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="build_qformer_qa_pairs output")
    parser.add_argument("--xbar-dir", required=True,
                        help="the extraction the pairs reference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--qformer-hidden", type=int, default=1024)
    parser.add_argument("--qformer-heads", type=int, default=8)
    parser.add_argument("--qformer-layers", type=int, default=4)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--accumulate", type=int, default=8,
                        help="micro-batches of 1 per optimizer step")
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--validation-rows", type=int, default=96)
    parser.add_argument("--patience-evals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()

    pairs = np.load(args.pairs, allow_pickle=False)
    metadata = json.loads(str(np.asarray(pairs["metadata"])))
    if metadata.get("protocol") != PAIRS_PROTOCOL:
        raise ValueError(f"pairs protocol {metadata.get('protocol')!r}")
    if metadata.get("excluded") != "web":
        raise ValueError("the evaluation subcategory must be excluded from the pairs")

    device = torch.device(args.device)
    processor, model = _load_model(args.model, device, False)
    # Seed before construction; both modules draw from the global generator.
    torch.manual_seed(args.seed)
    joint = QFormerStateReader(
        StateQFormer(
            num_queries=args.queries, hidden=args.qformer_hidden,
            heads=args.qformer_heads, layers=args.qformer_layers,
            modalities=NUM_MODALITIES,
        ),
        InputSoftTokenConnector(slots=args.queries),
    ).to(device)
    optimizer = torch.optim.AdamW(joint.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    store = ObservationStore(args.xbar_dir)

    split = pairs["split"].astype(str)
    train_rows = np.flatnonzero(split == "train")
    val_rows = np.flatnonzero(split == "validation")
    chosen_val = (
        val_rows if len(val_rows) <= args.validation_rows
        else np.random.default_rng(args.seed + 1).choice(
            val_rows, args.validation_rows, replace=False
        )
    )
    trainable = sum(p.numel() for p in joint.parameters() if p.requires_grad)
    print(f"[qformer] {len(train_rows)} train / {len(val_rows)} validation pairs, "
          f"{len(chosen_val)} scored per eval", flush=True)
    print(f"[qformer] {trainable/1e6:.1f}M trainable, {args.queries} queries, "
          f"accumulate {args.accumulate}, distill weight {args.distill_weight}", flush=True)

    def latents_for(row: int):
        sample_id = str(pairs["sample_id"][row])
        index = int(pairs["record_index"][row])
        record = store.observation(sample_id, index)
        with Image.open(record["screenshot"]) as handle:
            image = handle.convert("RGB")
        states = trunk_states(
            processor, model, image, record["synthetic_axtree"],
            layer=args.layer, device=device,
        )
        soft, xbar, valid = joint(*collate([states]))
        return soft, xbar, valid, record, sample_id, index

    def loss_for(row: int, weight: float):
        soft, xbar, valid, record, _sample, _index = latents_for(row)
        loss, ce, kl = answer_ce_and_distill_kl(
            model, processor, soft, valid,
            str(pairs["question"][row]), str(pairs["answer"][row]),
            str(record.get("fused_text", "")),
            max_answer_tokens=args.max_answer_tokens, weight=weight,
        )
        return loss, ce, kl, xbar

    best, best_step, stale, history = math.inf, 0, 0, []
    for step in range(1, args.max_steps + 1):
        joint.train()
        optimizer.zero_grad(set_to_none=True)
        totals = collections.Counter()
        for _ in range(args.accumulate):
            row = int(rng.choice(train_rows))
            loss, ce, kl, _ = loss_for(row, args.distill_weight)
            (loss / args.accumulate).backward()
            totals["ce"] += ce
            totals["kl"] += kl
        torch.nn.utils.clip_grad_norm_(joint.parameters(), args.clip_norm)
        optimizer.step()

        if step % args.eval_every and step != args.max_steps:
            continue

        joint.eval()
        losses, learned, pooled = [], [], []
        with torch.no_grad():
            for row in chosen_val:
                _loss, ce, _kl, xbar = loss_for(int(row), 0.0)
                losses.append(ce)
                learned.append(xbar[0].float().cpu().numpy())
                pooled.append(store.pooled_xbar(
                    str(pairs["sample_id"][row]), int(pairs["record_index"][row])
                ))
        validation = float(np.mean(losses))
        # Same observations, both representations -- the comparison is paired.
        monitors = {"learned": spread(np.stack(learned)),
                    "pooled": spread(np.stack(pooled))}
        regressed = (
            monitors["learned"]["pairwise_cosine"] > monitors["pooled"]["pairwise_cosine"]
            or monitors["learned"]["effective_rank"] < monitors["pooled"]["effective_rank"]
        )
        history.append({
            "step": step, "validation_answer_ce": validation,
            "train_ce": totals["ce"] / args.accumulate,
            "train_kl": totals["kl"] / args.accumulate,
            "monitors": monitors, "collapse_regressed": regressed,
        })
        flag = "  COLLAPSE-REGRESSED" if regressed else ""
        print(
            f"[qformer] step {step:5d}  val CE {validation:.4f}  "
            f"(train CE {totals['ce']/args.accumulate:.4f}, "
            f"KL {totals['kl']/args.accumulate:.4f})  "
            f"cos {monitors['learned']['pairwise_cosine']:+.4f}"
            f"/{monitors['pooled']['pairwise_cosine']:+.4f}  "
            f"rank {monitors['learned']['effective_rank']:.1f}"
            f"/{monitors['pooled']['effective_rank']:.1f}  "
            f"mean/dev {monitors['learned']['mean_to_deviation']:.1f}"
            f"/{monitors['pooled']['mean_to_deviation']:.1f}{flag}",
            flush=True,
        )
        if validation < best:
            best, best_step, stale = validation, step, 0
            save_bridge(
                args.output, joint, protocol=QFORMER_PROTOCOL,
                queries=args.queries, layer=args.layer,
                best_step=step, validation_answer_ce=validation,
                distill_weight=args.distill_weight,
                objective="answer_ce+distill_kl",
                qformer_sha256=qformer_hash(joint.qformer),
                monitors=monitors,
            )
        else:
            stale += 1
            if stale >= args.patience_evals:
                break

    report = {
        "protocol": QFORMER_PROTOCOL, "objective": "answer_ce+distill_kl",
        "queries": args.queries, "distill_weight": args.distill_weight,
        "accumulate": args.accumulate, "best_step": best_step,
        "best_validation_answer_ce": best, "pairs_metadata": metadata,
        "history": history,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
