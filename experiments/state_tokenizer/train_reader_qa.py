"""Train the input connector to answer questions, not to recite observations.

``train_reader_bridge.py`` optimizes next-token CE on the observation text, so
the latent becomes a prefix that regenerates that text. Inference asks something
different -- answer this question from this latent -- and the mismatch is
visible: the model answers and then keeps going, narrating the observation.
Hallucination measured 33% in the latent condition against 0% in the text one.

This trains on the shape inference actually uses, with two terms:

* **answer CE** -- teacher-forced on the gold answer, given the latent and the
  question, in the same prompt the reader builds at inference.
* **distillation KL** -- the frozen generator reading the *observation text*
  plus the same question defines the target distribution; the student sees the
  latent instead. Token-level KL along the teacher's own greedy trajectory.
  This is the Latent-Memory term: it aligns the latent with what the generator
  needs to answer, rather than with the text's surface form.

The teacher trajectory is recomputed each step rather than cached. Caching it
would mean storing vocabulary-wide logits, and the teacher forward is one extra
pass over a frozen 9B -- cheaper than the storage and simpler than a top-k
approximation.

Only non-evaluation subcategories supply pairs; see build_reader_qa_pairs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from residualmem.latent.instruct_bridge import (
    READER_BRIDGE_PROTOCOL,
    InputSoftTokenConnector,
    save_bridge,
)

from .extract_qwen import _load_model
from .reader_losses import answer_ce_and_distill_kl


def build_loss(model, processor, connector, row, device, max_answer_tokens, weight):
    """Cached ``xbar`` -> soft tokens -> the shared reader losses.

    The loss itself lives in ``reader_losses`` because ``train_qformer_joint``
    optimizes exactly the same objective from a learnable latent, and keeping
    two copies is how the distillation term ended up mis-scaled in one of them.
    """
    xbar, valid, question, answer, memory_text = row
    valid_mask = torch.as_tensor(valid[None], dtype=torch.bool, device=device)
    latent = connector(
        torch.as_tensor(xbar[None], dtype=torch.float32, device=device),
        valid_mask,
    )
    return answer_ce_and_distill_kl(
        model, processor, latent, valid_mask, question, answer, memory_text,
        max_answer_tokens=max_answer_tokens, weight=weight,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--texts", required=True,
                        help="cache carrying target_text, for the teacher's context")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--validation-rows", type=int, default=128)
    parser.add_argument("--patience-evals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()

    pairs = np.load(args.pairs, allow_pickle=False)
    metadata = json.loads(str(np.asarray(pairs["metadata"])))
    if metadata.get("excluded") != "web":
        raise ValueError("the evaluation subcategory must be excluded from the pairs")

    # The observation text the teacher reads. Keyed by (sample, row) is overkill
    # here: the pairs carry the xbar directly, so the text is looked up by
    # matching the observation's own reconstruction target.
    texts = np.load(args.texts, allow_pickle=False)
    lookup = {}
    if "target_text" in texts.files:
        for key, text in zip(texts["xbar"], texts["target_text"]):
            lookup[key.tobytes()] = str(text)

    device = torch.device(args.device)
    processor, model = _load_model(args.model, device, False)
    # Seed *before* constructing the connector: its Linear layers draw from the
    # global torch generator, so seeding afterwards left the initialization
    # unreproducible. train_retrieval_bridge.py:131-132 is the correct order.
    torch.manual_seed(args.seed)
    connector = InputSoftTokenConnector().to(device)
    optimizer = torch.optim.AdamW(connector.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)

    split = pairs["split"].astype(str)
    train_rows = np.flatnonzero(split == "train")
    val_rows = np.flatnonzero(split == "validation")
    # Drawn once, so every eval scores the same rows and the curve is
    # comparable across steps. Taking the first N in index order instead
    # sampled the earliest-sorted held-out samples rather than the split.
    chosen_val = (
        val_rows if len(val_rows) <= args.validation_rows
        else np.random.default_rng(args.seed + 1).choice(
            val_rows, args.validation_rows, replace=False
        )
    )
    missing = 0

    def row_at(index):
        nonlocal missing
        xbar = pairs["xbar"][index]
        text = lookup.get(xbar.tobytes())
        if text is None:
            missing += 1
            text = ""
        return (xbar, pairs["valid"][index], str(pairs["question"][index]),
                str(pairs["answer"][index]), text)

    print(f"[qa-reader] {len(train_rows)} train / {len(val_rows)} validation pairs, "
          f"distill weight {args.distill_weight}", flush=True)

    best, best_step, stale, history = math.inf, 0, 0, []
    for step in range(1, args.max_steps + 1):
        connector.train()
        optimizer.zero_grad(set_to_none=True)
        index = int(rng.choice(train_rows))
        row = row_at(index)
        weight = args.distill_weight if row[4] else 0.0
        loss, ce, kl = build_loss(
            model, processor, connector, row, device, args.max_answer_tokens, weight
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(connector.parameters(), args.clip_norm)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.max_steps:
            connector.eval()
            losses = []
            with torch.no_grad():
                for index in chosen_val:
                    row = row_at(int(index))
                    _, ce_value, _ = build_loss(
                        model, processor, connector, row, device,
                        args.max_answer_tokens, 0.0
                    )
                    losses.append(ce_value)
            validation = float(np.mean(losses))
            history.append({"step": step, "validation_answer_ce": validation,
                            "train_ce": ce, "train_kl": kl})
            print(f"[qa-reader] step {step:6d}  val answer CE {validation:.4f}  "
                  f"(train CE {ce:.4f}, KL {kl:.4f})", flush=True)
            if validation < best:
                best, best_step, stale = validation, step, 0
                save_bridge(args.output, connector, protocol=READER_BRIDGE_PROTOCOL,
                            mode="input", representation="xbar",
                            best_step=step, validation_answer_ce=validation,
                            distill_weight=args.distill_weight,
                            objective="answer_ce+distill_kl")
            else:
                stale += 1
                if stale >= args.patience_evals:
                    break

    report = {"protocol": READER_BRIDGE_PROTOCOL, "objective": "answer_ce+distill_kl",
              "distill_weight": args.distill_weight, "best_step": best_step,
              "best_validation_answer_ce": best, "rows_without_text": missing,
              "pairs_metadata": metadata, "history": history}
    Path(args.output).with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
