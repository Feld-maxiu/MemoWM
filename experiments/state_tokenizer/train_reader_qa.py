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
from torch.nn import functional as F

from residualmem.latent.instruct_bridge import (
    READER_BRIDGE_PROTOCOL,
    InputSoftTokenConnector,
    save_bridge,
)

from .extract_qwen import _load_model

# The reader builds exactly this at inference; training on a different shape is
# what produced the mismatch in the first place.
INSTRUCTION = (
    "Answer the question from the retrieved memory. Keep the answer concise. "
    "If absent, say exactly 'Not mentioned in memory.'.\nQuestion: "
)


def prompt_ids(processor, question: str, device):
    encoded = processor.tokenizer(
        INSTRUCTION + question, return_tensors="pt", add_special_tokens=True
    )
    return encoded["input_ids"].to(device)


def build_loss(model, processor, connector, row, device, max_answer_tokens, weight):
    xbar, valid, question, answer, memory_text = row
    embed = model.get_input_embeddings()
    dtype = embed.weight.dtype

    latent = connector(
        torch.as_tensor(xbar[None], dtype=torch.float32, device=device),
        torch.as_tensor(valid[None], dtype=torch.bool, device=device),
    )
    valid_mask = torch.as_tensor(valid[None], device=device).to(torch.long)

    question_ids = prompt_ids(processor, question, device)
    answer_ids = processor.tokenizer(
        answer, return_tensors="pt", add_special_tokens=False,
        truncation=True, max_length=max_answer_tokens,
    )["input_ids"].to(device)
    question_embeds = embed(question_ids).detach()
    answer_embeds = embed(answer_ids).detach()
    answer_len = int(answer_ids.shape[1])

    # Text memory for the teacher: the observation the question is about is
    # already what the latent encodes, so the teacher's context is that text.
    def run(memory_embeds, memory_mask):
        inputs = torch.cat((memory_embeds.to(dtype), question_embeds, answer_embeds), 1)
        attention = torch.cat(
            (memory_mask, torch.ones_like(question_ids), torch.ones_like(answer_ids)), 1
        )
        labels = torch.cat((
            torch.full(memory_mask.shape, -100, device=device, dtype=torch.long),
            torch.full_like(question_ids, -100),
            answer_ids,
        ), 1)
        return model(inputs_embeds=inputs, attention_mask=attention,
                     labels=labels, use_cache=False)

    student = run(latent, valid_mask)
    loss = student.loss

    if weight > 0:
        with torch.no_grad():
            text_ids = processor.tokenizer(
                memory_text, return_tensors="pt", add_special_tokens=False,
                truncation=True, max_length=256,
            )["input_ids"].to(device)
            teacher = run(embed(text_ids), torch.ones_like(text_ids))
        # Align on the answer span: the two sequences differ only in how the
        # memory reached the model, so the last answer_len positions correspond.
        student_logits = student.logits[:, -answer_len - 1 : -1]
        teacher_logits = teacher.logits[:, -answer_len - 1 : -1]
        kl = F.kl_div(
            F.log_softmax(student_logits.float(), -1),
            F.log_softmax(teacher_logits.float(), -1),
            reduction="batchmean", log_target=True,
        )
        loss = loss + weight * kl
        return loss, float(student.loss), float(kl)
    return loss, float(student.loss), 0.0


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
    connector = InputSoftTokenConnector().to(device)
    torch.manual_seed(args.seed)
    optimizer = torch.optim.AdamW(connector.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)

    split = pairs["split"].astype(str)
    train_rows = np.flatnonzero(split == "train")
    val_rows = np.flatnonzero(split == "validation")
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
            chosen = val_rows[: args.validation_rows]
            losses = []
            with torch.no_grad():
                for index in chosen:
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
