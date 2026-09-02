"""Generate many questions per observation, so E_q[U] becomes estimable.

The utility a gate needs is ``E_q[U]`` -- how much a code is worth averaged over
the questions that will be asked. Half the variance of U sits in "same code,
different question", and each observation in the fit corpus carries only 2.75
questions, so the cell mean has a split-half reliability of 0.12. No feature and
no model can predict a target that noisy; the ceiling is the reliability itself.

Spearman-Brown puts the single-question reliability at about 0.047, so:

    2.75 questions -> 0.12      20 questions -> 0.50      47 questions -> 0.70

The lever is therefore questions per observation, not observations. This
generates them.

**The answer is not generated, it is elicited.** The model writes a question;
the gold answer is then produced by the same model *reading the full
observation*. That makes the pair self-consistent by construction -- the answer
is answerable from this screen, because it was read off this screen -- and it
makes the resulting CE term a question-conditioned observation distillation
rather than supervision. That is worth stating plainly: these are not labels.
What they buy is selectivity. A description probe elicits a generic summary; a
question forces the latent to keep one specific fact, and which facts get kept
is what the utility measurement is about.

Style is copied from the corpus rather than invented. Real WorldMemArena
questions of the requested type go in as exemplars, and the measured shape --
22-40 words, answers around 50 words ending in a period -- is stated in the
prompt and enforced afterwards. Questions that come back too short, too long, or
duplicated within an observation are dropped rather than repaired.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from experiments.state_tokenizer.extract_qwen import _load_model, prepare_inputs

# Types a single observation can support. The corpus is dominated by
# trajectory-level questions -- temporal_reasoning, time_to_live and friends are
# 94% cross-frame -- and asking those of one screenshot would produce questions
# the frame cannot answer, which is noise in U rather than signal.
SINGLE_FRAME_TYPES = (
    "visual_factual_recall", "visual_search", "factual_recall",
    "cross_modal_reasoning", "memory_change",
)

ASK = (
    "Write {n} different questions that can be answered by looking at this "
    "screen alone. Follow these rules exactly:\n"
    "- the answer must be readable off this screen. Do not ask what something "
    "means, what it is normally used for, or anything answerable from general "
    "knowledge without looking;\n"
    "- ask about {n} different parts of the screen -- do not ask about the same "
    "element twice;\n"
    "- every question must begin with What, Which, Where, How many, or How. "
    "Never write a yes/no question;\n"
    "- each question is one sentence of 22 to 40 words;\n"
    "- vary how the questions open; do not start them all the same way;\n"
    "- write only the questions, one per line, with no numbering.\n\n"
    "Here are real questions of the kind wanted, for style only -- do not copy "
    "their content:\n{exemplars}\n"
)

# A yes/no question is answered by "Yes" plus a restatement of the premise, so
# almost none of its gold-answer NLL depends on the screen -- it is a weak probe
# of utility, and the corpus being imitated has essentially none of the form.
YES_NO = re.compile(r"^(is|are|was|were|does|do|did|can|could|has|have|had|will"
                    r"|would|should|may|might)\b", re.IGNORECASE)
OPEN = re.compile(r"^(what|which|where|how)\b", re.IGNORECASE)

ANSWER = (
    "Answer the question from what is visible on this screen. Write one to three "
    "sentences, about 50 words, ending with a period. If the screen does not "
    "show it, say so plainly.\nQuestion: {question}"
)


def exemplars_by_type(pairs_path: Path, per_type: int = 2) -> dict:
    pairs = np.load(pairs_path, allow_pickle=True)
    chosen: dict[str, list[str]] = {}
    for question, kind in zip(pairs["question"], pairs["question_type"]):
        kind = str(kind)
        if kind not in SINGLE_FRAME_TYPES:
            continue
        chosen.setdefault(kind, [])
        if len(chosen[kind]) < per_type:
            chosen[kind].append(str(question))
    return chosen


def observation_records(xbar_dir: Path) -> dict:
    records: dict[str, list[dict]] = {}
    for path in sorted(xbar_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata"])))
        records[path.stem] = metadata.get("records") or []
    return records


def generate(model, processor, image, dom, prompt, *, max_new_tokens,
             temperature, device, seed):
    inputs, _truncated, _kept, _length = prepare_inputs(
        processor, image, dom, prompt, 8192, "instruct", enable_thinking=False)
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    # This transformers build takes no `generator`; seeding globally is the only
    # handle on sampling, so it is set per call to keep runs reproducible.
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            **({"temperature": temperature, "top_p": 0.95} if temperature > 0 else {}),
        )
    text = processor.tokenizer.decode(
        out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip()


def clean_questions(text: str, want: int, low: int, high: int) -> list[str]:
    seen, kept = set(), []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        if not line.endswith("?"):
            continue
        if YES_NO.match(line) or not OPEN.match(line):
            continue
        words = len(line.split())
        if not low <= words <= high:
            continue
        key = " ".join(line.lower().split()[:6])
        if key in seen:                        # same opening -> same template
            continue
        seen.add(key)
        kept.append(line)
        if len(kept) >= want:
            break
    return kept


# Two ways a generated pair carries no utility, both visible in the answer.
#
# A hallucinated premise -- "which icon shows a red ghost face" when there is
# none -- gets a truthful denial back. The denial is a negation of the question,
# so almost none of its NLL depends on which codes survive.
#
# A world-knowledge question -- "what does This PC typically represent" -- is
# answered without consulting the screen at all, so omitting any code cannot
# change it. Its utility is zero by construction.
#
# An earlier version tested the answer against the observation's caption
# instead. That was the wrong instrument: the model answers from the screenshot,
# while the caption is one short paragraph (38 distinct content words at the
# median), so legitimate visual detail -- icon colours, positions -- has nowhere
# to match and the filter rejected good pairs at four times the rate of bad ones.
DENIAL = re.compile(
    r"\b(there (is|are) no|does not (show|display|contain|appear)|"
    r"do not (show|display|contain|appear)|no (application|icon|button|element|"
    r"text|window|visible)|not visible|cannot be seen|is not present)\b",
    re.IGNORECASE)
GENERIC = re.compile(r"\b(typically|commonly|generally|usually|in general|"
                     r"is used to|are used to|refers to)\b", re.IGNORECASE)


def usable_answer(answer: str) -> str | None:
    """``None`` if the pair is fine, else the reason it was dropped."""
    if len(answer.split()) < 8:
        return "short"
    if DENIAL.search(answer):
        return "denial"
    if GENERIC.search(answer):
        return "generic"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbar-dir", required=True)
    parser.add_argument("--pairs", required=True, help="for style exemplars")
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--observations", type=int, default=100)
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--min-words", type=int, default=20)
    parser.add_argument("--max-words", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    exemplars = exemplars_by_type(Path(args.pairs))
    records = observation_records(Path(args.xbar_dir))
    usable = [(sample, index)
              for sample, rows in sorted(records.items())
              for index, row in enumerate(rows)
              if row.get("synthetic_axtree") and row.get("screenshot")
              and Path(row["screenshot"]).exists()]
    rng = np.random.default_rng(args.seed)
    usable = [usable[int(i)] for i in rng.permutation(len(usable))]
    usable = usable[: args.observations][args.shard_index::args.shard_count]
    print(f"[gen] {len(usable)} observations, {args.questions} questions each",
          flush=True)

    device = torch.device(args.device)
    processor, model = _load_model(args.model, device, False)

    rows: list[dict] = []
    dropped: Counter = Counter()
    kinds = list(exemplars)
    for done, (sample, index) in enumerate(usable):
        record = records[sample][index]
        dom = record["synthetic_axtree"]
        with Image.open(record["screenshot"]) as handle:
            image = handle.convert("RGB")
        # Two passes with different type conditioning, so the questions are not
        # all drawn from one exemplar's basin.
        questions: list[str] = []
        for pass_index in range(3):
            kind = kinds[(done + pass_index) % len(kinds)]
            prompt = ASK.format(
                n=args.questions + 10,
                exemplars="\n".join(f"- {e}" for e in exemplars[kind]))
            text = generate(model, processor, image, dom, prompt,
                            max_new_tokens=64 * args.questions,
                            temperature=args.temperature, device=device,
                            seed=args.seed + 1000 * done + pass_index)
            questions.extend(clean_questions(
                text, args.questions, args.min_words, args.max_words))
            if len(questions) >= args.questions:
                break
        seen, unique = set(), []
        for question in questions:
            key = " ".join(question.lower().split()[:6])
            if key not in seen:
                seen.add(key)
                unique.append(question)
        questions = unique[: args.questions]

        for question in questions:
            answer = generate(model, processor, image, dom,
                              ANSWER.format(question=question),
                              max_new_tokens=128, temperature=0.0,
                              device=device, seed=0)
            answer = " ".join(answer.split())
            reason = usable_answer(answer)
            if reason:
                dropped[reason] += 1
                continue
            rows.append({"sample_id": sample, "record_index": index,
                         "question": question, "answer": answer,
                         "question_type": "generated"})
        if done % 5 == 0:
            print(f"[gen] {done + 1}/{len(usable)}  {len(rows)} pairs", flush=True)

    if not rows:
        raise SystemExit("nothing generated")
    per_observation = Counter((r["sample_id"], r["record_index"]) for r in rows)
    np.savez_compressed(
        args.output,
        sample_id=np.asarray([r["sample_id"] for r in rows]),
        record_index=np.asarray([r["record_index"] for r in rows], np.int32),
        question=np.asarray([r["question"] for r in rows]),
        answer=np.asarray([r["answer"] for r in rows]),
        question_type=np.asarray([r["question_type"] for r in rows]),
        split=np.asarray(["generated"] * len(rows)),
        metadata=json.dumps({
            "protocol": "residualmem_generated_qa_v1",
            "purpose": "raise questions-per-observation so E_q[U] is estimable",
            "answers": "elicited from the model reading the full observation; "
                       "not annotations",
            "observations": len(per_observation),
            "questions_per_observation_mean": float(np.mean(list(per_observation.values()))),
            "temperature": args.temperature, "seed": args.seed,
            "word_range": [args.min_words, args.max_words],
        }, ensure_ascii=False),
    )
    print(json.dumps({
        "pairs": len(rows), "observations": len(per_observation),
        "per_observation_mean": float(np.mean(list(per_observation.values()))),
        "per_observation_min": int(min(per_observation.values())),
        "dropped": dict(dropped),
    }, indent=2))


if __name__ == "__main__":
    main()
