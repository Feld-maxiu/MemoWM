"""Pair WorldMemArena questions with the observation they are about.

The connector trained by text reconstruction learns to make its latent a prefix
that regenerates the observation text. At inference we ask it something else --
answer a question from this latent -- and the mismatch shows: the latent pulls
the model back toward reconstruction, so it answers and then keeps going,
narrating the observation. Measured on a 12-question smoke, hallucination went
from 0% in the text condition to 33% in the latent one.

Real question/answer pairs remove that mismatch at the source. Evidence links
them to observations exactly: 82% of non-web questions carry an ``image_id``,
and every extracted observation records the ``image_ids`` it was built from, so
the join is at observation level rather than the session-level fallback the
benchmark's own matcher settles for.

**Only non-evaluation subcategories.** The 27 ``agent/gui/web`` samples supply
nothing here -- not their questions, answers, memory points, or evidence.
Note what this does to the claim: the retrieval side was adapted on WorldMemArena
*observations*, which is domain adaptation. Training on its questions is
training on the benchmark's task, and the 12 question types are shared with the
evaluation split, so answer style transfers. That has to be stated plainly, and
next to the fact that the Raw-Fused control is trained on nothing at all.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np

EVAL_SUBCATEGORY = "web"


def question_index(dataset: Path, subcategories: list[str]) -> dict[str, list[dict]]:
    """``image_id -> [{question, answer, type}]`` over non-evaluation samples."""
    by_image: dict[str, list[dict]] = collections.defaultdict(list)
    for subcategory in subcategories:
        if subcategory == EVAL_SUBCATEGORY:
            raise ValueError("the evaluation subcategory must not be indexed")
        for path in sorted(glob.glob(str(dataset / "agent" / "gui" / subcategory / "*.json"))):
            payload = json.loads(Path(path).read_text())
            for checkpoint in payload.get("qa_checkpoints") or []:
                for question in checkpoint.get("questions") or []:
                    images = [
                        str(entry["image_id"])
                        for entry in (question.get("evidence") or [])
                        if entry.get("image_id")
                    ]
                    if not images or not question.get("answer"):
                        continue
                    record = {
                        "question": str(question["question"]),
                        "answer": str(question["answer"]),
                        "question_type": str(question.get("question_type", "")),
                        "evidence_images": len(images),
                    }
                    for image_id in images:
                        by_image[image_id].append(record)
    return by_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--xbar-dir", required=True, help="the non-web fit corpus")
    parser.add_argument("--output", required=True)
    parser.add_argument("--subcategory", action="append", default=None)
    parser.add_argument("--max-per-observation", type=int, default=4,
                        help="cap so a heavily-questioned observation cannot dominate")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()

    subcategories = args.subcategory or [
        "excel", "file_mgmt", "image_edit", "mobile", "webarena_lite", "word_docs"
    ]
    by_image = question_index(Path(args.dataset), subcategories)
    print(f"[qa] {len(by_image)} images carry questions across {len(subcategories)} subcategories")

    xbars, valids, questions, answers, samples, kinds = [], [], [], [], [], []
    matched_images = set()
    for path in sorted(Path(args.xbar_dir).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata"])))
            keys = sorted(k for k in data.files if k.startswith("m11/xbar/"))
            records = metadata.get("records") or []
            if len(records) != len(keys):
                raise ValueError(f"{path.name}: {len(records)} records, {len(keys)} states")
            for key, record in zip(keys, records):
                hits: list[dict] = []
                for image_id in record.get("image_ids") or []:
                    hits.extend(by_image.get(str(image_id), []))
                    if str(image_id) in by_image:
                        matched_images.add(str(image_id))
                if not hits:
                    continue
                index = key.rsplit("/", 1)[1]
                xbar = np.asarray(data[key], np.float32)
                valid = np.asarray(data[f"m11/valid/{index}"], bool)
                for hit in hits[: args.max_per_observation]:
                    xbars.append(xbar)
                    valids.append(valid)
                    questions.append(hit["question"])
                    answers.append(hit["answer"])
                    samples.append(path.stem)
                    kinds.append(hit["question_type"])

    if not xbars:
        raise RuntimeError("no observation matched a question by image_id")

    # Split by sample, as everywhere else here: observations inside one sample
    # come from consecutive rounds and are heavily correlated.
    unique = sorted(set(samples))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(unique))
    held = {unique[i] for i in order[: max(1, int(round(len(unique) * args.validation_fraction)))]}
    split = np.asarray(["validation" if s in held else "train" for s in samples])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        xbar=np.stack(xbars),
        valid=np.stack(valids),
        question=np.asarray(questions),
        answer=np.asarray(answers),
        sample_id=np.asarray(samples),
        question_type=np.asarray(kinds),
        split=split,
        metadata=np.asarray(json.dumps({
            "protocol": "wma_reader_qa_v1",
            "subcategories": subcategories,
            "excluded": EVAL_SUBCATEGORY,
            "max_per_observation": args.max_per_observation,
            "seed": args.seed,
            "images_with_questions": len(by_image),
            "images_matched": len(matched_images),
        })),
    )
    counts = collections.Counter(split.tolist())
    print(f"\n{len(xbars)} pairs over {len(unique)} samples  {dict(counts)}")
    print(f"images matched: {len(matched_images)}/{len(by_image)}")
    top = collections.Counter(kinds).most_common(5)
    print("question types: " + ", ".join(f"{k or '?'}={v}" for k, v in top))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
