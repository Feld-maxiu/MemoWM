"""QA pairs for the Q-Former, carrying references rather than latents.

``build_reader_qa_pairs`` writes the 64x512 ``xbar`` inline, once per question,
so the shipped file is 1.22 GB for 9,041 pairs over 3,182 distinct observations.
A learnable tokenizer cannot use a precomputed ``xbar`` at all -- it needs the
*inputs* the trunk saw -- so this stores ``(sample_id, record_index)`` and lets
the trainer resolve the observation from the extraction npz. The result is a few
megabytes and, more importantly, one copy of each observation rather than four.

Resolving requires the extraction to have persisted ``synthetic_axtree``; runs
before that field was added carry only ``synthetic_axtree_chars`` and are
rejected here rather than silently producing an empty tree.

**Only non-evaluation subcategories.** The 27 ``agent/gui/web`` samples supply
nothing -- not their questions, answers, memory points, or evidence.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from .build_reader_qa_pairs import EVAL_SUBCATEGORY, question_index

PROTOCOL = "wma_qformer_qa_v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--xbar-dir", required=True,
                        help="an extraction that persisted synthetic_axtree")
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
    print(f"[qformer-qa] {len(by_image)} images carry questions "
          f"across {len(subcategories)} subcategories")

    samples, indices, questions, answers, kinds = [], [], [], [], []
    matched_images, observations = set(), 0
    for path in sorted(Path(args.xbar_dir).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata"])))
        records = metadata.get("records") or []
        missing_tree = [i for i, r in enumerate(records) if not r.get("synthetic_axtree")]
        if missing_tree:
            raise ValueError(
                f"{path.name}: {len(missing_tree)} records have no synthetic_axtree. "
                "Re-run wma_extract_xbar with the field persisted -- the trainer has "
                "to re-run the trunk over the identical input, and an empty tree "
                "would change H_t silently."
            )
        for index, record in enumerate(records):
            hits: list[dict] = []
            for image_id in record.get("image_ids") or []:
                hits.extend(by_image.get(str(image_id), []))
                if str(image_id) in by_image:
                    matched_images.add(str(image_id))
            if not hits:
                continue
            observations += 1
            for hit in hits[: args.max_per_observation]:
                samples.append(path.stem)
                indices.append(index)
                questions.append(hit["question"])
                answers.append(hit["answer"])
                kinds.append(hit["question_type"])

    if not samples:
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
        sample_id=np.asarray(samples),
        record_index=np.asarray(indices, np.int32),
        question=np.asarray(questions),
        answer=np.asarray(answers),
        question_type=np.asarray(kinds),
        split=split,
        metadata=np.asarray(json.dumps({
            "protocol": PROTOCOL,
            "xbar_dir": str(Path(args.xbar_dir).resolve()),
            "subcategories": subcategories,
            "excluded": EVAL_SUBCATEGORY,
            "max_per_observation": args.max_per_observation,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "images_with_questions": len(by_image),
            "images_matched": len(matched_images),
            "observations_used": observations,
        })),
    )
    counts = collections.Counter(split.tolist())
    print(f"\n{len(samples)} pairs over {observations} observations / "
          f"{len(unique)} samples  {dict(counts)}")
    print(f"images matched: {len(matched_images)}/{len(by_image)}")
    top = collections.Counter(kinds).most_common(5)
    print("question types: " + ", ".join(f"{k or '?'}={v}" for k, v in top))
    print(f"wrote {args.output} "
          f"({Path(args.output).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
