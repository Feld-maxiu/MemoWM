"""Build question/answer pairs for the held-out web samples -- evaluation only.

``build_qformer_qa_pairs`` refuses to index ``agent/gui/web``, and that refusal is
correct: those 27 samples are the official evaluation set, and letting their
questions, answers or evidence reach anything that is fitted would quietly
invalidate every number computed downstream. That guard is left alone.

This is the other half of the same discipline. A mask fitted on the fit corpus
has to be *measured* somewhere out of domain, and the web samples are exactly
that. The pairs written here are consumed by the evaluation curve and nothing
else -- nothing is fitted on them, no threshold is chosen from them, and they
never enter a training loop.

The join is ``image_id``: the converter records each observation's image ids, and
the dataset's questions carry their evidence as image ids. State ids follow
``convert_wma``'s ``f"{sample_id}-{index:04d}"``, so the sample and the record
index can be read straight back off them.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np

SUBCATEGORY = "web"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="the directory holding agent/gui")
    parser.add_argument("--records", required=True,
                        help="records.jsonl from the test-set build")
    parser.add_argument("--max-per-observation", type=int, default=4,
                        help="the same cap the fit-corpus pairs use, so the two "
                             "are comparable per observation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    by_image: dict[str, list[dict]] = collections.defaultdict(list)
    pattern = str(Path(args.dataset) / "agent" / "gui" / SUBCATEGORY / "*.json")
    samples = sorted(glob.glob(pattern))
    if not samples:
        raise SystemExit(f"no samples under {pattern}")
    for path in samples:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for checkpoint in payload.get("qa_checkpoints") or []:
            for question in checkpoint.get("questions") or []:
                images = [str(entry["image_id"])
                          for entry in (question.get("evidence") or [])
                          if entry.get("image_id")]
                if not images or not question.get("answer"):
                    continue
                record = {
                    "question": str(question["question"]),
                    "answer": str(question["answer"]),
                    "question_type": str(question.get("question_type", "")),
                }
                for image_id in images:
                    by_image[image_id].append(record)

    rows: list[dict] = []
    used = 0
    for line in Path(args.records).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        state_id = str(entry["state_id"])
        sample_id, _, index = state_id.rpartition("-")
        matched: list[dict] = []
        for image_id in entry.get("image_ids") or []:
            matched.extend(by_image.get(str(image_id), []))
        if not matched:
            continue
        used += 1
        for record in matched[: args.max_per_observation]:
            rows.append({"sample_id": sample_id, "record_index": int(index), **record})

    if not rows:
        raise SystemExit("no question matched any observation")
    np.savez_compressed(
        args.output,
        sample_id=np.asarray([r["sample_id"] for r in rows]),
        record_index=np.asarray([r["record_index"] for r in rows], np.int32),
        question=np.asarray([r["question"] for r in rows]),
        answer=np.asarray([r["answer"] for r in rows]),
        question_type=np.asarray([r["question_type"] for r in rows]),
        split=np.asarray(["test"] * len(rows)),
        metadata=json.dumps({
            "protocol": "wma_qformer_qa_eval_only_v1",
            "subcategory": SUBCATEGORY,
            "purpose": "evaluation only; never fitted on",
            "samples": len(samples), "observations_used": used,
            "pairs": len(rows), "max_per_observation": args.max_per_observation,
        }, ensure_ascii=False),
    )
    print(json.dumps({"samples": len(samples), "observations_used": used,
                      "pairs": len(rows),
                      "unique_questions": len({r["question"] for r in rows})},
                     indent=2))


if __name__ == "__main__":
    main()
