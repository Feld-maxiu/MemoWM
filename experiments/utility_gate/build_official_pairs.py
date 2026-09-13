"""Pair every official AMA question with the state the deployed system serves it.

The utility labels need three things per question that the episode caches do not
hand over together:

* the retrieved step -- the cache already holds the retrieval-head keys
  (``document_embeddings``) and the question embeddings, so top-1 is a cosine
  over those, no model needed;
* that step's reader-side text -- the frozen wire format, regenerated from the
  official trajectory with the same adapter the cache used;
* the world-model state id for the same turn -- because the gate's codes are
  indexed by ``state_id`` in the deduplicated code table, not by
  (episode, turn).

Turns are matched by enumeration position: both the world-model records and the
cache walk the trajectory in order, so position ``i`` is the same turn even when
``turn_idx`` is irregular.

Fit rows come from train episodes, evaluation rows from validation episodes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.state_tokenizer.common import iter_jsonl
from experiments.utility_gate.label_counterfactual_ama import (
    load_codes, split_of_state,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--records", default="outputs/wm_train/ama-v1/records.jsonl")
    parser.add_argument("--states", default="outputs/wm_train/ama-v1/states.jsonl")
    parser.add_argument("--pq", default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    parser.add_argument("--dataset",
                        default="third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl")
    parser.add_argument("--retrieval-k", type=int, default=8,
                        help="retrieved screens kept per question; the anchor "
                             "block is built from these rows' step texts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    _codes, state_ids = load_codes(Path(args.pq))
    code_row = {sid: index for index, sid in enumerate(state_ids)}
    split, shared = split_of_state(Path(args.states), Path(args.records))

    # "ama:000177" -> {position: state_id}
    turn_state: dict[str, dict[int, str]] = {}
    for row in iter_jsonl(Path(args.records)):
        turn_state.setdefault(str(row["episode_id"]), {})[int(row["step"])] = str(
            row["state_id"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    dropped = {"missing_turn_state": 0, "missing_code_row": 0, "missing_step_text": 0}

    with out_path.open("w", encoding="utf-8") as handle:
        for path in sorted(Path(args.cache_dir).glob("episode-*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            key = f"ama:{int(payload['metadata']['episode_id']):06d}"
            positions = turn_state.get(key, {})
            if not positions:
                continue

            questions = [str(q) for q in payload["questions"]]
            gold = [str(a) for a in payload["gold_answers"]]
            qa_types = list(payload.get("qa_types") or [None] * len(questions))
            scores = (payload["query_embeddings"].float()
                      @ payload["document_embeddings"].float().T)
            step_texts = [str(t) for t in payload["step_texts"]]

            for index, question in enumerate(questions):
                position = int(scores[index].argmax())
                state_id = positions.get(position)
                if state_id is None:
                    dropped["missing_turn_state"] += 1
                    continue
                if state_id not in code_row:
                    dropped["missing_code_row"] += 1
                    continue
                if position >= len(step_texts):
                    dropped["missing_step_text"] += 1
                    continue
                order = scores[index].argsort(descending=True)[
                    : args.retrieval_k].tolist()
                topk = []
                for rank, candidate in enumerate(order):
                    candidate_state = positions.get(int(candidate))
                    topk.append({
                        "rank": rank,
                        "position": int(candidate),
                        "step_index": int(payload["step_indices"][candidate]),
                        "state_id": candidate_state,
                        "has_code_row": candidate_state in code_row,
                    })
                state_split = split.get(state_id, "?")
                counts[state_split] = counts.get(state_split, 0) + 1
                handle.write(json.dumps({
                    "episode_id": key,
                    "qa_index": index,
                    "question": question,
                    "answer": gold[index],
                    "qa_type": qa_types[index],
                    "retrieved_step": position,
                    "state_id": state_id,
                    "state_row": code_row[state_id],
                    "split": state_split,
                    "task_type": payload.get("task_type"),
                    "domain": payload["metadata"].get("domain"),
                    "step_text": step_texts[position],
                    "topk": topk,
                    "in_both_splits": state_id in shared,
                }, ensure_ascii=False) + "\n")

    print(json.dumps({"pairs_by_split": counts, "dropped": dropped,
                      "output": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
