"""Do the world model's codes and the reader's latents live in the same space?

The AMA pipeline encodes each state twice, from two different texts:

* the world model's codes come from ``states.jsonl``, whose text is the raw
  *observation* (``residualmem_text_wm_qformer_encode_v1``);
* the reader's latents come from the cached ``step_text``, which is the frozen
  wire format ``"AMA Step N:\\nAction: ...\\nObservation:\\n..."``.

Same Q-Former checkpoint, different input text, so the two latents are close but
not identical (measured cosine 0.95-0.98 per state). The utility gate governs the
world model's codes; the reader consumes the bridge's output. If feeding a
decoded world-model state through the bridge degrades the reader beyond the
deployed latent, then a mask fitted on those codes is not measuring the deployed
system and the gap has to be reported, not hidden.

This script scores the same (question, retrieved step) twice -- once with the
cached latent, once with the world model's code reconstruction -- and prints the
difference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.state_tokenizer.common import iter_jsonl
from experiments.utility_gate.label_counterfactual_ama import (
    load_ama_reader, load_codes, load_posteriors, place_bridge,
)
from experiments.utility_gate.pilot_text_vs_latent import score_segments
from experiments.utility_gate.verify_scorer import load_codebook, rebuild


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=(
        "outputs/ama_latent_memory/ama_eval/web-latent-formal/cache"))
    parser.add_argument("--bridge", default=(
        "outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms.pt"))
    parser.add_argument("--pq", default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    parser.add_argument("--codebook",
                        default="outputs/wm_train/ama-v1/ama-pq-c64-v2codebook.npz")
    parser.add_argument("--records", default="outputs/wm_train/ama-v1/records.jsonl")
    parser.add_argument("--reader-model", default="/data1/models/Qwen3-32B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--episodes", type=int, nargs="+", default=[177, 178])
    parser.add_argument("--questions-per-episode", type=int, default=2)
    parser.add_argument("--max-answer-tokens", type=int, default=104)
    parser.add_argument("--max-model-len", type=int, default=32000)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    codes, state_ids = load_codes(Path(args.pq))
    by_id = {sid: i for i, sid in enumerate(state_ids)}
    book = load_codebook(Path(args.codebook))
    records = list(iter_jsonl(Path(args.records)))
    # (episode_id string, step) -> unique state_id, i.e. the world model's view.
    wm_state = {}
    for row in records:
        key = (str(row["episode_id"]), int(row["step"]))
        wm_state.setdefault(key, str(row["state_id"]))

    tokenizer, model = load_ama_reader(args.reader_model, args.dtype, args.device)
    bridge, metadata = place_bridge(Path(args.bridge), model)
    enable_thinking = bool(metadata.get("enable_thinking"))

    from xt_ama_adapter.adapter import adapt_ama_trajectory

    official = {}
    for row in iter_jsonl(Path("third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl")):
        official[int(row["episode_id"])] = row

    results = []
    for episode_id in args.episodes:
        episode = official[episode_id]
        payload = torch.load(
            Path(args.cache_dir) / f"episode-{episode_id:06d}.pt",
            map_location="cpu", weights_only=False)
        adapted = adapt_ama_trajectory(
            episode["trajectory"], episode_id=str(episode_id),
            task=str(episode.get("task", "")))
        step_texts = [r.step_text for r in adapted]
        step_index = payload["step_indices"].tolist()
        xbar = payload["xbar"].float()
        valid = payload["valid"].bool()
        questions = [str(q) for q in payload["questions"]]
        gold = [str(a) for a in payload["gold_answers"]]
        scores = payload["query_embeddings"].float() @ payload["document_embeddings"].float().T
        top1 = scores.argmax(-1)

        for index in range(min(args.questions_per_episode, len(questions))):
            step = int(top1[index])
            turn = int(step_index[step])
            state_id = wm_state.get((f"ama:{episode_id:06d}", turn))
            row = {"episode_id": episode_id, "qa_index": index, "step": step,
                   "turn": turn, "state_id": state_id}
            if state_id is None or state_id not in by_id:
                row["skipped"] = "no world-model state for this turn"
                results.append(row)
                print(f"[space] ep{episode_id} q{index} {row['skipped']}", flush=True)
                continue

            # The world model's reconstruction of the same state: codes -> x_hat.
            reconstructed = rebuild(codes[by_id[state_id]][None], book)[0]
            deployed = xbar[step].numpy().astype(np.float32)
            row["latent_cosine"] = float(
                (deployed * reconstructed).sum()
                / (np.linalg.norm(deployed) * np.linalg.norm(reconstructed) + 1e-9))

            common = dict(enable_thinking=enable_thinking,
                          max_answer_tokens=args.max_answer_tokens,
                          max_model_len=args.max_model_len,
                          max_new_tokens=args.max_new_tokens)
            text_row = step_texts[step]
            nll_deployed = score_segments(
                model, tokenizer, bridge, questions[index], gold[index],
                [("latent", (torch.as_tensor(deployed), valid[step])),
                 ("text", text_row)], **common)
            nll_wm = score_segments(
                model, tokenizer, bridge, questions[index], gold[index],
                [("latent", (torch.as_tensor(reconstructed), valid[step])),
                 ("text", text_row)], **common)
            nll_text_only = score_segments(
                model, tokenizer, bridge, questions[index], gold[index],
                [("text", text_row)], **common)
            row.update({"nll_deployed": nll_deployed, "nll_wm_codes": nll_wm,
                        "nll_text_only": nll_text_only,
                        "penalty_of_wm_codes": nll_wm - nll_deployed})
            results.append(row)
            print(f"[space] ep{episode_id} q{index} cos {row['latent_cosine']:.4f} | "
                  f"deployed {nll_deployed:.2f} | wm-codes {nll_wm:.2f} | "
                  f"text-only {nll_text_only:.2f} | penalty "
                  f"{row['penalty_of_wm_codes']:+.2f} bits", flush=True)

    penalties = [r["penalty_of_wm_codes"] for r in results if "penalty_of_wm_codes" in r]
    cosines = [r["latent_cosine"] for r in results if "latent_cosine" in r]
    summary = {
        "rows": len(results),
        "cosine_mean": float(np.mean(cosines)) if cosines else None,
        "penalty_bits_mean": float(np.mean(penalties)) if penalties else None,
        "penalty_bits_max": float(np.max(penalties)) if penalties else None,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": results}, indent=1,
                              ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
