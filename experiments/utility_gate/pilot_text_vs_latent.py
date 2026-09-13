"""Pilot: how many official AMA questions actually depend on the latent row?

The AMA reader prompt puts two kinds of memory in front of the model: the
retrieved step's latent soft tokens and that step's raw observation text
(``memory_mode="matched"``, the configuration every published AMA number was
produced under). A question whose answer is readable from the *text* row -- and
many are, since the text is the uncompressed observation -- cannot tell us
anything about the utility of the code positions: dropping codes is free by
construction, and a mask fitted on such questions would report a free lunch.

So before labelling anything, measure the marginal value of the latent row:

    latent gain = NLL(gold | text only) - NLL(gold | latent + text)

Only questions with a real gain carry information about the codes. This script
runs that measurement over an existing episode-cache directory and prints the
distribution, so the threshold and the surviving sample size are chosen from
data rather than assumed.

Runs under the torch/vLLM venv.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.utility_gate.label_counterfactual_ama import (
    load_ama_reader, latent_position_ids, place_bridge,
)


def load_reader(reader_model: str, bridge: str, dtype: str, device: str):
    tokenizer, model = load_ama_reader(reader_model, dtype, device)
    bridge_module, metadata = place_bridge(Path(bridge), model)
    return tokenizer, model, bridge_module, metadata


def score_segments(model, tokenizer, bridge, question: str, answer: str,
                   segments, *, enable_thinking: bool,
                   max_answer_tokens: int, max_model_len: int | None,
                   max_new_tokens: int | None) -> float:
    """Teacher-forced gold-answer NLL (bits) for one assembled memory prompt."""
    from xt_ama_adapter.qwen32_bridge import Qwen32LatentReader

    reader = Qwen32LatentReader(
        model, tokenizer, bridge, enable_thinking=bool(enable_thinking))
    if max_model_len is not None and max_new_tokens is not None:
        inputs, _audit = reader.build_inputs_segments(
            question, segments, max_model_len=max_model_len,
            max_new_tokens=max_new_tokens)
    else:
        inputs, _audit = reader.build_inputs_segments(question, segments)

    embeds = inputs["inputs_embeds"]
    attention = inputs["attention_mask"]
    device = embeds.device
    embed = model.get_input_embeddings()
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)[:max_answer_tokens]
    answer_len = len(answer_ids)
    if answer_len == 0:
        return float("nan")
    answer_embeds = embed(torch.tensor([answer_ids], dtype=torch.long, device=device))

    full = torch.cat((embeds, answer_embeds), 1)
    mask = torch.cat((
        attention,
        torch.ones((1, answer_len), dtype=attention.dtype, device=device)), 1)
    kwargs = {"use_cache": False}
    import inspect
    parameters = inspect.signature(model.forward).parameters
    if "position_ids" in parameters:
        kwargs["position_ids"] = latent_position_ids(mask)
    if "logits_to_keep" in parameters:
        kwargs["logits_to_keep"] = answer_len + 1
    with torch.no_grad():
        out = model(inputs_embeds=full.to(embed.weight.dtype), attention_mask=mask,
                    **kwargs)
        logits = out.logits[:, -answer_len - 1:-1, :].float()
        logprob = torch.nn.functional.log_softmax(logits, -1)
        target = torch.tensor([answer_ids], dtype=torch.long, device=logits.device)
        gathered = logprob.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(-gathered.sum() / math.log(2.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--reader-model", default="/data1/models/Qwen3-32B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-questions-per-episode", type=int, default=0)
    parser.add_argument("--max-answer-tokens", type=int, default=104)
    parser.add_argument("--max-model-len", type=int, default=32000)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    files = sorted(Path(args.cache_dir).glob("episode-*.pt"))
    if args.max_episodes:
        files = files[: args.max_episodes]
    print(f"[pilot] {len(files)} episode caches", flush=True)

    tokenizer, model, bridge, metadata = load_reader(
        args.reader_model, args.bridge, args.dtype, args.device)
    enable_thinking = bool(metadata.get("enable_thinking"))

    records = []
    for path in files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        questions = [str(q) for q in payload["questions"]]
        gold = [str(a) for a in payload["gold_answers"]]
        documents = payload["document_embeddings"].float()
        queries = payload["query_embeddings"].float()
        xbar = payload["xbar"].float()
        valid = payload["valid"].bool()
        texts = [str(t) for t in payload["step_texts"]]
        scores = queries @ documents.T
        # Retrieval is within the episode, exactly as the answer path does it.
        top1 = scores.argmax(-1)
        episode_id = int(payload["metadata"]["episode_id"])
        order = range(len(questions))
        if args.max_questions_per_episode:
            order = list(order)[: args.max_questions_per_episode]
        for index in order:
            step = int(top1[index])
            matched = score_segments(
                model, tokenizer, bridge, questions[index], gold[index],
                [("latent", (xbar[step], valid[step])), ("text", texts[step])],
                enable_thinking=enable_thinking,
                max_answer_tokens=args.max_answer_tokens,
                max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens)
            text_only = score_segments(
                model, tokenizer, bridge, questions[index], gold[index],
                [("text", texts[step])],
                enable_thinking=enable_thinking,
                max_answer_tokens=args.max_answer_tokens,
                max_model_len=args.max_model_len,
                max_new_tokens=args.max_new_tokens)
            records.append({
                "episode_id": episode_id, "qa_index": index, "step": step,
                "task_type": payload.get("task_type"),
                "matched_nll_bits": matched, "text_only_nll_bits": text_only,
                "latent_gain_bits": text_only - matched,
                "answer_len": len(tokenizer.encode(
                    gold[index], add_special_tokens=False)[: args.max_answer_tokens]),
            })
            print(f"[pilot] ep{episode_id} q{index} step{step} "
                  f"matched {matched:.3f} text-only {text_only:.3f} "
                  f"gain {text_only - matched:+.3f}", flush=True)

    gains = np.asarray([r["latent_gain_bits"] for r in records])
    summary = {
        "questions": len(records),
        "latent_gain_bits": {
            "mean": float(gains.mean()),
            "median": float(np.median(gains)),
            "p10": float(np.percentile(gains, 10)),
            "p90": float(np.percentile(gains, 90)),
            "max": float(gains.max()),
            "min": float(gains.min()),
        },
        "survivors": {str(t): int((gains >= t).sum())
                      for t in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)},
        "matched_nll_bits_mean": float(np.mean(
            [r["matched_nll_bits"] for r in records])),
        "text_only_nll_bits_mean": float(np.mean(
            [r["text_only_nll_bits"] for r in records])),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records}, indent=1,
                              ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
