"""Generate HumanTrajs validation answers through a frozen Qwen3-32B bridge.

This is an evaluation-only, resumable five-arm gate.  It never reads the test
split and never updates the reader or bridge.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import unicodedata
from pathlib import Path

import numpy as np
import torch

from xt_ama_adapter.qwen32_bridge import (
    file_sha256,
    latent_position_ids,
    load_qwen32_bridge,
    render_ama_openend_parts,
)
from .train_qwen32_bridge import (
    DATA_PROTOCOL,
    _load_dataset,
    _reader_artifact_sha256,
    _shuffle_partner,
)


PROTOCOL = "qwen32_bridge_generation_eval_v1"
ARMS = ("matched", "shuffled", "zero", "question_only", "oracle_text")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    return " ".join(text.split())


def _token_f1(prediction: str, gold: str) -> float:
    predicted, expected = _normalize(prediction).split(), _normalize(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((collections.Counter(predicted) & collections.Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _clean_prediction(text: str) -> str:
    value = text.strip()
    if "</think>" in value:
        value = value.rsplit("</think>", 1)[-1].strip()
    value = re.sub(r"^\s*(?:Answer\s*\[?1\]?|Answer)\s*:\s*", "", value,
                   flags=re.IGNORECASE)
    return value.strip()


def _prompt_row(model, tokenizer, bridge, arrays, index: int, source: int,
                arm: str, enable_thinking: bool, latent_output_scale: float):
    question = str(arrays["question"][index])
    prefix, suffix, _ = render_ama_openend_parts(
        tokenizer, question, enable_thinking=enable_thinking
    )
    embedding = model.get_input_embeddings()
    device, dtype = embedding.weight.device, embedding.weight.dtype
    prefix_ids = torch.tensor(
        [tokenizer.encode(prefix, add_special_tokens=False)], device=device
    )
    suffix_ids = torch.tensor(
        [tokenizer.encode(suffix, add_special_tokens=False)], device=device
    )
    pieces = [embedding(prefix_ids)]
    masks = [torch.ones(prefix_ids.shape, dtype=torch.bool, device=device)]
    if arm in {"matched", "shuffled", "zero"}:
        latent, valid = bridge(
            torch.as_tensor(arrays["xbar"][source:source + 1],
                            dtype=torch.float32, device=device),
            torch.as_tensor(arrays["valid"][source:source + 1],
                            dtype=torch.bool, device=device),
        )
        latent = latent * latent_output_scale
        if arm == "zero":
            latent = torch.zeros_like(latent)
        pieces.append(latent.to(dtype))
        masks.append(valid)
    elif arm == "oracle_text":
        oracle_ids = torch.tensor(
            [tokenizer.encode(str(arrays["oracle_text"][index]),
                              add_special_tokens=False)], device=device
        )
        pieces.append(embedding(oracle_ids))
        masks.append(torch.ones(oracle_ids.shape, dtype=torch.bool, device=device))
    elif arm != "question_only":
        raise ValueError(f"unknown arm {arm!r}")
    pieces.append(embedding(suffix_ids))
    masks.append(torch.ones(suffix_ids.shape, dtype=torch.bool, device=device))
    return torch.cat(pieces, 1), torch.cat(masks, 1)


def _left_pad(rows):
    maximum = max(int(embedding.shape[1]) for embedding, _ in rows)
    padded_embeddings, padded_masks = [], []
    for embedding, mask in rows:
        amount = maximum - int(embedding.shape[1])
        if amount:
            zeros = torch.zeros(
                (1, amount, embedding.shape[-1]), dtype=embedding.dtype,
                device=embedding.device,
            )
            embedding = torch.cat((zeros, embedding), 1)
            mask = torch.cat((torch.zeros((1, amount), dtype=torch.bool,
                                          device=mask.device), mask), 1)
        padded_embeddings.append(embedding)
        padded_masks.append(mask)
    embeddings = torch.cat(padded_embeddings, 0)
    masks = torch.cat(padded_masks, 0)
    return embeddings, masks


def _load_existing(path: Path):
    records = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    keys = {(str(row["arm"]), int(row["row_index"])) for row in records}
    return records, keys


def _summarize(records):
    report = {}
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm]
        if not rows:
            continue
        report[arm] = {
            "rows": len(rows),
            "normalized_exact_match": float(np.mean([row["exact_match"] for row in rows])),
            "token_f1": float(np.mean([row["token_f1"] for row in rows])),
            "gold_contained": float(np.mean([row["gold_contained"] for row in rows])),
            "empty_rate": float(np.mean([not row["prediction"].strip() for row in rows])),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-memory-gib", type=int, default=34)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only the first N validation rows; 0 means all")
    parser.add_argument("--retrieval-hit-only", action="store_true")
    parser.add_argument("--latent-output-scale", type=float, default=1.0,
                        help="Inference-only multiplier for bridge soft tokens")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--allow-prompt-mismatch", action="store_true",
                        help="Diagnostic only: evaluate a checkpoint under a new prompt")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size and max-new-tokens must be positive")
    if args.limit < 0 or not np.isfinite(args.latent_output_scale) or args.latent_output_scale <= 0:
        raise ValueError("limit must be nonnegative and latent-output-scale positive")
    visible = ",".join(x.strip() for x in
                       (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if x.strip())
    if visible != "0,1,2":
        raise RuntimeError(f"evaluation requires CUDA_VISIBLE_DEVICES=0,1,2; got {visible!r}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset_path, bridge_path, model_path = map(
        Path, (args.dataset, args.bridge, args.model)
    )
    arrays, metadata = _load_dataset(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as data:
        if "oracle_text" not in data.files:
            raise ValueError("generation evaluation requires oracle_text")
        arrays["oracle_text"] = np.asarray(data["oracle_text"])
    if metadata.get("protocol") != DATA_PROTOCOL:
        raise ValueError("dataset protocol mismatch")
    validation = np.flatnonzero(arrays["split"].astype(str) == "validation")
    if args.retrieval_hit_only:
        if "retrieval_gold_hit" not in arrays:
            raise ValueError("retrieval-hit evaluation needs retrieval_gold_hit")
        validation = validation[arrays["retrieval_gold_hit"][validation].astype(bool)]
    if not len(validation):
        raise ValueError("validation split is empty")
    partner = _shuffle_partner(validation, arrays["trajectory_id"].astype(str))
    if args.limit:
        validation = validation[:args.limit]
    bridge, bridge_metadata = load_qwen32_bridge(
        bridge_path,
        expected_qformer_sha256=metadata["qformer_artifact_sha256"],
        expected_retrieval_head_sha256=metadata["retrieval_head_artifact_sha256"],
    )
    if bridge_metadata["data_manifest_sha256"] != file_sha256(dataset_path):
        raise ValueError("bridge dataset hash mismatch")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _, _, prompt_hash = render_ama_openend_parts(
        tokenizer, "{QUESTION}", enable_thinking=args.enable_thinking
    )
    prompt_mismatch = bridge_metadata["prompt_sha256"] != prompt_hash
    if prompt_mismatch and not args.allow_prompt_mismatch:
        raise ValueError("bridge prompt hash mismatch")
    print("[generation-eval] hashing the complete Qwen3-32B reader artifact", flush=True)
    reader_hash = _reader_artifact_sha256(model_path)
    if bridge_metadata["reader_model_sha256"] != reader_hash:
        raise ValueError("bridge reader hash mismatch")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="balanced",
        max_memory={i: f"{args.max_memory_gib}GiB"
                    for i in range(torch.cuda.device_count())},
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bridge = bridge.to(model.get_input_embeddings().weight.device).eval()

    output_path, report_path = Path(args.output), Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records, completed = _load_existing(output_path)
    with torch.inference_mode(), output_path.open("a", encoding="utf-8") as handle:
        for arm in args.arms:
            pending = [int(i) for i in validation if (arm, int(i)) not in completed]
            for start in range(0, len(pending), args.batch_size):
                indices = pending[start:start + args.batch_size]
                rows = [
                    _prompt_row(model, tokenizer, bridge, arrays, index,
                                partner[index] if arm == "shuffled" else index,
                                arm, args.enable_thinking, args.latent_output_scale)
                    for index in indices
                ]
                inputs_embeds, attention_mask = _left_pad(rows)
                sequences = model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=latent_position_ids(attention_mask),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=(tokenizer.pad_token_id
                                  if tokenizer.pad_token_id is not None
                                  else tokenizer.eos_token_id),
                )
                decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                for index, raw in zip(indices, decoded):
                    prediction = _clean_prediction(raw)
                    gold = str(arrays["answer"][index]).strip()
                    normalized_prediction, normalized_gold = _normalize(prediction), _normalize(gold)
                    record = {
                        "protocol": PROTOCOL,
                        "arm": arm,
                        "row_index": index,
                        "trajectory_id": str(arrays["trajectory_id"][index]),
                        "question": str(arrays["question"][index]),
                        "gold": gold,
                        "prediction": prediction,
                        "exact_match": normalized_prediction == normalized_gold,
                        "token_f1": _token_f1(prediction, gold),
                        "gold_contained": bool(normalized_gold and
                                               normalized_gold in normalized_prediction),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    records.append(record)
                    completed.add((arm, index))
                print(f"[generation-eval] {arm} {start + len(indices)}/{len(pending)}", flush=True)

    summary = {
        "protocol": PROTOCOL,
        "split": "validation",
        "validation_rows": len(validation),
        "retrieval_hit_only": bool(args.retrieval_hit_only),
        "official_ama_test_included": False,
        "dataset_sha256": file_sha256(dataset_path),
        "bridge_sha256": file_sha256(bridge_path),
        "reader_model_sha256": reader_hash,
        "prompt_sha256": prompt_hash,
        "checkpoint_prompt_sha256": bridge_metadata["prompt_sha256"],
        "prompt_mismatch_diagnostic": bool(prompt_mismatch),
        "enable_thinking": bool(args.enable_thinking),
        "max_new_tokens": args.max_new_tokens,
        "latent_output_scale": args.latent_output_scale,
        "requested_arms": list(args.arms),
        "arms": _summarize(records),
    }
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                      sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
