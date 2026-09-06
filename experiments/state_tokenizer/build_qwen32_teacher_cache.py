"""Cache Qwen3-32B oracle-text answer distributions for bridge distillation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from xt_ama_adapter.qwen32_bridge import file_sha256, render_ama_openend_parts

from .train_qwen32_bridge import (
    DATA_PROTOCOL,
    HiddenAnchorCapture,
    _load_dataset,
    _reader_artifact_sha256,
    _row_parts,
    _select_split_indices,
    _token_ids,
    _visible_gpu_guard,
)


PROTOCOL = "qwen32_oracle_text_answer_topk_hidden_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[16, 32, 48])
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--max-memory-gib", type=int, default=34)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--train-retrieval-hit-only", action="store_true")
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()
    _visible_gpu_guard("0,1,2")
    if args.top_k < 1 or not 0 < args.min_coverage <= 1 or args.batch_size < 1:
        raise ValueError("invalid top-k or coverage threshold")
    hidden_layers = tuple(map(int, args.hidden_layers))
    if not hidden_layers or len(set(hidden_layers)) != len(hidden_layers):
        raise ValueError("hidden layers must be a non-empty unique list")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset_path, model_path, output = Path(args.dataset), Path(args.model), Path(args.output)
    arrays, dataset_metadata = _load_dataset(dataset_path)
    selected_train = _select_split_indices(
        arrays, "train", retrieval_hit_only=args.train_retrieval_hit_only,
        limit=args.train_limit, seed=args.seed,
    )
    selected_set = set(map(int, selected_train))
    if "oracle_text" not in arrays:
        # _load_dataset intentionally returns only its required core columns;
        # read the verified text separately without weakening core validation.
        with np.load(dataset_path, allow_pickle=False) as raw:
            if "oracle_text" not in raw.files:
                raise ValueError("bridge dataset has no verified oracle_text column")
            oracle_text = np.asarray(raw["oracle_text"]).astype(str)
    else:
        oracle_text = arrays["oracle_text"].astype(str)
    if len(oracle_text) != len(arrays["question"]) or np.any(oracle_text == ""):
        raise ValueError("oracle text is empty or misaligned")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _, _, template_prompt_hash = render_ama_openend_parts(
        tokenizer, "{QUESTION}", enable_thinking=args.enable_thinking
    )
    print("[teacher-cache] hashing the complete Qwen3-32B artifact", flush=True)
    reader_hash = _reader_artifact_sha256(model_path)
    dataset_hash = file_sha256(dataset_path)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    expected_manifest = {
        "protocol": PROTOCOL,
        "dataset_protocol": dataset_metadata["protocol"],
        "dataset_sha256": dataset_hash,
        "reader_model_sha256": reader_hash,
        "prompt_sha256": template_prompt_hash,
        "enable_thinking": bool(args.enable_thinking),
        "top_k": args.top_k,
        "generation_batch_size": int(args.batch_size),
        "hidden_layers": list(hidden_layers),
        "hidden_width": 5120,
        "hidden_anchor_position": "last_prompt_token_before_answer",
        "rows": len(arrays["question"]),
        "cached_split": "train",
        "cached_rows": int(len(selected_train)),
        "cached_indices": list(map(int, selected_train)),
        "train_limit": int(args.train_limit),
        "train_retrieval_hit_only": bool(args.train_retrieval_hit_only),
        "selection_seed": int(args.seed),
        "teacher_context": "verified_pseudo_web_text_observation",
        "official_ama_test_included": False,
        "qformer_artifact_sha256": dataset_metadata["qformer_artifact_sha256"],
        "retrieval_head_artifact_sha256": dataset_metadata[
            "retrieval_head_artifact_sha256"
        ],
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, value in expected_manifest.items():
            if existing.get(name) != value:
                raise ValueError(f"existing teacher-cache manifest differs at {name}")
    else:
        stale_rows = list(output.glob("*.npz"))
        if stale_rows:
            raise ValueError(
                "teacher-cache rows exist without a provenance manifest; refusing resume"
            )
        manifest_path.write_text(
            json.dumps({**expected_manifest, "status": "building"}, indent=2,
                       sort_keys=True),
            encoding="utf-8",
        )

    max_memory = {index: f"{args.max_memory_gib}GiB"
                  for index in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="balanced",
        max_memory=max_memory, low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval()
    hidden_capture = HiddenAnchorCapture(
        model, hidden_layers, detach_to_cpu=True
    )
    first_device = model.get_input_embeddings().weight.device
    coverages = []
    completed = 0
    pending = []

    def flush(rows):
        nonlocal completed
        if not rows:
            return
        lengths = [len(row[2]) for row in rows]
        max_length = max(lengths)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        input_ids = torch.full(
            (len(rows), max_length), int(pad_id), dtype=torch.long,
            device=first_device,
        )
        attention_mask = torch.zeros(
            (len(rows), max_length), dtype=torch.bool, device=first_device
        )
        boundaries = []
        for row, length in zip(rows, lengths):
            ids, answer_ids = row[2], row[3]
            input_ids[row[4], :length] = torch.as_tensor(ids, device=first_device)
            attention_mask[row[4], :length] = True
            boundaries.append(length - len(answer_ids) - 1)
        hidden_capture.begin(boundaries)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            ).logits
            hidden_anchors = hidden_capture.stacked()
        hidden_capture.clear()
        for row, length, hidden_anchor in zip(rows, lengths, hidden_anchors):
            index, target, _ids, answer_ids, batch_row = row
            answer_logits = logits[
                batch_row, length - len(answer_ids) - 1:length - 1
            ].float()
            logprob = torch.log_softmax(answer_logits, dim=-1)
            values, indices = torch.topk(
                logprob, k=min(args.top_k, logprob.shape[-1]), dim=-1
            )
            coverage = values.exp().sum(-1)
            coverage_np = coverage.cpu().numpy().astype(np.float32)
            coverages.extend(coverage_np.tolist())
            temporary = target.with_suffix(".tmp.npz")
            np.savez(
                temporary,
                row_index=np.asarray(index, dtype=np.int64),
                answer_tokens=np.asarray(len(answer_ids), dtype=np.int64),
                topk_index=indices.to(torch.int32).cpu().numpy(),
                topk_logprob=values.to(torch.float16).cpu().numpy(),
                coverage=coverage_np,
                hidden_anchor=hidden_anchor.numpy(),
            )
            temporary.replace(target)
            completed += 1
        if completed % 100 < len(rows) or completed == expected_manifest["cached_rows"]:
            print(f"[teacher-cache] {completed}/{expected_manifest['cached_rows']}",
                  flush=True)

    for index, (question, answer, memory) in enumerate(zip(
        arrays["question"].astype(str), arrays["answer"].astype(str), oracle_text
    )):
        if index not in selected_set:
            continue
        target = output / f"{index:06d}.npz"
        if args.resume and target.exists():
            with np.load(target, allow_pickle=False) as cached:
                if int(np.asarray(cached["row_index"])) != index:
                    raise ValueError(f"{target}: row index mismatch")
                coverage = np.asarray(cached["coverage"], dtype=np.float32)
                hidden_anchor = np.asarray(cached["hidden_anchor"])
                if hidden_anchor.shape != (len(hidden_layers), 5120):
                    raise ValueError(f"{target}: malformed hidden anchor")
            coverages.extend(coverage.tolist())
            completed += 1
            continue
        prefix, suffix, answer_ids = _row_parts(
            tokenizer, question, answer, args.enable_thinking
        )
        memory_ids = _token_ids(tokenizer, memory)
        ids = prefix + memory_ids + suffix + answer_ids
        pending.append((index, target, ids, answer_ids, len(pending)))
        if len(pending) == args.batch_size:
            flush(pending)
            pending = []

    flush(pending)

    coverage_array = np.asarray(coverages, dtype=np.float32)
    if not len(coverage_array):
        raise ValueError("teacher cache contains no answer positions")
    final_manifest = {
        **expected_manifest,
        "status": "complete",
        "answer_positions": int(len(coverage_array)),
        "coverage_min": float(coverage_array.min()),
        "coverage_median": float(np.median(coverage_array)),
        "coverage_p05": float(np.quantile(coverage_array, 0.05)),
        "min_coverage_required": args.min_coverage,
        "coverage_pass": bool(np.median(coverage_array) >= args.min_coverage),
    }
    manifest_path.write_text(json.dumps(final_manifest, indent=2, sort_keys=True),
                             encoding="utf-8")
    hidden_capture.close()
    print(json.dumps(final_manifest, indent=2, sort_keys=True))
    if not final_manifest["coverage_pass"]:
        raise RuntimeError("teacher top-k median retained mass is below threshold")


if __name__ == "__main__":
    main()
