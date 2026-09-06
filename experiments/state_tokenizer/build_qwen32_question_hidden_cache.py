"""Cache frozen Qwen3-32B question-only hidden anchors for delta distillation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from xt_ama_adapter.qwen32_bridge import file_sha256, render_ama_openend_parts
from .train_qwen32_bridge import (
    HiddenAnchorCapture,
    QUESTION_HIDDEN_PROTOCOL,
    _load_dataset,
    _reader_artifact_sha256,
    _row_parts,
    _select_split_indices,
    _visible_gpu_guard,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[16, 32, 48])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-memory-gib", type=int, default=34)
    parser.add_argument("--train-limit", type=int, default=2048)
    parser.add_argument("--train-retrieval-hit-only", action="store_true")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=35)
    args = parser.parse_args()
    _visible_gpu_guard("0,1,2")
    if args.batch_size < 1 or args.train_limit < 1:
        raise ValueError("batch-size and train-limit must be positive")
    hidden_layers = tuple(map(int, args.hidden_layers))
    if not hidden_layers or len(set(hidden_layers)) != len(hidden_layers):
        raise ValueError("hidden-layers must be non-empty and unique")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset_path = Path(args.dataset)
    model_path = Path(args.model)
    output = Path(args.output)
    arrays, metadata = _load_dataset(dataset_path)
    selected = _select_split_indices(
        arrays, "train", retrieval_hit_only=args.train_retrieval_hit_only,
        limit=args.train_limit, seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _, _, prompt_hash = render_ama_openend_parts(
        tokenizer, "{QUESTION}", enable_thinking=args.enable_thinking
    )
    print("[question-hidden] hashing the complete Qwen3-32B artifact", flush=True)
    reader_hash = _reader_artifact_sha256(model_path)
    dataset_hash = file_sha256(dataset_path)
    expected = {
        "protocol": QUESTION_HIDDEN_PROTOCOL,
        "dataset_protocol": metadata["protocol"],
        "dataset_sha256": dataset_hash,
        "reader_model_sha256": reader_hash,
        "prompt_sha256": prompt_hash,
        "enable_thinking": bool(args.enable_thinking),
        "hidden_layers": list(hidden_layers),
        "hidden_width": 5120,
        "hidden_anchor_position": "last_prompt_token_before_answer",
        "rows": len(arrays["question"]),
        "cached_split": "train",
        "cached_rows": int(len(selected)),
        "cached_indices": list(map(int, selected)),
        "train_limit": int(args.train_limit),
        "train_retrieval_hit_only": bool(args.train_retrieval_hit_only),
        "selection_seed": int(args.seed),
        "context": "question_only_no_latent_no_oracle_text",
        "official_ama_test_included": False,
        "qformer_artifact_sha256": metadata["qformer_artifact_sha256"],
        "retrieval_head_artifact_sha256": metadata["retrieval_head_artifact_sha256"],
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, value in expected.items():
            if existing.get(name) != value:
                raise ValueError(f"existing question-hidden manifest differs at {name}")
    else:
        if list(output.glob("*.npz")):
            raise ValueError("cache rows exist without a provenance manifest")
        manifest_path.write_text(
            json.dumps({**expected, "status": "building"}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    max_memory = {
        index: f"{args.max_memory_gib}GiB"
        for index in range(torch.cuda.device_count())
    }
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="balanced",
        max_memory=max_memory, low_cpu_mem_usage=True, trust_remote_code=True,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    capture = HiddenAnchorCapture(model, hidden_layers, detach_to_cpu=True)
    first_device = model.get_input_embeddings().weight.device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    completed = 0
    pending: list[tuple[int, Path, list[int]]] = []

    def flush(rows: list[tuple[int, Path, list[int]]]) -> None:
        nonlocal completed
        if not rows:
            return
        lengths = [len(row[2]) for row in rows]
        maximum = max(lengths)
        input_ids = torch.full(
            (len(rows), maximum), int(pad_id), dtype=torch.long, device=first_device
        )
        attention_mask = torch.zeros(
            (len(rows), maximum), dtype=torch.bool, device=first_device
        )
        for batch_row, ((_, _, ids), length) in enumerate(zip(rows, lengths)):
            input_ids[batch_row, :length] = torch.as_tensor(ids, device=first_device)
            attention_mask[batch_row, :length] = True
        capture.begin([length - 1 for length in lengths])
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            anchors = capture.stacked()
        capture.clear()
        for (index, target, _), hidden_anchor in zip(rows, anchors):
            temporary = target.with_suffix(".tmp.npz")
            np.savez(
                temporary,
                row_index=np.asarray(index, dtype=np.int64),
                hidden_anchor=hidden_anchor.numpy(),
            )
            temporary.replace(target)
            completed += 1
        if completed % 100 < len(rows) or completed == len(selected):
            print(f"[question-hidden] {completed}/{len(selected)}", flush=True)

    for raw_index in selected:
        index = int(raw_index)
        target = output / f"{index:06d}.npz"
        if args.resume and target.exists():
            with np.load(target, allow_pickle=False) as data:
                if int(np.asarray(data["row_index"])) != index:
                    raise ValueError(f"{target}: row index mismatch")
                hidden = np.asarray(data["hidden_anchor"])
            if hidden.shape != (len(hidden_layers), 5120):
                raise ValueError(f"{target}: malformed hidden anchor")
            completed += 1
            continue
        prefix, suffix, _answer = _row_parts(
            tokenizer, str(arrays["question"][index]),
            str(arrays["answer"][index]), args.enable_thinking,
        )
        ids = prefix + suffix
        if not ids:
            raise ValueError(f"row {index} has an empty question-only prompt")
        pending.append((index, target, ids))
        if len(pending) == args.batch_size:
            flush(pending)
            pending = []
    flush(pending)
    capture.close()
    final = {**expected, "status": "complete"}
    manifest_path.write_text(
        json.dumps(final, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
