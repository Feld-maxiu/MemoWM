"""Train Input/L16 connectors by BrowserGym observation-text reconstruction.

The Qwen3.5-Instruct backbone is frozen. Only the connector is optimized, and
WorldMemArena data is never read. This is a practical native-reader warm-up;
retrieval quality remains governed by the independently trained shared head.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from residualmem.latent.instruct_bridge import (
    BRIDGE_CACHE_PROTOCOL,
    READER_BRIDGE_PROTOCOL,
    InputSoftTokenConnector,
    Layer16Connector,
    Layer16Restorer,
    save_bridge,
)

from .build_instruct_bridge_cache import browsergym_teacher_text
from .common import iter_jsonl
from .extract_qwen import _load_model
from .train_retrieval_bridge import load_cache


def _targets(records_path: str | Path, indices: np.ndarray) -> list[str]:
    records = list(iter_jsonl(records_path))
    by_index = {int(row["global_index"]): row for row in records}
    missing = [int(index) for index in indices if int(index) not in by_index]
    if missing:
        raise KeyError(f"records miss global indices {missing[:20]}")
    return [browsergym_teacher_text(by_index[int(index)]) for index in indices]


def _forward_loss(model, processor, connector, mode, xbar, valid, texts, max_tokens):
    encoded = processor.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_tokens, add_special_tokens=True,
    )
    device = xbar.device
    ids = encoded["input_ids"].to(device)
    text_mask = encoded["attention_mask"].to(device)
    text_embeddings = model.get_input_embeddings()(ids).detach()
    latent = connector(xbar, valid)
    prefix = latent.shape[1]
    attention = torch.cat((valid.to(text_mask.dtype), text_mask), 1)
    labels = torch.cat((
        torch.full((len(xbar), prefix), -100, device=device, dtype=torch.long),
        ids.masked_fill(~text_mask.bool(), -100),
    ), 1)
    hook_handle = None
    if mode == "input":
        inputs_embeds = torch.cat((latent.to(text_embeddings.dtype), text_embeddings), 1)
    else:
        inputs_embeds = torch.cat((torch.zeros_like(latent), text_embeddings), 1)
        target_layer = model.model.language_model.layers[16]

        def replace_prefix(_module, args):
            hidden = args[0]
            updated = hidden.clone()
            updated[:, :prefix] = latent.to(updated.dtype)
            return (updated, *args[1:])

        hook_handle = target_layer.register_forward_pre_hook(replace_prefix)
    try:
        output = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention,
            labels=labels,
            use_cache=False,
        )
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    return output.loss


def train(args: argparse.Namespace) -> dict:
    cache = load_cache(args.cache)
    metadata = cache["metadata"]
    if metadata.get("protocol") != BRIDGE_CACHE_PROTOCOL:
        raise ValueError("reader cache protocol mismatch")
    records_path = args.records or metadata["records"]
    with np.load(args.cache, allow_pickle=False) as raw:
        indices = np.asarray(raw["global_indices"], np.int64)
    texts = _targets(records_path, indices)
    sources = (
        [cache["xbar"], cache["a2_xbar"]]
        if args.representation == "both"
        else [cache["a2_xbar"] if args.representation == "a2" else cache["xbar"]]
    )
    split = cache["split"].astype(str)
    train_rows = np.flatnonzero(split == "train")
    val_rows = np.flatnonzero(split == "validation")
    device = torch.device(args.device)
    processor, model = _load_model(args.model, device, args.use_kernels)
    if args.mode == "input":
        connector = InputSoftTokenConnector()
    else:
        connector = Layer16Connector(Layer16Restorer.from_artifacts(
            args.normalization, args.pca
        ))
    connector.to(device).train()
    optimizer = torch.optim.AdamW(connector.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    best, best_step, stale = math.inf, 0, 0

    def loss_for(source, rows, training):
        values = torch.as_tensor(source[rows], dtype=torch.float32, device=device)
        valid = torch.as_tensor(cache["valid"][rows], dtype=torch.bool, device=device)
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            return _forward_loss(
                model, processor, connector, args.mode, values, valid,
                [texts[int(row)] for row in rows], args.max_target_tokens,
            )

    history = []
    for step in range(1, args.max_steps + 1):
        rows = rng.choice(train_rows, min(args.batch_size, len(train_rows)), replace=False)
        connector.train()
        optimizer.zero_grad(set_to_none=True)
        loss = sum(loss_for(source, rows, True) for source in sources) / len(sources)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(connector.parameters(), args.clip_norm)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.max_steps:
            connector.eval()
            chosen = val_rows[:min(args.validation_states, len(val_rows))]
            losses = []
            for start in range(0, len(chosen), args.batch_size):
                block = chosen[start:start + args.batch_size]
                losses.extend(float(loss_for(source, block, False)) for source in sources)
            validation = float(np.mean(losses))
            history.append({"step": step, "validation_loss": validation})
            if validation < best:
                best, best_step, stale = validation, step, 0
                save_bridge(
                    args.output, connector, protocol=READER_BRIDGE_PROTOCOL,
                    mode=args.mode, representation=args.representation,
                    best_step=step, validation_loss=validation,
                )
            else:
                stale += 1
                if stale >= args.patience_evals:
                    break
    report = {
        "protocol": READER_BRIDGE_PROTOCOL,
        "mode": args.mode,
        "representation": args.representation,
        "best_step": best_step,
        "best_validation_loss": best,
        "history": history,
    }
    Path(args.output).with_suffix(".json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--records")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("input", "layer16"), required=True)
    parser.add_argument("--representation", choices=("xbar", "a2", "both"), default="both")
    parser.add_argument("--pca")
    parser.add_argument("--normalization")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-target-tokens", type=int, default=256)
    parser.add_argument("--validation-states", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience-evals", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--use-kernels", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.mode == "layer16" and not (args.pca and args.normalization):
        parser.error("layer16 mode requires --pca and --normalization")
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
