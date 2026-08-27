"""Where does the K=64 resampler first go non-finite?

Two runs at 64 queries diverged: ``lr 1e-4`` produced NaN between steps 500 and
800, and ``lr 3e-5`` survived to step 3250 and then produced the same
``xbar contains non-finite values``. Both times the response was to lower the
learning rate and eventually to abandon the slot count. Neither time was the
cause looked at, so "K=64 is unstable" has been sitting in the worklog as a
brute fact for two days while the slot-count question kept coming back.

That error is raised by the connector's input check, which means it is the
*last* place the corruption is visible, not the first. This walks the forward
with hooks and reports the first tensor that stops being finite, plus the
running magnitude of every intermediate and of the gradients, so the failure has
a location and a shape instead of a symptom.

Three candidates it is built to distinguish:

* **no warmup** -- the trainer builds a bare ``AdamW`` with no scheduler, so the
  first optimizer step takes the full learning rate. Sixty-four queries mean a
  4x larger query table and four times the attention rows of the configuration
  that trains cleanly. Signature: magnitudes climb monotonically from step one.
* **bf16 overflow** -- the resampler runs in the projection's dtype end to end
  with no fp32 promotion around the attention softmax. Signature: a single
  block's output jumps to inf while its inputs are still ordinary.
* **a degenerate observation** -- one input that is pathological regardless of
  slot count. Signature: the step that fails is reproducible by sample id, and
  K=16 fails on it too.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from residualmem.latent.instruct_bridge import InputSoftTokenConnector
from residualmem.latent.qformer import QFormerStateReader, StateQFormer

from .extract_qwen import _load_model
from .reader_losses import answer_ce_and_distill_kl
from .train_qformer_joint import ObservationStore, PAIRS_PROTOCOL
from .trunk_states import NUM_MODALITIES, collate, trunk_states


def finite_report(tensor: torch.Tensor) -> dict:
    finite = torch.isfinite(tensor)
    values = tensor[finite].float()
    return {
        "finite": bool(finite.all()),
        "nonfinite": int((~finite).sum()),
        "absmax": float(values.abs().max()) if values.numel() else math.inf,
        "rms": float(values.pow(2).mean().sqrt()) if values.numel() else math.inf,
    }


def attach_probes(module: torch.nn.Module, seen: dict) -> list:
    """Record the first non-finite output, and every module's magnitude."""
    handles = []

    def make(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                return
            report = finite_report(tensor.detach())
            seen["magnitudes"][name] = report["absmax"]
            if not report["finite"] and seen["first_nonfinite"] is None:
                seen["first_nonfinite"] = {"module": name, **report}
        return hook

    for name, child in module.named_modules():
        if name and not list(child.children()):
            handles.append(child.register_forward_hook(make(name)))
    return handles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--xbar-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--qformer-hidden", type=int, default=1024)
    parser.add_argument("--qformer-heads", type=int, default=8)
    parser.add_argument("--qformer-layers", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--distill-weight", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=5.0)
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="linear warmup; 0 reproduces the trainer, which has "
                             "no scheduler at all")
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--output")
    args = parser.parse_args()

    pairs = np.load(args.pairs, allow_pickle=False)
    metadata = json.loads(str(np.asarray(pairs["metadata"])))
    if metadata.get("protocol") != PAIRS_PROTOCOL:
        raise ValueError(f"pairs protocol {metadata.get('protocol')!r}")

    device = torch.device(args.device)
    processor, model = _load_model(args.model, device, False)
    torch.manual_seed(args.seed)
    joint = QFormerStateReader(
        StateQFormer(num_queries=args.queries, hidden=args.qformer_hidden,
                     heads=args.qformer_heads, layers=args.qformer_layers,
                     modalities=NUM_MODALITIES, self_attention=False),
        InputSoftTokenConnector(slots=args.queries),
        None,
    ).to(device)
    optimizer = torch.optim.AdamW(joint.parameters(), lr=args.learning_rate)
    store = ObservationStore(args.xbar_dir)
    rng = np.random.default_rng(args.seed)

    split = pairs["split"].astype(str)
    train_rows = np.flatnonzero(split == "train")
    seen = {"first_nonfinite": None, "magnitudes": {}}
    handles = attach_probes(joint, seen)

    history = []
    trace = None
    for step in range(1, args.max_steps + 1):
        if args.warmup_steps:
            scale = min(1.0, step / args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * scale
        joint.train()
        optimizer.zero_grad(set_to_none=True)
        rows = [int(rng.choice(train_rows)) for _ in range(args.accumulate)]
        losses = []
        for row in rows:
            sample_id = str(pairs["sample_id"][row])
            index = int(pairs["record_index"][row])
            record = store.observation(sample_id, index)
            with Image.open(record["screenshot"]) as handle:
                image = handle.convert("RGB")
            states = trunk_states(processor, model, image, record["synthetic_axtree"],
                                  device=device)
            soft, xbar, valid = joint(*collate([states]))
            if not torch.isfinite(xbar).all():
                trace = {
                    "step": step, "sample_id": sample_id, "record_index": index,
                    "first_nonfinite": seen["first_nonfinite"],
                    "magnitudes": dict(sorted(
                        seen["magnitudes"].items(), key=lambda kv: -kv[1]
                    )[:12]),
                }
                break
            loss, ce, kl = answer_ce_and_distill_kl(
                model, processor, soft, valid,
                str(pairs["question"][row]), str(pairs["answer"][row]),
                str(record.get("fused_text", "")), weight=args.distill_weight,
            )
            (loss / args.accumulate).backward()
            losses.append((ce, kl))
        if trace is not None:
            break
        grad = torch.nn.utils.clip_grad_norm_(joint.parameters(), args.clip_norm)
        optimizer.step()
        if step % args.report_every == 0 or step == 1:
            with torch.no_grad():
                query_absmax = float(joint.qformer.queries.abs().max())
                xbar_absmax = float(xbar.abs().max())
            row = {
                "step": step,
                "ce": float(np.mean([c for c, _ in losses])),
                "kl": float(np.mean([k for _, k in losses])),
                "grad_norm": float(grad),
                "query_absmax": query_absmax,
                "xbar_absmax": xbar_absmax,
                "top_module_absmax": max(seen["magnitudes"].values()),
            }
            history.append(row)
            print(f"[nan] step {step:5d}  ce {row['ce']:.4f}  kl {row['kl']:.4f}  "
                  f"|g| {row['grad_norm']:9.2f}  queries {query_absmax:8.3f}  "
                  f"xbar {xbar_absmax:9.3f}  peak {row['top_module_absmax']:.3e}",
                  flush=True)
    for handle in handles:
        handle.remove()

    report = {
        "queries": args.queries, "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps, "clip_norm": args.clip_norm,
        "diverged": trace is not None, "trace": trace, "history": history,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))
    if trace is None:
        print(f"\n[nan] survived {args.max_steps} steps at "
              f"queries={args.queries} lr={args.learning_rate} "
              f"warmup={args.warmup_steps}")
    else:
        print(f"\n[nan] DIVERGED at step {trace['step']} on "
              f"{trace['sample_id']}[{trace['record_index']}]")
        print(f"  first non-finite module: {trace['first_nonfinite']}")
        print("  largest magnitudes at that moment:")
        for name, value in trace["magnitudes"].items():
            print(f"    {value:12.3e}  {name}")


if __name__ == "__main__":
    main()
