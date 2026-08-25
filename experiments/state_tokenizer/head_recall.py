"""Does a retrieval head actually retrieve? Measured the way the benchmark asks.

``train_retrieval_bridge`` reports loss, contrastive and cosine, and none of
those says whether the right row comes back. Its own objective is also easier
than the task: a batch of 64 drawn from 4,379 rows over 156 samples almost never
contains two observations from the same session, so the in-batch negatives ask
"which website is this" -- while WorldMemArena retrieval has to pick round 3
from round 7 of one session.

That distinction is not hypothetical here. Measured per session and normalized
by the n-1 ceiling that centring imposes, the Q-Former retains 59.9% of the
available within-session directions against the fixed pooling's 75.6%, while
its *global* effective rank is 138.6 against 450.1 -- it discards cross-session
variety far more aggressively than within-session detail. A global recall
number would therefore misreport it in both directions.

(An earlier draft of this file claimed the opposite -- "variance dominated by a
few website-identity directions, effective rank 7.2 over 23 samples". That came
from an ad-hoc measurement with no ceiling normalization and is retracted;
``within_sample_spread`` below is the measurement that replaced it.)

So the headline is ``same_sample_rank_metrics`` from ``wma_shift_attribution``,
which ranks each state's own teacher against the *other rows of its own sample*.
Global recall is reported next to it as a reference, not as the verdict.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
import torch

from residualmem.latent.instruct_bridge import (
    RETRIEVAL_BRIDGE_PROTOCOL,
    MaskedAttentionRetrievalHead,
    load_bridge,
)

from .train_qformer_joint import spread
from .wma_shift_attribution import same_sample_rank_metrics


def within_sample_spread(xbar: np.ndarray, samples: np.ndarray) -> dict:
    """How many directions survive *inside* one session, which is all that counts.

    The effective rank the trainer prints is measured over observations drawn
    from many different websites, so most of it can be "which site is this" --
    a question the benchmark never asks. A representation can read 41.7 there
    and still carry three usable directions within a session, and that is
    exactly the failure that would sink retrieval while the trainer's monitor
    stays quiet.

    Centring n rows caps the rank at n-1, so the raw number is meaningless
    without its ceiling; ``saturation`` is the fraction of that ceiling reached
    and is what compares across samples of different length. Both arms have
    slot*512 >> n, so neither is limited by its slot count and the 16-slot and
    64-slot numbers are comparable.
    """
    ranks, saturations, weights = [], [], []
    for name in sorted(set(samples.tolist())):
        rows = samples == name
        n = int(rows.sum())
        if n < 3:
            continue        # ceiling of 1 direction says nothing
        rank = spread(xbar[rows])["effective_rank"]
        ranks.append(rank)
        saturations.append(rank / (n - 1))
        weights.append(n)
    if not ranks:
        return {}
    total = float(sum(weights))
    return {
        "samples": len(ranks),
        "mean_rows": total / len(ranks),
        "effective_rank": float(sum(r * w for r, w in zip(ranks, weights)) / total),
        "saturation": float(sum(s * w for s, w in zip(saturations, weights)) / total),
    }


def global_rank_metrics(pred: np.ndarray, teacher: np.ndarray) -> dict:
    """The easy version, for contrast: rank against every held-out row."""
    teacher = teacher / np.maximum(np.linalg.norm(teacher, axis=1, keepdims=True), 1e-12)
    scores = pred @ teacher.T
    n = len(scores)
    order = np.argsort(-scores, axis=1)
    rank = (order == np.arange(n)[:, None]).argmax(axis=1) + 1
    return {
        "n_way": n,
        "recall_at_1": float((rank <= 1).mean()),
        "recall_at_10": float((rank <= 10).mean()),
        "chance_recall_at_1": 1.0 / n,
    }


def load_head(path: str, *, joint: bool) -> MaskedAttentionRetrievalHead:
    """Either a standalone A1 head, or the one inside a joint checkpoint.

    Both need measuring, and until now only the first was. The joint head is
    what the WorldMemArena adapter actually deploys -- it takes
    ``tokenizer.retrieval_head`` when present -- while the gate has been reading
    a head that ``train_retrieval_bridge`` fits afterwards. That gap did not
    matter while the joint head was ``1 - cos`` at batch 1 and scored R@1 of
    0.045. Under ``--sem-mode same-session`` it is trained with the same
    InfoNCE, at the same temperature, on the negatives the benchmark actually
    poses, so which head is better is now an open question rather than a
    settled one -- and the deployed one is the joint one.
    """
    head = MaskedAttentionRetrievalHead()
    if not joint:
        load_bridge(path, head, expected_protocol=RETRIEVAL_BRIDGE_PROTOCOL)
        return head
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    prefix = "retrieval_head."
    inner = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not inner:
        raise ValueError(f"{path} carries no retrieval_head.* weights (trained without L_sem)")
    head.load_state_dict(inner, strict=True)
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True)
    parser.add_argument("--joint-head", action="store_true",
                        help="--head is a QFormerStateReader checkpoint; measure "
                             "the retrieval head trained inside it, which is the "
                             "one the WMA adapter deploys")
    parser.add_argument("--cache", required=True,
                        help="must carry sample_id, or same-sample ranking is impossible")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    args = parser.parse_args()

    with np.load(args.cache, allow_pickle=True) as data:
        if "sample_id" not in data.files:
            raise ValueError(
                f"{args.cache} has no sample_id column, so rows cannot be grouped "
                "by session and only the easy global ranking is available"
            )
        keep = data["split"].astype(str) == args.split
        xbar = np.asarray(data["xbar"], np.float32)[keep]
        valid = np.asarray(data["valid"], bool)[keep]
        teacher = np.asarray(data["teacher_fused_embedding"], np.float32)[keep]
        samples = data["sample_id"].astype(str)[keep]
        domains = (data["domain"].astype(str)[keep] if "domain" in data.files
                   else np.full(len(samples), "-"))

    head = load_head(args.head, joint=args.joint_head)
    device = torch.device(args.device)
    head.to(device).eval()
    with torch.no_grad():
        predicted = head(
            torch.as_tensor(xbar, device=device),
            torch.as_tensor(valid, device=device),
        ).float().cpu().numpy()

    # Per sample, then averaged over samples weighted by their row count, so a
    # 60-round session does not count the same as a 4-round one.
    per_sample, weights = [], []
    for name in sorted(set(samples)):
        rows = samples == name
        if int(rows.sum()) < 2:
            continue        # a one-row sample has nothing to rank against
        per_sample.append(same_sample_rank_metrics(predicted[rows], teacher[rows]))
        weights.append(int(rows.sum()))
    if not per_sample:
        raise ValueError("no sample has two or more rows")
    total = float(sum(weights))
    within = {
        key: float(sum(m[key] * w for m, w in zip(per_sample, weights)) / total)
        for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr",
                    "chance_recall_at_1", "n_way")
    }

    report = {
        "head": str(Path(args.head).resolve()),
        "head_source": "joint" if args.joint_head else "standalone",
        "cache": str(Path(args.cache).resolve()),
        "split": args.split,
        "rows": int(len(xbar)),
        "samples": len(per_sample),
        "slots": int(xbar.shape[1]),
        "within_sample": within,
        "within_sample_spread": within_sample_spread(xbar, samples),
        "global": global_rank_metrics(predicted, teacher),
        "global_spread": spread(xbar),
        "by_domain": {
            name: global_rank_metrics(predicted[domains == name], teacher[domains == name])
            for name in sorted(set(domains.tolist())) if (domains == name).sum() > 1
        } if len(set(domains.tolist())) > 1 else {},
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2))

    print(f"{Path(args.head).name}  ({report['rows']} rows / {report['samples']} samples, "
          f"{report['slots']} slots)")
    w, g = report["within_sample"], report["global"]
    print(f"  within-sample  R@1 {w['recall_at_1']:.4f}  R@5 {w['recall_at_5']:.4f}  "
          f"R@10 {w['recall_at_10']:.4f}  MRR {w['mrr']:.4f}"
          f"   (avg {w['n_way']:.1f}-way, chance R@1 {w['chance_recall_at_1']:.4f})")
    print(f"  global         R@1 {g['recall_at_1']:.4f}  R@10 {g['recall_at_10']:.4f}"
          f"   ({g['n_way']}-way, chance {g['chance_recall_at_1']:.5f})  <- reference only")
    ws, gs = report["within_sample_spread"], report["global_spread"]
    if ws:
        print(f"  within-sample  eff.rank {ws['effective_rank']:.1f} "
              f"({ws['saturation']:.1%} of the n-1 ceiling, {ws['samples']} samples, "
              f"{ws['mean_rows']:.1f} rows each)")
    print(f"  global         eff.rank {gs['effective_rank']:.1f}  "
          f"cos {gs['pairwise_cosine']:+.4f}  mean/dev {gs['mean_to_deviation']:.2f}")
    for name, metrics in report["by_domain"].items():
        print(f"    [{name}] global R@1 {metrics['recall_at_1']:.4f} ({metrics['n_way']}-way)")


if __name__ == "__main__":
    main()
