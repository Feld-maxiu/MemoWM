"""Run one disjoint shard of the official WMA Web ResidualMem evaluation.

This wrapper changes only which samples are handed to ``run_eval``.  Each
sample still follows the official session, retrieval, native Reader, and Judge
paths.  It exists so two GPUs can evaluate disjoint sample sets without loading
the 9B Reader once per individual ``--sample-index`` invocation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wma-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--baseline", default="ResidualMem-Instruct-Xbar-Input-RAG"
    )
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    wma_root = Path(args.wma_root).resolve()
    if str(wma_root) not in sys.path:
        sys.path.insert(0, str(wma_root))
    from eval_framework.cli import run_eval
    from eval_framework.config import EvalConfig
    from eval_framework.datasets.wma_bundle import EvalBundle
    from eval_framework.datasets.worldmemarena import load_worldmemarena

    dataset = wma_root / "WorldMemArena"

    def load_shard(path: Path) -> EvalBundle:
        bundle = load_worldmemarena(path, split="all")
        web = tuple(
            sample for sample in bundle.samples
            if getattr(sample, "_subcategory", None) == "agent/arena/web"
        )
        selected = web[args.shard_index::args.num_shards]
        if not selected:
            raise ValueError("selected WMA shard is empty")
        print(
            f"[codec-shard {args.shard_index}/{args.num_shards}] "
            f"{len(selected)} samples: "
            + ", ".join(sample.sample_id for sample in selected),
            flush=True,
        )
        return EvalBundle(samples=selected)

    config = EvalConfig(
        dataset_path=dataset,
        output_dir=Path(args.output).resolve(),
        baseline=args.baseline,
    )
    run_eval(
        config,
        load_domain_bundle=load_shard,
        max_eval_workers=args.workers,
    )


if __name__ == "__main__":
    main()
