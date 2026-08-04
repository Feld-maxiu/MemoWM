"""Build the episode-unique 3,500-state v2 reference subset."""
from __future__ import annotations

import argparse
import json

from .common import iter_jsonl
from .v2_data import select_subset, write_subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--validation", type=int, default=500)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    source = list(iter_jsonl(args.records))
    split_sizes = {"train": args.train, "validation": args.validation, "test": args.test}
    subset = select_subset(source, split_sizes, args.seed)
    summary = write_subset(
        subset, args.output, source_manifest=args.records, seed=args.seed
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
