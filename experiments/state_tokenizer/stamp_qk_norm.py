#!/usr/bin/env python
"""Stamp ``qk_norm`` into checkpoints written before the field existed.

QK-norm is parameter-free. That is what makes it safe to sweep on and off on
identical weights, and it is also what makes a missing metadata field
dangerous: a checkpoint trained with it loads without complaint into a module
built without it, ``strict=True`` has no key to miss, and every downstream
number is then computed through the wrong forward with nothing to raise on.

The K32e arms need this because their trainer process had already imported the
module before the metadata field was added, so no checkpoint they write will
carry it no matter how long they run.

    python -m experiments.state_tokenizer.stamp_qk_norm --true PATH [PATH ...]

Refuses to change a file that already records a conflicting value.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def stamp(path: Path, value: bool, dry_run: bool) -> str:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if metadata is None:
        return f"{path.name}: no metadata block, skipped"
    current = metadata.get("qk_norm")
    if current is not None and bool(current) != value:
        raise ValueError(
            f"{path.name} already records qk_norm={current!r}, refusing to "
            f"overwrite it with {value!r}"
        )
    if current is not None:
        return f"{path.name}: already qk_norm={current}"
    if dry_run:
        return f"{path.name}: would set qk_norm={value}"
    metadata["qk_norm"] = value
    payload["metadata"] = metadata
    torch.save(payload, path)
    return f"{path.name}: set qk_norm={value}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--true", dest="value", action="store_true")
    group.add_argument("--false", dest="value", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for raw in args.paths:
        print(" ", stamp(Path(raw), bool(args.value), args.dry_run))


if __name__ == "__main__":
    main()
