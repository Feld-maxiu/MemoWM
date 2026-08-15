"""Rebuild only the action channel of the frozen cache.

Action semantics do not change the state encoding: ``codes`` and ``valid`` are
bit-identical to the source cache. So instead of re-running 100,008 states
through A1/A2 -- which would mean reading 55 GB of features and re-deriving
values we already have -- this copies those arrays verbatim and rebuilds only
the transition archive from the records manifest.

Copying rather than re-encoding also makes "the codes were not disturbed" a
constructive fact rather than something to be checked afterwards.

The new cache is written to a separate directory; the frozen v8 cache is never
modified.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from experiments.state_tokenizer.common import sha256_file, write_json

from ..cache import (
    CACHE_FILES,
    V8_EXPECTED_STATES,
    V8_EXPECTED_TRANSITIONS,
    _minimal_records,
    build_transition_archive,
)
from ..schema import CACHE_FORMAT_VERSION, PROTOCOL

COPIED = ("codes", "valid", "global_indices")


def run(args: argparse.Namespace) -> dict:
    source = Path(args.source)
    output = Path(args.output)
    if source.resolve() == output.resolve():
        raise ValueError("refusing to rebuild in place; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)

    # 1. Carry the state encoding across untouched.
    copied = {}
    for name in COPIED:
        src = source / CACHE_FILES[name]
        dst = output / CACHE_FILES[name]
        shutil.copy2(src, dst)
        digest = sha256_file(dst)
        if digest != sha256_file(src):
            raise RuntimeError(f"{name}: copy does not match source")
        copied[name] = digest
    print(f"copied {len(COPIED)} state arrays unchanged", flush=True)

    # 2. Rebuild the transition archive, now carrying target semantics.
    records, state_counts = _minimal_records(args.records)
    if state_counts != V8_EXPECTED_STATES:
        raise ValueError(f"state counts {state_counts} != {V8_EXPECTED_STATES}")
    print(f"parsed {len(records)} records", flush=True)
    transitions = build_transition_archive(
        records, output / CACHE_FILES["transitions"],
        expected_counts=V8_EXPECTED_TRANSITIONS,
    )

    # Fields describing the state encoding are inherited verbatim: those arrays
    # were copied, not rebuilt, so anything derived from them must not change.
    source_manifest = json.loads(
        (source / CACHE_FILES["manifest"]).read_text(encoding="utf-8")
    )
    inherited = {
        key: source_manifest[key]
        for key in ("layout", "arrays", "numerics", "tasks", "episode_names",
                    "episodes", "policy_names")
        if key in source_manifest
    }
    manifest = {
        "protocol": PROTOCOL,
        "format_version": CACHE_FORMAT_VERSION,
        **inherited,
        "derived_from": {
            "cache": str(source.resolve()),
            "manifest_sha256": sha256_file(source / CACHE_FILES["manifest"]),
            "note": (
                "codes/valid/global_indices copied verbatim; only the action "
                "channel was rebuilt, so the state encoding is unchanged"
            ),
        },
        "records": str(Path(args.records).resolve()),
        "records_sha256": sha256_file(args.records),
        "state_counts": state_counts,
        "transition_counts": transitions.get("counts", V8_EXPECTED_TRANSITIONS),
        "transitions": sum(V8_EXPECTED_TRANSITIONS.values()),
        "artifact_sha256": {
            **copied,
            "transitions": sha256_file(output / CACHE_FILES["transitions"]),
        },
        "test_locked": True,
    }
    write_json(output / CACHE_FILES["manifest"], manifest)
    print(f"wrote {output}", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/world_model/v8/cache")
    parser.add_argument(
        "--records", default="outputs/state_tokenizer/v8/full-721.jsonl"
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
