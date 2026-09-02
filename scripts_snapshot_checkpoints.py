"""Record what every checkpoint on this machine is, before any of them is deleted.

Checkpoints are gitignored, so they exist on exactly one disk and nowhere else.
Deleting one destroys not just the weights but the record that it ever existed --
which arm it was, when it was trained, what it selected on. That record is small
and is worth keeping even when the 300 MB behind it is not.

Lives at the repo root, not under ``outputs/`` -- that directory is
gitignored, and git cannot re-include a file whose parent directory is
excluded, so a negation pattern there silently does nothing.

Written before the cleanup, committed, and never deleted. If a result later needs
explaining, this says what was on disk on the day it was measured.

Metadata extraction is best-effort: reading a ``.pt`` needs torch and a ``.pkl``
needs the training code's classes on the path. The hash, size and mtime are
always recorded, because those are what identify a file after it is gone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_ROOTS = ["outputs", "gate", "run"]
SUFFIXES = (".pt", ".pkl")

# Metadata keys worth keeping. The full dict can hold optimizer state and
# megabytes of config; these are the fields that say which arm a file is.
KEEP_KEYS = ("queries", "selected_by", "step", "val_answer_ce", "seed", "variant",
             "obs_weight", "qk_norm", "protocol", "arm", "qformer_sha256",
             "slots", "layout")


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_metadata(path: Path) -> dict | str:
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:                                   # noqa: BLE001
        return f"unreadable: {type(error).__name__}"
    if not isinstance(payload, dict):
        return f"not a dict: {type(payload).__name__}"
    meta = payload.get("metadata")
    if not isinstance(meta, dict):
        return {"top_level_keys": sorted(payload)[:12]}
    return {k: meta[k] for k in KEEP_KEYS if k in meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument("--no-metadata", action="store_true",
                        help="skip torch.load; hashes and sizes only")
    parser.add_argument("--output", default="CHECKPOINT_INVENTORY.json")
    args = parser.parse_args()

    entries = []
    for root in args.roots:
        base = REPO / root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            stat = path.stat()
            entry = {
                "path": str(path.relative_to(REPO)),
                "bytes": stat.st_size,
                "mtime": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc).isoformat(),
                "sha256": sha256_file(path),
            }
            if not args.no_metadata:
                entry["metadata"] = read_metadata(path)
            entries.append(entry)
            print(f"  {entry['path']}  {stat.st_size / 1048576:.0f} MB", flush=True)

    # Files sharing a hash are the same bytes under two names; worth flagging so
    # a "deletion" that only removes one of them is not mistaken for a saving.
    by_hash: dict[str, list[str]] = {}
    for entry in entries:
        by_hash.setdefault(entry["sha256"], []).append(entry["path"])
    duplicates = {h: paths for h, paths in by_hash.items() if len(paths) > 1}

    report = {
        "protocol": "residualmem_checkpoint_inventory_v1",
        "taken": datetime.now(timezone.utc).isoformat(),
        "why": "checkpoints are gitignored and exist on one disk only; this is "
               "the record that survives their deletion",
        "roots": args.roots,
        "count": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "identical_content_groups": duplicates,
        "entries": entries,
    }
    output = REPO / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"\n{len(entries)} files, {report['total_bytes'] / 2**30:.2f} GB")
    if duplicates:
        print(f"{len(duplicates)} groups of byte-identical files:")
        for paths in duplicates.values():
            print("   " + " == ".join(paths))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
