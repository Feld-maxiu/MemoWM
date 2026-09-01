"""Build the WorldMemArena web test set from the official dataset repository.

The point of this script is auditability. It starts from a pinned checkout of
the official dataset, walks the published pipeline end to end, and records the
provenance of everything it touched, so the test set can be regenerated and
checked by someone with no access to this machine:

    official agent/gui/web/*.json
      -> convert_wma            records with canonical actions
      -> Q-Former encode        (32, 512) states, frozen checkpoint
      -> qformer_pq_apply       discrete codes, frozen codebook, never refitted
      -> cache_web              an evaluable cache

Steps 1, 3 and 4 run under the jax interpreter; step 2 needs torch and runs
under conda qwen-vl. Pass both.

The written provenance.json carries the dataset repo remote and commit, the
sha256 of the Q-Former checkpoint and of the codebook, and the row counts at
every stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(("git", "-C", str(repo)) + args, check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"failed ({result.returncode}): {' '.join(command)}\n"
                         f"see {log}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-repo", required=True,
                        help="checkout of the official dataset; the directory "
                             "holding agent/gui")
    parser.add_argument("--checkpoint", required=True, help="Q-Former checkpoint")
    parser.add_argument("--codebook", required=True, help="frozen PQ artifact")
    parser.add_argument("--output", required=True)
    parser.add_argument("--jax-python", required=True)
    parser.add_argument("--torch-python", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--subcategory", default="web")
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    dataset = Path(args.dataset_repo).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env_repo = str(repo)

    records = output / "records.jsonl"
    states = output / "states.npz"
    codes = output / "codes.npz"
    cache = output / "cache"

    run([args.jax_python, "-m", "experiments.state_tokenizer.convert_wma",
         "--root", str(dataset / "agent" / "gui"),
         "--subcategory", args.subcategory,
         "--split", "validation",
         "--output", str(records)], output / "01-convert.log")

    run([args.torch_python, "-u", "-m", "experiments.state_tokenizer.encode_molmoweb",
         "--source", "wma",
         "--wma-root", str(dataset / "agent" / "gui" / args.subcategory),
         "--checkpoint", args.checkpoint,
         "--queries", str(args.queries),
         "--device", args.device,
         "--output", str(states)], output / "02-encode.log")

    run([args.jax_python, "-m", "experiments.state_tokenizer.qformer_pq_apply",
         "--artifact", args.codebook,
         "--states", str(states),
         "--label", "test",
         "--output", str(codes)], output / "03-apply.log")

    # The codebook goes first because cache_web reads the alphabet off an
    # artifact that carries centroids; the applied codes file has none. Its own
    # code rows are for states no record mentions, so they are ignored.
    run([args.jax_python, "-m", "experiments.world_model.cache_web",
         "--codes", args.codebook,
         "--codes", str(codes),
         "--records", str(records),
         "--output", str(cache),
         "--max-history", "32",
         "--include-splits", "validation"], output / "04-cache.log")

    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    provenance = {
        "dataset_repo": {
            "path": str(dataset),
            "remote": git(dataset, "config", "--get", "remote.origin.url"),
            "commit": git(dataset, "rev-parse", "HEAD"),
            "subcategory": args.subcategory,
            "samples": len(sorted(
                (dataset / "agent" / "gui" / args.subcategory).glob("*.json"))),
        },
        "qformer_checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "file_sha256": sha256(Path(args.checkpoint)),
        },
        "codebook": {
            "path": str(Path(args.codebook).resolve()),
            "file_sha256": sha256(Path(args.codebook)),
        },
        "counts": {
            "states": sum(int(v) for v in manifest["state_counts"].values()),
            "transitions": int(manifest["transitions"]),
            "episodes": int(manifest["episodes"]),
        },
        "cache": {
            "fixed_width_bits": manifest["fixed_width_bits"],
            "max_history": manifest["max_history"],
            "num_categories": manifest["num_categories"],
            "num_subspaces": manifest["num_subspaces"],
            "tasks": manifest["tasks"],
        },
        "pipeline": ["convert_wma", "encode_molmoweb --source wma",
                     "qformer_pq_apply", "cache_web"],
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
