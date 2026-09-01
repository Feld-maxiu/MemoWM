"""Build the WorldMemArena fit-corpus cache that the utility gate scores against.

The gate needs, for every labelled observation, the world model's categorical
distribution over each of the 1024 code positions. That means the observation has
to sit in a cache the world model can read: history codes plus the action that
led to it. This script produces that cache from artifacts already on disk.

    official agent/gui/{6 non-web subcategories}/*.json
      -> convert_wma          records with canonical actions and episode ids
      -> (reuse)              32-slot xbar, already encoded by the Q-Former
      -> qformer_pq_apply     discrete codes, frozen codebook, never refitted
      -> cache_web            a scoreable cache

It differs from ``scripts_build_wm_testset.py`` in two ways. It never re-runs the
Q-Former -- the 32-slot xbar for the fit corpus already exists as encoder shards,
so the expensive vision forward is skipped. And it refuses to proceed unless the
codes it produces are the ones the world model actually trained on.

That last check is not hypothetical. The first run of this script produced codes
agreeing with the world model's training copy at **0 of 4199 rows**, because two
plausible-looking inputs were both wrong:

* **The xbar.** A bridge cache under ``outputs/instruct_bridge/`` holds a
  (4379, 32, 512) xbar from the same Q-Former checkpoint -- but it is a different
  forward pass than the ``encoded/wma-fitcorpus.*`` shards the codes came from.
  They differ by only 0.6% RMS, and that alone moves 4.8% of the codes.
* **The codebook.** ``pq/opq-shared-mix10-M32-C64.npz`` and
  ``pq-full/opq-shared-mix10-M32-C64.npz`` are the same configuration fitted
  twice. Only the second is bit-identical to the decoder embedded in the training
  npz; the first disagrees on every field (centroids max|Δ| 47.3).

Neither would raise anywhere. The codes stay (32, 32) uint8 in range, the cache
builds, the model scores it, and every per-position rate is drawn from a prior
fitted to a different quantisation -- which the manual measures at ~972 bits of
NLL. So the codes are looked up by hash in ``cache/codes.npy`` and a low hit rate
is a hard stop, not a warning.

Runs under either interpreter (stdlib + numpy only) and shells out to the jax
one for the three module invocations, following the split the repo enforces:
the extraction pipeline is torch-only, the world model jax-only, and nothing
crosses except npz and jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

SUBCATEGORIES = ("excel", "file_mgmt", "image_edit", "mobile",
                 "webarena_lite", "word_docs")


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
    except Exception:                                        # noqa: BLE001
        return ""


def run(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        raise SystemExit(f"failed ({result.returncode}): {' '.join(command)}\n"
                         f"see {log}\n" + "\n".join(tail))


def audit_samples(state_ids: list[str], records: list[dict]) -> dict:
    """The encoder's states and the converter's records must name the same set.

    Both enumerate the same observations, but by different routes: ``convert_wma``
    walks user turns carrying attachments, while the encoder shards were written
    by ``encode_molmoweb --source wma``. If either drifts, ``cache_web`` would
    either refuse (a record with no code) or, worse, line a record up against a
    different state's codes. Set equality is the check; counts alone are not.
    """
    from_records = [str(r["state_id"]) for r in records]
    encoded, converted = set(state_ids), set(from_records)
    per_sample_encoded = Counter(sid.rsplit("-", 1)[0] for sid in encoded)
    per_sample_records = Counter(sid.rsplit("-", 1)[0] for sid in converted)
    disagreeing = sorted(
        stem for stem in set(per_sample_encoded) | set(per_sample_records)
        if per_sample_encoded[stem] != per_sample_records[stem]
    )
    return {
        "encoded_states": len(encoded), "converted_records": len(converted),
        "missing_codes": sorted(converted - encoded)[:20],
        "missing_codes_total": len(converted - encoded),
        "unused_states": sorted(encoded - converted)[:20],
        "unused_states_total": len(encoded - converted),
        "samples_disagreeing": disagreeing,
        "per_sample_counts_agree": not disagreeing,
    }


def build_states(patterns: list[str], output: Path) -> dict:
    """Concatenate the encoder shards in the order ``load_states`` would.

    Written out as one file so every downstream stage reads a single artifact
    whose row order is, by construction, the row order of the applied codes.
    """
    from experiments.state_tokenizer.qformer_pq import load_states

    states, ids, checkpoint = load_states(patterns)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        xbar=states.astype(np.float32),
        state_ids=np.asarray(ids, dtype=object),
        metadata=json.dumps({
            "checkpoint": checkpoint,
            "source": "encode_molmoweb --source wma",
            "patterns": list(patterns),
            "queries": int(states.shape[1]),
        }, ensure_ascii=False),
    )
    return {"rows": int(len(ids)), "queries": int(states.shape[1]),
            "channels": int(states.shape[2]), "checkpoint": checkpoint,
            "state_ids": ids}



def strip_codebook(codebook: Path, output: Path) -> None:
    """Copy a codebook keeping only the alphabet.

    ``cache_web`` reads the alphabet size off an artifact carrying centroids, but
    a full codebook also ships its own ``codes/*`` rows. Those were fitted on a
    different corpus that includes WorldMemArena states, so their ids collide with
    the freshly applied ones while disagreeing bit for bit -- the collision the
    manual documents at §3.3. Keeping only the alphabet removes the ambiguity.
    """
    with np.load(codebook, allow_pickle=True) as data:
        kept = {name: data[name] for name in data.files
                if not name.startswith(("codes/", "state_ids/"))}
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **kept)


def code_copy_hit_rate(training_codes: Path, applied: Path,
                       label: str) -> dict:
    """Fraction of applied code arrays that appear byte-identically in training.

    A miss is not fatal by itself -- the fit corpus may simply not be in the
    shipped training file -- but it means the world model's prior was fitted
    against a different quantisation of the same observations, and the manual
    records that costing ~972 bits of NLL. Either way the number belongs in
    provenance rather than in someone's memory.
    """
    with np.load(applied, allow_pickle=True) as data:
        codes = np.asarray(data[f"codes/{label}"], np.uint8)
    if not training_codes.exists():
        return {"checked": False, "reason": f"{training_codes} is absent"}

    training = np.load(training_codes, mmap_mode="r")
    seen: set[bytes] = set()
    for start in range(0, len(training), 20000):
        block = np.ascontiguousarray(training[start:start + 20000])
        seen.update(hashlib.blake2b(row.tobytes(), digest_size=16).digest()
                    for row in block)
    hits = sum(
        hashlib.blake2b(np.ascontiguousarray(row).tobytes(),
                        digest_size=16).digest() in seen
        for row in codes
    )
    return {"checked": True, "training_rows": int(len(training)),
            "applied_rows": int(len(codes)), "hits": int(hits),
            "hit_rate": round(hits / max(len(codes), 1), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True,
                        help="the directory holding agent/gui subcategories")
    parser.add_argument("--states", action="append", required=True,
                        help="glob over the encoder shards; repeatable. Must be "
                             "the same xbar the world model's codes came from")
    parser.add_argument("--codebook", required=True,
                        help="frozen C=64 artifact. ☠️ pq-full/, not pq/ -- the "
                             "two are different k-means fits of the same config "
                             "and only one matches the training cache")
    parser.add_argument("--codebook-coarse", default=None,
                        help="frozen C=16 artifact, for the coarse-quantiser arm")
    parser.add_argument("--training-codes", default="cache/codes.npy",
                        help="the world model's training codes, for the hash check")
    parser.add_argument("--output", required=True)
    parser.add_argument("--jax-python", required=True)
    parser.add_argument("--max-history", type=int, default=32)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-hit-rate", type=float, default=0.9,
                        help="fraction of applied codes that must appear "
                             "byte-identically in the training cache")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    records_path = output / "records.jsonl"
    states = output / "states.npz"
    codes = output / "codes.npz"
    codes_coarse = output / "codes-coarse.npz"
    alphabet = output / "codebook-alphabet-only.npz"
    cache = output / "cache"

    command = [args.jax_python, "-m", "experiments.state_tokenizer.convert_wma",
               "--root", str(dataset_root), "--split", args.split,
               "--output", str(records_path)]
    for subcategory in SUBCATEGORIES:
        command += ["--subcategory", subcategory]
    run(command, output / "01-convert.log")

    records = [json.loads(line) for line in
               records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = build_states(args.states, states)
    audit = audit_samples(summary.pop("state_ids"), records)
    print(f"[audit] {audit['encoded_states']} encoded states, "
          f"{audit['converted_records']} converted records, "
          f"{audit['missing_codes_total']} records without a code")
    if audit["missing_codes_total"]:
        raise SystemExit(
            f"{audit['missing_codes_total']} records have no encoded state, e.g. "
            f"{audit['missing_codes'][:3]}. cache_web would refuse; resolve the "
            f"enumeration mismatch rather than filtering silently."
        )

    for artifact, destination in ((args.codebook, codes),
                                  (args.codebook_coarse, codes_coarse)):
        if artifact is None:
            continue
        command = [args.jax_python, "-m",
                   "experiments.state_tokenizer.qformer_pq_apply",
                   "--artifact", str(Path(artifact).resolve()),
                   "--label", args.split, "--output", str(destination)]
        for pattern in args.states:
            command += ["--states", pattern]
        run(command, output / f"02-apply-{destination.stem}.log")

    check = code_copy_hit_rate(Path(args.training_codes), codes, args.split)
    print(f"[codes] hit rate {check.get('hit_rate')} "
          f"({check.get('hits')}/{check.get('applied_rows')})")
    if check.get("checked") and check["hit_rate"] < args.min_hit_rate:
        raise SystemExit(
            f"code-copy hit rate {check['hit_rate']:.4f} < {args.min_hit_rate}. "
            f"The applied codes are not the copy the world model trained "
            f"against, so its per-position prior does not describe them and the "
            f"rate axis would measure a different quantisation (the manual puts "
            f"that at ~972 bits). Check that --states is the encoder shard set "
            f"and --codebook is the pq-full fit, not the pq/ one."
        )

    strip_codebook(Path(args.codebook).resolve(), alphabet)
    run([args.jax_python, "-m", "experiments.world_model.cache_web",
         "--codes", str(alphabet), "--codes", str(codes),
         "--records", str(records_path), "--output", str(cache),
         "--max-history", str(args.max_history),
         "--include-splits", args.split], output / "03-cache.log")

    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    provenance = {
        "protocol": "residualmem_utility_gate_cache_v1",
        "dataset_root": {
            "path": str(dataset_root),
            "remote": git(dataset_root, "config", "--get", "remote.origin.url"),
            "commit": git(dataset_root, "rev-parse", "HEAD"),
            "subcategories": list(SUBCATEGORIES),
        },
        "alignment_audit": audit,
        "states": {"patterns": list(args.states), **summary},
        "codebook": {
            "path": str(Path(args.codebook).resolve()),
            "file_sha256": sha256(Path(args.codebook)),
        },
        "codebook_coarse": None if args.codebook_coarse else None,
        "code_copy_check": check,
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
            "num_latent_tokens": manifest["num_latent_tokens"],
            "tasks": manifest["tasks"],
        },
        "pipeline": ["convert_wma", "reuse encoder shards",
                     "qformer_pq_apply", "cache_web"],
    }
    if args.codebook_coarse:
        provenance["codebook_coarse"] = {
            "path": str(Path(args.codebook_coarse).resolve()),
            "file_sha256": sha256(Path(args.codebook_coarse)),
        }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in provenance.items()
                      if k != "alignment_audit"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
