"""Build the single-file WM training dataset ``amabench_wm_train.npz``.

Merges the two training corpora used by the mixed-corpus world model into
one FrozenCache-loadable npz:

- WebWorldData v2 (corpus 0): full cache, train + validation splits (public
  corpus; validation is the anti-forgetting reference eval).
- AMA-bench (corpus 1): TRAIN SPLIT ONLY.  The AMA validation split (model
  selection) and the access-controlled test split are deliberately excluded;
  benchmark evaluation must go through the official ama-bench release.

AMA rows are extracted train-only with state-level index remapping, then
history/action windows are left-padded from width 4 to width 8 following the
same convention as FrozenCache's cross-cache padding (invalid history slots,
present=False makes padding invisible).

Usage:
    python -m experiments.world_model.build_amabench_wm_train \
        --output outputs/release/amabench_wm_train.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WINDOW = 8
CORPUS_WEBWORLD = 0
CORPUS_AMA = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pad_windows(value: np.ndarray, width: int) -> np.ndarray:
    """Left-pad a (N, w, ...) window array to `width` columns."""
    pad = width - value.shape[1]
    if pad <= 0:
        return value
    if value.dtype == np.bool_:
        fill = False
    elif value.dtype.kind in "iu":
        fill = -1
    else:
        fill = 0
    shape = (len(value), pad) + value.shape[2:]
    return np.concatenate([np.full(shape, fill, value.dtype), value], axis=1)


def load_ama_train(cache_dir: Path) -> tuple[dict, np.ndarray, dict]:
    """Extract AMA train transitions with remapped, width-8 windows."""
    cache_dir = Path(cache_dir)
    transitions = dict(np.load(cache_dir / "transitions.npz"))
    keep_t = np.flatnonzero(transitions["split_ids"] == 0)
    hist = transitions["history_indices"][keep_t]
    targets = transitions["target_indices"][keep_t]
    keep_s = np.unique(np.concatenate([targets, hist[hist >= 0].ravel()]))
    n_states = len(np.load(cache_dir / "codes.npy", mmap_mode="r"))
    remap_s = -np.ones(n_states, np.int64)
    remap_s[keep_s] = np.arange(len(keep_s))
    remap_e = -np.ones(int(transitions["episode_ids"].max()) + 1, np.int64)
    keep_eps = np.unique(transitions["episode_ids"][keep_t])
    remap_e[keep_eps] = np.arange(len(keep_eps))

    out = {}
    for key, value in transitions.items():
        value = value[keep_t]
        if key == "history_indices":
            out[key] = np.where(value >= 0, remap_s[np.clip(value, 0, None)], -1)
        elif key == "target_indices":
            out[key] = remap_s[value]
        elif key == "episode_ids":
            out[key] = remap_e[value]
        elif key == "split_ids":
            out[key] = np.zeros(len(keep_t), np.uint8)
        elif value.ndim >= 2 and value.shape[1] == 4:
            out[key] = _pad_windows(value, WINDOW)
        else:
            out[key] = value
    codes = np.load(cache_dir / "codes.npy")[keep_s]
    valid = np.load(cache_dir / "valid.npy")[keep_s]
    provenance = json.loads((cache_dir / "manifest.json").read_text())
    return out, keep_s, {
        "codes": codes, "valid": valid,
        "episodes": int(len(keep_eps)),
        "episode_names": [provenance["episode_names"][e] for e in keep_eps],
        "codebook": Path(provenance["codes"][0]).name,
        "manifest": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webworld-cache",
                        default="outputs/wm_train/webworld-v2/cache-h8")
    parser.add_argument("--ama-cache", default="outputs/wm_train/ama-v1/cache-h4")
    parser.add_argument("--output",
                        default="outputs/release/amabench_wm_train.npz")
    args = parser.parse_args()

    ww_dir, ama_dir = Path(args.webworld_cache), Path(args.ama_cache)
    ww_t = dict(np.load(ww_dir / "transitions.npz"))
    ww_codes = np.load(ww_dir / "codes.npy", mmap_mode="r")
    ww_valid = np.load(ww_dir / "valid.npy", mmap_mode="r")
    ww_manifest = json.loads((ww_dir / "manifest.json").read_text())
    ama_t, ama_states, ama_meta = load_ama_train(ama_dir)

    n_ww_states, n_ama_states = len(ww_codes), len(ama_meta["codes"])
    n_ww, n_ama = len(ww_t["target_indices"]), len(ama_t["target_indices"])
    ww_episode_base = int(ww_t["episode_ids"].max()) + 1

    codes = np.concatenate([np.asarray(ww_codes), ama_meta["codes"]])
    valid = np.concatenate([np.asarray(ww_valid), ama_meta["valid"]])
    global_indices = np.arange(len(codes), dtype=np.int64)

    merged = {}
    for key, value in ww_t.items():
        if key not in ama_t:
            continue  # key absent from one corpus (e.g. tags/refs) -> drop
        av = ama_t[key]
        if key in ("episode_ids",):
            av = av + ww_episode_base
        elif key == "history_indices":
            av = _pad_windows(np.where(av >= 0, av + n_ww_states, -1), WINDOW)
        elif key == "target_indices":
            av = av + n_ww_states
        merged["t_" + key] = np.concatenate([value, av])
    merged["t_corpus_ids"] = np.concatenate(
        [np.full(n_ww, CORPUS_WEBWORLD, np.uint8),
         np.full(n_ama, CORPUS_AMA, np.uint8)])

    manifest = {
        "protocol": ww_manifest["protocol"],
        "format_version": ww_manifest["format_version"],
        "max_history": WINDOW,
        "num_categories": ww_manifest["num_categories"],
        "num_latent_tokens": ww_manifest["num_latent_tokens"],
        "num_subspaces": ww_manifest["num_subspaces"],
        "layout": list(ww_manifest["layout"]),
        "tasks": ["webworld"],
        "episode_names": list(ww_manifest["episode_names"]) + list(
            ama_meta["episode_names"]),
        "transitions": int(n_ww + n_ama),
        "state_counts": {"all": int(len(codes))},
        "transition_counts": {
            "webworld_train": int(n_ww - int((ww_t["split_ids"] == 1).sum())),
            "webworld_validation": int((ww_t["split_ids"] == 1).sum()),
            "ama_train": int(n_ama),
        },
        "corpora": {
            "webworld_v2": {
                "corpus_id": CORPUS_WEBWORLD,
                "source_cache": str(ww_dir),
                "codebook": Path(ww_manifest["codes"][0]).name,
                "note": "public WebWorldData corpus; validation split is the "
                        "anti-forgetting reference eval",
            },
            "ama_bench_train": {
                "corpus_id": CORPUS_AMA,
                "source_cache": str(ama_dir),
                "codebook": ama_meta["codebook"],
                "episodes": int(ama_meta["episodes"]),
                "note": "TRAIN SPLIT ONLY. AMA validation (model-selection) "
                        "and test splits are excluded; evaluate via the "
                        "official ama-bench release.",
            },
        },
        "release_note": (
            "Single-file WM training dataset. Contains no AMA held-out data."
        ),
    }
    merged["ama_source_state_indices"] = ama_states
    merged["ama_episode_names"] = np.array(ama_meta["episode_names"])
    merged["manifest_json"] = np.array(json.dumps(manifest))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, codes=codes, valid=valid,
             global_indices=global_indices, **merged)
    tmp.replace(out)
    digest = _sha256(out)
    out.with_suffix(".sha256").write_text(f"{digest}  {out.name}\n")

    print(f"dataset: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  states {len(codes):,} | transitions {n_ww + n_ama:,} "
          f"(webworld {n_ww:,} [train+val], ama train {n_ama:,})")
    print(f"  sha256 {digest}  (sidecar {out.with_suffix('.sha256').name})")


if __name__ == "__main__":
    main()
