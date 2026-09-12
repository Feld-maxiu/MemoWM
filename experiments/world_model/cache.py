from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.state_tokenizer.common import iter_jsonl, sha256_file, write_json
from experiments.state_tokenizer.slot_layout import KEY64_LAYOUT

from .schema_web import WEB_CACHE_PROTOCOL

from .schema import (
    ACTION_TYPE_IDS,
    CACHE_FORMAT_VERSION,
    MAX_HISTORY,
    MAX_PAYLOAD_BYTES,
    MAX_TARGET_BYTES,
    NUM_CATEGORIES,
    NUM_LATENT_TOKENS,
    NUM_OBSERVATION_SLOTS,
    NUM_SUBSPACES,
    POLICY_NAMES,
    PROTOCOL,
    REF_PAD_ID,
    SPLIT_IDS,
    SPLITS,
    TAG_PAD_ID,
    Action,
    action_side_information_bits,
    canonicalize_action,
)


CACHE_FILES = {
    "codes": "codes.npy",
    "valid": "valid.npy",
    "global_indices": "global_indices.npy",
    "transitions": "transitions.npz",
    "manifest": "manifest.json",
}

ACCEPTED_PROTOCOLS = frozenset({PROTOCOL, WEB_CACHE_PROTOCOL})

V8_EXPECTED_STATES = {"train": 70_018, "validation": 20_011, "test": 9_979}
V8_EXPECTED_TRANSITIONS = {"train": 39_365, "validation": 11_263, "test": 5_629}


@dataclasses.dataclass(frozen=True)
class _Record:
    global_index: int
    task: str
    episode_id: str
    step: int
    split: str
    action: dict | None
    compact_axtree: str


def _minimal_records(path: str | Path) -> tuple[list[_Record], dict[str, int]]:
    records: list[_Record] = []
    counts: Counter[str] = Counter()
    seen_states: set[str] = set()
    for row, raw in enumerate(iter_jsonl(path)):
        index = int(raw.get("global_index", -1))
        if index != row:
            raise ValueError(
                f"global_index must equal manifest row: row {row} stores {index}"
            )
        split = str(raw.get("split"))
        if split not in SPLIT_IDS:
            raise ValueError(f"row {row} has invalid split {split!r}")
        state_id = str(raw.get("state_id", ""))
        if not state_id or state_id in seen_states:
            raise ValueError(f"row {row} has missing or duplicate state_id {state_id!r}")
        seen_states.add(state_id)
        action = raw.get("action")
        needs_tree = isinstance(action, dict)
        records.append(_Record(
            global_index=index,
            task=str(raw["task"]),
            episode_id=str(raw["episode_id"]),
            step=int(raw["step"]),
            split=split,
            action=action,
            compact_axtree=str(raw.get("dom", "")) if needs_tree else "",
        ))
        counts[split] += 1
    if not records:
        raise ValueError("records manifest is empty")
    return records, {name: int(counts[name]) for name in SPLITS}


def _assert_expected(actual: dict[str, int], expected: dict[str, int] | None, kind: str):
    if expected is not None and actual != expected:
        raise ValueError(f"{kind} counts mismatch: expected {expected}, got {actual}")


def _canonical_actions(episode: list[_Record]) -> list[Action]:
    output = []
    for record in episode[:-1]:
        if record.action is None:
            raise ValueError(
                f"{record.episode_id} step {record.step} has a successor but no action"
            )
        output.append(canonicalize_action(record.action, record.compact_axtree))
    return output


def build_transition_archive(
    records: Iterable[_Record],
    output: str | Path,
    *,
    expected_counts: dict[str, int] | None = None,
) -> dict:
    episodes: dict[str, list[_Record]] = defaultdict(list)
    for record in records:
        episodes[record.episode_id].append(record)

    task_names = sorted({record.task for values in episodes.values() for record in values})
    task_ids = {name: index for index, name in enumerate(task_names)}
    episode_names = sorted(episodes)
    episode_ids = {name: index for index, name in enumerate(episode_names)}

    history_indices = []
    target_indices = []
    out_task_ids = []
    out_episode_ids = []
    steps = []
    split_ids = []
    action_types = []
    action_tags = []
    action_refs = []
    action_payloads = []
    action_lengths = []
    action_policies = []
    action_targets = []
    action_target_lengths = []
    structural_bits = []
    semantic_bits = []
    full_action_bits = []
    counts: Counter[str] = Counter()

    for episode_name in episode_names:
        values = sorted(episodes[episode_name], key=lambda item: item.step)
        if values[0].step != 0:
            raise ValueError(f"episode {episode_name} does not start at step 0")
        if len({item.task for item in values}) != 1:
            raise ValueError(f"episode {episode_name} spans tasks")
        if len({item.split for item in values}) != 1:
            raise ValueError(f"episode {episode_name} spans splits")
        if len({item.step for item in values}) != len(values):
            raise ValueError(f"episode {episode_name} contains duplicate steps")
        for left, right in zip(values, values[1:]):
            if right.step != left.step + 1:
                raise ValueError(
                    f"episode {episode_name} is not consecutive: {left.step} -> {right.step}"
                )
        if len(values) < 2:
            continue
        actions = _canonical_actions(values)
        for position in range(len(values) - 1):
            current, target = values[position], values[position + 1]
            start = max(0, position - MAX_HISTORY + 1)
            prefix_records = values[start:position + 1]
            prefix_actions = actions[start:position + 1]
            pad = MAX_HISTORY - len(prefix_records)

            history_indices.append(
                [-1] * pad + [item.global_index for item in prefix_records]
            )
            padded_actions: list[Action | None] = [None] * pad + prefix_actions
            type_row, tag_row, ref_row = [], [], []
            payload_row, length_row, policy_row = [], [], []
            target_row, target_length_row = [], []
            for action in padded_actions:
                if action is None:
                    type_row.append(ACTION_TYPE_IDS["PAD"])
                    tag_row.append(TAG_PAD_ID)
                    ref_row.append(REF_PAD_ID)
                    payload_row.append(np.zeros((MAX_PAYLOAD_BYTES,), np.uint8))
                    length_row.append(0)
                    policy_row.append(255)
                    target_row.append(np.zeros((MAX_TARGET_BYTES,), np.uint8))
                    target_length_row.append(0)
                else:
                    type_row.append(action.type_id)
                    tag_row.append(action.tag_id)
                    ref_row.append(action.ref)
                    payload_row.append(action.padded_payload())
                    length_row.append(action.payload_length)
                    policy_row.append(action.policy_id)
                    target_row.append(action.padded_target())
                    target_length_row.append(action.target_length)

            action_types.append(type_row)
            action_tags.append(tag_row)
            action_refs.append(ref_row)
            action_payloads.append(payload_row)
            action_lengths.append(length_row)
            action_policies.append(policy_row)
            action_targets.append(target_row)
            action_target_lengths.append(target_length_row)
            target_indices.append(target.global_index)
            out_task_ids.append(task_ids[current.task])
            out_episode_ids.append(episode_ids[episode_name])
            steps.append(current.step)
            split_ids.append(SPLIT_IDS[current.split])
            counts[current.split] += 1
            current_action = prefix_actions[-1]
            structural_bits.append(
                action_side_information_bits(current_action, include_payload=False)
            )
            semantic_bits.append(action_side_information_bits(
                current_action, include_target=True, include_payload=False
            ))
            full_action_bits.append(action_side_information_bits(
                current_action, include_target=True, include_payload=True
            ))

    actual_counts = {name: int(counts[name]) for name in SPLITS}
    _assert_expected(actual_counts, expected_counts, "transition")
    arrays = {
        "history_indices": np.asarray(history_indices, np.int64),
        "target_indices": np.asarray(target_indices, np.int64),
        "task_ids": np.asarray(out_task_ids, np.uint8),
        "episode_ids": np.asarray(out_episode_ids, np.int32),
        "steps": np.asarray(steps, np.uint8),
        "split_ids": np.asarray(split_ids, np.uint8),
        "action_types": np.asarray(action_types, np.uint8),
        "action_tags": np.asarray(action_tags, np.uint8),
        "action_refs": np.asarray(action_refs, np.uint8),
        "action_payloads": np.asarray(action_payloads, np.uint8),
        "action_lengths": np.asarray(action_lengths, np.uint8),
        "action_policies": np.asarray(action_policies, np.uint8),
        "action_targets": np.asarray(action_targets, np.uint8),
        "action_target_lengths": np.asarray(action_target_lengths, np.uint8),
        "structural_action_bits": np.asarray(structural_bits, np.uint16),
        "semantic_action_bits": np.asarray(semantic_bits, np.uint16),
        "full_action_bits": np.asarray(full_action_bits, np.uint16),
    }
    expected_shapes = {
        "history_indices": (len(target_indices), MAX_HISTORY),
        "action_payloads": (len(target_indices), MAX_HISTORY, MAX_PAYLOAD_BYTES),
        "action_targets": (len(target_indices), MAX_HISTORY, MAX_TARGET_BYTES),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise AssertionError(f"{name}: expected {shape}, got {arrays[name].shape}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, output)
    return {
        "transitions": len(target_indices),
        "transition_counts": actual_counts,
        "episodes": len(episode_names),
        "tasks": task_names,
        "episode_names": episode_names,
        "policy_names": list(POLICY_NAMES),
    }


def _save_array_atomic(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    os.replace(temporary, path)


def build_cache(args: argparse.Namespace) -> dict:
    import jax
    import jax.numpy as jnp

    from residualmem.encoders.normalization import GroupChannelNormalizer
    from residualmem.world_model import categorical_bottleneck as Q
    from experiments.state_tokenizer.a1_continuous_bottleneck import FeatureStore
    from experiments.state_tokenizer.a2_categorical_bottleneck import (
        load_checkpoint,
        read_checkpoint_config,
    )

    output = Path(args.output)
    existing = [output / name for name in CACHE_FILES.values() if (output / name).exists()]
    if existing and not args.force:
        raise FileExistsError(
            f"cache already contains {existing[0]}; pass --force to replace exact files"
        )
    output.mkdir(parents=True, exist_ok=True)

    records, state_counts = _minimal_records(args.records)
    expected_states = None if args.no_v8_count_gate else V8_EXPECTED_STATES
    expected_transitions = None if args.no_v8_count_gate else V8_EXPECTED_TRANSITIONS
    _assert_expected(state_counts, expected_states, "state")

    stored = read_checkpoint_config(args.checkpoint)
    if tuple(stored["group_sizes"]) != tuple(KEY64_LAYOUT):
        raise ValueError(
            f"A2 layout {tuple(stored['group_sizes'])} != frozen {tuple(KEY64_LAYOUT)}"
        )
    config = Q.CategoricalBottleneckConfig(
        **{**stored, "group_sizes": tuple(stored["group_sizes"])}
    )
    if (
        config.num_e_tokens != NUM_LATENT_TOKENS
        or config.num_subspaces != NUM_SUBSPACES
        or config.num_categories != NUM_CATEGORIES
    ):
        raise ValueError(f"unexpected frozen code shape/config: {config}")

    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    device = jax.devices(args.platform)[args.device_index]
    params = jax.device_put(load_checkpoint(args.checkpoint, config), device)
    store = FeatureStore(args.features)
    normalizer = GroupChannelNormalizer.from_npz(args.normalization)

    total = len(records)
    codes_path = output / CACHE_FILES["codes"]
    valid_path = output / CACHE_FILES["valid"]
    code_tmp = codes_path.with_name(codes_path.name + ".tmp")
    valid_tmp = valid_path.with_name(valid_path.name + ".tmp")
    codes = np.lib.format.open_memmap(
        code_tmp, mode="w+", dtype=np.uint8,
        shape=(total, NUM_LATENT_TOKENS, NUM_SUBSPACES),
    )
    valid_out = np.lib.format.open_memmap(
        valid_tmp, mode="w+", dtype=np.bool_,
        shape=(total, NUM_OBSERVATION_SLOTS),
    )
    for start in range(0, total, args.batch_size):
        stop = min(start + args.batch_size, total)
        indices = list(range(start, stop))
        x_t, valid = store.load(indices)
        xbar = np.asarray(normalizer.normalize(x_t, valid), np.float32)
        if not np.isfinite(xbar).all() or np.count_nonzero(xbar[~valid]):
            raise ValueError(f"normalization invariant failed in rows {start}:{stop}")
        _, batch_codes = Q.reconstruct(
            params,
            jax.device_put(jnp.asarray(xbar), device),
            jax.device_put(jnp.asarray(valid), device),
            config,
            return_codes=True,
        )
        batch_codes = np.asarray(batch_codes)
        if batch_codes.min() < 0 or batch_codes.max() >= NUM_CATEGORIES:
            raise ValueError(f"invalid A2 code in rows {start}:{stop}")
        codes[start:stop] = batch_codes.astype(np.uint8)
        valid_out[start:stop] = valid
    codes.flush()
    valid_out.flush()
    del codes, valid_out
    os.replace(code_tmp, codes_path)
    os.replace(valid_tmp, valid_path)
    _save_array_atomic(
        output / CACHE_FILES["global_indices"], np.arange(total, dtype=np.int64)
    )

    transition_info = build_transition_archive(
        records,
        output / CACHE_FILES["transitions"],
        expected_counts=expected_transitions,
    )
    sources = {
        "records": str(Path(args.records).resolve()),
        "features": str(Path(args.features).resolve()),
        "normalization": str(Path(args.normalization).resolve()),
        "a1_checkpoint": str(Path(args.a1_checkpoint).resolve()),
        "a2_checkpoint": str(Path(args.checkpoint).resolve()),
    }
    source_hashes = {
        "records": sha256_file(args.records),
        "normalization": sha256_file(args.normalization),
        "a1_checkpoint": sha256_file(args.a1_checkpoint),
        "a2_checkpoint": sha256_file(args.checkpoint),
    }
    manifest = {
        "protocol": PROTOCOL,
        "format_version": CACHE_FORMAT_VERSION,
        "state_counts": state_counts,
        **transition_info,
        "layout": list(KEY64_LAYOUT),
        "arrays": {
            "codes": {"file": CACHE_FILES["codes"], "dtype": "uint8",
                      "shape": [total, NUM_LATENT_TOKENS, NUM_SUBSPACES]},
            "valid": {"file": CACHE_FILES["valid"], "dtype": "bool",
                      "shape": [total, NUM_OBSERVATION_SLOTS]},
            "global_indices": {"file": CACHE_FILES["global_indices"],
                               "dtype": "int64", "shape": [total]},
            "transitions": {"file": CACHE_FILES["transitions"]},
        },
        "sources": sources,
        "source_sha256": source_hashes,
        "artifact_sha256": {
            name: sha256_file(output / filename)
            for name, filename in CACHE_FILES.items()
            if name != "manifest"
        },
        "numerics": {
            "matmul_precision": args.matmul_precision,
            "activations": "float32",
            "device": str(device),
        },
        "test_locked": True,
    }
    write_json(output / CACHE_FILES["manifest"], manifest)
    return manifest


class FrozenCache:

    HISTORY_COLUMNS = (
        "history_codes", "history_valid", "history_present", "action_types",
        "action_payloads", "action_lengths", "action_tags", "action_refs",
        "action_targets", "action_target_lengths", "action_x", "action_y",
        "action_dx", "action_dy", "action_has_coord", "action_has_delta",
    )

    def __init__(self, root: str | Path, *, verify_hashes: bool = False,
                 max_history: int | None = None):
        self.max_history = max_history
        self.root = Path(root)
        if self.root.is_file() and self.root.suffix == ".npz":
            # Single-file merged dataset (see build_amabench_wm_train.py):
            # one npz holding codes/valid/global_indices, the transition
            # arrays under a "t_" prefix, and the manifest as a JSON string.
            archive = np.load(self.root, allow_pickle=False)
            self.manifest = json.loads(str(archive["manifest_json"].item()))
            if self.manifest.get("protocol") not in ACCEPTED_PROTOCOLS:
                raise ValueError(
                    f"unsupported cache protocol {self.manifest.get('protocol')!r}"
                )
            self.codes = archive["codes"]
            self.valid = archive["valid"]
            self.global_indices = archive["global_indices"]
            self.transitions = {
                name[2:]: archive[name]
                for name in archive.files if name.startswith("t_")
            }
            self._validate_arrays()
            if verify_hashes:
                digest = sha256_file(self.root)
                expected = self.manifest.get("artifact_sha256", {}).get("bundle")
                if expected and digest != expected:
                    raise ValueError(f"cache hash mismatch: {digest} != {expected}")
            return
        manifest_path = self.root / CACHE_FILES["manifest"]
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("protocol") not in ACCEPTED_PROTOCOLS:
            raise ValueError(f"unsupported cache protocol {self.manifest.get('protocol')!r}")
        self.codes = np.load(self.root / CACHE_FILES["codes"], mmap_mode="r")
        self.valid = np.load(self.root / CACHE_FILES["valid"], mmap_mode="r")
        self.global_indices = np.load(
            self.root / CACHE_FILES["global_indices"], mmap_mode="r"
        )
        archive = np.load(self.root / CACHE_FILES["transitions"], allow_pickle=False)
        self.transitions = {name: archive[name] for name in archive.files}
        self._validate_arrays()
        if verify_hashes:
            for name, digest in self.manifest["artifact_sha256"].items():
                actual = sha256_file(self.root / CACHE_FILES[name])
                if actual != digest:
                    raise ValueError(f"cache hash mismatch for {name}: {actual} != {digest}")

    def _validate_arrays(self):
        total = sum(int(v) for v in self.manifest["state_counts"].values())
        if self.codes.dtype != np.uint8 or self.codes.ndim != 3 or len(self.codes) != total:
            raise ValueError(f"invalid codes array {self.codes.dtype} {self.codes.shape}")
        if self.valid.dtype != np.bool_ or self.valid.shape != (
            total, self.codes.shape[1]
        ):
            raise ValueError(f"invalid valid array {self.valid.dtype} {self.valid.shape}")
        layout = tuple(int(v) for v in self.manifest.get("layout", ()))
        if not layout or sum(layout) != self.valid.shape[1]:
            raise ValueError(
                f"layout {layout} does not describe {self.valid.shape[1]} mask slots"
            )
        if self.global_indices.dtype != np.int64 or not np.array_equal(
            self.global_indices, np.arange(total, dtype=np.int64)
        ):
            raise ValueError("global_indices are not contiguous int64 0..N-1")
        n = int(self.manifest["transitions"])
        window = int(self.manifest.get("max_history", MAX_HISTORY))
        if self.transitions["history_indices"].shape != (n, window):
            raise ValueError(
                f"history_indices is {self.transitions['history_indices'].shape}, "
                f"not {(n, window)} as the manifest declares"
            )
        if self.transitions["action_payloads"].shape != (
            n, window, MAX_PAYLOAD_BYTES
        ):
            raise ValueError("transition payload shape mismatch")
        source = self.transitions["history_indices"]
        target = self.transitions["target_indices"]
        if np.any(source < -1) or np.any(source >= total):
            raise ValueError("transition history index outside cache")
        if np.any(target < 0) or np.any(target >= total):
            raise ValueError("transition target index outside cache")

    @property
    def task_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["tasks"])

    @property
    def episode_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["episode_names"])

    def indices_for_split(
        self, split: str, *, test_freeze_manifest: str | Path | None = None
    ) -> np.ndarray:
        if split not in SPLIT_IDS:
            raise ValueError(f"unknown split {split!r}")
        if split == "test":
            self._assert_test_unlocked(test_freeze_manifest)
        return np.flatnonzero(self.transitions["split_ids"] == SPLIT_IDS[split])

    def _assert_test_unlocked(self, freeze_manifest: str | Path | None):
        if freeze_manifest is None:
            raise PermissionError(
                "test split is locked; provide the final test freeze manifest"
            )
        value = json.loads(Path(freeze_manifest).read_text(encoding="utf-8"))
        expected = sha256_file(self.root / CACHE_FILES["manifest"])
        authorized = (
            value.get("protocol") == "v8_wm_test_freeze_v1"
            and value.get("test_unlocked") is True
            and value.get("cache_manifest_sha256") == expected
            and value.get("selected_variant") in {"full", "no_history"}
            and len(value.get("checkpoints", ())) == 3
        )
        if not authorized:
            raise PermissionError("test freeze manifest does not authorize this cache")

    def batch(
        self,
        transition_indices: np.ndarray,
        *,
        window_rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        rows = np.asarray(transition_indices, np.int64)
        history = self.transitions["history_indices"][rows]
        keep = self.max_history
        present = history >= 0
        if window_rng is not None:
            # Random window sampling: keep only a random suffix of each row's
            # history (right-aligned layout, so columns >= width - w), with
            # w drawn uniformly from 1..depth.  Training-time regulariser that
            # exposes the model to every context length, not just full windows.
            depth = present.sum(-1)
            width = history.shape[1]
            w = np.minimum(
                window_rng.integers(1, np.maximum(depth, 1) + 1), depth
            )
            present = present & (
                np.arange(width)[None, :] >= (width - w)[:, None]
            )
        safe = np.maximum(history, 0)
        target = self.transitions["target_indices"][rows]
        batch = {
            "transition_indices": rows,
            "history_codes": np.asarray(self.codes[safe], np.uint8),
            "history_valid": np.asarray(self.valid[safe] & present[..., None], np.bool_),
            "history_present": present,
            "action_types": self.transitions["action_types"][rows],
            "action_payloads": self.transitions["action_payloads"][rows],
            "action_lengths": self.transitions["action_lengths"][rows],
            "task_ids": self.transitions["task_ids"][rows],
            "target_codes": np.asarray(self.codes[target], np.uint8),
            "target_valid": np.asarray(self.valid[target], np.bool_),
            "target_indices": target,
            "episode_ids": self.transitions["episode_ids"][rows],
            "steps": self.transitions["steps"][rows],
            "structural_action_bits": self.transitions["structural_action_bits"][rows],
            "full_action_bits": self.transitions["full_action_bits"][rows],
        }
        for name in ("action_tags", "action_refs", "action_targets",
                     "action_target_lengths", "action_x", "action_y",
                     "action_dx", "action_dy", "action_has_coord",
                     "action_has_delta"):
            batch[name] = (
                self.transitions[name][rows] if name in self.transitions else None
            )
        if keep is not None:
            width = int(history.shape[1])
            if keep > width:
                # Cross-cache eval (e.g. an h4 selection cache under an h8
                # training config): left-pad with invalid history slots so the
                # batch width matches; present=False makes padding invisible.
                pad = keep - width
                for name in self.HISTORY_COLUMNS:
                    value = batch.get(name)
                    if value is None:
                        continue
                    if name == "history_codes":
                        shape = value.shape[2:]
                        batch[name] = np.concatenate(
                            [np.zeros((len(value), pad) + shape, value.dtype),
                             value], axis=1)
                    elif value.dtype == np.bool_:
                        batch[name] = np.concatenate(
                            [np.zeros((len(value), pad) + value.shape[2:], bool),
                             value], axis=1)
                    elif value.dtype.kind in "iu":
                        batch[name] = np.concatenate(
                            [np.full((len(value), pad) + value.shape[2:], -1,
                                     value.dtype), value], axis=1)
                    else:
                        batch[name] = np.concatenate(
                            [np.zeros((len(value), pad) + value.shape[2:],
                                      value.dtype), value], axis=1)
            if keep < width:
                for name in self.HISTORY_COLUMNS:
                    value = batch.get(name)
                    if value is not None:
                        batch[name] = value[:, -keep:]
        batch["policies"] = (
            self.transitions["action_policies"][rows, -1]
            if "action_policies" in self.transitions else None
        )
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--a1-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True, help="frozen A2 .npz checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--platform", default="gpu")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--matmul-precision", default="highest")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--no-v8-count-gate", action="store_true",
        help="only for synthetic/development data; production v8 must keep the gate",
    )
    return parser.parse_args()


def main() -> None:
    manifest = build_cache(parse_args())
    print(json.dumps({
        "state_counts": manifest["state_counts"],
        "transition_counts": manifest["transition_counts"],
        "artifact_sha256": manifest["artifact_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
