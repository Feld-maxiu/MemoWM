"""Build leakage-free BrowserGym cache for v9 retrieval/reader bridges.

Teacher documents are *fused observations*: the same screenshot plus task and
AXTree text in one Qwen3-VL-Embedding-8B document call.  Screenshot-only legacy
caches use a different key and are intentionally rejected by bridge trainers.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import torch

from residualmem.encoders.normalization import GroupChannelNormalizer
from residualmem.latent.instruct_bridge import (
    BRIDGE_CACHE_PROTOCOL,
    BROWSERGYM_TEACHER_TEXT_PROTOCOL,
    FUSED_OBSERVATION_TEACHER_PROTOCOL,
    cache_metadata_json,
)

from .common import iter_jsonl


def select_task_balanced_rows(
    records: list[dict], num_states: int, seed: int, *, split: str
) -> list[int]:
    rng = np.random.default_rng(seed)
    by_task: dict[str, list[int]] = {}
    for row, record in enumerate(records):
        if record["split"] == split:
            by_task.setdefault(str(record["task"]), []).append(row)
    for rows in by_task.values():
        rng.shuffle(rows)
    output: list[int] = []
    cursors = {task: 0 for task in by_task}
    while len(output) < num_states:
        progressed = False
        for task in sorted(by_task):
            cursor = cursors[task]
            if cursor < len(by_task[task]):
                output.append(by_task[task][cursor])
                cursors[task] += 1
                progressed = True
                if len(output) == num_states:
                    break
        if not progressed:
            raise ValueError(f"requested {num_states} {split} states, but split is exhausted")
    return output


def load_static_features(root: str | Path, global_indices: list[int]):
    locations: dict[int, tuple[Path, int]] = {}
    for worker in sorted(Path(root).glob("worker*")):
        indices_path = worker / "record_indices.npy"
        if not indices_path.exists():
            continue
        for local, index in enumerate(np.load(indices_path)):
            locations[int(index)] = (worker, local)
    missing = [index for index in global_indices if index not in locations]
    if missing:
        raise KeyError(f"feature store misses indices {missing[:20]}")
    opened: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    rows, masks = [], []
    for index in global_indices:
        worker, local = locations[index]
        if worker not in opened:
            opened[worker] = (
                np.load(worker / "key64-static-pca-bf16.npy", mmap_mode="r"),
                np.load(worker / "key64-static-valid.npy", mmap_mode="r"),
            )
        bits, valid = opened[worker]
        # uint16 stores the literal BF16 bit pattern.
        values = torch.from_numpy(np.asarray(bits[local], np.uint16).copy()).view(torch.bfloat16)
        rows.append(values.float().numpy())
        masks.append(np.asarray(valid[local], np.bool_).copy())
    return np.stack(rows), np.stack(masks)


def browsergym_teacher_text(record: dict) -> str:
    return (
        f"Task instruction:\n{record['instruction']}\n\n"
        f"Accessibility tree:\n{record['dom']}"
    )


def _data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime or 'image/png'};base64,{encoded}"


def _teacher_message(image: Path, text: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": "Represent the user's input."}]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _data_url(image)}},
            {"type": "text", "text": text},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]


def encode_teacher(
    rows: list[tuple[Path, str]], *, base_url: str, model: str,
    api_key: str, workers: int,
) -> np.ndarray:
    url = base_url.rstrip("/") + "/embeddings"
    headers = {"Authorization": f"Bearer {api_key}"}

    def one(row: tuple[Path, str]) -> np.ndarray:
        image, text = row
        with httpx.Client(timeout=180.0) as client:
            response = client.post(url, headers=headers, json={
                "messages": _teacher_message(image, text),
                "model": model,
                "encoding_format": "float",
                "continue_final_message": True,
                "add_special_tokens": True,
            })
            response.raise_for_status()
            return np.asarray(response.json()["data"][0]["embedding"], np.float32)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        vectors = list(pool.map(one, rows))
    output = np.stack(vectors)
    output /= np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)
    if output.shape[1] != 4096:
        raise ValueError(f"teacher dimension must be 4096, got {output.shape}")
    return output


def _a2_reconstruct(
    path: str, xbar: np.ndarray, valid: np.ndarray, batch: int, jax_python: str
) -> np.ndarray:
    # Qwen/torch and JAX intentionally live in different environments. Keep
    # their dependency stacks isolated and exchange only an NPZ/NPY pair.
    with tempfile.TemporaryDirectory(prefix="residualmem-a2-") as directory:
        input_path = Path(directory) / "input.npz"
        output_path = Path(directory) / "output.npy"
        np.savez(input_path, xbar=xbar, valid=valid)
        subprocess.run([
            jax_python, "-m", "experiments.state_tokenizer.reconstruct_a2_bridge",
            "--input", str(input_path), "--checkpoint", str(path),
            "--output", str(output_path), "--batch-size", str(batch),
        ], check=True)
        return np.asarray(np.load(output_path), np.float32)


def build(args: argparse.Namespace) -> dict:
    records_path = Path(args.records).resolve()
    records = list(iter_jsonl(records_path))
    selected: list[int] = []
    split_labels: list[str] = []
    for split, count in (("train", args.train_states), ("validation", args.validation_states)):
        rows = select_task_balanced_rows(records, count, args.seed, split=split)
        selected.extend(rows)
        split_labels.extend([split] * len(rows))
    global_indices = [int(records[row]["global_index"]) for row in selected]
    x_t, valid = load_static_features(args.features, global_indices)
    normalizer = GroupChannelNormalizer.from_npz(args.normalization)
    xbar = normalizer.normalize(x_t, valid)
    a2_xbar = _a2_reconstruct(
        args.a2_checkpoint, xbar, valid, args.a2_batch, args.jax_python
    )
    teacher_rows = []
    for row in selected:
        record = records[row]
        screenshot = records_path.parent / record["screenshot"]
        if not screenshot.is_file():
            raise FileNotFoundError(screenshot)
        teacher_rows.append((screenshot, browsergym_teacher_text(record)))
    teacher = encode_teacher(
        teacher_rows,
        base_url=args.teacher_base_url,
        model=args.teacher_model,
        api_key=args.teacher_api_key,
        workers=args.teacher_workers,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = cache_metadata_json(
        records=str(records_path),
        features=str(Path(args.features).resolve()),
        normalization=str(Path(args.normalization).resolve()),
        a2_checkpoint=str(Path(args.a2_checkpoint).resolve()),
        seed=args.seed,
        train_states=args.train_states,
        validation_states=args.validation_states,
    )
    np.savez_compressed(
        output,
        xbar=xbar.astype(np.float32),
        a2_xbar=a2_xbar.astype(np.float32),
        valid=valid.astype(np.bool_),
        teacher_fused_embedding=teacher.astype(np.float32),
        global_indices=np.asarray(global_indices, np.int64),
        split=np.asarray(split_labels),
        metadata=np.asarray(metadata),
    )
    return {
        "protocol": BRIDGE_CACHE_PROTOCOL,
        "teacher_protocol": FUSED_OBSERVATION_TEACHER_PROTOCOL,
        "teacher_text_protocol": BROWSERGYM_TEACHER_TEXT_PROTOCOL,
        "states": len(selected),
        "train_states": args.train_states,
        "validation_states": args.validation_states,
        "output": str(output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--a2-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-states", type=int, default=5000)
    parser.add_argument("--validation-states", type=int, default=500)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--a2-batch", type=int, default=64)
    parser.add_argument("--jax-python", default=os.getenv("JAX_PYTHON", ".venv-jax/bin/python"))
    parser.add_argument("--teacher-base-url", default=os.getenv("QWEN_VL_EMBED_BASE_URL", "http://127.0.0.1:8014/v1"))
    parser.add_argument("--teacher-model", default=os.getenv("QWEN_VL_EMBED_MODEL", "Qwen3-VL-Embedding-8B"))
    parser.add_argument("--teacher-api-key", default=os.getenv("QWEN_VL_EMBED_API_KEY", "EMPTY"))
    parser.add_argument("--teacher-workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
