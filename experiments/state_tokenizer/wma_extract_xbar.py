"""Extract frozen ``xbar`` for WorldMemArena web observations, under ablation arms.

The observation set must match the retrieval path exactly, or the diagnosis
describes a different set of rows than the smoke did. Rather than re-walking
sessions here -- which would drift the moment the adapter's round pairing or its
``_observation_only_turn`` gate changed -- this drives the real
``ResidualMemInstructAdapter`` over the real ``run_eval_sample`` and wraps the
tokenizer, recording every arm at the moment the adapter asks for the one it
uses. The recorded set is then identical by construction.

Arms, following the modality decomposition (V = screenshot, T = synthetic-AXTree
text built from user text + captions):

    m11  real screenshot   + real text     the arm the adapter consumes
    m10  real screenshot   + empty AXTree  visual only
    m01  blank white image + real text     text only
    m00  blank white image + empty AXTree  empty control

``m00`` is not "no information": the tokenizer still emits 64 slots from a blank
image and a structural-only AXTree row, and the head still maps them somewhere.
That output is the floor every other arm is measured against, which is why the
decomposition subtracts it rather than treating zero as the baseline.

A fifth arm, ``m11_resampled``, repeats ``m11`` with the screenshot resized to
the training capture size, separating "the pixels are different" from "the
resolution and merged-grid geometry are different".

Writes one npz per sample so the run is restartable; ``/tmp`` does not survive a
session-container rebuild and neither does an hour of 9B forwards.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TRAINING_CAPTURE_SIZE = (498, 321)

ARMS = ("m11", "m10", "m01", "m00", "m11_resampled", "m11_letterbox")


def resample_stretch(image: Image.Image, size=TRAINING_CAPTURE_SIZE) -> Image.Image:
    """Resize onto the training capture size, ignoring aspect ratio."""
    return image.resize(size, Image.BICUBIC)


def resample_letterbox(image: Image.Image, size=TRAINING_CAPTURE_SIZE) -> Image.Image:
    """Resize onto the training capture size preserving aspect ratio, padding white.

    Needed because the two domains do not share an aspect ratio: WorldMemArena
    captures 1280x720 (1.778) and the training domain 498x321 (1.551). A direct
    resize therefore changes resolution *and* squashes the content horizontally,
    so a shift measured against it cannot say which of the two caused it. Padding
    matches ``_blank_image``'s white so the added border is the same colour the
    empty-image arm already uses.
    """
    target_w, target_h = size
    scale = min(target_w / image.width, target_h / image.height)
    scaled = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.BICUBIC,
    )
    canvas = Image.new("RGB", size, color=(255, 255, 255))
    canvas.paste(scaled, ((target_w - scaled.width) // 2, (target_h - scaled.height) // 2))
    return canvas


def _wma_root() -> Path:
    configured = os.getenv("WORLDMEMARENA_ROOT")
    root = Path(configured) if configured else Path(__file__).resolve().parents[2].parent / "WorldMemArena"
    root = root.resolve()
    if not (root / "eval_framework").is_dir():
        raise FileNotFoundError(f"WorldMemArena checkout not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class RecordingTokenizer:
    """Delegates to the frozen tokenizer, recording every ablation arm.

    The adapter only ever asks for ``m11``; the other arms are computed on the
    same observation while it is in hand. Attribute access falls through so the
    adapter's other uses of the tokenizer (``model``, ``processor``) still work.
    """

    def __init__(self, inner, *, arms=ARMS, resample_to=TRAINING_CAPTURE_SIZE,
                 save_key64=False):
        self._inner = inner
        self._arms = tuple(arms)
        self._resample_to = resample_to
        self._save_key64 = bool(save_key64)
        self.records: list[dict] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _blank(self):
        return self._inner._blank_image()

    def encode(self, observation):
        from residualmem.benchmarks.worldmemarena_tokenizer import (
            WebObservation,
            fused_observation_text,
        )

        stripped = dataclasses.replace(observation, user_text="", captions=())
        no_image = dataclasses.replace(observation, screenshot=None)
        empty = WebObservation(screenshot=None, user_text="", captions=())

        outputs: dict[str, object] = {}
        # m11 first: it is what the adapter receives, so a failure here must
        # surface exactly as it would without this wrapper. Only this arm keeps
        # the pre-PCA Key64 -- it is 8x the size of xbar, and the pre-PCA
        # questions are all about the real observation, not the ablations.
        outputs["m11"] = self._inner.encode(observation, return_key64=self._save_key64)

        if "m10" in self._arms:
            outputs["m10"] = self._inner.encode(stripped, validate=False)
        if "m01" in self._arms:
            outputs["m01"] = self._inner.encode(no_image, validate=False)
        if "m00" in self._arms:
            outputs["m00"] = self._inner.encode(empty, validate=False)
        if "m11_resampled" in self._arms and observation.screenshot:
            with Image.open(observation.screenshot) as handle:
                native = handle.convert("RGB")
            outputs["m11_resampled"] = self._inner.encode(
                observation, validate=False, image=resample_stretch(native, self._resample_to)
            )
        if "m11_letterbox" in self._arms and observation.screenshot:
            with Image.open(observation.screenshot) as handle:
                native = handle.convert("RGB")
            outputs["m11_letterbox"] = self._inner.encode(
                observation, validate=False, image=resample_letterbox(native, self._resample_to)
            )

        record = {
            "has_screenshot": bool(observation.screenshot),
            "screenshot": observation.screenshot,
            "user_text_chars": len(observation.user_text),
            "caption_count": len(observation.captions),
            "caption_chars": sum(len(c) for c in observation.captions),
            "image_ids": list(observation.image_ids),
            # The teacher's document text, stored rather than re-derived later:
            # the teacher pass has to pair the same text with the same image, and
            # rebuilding it from a second walk of the dataset would be one more
            # place for the two observation sets to drift apart.
            "fused_text": fused_observation_text(observation.user_text, observation.captions),
            "native_image_size": list(outputs["m11"].metadata.get("image_size", ())),
            "synthetic_axtree_chars": len(outputs["m11"].metadata.get("synthetic_axtree", "")),
            "key64": outputs["m11"].key64,
            "arms": {
                name: {
                    "xbar": out.xbar.astype(np.float32),
                    "valid": out.valid.astype(np.bool_),
                }
                for name, out in outputs.items()
            },
        }
        self.records.append(record)
        return outputs["m11"]


def _sample_ids(bundle, subcategory: str) -> list[str]:
    return [
        s.sample_id for s in bundle.samples
        if getattr(s, "_subcategory", "") == subcategory
    ]


def run_sample(sample, tokenizer, *, top_k: int, checkpoint_index: int,
               save_key64: bool = False) -> dict:
    from eval_framework.memory_adapters.residualmem_instruct_adapter import (
        ResidualMemInstructAdapter,
    )
    from eval_framework.pipeline.runner import run_eval_sample

    class _XbarOnlyAdapter(ResidualMemInstructAdapter):
        """Skips all embedding; extraction wants ``xbar``, not retrieval.

        Which rows exist is decided by ``_observation_only_turn`` and the round
        pairing, neither of which depends on the vectors -- so stubbing these
        keeps the recorded observation set exact while avoiding a Qwen3-VL
        forward per text row and per question. The retrieval output of this run
        is meaningless and is discarded; ranking metrics are computed later,
        from the teacher embeddings, in the attribution step.

        Both sides must be stubbed: ``run_eval_sample`` also runs the QA phase,
        which encodes each question through ``_encode_query``.
        """

        def _encode_docs(self, inputs):
            return [np.zeros(4096, np.float32) for _ in inputs]

        def _encode_query(self, query):
            return np.zeros(4096, np.float32)

    checkpoint = sample.normalized_checkpoints[checkpoint_index]
    sample = dataclasses.replace(sample, normalized_checkpoints=(checkpoint,))
    session_index = {row.session_id: i for i, row in enumerate(sample.sessions)}
    max_sessions = max(session_index[s] for s in checkpoint.covered_sessions) + 1

    # The base adapter demands an embedding backend, and picks eagerly: with no
    # base URL it loads a 16 GB SentenceTransformer in __init__. Extraction never
    # embeds anything -- both encode paths above are stubbed -- so declaring a
    # remote backend keeps self._model at None and skips the load entirely. The
    # address is deliberately unreachable: if a future change adds a third encode
    # path, this fails loudly instead of silently embedding against some server.
    os.environ.setdefault("QWEN_VL_EMBED_BASE_URL", "http://xbar-extraction-never-calls-this.invalid/v1")

    recorder = RecordingTokenizer(tokenizer, save_key64=save_key64)
    adapter = _XbarOnlyAdapter(
        baseline_name="ResidualMem-Instruct-Xbar-Input-RAG",
        tokenizer=recorder,
    )
    run_eval_sample(
        adapter, sample, top_k=top_k,
        answer_fn=lambda _q, _r: "", max_sessions=max_sessions,
    )

    observation_rows = [r for r in adapter._rounds if r["row_kind"] == "residualmem_xbar"]
    if len(observation_rows) != len(recorder.records):
        raise RuntimeError(
            f"{sample.sample_id}: {len(observation_rows)} observation rows but "
            f"{len(recorder.records)} recorded encodes; the wrapper and the "
            "adapter disagree on the observation set"
        )
    for row, record in zip(observation_rows, recorder.records):
        record["memory_id"] = row["memory_id"]
        record["session_id"] = row["session_id"]
    return {
        "sample_id": sample.sample_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "sessions_covered": len(checkpoint.covered_sessions),
        "max_sessions": max_sessions,
        "total_rows": len(adapter._rounds),
        "observation_rows": len(observation_rows),
        "records": recorder.records,
    }


def save_sample(path: Path, payload: dict) -> None:
    arrays: dict[str, np.ndarray] = {}
    meta = []
    for index, record in enumerate(payload["records"]):
        for arm, blob in record["arms"].items():
            arrays[f"{arm}/xbar/{index:04d}"] = blob["xbar"]
            arrays[f"{arm}/valid/{index:04d}"] = blob["valid"]
        if record.get("key64") is not None:
            # bf16 bit pattern, matching the on-disk feature store, so the
            # in-domain and WMA sides of the pre-PCA comparison carry the same
            # precision rather than one being silently finer than the other.
            arrays[f"key64/{index:04d}"] = _to_bf16_bits(record["key64"])
        meta.append({k: v for k, v in record.items() if k not in ("arms", "key64")})
    arrays["metadata"] = np.asarray(json.dumps({
        **{k: v for k, v in payload.items() if k != "records"},
        "records": meta,
        "arms": list(ARMS),
        "protocol": "wma_xbar_ablation_v1",
    }, ensure_ascii=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _to_bf16_bits(values: np.ndarray) -> np.ndarray:
    import torch

    return torch.from_numpy(np.ascontiguousarray(values, np.float32)) \
        .to(torch.bfloat16).view(torch.uint16).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("all", "small"), default="all")
    parser.add_argument("--subcategory", default="agent/arena/web")
    parser.add_argument("--sample-id", action="append", default=None,
                        help="restrict to these samples; repeatable")
    parser.add_argument("--checkpoint-index", type=int, default=-1,
                        help="-1 selects the final cumulative checkpoint")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--qwen35-model", required=True)
    parser.add_argument("--pca", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-key64", action="store_true",
                        help="also store the pre-PCA 64x4096 Key64 for the m11 arm")
    args = parser.parse_args()

    _wma_root()
    from eval_framework.datasets.worldmemarena import load_worldmemarena
    from residualmem.latent.frozen_v8_runtime import FrozenV9InstructTokenizer

    bundle = load_worldmemarena(Path(args.dataset), split=args.split)
    ids = args.sample_id or _sample_ids(bundle, args.subcategory)
    by_id = {s.sample_id: s for s in bundle.samples}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise KeyError(f"samples not found: {missing}")
    print(f"[extract] {len(ids)} samples: {', '.join(ids[:6])}{' ...' if len(ids) > 6 else ''}")

    out_dir = Path(args.output_dir)
    tokenizer = FrozenV9InstructTokenizer(
        model_path=args.qwen35_model,
        pca_path=args.pca,
        normalization_path=args.normalization,
        device=args.device,
        use_kernels=False,
    )

    done = skipped = 0
    for sample_id in ids:
        target = out_dir / f"{sample_id}.npz"
        if args.resume and target.exists():
            skipped += 1
            print(f"[extract] skip {sample_id} (exists)")
            continue
        payload = run_sample(
            by_id[sample_id], tokenizer,
            top_k=args.top_k, checkpoint_index=args.checkpoint_index,
            save_key64=args.save_key64,
        )
        save_sample(target, payload)
        done += 1
        print(f"[extract] {sample_id}: {payload['observation_rows']} observations "
              f"/ {payload['total_rows']} rows -> {target.name}", flush=True)
    print(f"[extract] done={done} skipped={skipped}")


if __name__ == "__main__":
    main()
