from __future__ import annotations

import argparse
import glob
import json
import logging
import time
from pathlib import Path

import numpy as np


PROTOCOL = "residualmem_molmoweb_qformer_encode_v1"



def _jsonl_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield line


def _load_records(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        for line in _jsonl_lines(path):
            if line.strip():
                records.append(json.loads(line))
    return records


def _load_captions(paths: list[Path]) -> dict[str, str]:
    captions: dict[str, str] = {}
    for path in paths:
        for line in _jsonl_lines(path):
            if not line.strip():
                continue
            row = json.loads(line)
            captions[row["state_id"]] = row["caption"]
    return captions


def _arm_a_captions(record: dict) -> tuple[str, ...]:
    parts = [value for value in (record.get("page_title"), record.get("source_url")) if value]
    return (" — ".join(parts),) if parts else ()


def molmoweb_observations(
    records: list[dict], images_root: Path, arm: str, captions: dict[str, str]
):
    from residualmem.benchmarks.worldmemarena_tokenizer import WebObservation

    for record in records:
        state_id = record["state_id"]
        if arm == "arm-a":
            text = _arm_a_captions(record)
        else:
            caption = captions.get(state_id, "").strip()
            text = (caption,) if caption else ()
        yield state_id, WebObservation(
            screenshot=str(images_root / record["screenshot"]),
            user_text="",
            captions=text,
            image_ids=(record["image_id"],),
        )


def wma_observations(root: Path):
    from residualmem.benchmarks.worldmemarena_tokenizer import WebObservation

    for path in sorted(glob.glob(str(root / "*.json"))):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        sample_id = payload["sample_id"]
        index = 0
        for session in payload["sessions"]:
            for turn in session["dialogue"]:
                if turn.get("role") != "user":
                    continue
                attachments = turn.get("attachments") or []
                if not attachments:
                    continue
                captions = tuple(
                    str(a.get("caption") or "").strip()
                    for a in attachments
                    if str(a.get("caption") or "").strip()
                )
                image_ids = tuple(str(a.get("image_id")) for a in attachments if a.get("image_id"))
                screenshot = str(root / attachments[0]["file_path"])
                yield f"{sample_id}-{index:04d}", WebObservation(
                    screenshot=screenshot, user_text="",
                    captions=captions, image_ids=image_ids,
                )
                index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=("arm-a", "arm-b", "wma"))
    parser.add_argument("--records", type=Path, action="append", default=[])
    parser.add_argument("--captions", type=Path, action="append", default=[])
    parser.add_argument("--images", type=Path)
    parser.add_argument("--wma-root", type=Path, action="append", default=[],
                        help="a WorldMemArena subcategory directory; repeatable")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default="models/Qwen3.5-9B")
    parser.add_argument("--queries", type=int, required=True,
                        help="must match the checkpoint; the class default is 16")
    parser.add_argument("--qformer-layers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-limit", type=int, default=None,
                        help="whole trajectories only, so transitions stay intact")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.source == "wma":
        pairs = [p for root in args.wma_root for p in wma_observations(root)]
    else:
        records = _load_records(args.records)
        if args.episode_limit is not None:
            seen: set[str] = set()
            kept = []
            for record in records:
                if record["episode_id"] not in seen:
                    if len(seen) >= args.episode_limit:
                        continue
                    seen.add(record["episode_id"])
                kept.append(record)
            records = kept
        captions = _load_captions(args.captions) if args.source == "arm-b" else {}
        if args.source == "arm-b":
            missing = [r["state_id"] for r in records if r["state_id"] not in captions]
            if missing:
                raise SystemExit(
                    f"arm-b needs a caption per state; {len(missing)} are missing, "
                    f"first {missing[:3]}"
                )
        pairs = list(molmoweb_observations(records, args.images, args.source, captions))

    if args.shard_count > 1:
        pairs = [p for i, p in enumerate(pairs) if i % args.shard_count == args.shard_index]

    from residualmem.latent.qformer_runtime import QFormerInstructTokenizer

    tokenizer = QFormerInstructTokenizer(
        model_path=args.model, checkpoint=args.checkpoint,
        queries=args.queries, layers=args.qformer_layers, device=args.device,
    )
    logging.info("%s: %d observations, checkpoint metadata %s",
                 args.source, len(pairs), tokenizer.metadata)

    ids: list[str] = []
    states: list[np.ndarray] = []
    tree_chars: list[int] = []
    sequence_lengths: list[int] = []
    started = time.time()
    for position, (state_id, observation) in enumerate(pairs):
        output = tokenizer.encode(observation)
        ids.append(state_id)
        states.append(output.xbar.astype(np.float32))
        tree_chars.append(len(output.metadata.get("synthetic_axtree", "")))
        sequence_lengths.append(int(output.metadata.get("original_sequence_length", 0)))
        if position % 200 == 0 and position:
            rate = position / max(time.time() - started, 1e-9)
            logging.info("%d/%d  %.2f obs/s  eta %.1f min",
                         position, len(pairs), rate,
                         (len(pairs) - position) / max(rate, 1e-9) / 60)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        xbar=np.stack(states),
        state_ids=np.asarray(ids, dtype=object),
        synthetic_axtree_chars=np.asarray(tree_chars, np.int32),
        original_sequence_length=np.asarray(sequence_lengths, np.int32),
        metadata=json.dumps({
            "protocol": PROTOCOL,
            "source": args.source,
            "checkpoint": str(args.checkpoint),
            "checkpoint_metadata": {
                k: v for k, v in tokenizer.metadata.items() if k != "monitors"
            },
            "queries": args.queries,
            "count": len(ids),
        }, ensure_ascii=False),
    )
    logging.info("wrote %s  xbar %s", args.output, np.stack(states).shape)


if __name__ == "__main__":
    main()
