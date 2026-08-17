"""Write the modality-length stub that ``extract_fixed_prompt`` reads.

``extract_fixed_prompt`` needs the per-record (image, DOM, prompt) token counts up
front, to size the ragged token array before any forward pass. It normally reads
them from an ``extract_qwen`` run -- but that run also computes layer-16 hidden
states for every state, which is by far the most expensive step in the pipeline
and is not otherwise needed for the ``key64_static_pca`` chain.

The counts depend only on the processor's tokenisation of the screenshot and the
DOM, so they can be produced without a single forward pass. Nothing here is
trusted on faith: ``extract_fixed_prompt`` re-derives the true lengths per record
and raises on any mismatch, so a wrong stub fails loudly rather than corrupting
the extraction.

The model is loaded only for ``config.image_token_id`` and the marker token ids;
no weights are ever executed.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .common import iter_jsonl, write_json
from .extract_qwen import _load_model, modality_indices, prepare_inputs

LOGGER = logging.getLogger("modality_lengths")


def run(args: argparse.Namespace) -> dict:
    records_path = Path(args.records).resolve()
    records = list(iter_jsonl(records_path))
    device = torch.device(args.device)
    processor, model = _load_model(args.model, device, args.use_kernels)

    rows = list(range(args.rank, len(records), args.world_size))
    lengths = np.zeros((len(rows), 3), np.int32)
    truncated = np.zeros((len(rows),), bool)
    for position, row in enumerate(rows):
        record = records[row]
        image_path = records_path.parent / record["screenshot"]
        with Image.open(image_path) as handle:
            inputs, cut, _, _ = prepare_inputs(
                processor, handle.convert("RGB"), record["dom"],
                record["instruction"], args.max_length,
            )
        indices = modality_indices(processor, model, inputs["input_ids"])
        lengths[position] = [len(index) for index in indices]
        truncated[position] = cut
        if position % 250 == 0:
            LOGGER.info("rank %d: %d/%d", args.rank, position, len(rows))

    output = Path(args.output) / f"worker{args.rank:02d}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "subset_rows.npy", np.asarray(rows, np.int64))
    np.save(output / "record_indices.npy",
            np.asarray([int(records[row]["global_index"]) for row in rows], np.int64))
    np.save(output / "modality-lengths.npy", lengths)
    np.save(output / "truncated.npy", truncated)
    np.save(output / "done.npy", np.ones((len(rows),), bool))
    summary = {
        "records": len(rows),
        "truncated": int(truncated.sum()),
        "note": "modality lengths only; no hidden states, no forward pass",
        "median_dom_tokens": int(np.median(lengths[:, 1])),
        "max_total_tokens": int(lengths.sum(axis=1).max()),
    }
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--use-kernels", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)
    print(run(args))


if __name__ == "__main__":
    main()
