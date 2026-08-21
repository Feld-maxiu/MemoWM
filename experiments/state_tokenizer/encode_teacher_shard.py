"""Independent GPU worker for official Qwen3-VL fused-document embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def encode_payloads(
    payloads: list[dict[str, str]], *, model_path: str, device: str,
    batch_size: int, show_progress_bar: bool = False,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_path,
        device=device,
        trust_remote_code=True,
        model_kwargs={"dtype": torch.bfloat16},
    )
    output = model.encode(
        payloads,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    output = np.asarray(output, dtype=np.float32)
    if output.shape != (len(payloads), 4096):
        raise ValueError(f"teacher dimension must be 4096, got {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("teacher output contains non-finite values")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    payloads = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    output = encode_payloads(
        payloads, model_path=args.model, device=args.device,
        batch_size=args.batch_size,
    )
    np.save(args.output, output)
    print(json.dumps({"device": args.device, "states": len(payloads), "output": args.output}))


if __name__ == "__main__":
    main()
