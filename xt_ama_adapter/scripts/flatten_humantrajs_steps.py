"""Flatten selected HumanTrajs trajectories into one record per step.

The script preserves trajectory order and materializes the corresponding image
bytes. It does not invent QA labels; those are generated in a later stage.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="JSONL containing sample_id (semantic or structural manifest)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, required=True)
    args = ap.parse_args()

    wanted = set()
    for line in args.manifest.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            if row.get("sample_id"):
                wanted.add(row["sample_id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)
    n_traj = n_steps = n_images = 0
    with args.output.open("w", encoding="utf-8") as out:
        table = pq.read_table(args.parquet)
        for row in table.to_pylist():
            sid = row.get("sample_id")
            if sid not in wanted:
                continue
            traj = json.loads(row["trajectory"])
            image_bytes = row.get("images") or []
            image_paths = row.get("image_paths") or []
            step_keys = sorted(traj, key=lambda x: int(x) if str(x).isdigit() else str(x))
            for pos, key in enumerate(step_keys):
                step = traj[key]
                rel = image_paths[pos] if pos < len(image_paths) else step.get("screenshot", f"step_{key}.png")
                suffix = Path(rel).suffix or ".png"
                image_path = args.image_dir / f"{sid}_step_{key}{suffix}"
                if pos < len(image_bytes) and image_bytes[pos] and not image_path.exists():
                    image_path.write_bytes(image_bytes[pos])
                    n_images += 1
                action = step.get("action", {})
                obs = step.get("other_obs", {})
                record = {
                    "trajectory_id": sid,
                    "step_idx": int(key) if str(key).isdigit() else pos,
                    "action": action,
                    "observation": obs,
                    "image_path": str(image_path),
                    "instruction": row.get("instruction"),
                    "question": None,
                    "answer": None,
                    "gold_step_ids": [],
                    "qa_source": None,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_steps += 1
            n_traj += 1
    print(json.dumps({"trajectories": n_traj, "steps": n_steps, "images_written": n_images,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
