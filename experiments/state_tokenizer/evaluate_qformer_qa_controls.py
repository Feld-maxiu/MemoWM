"""Evaluate a frozen Q-Former against shuffled/zero/question-only controls.

The trainer's validation CE scores only matched latents.  This audit reuses its
fixed observation-uniform validation selection and proves that the reader is
using observation-specific state rather than merely the question prior.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from residualmem.latent.qformer_runtime import QFormerInstructTokenizer

from .reader_losses import answer_ce_and_distill_kl
from .train_qformer_joint import ObservationStore, validate_pairs_metadata
from .trunk_states import collate, text_trunk_states, trunk_states


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _groups(pairs: dict[str, np.ndarray], rows: np.ndarray):
    grouped: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    for raw in rows:
        row = int(raw)
        grouped[(str(pairs["sample_id"][row]), int(pairs["record_index"][row]))].append(row)
    # Preserve first occurrence order exactly as train_qformer_joint.grouped;
    # the seed+1 choice is only the same validation subset under that order.
    return [(key, np.asarray(value, dtype=np.int64))
            for key, value in grouped.items()]


def _partners(groups):
    """Prefer a wrong state from the same trajectory, then any other state."""
    result = {}
    for position, (key, _rows) in enumerate(groups):
        sample_id, _ = key
        same = [i for i, (candidate, _r) in enumerate(groups)
                if i != position and candidate[0] == sample_id]
        if same:
            result[position] = same[0]
            continue
        if len(groups) < 2:
            raise ValueError("shuffled control needs at least two observations")
        result[position] = (position + 1) % len(groups)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--xbar-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--qformer-layers", type=int, default=4)
    parser.add_argument("--observations", type=int, default=96)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.queries != 32:
        raise ValueError("formal HumanTrajs control audit is registered for K32")

    with np.load(args.pairs, allow_pickle=False) as raw:
        required = {"sample_id", "record_index", "question", "answer", "split", "metadata"}
        missing = required - set(raw.files)
        if missing:
            raise ValueError(f"pairs lacks {sorted(missing)}")
        pairs = {name: np.asarray(raw[name]) for name in required - {"metadata"}}
        pairs_metadata = json.loads(str(np.asarray(raw["metadata"]).item()))
    validate_pairs_metadata(pairs_metadata)
    all_groups = _groups(pairs, np.flatnonzero(pairs["split"].astype(str) == "validation"))
    rng = np.random.default_rng(args.seed + 1)
    if len(all_groups) > args.observations:
        chosen = sorted(map(int, rng.choice(len(all_groups), args.observations, replace=False)))
        groups = [all_groups[index] for index in chosen]
    else:
        groups = all_groups
    if not groups:
        raise ValueError("no validation observations")

    tokenizer = QFormerInstructTokenizer(
        model_path=args.model,
        checkpoint=args.checkpoint,
        queries=args.queries,
        layers=args.qformer_layers,
        self_attention=False,
        qk_norm=True,
        device=args.device,
    )
    processor, model, reader = tokenizer.processor, tokenizer.model, tokenizer.reader
    store = ObservationStore(args.xbar_dir)
    states_on_cpu = []
    for position, ((sample_id, index), _rows) in enumerate(groups, 1):
        record = store.observation(sample_id, index)
        if record.get("observation_protocol") == "qwen35_9b_visual_transcription_v1":
            states = text_trunk_states(
                processor, model, str(record["text_observation"]),
                layer=tokenizer.layer, device=tokenizer.device,
            )
        else:
            with Image.open(record["screenshot"]) as handle:
                image = handle.convert("RGB")
            states = trunk_states(
                processor, model, image, record["synthetic_axtree"],
                layer=tokenizer.layer, device=tokenizer.device,
            )
        with torch.inference_mode():
            _soft, xbar, valid = reader(*collate([states]))
        states_on_cpu.append((xbar.float().cpu(), valid.cpu()))
        if position % 8 == 0 or position == len(groups):
            print(f"[qa-controls] encoded {position}/{len(groups)} observations", flush=True)

    partners = _partners(groups)
    totals = {name: [] for name in ("matched", "shuffled", "zero", "question_only")}
    matched_beats_shuffled = []
    matched_beats_zero = []
    with torch.inference_mode():
        for position, ((_key, rows), (xbar_cpu, valid_cpu)) in enumerate(
            zip(groups, states_on_cpu), 1
        ):
            xbar = xbar_cpu.to(tokenizer.device)
            valid = valid_cpu.to(tokenizer.device)
            other_xbar, other_valid = states_on_cpu[partners[position - 1]]
            soft_matched = reader.connector(xbar, valid)
            soft_shuffled = reader.connector(
                other_xbar.to(tokenizer.device), other_valid.to(tokenizer.device)
            )
            soft_zero = torch.zeros_like(soft_matched)
            empty_soft = soft_matched[:, :0]
            empty_valid = valid[:, :0]
            observation_values = {name: [] for name in totals}
            for raw_row in rows:
                row = int(raw_row)
                question, answer = str(pairs["question"][row]), str(pairs["answer"][row])
                controls = {
                    "matched": (soft_matched, valid),
                    "shuffled": (soft_shuffled, other_valid.to(tokenizer.device)),
                    "zero": (soft_zero, valid),
                    "question_only": (empty_soft, empty_valid),
                }
                values = {}
                for name, (soft, mask) in controls.items():
                    _loss, ce, _kl = answer_ce_and_distill_kl(
                        model, processor, soft, mask, question, answer, "", weight=0.0
                    )
                    values[name] = ce
                    observation_values[name].append(ce)
                matched_beats_shuffled.append(values["matched"] < values["shuffled"])
                matched_beats_zero.append(values["matched"] < values["zero"])
            for name in totals:
                totals[name].append(float(np.mean(observation_values[name])))
            if position % 8 == 0 or position == len(groups):
                print(f"[qa-controls] scored {position}/{len(groups)} observations", flush=True)

    means = {f"{name}_validation_answer_ce": float(np.mean(values))
             for name, values in totals.items()}
    report = {
        "protocol": "qformer_qa_controls_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": _sha256(Path(args.checkpoint)),
        "pairs": str(Path(args.pairs).resolve()),
        "pairs_sha256": _sha256(Path(args.pairs)),
        "selection": "trainer_seed_plus_one_observation_uniform",
        "seed": args.seed,
        "observations": len(groups),
        "qa_rows": int(sum(len(rows) for _key, rows in groups)),
        **means,
        "matched_minus_shuffled_ce": (
            means["matched_validation_answer_ce"] - means["shuffled_validation_answer_ce"]
        ),
        "matched_minus_zero_ce": (
            means["matched_validation_answer_ce"] - means["zero_validation_answer_ce"]
        ),
        "matched_beats_shuffled": float(np.mean(matched_beats_shuffled)),
        "matched_beats_zero": float(np.mean(matched_beats_zero)),
        "write_side_question_independent": True,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
