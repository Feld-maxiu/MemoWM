"""D1-D4: where the cross-domain loss comes from, decomposed.

Two decompositions, both Shapley, both reported against the empty control.

**D2, modality (2 players: V = screenshot, T = synthetic-AXTree text).** The
retrieval head is nonlinear, so ``f(V,T) != f(V,0) + f(0,T) - f(0,0)`` and the
naive "visual explains X%, text explains Z%" reading is unavailable. With

    phi_V = 1/2 [(m10 - m00) + (m11 - m01)]
    phi_T = 1/2 [(m01 - m00) + (m11 - m10)]

efficiency holds exactly -- ``phi_V + phi_T == m11 - m00`` -- so the shares are a
legitimate partition of the signal the full representation provides *over the
empty control*. The interaction

    I_VT = m11 - m10 - m01 + m00

is reported alongside and is not optional: when it is not small next to the two
attributions, the modalities are entangled and no per-modality share is
meaningful no matter how the arithmetic is arranged.

**D1, slot group (3 players: image / detail / context).** Same construction over
2^3 subsets. This one is nearly free -- the subsets are post-hoc masks on the
already-extracted ``xbar``, and the head is tiny -- whereas D2's arms each cost a
separate 9B forward, which is why they were extracted up front.

Value functions are evaluated per observation and only then averaged. Averaging
first and substituting into the formulas is not the same thing for a nonlinear
metric, and would silently change the answer.

**D3, resolution.** Two controls sit beside ``m11``: a direct resize to the
training capture size, and an aspect-preserving letterbox onto the same size.
The direct resize changes resolution *and* squashes 16:9 into 1.55:1, so only
the letterbox arm isolates resolution from geometry.

**D4, bucketing.** Per handover 9.8. On WorldMemArena web the screenshot bucket
is degenerate -- every observation has one -- so only caption length and session
length carry variation, and that is stated rather than silently dropped.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.state_tokenizer.slot_layout import GROUP_NAMES, KEY64_LAYOUT
from residualmem.latent.instruct_bridge import (
    RETRIEVAL_BRIDGE_PROTOCOL,
    MaskedAttentionRetrievalHead,
    load_bridge,
)

MODALITY_ARMS = {(1, 1): "m11", (1, 0): "m10", (0, 1): "m01", (0, 0): "m00"}
SLOT_GROUPS = ("image", "detail", "context")

GROUP_SPANS = {}
_start = 0
for _name, _size in zip(GROUP_NAMES, KEY64_LAYOUT):
    GROUP_SPANS[_name] = (_start, _start + _size)
    _start += _size


def head_vectors(head, xbar: np.ndarray, valid: np.ndarray, device) -> np.ndarray:
    out = []
    for start in range(0, len(xbar), 128):
        x = torch.as_tensor(xbar[start:start + 128], device=device)
        v = torch.as_tensor(valid[start:start + 128], device=device)
        with torch.inference_mode():
            out.append(head(x, v).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def paired_cosine(pred: np.ndarray, teacher: np.ndarray) -> np.ndarray:
    teacher = teacher / np.maximum(np.linalg.norm(teacher, axis=1, keepdims=True), 1e-12)
    return (pred * teacher).sum(axis=1)


def same_sample_rank_metrics(pred: np.ndarray, teacher: np.ndarray) -> dict:
    """N-way retrieval of the matching teacher row, within one sample.

    Ranking across samples would be a different and easier task: rows from other
    sessions are trivially distinguishable, which inflates every number.
    """
    teacher = teacher / np.maximum(np.linalg.norm(teacher, axis=1, keepdims=True), 1e-12)
    scores = pred @ teacher.T
    n = len(scores)
    order = np.argsort(-scores, axis=1)
    rank = (order == np.arange(n)[:, None]).argmax(axis=1) + 1
    return {
        "n_way": n,
        "recall_at_1": float((rank <= 1).mean()),
        "recall_at_5": float((rank <= 5).mean()),
        "recall_at_10": float((rank <= 10).mean()),
        "mrr": float((1.0 / rank).mean()),
        "chance_recall_at_1": 1.0 / n,
    }


def shapley(values: dict[tuple[int, ...], np.ndarray], players: int) -> dict:
    """Exact Shapley over ``players`` binary features, per element.

    ``values`` maps a coalition bitmask tuple to a per-observation value vector.
    Returns per-player attributions averaged over observations, plus the
    efficiency residual, which must be ~0 by construction.
    """
    n = players
    factorial = math.factorial
    phi = [np.zeros_like(values[(0,) * n]) for _ in range(n)]
    for player in range(n):
        for mask in itertools.product((0, 1), repeat=n):
            if mask[player]:
                continue
            with_player = list(mask)
            with_player[player] = 1
            size = sum(mask)
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            phi[player] += weight * (values[tuple(with_player)] - values[mask])
    full = values[(1,) * n]
    empty = values[(0,) * n]
    total = full - empty
    return {
        "phi": [float(p.mean()) for p in phi],
        "phi_std": [float(p.std()) for p in phi],
        "grand": float(total.mean()),
        "empty": float(empty.mean()),
        "full": float(full.mean()),
        "efficiency_residual": float((sum(phi) - total).mean()),
        "max_abs_efficiency_residual": float(np.abs(sum(phi) - total).max()),
    }


def modality_interaction(values: dict) -> dict:
    inter = values[(1, 1)] - values[(1, 0)] - values[(0, 1)] + values[(0, 0)]
    return {"mean": float(inter.mean()), "abs_mean": float(np.abs(inter).mean())}


def mask_groups(xbar, valid, keep: tuple[str, ...]):
    """Zero every slot outside ``keep`` and drop it from the mask."""
    new_valid = np.zeros_like(valid)
    for group in keep:
        lo, hi = GROUP_SPANS[group]
        new_valid[:, lo:hi] = valid[:, lo:hi]
    return xbar * new_valid[..., None], new_valid


def load_sample(xbar_dir: Path, teacher_dir: Path, name: str):
    with np.load(xbar_dir / name, allow_pickle=False) as z:
        meta = json.loads(str(np.asarray(z["metadata"])))
        arms = {}
        for arm in meta["arms"]:
            keys = sorted(k for k in z.files if k.startswith(f"{arm}/xbar/"))
            if not keys:
                continue
            xs = np.stack([z[k] for k in keys])
            vs = np.stack([z[f"{arm}/valid/{k.rsplit('/', 1)[1]}"] for k in keys])
            arms[arm] = (xs, vs)
    with np.load(teacher_dir / name, allow_pickle=False) as z:
        teacher = np.asarray(z["teacher"], np.float32)
    if len(teacher) != len(arms["m11"][0]):
        raise ValueError(f"{name}: teacher/xbar count mismatch")
    return meta, arms, teacher


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xbar-dir", required=True)
    parser.add_argument("--teacher-dir", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-observations", type=int, default=2,
                        help="samples with fewer rows are excluded from ranking metrics")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    head = MaskedAttentionRetrievalHead()
    load_bridge(args.head, head, expected_protocol=RETRIEVAL_BRIDGE_PROTOCOL)
    device = torch.device(args.device)
    head.to(device).eval()

    xbar_dir, teacher_dir = Path(args.xbar_dir), Path(args.teacher_dir)
    names = sorted(p.name for p in xbar_dir.glob("*.npz"))

    modality_values: dict[tuple[int, int], list[np.ndarray]] = {k: [] for k in MODALITY_ARMS}
    slot_values: dict[tuple[int, int, int], list[np.ndarray]] = {
        m: [] for m in itertools.product((0, 1), repeat=3)
    }
    per_sample, records, resolution = [], [], {}
    for name in names:
        meta, arms, teacher = load_sample(xbar_dir, teacher_dir, name)
        n = len(teacher)

        cosines = {}
        for bits, arm in MODALITY_ARMS.items():
            xs, vs = arms[arm]
            cos = paired_cosine(head_vectors(head, xs, vs, device), teacher)
            modality_values[bits].append(cos)
            cosines[arm] = cos

        xs, vs = arms["m11"]
        for mask in slot_values:
            keep = tuple(g for g, bit in zip(SLOT_GROUPS, mask) if bit)
            if not keep:
                # No valid slot at all would make the head raise; the empty slot
                # coalition is defined as "the head sees nothing", which is the
                # m00 arm's role in the modality decomposition. Use a single
                # image slot's zeroed value instead: mask everything to zero but
                # keep one slot valid so pooling is defined.
                mx, mv = mask_groups(xs, vs, ("image",))
                mv = np.zeros_like(mv)
                mv[:, 0] = True
                mx = np.zeros_like(mx)
            else:
                mx, mv = mask_groups(xs, vs, keep)
                empty_rows = mv.sum(axis=1) == 0
                if empty_rows.any():
                    mv = mv.copy()
                    mv[empty_rows, GROUP_SPANS[keep[0]][0]] = True
            slot_values[mask].append(paired_cosine(head_vectors(head, mx, mv, device), teacher))

        entry = {
            "sample_id": meta["sample_id"],
            "observations": n,
            "paired_cosine": {a: float(c.mean()) for a, c in cosines.items()},
        }
        for extra in ("m11_resampled", "m11_letterbox"):
            if extra in arms:
                xs2, vs2 = arms[extra]
                cos = paired_cosine(head_vectors(head, xs2, vs2, device), teacher)
                entry["paired_cosine"][extra] = float(cos.mean())
                resolution.setdefault(extra, []).append(cos)
        if n >= args.min_observations:
            pred = head_vectors(head, *arms["m11"], device)
            entry["ranking_m11"] = same_sample_rank_metrics(pred, teacher)
        per_sample.append(entry)
        for record in meta["records"]:
            record["sample_id"] = meta["sample_id"]
            records.append(record)
        print(f"[attribution] {meta['sample_id']}: n={n} "
              f"cos(m11)={entry['paired_cosine']['m11']:.4f}", flush=True)

    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    modality = cat(modality_values)
    slots = cat(slot_values)

    report = {
        "protocol": "wma_shift_attribution_v1",
        "samples": len(names),
        "observations": int(len(modality[(1, 1)])),
        "d2_modality": {
            "players": ["visual", "text"],
            "arms_mean": {MODALITY_ARMS[k]: float(v.mean()) for k, v in modality.items()},
            "shapley": shapley(modality, 2),
            "interaction_I_VT": modality_interaction(modality),
        },
        "d1_slot_groups": {
            "players": list(SLOT_GROUPS),
            "shapley": shapley(slots, 3),
        },
        "d3_resolution": {
            arm: {"paired_cosine_mean": float(np.concatenate(v).mean())}
            for arm, v in resolution.items()
        },
        "per_sample": per_sample,
    }

    phi = report["d2_modality"]["shapley"]["phi"]
    inter = report["d2_modality"]["interaction_I_VT"]["abs_mean"]
    grand = report["d2_modality"]["shapley"]["grand"]
    report["d2_modality"]["share"] = (
        {"visual": phi[0] / grand, "text": phi[1] / grand} if abs(grand) > 1e-9 else None
    )
    report["d2_modality"]["share_is_reportable"] = bool(
        abs(grand) > 1e-9 and inter < 0.25 * max(abs(p) for p in phi)
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {k: report[k] for k in ("observations", "d2_modality", "d1_slot_groups", "d3_resolution")}
    summary["d2_modality"].pop("per_sample", None)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
