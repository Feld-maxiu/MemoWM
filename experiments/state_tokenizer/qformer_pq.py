from __future__ import annotations

import argparse
import glob
import json
import logging
import time
from pathlib import Path

import numpy as np


PROTOCOL = "residualmem_qformer_pq_v1"



def _jsonl_lines(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield line


def load_states(patterns: list[str]) -> tuple[np.ndarray, list[str], str]:
    paths: list[str] = []
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if not found:
            raise SystemExit(f"no files match {pattern!r}")
        paths.extend(found)
    blocks, ids, checkpoints = [], [], set()
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            blocks.append(np.asarray(data["xbar"], np.float32))
            ids.extend(str(value) for value in np.asarray(data["state_ids"]))
            checkpoints.add(json.loads(str(np.asarray(data["metadata"])))["checkpoint"])
    if len(checkpoints) != 1:
        raise SystemExit(f"states span multiple checkpoints: {sorted(checkpoints)}")
    states = np.concatenate(blocks)
    if len(states) != len(set(ids)):
        raise SystemExit(f"{len(states)} states but {len(set(ids))} unique ids")
    return states, ids, next(iter(checkpoints))


def load_splits(paths: list[Path]) -> dict[str, str]:
    splits: dict[str, str] = {}
    for path in paths:
        for line in _jsonl_lines(path):
            if line.strip():
                row = json.loads(line)
                splits[row["state_id"]] = row["split"]
    return splits


def fit_standardizer(train: np.ndarray, *, floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), floor)
    return mean.astype(np.float32), scale.astype(np.float32)


def _balanced_partition(variance: np.ndarray, num_subspaces: int) -> np.ndarray:
    per_subspace = len(variance) // num_subspaces
    order = np.argsort(variance)[::-1]
    buckets: list[list[int]] = [[] for _ in range(num_subspaces)]
    totals = np.zeros(num_subspaces)
    for dimension in order:
        candidates = [i for i in range(num_subspaces) if len(buckets[i]) < per_subspace]
        chosen = candidates[int(np.argmin(totals[candidates]))]
        buckets[chosen].append(int(dimension))
        totals[chosen] += variance[dimension]
    return np.asarray([d for bucket in buckets for d in bucket], np.int64)


def fit_rotation(
    train: np.ndarray, *, num_subspaces: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    if mode not in ("per-slot", "shared"):
        raise ValueError(f"unknown rotation mode {mode!r}")
    count, slots, dim = train.shape
    bases = np.empty((slots if mode == "per-slot" else 1, dim, dim), np.float32)
    order = np.empty((slots if mode == "per-slot" else 1, dim), np.int64)

    if mode == "shared":
        pooled = train.reshape(count * slots, dim)
        centred = pooled - pooled.mean(0)
        _, _, basis = np.linalg.svd(centred, full_matrices=False)
        bases[0] = basis
        order[0] = _balanced_partition(np.square(centred @ basis.T).mean(0), num_subspaces)
        return bases, order

    for slot in range(slots):
        centred = train[:, slot, :] - train[:, slot, :].mean(0)
        _, _, basis = np.linalg.svd(centred, full_matrices=False)
        bases[slot] = basis
        order[slot] = _balanced_partition(
            np.square(centred @ basis.T).mean(0), num_subspaces
        )
    return bases, order


def rotate(states: np.ndarray, bases: np.ndarray, order: np.ndarray) -> np.ndarray:
    output = np.empty_like(states)
    for slot in range(states.shape[1]):
        index = slot if len(bases) > 1 else 0
        output[:, slot, :] = (states[:, slot, :] @ bases[index].T)[:, order[index]]
    return output


def unrotate(rotated: np.ndarray, bases: np.ndarray, order: np.ndarray) -> np.ndarray:
    output = np.empty_like(rotated)
    for slot in range(rotated.shape[1]):
        index = slot if len(bases) > 1 else 0
        restored = np.empty_like(rotated[:, slot, :])
        restored[:, order[index]] = rotated[:, slot, :]
        output[:, slot, :] = restored @ bases[index]
    return output


def fit_codebook(
    train: np.ndarray, *, num_subspaces: int, num_categories: int,
    seed: int, iterations: int, problem_batch: int,
) -> tuple[np.ndarray, np.ndarray]:
    from .a2_categorical_bottleneck import kmeans_codebook

    count, slots, dim = train.shape
    if dim % num_subspaces:
        raise SystemExit(f"{dim} channels do not divide into {num_subspaces} subspaces")
    subspace_dim = dim // num_subspaces
    if num_categories > count:
        raise SystemExit(
            f"{num_categories} clusters exceed {count} train states per problem"
        )

    blocked = train.reshape(count, slots, num_subspaces, subspace_dim)
    problems = slots * num_subspaces
    centroids = np.empty((problems, num_categories, subspace_dim), np.float32)
    occupancy = np.empty((problems, num_categories), np.int64)

    started = time.time()
    for start in range(0, problems, problem_batch):
        stop = min(start + problem_batch, problems)
        ids = np.arange(start, stop)
        points = np.ascontiguousarray(
            np.transpose(blocked[:, ids // num_subspaces, ids % num_subspaces, :], (1, 0, 2))
        )
        block_centroids, _, block_occupancy = kmeans_codebook(
            points, num_categories, seed, iterations,
            chunk=problem_batch, problem_offset=start, return_occupancy=True,
        )
        centroids[start:stop] = np.asarray(block_centroids, np.float32)
        occupancy[start:stop] = block_occupancy
        elapsed = time.time() - started
        logging.info("k-means %d/%d problems  %.1f min elapsed  eta %.1f min",
                     stop, problems, elapsed / 60,
                     elapsed / max(stop, 1) * (problems - stop) / 60)

    return centroids.reshape(slots, num_subspaces, num_categories, subspace_dim), occupancy


def encode(
    states: np.ndarray, mean, scale, centroids, *,
    bases=None, order=None, batch: int = 512,
) -> np.ndarray:
    count, slots, dim = states.shape
    _, num_subspaces, num_categories, subspace_dim = centroids.shape
    codes = np.empty((count, slots, num_subspaces), np.uint8)
    centroid_sq = np.square(centroids, dtype=np.float32).sum(-1)
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        block = ((states[start:stop] - mean) / scale).astype(np.float32)
        if bases is not None:
            block = rotate(block, bases, order)
        block = block.reshape(stop - start, slots, num_subspaces, subspace_dim)
        cross = np.einsum("bsmd,smcd->bsmc", block, centroids, optimize=True)
        codes[start:stop] = (centroid_sq - 2.0 * cross).argmin(-1).astype(np.uint8)
    return codes


def decode(codes: np.ndarray, mean, scale, centroids, *, bases=None, order=None) -> np.ndarray:
    count, slots, num_subspaces = codes.shape
    subspace_dim = centroids.shape[-1]
    flat = np.empty((count, slots, num_subspaces, subspace_dim), np.float32)
    for slot in range(slots):
        for subspace in range(num_subspaces):
            flat[:, slot, subspace] = centroids[slot, subspace][codes[:, slot, subspace]]
    rebuilt = flat.reshape(count, slots, num_subspaces * subspace_dim)
    if bases is not None:
        rebuilt = unrotate(rebuilt, bases, order)
    return rebuilt * scale + mean


def opq_round(
    standardized: np.ndarray, bases: np.ndarray, order: np.ndarray, *,
    num_subspaces: int, num_categories: int, seed: int, iterations: int,
    problem_batch: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    dim = standardized.shape[-1]
    rotated = rotate(standardized, bases, order)
    centroids, _ = fit_codebook(
        rotated, num_subspaces=num_subspaces, num_categories=num_categories,
        seed=seed, iterations=iterations, problem_batch=problem_batch,
    )
    zero, one = np.zeros((), np.float32), np.ones((), np.float32)
    target = decode(encode(rotated, zero, one, centroids, bases=None),
                    zero, one, centroids, bases=None)
    mse = float(np.square(rotated - target, dtype=np.float64).mean())

    def procrustes(left_block, right_block):
        left, _, right = np.linalg.svd(
            left_block.reshape(-1, dim).T.astype(np.float64)
            @ right_block.reshape(-1, dim).astype(np.float64)
        )
        return (left @ right).astype(np.float32)

    if len(bases) == 1:
        step = procrustes(rotated, target)
        new_bases = (bases[0].T[:, order[0]] @ step).T[None, :, :].copy()
        new_order = np.arange(dim, dtype=order.dtype)[None, :]
    else:
        new_bases = np.empty_like(bases)
        for slot in range(standardized.shape[1]):
            step = procrustes(rotated[:, slot, :], target[:, slot, :])
            new_bases[slot] = (bases[slot].T[:, order[slot]] @ step).T
        new_order = np.tile(np.arange(dim, dtype=order.dtype),
                            (len(bases), 1))
    return new_bases, new_order, mse


def reconstruction_metrics(states: np.ndarray, rebuilt: np.ndarray) -> dict:
    residual = np.square(states - rebuilt, dtype=np.float64).sum()
    spread = np.square(states - states.mean(axis=0, keepdims=True), dtype=np.float64).sum()
    return {
        "mse": float(residual / states.size),
        "rmse": float(np.sqrt(residual / states.size)),
        "r2": float(1.0 - residual / max(spread, 1e-12)),
        "states": int(len(states)),
    }


def codebook_health(codes: np.ndarray, num_categories: int) -> dict:
    count, slots, num_subspaces = codes.shape
    flat = codes.reshape(count, slots * num_subspaces).astype(np.int64)
    offsets = np.arange(flat.shape[1], dtype=np.int64) * num_categories
    histogram = np.bincount(
        (flat + offsets).ravel(), minlength=flat.shape[1] * num_categories
    ).reshape(slots * num_subspaces, num_categories).astype(np.float64)
    frequency = histogram / np.maximum(histogram.sum(-1, keepdims=True), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(frequency > 0, frequency * np.log2(frequency), 0.0)
    perplexity = np.power(2.0, -terms.sum(-1))
    active = (histogram > 0).sum(-1)
    return {
        "perplexity_min": float(perplexity.min()),
        "perplexity_median": float(np.median(perplexity)),
        "perplexity_mean": float(perplexity.mean()),
        "active_codes_min": int(active.min()),
        "active_codes_median": int(np.median(active)),
        "active_codes_mean": float(active.mean()),
        "num_categories": int(num_categories),
        "problems": int(slots * num_subspaces),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", action="append", required=True,
                        help="glob over encoded .npz shards; repeatable")
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-subspaces", type=int, default=32)
    parser.add_argument("--num-categories", type=int, default=256)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--problem-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fit-states", type=int, default=0,
                        help="cap the train states used for k-means; 0 = all")
    parser.add_argument("--extra-states", action="append", default=[],
                        help="further shards to encode but never fit on, e.g. WMA web")
    parser.add_argument("--extra-label", default="extra")
    parser.add_argument("--mix-states", action="append", default=[],
                        help="glob over states to include in the FIT only; never "
                             "encoded, never evaluated. Must not be the eval set")
    parser.add_argument("--mix-share", type=float, default=0.10,
                        help="fraction of the fit set drawn from --mix-states")
    parser.add_argument("--rotation", choices=("per-slot", "shared", "none"),
                        default="shared",
                        help="OPQ basis; 'none' is the un-rotated contiguous split")
    parser.add_argument("--opq-rounds", type=int, default=0,
                        help="OPQ_NP alternations after the parametric init. 0 "
                             "(the default) reproduces the previous behaviour "
                             "byte for byte. 6 rounds measured R^2 0.98755 -> "
                             "0.98963 on MolmoWeb and 0.97052 -> 0.97243 on WMA "
                             "web, still improving, with the residual-coding "
                             "ratio unchanged at 1.109")
    parser.add_argument("--opq-kmeans-iterations", type=int, default=10,
                        help="k-means iterations inside each alternation; the "
                             "final codebook still uses --kmeans-iterations")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="also report the held-out test split; off by default")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    states, ids, checkpoint = load_states(args.states)
    splits = load_splits(args.records)
    missing = [i for i in ids if i not in splits]
    if missing:
        raise SystemExit(f"{len(missing)} encoded states have no record, e.g. {missing[:3]}")
    assignment = np.asarray([splits[i] for i in ids])

    train_rows = np.flatnonzero(assignment == "train")
    if args.fit_states and len(train_rows) > args.fit_states:
        rng = np.random.default_rng(args.seed)
        train_rows = np.sort(rng.choice(train_rows, args.fit_states, replace=False))
    train = states[train_rows]

    mix_count = 0
    if args.mix_states:
        mix, mix_ids, mix_checkpoint = load_states(args.mix_states)
        if Path(mix_checkpoint).resolve() != Path(checkpoint).resolve():
            raise SystemExit(
                f"mix states came from {mix_checkpoint}, not {checkpoint}"
            )
        wanted = int(round(len(train) * args.mix_share / (1.0 - args.mix_share)))
        rng = np.random.default_rng(args.seed)
        take = min(wanted, len(mix))
        if take < wanted:
            logging.warning("mix wants %d states but only %d exist; using all",
                            wanted, len(mix))
        chosen = rng.choice(len(mix), take, replace=False)
        train = np.concatenate([train, mix[chosen]])
        mix_count = take
        logging.info("mixing %d fit-only states (%.1f%% of the fit set)",
                     take, 100.0 * take / len(train))

    logging.info("fitting on %d states (%d MolmoWeb train + %d mixed) of %d total",
                 len(train), len(train) - mix_count, mix_count, len(states))

    mean, scale = fit_standardizer(train)
    standardized = (train - mean) / scale
    bases = order = None
    opq_trace: list[float] = []
    if args.rotation != "none":
        bases, order = fit_rotation(
            standardized, num_subspaces=args.num_subspaces, mode=args.rotation
        )
        probe = standardized[:64]
        drift = float(np.abs(probe - unrotate(rotate(probe, bases, order), bases, order)).max())
        if drift > 1e-3:
            raise SystemExit(f"rotation round-trip is not the identity: max drift {drift:.2e}")
        logging.info("rotation %s: bases %s, round-trip drift %.2e",
                     args.rotation, bases.shape, drift)
        for round_index in range(1, args.opq_rounds + 1):
            bases, order, in_frame_mse = opq_round(
                standardized, bases, order,
                num_subspaces=args.num_subspaces,
                num_categories=args.num_categories,
                seed=args.seed, iterations=args.opq_kmeans_iterations,
                problem_batch=args.problem_batch,
            )
            opq_trace.append(in_frame_mse)
            logging.info("opq round %d/%d: in-frame MSE %.6f",
                         round_index, args.opq_rounds, in_frame_mse)
        if args.opq_rounds:
            probe = standardized[:64]
            drift = float(np.abs(
                probe - unrotate(rotate(probe, bases, order), bases, order)
            ).max())
            if drift > 1e-3:
                raise SystemExit(
                    f"rotation round-trip broke after OPQ: max drift {drift:.2e}"
                )
            logging.info("after %d OPQ rounds: round-trip drift %.2e",
                         args.opq_rounds, drift)
        standardized = rotate(standardized, bases, order)
    centroids, occupancy = fit_codebook(
        standardized,
        num_subspaces=args.num_subspaces, num_categories=args.num_categories,
        seed=args.seed, iterations=args.kmeans_iterations,
        problem_batch=args.problem_batch,
    )

    report: dict = {
        "protocol": PROTOCOL,
        "checkpoint": checkpoint,
        "slots": int(states.shape[1]),
        "channels": int(states.shape[2]),
        "num_subspaces": args.num_subspaces,
        "num_categories": args.num_categories,
        "fixed_width_bits": int(states.shape[1] * args.num_subspaces
                                * np.log2(args.num_categories)),
        "rotation": args.rotation,
        "opq_rounds": args.opq_rounds,
        "opq_in_frame_mse_trace": opq_trace,
        "mix": {"states": mix_count, "share": args.mix_share,
                "sources": list(args.mix_states)},
        "kmeans": {
            "iterations": args.kmeans_iterations,
            "seed": args.seed,
            "fit_states": int(len(train)),
            "empty_clusters": int((occupancy == 0).sum()),
            "occupancy_min": int(occupancy.min()),
            "occupancy_median": float(np.median(occupancy)),
        },
        "splits": {},
    }

    codes_by_split: dict[str, np.ndarray] = {}
    ids_by_split: dict[str, list[str]] = {}
    for name in sorted(set(assignment.tolist())):
        if name == "test" and not args.evaluate_test:
            logging.info("skipping the test split; pass --evaluate-test to report it")
            continue
        rows = np.flatnonzero(assignment == name)
        subset = states[rows]
        codes = encode(subset, mean, scale, centroids, bases=bases, order=order)
        rebuilt = decode(codes, mean, scale, centroids, bases=bases, order=order)
        codes_by_split[name] = codes
        ids_by_split[name] = [ids[r] for r in rows]
        report["splits"][name] = {
            **reconstruction_metrics(subset, rebuilt),
            "code_health": codebook_health(codes, args.num_categories),
        }
        logging.info("%s: r2 %.4f  rmse %.4f  perplexity(median) %.1f",
                     name, report["splits"][name]["r2"],
                     report["splits"][name]["rmse"],
                     report["splits"][name]["code_health"]["perplexity_median"])

    for pattern in args.extra_states:
        extra, extra_ids, extra_checkpoint = load_states([pattern])
        if extra_checkpoint != checkpoint:
            raise SystemExit(
                f"{pattern} was encoded by {extra_checkpoint}, not {checkpoint}"
            )
        codes = encode(extra, mean, scale, centroids, bases=bases, order=order)
        rebuilt = decode(codes, mean, scale, centroids, bases=bases, order=order)
        codes_by_split[args.extra_label] = codes
        ids_by_split[args.extra_label] = extra_ids
        report["splits"][args.extra_label] = {
            **reconstruction_metrics(extra, rebuilt),
            "code_health": codebook_health(codes, args.num_categories),
        }
        logging.info("%s: r2 %.4f  rmse %.4f",
                     args.extra_label, report["splits"][args.extra_label]["r2"],
                     report["splits"][args.extra_label]["rmse"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mean": mean, "scale": scale, "centroids": centroids,
        "occupancy": occupancy, "report": json.dumps(report, ensure_ascii=False),
    }
    if bases is not None:
        payload["rotation_bases"] = bases
        payload["rotation_order"] = order
    for name, codes in codes_by_split.items():
        payload[f"codes/{name}"] = codes
        payload[f"state_ids/{name}"] = np.asarray(ids_by_split[name], dtype=object)
    np.savez_compressed(args.output, **payload)
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
