from __future__ import annotations

import argparse
import glob
import json
import re
import time

import numpy as np

from experiments.state_tokenizer.qformer_pq import (
    decode,
    encode,
    fit_codebook,
    fit_rotation,
    fit_standardizer,
    reconstruction_metrics,
    rotate,
    unrotate,
)

STATE_ID = re.compile(r"^(.*)-s?(\d+)$")


def parse_state_id(state_id: str) -> tuple[str, int]:
    match = STATE_ID.match(state_id)
    if not match:
        raise ValueError(f"cannot parse episode/step from {state_id!r}")
    return match.group(1), int(match.group(2))


def _episode_index(ids):
    parsed = [parse_state_id(str(v)) for v in ids]
    return np.array([p[0] for p in parsed]), np.array([p[1] for p in parsed], np.int64)


def load_by_episode(pattern, n_fit, n_eval, seed):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no shards match {pattern}")

    all_ids = []
    for path in files:
        with np.load(path, allow_pickle=True) as handle:
            all_ids.append(np.asarray([str(v) for v in handle["state_ids"]]))
    offsets = np.cumsum([0] + [len(x) for x in all_ids])
    ids = np.concatenate(all_ids)
    episodes, steps = _episode_index(ids)

    rng = np.random.default_rng(seed)
    unique = np.unique(episodes)
    rng.shuffle(unique)
    rank = {name: i for i, name in enumerate(unique)}
    by_episode: dict[str, list[int]] = {}
    for position, name in enumerate(episodes):
        by_episode.setdefault(name, []).append(position)
    wanted_fit, wanted_eval = [], []
    for name in sorted(unique, key=lambda n: rank[n]):
        target = wanted_fit if len(wanted_fit) < n_fit else wanted_eval
        if len(wanted_eval) >= n_eval and len(wanted_fit) >= n_fit:
            break
        target.extend(by_episode[name])
    wanted_fit = np.array(wanted_fit[:n_fit + 4096], np.int64)
    wanted_eval = np.array(wanted_eval[:n_eval + 4096], np.int64)

    selected = np.concatenate([wanted_fit, wanted_eval])
    order = np.argsort(selected)
    gathered = np.empty((len(selected), 32, 512), np.float32)
    for shard, path in enumerate(files):
        low, high = offsets[shard], offsets[shard + 1]
        here = order[(selected[order] >= low) & (selected[order] < high)]
        if not len(here):
            continue
        with np.load(path, allow_pickle=True) as handle:
            xbar = handle["xbar"]
            gathered[here] = np.asarray(xbar[selected[here] - low], np.float32)

    cut = len(wanted_fit)
    return (
        gathered[:cut], episodes[wanted_fit], steps[wanted_fit],
        gathered[cut:], episodes[wanted_eval], steps[wanted_eval],
    )


def consecutive_pairs(episodes, steps):
    order = np.lexsort((steps, episodes))
    same = episodes[order[1:]] == episodes[order[:-1]]
    adjacent = steps[order[1:]] == steps[order[:-1]] + 1
    keep = same & adjacent
    return order[:-1][keep], order[1:][keep]


def load_plain(path):
    if not path:
        return None
    with np.load(path, allow_pickle=True) as handle:
        return np.asarray(handle["xbar"], np.float32)


def predictability(fit_codes, fit_pairs, eval_codes, eval_pairs, categories,
                   alpha=0.5):
    fit_prev, fit_cur = fit_pairs
    ev_prev, ev_cur = eval_pairs
    positions = fit_codes.shape[1] * fit_codes.shape[2]
    fp = fit_codes.reshape(len(fit_codes), positions)
    ep = eval_codes.reshape(len(eval_codes), positions)

    marginal_bits = np.empty(positions)
    bigram_bits = np.empty(positions)
    for j in range(positions):
        prior = np.bincount(fp[fit_cur, j], minlength=categories) + alpha
        marginal = prior / prior.sum()
        marginal_bits[j] = -np.log2(marginal[ep[ev_cur, j]]).mean()

        joint = np.zeros((categories, categories))
        np.add.at(joint, (fp[fit_prev, j], fp[fit_cur, j]), 1.0)
        joint += alpha
        conditional = joint / joint.sum(1, keepdims=True)
        bigram_bits[j] = -np.log2(
            conditional[ep[ev_prev, j], ep[ev_cur, j]]
        ).mean()

    persist = float((ep[ev_prev] == ep[ev_cur]).mean())
    return {
        "positions": int(positions),
        "eval_pairs": int(len(ev_cur)),
        "persistence": persist,
        "marginal_bits_per_position": float(marginal_bits.mean()),
        "bigram_bits_per_position": float(bigram_bits.mean()),
        "marginal_bits_per_transition": float(marginal_bits.sum()),
        "bigram_bits_per_transition": float(bigram_bits.sum()),
    }


def fit_stages(rotated, stages, *, subspaces, seed, iterations):
    residual = rotated
    codebooks = []
    zero, one = np.zeros((), np.float32), np.ones((), np.float32)
    for categories in stages:
        centroids, _ = fit_codebook(
            residual, num_subspaces=subspaces, num_categories=categories,
            seed=seed, iterations=iterations, problem_batch=64,
        )
        codebooks.append(centroids)
        codes = encode(residual, zero, one, centroids, bases=None)
        residual = residual - decode(codes, zero, one, centroids, bases=None)
    return codebooks


def stage_encode(rotated, codebooks):
    zero, one = np.zeros((), np.float32), np.ones((), np.float32)
    residual, out = rotated, []
    for centroids in codebooks:
        codes = encode(residual, zero, one, centroids, bases=None)
        out.append(codes)
        residual = residual - decode(codes, zero, one, centroids, bases=None)
    return out


def stage_decode(code_list, codebooks):
    zero, one = np.zeros((), np.float32), np.ones((), np.float32)
    total = None
    for codes, centroids in zip(code_list, codebooks):
        part = decode(codes, zero, one, centroids, bases=None)
        total = part if total is None else total + part
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True)
    parser.add_argument("--wma-web",
                        help="WorldMemArena agent/gui/web xbar, 956 states -- the "
                             "target distribution, held out from every fit here "
                             "and from the full-corpus codebook's 10%% mix")
    parser.add_argument("--wma-other",
                        help="the 4,558 non-web WMA states; held out in this "
                             "harness too, so same domain at 5x the sample size")
    parser.add_argument("--n-fit", type=int, default=50000)
    parser.add_argument("--n-eval", type=int, default=20000)
    parser.add_argument("--num-subspaces", type=int, default=32)
    parser.add_argument("--num-categories", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kmeans-iterations", type=int, default=25)
    parser.add_argument("--opq-rounds", type=int, default=6)
    parser.add_argument("--opq-kmeans-iterations", type=int, default=10)
    parser.add_argument("--mode", choices=("frontier", "rvq"), default="frontier",
                        help="frontier: rotation variants at one scale. rvq: the "
                             "same 6144-bit budget split across residual stages")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.time()
    width = 32 * args.num_subspaces * int(np.log2(args.num_categories))

    (fit, fit_ep, fit_step,
     ev, ev_ep, ev_step) = load_by_episode(args.states, args.n_fit, args.n_eval,
                                           args.seed)
    fit_pairs = consecutive_pairs(fit_ep, fit_step)
    eval_pairs = consecutive_pairs(ev_ep, ev_step)
    print(f"[load] fit {fit.shape} ({len(np.unique(fit_ep))} episodes, "
          f"{len(fit_pairs[0])} pairs)  eval {ev.shape} "
          f"({len(np.unique(ev_ep))} episodes, {len(eval_pairs[0])} pairs)  "
          f"{time.time() - started:.0f}s", flush=True)

    eval_sets = {"molmoweb_eval": ev}
    for name, path in (("wma_web", args.wma_web), ("wma_other", args.wma_other)):
        loaded = load_plain(path)
        if loaded is not None:
            eval_sets[name] = loaded
            print(f"[load] {name}: {loaded.shape}", flush=True)

    results = {}

    def run_stages(name, bases, order, stages, iterations):
        mean, scale = fit_standardizer(fit)
        fit_s = ((fit - mean) / scale).astype(np.float32)
        rotated = rotate(fit_s, bases, order) if bases is not None else fit_s
        codebooks = fit_stages(rotated, stages, subspaces=args.num_subspaces,
                               seed=args.seed, iterations=iterations)

        def to_rotated(block):
            standard = ((block - mean) / scale).astype(np.float32)
            return rotate(standard, bases, order) if bases is not None else standard

        fidelity = {}
        for split_name, split in eval_sets.items():
            rebuilt_rot = stage_decode(stage_encode(to_rotated(split), codebooks),
                                       codebooks)
            rebuilt = (unrotate(rebuilt_rot, bases, order)
                       if bases is not None else rebuilt_rot)
            fidelity[split_name] = reconstruction_metrics(
                split, rebuilt * scale + mean)

        fit_stage_codes = stage_encode(rotated, codebooks)
        eval_stage_codes = stage_encode(to_rotated(ev), codebooks)
        per_stage, total_bits, weighted_persist = [], 0.0, 0.0
        for index, categories in enumerate(stages):
            stat = predictability(fit_stage_codes[index], fit_pairs,
                                  eval_stage_codes[index], eval_pairs, categories)
            stat["categories"] = int(categories)
            stat["raw_bits_per_position"] = float(np.log2(categories))
            per_stage.append(stat)
            total_bits += stat["bigram_bits_per_transition"]
            weighted_persist += stat["persistence"] * np.log2(categories)
        weighted_persist /= sum(float(np.log2(c)) for c in stages)

        width_here = 32 * args.num_subspaces * sum(int(np.log2(c)) for c in stages)
        positions = per_stage[0]["positions"] * len(stages)
        results[name] = {
            "stages": [int(c) for c in stages],
            "fidelity": fidelity, "per_stage": per_stage,
            "fixed_width_bits": width_here,
            "bigram_bits_per_transition": total_bits,
            "source_ratio": width_here / total_bits,
            "projected_wm_ratio": width_here / max(total_bits - 0.41 * positions, 1.0),
        }
        wma = fidelity.get("wma_web", {}).get("r2", float("nan"))
        stage_note = " ".join(
            f"C{s['categories']}:{s['persistence']*100:.1f}%/"
            f"{s['bigram_bits_per_position']:.2f}b" for s in per_stage)
        print(f"[{name}]  R2 mw {fidelity['molmoweb_eval']['r2']:.5f}  "
              f"wma {wma:.5f}  |  {stage_note}  |  width {width_here}  "
              f"bigram {total_bits:.0f}  ratio {width_here/total_bits:.3f}  "
              f"(WM~{results[name]['projected_wm_ratio']:.3f})  "
              f"{time.time() - started:.0f}s", flush=True)

    def run(name, bases, order, iterations):
        run_stages(name, bases, order, (args.num_categories,), iterations)

    mean, scale = fit_standardizer(fit)
    fit_s = ((fit - mean) / scale).astype(np.float32)

    if args.mode == "rvq":
        shared_bases, shared_order = fit_rotation(
            fit_s, num_subspaces=args.num_subspaces, mode="shared")
        for name, stages in (
            ("flat-C64", (64,)),
            ("rvq-8x8", (8, 8)),
            ("rvq-16x4", (16, 4)),
            ("rvq-4x4x4", (4, 4, 4)),
            ("rvq-32x2", (32, 2)),
        ):
            run_stages(name, shared_bases, shared_order, stages,
                       args.kmeans_iterations)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump({
                "protocol": "residualmem_pq_rvq_probe_v1",
                "n_fit": int(len(fit)), "n_eval": int(len(ev)),
                "num_subspaces": args.num_subspaces,
                "seed": args.seed, "results": results,
            }, handle, indent=2)
        print(f"[done] wrote {args.output}", flush=True)
        return

    run("pq-contiguous", None, None, args.kmeans_iterations)

    shared_bases, shared_order = fit_rotation(
        fit_s, num_subspaces=args.num_subspaces, mode="shared")
    run("opq-shared", shared_bases, shared_order, args.kmeans_iterations)

    slot_bases, slot_order = fit_rotation(
        fit_s, num_subspaces=args.num_subspaces, mode="per-slot")
    run("opq-per-slot", slot_bases, slot_order, args.kmeans_iterations)

    bases, order = shared_bases, shared_order
    for index in range(1, args.opq_rounds + 1):
        rotated = rotate(fit_s, bases, order)
        centroids, _ = fit_codebook(
            rotated, num_subspaces=args.num_subspaces,
            num_categories=args.num_categories, seed=args.seed,
            iterations=args.opq_kmeans_iterations, problem_batch=64,
        )
        zero, one = np.zeros((), np.float32), np.ones((), np.float32)
        target = decode(encode(rotated, zero, one, centroids, bases=None),
                        zero, one, centroids, bases=None)
        left, _, right = np.linalg.svd(
            rotated.reshape(-1, 512).T.astype(np.float64)
            @ target.reshape(-1, 512).astype(np.float64)
        )
        step = (left @ right).astype(np.float32)

        current = bases[0].T[:, order[0]]
        combined = (current @ step)
        bases = combined.T[None, :, :].copy()
        order = np.arange(combined.shape[0])[None, :]
        run(f"opq-iterative-r{index}", bases, order, args.kmeans_iterations)

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump({
            "protocol": "residualmem_pq_frontier_probe_v1",
            "n_fit": int(len(fit)), "n_eval": int(len(ev)),
            "num_subspaces": args.num_subspaces,
            "num_categories": args.num_categories,
            "fixed_width_bits": width,
            "seed": args.seed,
            "results": results,
        }, handle, indent=2)
    print(f"[done] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
