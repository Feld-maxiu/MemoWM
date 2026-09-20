"""Fit the e·V factorized utility gate for WMA from existing counterfactual labels.

Mirrors the LongMemEval Stage-1 protocol (wmaexperiment/lme_gate/ev_gate_stage1.py)
on the WMA gate corpus, reusing the artifacts the published fixed-vector gate was
fitted from -- no new GPU labelling is required:

  labels:  gate/labels-wmfill.npz, gate/labels-slot-s{0,1}.npz  (answer-level |delta NLL|)
  posteriors: gate/posteriors.npz  (entropy_bits, wm_argmax AND target_codes in
              one table, so the per-position error indicator needs no alignment)

Decomposition (exact, since argmax==true makes the substitution a no-op and
contributes 0 to both sides):

    |U_j| = e_bar_j * V_j,   V_j = E[|delta| | argmax != true]  (global),
    e_{t,j} = calibrated error rate of the carrier bin at state t.

The decision rule evaluated at decode time is  e_{t,j} * V_j >= lambda * H_{t,j}
(arm "ev" in closed_loop_rate / materialize_wma_codec_reconstructions).

Outputs (written next to the old gate artifacts):
  gate/ev-vectors.npz       -- v, e_bar, u, h_edges, h_rates, lam + metadata
  gate/ev-calibration.json  -- human-readable fit report + lambda sweep

The lambda recorded here is the open-loop keep-matched operating point
(keep fraction matched to the published 0.8359); the closed-loop keep fraction
will differ and must be re-measured on the web split before any claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NUM_POSITIONS = 1024
MIN_BIN_COUNT = 1000
PUBLISHED_KEEP = 0.8359


def _labels(paths: list[str]) -> dict[str, np.ndarray]:
    rows, poss, ad, sids = [], [], [], []
    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            rows.append(np.asarray(data["state_row"], np.int64))
            poss.append(np.asarray(data["position"], np.int64))
            ad.append(np.asarray(data["abs_delta_nll_bits"], np.float64))
            sids.append(np.asarray(data["state_ids"], object)[
                np.asarray(data["state_row"], np.int64)])
    return {"row": np.concatenate(rows), "pos": np.concatenate(poss),
            "ad": np.concatenate(ad), "sid": np.concatenate(sids)}


def _cache_order(records_path: Path, split: str) -> dict[str, int]:
    """state_id -> global cache row; reproduces cache_web's ordering exactly
    (imported semantics from label_counterfactual.cache_order)."""
    records = [json.loads(line) for line in
               records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in records if r["split"] == split]
    records.sort(key=lambda r: (r["episode_id"], int(r["step"])))
    return {str(r["state_id"]): index for index, r in enumerate(records)}


def fit_factors(labels: dict, argmax: np.ndarray, codes: np.ndarray,
                posterior_row_of_label_row: np.ndarray) -> dict:
    """Per-position U/e/V from labels joined to the posterior table.

    Label ``state_row`` indexes the pairs-corpus state table; the label file
    stores ``state_ids`` for that table, and the two-hop mapping
    ``state_ids -> cache_order(records.jsonl) -> posteriors.target_indices``
    (the same join label_counterfactual used when it filled from the WM) turns
    it into a posterior row.  The per-label error indicator is then read off
    ``wm_argmax != target_codes`` at the label's flat position.
    """
    flat_arg = argmax.reshape(len(argmax), -1)
    flat_code = codes.reshape(len(codes), -1)
    rows = posterior_row_of_label_row
    if (rows < 0).any() or rows.max() >= len(flat_arg):
        raise SystemExit("label -> posterior mapping left the posterior table")
    err_at = flat_arg[rows, labels["pos"]] != flat_code[rows, labels["pos"]]

    u = np.zeros(NUM_POSITIONS); e_lab = np.zeros(NUM_POSITIONS); v = np.zeros(NUM_POSITIONS)
    coverage = np.zeros(NUM_POSITIONS, np.int64)
    ad, pos, err = labels["ad"], labels["pos"], err_at
    for j in range(NUM_POSITIONS):
        m = pos == j
        coverage[j] = int(m.sum())
        if not m.any():
            continue
        u[j] = ad[m].mean()
        e_lab[j] = err[m].mean()
        sel = m & err
        if sel.any():
            v[j] = ad[sel].mean()
    return {"u": u, "e_lab": e_lab, "v": v, "coverage": coverage}


def _h_table(stat: np.ndarray, err: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.clip(np.digitize(stat, edges) - 1, 0, len(edges) - 2)
    rates = np.full(len(edges) - 1, err.mean())
    counts = np.zeros(len(rates), np.int64)
    for b in range(len(rates)):
        sel = idx == b
        counts[b] = int(sel.sum())
        if counts[b] >= MIN_BIN_COUNT:
            rates[b] = err[sel].mean()
    return rates, counts


def _margin_table(stat: np.ndarray, err: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _h_table(stat, err, edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--posteriors", required=True)
    parser.add_argument("--gate", default="gate",
                        help="gate cache dir holding records.jsonl (the join "
                             "label_counterfactual used for the WM fill)")
    parser.add_argument("--label", default="train",
                        help="records split the labels were cached under")
    parser.add_argument("--gate-dir", default="gate")
    parser.add_argument("--keep", type=float, default=PUBLISHED_KEEP,
                        help="open-loop keep fraction the recorded lambda matches")
    args = parser.parse_args()

    with np.load(args.posteriors, allow_pickle=True) as data:
        entropy = np.asarray(data["entropy_bits"], np.float64).reshape(len(data["entropy_bits"]), -1)
        argmax = np.asarray(data["wm_argmax"], np.uint8)
        codes = np.asarray(data["target_codes"], np.uint8)
        top1 = np.asarray(data["top1_logprob_bits"], np.float64).reshape(entropy.shape)
    labels = _labels(args.labels)
    print(f"[ev] {len(labels['ad'])} labels over "
          f"{len(np.unique(labels['sid']))} states, {NUM_POSITIONS} positions")

    # label row -> posterior row: state_ids -> cache order -> target_indices,
    # the same two-hop join label_counterfactual used for the WM fill.
    order = _cache_order(Path(args.gate) / "records.jsonl", args.label)
    with np.load(args.posteriors, allow_pickle=True) as data:
        targets = np.asarray(data["target_indices"], np.int64)
    row_of_global = {int(g): r for r, g in enumerate(targets)}
    global_rows = np.array([order.get(str(s), -1) for s in labels["sid"]], np.int64)
    post_rows = np.array([row_of_global.get(int(g), -1) for g in global_rows], np.int64)
    if (post_rows < 0).any():
        raise SystemExit(
            f"{int((post_rows < 0).sum())} labels have no posterior row; the "
            f"WM fill joined all of them, so the mapping must too")
    print(f"[ev] label -> posterior join: all rows mapped")

    factors = fit_factors(labels, argmax, codes, post_rows)
    u, v, e_lab = factors["u"], factors["v"], factors["e_lab"]
    covered = factors["coverage"] > 0
    print(f"[ev] coverage: min {factors['coverage'].min()} / median "
          f"{int(np.median(factors['coverage']))} per position, "
          f"{NUM_POSITIONS - covered.sum()} positions without any label")
    if not covered.all():
        print("[ev] WARNING: uncovered positions get V=0 -> dropped at every lambda")

    # Exactness of the factorization: |U_j| = e_bar_j * V_j up to the labels'
    # own error-rate estimate (the two estimates of e agree per position).
    prod = e_lab * v
    ok = u > 0
    corr = float(np.corrcoef(np.log(u[ok] + 1e-12), np.log(prod[ok] + 1e-12))[0, 1])
    factor_corr = float(np.corrcoef(np.log(e_lab[ok] + 1e-12), np.log(v[ok] + 1e-12))[0, 1])
    print(f"[ev] corr(log U, log e*V) = {corr:.6f} (1.0 == exact), "
          f"corr(log e, log V) = {factor_corr:.4f}")

    # Calibration on every (state, position) pair of the posterior table.
    err_all = (argmax.reshape(len(argmax), -1) != codes.reshape(len(codes), -1)).ravel()
    global_rate = float(err_all.mean())
    h_edges = np.concatenate([np.arange(0.0, 6.0, 0.25), [np.inf]])
    h_rates, h_counts = _h_table(entropy.ravel(), err_all, h_edges)
    print(f"[ev] global error rate {global_rate:.4f}; H-table rates "
          f"{np.round(h_rates, 3).tolist()}")

    # Backup carrier: p_max = 2**(-top1_logprob_bits), uneven bins as in LME.
    pmax = np.power(2.0, -top1.ravel())
    p_edges = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0000001])
    p_rates, p_counts = _margin_table(pmax, err_all, p_edges)
    e_h = h_rates[np.clip(np.digitize(entropy.ravel(), h_edges) - 1, 0, len(h_rates) - 1)]
    e_p = p_rates[np.clip(np.digitize(pmax, p_edges) - 1, 0, len(p_rates) - 1)]
    carrier_corr = float(np.corrcoef(e_h, e_p)[0, 1])
    print(f"[ev] corr(e_H, e_pmax) over pairs = {carrier_corr:.4f} -> carrier H")

    # Open-loop lambda sweep on the fit corpus posteriors (in-sample; the
    # closed-loop web keep fraction must be re-measured before any claim).
    e_map = h_rates[np.clip(np.digitize(entropy, h_edges) - 1, 0, len(h_rates) - 1)]
    risk = e_map * v[None, :]
    cost = entropy
    sweep = []
    for lam in np.logspace(-6, -1, 26):
        keep = risk >= lam * cost
        sweep.append({"lambda": float(lam), "keep_fraction": float(keep.mean())})
    target = min(sweep, key=lambda s: abs(s["keep_fraction"] - args.keep))
    # Bisect to an exact keep match. keep() decreases in lambda, so a grid point
    # below the target keep needs the answer at smaller lambda and vice versa.
    if target["keep_fraction"] <= args.keep:
        lo, hi = target["lambda"] * 10 ** -0.2, target["lambda"]
    else:
        lo, hi = target["lambda"], target["lambda"] * 10 ** 0.2
    keep = target["keep_fraction"]
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        keep = float((risk >= mid * cost).mean())
        if abs(keep - args.keep) < 1e-4:
            break
        if keep > args.keep:
            lo = mid
        else:
            hi = mid
    lam = mid
    target = {"lambda": float(lam), "keep_fraction": float(keep)}
    sweep.append(target)
    print(f"[ev] operating point lambda = {lam:.6e} (open-loop keep "
          f"{target['keep_fraction']:.4f} vs published {args.keep})")

    gate_dir = Path(args.gate_dir)
    gate_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        gate_dir / "ev-vectors.npz",
        v=v, e_bar=e_lab, u=u, h_edges=h_edges, h_rates=h_rates, lam=np.float64(lam),
        metadata=json.dumps({
            "protocol": "residualmem_ev_gate_v1",
            "rule": "keep j iff e(H_{t,j}) * V_j >= lambda * H_{t,j}, "
                    "e from the frozen H calibration table",
            "carrier": "H", "lambda": lam,
            "labels": [Path(p).name for p in args.labels],
            "posteriors": str(args.posteriors),
            "coverage_min": int(factors["coverage"].min()),
            "corr_logU_vs_eV": corr, "corr_log_e_vs_V": factor_corr,
            "corr_eH_vs_epmax": carrier_corr,
            "global_error_rate": global_rate,
        }))
    (gate_dir / "ev-calibration.json").write_text(json.dumps({
        "h_edges": h_edges.tolist(), "h_rates": h_rates.tolist(),
        "h_counts": h_counts.tolist(),
        "pmax_edges": p_edges.tolist(), "pmax_rates": p_rates.tolist(),
        "global_error_rate": global_rate,
        "lambda_sweep": sweep, "operating_point": target,
    }, indent=2))
    print(f"[ev] wrote {gate_dir}/ev-vectors.npz and {gate_dir}/ev-calibration.json")


if __name__ == "__main__":
    main()
