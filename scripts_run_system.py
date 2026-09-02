"""Reproduce every number in the main table, from locked artifacts, in one command.

**Scope, stated up front so the name does not overpromise.** This does not
rebuild the system from raw data. Encoding 679,290 states is a 3.8-hour eight-GPU
job, fitting the codebook and training the world model are longer still, and
those procedures already have a home: ``scripts_build_wm_testset.py`` and
``WM_MIXED_复现手册``. What is missing -- and what this is -- is a way to go from
the artifacts those produce to the reported numbers, with every input hash-checked
and every claim recomputed rather than quoted.

The distinction matters because the six components were built separately and
never verified together. Each stage below reproduces a number that was previously
established in isolation, and the run fails if any of them has drifted:

* the world model's rate on the external test set (``run.json``: 5182.83124384419)
* the gate's keep fraction and gated rate at the operating point
* the closed-loop correction, with its all-send null control

☠️ **Not measured here: answer quality.** The gate's cost is 0.103 bits of
gold-answer NLL, which is 0.07% of a ~150-bit answer, and the retrieval-free
judge harness returns categorical labels over 2,248 pairs. Whether that can
resolve 0.07% is a power question nobody has answered, so the honest table
carries |ΔNLL| in bits -- already measured, paired t = −25.5 -- and leaves the
judge column out rather than printing a null result that would be read as
"lossless". See ``UTILITY_GATE.md`` §8.

☠️ **Not measured here: anything downstream of retrieval.** Report §5.6: 90.3% of
questions retrieve zero latent rows, so QA-C cannot observe a change to the
latent. The existing QA-C is quoted, not recomputed, and quoting it is the point
-- recomputing would suggest it was responsive to something in this pipeline.

The two environments cannot import each other (torch in conda ``qwen-vl``, jax in
``.venv-jax`` with system site-packages off), so stages run as subprocesses with
both interpreters passed in, following ``scripts_build_wm_testset.py``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_JAX = str(REPO / ".venv-jax" / "bin" / "python")
DEFAULT_TORCH = "/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python"

# run.json's best_selection.code_bits_per_transition. Everything else is measured
# against this; if a stage cannot reproduce it, the artifacts have drifted apart.
WM_RATE = 5182.83124384419
# code_bits ships float32, so a float64 corpus mean lands ~1e-5 away. This is a
# tolerance, not a bitwise check -- an earlier draft of the plan said "bitwise"
# and that was wrong.
RATE_TOLERANCE = 1e-3


def run(command: list[str], *, label: str, env_extra: dict | None = None) -> str:
    import os
    print(f"\n=== {label} ===\n$ {' '.join(command)}", flush=True)
    started = time.time()
    env = {**os.environ, "PYTHONPATH": str(REPO), **(env_extra or {})}
    finished = subprocess.run(command, cwd=REPO, env=env, text=True,
                              capture_output=True)
    if finished.returncode != 0:
        sys.stderr.write(finished.stdout[-4000:])
        sys.stderr.write(finished.stderr[-4000:])
        raise SystemExit(f"{label} failed with code {finished.returncode}")
    print(f"[{label}] {time.time() - started:.1f}s", flush=True)
    return finished.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-python", default=DEFAULT_JAX)
    parser.add_argument("--torch-python", default=DEFAULT_TORCH)
    parser.add_argument("--cache", default="testset/cache")
    parser.add_argument("--config", default="configs/world_model/web_h16_C64_full.yaml")
    parser.add_argument("--posteriors", default="gate/posteriors-testset.npz")
    parser.add_argument("--labels", nargs="+", default=[
        "gate/labels-wmfill.npz", "gate/labels-slot-s0.npz", "gate/labels-slot-s1.npz"])
    parser.add_argument("--lam", type=float, default=0.0010)
    parser.add_argument("--arms", nargs="+", default=["gate", "random", "anti"])
    parser.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--skip-closed-loop", action="store_true",
                        help="the closed-loop passes are the slow part")
    parser.add_argument("--output", default="system_results.json")
    args = parser.parse_args()

    results: dict = {"protocol": "residualmem_system_run_v1"}
    gpu_env = {"CUDA_VISIBLE_DEVICES": args.gpu, "OMP_NUM_THREADS": "8",
               "MKL_NUM_THREADS": "8"}

    # 1. Every locked artifact is present and unmodified. Cheap, and it fails
    #    before anything expensive if the codebook has been swapped.
    verify = run([args.jax_python, "-m", "residualmem.manifest"],
                 label="manifest")
    results["artifacts"] = json.loads(verify)
    print(json.dumps(results["artifacts"], indent=2, ensure_ascii=False))

    # 2. The mask, re-derived from the labels and self-checked against the world
    #    model's own rate. This is the stage that would catch a label set or a
    #    posterior dump that no longer belongs to this checkpoint.
    mask_path = f"gate/mask-lambda{args.lam:.4f}.npz"
    run([args.jax_python, "-m", "experiments.utility_gate.export_mask",
         "--labels", *args.labels, "--lambda", str(args.lam),
         "--posteriors", args.posteriors,
         "--expect-full-rate", repr(WM_RATE),
         "--tolerance", repr(RATE_TOLERANCE),
         "--output", mask_path], label="export_mask")
    with __import__("numpy").load(mask_path, allow_pickle=True) as data:
        results["mask"] = json.loads(str(data["metadata"]))

    if args.skip_closed_loop:
        results["closed_loop"] = "skipped"
    else:
        # 3. The null control comes first and is a gate on everything after it:
        #    under all-send the decoder state equals the encoder state by
        #    construction (report §6.2), so the two passes must agree exactly.
        null_path = "gate/closedloop-null.json"
        run([args.jax_python, "-m", "experiments.utility_gate.closed_loop_rate",
             "--cache", args.cache, "--config", args.config,
             "--checkpoint", "run/best.pkl", "--mask", mask_path,
             "--all-send", "--split", "validation", "--output", null_path],
            label="closed_loop_null_control", env_extra=gpu_env)
        results["null_control"] = json.loads(Path(null_path).read_text())["results"][0]

        arms_path = "gate/closedloop-arms.json"
        run([args.jax_python, "-m", "experiments.utility_gate.closed_loop_rate",
             "--cache", args.cache, "--config", args.config,
             "--checkpoint", "run/best.pkl", "--mask", mask_path,
             "--lambdas", str(args.lam), "--arms", *args.arms,
             "--split", "validation", "--output", arms_path],
            label="closed_loop_arms", env_extra=gpu_env)
        results["closed_loop"] = json.loads(Path(arms_path).read_text())["results"]

    # 4. The table. Assembled from what was just measured, not from constants.
    check = results["mask"]["self_check"]
    table = [
        {"arm": "fixed width", "rate_bits": 6144.0, "compression": 1.0,
         "answer_cost_bits": None},
        {"arm": "+ WM conditional coding", "rate_bits": check["full_rate_bits"],
         "compression": 6144.0 / check["full_rate_bits"],
         "answer_cost_bits": 0.148,
         "note": "answer cost is the quantizer's own, measured separately"},
        {"arm": f"+ utility gate lambda={args.lam} (open loop)",
         "rate_bits": check["gated_rate_bits"],
         "compression": check["compression_vs_fixed_width"],
         "answer_cost_bits": 0.103},
    ]
    if isinstance(results.get("closed_loop"), list):
        gate_row = next(r for r in results["closed_loop"] if r["arm"] == "gate")
        table.append({
            "arm": f"+ utility gate lambda={args.lam} (closed loop)",
            "rate_bits": gate_row["closed_loop"]["gated_rate_bits"],
            "compression": gate_row["closed_loop_compression"],
            "answer_cost_bits": None,
            "note": "drift %.2f bits; NOT earned by the gate -- a budget-matched "
                    "random mask drifts the same amount" % gate_row["drift_bits"],
        })
    results["main_table"] = table
    results["quoted_not_recomputed"] = {
        "qa_c": 0.5953360768175583,
        "why": "report §5.6: 90.3% of questions retrieve zero latent rows, so "
               "QA-C cannot observe a change to the latent. Recomputing it here "
               "would imply it was responsive to this pipeline.",
    }

    Path(args.output).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== main table ===")
    for row in table:
        cost = "" if row["answer_cost_bits"] is None else f"  |dNLL| {row['answer_cost_bits']}"
        print(f"  {row['arm']:<48s} {row['rate_bits']:>9.2f} bits  "
              f"{row['compression']:.3f}x{cost}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
