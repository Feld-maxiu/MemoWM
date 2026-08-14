# ResidualMem v8 World Model Experiment Tracker

| Run ID | Milestone | Purpose | Variant | Split | Priority | Status | Artifact / note |
|---|---|---|---|---|---|---|---|
| M0-CACHE | M0 | Freeze all v8 codes and strict transitions | — | all | MUST | DONE | `outputs/world_model/v8/cache/manifest.json`; counts and hashes pass |
| M0-SUBSET | M0 | Nested episode-complete train subsets | — | train | MUST | DONE | 10,015 ⊂ 30,015 ⊂ 39,365 transitions |
| M0-BASE | M0 | Train-fit held-out baselines | marginal/copy/source | validation | MUST | DONE | 10,635.35 / 9,564.59 / 8,990.72 bits/transition |
| M1-SMOKE | M1 | One-update end-to-end smoke | full seed 0 | train-4 | MUST | DONE | Training/eval/checkpoint pipeline runs; intentionally not a gate run |
| M1-OVERFIT | M1 | 90% NLL-drop memorisation gate | full seed 0 | train-128 | MUST | PASSED | 16,466.01 → 20.70 bits/transition (99.874% drop) |
| M1-FULL0 | M1 | First held-out point gate | full seed 0 | validation | MUST | FAILED | 9,058.17; beats copy by 506.42, loses to source by 67.45 bits/transition |
| M2-ABL0 | M2 | Five decisive variants | all seed 0 | validation | MUST | NOT RUN | Hard-stopped by M1-FULL0 as preregistered |
| M3-SEEDS | M3 | Three-seed final statistics | all seeds 0/1/2 | validation | MUST | NOT RUN | Requires M1 point gate |
| M3-SCALE | M3 | Data-size curve | selected variant | validation | MUST | NOT RUN | Requires M1 point gate |
| M4-ROLL | M4 | Closed-loop 1–7 steps | selected seeds | validation | MUST | NOT RUN | Requires frozen M2/M3 conclusion |
| M4-TEST | M4 | One-time frozen confirmation | selected seeds | test | MUST | LOCKED | No freeze manifest exists; failed validation gate forbids unlock |

## M1 decision

The neural model passes the implementation/learnability gate and beats the
registered copy-residual baseline, but it does not beat the stronger
source-conditioned Markov control. The stop is therefore a scientific outcome,
not a training failure: this architecture has not established incremental
conditional-rate evidence beyond a sparse train-fit Markov table.

The cache contains 12 state-level tasks, but only 10 have valid successor
transitions. `miniwob/click-dialog-2-v1` and `miniwob/focus-text-v1` consist of
single-state episodes under this collector, so they cannot enter any transition
rate or transition-stratified CI. This scope difference is explicit in every
reported WM claim.

Available M1-only artifacts are `outputs/world_model/v8/diagnostics/m1_seed0.json`
and `outputs/world_model/v8/figures/m1/model_vs_baselines.{png,pdf}`. The other
two preregistered main figures require stopped M2–M4 results and do not exist.

The tracker records execution state, not claimed results. Test remains closed.
