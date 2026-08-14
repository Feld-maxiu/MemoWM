# v8 discrete World Model implementation status

Implemented and executed through the preregistered M1 gate on 2026-08-12:

- Immutable code/valid cache with provenance hashes, strict successor joins,
  canonical action tensors, test lock, and deterministic nested subsets.
- Task-conditioned Jeffreys marginal/copy-residual/source-conditioned baselines
  with per-transition and per-episode ideal codelengths.
- Pure-JAX/Optax block-causal Transformer with tied per-position code tables,
  byte-GRU payload encoder, five same-parameter variants, FP64 host metric sums,
  best/last resumable checkpoints, and explicit action/task cost auditing.
- Paired episode bootstrap, closed-loop rollout, decoded frozen-probe stores,
  optional predicted-prefix curriculum, hard final test freeze, and the three
  required figure generators.
- Final provenance guards require full-train/complete-validation runs, one shared
  resolved architecture and hyperparameter set, matching parameter shapes, and
  checkpoints whose config hashes equal the canonical frozen YAML.
- **28/28 CPU tests pass** for action alignment, transition integrity, test lock, smoothing,
  full-code accounting, causal masking, ablation isolation, gradients,
  checkpoint recovery, bootstrap behavior, side-information billing, and
  predicted-mask semantic evaluation.

## Measured results

| System | validation bits/transition |
|---|---:|
| task marginal | 10,635.3489 |
| copy-residual | 9,564.5895 |
| source-conditioned Markov | **8,990.7231** |
| neural WM full, seed 0 | 9,058.1713 |

The 128-transition memorisation gate passed: 16,466.0136 → 20.6996
bits/transition, a 99.874% NLL reduction. The formal full model then beat copy
by 506.4182 bits/transition (5.294%) but lost to the stronger
source-conditioned Markov control by 67.4483 bits/transition (0.750%).

Consequently M1 failed and the protocol stopped. M2 ablations, M3 scale/seeds,
M4 rollout, figures based on those runs, and test evaluation were deliberately
not executed. No test-unlock manifest exists.
The available M1-only comparison is
`outputs/world_model/v8/figures/m1/model_vs_baselines.{png,pdf}`; paired task,
policy, and source-step slices are in
`outputs/world_model/v8/diagnostics/m1_seed0.json`.

The state cache has 12 tasks, while only 10 have valid transitions:
`click-dialog-2` and `focus-text` contain only single-state episodes. WM rate
claims therefore cover the 10 transition-bearing tasks.
