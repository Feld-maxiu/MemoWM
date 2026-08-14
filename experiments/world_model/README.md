# v8 discrete World Model evidence gate

This directory is the independent, pure-JAX experiment line for predicting the
frozen v8 A2 representation. It does not import or modify the existing RSSM.

The central accounting invariant is that A1/A2 latent indices are distributed:
all `64×32=2048` codes are charged on every transition. `valid[64]` describes
observation slots and is predicted as a separate mask; it never removes code CE.
The primary comparison is held-out ideal conditional NLL. Fixed action payload
costs are reported separately, and the 12-way task ID is audited as 4 bits once
per episode. This line does not claim a realised entropy coder.

## Current decision (2026-08-12)

M0 and M1 have run. The 128-transition overfit gate passed (99.874% NLL drop).
The formal full seed-0 model reached 9,058.1713 validation bits/transition: it
beat copy-residual (9,564.5895) but did not beat source-conditioned Markov
(8,990.7231). The preregistered M1 gate therefore failed. M2–M4 were not run,
and test remains locked. See `refine-logs/EXPERIMENT_TRACKER.md`.

The state cache contains 12 tasks, but only 10 have successor transitions.
`click-dialog-2-v1` and `focus-text-v1` contain only single-state episodes and
are excluded by construction from transition metrics.

## M0: freeze cache, subsets, and validation baselines

Run in the JAX environment from the repository root:

```sh
JX=/root/nas/users/luzheng/workspace/enter/envs/ResidualMem/bin/python3.11
export PYTHONPATH=.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

$JX -m experiments.world_model.cache \
  --records outputs/state_tokenizer/v8/full-721.jsonl \
  --features outputs/state_tokenizer/v8/static_features \
  --normalization outputs/state_tokenizer/v8/key64-static-pca-normalization.npz \
  --a1-checkpoint outputs/a1/v8.npz \
  --checkpoint outputs/a2/v8-m32-gw.npz \
  --output outputs/world_model/v8/cache

$JX -m experiments.world_model.subsets \
  --cache outputs/world_model/v8/cache \
  --output outputs/world_model/v8/subsets --targets 10000 30000

$JX -m experiments.world_model.baselines \
  --cache outputs/world_model/v8/cache \
  --output outputs/world_model/v8/baselines/validation \
  --fit-split train --eval-split validation --verify-cache-hashes
```

The baseline command rejects same-split fitting by default. Test access is also
rejected unless a matching final freeze manifest is supplied.

## M1/M2: sanity, main run, and ablations

```sh
# Required 128-transition memorisation gate.
$JX -m experiments.world_model.train \
  --cache outputs/world_model/v8/cache \
  --config configs/world_model/v8_discrete.yaml \
  --variant full --seed 0 --overfit-transitions 128 \
  --output outputs/world_model/v8/runs/overfit128_full

# First formal held-out run, with both point gates enforced.
$JX -m experiments.world_model.train \
  --cache outputs/world_model/v8/cache \
  --config configs/world_model/v8_discrete.yaml \
  --variant full --seed 0 \
  --baseline-json outputs/world_model/v8/baselines/validation/baseline.json \
  --enforce-c1-point-gate \
  --output outputs/world_model/v8/runs/full_seed0
```

Only if that command passes should the same command be run for `t_only`,
`no_action`, `structural_action`, and `no_history`. A run directory stores
best/last checkpoints, optimizer/PRNG/sampler state, resolved YAML, JSONL
metrics, and per-transition/per-episode rates.

## M3/M4: statistics, rollout, figures, and test freeze

`statistics.py` requires all five variants with exactly seeds 0/1/2. It verifies
each sibling `run.json`, automatically selects `full` only when the history-gain
CI is strictly positive (otherwise `no_history`), and then computes C1 for that
selected model. It rejects subset/smoke/overfit inputs and requires every final
run to share the same resolved architecture, hyperparameters, parameter shapes,
full train count, and complete validation selection. Its 10,000 paired bootstrap
replicates draw a seed and complete episodes within each task.

`rollout.py` recursively feeds greedy predicted code/mask states for horizons
1–7 while retaining known actions. Optional A2 decoding writes drop-in raw-PCA
stores for `semantic_rollout.py`; predicted validity is deliberately allowed to
differ from clean validity so mask failures are measured rather than rejected.

`curriculum.py` is the optional fixed 5,000-step, ratio 0.1→0.5 predicted-prefix
stage. `curriculum_decision.py` keeps it only when the three-seed horizon-3 and
horizon-7 paired CIs are positive and every one-step degradation is at most 1%.

`figures.py` emits the three preregistered PNG/PDF figures from baseline,
run/statistics, scale, and rollout JSON artifacts. `freeze.py` has no negative-
gate override: it requires passed validation gates and exactly seeds 0/1/2 of
the statistically selected variant before it can create the sole test key. Each
checkpoint must also carry the hash of the canonical resolved formal YAML for
its variant and seed.

## Tests

```sh
PYTHONPATH=. JAX_PLATFORMS=cpu $JX tests/run_tests.py tests/world_model/test_*.py
```

Current result: **28/28 passed** on CPU (2026-08-12).
