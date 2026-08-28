#!/usr/bin/env bash
# One arm of the K=32 observation-distillation sweep.
#
#   scripts_qformer_train_arm.sh <OBS_WEIGHT> <DEVICE_ORDINAL>
#
# --qk-norm is load-bearing and is a root-cause fix, not a workaround:
# parameter-free RMS norm on q and k before the cross-attention dot product.
# Same checkpoint, same 1000 rows, only that flag differing:
#
#            non-finite    over 1e3      median      p90        p99       max
#   without   20 (2.0%)   295 (29.5%)     8.558   2.374e15  4.144e19  8.406e19
#   with       0 (0.0%)     0 (0.0%)      1.73    3.363     8.364     14.86
#
# Because it adds no parameters, a checkpoint trained with it loads cleanly into
# a module built without it and every downstream number is then computed through
# the wrong forward with nothing to raise on. The flag is therefore recorded in
# the checkpoint metadata and read back from there, never re-specified
# downstream. See QFORMER_实验手册.md §4.1.
#
# --drop-microbatches must stay off: its threshold is the p99 of a randomly
# initialised model, so it has no headroom from step 1 and corrupts the weights
# it is meant to protect. The rate-recovery policy below should be inert under
# QK-norm (K32e took zero skipped steps) but costs nothing if it is.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
W="$1"
DEV="$2"
D="${QFORMER_BRIDGE_DIR:-$ROOT/outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar}"
MODEL="${QWEN35_MODEL:-$ROOT/models/Qwen3.5-9B}"

# conda's activate.d/activate-gcc_linux-64.sh dereferences an unset SYS_SYSROOT,
# so `set -u` kills the shell here before a single python line runs. Relax it for
# the activation only; everything after still gets the unbound-variable guard.
set +u
source /mnt/data/public_tools/miniconda3/etc/profile.d/conda.sh
conda activate "${QFORMER_CONDA_ENV:-qwen-vl}"
set -u

cd "$ROOT"
export TOKENIZERS_PARALLELISM=false

exec python -u -m experiments.state_tokenizer.train_qformer_joint \
  --pairs         "$D/qformer-qa-pairs.npz" \
  --xbar-dir      "$D/wma-xbar-fitcorpus-axtree" \
  --teacher-dir   "$D/wma-teacher-fitcorpus" \
  --teacher-cache "$D/wma-observation-teacher" \
  --model         "$MODEL" \
  --output        "$D/qformer-K32e-obs${W}.pt" \
  --device "cuda:$DEV" \
  --queries 32 --qformer-layers 4 --accumulate 4 \
  --qk-norm \
  --distill-weight 0.3 --distill-teacher observation --obs-weight "$W" \
  --sem-weight 1.0 --sem-mode same-session --sem-batch 4 --sem-extra-negatives 0 \
  --learning-rate 1e-4 --clip-norm 5.0 --seed 35 \
  --no-drop-microbatches \
  --lr-recover-steps 10 --min-lr-fraction 0.25 --max-skipped-steps 500 \
  --held-out-probe P4 --probe-observations 24 \
  --max-steps 9000 --eval-every 250 --validation-observations 96 --patience-evals 8
