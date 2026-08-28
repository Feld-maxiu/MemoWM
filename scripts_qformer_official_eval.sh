#!/usr/bin/env bash
# Step 5: the official WorldMemArena QA eval for one Q-Former arm.
# See QFORMER_实验手册.md §3.4.
#
#   scripts_qformer_official_eval.sh <ARM> <CHECKPOINT_BASENAME> <DEVICE_ORDINAL>
#
# The Q-Former is injected through environment variables, NOT CLI flags
# (residualmem_instruct_adapter.py:113-120). Three traps:
#   - RESIDUALMEM_RETRIEVAL_HEAD must be set explicitly. Leaving it unset
#     silently falls back to the joint head baked into the checkpoint, which
#     measured a full point worse (0.2131 vs 0.1988).
#   - RESIDUALMEM_INPUT_CONNECTOR must NOT be set alongside RESIDUALMEM_QFORMER
#     (:199-204 raises) -- the resampler brings its own connector.
#   - --baseline must be ResidualMem-Instruct-Xbar-Input-RAG. The -A2- variant
#     errors out (a Q-Former emits no a2_xbar) and -L16- needs a different
#     connector.
#
# WORKERS defaults to 24, not the 48 the single-arm runs used: two arms at 48
# each put 96 concurrent requests on one judge server and drove 1400+ CUDA OOM
# retries. Every retry succeeded and no question was dropped, but it is why that
# pair took 2h28m. Raise it to 48 only when running a single arm.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ARM="$1"
CKPT="$2"
DEV="$3"
WORKERS="${WORKERS:-24}"
WMA="${WMA_ROOT:-$ROOT/../WorldMemArena}"
D="${QFORMER_BRIDGE_DIR:-$ROOT/outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar}"

# Fail before burning two hours if an artifact is missing or misnamed.
test -f "$D/$CKPT"        || { echo "missing checkpoint: $D/$CKPT" >&2; exit 1; }
test -f "$D/head-$ARM.pt" || { echo "missing head: $D/head-$ARM.pt" >&2; exit 1; }
test -d "$WMA"            || { echo "missing WorldMemArena: $WMA" >&2; exit 1; }

# See scripts_qformer_train_arm.sh for why `set -u` has to be relaxed here.
set +u
source /mnt/data/public_tools/miniconda3/etc/profile.d/conda.sh
conda activate "${QFORMER_CONDA_ENV:-qwen-vl}"
set -u

cd "$WMA"
export PYTHONPATH="$PWD"
export QWEN_EMBED_DEVICE="cuda:$DEV"
export QWEN_VL_EMBED_LOCAL=1
export LLM_MAX_CONCURRENT="$WORKERS"

export RESIDUALMEM_ROOT="$ROOT"
export RESIDUALMEM_QWEN35_MODEL="${QWEN35_MODEL:-$ROOT/models/Qwen3.5-9B}"
export RESIDUALMEM_QFORMER="$D/$CKPT"
export RESIDUALMEM_QFORMER_QUERIES=32
export RESIDUALMEM_QFORMER_LAYERS=4
export RESIDUALMEM_RETRIEVAL_HEAD="$D/head-$ARM.pt"
export RESIDUALMEM_DEVICE="cuda:$DEV"
export RESIDUALMEM_HEAD_DEVICE="cuda:$DEV"

echo "=== [$ARM] official QA on cuda:$DEV  ckpt=$CKPT  workers=$WORKERS  $(date '+%F %T') ==="

python -u -m eval_framework.cli \
  --dataset ./WorldMemArena --split all --subcategory agent/arena/web \
  --max-eval-workers "$WORKERS" \
  --baseline ResidualMem-Instruct-Xbar-Input-RAG \
  --output-dir "./exp_results/$ARM"

echo "=== [$ARM] done  $(date '+%F %T') ==="
