#!/usr/bin/env bash
# Step 5: the official WorldMemArena QA eval for one Q-Former arm.
# See WMA_RESIDUAL_复现手册.md §3.4–3.7.
#
#   scripts_qformer_official_eval.sh <ARM> <CHECKPOINT_BASENAME> <DEVICE_ORDINAL>
#
# Optional paired-eval overrides:
#   RETRIEVAL_HEAD_ARM=<artifact arm>  # when output arm has a distinct name
#   OUTPUT_ARM=<result directory name>
#   QFORMER_PYTHON=/path/to/python      # skip conda activation and use this
#   ANCHOR_RECORDS=<records.jsonl>      # post-retrieval OCR exact-value anchors
#   CODEC_RECONSTRUCTIONS=<states.npz>  # closed-loop WM + utility reconstruction
#   SMOKE=1                             # first Web sample only
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
HEAD_ARM="${RETRIEVAL_HEAD_ARM:-$ARM}"
RESULT_ARM="${OUTPUT_ARM:-$ARM}"
WMA="${WMA_ROOT:-$ROOT/../WorldMemArena}"
D="${QFORMER_BRIDGE_DIR:-$ROOT/outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar}"

# Fail before burning two hours if an artifact is missing or misnamed.
test -f "$D/$CKPT"        || { echo "missing checkpoint: $D/$CKPT" >&2; exit 1; }
test -f "$D/head-$HEAD_ARM.pt" || { echo "missing head: $D/head-$HEAD_ARM.pt" >&2; exit 1; }
test -d "$WMA"            || { echo "missing WorldMemArena: $WMA" >&2; exit 1; }
if [[ -n "${ANCHOR_RECORDS:-}" ]]; then
  test -f "$ANCHOR_RECORDS" || { echo "missing anchor records: $ANCHOR_RECORDS" >&2; exit 1; }
fi
if [[ -n "${CODEC_RECONSTRUCTIONS:-}" ]]; then
  test -f "$CODEC_RECONSTRUCTIONS" || { echo "missing codec reconstructions: $CODEC_RECONSTRUCTIONS" >&2; exit 1; }
fi

if [[ -n "${QFORMER_PYTHON:-}" ]]; then
  PYTHON_BIN="$QFORMER_PYTHON"
  test -x "$PYTHON_BIN" || { echo "QFORMER_PYTHON is not executable: $PYTHON_BIN" >&2; exit 1; }
else
  # Backwards-compatible cluster path, with a portable conda discovery fallback.
  CONDA_SH="${CONDA_SH:-}"
  if [[ -z "$CONDA_SH" ]] && command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
  fi
  if [[ -z "$CONDA_SH" ]] && [[ -f /mnt/data/public_tools/miniconda3/etc/profile.d/conda.sh ]]; then
    CONDA_SH=/mnt/data/public_tools/miniconda3/etc/profile.d/conda.sh
  fi
  test -f "$CONDA_SH" || {
    echo "cannot locate conda.sh; set QFORMER_PYTHON=/path/to/python" >&2
    exit 1
  }
  # See scripts_qformer_train_arm.sh for why `set -u` is relaxed around conda.
  set +u
  source "$CONDA_SH"
  conda activate "${QFORMER_CONDA_ENV:-qwen-vl}"
  set -u
  PYTHON_BIN=python
fi

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
export RESIDUALMEM_RETRIEVAL_HEAD="$D/head-$HEAD_ARM.pt"
export RESIDUALMEM_DEVICE="cuda:$DEV"
export RESIDUALMEM_HEAD_DEVICE="cuda:$DEV"
if [[ -n "${ANCHOR_RECORDS:-}" ]]; then
  export RESIDUALMEM_WMA_ANCHOR_RECORDS="$ANCHOR_RECORDS"
else
  unset RESIDUALMEM_WMA_ANCHOR_RECORDS || true
fi
if [[ -n "${CODEC_RECONSTRUCTIONS:-}" ]]; then
  export RESIDUALMEM_CODEC_RECONSTRUCTIONS="$CODEC_RECONSTRUCTIONS"
else
  unset RESIDUALMEM_CODEC_RECONSTRUCTIONS || true
fi

EXTRA_EVAL_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  EXTRA_EVAL_ARGS+=(--smoke)
fi

echo "=== [$RESULT_ARM] official QA on cuda:$DEV  ckpt=$CKPT  head=$HEAD_ARM  workers=$WORKERS  anchors=${ANCHOR_RECORDS:-off}  codec=${CODEC_RECONSTRUCTIONS:-off}  $(date '+%F %T') ==="

"$PYTHON_BIN" -u -m eval_framework.cli \
  --dataset ./WorldMemArena --split all --subcategory agent/arena/web \
  --max-eval-workers "$WORKERS" \
  --baseline ResidualMem-Instruct-Xbar-Input-RAG \
  --output-dir "./exp_results/$RESULT_ARM" \
  "${EXTRA_EVAL_ARGS[@]}"

echo "=== [$RESULT_ARM] done  $(date '+%F %T') ==="
