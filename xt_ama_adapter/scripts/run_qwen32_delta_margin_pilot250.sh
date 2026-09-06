#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
PY="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
DATASET="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge/syqa-matched-k32.npz"
MODEL=/data1/models/Qwen3-32B
BRIDGE_DIR="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge"
TEACHER="$BRIDGE_DIR/qwen32-teacher-top128-hidden16-32-48-b4"
QUESTION_HIDDEN="$BRIDGE_DIR/qwen32-question-only-hidden16-32-48-train2048"
INIT_BRIDGE="$BRIDGE_DIR/qwen32-input-k32-rms-semantic.pt"
OUTPUT="$BRIDGE_DIR/qwen32-input-k32-rms-delta-margin-pilot250.pt"
LOG_DIR="$RUN_ROOT/logs/ama_latent_memory/syqa/post-qformer"
PIPELINE_LOG="$LOG_DIR/qwen32-delta-margin-pilot250-pipeline.log"
QUESTION_LOG="$LOG_DIR/qwen32-question-hidden-train2048.log"
TRAIN_LOG="$LOG_DIR/qwen32-delta-margin-pilot250.log"
COMPLETE="$BRIDGE_DIR/DELTA_MARGIN_PILOT250_COMPLETE"
FAILED="$BRIDGE_DIR/DELTA_MARGIN_PILOT250_FAILED"

mkdir -p "$BRIDGE_DIR" "$LOG_DIR"
cd "$RUN_ROOT"
echo "[delta-margin-pilot] question cache start $(date --iso-8601=seconds)" >> "$PIPELINE_LOG"

if ! CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=.:xt_ama_adapter \
  "$PY" -u -m experiments.state_tokenizer.build_qwen32_question_hidden_cache \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --output "$QUESTION_HIDDEN" \
    --hidden-layers 16 32 48 \
    --batch-size 4 \
    --max-memory-gib 34 \
    --train-limit 2048 \
    --seed 35 \
    >> "$QUESTION_LOG" 2>&1; then
  touch "$FAILED"
  echo "[delta-margin-pilot] question cache failed $(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
  exit 1
fi

echo "[delta-margin-pilot] training start $(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
if ! CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=.:xt_ama_adapter \
  "$PY" -u -m experiments.state_tokenizer.train_qwen32_bridge \
    --dataset "$DATASET" \
    --teacher-cache "$TEACHER" \
    --question-hidden-cache "$QUESTION_HIDDEN" \
    --model "$MODEL" \
    --init-bridge "$INIT_BRIDGE" \
    --output "$OUTPUT" \
    --train-limit 2048 \
    --max-steps 250 \
    --eval-every 250 \
    --eval-limit 0 \
    --micro-batch-size 4 \
    --accumulate 2 \
    --learning-rate 2e-5 \
    --weight-decay 1e-4 \
    --clip-norm 5.0 \
    --distill-weight 0.30 \
    --hidden-distill-weight 0.10 \
    --hidden-delta-whiten \
    --hidden-whiten-floor-ratio 0.10 \
    --hidden-layers 16 32 48 \
    --shuffle-margin-weight 0.20 \
    --shuffle-margin 0.10 \
    --semantic-warmup-steps 25 \
    --rms-calibrate-to-reader \
    --no-gradient-checkpointing \
    --max-memory-gib 34 \
    --seed 35 \
    >> "$TRAIN_LOG" 2>&1; then
  touch "$FAILED"
  echo "[delta-margin-pilot] training failed $(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
  exit 1
fi

touch "$COMPLETE"
echo "[delta-margin-pilot] complete $(date --iso-8601=seconds)" >> "$PIPELINE_LOG"
