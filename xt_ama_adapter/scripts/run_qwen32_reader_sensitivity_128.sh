#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
PY="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
DATASET="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge/syqa-matched-k32.npz"
BRIDGE="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-semantic.pt"
MODEL=/data1/models/Qwen3-32B
OUTPUT_DIR="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/diagnostics"
OUTPUT="$OUTPUT_DIR/qwen32-reader-sensitivity-semantic-128-v5.json"
LOG="$RUN_ROOT/logs/ama_latent_memory/syqa/post-qformer/qwen32-reader-sensitivity-semantic-128-v5.log"

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG")"
cd "$RUN_ROOT"
echo "[reader-sensitivity] start $(date --iso-8601=seconds)" >> "$LOG"

if CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=.:xt_ama_adapter \
  "$PY" -u -m experiments.state_tokenizer.diagnose_qwen32_reader_sensitivity \
    --dataset "$DATASET" \
    --bridge "$BRIDGE" \
    --model "$MODEL" \
    --output "$OUTPUT" \
    --sample-size 128 \
    --batch-size 4 \
    --random-directions 32 \
    --finite-difference-epsilon 0.01 \
    --alphas 0 1 2 4 \
    --max-memory-gib 34 \
    --seed 35 \
    >> "$LOG" 2>&1; then
  echo "[reader-sensitivity] complete $(date --iso-8601=seconds)" >> "$LOG"
else
  code=$?
  touch "$OUTPUT_DIR/qwen32-reader-sensitivity-semantic-128-v5.failed"
  echo "[reader-sensitivity] failed code=$code $(date --iso-8601=seconds)" >> "$LOG"
  exit "$code"
fi
