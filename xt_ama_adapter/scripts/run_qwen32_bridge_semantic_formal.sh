#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
DATASET="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge/syqa-matched-k32.npz"
MODEL=/data1/models/Qwen3-32B
OUTPUT_DIR="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge"
TEACHER="$OUTPUT_DIR/qwen32-teacher-top128-hidden16-32-48-b4"
BRIDGE="$OUTPUT_DIR/qwen32-input-k32-rms-semantic.pt"
LOG_DIR="$RUN_ROOT/logs/ama_latent_memory/syqa/post-qformer"
PY="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$RUN_ROOT"

echo "[semantic-bridge] hidden teacher cache start $(date --iso-8601=seconds)" \
  | tee -a "$LOG_DIR/qwen32-bridge-semantic-pipeline.log"
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=.:xt_ama_adapter \
  "$PY" -u -m experiments.state_tokenizer.build_qwen32_teacher_cache \
    --dataset "$DATASET" --model "$MODEL" --output "$TEACHER" \
    --top-k 128 --hidden-layers 16 32 48 --batch-size 4 \
    --max-memory-gib 34 \
    > "$LOG_DIR/qwen32-teacher-hidden-cache.log" 2>&1

echo "[semantic-bridge] hidden teacher cache complete; training start $(date --iso-8601=seconds)" \
  | tee -a "$LOG_DIR/qwen32-bridge-semantic-pipeline.log"
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=.:xt_ama_adapter \
  "$PY" -u -m experiments.state_tokenizer.train_qwen32_bridge \
    --dataset "$DATASET" --teacher-cache "$TEACHER" \
    --model "$MODEL" --output "$BRIDGE" \
    --max-steps 3000 --eval-every 250 --eval-limit 0 \
    --micro-batch-size 4 --accumulate 2 \
    --learning-rate 1e-4 --weight-decay 1e-4 --clip-norm 5.0 \
    --distill-weight 0.30 \
    --hidden-distill-weight 0.10 --hidden-layers 16 32 48 \
    --semantic-reconstruction-weight 0.05 \
    --semantic-relation-weight 0.05 \
    --semantic-warmup-steps 200 \
    --semantic-decoder-dropout 0.10 --semantic-noise-ratio 0.01 \
    --rms-calibrate-to-reader --no-gradient-checkpointing \
    --max-memory-gib 34 --seed 35 \
    > "$LOG_DIR/qwen32-bridge-semantic-formal.log" 2>&1

touch "$OUTPUT_DIR/SEMANTIC_BRIDGE_COMPLETE"
echo "[semantic-bridge] formal training complete $(date --iso-8601=seconds)" \
  | tee -a "$LOG_DIR/qwen32-bridge-semantic-pipeline.log"
