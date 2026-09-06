#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
TEST_FILE="$RUN_ROOT/third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl"
QWEN32=/data1/models/Qwen3-32B
QFORMER="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt"
HEAD="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt"
BRIDGE="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-delta-margin-total1000.pt"
CACHE="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/web-latent-formal/cache"
OUTPUT="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/web-latent-delta-margin-total1000"
LOGS="$RUN_ROOT/logs/ama_latent_memory/ama_eval/web-latent-delta-margin-total1000"
PY32="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
FAILED="$OUTPUT/ANSWERS_FAILED"

mkdir -p "$OUTPUT" "$LOGS"
cd "$RUN_ROOT"
rm -f "$OUTPUT/ANSWERS_COMPLETE" "$FAILED"
echo "[ama-web-delta1000] answer generation start $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/pipeline.log"
sha256sum "$BRIDGE" | tee -a "$LOGS/pipeline.log"

if ! CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=.:xt_ama_adapter \
  "$PY32" -u -m experiments.state_tokenizer.run_ama_web_latent answer \
    --test-file "$TEST_FILE" --cache-dir "$CACHE" \
    --qformer "$QFORMER" --retrieval-head "$HEAD" --bridge "$BRIDGE" \
    --reader-model "$QWEN32" \
    --checkpoint "$OUTPUT/answer-checkpoint.jsonl" \
    --answers-output "$OUTPUT/answers.jsonl" \
    --top-k 1 --batch-size 4 --max-new-tokens 1024 --max-memory-gib 34 \
    --memory-mode matched --skip-episode-ids 184 \
    > "$LOGS/answers.log" 2>&1; then
  touch "$FAILED"
  echo "[ama-web-delta1000] answer generation failed $(date --iso-8601=seconds)" \
    | tee -a "$LOGS/pipeline.log"
  exit 1
fi

touch "$OUTPUT/ANSWERS_COMPLETE"
echo "[ama-web-delta1000] all 360 answers complete $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/pipeline.log"
