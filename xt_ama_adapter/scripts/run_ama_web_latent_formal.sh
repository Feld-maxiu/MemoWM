#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
TEST_FILE="$RUN_ROOT/third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl"
QWEN35=/data1/models/Qwen3.5-9B
QWEN32=/data1/models/Qwen3-32B
QUERY_MODEL=/data1/models/Qwen3-VL-Embedding-8B
QFORMER="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt"
HEAD="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt"
BRIDGE="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms.pt"
OUTPUT="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/web-latent-formal"
CACHE="$OUTPUT/cache"
LOGS="$RUN_ROOT/logs/ama_latent_memory/ama_eval/web-latent-formal"
PY9="$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python"
PY32="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"

mkdir -p "$OUTPUT" "$CACHE" "$LOGS"
cd "$RUN_ROOT"
echo "[ama-web] formal start $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"

while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [[ "${used[4]}" -lt 5000 && "${used[5]}" -lt 5000 ]]; then
    break
  fi
  echo "[ama-web] waiting for GPU4-5 release: ${used[4]},${used[5]} MiB $(date --iso-8601=seconds)" \
    | tee -a "$LOGS/pipeline.log"
  sleep 30
done
echo "[ama-web] GPU4-5 available; resuming with episode 184 skipped $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/pipeline.log"

cache_shard() {
  local gpu="$1"
  local shard="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=.:xt_ama_adapter \
    "$PY9" -u -m experiments.state_tokenizer.run_ama_web_latent cache-shard \
      --test-file "$TEST_FILE" --residualmem-root "$RUN_ROOT" \
      --qwen35-model "$QWEN35" --qformer "$QFORMER" --retrieval-head "$HEAD" \
      --query-model "$QUERY_MODEL" --output-dir "$CACHE" --device cuda:0 \
      --shard-index "$shard" --shard-count 2 \
      --skip-episode-ids 184 \
      > "$LOGS/cache-resume-shard${shard}.log" 2>&1
}

cache_shard 4 0 & pid0=$!
cache_shard 5 1 & pid1=$!
wait "$pid0"
wait "$pid1"

cache_count=$(find "$CACHE" -maxdepth 1 -type f -name 'episode-*.pt' | wc -l)
if [[ "$cache_count" -ne 30 ]]; then
  echo "[ama-web] expected 30 caches after skipping episode 184, found $cache_count" \
    | tee -a "$LOGS/pipeline.log" >&2
  exit 1
fi
echo "[ama-web] all 30 trajectory caches complete; episode 184 skipped $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/pipeline.log"

CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=.:xt_ama_adapter \
  "$PY32" -u -m experiments.state_tokenizer.run_ama_web_latent answer \
    --test-file "$TEST_FILE" --cache-dir "$CACHE" \
    --qformer "$QFORMER" --retrieval-head "$HEAD" --bridge "$BRIDGE" \
    --reader-model "$QWEN32" \
    --checkpoint "$OUTPUT/answer-checkpoint.jsonl" \
    --answers-output "$OUTPUT/answers.jsonl" \
    --top-k 1 --batch-size 4 --max-new-tokens 1024 --max-memory-gib 34 \
    --memory-mode matched --skip-episode-ids 184 \
    > "$LOGS/answers.log" 2>&1

touch "$OUTPUT/ANSWERS_COMPLETE"
echo "[ama-web] all 360 answers complete; episode 184 skipped $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/pipeline.log"
