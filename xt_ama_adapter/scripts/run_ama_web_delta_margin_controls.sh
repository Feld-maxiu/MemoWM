#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
AMA_ROOT="$RUN_ROOT/third_party/AMA-Bench"
TEST_FILE="$AMA_ROOT/dataset/test/open_end_qa_set.jsonl"
QWEN32=/data1/models/Qwen3-32B
QFORMER="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt"
HEAD="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt"
BRIDGE="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-delta-margin-total1000.pt"
CACHE="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/web-latent-formal/cache"
ROOT_OUTPUT="$RUN_ROOT/outputs/ama_latent_memory/ama_eval"
ROOT_LOGS="$RUN_ROOT/logs/ama_latent_memory/ama_eval"
PY32="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
VLLM_PY="$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python"
CONFIG="$RUN_ROOT/xt_ama_adapter/configs/ama_web_judge_controls.yaml"
PORT=8059
MODES=(question-only shuffled)

cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$RUN_ROOT"
for mode in "${MODES[@]}"; do
  output="$ROOT_OUTPUT/web-latent-delta-margin-total1000-$mode"
  logs="$ROOT_LOGS/web-latent-delta-margin-total1000-$mode"
  mkdir -p "$output" "$logs"
  rm -f "$output/ANSWERS_COMPLETE" "$output/ANSWERS_FAILED"
  echo "[ama-control] $mode answers start $(date --iso-8601=seconds)" \
    | tee -a "$logs/pipeline.log"
  if ! CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=.:xt_ama_adapter \
    "$PY32" -u -m experiments.state_tokenizer.run_ama_web_latent answer \
      --test-file "$TEST_FILE" --cache-dir "$CACHE" \
      --qformer "$QFORMER" --retrieval-head "$HEAD" --bridge "$BRIDGE" \
      --reader-model "$QWEN32" \
      --checkpoint "$output/answer-checkpoint.jsonl" \
      --answers-output "$output/answers.jsonl" \
      --top-k 1 --batch-size 4 --max-new-tokens 1024 --max-memory-gib 34 \
      --memory-mode "$mode" --shuffle-seed 35 --skip-episode-ids 184 \
      > "$logs/answers.log" 2>&1; then
    touch "$output/ANSWERS_FAILED"
    exit 1
  fi
  touch "$output/ANSWERS_COMPLETE"
  echo "[ama-control] $mode answers complete $(date --iso-8601=seconds)" \
    | tee -a "$logs/pipeline.log"
done

echo "[ama-control] launching shared Judge GPU0-2 $(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES=0,1,2 VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$QWEN32" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 8192 --tensor-parallel-size 1 --data-parallel-size 3 \
    --gpu-memory-utilization 0.88 --enforce-eager \
    > "$ROOT_LOGS/web-latent-delta-margin-controls-judge-vllm.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    exit 1
  fi
  sleep 2
done
[[ "$ready" -eq 1 ]]

for mode in "${MODES[@]}"; do
  output="$ROOT_OUTPUT/web-latent-delta-margin-total1000-$mode"
  logs="$ROOT_LOGS/web-latent-delta-margin-total1000-$mode"
  rm -f "$output/JUDGE_COMPLETE" "$output/JUDGE_FAILED"
  echo "[ama-control] $mode Judge start $(date --iso-8601=seconds)" \
    | tee -a "$logs/judge-pipeline.log"
  if ! (cd "$AMA_ROOT" && \
    PYTHONPATH="$AMA_ROOT:$RUN_ROOT" "$PY32" src/evaluate.py \
      --answers-file "$output/answers.jsonl" --test-file "$TEST_FILE" \
      --judge-config "$CONFIG" --judge-server vllm \
      --output-file "$output/evaluation.json" \
      > "$logs/judge-evaluate.log" 2>&1); then
    touch "$output/JUDGE_FAILED"
    exit 1
  fi
  cd "$RUN_ROOT"
  touch "$output/JUDGE_COMPLETE"
  echo "[ama-control] $mode Judge complete $(date --iso-8601=seconds)" \
    | tee -a "$logs/judge-pipeline.log"
done

touch "$ROOT_OUTPUT/AMA_WEB_DELTA_MARGIN_CONTROLS_COMPLETE"
