#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
AMA_ROOT="$RUN_ROOT/third_party/AMA-Bench"
OUTPUT="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/web-latent-formal"
LOGS="$RUN_ROOT/logs/ama_latent_memory/ama_eval/web-latent-formal"
CONFIG="$RUN_ROOT/xt_ama_adapter/configs/ama_web_judge_gpu3.yaml"
VLLM_PY="$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python"
CLIENT_PY="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
PORT=8057
mkdir -p "$OUTPUT" "$LOGS"

cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[ama-judge] waiting for complete answers $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/judge-pipeline.log"
while [[ ! -e "$OUTPUT/ANSWERS_COMPLETE" ]]; do
  sleep 30
done

if [[ ! -s "$OUTPUT/answers.jsonl" ]]; then
  echo "[ama-judge] answers marker exists but answers.jsonl is missing or empty" \
    | tee -a "$LOGS/judge-pipeline.log" >&2
  exit 1
fi

echo "[ama-judge] launching 3x Qwen3-32B data parallel on GPU3,4,5 $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/judge-pipeline.log"
CUDA_VISIBLE_DEVICES=3,4,5 VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model /data1/models/Qwen3-32B --host 127.0.0.1 --port "$PORT" \
    --max-model-len 8192 --tensor-parallel-size 1 --data-parallel-size 3 \
    --gpu-memory-utilization 0.88 \
    --enforce-eager \
    > "$LOGS/judge-vllm.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "[ama-judge] vLLM exited during startup" | tee -a "$LOGS/judge-pipeline.log" >&2
    exit 1
  fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then
  echo "[ama-judge] vLLM startup timeout" | tee -a "$LOGS/judge-pipeline.log" >&2
  exit 1
fi
echo "[ama-judge] server ready; unified official evaluation start $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/judge-pipeline.log"
cd "$AMA_ROOT"
PYTHONPATH="$AMA_ROOT:$RUN_ROOT" "$CLIENT_PY" src/evaluate.py \
  --answers-file "$OUTPUT/answers.jsonl" \
  --test-file "$AMA_ROOT/dataset/test/open_end_qa_set.jsonl" \
  --judge-config "$CONFIG" --judge-server vllm \
  --output-file "$OUTPUT/evaluation.json" \
  > "$LOGS/judge-evaluate.log" 2>&1
touch "$OUTPUT/JUDGE_COMPLETE"
echo "[ama-judge] official evaluation complete $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/judge-pipeline.log"
