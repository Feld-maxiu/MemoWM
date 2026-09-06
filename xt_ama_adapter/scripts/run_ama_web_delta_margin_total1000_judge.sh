#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
AMA_ROOT="$RUN_ROOT/third_party/AMA-Bench"
OUTPUT="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/web-latent-delta-margin-total1000"
LOGS="$RUN_ROOT/logs/ama_latent_memory/ama_eval/web-latent-delta-margin-total1000"
CONFIG="$RUN_ROOT/xt_ama_adapter/configs/ama_web_judge_delta1000.yaml"
VLLM_PY="$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python"
CLIENT_PY="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
PORT=8058
FAILED="$OUTPUT/JUDGE_FAILED"
mkdir -p "$OUTPUT" "$LOGS"

cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

rm -f "$OUTPUT/JUDGE_COMPLETE" "$FAILED"
echo "[ama-judge-delta1000] waiting for complete answers $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/judge-pipeline.log"
while [[ ! -e "$OUTPUT/ANSWERS_COMPLETE" ]]; do
  if [[ -e "$OUTPUT/ANSWERS_FAILED" ]]; then
    touch "$FAILED"
    echo "[ama-judge-delta1000] answer generation failed" \
      | tee -a "$LOGS/judge-pipeline.log"
    exit 1
  fi
  sleep 30
done

while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [[ "${used[0]}" -lt 15000 && "${used[1]}" -lt 15000 && "${used[2]}" -lt 15000 ]]; then
    break
  fi
  echo "[ama-judge-delta1000] waiting for GPU0-2: ${used[0]},${used[1]},${used[2]} MiB $(date --iso-8601=seconds)" \
    | tee -a "$LOGS/judge-pipeline.log"
  sleep 30
done

echo "[ama-judge-delta1000] launching 3x Qwen3-32B on GPU0,1,2 $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/judge-pipeline.log"
CUDA_VISIBLE_DEVICES=0,1,2 VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model /data1/models/Qwen3-32B --host 127.0.0.1 --port "$PORT" \
    --max-model-len 8192 --tensor-parallel-size 1 --data-parallel-size 3 \
    --gpu-memory-utilization 0.88 --enforce-eager \
    > "$LOGS/judge-vllm.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    touch "$FAILED"
    echo "[ama-judge-delta1000] vLLM exited during startup" \
      | tee -a "$LOGS/judge-pipeline.log"
    exit 1
  fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then
  touch "$FAILED"
  echo "[ama-judge-delta1000] vLLM startup timeout" \
    | tee -a "$LOGS/judge-pipeline.log"
  exit 1
fi

echo "[ama-judge-delta1000] official evaluation start $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/judge-pipeline.log"
cd "$AMA_ROOT"
if ! PYTHONPATH="$AMA_ROOT:$RUN_ROOT" "$CLIENT_PY" src/evaluate.py \
  --answers-file "$OUTPUT/answers.jsonl" \
  --test-file "$AMA_ROOT/dataset/test/open_end_qa_set.jsonl" \
  --judge-config "$CONFIG" --judge-server vllm \
  --output-file "$OUTPUT/evaluation.json" \
  > "$LOGS/judge-evaluate.log" 2>&1; then
  touch "$FAILED"
  echo "[ama-judge-delta1000] evaluation failed $(date --iso-8601=seconds)" \
    | tee -a "$LOGS/judge-pipeline.log"
  exit 1
fi
touch "$OUTPUT/JUDGE_COMPLETE"
echo "[ama-judge-delta1000] official evaluation complete $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/judge-pipeline.log"
