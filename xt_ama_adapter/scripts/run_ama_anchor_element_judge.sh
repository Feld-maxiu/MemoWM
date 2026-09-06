#!/usr/bin/env bash
# Robust judge-only driver for the latent+anchor element arm (answers already
# complete).  vLLM segfaulted twice at startup for no logged reason, so this
# retries the 3x data-parallel server a few times and falls back to a single
# GPU if needed, then runs the official evaluate.py on 127.0.0.1:8057.
set -u

RUN_ROOT=/home/cbd/project/residual-mem
AMA_ROOT="$RUN_ROOT/third_party/AMA-Bench"
TEST_FILE="$AMA_ROOT/dataset/test/open_end_qa_set.jsonl"
OUT="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10-anchor/${JDIR:-element/k5}"
LOGS="$RUN_ROOT/logs/ama_latent_memory/ama_eval/latent-only-top10-anchor"
CONFIG="$RUN_ROOT/xt_ama_adapter/configs/ama_web_judge_gpu3.yaml"
VLLM_PY="$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python"
PY32="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
QWEN32=/data1/models/Qwen3-32B
PORT=8057
mkdir -p "$LOGS"

if [[ -e "$OUT/evaluation.json" ]]; then
  echo "[judge] evaluation.json already exists" | tee -a "$LOGS/judge-retry.log"
  exit 0
fi
[[ -s "$OUT/answers.jsonl" ]] || { echo "[judge] answers.jsonl missing/empty" >&2; exit 1; }
if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "[judge] port ${PORT} already serving; skip server launch" | tee -a "$LOGS/judge-retry.log"
else
  server_pid=""
  cleanup() {
    if [[ -n "${server_pid:-}" ]]; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM

  try_launch() {
    local gpus="$1" label="$2" extra="$3"
    echo "[judge] try: $label (GPUs $gpus) $(date --iso-8601=seconds)" | tee -a "$LOGS/judge-retry.log"
    : > "$LOGS/judge-vllm.log"
    CUDA_VISIBLE_DEVICES="$gpus" VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
      VLLM_USE_FLASHINFER_SAMPLER=0 \
      "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
        --model "$QWEN32" --host 127.0.0.1 --port "$PORT" \
        --max-model-len 8192 --tensor-parallel-size 1 $extra \
        --gpu-memory-utilization 0.86 --enforce-eager \
        > "$LOGS/judge-vllm.log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 240); do
      if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[judge] ready: $label" | tee -a "$LOGS/judge-retry.log"
        return 0
      fi
      if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "[judge] crashed: $label (see judge-vllm.log)" | tee -a "$LOGS/judge-retry.log"
        tail -3 "$LOGS/judge-vllm.log" 2>/dev/null | tee -a "$LOGS/judge-retry.log"
        wait "$server_pid" 2>/dev/null || true
        server_pid=""
        return 1
      fi
      sleep 2
    done
    echo "[judge] timeout: $label" | tee -a "$LOGS/judge-retry.log"
    return 1
  }

  ok=0
  for attempt in 1 2 3; do
    if try_launch "3,4,5" "3xDP attempt$attempt" "--data-parallel-size 3"; then ok=1; break; fi
    sleep 60
  done
  if [[ "$ok" -eq 0 ]]; then
    echo "[judge] 3xDP failed; falling back to single GPU" | tee -a "$LOGS/judge-retry.log"
    for g in 3 4 5 0 1 2; do
      if try_launch "$g" "single-GPU$g" ""; then ok=1; break; fi
      sleep 30
    done
  fi
  [[ "$ok" -eq 1 ]] || { echo "[judge] all attempts failed" >&2; exit 1; }
fi

echo "[judge] official evaluation start $(date --iso-8601=seconds)" | tee -a "$LOGS/judge-retry.log"
cd "$AMA_ROOT"
PYTHONPATH="$AMA_ROOT:$RUN_ROOT" "$PY32" src/evaluate.py \
  --answers-file "$OUT/answers.jsonl" --test-file "$TEST_FILE" \
  --judge-config "$CONFIG" --judge-server vllm \
  --output-file "$OUT/evaluation.json" \
  > "$LOGS/judge-evaluate.log" 2>&1
cd "$RUN_ROOT"
touch "$OUT/JUDGE_COMPLETE"
echo "[judge] evaluation complete $(date --iso-8601=seconds)" | tee -a "$LOGS/judge-retry.log"
