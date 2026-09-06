#!/usr/bin/env bash
# Official AMA-Bench baselines on the same WEB subset the latent arms use
# (30 episodes / 360 questions, episode 184 excluded): bm25 / embedding /
# longcontext, official run.py + official judge, Qwen3-32B via vLLM.
# Two servers, mirroring the official two-role design:
#   - answerer: GPUs 4,5 (tp=2, max_model_len 32000 -- longcontext needs it)
#   - judge:    GPU 3   (max_model_len 8192, judge prompts are short)
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
AMA_ROOT="$RUN_ROOT/third_party/AMA-Bench"
TEST_FILE="$AMA_ROOT/dataset/test/open_end_qa_set.jsonl"
QWEN32=/data1/models/Qwen3-32B
QWEN_EMBED=/data1/models/Qwen3-Embedding-4B
OUTPUT_ROOT="$RUN_ROOT/outputs/ama_latent_memory/baselines"
LOGS="$RUN_ROOT/logs/ama_latent_memory/baselines"
PY32="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
VLLM_PY="$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python"
ANSWER_PORT=8058
JUDGE_PORT=8059
# Restrict which official methods run via RUN_METHODS (default: all three).
read -r -a METHODS <<< "${RUN_METHODS:-bm25 embedding longcontext}"

mkdir -p "$OUTPUT_ROOT" "$LOGS" "$OUTPUT_ROOT/configs"
cd "$AMA_ROOT"

# ---------------------------------------------------------------------------
# 1. Local config overrides (official configs point at non-local paths).
# ---------------------------------------------------------------------------
cat > "$OUTPUT_ROOT/configs/qwen3-32B.local.yaml" <<EOF
provider: "vllm"
model: "$QWEN32"
vllm_host: "localhost"
vllm_port: $ANSWER_PORT
timeout: 600
enable_thinking: true
vllm_launch:
  gpus: "4,5"
  max_model_len: 32000
  max_response_len: 8192
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.85
EOF
cat > "$OUTPUT_ROOT/configs/llm_judge.local.yaml" <<EOF
provider: "vllm"
model: "$QWEN32"
vllm_host: "localhost"
vllm_port: $JUDGE_PORT
timeout: 600
enable_thinking: true
vllm_launch:
  gpus: "3"
  max_model_len: 8192
  max_response_len: 4096
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.88
EOF
cat > "$OUTPUT_ROOT/configs/embedding.local.json" <<EOF
{
  "top_k": 5,
  "embedding_model": "$QWEN_EMBED",
  "use_faiss": false,
  "embedding_engine": null
}
EOF

cleanup() {
  for pid in "${answer_pid:-}" "${judge_pid:-}"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

wait_gpus_free() {
  local threshold="$1"; shift
  while true; do
    local busy=0
    for gpu in "$@"; do
      used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
      [[ "$used" -ge "$threshold" ]] && busy=1
    done
    [[ "$busy" -eq 0 ]] && break
    echo "[baselines] waiting for GPU$* release $(date --iso-8601=seconds)" \
      | tee -a "$LOGS/pipeline.log"
    sleep 30
  done
}

wait_server() {
  local port="$1" pid="$2" name="$3"
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${port}/health" >/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || {
      echo "[baselines] $name vLLM exited during startup" >&2; return 1; }
    sleep 2
  done
  echo "[baselines] $name vLLM startup timeout" >&2
  return 1
}

# ---------------------------------------------------------------------------
# 2. Answerer server (GPUs 4,5, tp=2, 32000 context).
# ---------------------------------------------------------------------------
answer_pid=""
if ! curl -sf "http://127.0.0.1:${ANSWER_PORT}/health" >/dev/null; then
  wait_gpus_free 5000 4 5
  echo "[baselines] launching answer vLLM Qwen3-32B tp2 on GPU4-5 $(date --iso-8601=seconds)" \
    | tee -a "$LOGS/pipeline.log"
  CUDA_VISIBLE_DEVICES=4,5 VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --model "$QWEN32" --host 127.0.0.1 --port "$ANSWER_PORT" \
      --max-model-len 32000 --tensor-parallel-size 2 \
      --gpu-memory-utilization 0.85 --enforce-eager \
      > "$LOGS/vllm-answer.log" 2>&1 &
  answer_pid=$!
  wait_server "$ANSWER_PORT" "$answer_pid" "answer"
fi

# ---------------------------------------------------------------------------
# 3. Judge server (GPU 3, 8192 context).
# ---------------------------------------------------------------------------
judge_pid=""
if ! curl -sf "http://127.0.0.1:${JUDGE_PORT}/health" >/dev/null; then
  wait_gpus_free 5000 3
  echo "[baselines] launching judge vLLM Qwen3-32B on GPU3 $(date --iso-8601=seconds)" \
    | tee -a "$LOGS/pipeline.log"
  CUDA_VISIBLE_DEVICES=3 VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
      --model "$QWEN32" --host 127.0.0.1 --port "$JUDGE_PORT" \
      --max-model-len 8192 --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.88 --enforce-eager \
      > "$LOGS/vllm-judge.log" 2>&1 &
  judge_pid=$!
  wait_server "$JUDGE_PORT" "$judge_pid" "judge"
fi

# ---------------------------------------------------------------------------
# 4. WEB episode ids (episode 184 excluded, matching the latent arms).
# ---------------------------------------------------------------------------
EPISODE_IDS=$("$PY32" - "$TEST_FILE" <<'EOF'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
ids = sorted(int(row["episode_id"]) for row in rows
             if str(row.get("domain", "")).upper() == "WEB"
             and int(row["episode_id"]) != 184)
print(",".join(map(str, ids)))
EOF
)
echo "[baselines] ${METHODS[*]} on episode ids: $EPISODE_IDS" | tee -a "$LOGS/pipeline.log"

# ---------------------------------------------------------------------------
# 5. Official baselines.
# ---------------------------------------------------------------------------
for method in "${METHODS[@]}"; do
  output="$OUTPUT_ROOT/$method"
  mkdir -p "$output"
  if [[ -e "$output/EVALUATION_COMPLETE" ]]; then
    echo "[baselines] $method already complete; skipping"
    continue
  fi
  extra=()
  if [[ "$method" == "embedding" ]]; then
    extra=(--method-config "$OUTPUT_ROOT/configs/embedding.local.json")
  fi
  echo "[baselines] running $method $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
  PYTHONPATH="$AMA_ROOT:$RUN_ROOT" "$PY32" src/run.py \
    --llm-server vllm \
    --llm-config "$OUTPUT_ROOT/configs/qwen3-32B.local.yaml" \
    --subset openend --method "$method" \
    --test-file "$TEST_FILE" \
    --output-dir "$output" \
    --episode-ids "$EPISODE_IDS" \
    --judge-config "$OUTPUT_ROOT/configs/llm_judge.local.yaml" \
    --judge-server vllm --evaluate true \
    "${extra[@]}" \
    > "$LOGS/run-$method.log" 2>&1
  touch "$output/EVALUATION_COMPLETE"
done
touch "$OUTPUT_ROOT/BASELINES_COMPLETE"
echo "[baselines] all methods complete $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
