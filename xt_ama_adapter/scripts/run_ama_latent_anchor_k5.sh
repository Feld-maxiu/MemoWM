#!/usr/bin/env bash
# latent+anchor arm (P1 element anchors): identical retrieval/latent rows as the
# latent-only k=5 baseline, plus one verbatim element-anchor text block built
# from the injected rows' own step_texts. Pure inference change; no retraining,
# no re-encode. Judge = official evaluate.py on GPU3-5 (port 8057).
#
# Baseline cache: dirty cache-v2 by default (direct comparison vs 0.2556).
#   CACHE_VARIANT=clean  ->  cache-v3-clean (run after the clean arm is judged)
# Anchor granularity: element (P1, default). RUN_IDS=1 also runs ids (P2).
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
TEST_FILE="$RUN_ROOT/third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl"
AMA_ROOT="$RUN_ROOT/third_party/AMA-Bench"
QWEN32=/data1/models/Qwen3-32B
QFORMER="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt"
HEAD="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt"
BRIDGE="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-top10.pt"
JUDGE_CONFIG="$RUN_ROOT/xt_ama_adapter/configs/ama_web_judge_gpu3.yaml"
CACHE_VARIANT="${CACHE_VARIANT:-dirty}"
if [[ "$CACHE_VARIANT" == "clean" ]]; then
  CACHE="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/cache-v3-clean"
  OUTROOT="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10-clean-anchor"
else
  CACHE="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/cache-v2"
  OUTROOT="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10-anchor"
fi
LOGS="$RUN_ROOT/logs/ama_latent_memory/ama_eval/latent-only-top10-anchor"
PY32="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
VLLM_PY="$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python"
PORT=8057
K=5

mkdir -p "$OUTROOT/k$K" "$LOGS"
cd "$RUN_ROOT"

wait_gpus_free() {
  local threshold="$1"; shift
  while true; do
    local busy=0
    for gpu in "$@"; do
      used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)
      [[ "$used" -ge "$threshold" ]] && busy=1
    done
    [[ "$busy" -eq 0 ]] && break
    echo "[anchor] waiting for GPU$* release $(date --iso-8601=seconds)" \
      | tee -a "$LOGS/pipeline.log"
    sleep 30
  done
}

wait_port_free() {
  while curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
    echo "[anchor] waiting for judge port ${PORT} to free $(date --iso-8601=seconds)" \
      | tee -a "$LOGS/pipeline.log"
    sleep 30
  done
}

run_arm() {
  local style="$1"
  local out="$OUTROOT/$style/k$K"
  mkdir -p "$out"
  if [[ -e "$out/ANSWERS_COMPLETE" ]]; then
    echo "[anchor] $style answers already complete; skipping" | tee -a "$LOGS/pipeline.log"
    return
  fi
  echo "[anchor] answers $style k=$K on ${CACHE_VARIANT} cache $(date --iso-8601=seconds)" \
    | tee -a "$LOGS/pipeline.log"
  CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=.:xt_ama_adapter \
    "$PY32" -u -m experiments.state_tokenizer.run_ama_web_latent answer \
      --test-file "$TEST_FILE" --cache-dir "$CACHE" \
      --qformer "$QFORMER" --retrieval-head "$HEAD" --bridge "$BRIDGE" \
      --reader-model "$QWEN32" \
      --checkpoint "$out/answer-checkpoint.jsonl" \
      --answers-output "$out/answers.jsonl" \
      --memory-mode latent+anchor --anchor-style "$style" \
      --top-k-latent "$K" --top-k 5 \
      --enable-thinking true --max-model-len 32000 --max-new-tokens 8192 \
      --batch-size 4 --max-memory-gib 34 --skip-episode-ids 184 \
      > "$LOGS/answer-$style-k$K.log" 2>&1
  touch "$out/ANSWERS_COMPLETE"
  echo "[anchor] answers $style k=$K complete $(date --iso-8601=seconds)" \
    | tee -a "$LOGS/pipeline.log"
}

judge_arm() {
  local style="$1"
  local out="$OUTROOT/$style/k$K"
  [[ -s "$out/answers.jsonl" ]] || { echo "[anchor] $style answers.jsonl missing" >&2; exit 1; }
  if [[ -e "$out/evaluation.json" ]]; then
    echo "[anchor] $style already judged" | tee -a "$LOGS/pipeline.log"
    return
  fi
  echo "[anchor] judging $style $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
  cd "$AMA_ROOT"
  PYTHONPATH="$AMA_ROOT:$RUN_ROOT" "$PY32" src/evaluate.py \
    --answers-file "$out/answers.jsonl" --test-file "$TEST_FILE" \
    --judge-config "$JUDGE_CONFIG" --judge-server vllm \
    --output-file "$out/evaluation.json" \
    > "$LOGS/judge-$style.log" 2>&1
  cd "$RUN_ROOT"
}

# element arm is the agreed first run.
wait_gpus_free 5000 4 5
run_arm element

if [[ "${RUN_IDS:-0}" == "1" ]]; then
  wait_gpus_free 5000 4 5
  run_arm ids
fi

# Shared judge server (GPU3-5); port must be free of any prior judge.
wait_port_free
wait_gpus_free 5000 3 4 5
echo "[anchor] launching judge vLLM 3x Qwen3-32B on GPU3-5 $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/pipeline.log"
CUDA_VISIBLE_DEVICES=3,4,5 VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$QWEN32" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 8192 --tensor-parallel-size 1 --data-parallel-size 3 \
    --gpu-memory-utilization 0.88 --enforce-eager \
    > "$LOGS/judge-vllm.log" 2>&1 &
server_pid=$!
cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && break
  kill -0 "$server_pid" 2>/dev/null || { echo "[anchor] judge vLLM exited" >&2; exit 1; }
  sleep 2
done

judge_arm element
if [[ "${RUN_IDS:-0}" == "1" ]]; then
  judge_arm ids
fi
echo "[anchor] complete $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
