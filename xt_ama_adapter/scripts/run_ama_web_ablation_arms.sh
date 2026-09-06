#!/usr/bin/env bash
# Five-arm memory ablation on the AMA WEB subset (30 episodes / 360 questions,
# episode 184 excluded): question-only / text-only / latent-only / matched /
# shuffled, all with the same local Qwen3-32B reader, bridge and caches.
# Answers are generated on GPUs 4-5, then one shared vLLM judge (GPUs 3-5)
# evaluates every arm with the official evaluate.py.
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
TEST_FILE="$RUN_ROOT/third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl"
AMA_ROOT="$RUN_ROOT/third_party/AMA-Bench"
QWEN35=/data1/models/Qwen3.5-9B
QWEN32=/data1/models/Qwen3-32B
QUERY_MODEL=/data1/models/Qwen3-VL-Embedding-8B
QFORMER="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt"
HEAD="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt"
BRIDGE="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms.pt"
JUDGE_CONFIG="$RUN_ROOT/xt_ama_adapter/configs/ama_web_judge_gpu3.yaml"
OUTPUT_ROOT="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms"
CACHE="$OUTPUT_ROOT/cache-v2"
LOGS="$RUN_ROOT/logs/ama_latent_memory/ama_eval/ablation-arms"
PY9="$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python"
PY32="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
VLLM_PY="$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python"
PORT=8057
ARMS=(question-only text-only latent-only matched shuffled)

mkdir -p "$OUTPUT_ROOT" "$CACHE" "$LOGS"
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
    echo "[ablation] waiting for GPU$* release $(date --iso-8601=seconds)" \
      | tee -a "$LOGS/pipeline.log"
    sleep 30
  done
}

# ---------------------------------------------------------------------------
# 1. Cache v2 (step texts are required by the text-bearing arms). Never
#    touches the formal v1 cache directory.
# ---------------------------------------------------------------------------
cache_ok=$("$PY32" - "$CACHE" <<'EOF'
import sys
from pathlib import Path
import torch
files = sorted(Path(sys.argv[1]).glob("episode-*.pt"))
if len(files) != 30:
    print("no"); raise SystemExit
protocol = torch.load(files[0], map_location="cpu", weights_only=True).get("protocol")
print("yes" if str(protocol).endswith("_v2") else "no")
EOF
)
if [[ "$cache_ok" != "yes" ]]; then
  wait_gpus_free 5000 4 5
  echo "[ablation] building cache v2 on GPU4-5 $(date --iso-8601=seconds)" \
    | tee -a "$LOGS/pipeline.log"
  cache_shard() {
    local gpu="$1" shard="$2"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=.:xt_ama_adapter \
      "$PY9" -u -m experiments.state_tokenizer.run_ama_web_latent cache-shard \
        --test-file "$TEST_FILE" --residualmem-root "$RUN_ROOT" \
        --qwen35-model "$QWEN35" --qformer "$QFORMER" --retrieval-head "$HEAD" \
        --query-model "$QUERY_MODEL" --output-dir "$CACHE" --device cuda:0 \
        --shard-index "$shard" --shard-count 2 --skip-episode-ids 184 \
        > "$LOGS/cache-shard${shard}.log" 2>&1
  }
  cache_shard 4 0 & pid0=$!
  cache_shard 5 1 & pid1=$!
  wait "$pid0"; wait "$pid1"
fi
cache_count=$(find "$CACHE" -maxdepth 1 -type f -name 'episode-*.pt' | wc -l)
[[ "$cache_count" -eq 30 ]] || { echo "[ablation] expected 30 v2 caches, found $cache_count" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2. Five-arm answers (Qwen3-32B reader on GPU4-5).
# ---------------------------------------------------------------------------
for arm in "${ARMS[@]}"; do
  output="$OUTPUT_ROOT/$arm"
  mkdir -p "$output"
  if [[ -e "$output/ANSWERS_COMPLETE" ]]; then
    echo "[ablation] arm $arm already complete; skipping"
    continue
  fi
  wait_gpus_free 5000 4 5
  echo "[ablation] answers: $arm $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
  CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=.:xt_ama_adapter \
    "$PY32" -u -m experiments.state_tokenizer.run_ama_web_latent answer \
      --test-file "$TEST_FILE" --cache-dir "$CACHE" \
      --qformer "$QFORMER" --retrieval-head "$HEAD" --bridge "$BRIDGE" \
      --reader-model "$QWEN32" \
      --checkpoint "$output/answer-checkpoint.jsonl" \
      --answers-output "$output/answers.jsonl" \
      --memory-mode "$arm" --top-k 5 --top-k-latent 1 \
      --enable-thinking true --max-model-len 32000 --max-new-tokens 8192 \
      --batch-size 4 --max-memory-gib 34 --skip-episode-ids 184 \
      > "$LOGS/answers-$arm.log" 2>&1
  touch "$output/ANSWERS_COMPLETE"
done

# ---------------------------------------------------------------------------
# 3. Shared judge server (GPUs 3-5) + official evaluation per arm.
# ---------------------------------------------------------------------------
cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_gpus_free 5000 3 4 5
echo "[ablation] launching judge vLLM 3x Qwen3-32B on GPU3-5 $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/pipeline.log"
CUDA_VISIBLE_DEVICES=3,4,5 VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$QWEN32" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 8192 --tensor-parallel-size 1 --data-parallel-size 3 \
    --gpu-memory-utilization 0.88 --enforce-eager \
    > "$LOGS/judge-vllm.log" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && ready=1 && break
  kill -0 "$server_pid" 2>/dev/null || {
    echo "[ablation] judge vLLM exited during startup" >&2; exit 1; }
  sleep 2
done
[[ "$ready" -eq 1 ]] || { echo "[ablation] judge vLLM startup timeout" >&2; exit 1; }

for arm in "${ARMS[@]}"; do
  output="$OUTPUT_ROOT/$arm"
  [[ -s "$output/answers.jsonl" ]] || { echo "[ablation] $arm answers.jsonl missing" >&2; exit 1; }
  echo "[ablation] judging: $arm $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
  cd "$AMA_ROOT"
  PYTHONPATH="$AMA_ROOT:$RUN_ROOT" "$PY32" src/evaluate.py \
    --answers-file "$output/answers.jsonl" \
    --test-file "$TEST_FILE" \
    --judge-config "$JUDGE_CONFIG" --judge-server vllm \
    --output-file "$output/evaluation.json" \
    > "$LOGS/judge-$arm.log" 2>&1
  cd "$RUN_ROOT"
done
touch "$OUTPUT_ROOT/JUDGE_COMPLETE"
echo "[ablation] all five arms judged $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
