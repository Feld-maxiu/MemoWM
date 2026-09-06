#!/usr/bin/env bash
# Clean-AXTree arm: same latent-only k=5 evaluation as the dirty latent-only
# top10 arm (0.2556) but every observation goes through the deterministic
# R1-R4 redundancy filter (axtree_clean medium) before the frozen Q-Former
# encodes it into memory rows. All trainable components (Q-Former obs1.0-ce /
# head / rank-10 bridge / Qwen3-32B reader) are byte-identical to the dirty
# arm; only the cache input text differs (cache protocol v3-clean).
#
# Phases: (1) cache-shard clean on GPU4-5 (2 shards) -> (2) answers k=5
# latent-only on GPU4-5 -> (3) official judge on GPU3-5 (3x data parallel).
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
TEST_FILE="$RUN_ROOT/third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl"
AMA_ROOT="$RUN_ROOT/third_party/AMA-Bench"
QWEN35=/data1/models/Qwen3.5-9B
QWEN32=/data1/models/Qwen3-32B
QUERY_MODEL=/data1/models/Qwen3-VL-Embedding-8B
QFORMER="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt"
HEAD="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt"
BRIDGE="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-top10.pt"
JUDGE_CONFIG="$RUN_ROOT/xt_ama_adapter/configs/ama_web_judge_gpu3.yaml"
OUTROOT="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10-clean"
CACHE="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/cache-v3-clean"
LOGS="$RUN_ROOT/logs/ama_latent_memory/ama_eval/latent-only-top10-clean"
PY9="$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python"
PY32="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
VLLM_PY="$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python"
PORT=8057
K=5

mkdir -p "$OUTROOT/k$K" "$CACHE" "$LOGS"
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
    echo "[clean] waiting for GPU$* release $(date --iso-8601=seconds)" \
      | tee -a "$LOGS/pipeline.log"
    sleep 30
  done
}

wait_server() {
  local pid="$1"
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || { echo "[clean] judge vLLM exited" >&2; return 1; }
    sleep 2
  done
  echo "[clean] judge vLLM startup timeout" >&2
  return 1
}

# ---------------------------------------------------------------------------
# 1. Clean caches (v3-clean protocol) unless present.
# ---------------------------------------------------------------------------
cache_ok=$("$PY32" - "$CACHE" <<'EOF'
import sys
from pathlib import Path
import torch
files = sorted(Path(sys.argv[1]).glob("episode-*.pt"))
if len(files) != 30:
    print("no"); raise SystemExit
protocol = torch.load(files[0], map_location="cpu", weights_only=True).get("protocol")
meta = torch.load(files[0], map_location="cpu", weights_only=True).get("metadata") or {}
print("yes" if str(protocol).endswith("_v3-clean") and meta.get("axtree_clean") == "medium" else "no")
EOF
)
if [[ "$cache_ok" != "yes" ]]; then
  wait_gpus_free 5000 4 5
  echo "[clean] building v3-clean caches on GPU4-5 $(date --iso-8601=seconds)" \
    | tee -a "$LOGS/pipeline.log"
  cache_shard() {
    local gpu="$1" shard="$2"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=.:xt_ama_adapter \
      "$PY9" -u -m experiments.state_tokenizer.run_ama_web_latent cache-shard \
        --test-file "$TEST_FILE" --residualmem-root "$RUN_ROOT" \
        --qwen35-model "$QWEN35" --qformer "$QFORMER" --retrieval-head "$HEAD" \
        --query-model "$QUERY_MODEL" --output-dir "$CACHE" --device cuda:0 \
        --shard-index "$shard" --shard-count 2 --skip-episode-ids 184 \
        --clean-axtree medium \
        > "$LOGS/cache-shard${shard}.log" 2>&1
  }
  cache_shard 4 0 & pid0=$!
  cache_shard 5 1 & pid1=$!
  wait "$pid0"; wait "$pid1"
fi
cache_count=$(find "$CACHE" -maxdepth 1 -type f -name 'episode-*.pt' | wc -l)
[[ "$cache_count" -eq 30 ]] || { echo "[clean] expected 30 v3-clean caches, found $cache_count" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2. Answers: latent-only k=5 (Qwen3-32B reader on GPU4-5).
# ---------------------------------------------------------------------------
if [[ -e "$OUTROOT/k$K/ANSWERS_COMPLETE" ]]; then
  echo "[clean] k=$K answers already complete; skipping" | tee -a "$LOGS/pipeline.log"
else
  wait_gpus_free 5000 4 5
  echo "[clean] answers latent-only k=$K $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
  CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=.:xt_ama_adapter \
    "$PY32" -u -m experiments.state_tokenizer.run_ama_web_latent answer \
      --test-file "$TEST_FILE" --cache-dir "$CACHE" \
      --qformer "$QFORMER" --retrieval-head "$HEAD" --bridge "$BRIDGE" \
      --reader-model "$QWEN32" \
      --checkpoint "$OUTROOT/k$K/answer-checkpoint.jsonl" \
      --answers-output "$OUTROOT/k$K/answers.jsonl" \
      --memory-mode latent-only --top-k-latent "$K" --top-k 5 \
      --enable-thinking true --max-model-len 32000 --max-new-tokens 8192 \
      --batch-size 4 --max-memory-gib 34 --skip-episode-ids 184 \
      > "$LOGS/answer-k$K.log" 2>&1
  touch "$OUTROOT/k$K/ANSWERS_COMPLETE"
  echo "[clean] answers k=$K complete $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
fi

# ---------------------------------------------------------------------------
# 3. Official judge (vLLM 3x Qwen3-32B on GPU3-5).
# ---------------------------------------------------------------------------
cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ -e "$OUTROOT/k$K/evaluation.json" ]]; then
  echo "[clean] evaluation.json exists; skipping judge" | tee -a "$LOGS/pipeline.log"
  exit 0
fi
[[ -s "$OUTROOT/k$K/answers.jsonl" ]] || { echo "[clean] answers.jsonl missing" >&2; exit 1; }

wait_gpus_free 5000 3 4 5
echo "[clean] launching judge vLLM 3x Qwen3-32B on GPU3-5 $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/pipeline.log"
CUDA_VISIBLE_DEVICES=3,4,5 VLLM_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
    --model "$QWEN32" --host 127.0.0.1 --port "$PORT" \
    --max-model-len 8192 --tensor-parallel-size 1 --data-parallel-size 3 \
    --gpu-memory-utilization 0.88 --enforce-eager \
    > "$LOGS/judge-vllm.log" 2>&1 &
server_pid=$!
wait_server "$server_pid" || exit 1
echo "[clean] judge server ready; official evaluation $(date --iso-8601=seconds)" \
  | tee -a "$LOGS/pipeline.log"
cd "$AMA_ROOT"
PYTHONPATH="$AMA_ROOT:$RUN_ROOT" "$PY32" src/evaluate.py \
  --answers-file "$OUTROOT/k$K/answers.jsonl" \
  --test-file "$TEST_FILE" \
  --judge-config "$JUDGE_CONFIG" --judge-server vllm \
  --output-file "$OUTROOT/k$K/evaluation.json" \
  > "$LOGS/judge-evaluate.log" 2>&1
cd "$RUN_ROOT"
echo "[clean] k=$K judged $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
