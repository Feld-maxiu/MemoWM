#!/usr/bin/env bash
# H5: latent-only top-k evaluation on AMA WEB with the new rank-10 bridge.
# Runs k in {5, 8, 10}, each over all 360 questions (episode 184 skipped).
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
TEST_FILE="$RUN_ROOT/third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl"
CACHE="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/cache-v2"
QFORMER="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt"
HEAD="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt"
BRIDGE="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-top10.pt"
READER=/data1/models/Qwen3-32B
OUTROOT="$RUN_ROOT/outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10"
LOGS="$RUN_ROOT/logs/ama_latent_memory/ama_eval/latent-only-top10"
PY="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"

mkdir -p "$OUTROOT" "$LOGS"
cd "$RUN_ROOT"

run_k() {
  local k="$1"
  local out="$OUTROOT/k$k"
  mkdir -p "$out"
  if [[ -f "$out/ANSWERS_COMPLETE" ]]; then
    echo "[latent-only] k=$k already complete" | tee -a "$LOGS/pipeline.log"
    return
  fi
  echo "[latent-only] k=$k start $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
  CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=.:xt_ama_adapter \
    "$PY" -u -m experiments.state_tokenizer.run_ama_web_latent answer \
      --test-file "$TEST_FILE" --cache-dir "$CACHE" \
      --qformer "$QFORMER" --retrieval-head "$HEAD" --bridge "$BRIDGE" \
      --reader-model "$READER" \
      --checkpoint "$out/answer-checkpoint.jsonl" \
      --answers-output "$out/answers.jsonl" \
      --memory-mode latent-only --top-k-latent "$k" --top-k 5 \
      --enable-thinking true --max-model-len 32000 --max-new-tokens 8192 \
      --batch-size 4 --max-memory-gib 34 --skip-episode-ids 184 \
      > "$LOGS/answer-k$k.log" 2>&1
  touch "$out/ANSWERS_COMPLETE"
  echo "[latent-only] k=$k complete $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
}

while true; do
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [[ "${used[4]}" -lt 5000 && "${used[5]}" -lt 5000 ]]; then
    break
  fi
  echo "[latent-only] waiting for GPU4-5 release: ${used[4]},${used[5]} MiB $(date --iso-8601=seconds)" \
    | tee -a "$LOGS/pipeline.log"
  sleep 30
done

run_k 5

# k=8 and k=10 are opt-in: run them (they resume on the checkpoint) only after
# the user has reviewed the k=5 result.  The main comparison arm is k=5.
if [[ "${RUN_K8_K10:-0}" == "1" ]]; then
  run_k 8
  run_k 10
  echo "[latent-only] all three k done $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
else
  echo "[latent-only] k=5 done; pausing before k=8/10 (set RUN_K8_K10=1 to continue) $(date --iso-8601=seconds)" | tee -a "$LOGS/pipeline.log"
fi
