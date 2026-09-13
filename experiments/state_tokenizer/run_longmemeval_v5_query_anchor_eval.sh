#!/usr/bin/env bash
# Canonical LongMemEval Web Small evaluation: v5-query-anchor.
#
# This is the configuration that produced the current best local result:
# dense contextual anchors, head-K32-webchain-v5-query-b1, six direct latent
# hits, radius-1 slices, and the legacy short non-thinking reader path.
# Keep LME_GPUS configurable because judge services on 8023/8024 are external.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LME_PYTHON="${LME_PYTHON:-/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

LME_CACHE=.lme-cache/web-small-latents-K32-v5-query-b1
LME_OUTPUT=outputs/longmemeval/eval/web-small-v5-query-anchor-repro
LME_LOGS=outputs/longmemeval/v5/logs/query-anchor-repro
LME_GPUS="${LME_GPUS:-4,5,6}"
IFS=',' read -r -a lme_gpus <<< "$LME_GPUS"
if [[ "${#lme_gpus[@]}" -lt 1 ]]; then
  echo "LME_GPUS must contain at least one GPU index." >&2
  exit 2
fi

test -d "$LME_CACHE"
mkdir -p "$LME_OUTPUT" "$LME_LOGS"
if [[ -e "$LME_OUTPUT/results.jsonl" ]]; then
  echo "Merged output already exists; refusing to rerun." >&2
  exit 1
fi

pids=()
for rank in "${!lme_gpus[@]}"; do
  "$LME_PYTHON" -u -m experiments.state_tokenizer.run_longmemeval_local \
    --data-root LongMemEval-V2/data/longmemeval-v2 \
    --cache "$LME_CACHE" \
    --model models/Qwen3.5-9B \
    --checkpoint outputs/longmemeval/qformer/qformer-K32-webchain-v5-obs0.5.gapbest.pt \
    --embedding-model models/Qwen3-VL-Embedding-8B \
    --output "$LME_OUTPUT/results-r${rank}.jsonl" \
    --device "cuda:${lme_gpus[$rank]}" --rank "$rank" --world-size "${#lme_gpus[@]}" \
    --reader-disable-thinking --max-new-tokens 512 --reader-temperature 0 \
    --top-k 6 --candidate-k 40 --no-diverse-retrieval \
    --max-hits-per-trajectory 2 --no-structured-memory \
    --radius 1 --max-observations 18 \
    --anchor --resume --evaluator-base-url "http://127.0.0.1:$((8023 + rank % 2))/v1" \
    > "$LME_LOGS/eval-r${rank}.log" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then rc=1; fi
done
if [[ "$rc" != 0 ]]; then
  echo "At least one shard failed. Inspect logs before resuming." >&2
  exit 1
fi

merge_inputs=()
for rank in "${!lme_gpus[@]}"; do
  merge_inputs+=(--input "$LME_OUTPUT/results-r${rank}.jsonl")
done
"$LME_PYTHON" -m experiments.state_tokenizer.merge_longmemeval_eval \
  --data-root LongMemEval-V2/data/longmemeval-v2 "${merge_inputs[@]}" \
  --output "$LME_OUTPUT/results.jsonl" \
  --cache "$LME_CACHE" --anchor
