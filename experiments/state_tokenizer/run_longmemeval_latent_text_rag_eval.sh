#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
LME_PYTHON="${LME_PYTHON:-/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false

LME_CACHE="${LME_CACHE:-.lme-cache/web-small-latents-K32-v5-query-b1}"
LME_TEXT_INDEX="${LME_TEXT_INDEX:-outputs/longmemeval/retrieval/text-index-v1-compact.npz}"
LME_TEXT_MODEL="${LME_TEXT_MODEL:-models/Qwen3-Embedding-8B}"
LME_OUTPUT="${LME_OUTPUT:-outputs/longmemeval/eval/web-small-latent-text-rag-compact-official}"
LME_LOGS="${LME_LOGS:-outputs/longmemeval/text-rag/logs-latent-compact}"
LME_GPUS="${LME_GPUS:-4,5,6,7}"
LME_TEXT_MEMORY_LAYOUT="${LME_TEXT_MEMORY_LAYOUT:-legacy-windows}"
LME_READER_SYSTEM_PROMPT="${LME_READER_SYSTEM_PROMPT:-longmemeval}"
layout_args=()
merge_anchor=--no-anchor
case "$LME_TEXT_MEMORY_LAYOUT" in
  legacy-windows) ;;
  state-snapshots)
    layout_args=(--text-memory-layout state-snapshots)
    merge_anchor=--anchor
    ;;
  *)
    printf 'Unknown LME_TEXT_MEMORY_LAYOUT: %s\n' "$LME_TEXT_MEMORY_LAYOUT" >&2
    exit 2
    ;;
esac
IFS=',' read -r -a lme_gpus <<< "$LME_GPUS"
if [[ "${#lme_gpus[@]}" -lt 1 ]]; then
  echo "LME_GPUS must contain at least one GPU index." >&2
  exit 2
fi

test -d "$LME_CACHE"
test -f "$LME_TEXT_INDEX"
test -d "$LME_TEXT_MODEL"
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
    --text-index "$LME_TEXT_INDEX" \
    --text-embedding-model "$LME_TEXT_MODEL" \
    --output "$LME_OUTPUT/results-r${rank}.jsonl" \
    --device "cuda:${lme_gpus[$rank]}" --rank "$rank" --world-size "${#lme_gpus[@]}" \
    --reader-enable-thinking --max-new-tokens 20000 \
    --reader-temperature 0.6 --reader-top-p 0.95 --reader-top-k 20 \
    --reader-system-prompt "$LME_READER_SYSTEM_PROMPT" \
    --retrieval-mode text --top-k 6 --no-site-filter \
    --latent-payload "${layout_args[@]}" \
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
  --cache "$LME_CACHE" "$merge_anchor"
