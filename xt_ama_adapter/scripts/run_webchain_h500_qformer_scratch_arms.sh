#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

pairs="outputs/ama_latent_memory/qformer/webchain-h500-v1.pairs.npz"
store="outputs/ama_latent_memory/qformer/webchain-h500-v1.store"
fused="outputs/ama_latent_memory/qformer/mixed-web-v1.fused-teacher"
observation="outputs/ama_latent_memory/qformer/mixed-web-v1.observation-teacher"
log_dir="outputs/ama_latent_memory/qformer/logs"
mkdir -p "$log_dir"

run_arm() {
  local weight="$1"
  local gpu="$2"
  local tag="webchain-h500-v1-scratch-obs${weight}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=.:xt_ama_adapter \
    .venv-ama-vllm-qwen35/bin/python -u -m experiments.state_tokenizer.train_qformer_joint \
    --pairs "$pairs" --xbar-dir "$store" \
    --teacher-dir "$fused" --teacher-cache "$observation" \
    --model /data1/models/Qwen3.5-9B \
    --output "outputs/ama_latent_memory/qformer/qformer-K32-${tag}.pt" \
    --device cuda:0 --layer 16 \
    --queries 32 --qformer-hidden 1024 --qformer-heads 8 --qformer-layers 4 \
    --qk-norm --no-drop-microbatches \
    --obs-weight "$weight" --distill-teacher observation --distill-weight 0.3 \
    --sem-weight 1.0 --sem-mode same-session --sem-batch 4 --sem-extra-negatives 0 \
    --max-steps 9000 --accumulate 4 --max-answer-tokens 64 \
    --learning-rate 1e-4 --clip-norm 5.0 --seed 35 \
    --eval-every 250 --validation-observations 96 --probe-observations 48 \
    --patience-evals 8 --min-lr-fraction 0.25 --lr-recover-steps 10 \
    --max-skipped-steps 500 \
    > "$log_dir/qformer-K32-${tag}.log" 2>&1
}

run_arm 0.5 0 &
pid05=$!
run_arm 1.0 1 &
pid10=$!
printf '%s\n' "$pid05" > "$log_dir/qformer-K32-webchain-h500-v1-scratch-obs0.5.pid"
printf '%s\n' "$pid10" > "$log_dir/qformer-K32-webchain-h500-v1-scratch-obs1.0.pid"
wait "$pid05" "$pid10"
