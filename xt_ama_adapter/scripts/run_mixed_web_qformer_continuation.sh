#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

web_obs="outputs/ama_latent_memory/webchain/webchain-pass-review-v1.observation-teacher"
mixed_obs="outputs/ama_latent_memory/qformer/mixed-web-v1.observation-teacher"
log_dir="outputs/ama_latent_memory/qformer/logs"
mkdir -p "$mixed_obs" "$log_dir"

for rank in 0 1 2; do
  log="outputs/ama_latent_memory/webchain/logs/observation-teacher-rank${rank}.log"
  until grep -q "\[teacher\] rank ${rank} done=" "$log" 2>/dev/null; do
    sleep 30
  done
done

count="$(find "$web_obs" -maxdepth 1 -type f -name '*.npz' | wc -l)"
if [[ "$count" -ne 1190 ]]; then
  echo "expected 1190 WebChain observation-teacher files, found $count" >&2
  exit 1
fi

cp -asn "$(realpath outputs/ama_latent_memory/qformer/humantrajs-pseudo-web-v1.observation-teacher)/." "$mixed_obs/"
cp -asn "$(realpath "$web_obs")/." "$mixed_obs/"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:xt_ama_adapter \
  .venv-ama-vllm-qwen35/bin/python -u -m experiments.state_tokenizer.train_qformer_joint \
  --pairs outputs/ama_latent_memory/qformer/mixed-web-v1.pairs.npz \
  --xbar-dir outputs/ama_latent_memory/qformer/mixed-web-v1.store \
  --trunk-cache outputs/ama_latent_memory/qformer/mixed-web-v1.trunk-layer16 \
  --teacher-dir outputs/ama_latent_memory/qformer/mixed-web-v1.fused-teacher \
  --teacher-cache "$mixed_obs" \
  --model /data1/models/Qwen3.5-9B \
  --output outputs/ama_latent_memory/qformer/qformer-K32-mixed-web-v1-continue.pt \
  --init-from outputs/ama_latent_memory/qformer/selected/qformer-K32-obs1.0-gapbest-exploratory.pt \
  --device cuda:0 --layer 16 \
  --queries 32 --qformer-hidden 1024 --qformer-heads 8 --qformer-layers 4 \
  --qk-norm --no-drop-microbatches \
  --obs-weight 1.0 --distill-teacher observation --distill-weight 0.3 \
  --sem-weight 1.0 --sem-mode same-session --sem-batch 4 --sem-extra-negatives -1 \
  --max-steps 3000 --accumulate 4 --max-answer-tokens 64 \
  --learning-rate 3e-5 --warmup-steps 50 --clip-norm 5.0 \
  --eval-every 250 --validation-observations 96 --probe-observations 48 \
  --patience-evals 8 --min-lr-fraction 0.25 --lr-recover-steps 10 \
  --max-skipped-steps 500 \
  > "$log_dir/qformer-K32-mixed-web-v1-continue.log" 2>&1
