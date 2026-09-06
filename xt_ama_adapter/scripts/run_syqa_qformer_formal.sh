#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
PAIR_FILE="$RUN_ROOT/outputs/ama_latent_memory/syqa/qformer-pairs-teacher-complete.npz"
STORE_DIR="$RUN_ROOT/outputs/ama_latent_memory/syqa/qformer-store-teacher-complete"
OBS_TEACHER="$RUN_ROOT/outputs/ama_latent_memory/syqa/teacher-observation"
FUSED_TEACHER="$RUN_ROOT/outputs/ama_latent_memory/syqa/teacher-fused-complete"
TRUNK_CACHE=/data1/datasets/residual-mem/syqa-trunk-layer16-teacher-complete
MODEL_9B=/data1/models/Qwen3.5-9B
EMBED_MODEL=/data1/models/Qwen3-VL-Embedding-8B
LOG_DIR="$RUN_ROOT/logs/ama_latent_memory/syqa"
QFORMER_DIR="$RUN_ROOT/outputs/ama_latent_memory/syqa/qformer"

mkdir -p "$FUSED_TEACHER" "$TRUNK_CACHE" "$LOG_DIR" "$QFORMER_DIR"
cd "$RUN_ROOT"
export PYTHONPATH="$RUN_ROOT"
export TOKENIZERS_PARALLELISM=false

run_three_ranks() {
  local label=$1
  shift
  local pids=()
  for rank in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$rank "$@" "$rank" \
      > "$LOG_DIR/${label}-rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if [[ $failed -ne 0 ]]; then
    echo "$label failed; inspect $LOG_DIR/${label}-rank*.log" >&2
    exit 1
  fi
}

run_fused_rank() {
  local rank=$1
  exec "$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python" -u -m \
    experiments.state_tokenizer.wma_encode_teacher \
    --xbar-dir "$STORE_DIR" --output-dir "$FUSED_TEACHER" \
    --model "$EMBED_MODEL" --device cuda:0 --batch-size 8 --resume \
    --rank "$rank" --world-size 3
}
export -f run_fused_rank
export RUN_ROOT STORE_DIR FUSED_TEACHER EMBED_MODEL

run_trunk_rank() {
  local rank=$1
  exec "$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python" -u -m \
    experiments.state_tokenizer.build_trunk_state_cache \
    --store "$STORE_DIR" --output "$TRUNK_CACHE" --model "$MODEL_9B" \
    --device cuda:0 --layer 16 --resume --rank "$rank" --world-size 3
}
export -f run_trunk_rank
export TRUNK_CACHE MODEL_9B

echo "[pipeline] stage=fused_teacher start $(date --iso-8601=seconds)"
run_three_ranks fused-teacher bash -c 'run_fused_rank "$1"' _
fused_count=$(find "$FUSED_TEACHER" -maxdepth 1 -type f -name '*.npz' | wc -l)
if [[ $fused_count -ne 2829 ]]; then
  echo "fused teacher count $fused_count != 2829" >&2
  exit 1
fi
echo "[pipeline] stage=fused_teacher complete files=$fused_count $(date --iso-8601=seconds)"

echo "[pipeline] stage=trunk_cache start $(date --iso-8601=seconds)"
run_three_ranks trunk-cache bash -c 'run_trunk_rank "$1"' _
trunk_count=$(find "$TRUNK_CACHE" -maxdepth 1 -type f -name '*.npz' | wc -l)
if [[ $trunk_count -ne 6645 ]]; then
  echo "trunk cache count $trunk_count != 6645" >&2
  exit 1
fi
echo "[pipeline] stage=trunk_cache complete files=$trunk_count $(date --iso-8601=seconds)"

run_qformer_arm() {
  local gpu=$1
  local obs_weight=$2
  local name="qformer-K32-syqa-v1-obs${obs_weight}"
  CUDA_VISIBLE_DEVICES=$gpu exec "$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python" -u -m \
    experiments.state_tokenizer.train_qformer_joint \
    --pairs "$PAIR_FILE" --xbar-dir "$STORE_DIR" --trunk-cache "$TRUNK_CACHE" \
    --teacher-dir "$FUSED_TEACHER" --teacher-cache "$OBS_TEACHER" \
    --model "$MODEL_9B" --output "$QFORMER_DIR/${name}.pt" --device cuda:0 \
    --queries 32 --qformer-layers 4 --accumulate 4 --qk-norm \
    --distill-weight 0.3 --distill-teacher observation --obs-weight "$obs_weight" \
    --sem-weight 1.0 --sem-mode same-session --sem-batch 4 --sem-extra-negatives 0 \
    --learning-rate 1e-4 --clip-norm 5.0 --seed 35 --no-drop-microbatches \
    --lr-recover-steps 10 --min-lr-fraction 0.25 --max-skipped-steps 500 \
    --held-out-probe P4 --probe-observations 24 \
    --max-steps 9000 --eval-every 250 --validation-observations 96 \
    --patience-evals 8 > "$LOG_DIR/${name}.log" 2>&1
}

echo "[pipeline] stage=qformer start $(date --iso-8601=seconds)"
run_qformer_arm 0 0.5 &
pid_a=$!
run_qformer_arm 1 1.0 &
pid_b=$!
failed=0
wait "$pid_a" || failed=1
wait "$pid_b" || failed=1
if [[ $failed -ne 0 ]]; then
  echo "QFormer arm failed; inspect $LOG_DIR/qformer-K32-syqa-v1-obs*.log" >&2
  exit 1
fi
echo "[pipeline] stage=qformer complete $(date --iso-8601=seconds)"
touch "$QFORMER_DIR/FORMAL_QFORMER_TRAINING_COMPLETE"
