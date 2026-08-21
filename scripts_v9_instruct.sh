#!/usr/bin/env bash
# Explicit, restartable v9 pipeline. This script never backgrounds work.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
QWEN_PY="${QWEN_PY:-/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python}"
JAX_PY="${JAX_PY:-$ROOT/.venv-jax/bin/python}"
MODEL="${QWEN35_MODEL:-$ROOT/models/Qwen3.5-9B}"
V8="${V8_ROOT:-$ROOT/outputs/state_tokenizer/v8}"
V9="${V9_ROOT:-$ROOT/outputs/state_tokenizer/v9-instruct}"
BRIDGE="${BRIDGE_ROOT:-$ROOT/outputs/instruct_bridge/v9-instruct}"
A1_RESULT="${A1_RESULT:-$ROOT/outputs/a1/v9-instruct}"
A2_RESULT="${A2_RESULT:-$ROOT/outputs/a2/v9-instruct-m32-gw}"
A2="${A2_CHECKPOINT:-$A2_RESULT.npz}"
PCA_FIT_STATES="${PCA_FIT_STATES:-20000}"
PCA_FIT_SELECTION="${PCA_FIT_SELECTION:-task_balanced}"
A1_LABEL="${A1_LABEL:-a1-v9-instruct}"
A2_LABEL="${A2_LABEL:-a2-v9-instruct-m32-gw}"
BRIDGE_REPRESENTATION="${BRIDGE_REPRESENTATION:-xbar}"
TEACHER_BACKEND="${QWEN_VL_EMBED_BACKEND:-http}"
TEACHER_MODEL_PATH="${QWEN_VL_EMBED_MODEL_PATH:-$ROOT/models/Qwen3-VL-Embedding-8B}"
TEACHER_NUM_GPUS="${QWEN_VL_EMBED_NUM_GPUS:-1}"
TEACHER_BATCH_SIZE="${QWEN_VL_EMBED_BATCH_SIZE:-8}"
PHASE="${1:-help}"
RANK="${2:-0}"
WORLD_SIZE="${3:-1}"
DEVICE="${4:-cuda:0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

case "$PHASE" in
  modality-lengths)
    "$QWEN_PY" -m experiments.state_tokenizer.modality_lengths \
      --records "$V8/full-721.jsonl" --model "$MODEL" \
      --output "$V9/modality-lengths" \
      --rank "$RANK" --world-size "$WORLD_SIZE" --prompt-mode instruct
    ;;
  extract-full-h)
    "$QWEN_PY" -m experiments.state_tokenizer.extract_fixed_prompt \
      --records "$V8/full-721.jsonl" \
      --source-records "$V8/records-merged.jsonl" \
      --source-features "$V9/modality-lengths" \
      --model "$MODEL" --output "$V9/full-h" \
      --rank "$RANK" --world-size "$WORLD_SIZE" --device "$DEVICE" \
      --prompt-mode instruct --no-use-kernels
    ;;
  rebuild-static)
    "$QWEN_PY" -m experiments.state_tokenizer.rebuild_static_key64 \
      --records "$V8/full-721.jsonl" --full-h "$V9/full-h" \
      --model "$MODEL" --output "$V9/static_features" \
      --instruction-records "$V8/records-merged.jsonl" --filter-filler \
      --rank "$RANK" --world-size "$WORLD_SIZE" --device "$DEVICE" \
      --image-grid-thw 1 20 32 --prompt-mode instruct
    ;;
  pca-fit)
    "$QWEN_PY" -m experiments.state_tokenizer.key64_pca fit \
      --records "$V8/full-721.jsonl" --features "$V9/static_features" \
      --prefix key64-static --output "$V9/key64-static-pca.npz" \
      --device "$DEVICE" --fit-states "$PCA_FIT_STATES" \
      --fit-selection "$PCA_FIT_SELECTION"
    ;;
  pca-transform)
    "$QWEN_PY" -m experiments.state_tokenizer.key64_pca transform \
      --features "$V9/static_features" --pca "$V9/key64-static-pca.npz" \
      --prefix key64-static --rank "$RANK" --world-size "$WORLD_SIZE" \
      --device "$DEVICE"
    ;;
  normalization)
    "$JAX_PY" -m experiments.state_tokenizer.fit_normalization \
      --records "$V8/full-721.jsonl" --features "$V9/static_features" \
      --pca "$V9/key64-static-pca.npz" \
      --output "$V9/key64-static-pca-normalization.npz"
    ;;
  a1)
    mkdir -p "$(dirname "$A1_RESULT")"
    "$JAX_PY" -u -m experiments.state_tokenizer.a1_continuous_bottleneck \
      --records "$V8/full-721.jsonl" --features "$V9/static_features" \
      --normalization "$V9/key64-static-pca-normalization.npz" \
      --output "$A1_RESULT" --label "$A1_LABEL" \
      --num-e-tokens 64 --e-dim 512 \
      --stage-steps 2000 3000 5000 \
      --stage-learning-rates 1e-3 3e-4 1e-4 \
      --batch-size 32 --eval-batch-size 250 --eval-every 250 \
      --patience-evals 0 --gate-profile report_only \
      --platform gpu --device-index 0 --matmul-precision highest
    ;;
  a2)
    mkdir -p "$(dirname "$A2_RESULT")"
    "$JAX_PY" -u -m experiments.state_tokenizer.a2_categorical_bottleneck \
      --records "$V8/full-721.jsonl" --features "$V9/static_features" \
      --normalization "$V9/key64-static-pca-normalization.npz" \
      --init-a1-checkpoint "$A1_RESULT.npz" --a1-reference "$A1_RESULT" \
      --output "$A2_RESULT" --label "$A2_LABEL" \
      --num-e-tokens 64 --e-dim 512 --num-subspaces 32 --num-categories 256 \
      --kmeans-iterations 25 --kmeans-problem-batch 64 \
      --group-weights '1,2,0.5,0.5' \
      --max-steps 70000 --batch-size 32 --eval-batch-size 250 \
      --init-batch-size 250 --eval-every 500 --patience-evals 0 \
      --codebook-learning-rate 3e-4 --backbone-learning-rate 3e-5 \
      --gate-profile report_only --platform gpu --device-index 0 \
      --matmul-precision highest
    ;;
  bridge-cache)
    mkdir -p "$BRIDGE"
    BRIDGE_CACHE_ARGS=(
      --records "$V8/full-721.jsonl" --features "$V9/static_features"
      --normalization "$V9/key64-static-pca-normalization.npz"
      --representation "$BRIDGE_REPRESENTATION"
      --teacher-backend "$TEACHER_BACKEND"
      --jax-python "$JAX_PY"
      --output "$BRIDGE/train-cache-fused-observation.npz"
    )
    if [[ "$TEACHER_BACKEND" == "sentence_transformers" ]]; then
      BRIDGE_CACHE_ARGS+=(
        --teacher-model-path "$TEACHER_MODEL_PATH"
        --teacher-num-gpus "$TEACHER_NUM_GPUS"
        --teacher-batch-size "$TEACHER_BATCH_SIZE"
      )
    fi
    if [[ "$BRIDGE_REPRESENTATION" == "both" ]]; then
      BRIDGE_CACHE_ARGS+=(--a2-checkpoint "$A2")
    fi
    "$QWEN_PY" -m experiments.state_tokenizer.build_instruct_bridge_cache \
      "${BRIDGE_CACHE_ARGS[@]}"
    ;;
  retrieval-head)
    "$QWEN_PY" -m experiments.state_tokenizer.train_retrieval_bridge \
      --cache "$BRIDGE/train-cache-fused-observation.npz" \
      --output "$BRIDGE/retrieval-head-fused-observation.pt" \
      --representation "$BRIDGE_REPRESENTATION" --device "$DEVICE"
    ;;
  input-reader)
    "$QWEN_PY" -m experiments.state_tokenizer.train_reader_bridge \
      --cache "$BRIDGE/train-cache-fused-observation.npz" --model "$MODEL" \
      --mode input --representation "$BRIDGE_REPRESENTATION" \
      --output "$BRIDGE/input-connector.pt" --device "$DEVICE"
    ;;
  layer16-reader)
    "$QWEN_PY" -m experiments.state_tokenizer.train_reader_bridge \
      --cache "$BRIDGE/train-cache-fused-observation.npz" --model "$MODEL" \
      --mode layer16 --representation "$BRIDGE_REPRESENTATION" \
      --pca "$V9/key64-static-pca.npz" \
      --normalization "$V9/key64-static-pca-normalization.npz" \
      --output "$BRIDGE/layer16-connector.pt" --device "$DEVICE"
    ;;
  help|*)
    echo "usage: $0 {modality-lengths|extract-full-h|rebuild-static|pca-fit|pca-transform|normalization|a1|a2|bridge-cache|retrieval-head|input-reader|layer16-reader} [rank] [world-size] [device]"
    ;;
esac
