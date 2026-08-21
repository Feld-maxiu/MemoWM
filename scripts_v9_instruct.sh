#!/usr/bin/env bash
# Explicit, restartable v9 pipeline. This script never backgrounds work.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
QWEN_PY="${QWEN_PY:-/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python}"
JAX_PY="${JAX_PY:-$ROOT/.venv-jax/bin/python}"
MODEL="${QWEN35_MODEL:-$ROOT/models/Qwen3.5-9B}"
V8="$ROOT/outputs/state_tokenizer/v8"
V9="$ROOT/outputs/state_tokenizer/v9-instruct"
BRIDGE="$ROOT/outputs/instruct_bridge/v9-instruct"
A2="${A2_CHECKPOINT:-$ROOT/outputs/a2/v9-instruct-m32-gw.npz}"
PHASE="${1:-help}"
RANK="${2:-0}"
WORLD_SIZE="${3:-1}"
DEVICE="${4:-cuda:0}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

case "$PHASE" in
  extract-full-h)
    "$QWEN_PY" -m experiments.state_tokenizer.extract_full_h \
      --records "$V8/full-721.jsonl" \
      --source-records "$V8/records-merged.jsonl" \
      --features "$V8/features" \
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
      --prompt-mode instruct
    ;;
  pca-fit)
    "$QWEN_PY" -m experiments.state_tokenizer.key64_pca fit \
      --records "$V8/full-721.jsonl" --features "$V9/static_features" \
      --prefix key64-static --output "$V9/key64-static-pca.npz" \
      --device "$DEVICE" --fit-states 2000
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
  bridge-cache)
    mkdir -p "$BRIDGE"
    "$QWEN_PY" -m experiments.state_tokenizer.build_instruct_bridge_cache \
      --records "$V8/full-721.jsonl" --features "$V9/static_features" \
      --normalization "$V9/key64-static-pca-normalization.npz" \
      --a2-checkpoint "$A2" \
      --jax-python "$JAX_PY" \
      --output "$BRIDGE/train-cache-fused-observation.npz"
    ;;
  retrieval-head)
    "$QWEN_PY" -m experiments.state_tokenizer.train_retrieval_bridge \
      --cache "$BRIDGE/train-cache-fused-observation.npz" \
      --output "$BRIDGE/retrieval-head-fused-observation.pt" \
      --device "$DEVICE"
    ;;
  input-reader)
    "$QWEN_PY" -m experiments.state_tokenizer.train_reader_bridge \
      --cache "$BRIDGE/train-cache-fused-observation.npz" --model "$MODEL" \
      --mode input --representation both \
      --output "$BRIDGE/input-connector.pt" --device "$DEVICE"
    ;;
  layer16-reader)
    "$QWEN_PY" -m experiments.state_tokenizer.train_reader_bridge \
      --cache "$BRIDGE/train-cache-fused-observation.npz" --model "$MODEL" \
      --mode layer16 --representation both \
      --pca "$V9/key64-static-pca.npz" \
      --normalization "$V9/key64-static-pca-normalization.npz" \
      --output "$BRIDGE/layer16-connector.pt" --device "$DEVICE"
    ;;
  help|*)
    echo "usage: $0 {extract-full-h|rebuild-static|pca-fit|pca-transform|normalization|bridge-cache|retrieval-head|input-reader|layer16-reader} [rank] [world-size] [device]"
    ;;
esac
