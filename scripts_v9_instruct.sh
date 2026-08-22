#!/usr/bin/env bash
# Explicit, restartable v9 pipeline. This script never backgrounds work.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
QWEN_PY="${QWEN_PY:-/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python}"
JAX_PY="${JAX_PY:-$ROOT/.venv-jax/bin/python}"
MODEL="${QWEN35_MODEL:-$ROOT/models/Qwen3.5-9B}"
V8="${V8_ROOT:-$ROOT/outputs/state_tokenizer/v8}"

# Two trees, deliberately. Raw extraction (modality-lengths, full-h, the 4096-d
# Key64 store) lives in V9_RAW; the official PCA coordinates live in V9_COORD,
# which hardlinks the raw feature files back to V9_RAW and holds only its own
# projections. They have identical directory structure and identical file names,
# so mixing them raises nothing and silently answers in the wrong coordinates --
# see experiments/state_tokenizer/pca_binding.py. V9_COORD's PCA is the official
# 20k task-balanced fit (f98c517c...); V9_RAW's is the invalidated first-N fit
# (6d2df9b5...) kept only for audit, and cannot be removed because the ~420 GiB
# Full-H sits in the same directory.
V9_RAW="${V9_ROOT:-$ROOT/outputs/state_tokenizer/v9-instruct}"
V9_COORD="${V9_COORD_ROOT:-$ROOT/outputs/state_tokenizer/v9-instruct-pca20k-balanced}"

BRIDGE="${BRIDGE_ROOT:-$ROOT/outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar}"

# Named after the coordinate tree they are fitted in. The unsuffixed
# outputs/a1/v9-instruct and outputs/a2/v9-instruct-m32-gw are the 8-21 artifacts
# bound to the invalidated first-N PCA; they stay on disk for audit, like the
# invalidated PCA itself, and must never be mixed into a run on these axes.
A1_RESULT="${A1_RESULT:-$ROOT/outputs/a1/v9-instruct-pca20k-balanced}"
A2_RESULT="${A2_RESULT:-$ROOT/outputs/a2/v9-instruct-pca20k-balanced-m32-gw}"
A2="${A2_CHECKPOINT:-$A2_RESULT.npz}"
PCA_FIT_STATES="${PCA_FIT_STATES:-20000}"
PCA_FIT_SELECTION="${PCA_FIT_SELECTION:-task_balanced}"
A1_LABEL="${A1_LABEL:-a1-v9-instruct-pca20k-balanced}"
A2_LABEL="${A2_LABEL:-a2-v9-instruct-pca20k-balanced-m32-gw}"
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

# The official coordinate system for the v9-instruct line: 20,000 task-balanced
# states, explained variance 0.9186. Pinned because verify_binding only proves
# the artifacts agree with each other -- both trees are internally consistent, so
# a self-consistency check alone cannot catch "pointed V9_COORD_ROOT at the
# invalidated tree". Re-fitting the PCA (route (b) domain adaptation) legitimately
# changes this; override EXPECTED_PCA_SHA256 then, and update every downstream
# number, because every code word and historical result is tied to these axes.
EXPECTED_PCA_SHA256="${EXPECTED_PCA_SHA256:-f98c517cb23e778b78eb04080ddc01b375a2efe9a42be8992e4a5e8a8f6efb2a}"

# Fail before spending GPU hours in the wrong coordinate system. Every artifact
# already records the PCA it was built from; this reads those records back.
# Takes key=value pairs, where key is one of features/normalization/pca.
require_coord_binding() {
  EXPECTED_PCA_SHA256="$EXPECTED_PCA_SHA256" "$JAX_PY" - "$@" <<'PY'
import os
import sys
from experiments.state_tokenizer.pca_binding import verify_binding
kwargs = dict(item.split("=", 1) for item in sys.argv[1:])
sha = verify_binding(**kwargs)
expected = os.environ.get("EXPECTED_PCA_SHA256", "")
if expected and sha != expected:
    raise SystemExit(
        f"[pca-binding] FAIL: artifacts agree on {sha[:16]}... but the official "
        f"coordinate system is {expected[:16]}...\n"
        "  Either V9_COORD_ROOT points at the invalidated tree, or the PCA was "
        "re-fitted.\n"
        "  If the re-fit was intended, set EXPECTED_PCA_SHA256 and re-derive "
        "every downstream artifact -- A1/A2 codes and all historical numbers are "
        "tied to the old axes."
    )
print(f"[pca-binding] ok {sha[:16]}...  ({', '.join(sorted(kwargs))})")
PY
}

# Full three-way check, for stages that consume projections and normalization.
coord_preflight() {
  require_coord_binding \
    "features=$V9_COORD/static_features" \
    "normalization=$V9_COORD/key64-static-pca-normalization.npz" \
    "pca=$V9_COORD/key64-static-pca.npz"
}

case "$PHASE" in
  modality-lengths)
    "$QWEN_PY" -m experiments.state_tokenizer.modality_lengths \
      --records "$V8/full-721.jsonl" --model "$MODEL" \
      --output "$V9_RAW/modality-lengths" \
      --rank "$RANK" --world-size "$WORLD_SIZE" --prompt-mode instruct
    ;;
  extract-full-h)
    "$QWEN_PY" -m experiments.state_tokenizer.extract_fixed_prompt \
      --records "$V8/full-721.jsonl" \
      --source-records "$V8/records-merged.jsonl" \
      --source-features "$V9_RAW/modality-lengths" \
      --model "$MODEL" --output "$V9_RAW/full-h" \
      --rank "$RANK" --world-size "$WORLD_SIZE" --device "$DEVICE" \
      --prompt-mode instruct --no-use-kernels
    ;;
  rebuild-static)
    "$QWEN_PY" -m experiments.state_tokenizer.rebuild_static_key64 \
      --records "$V8/full-721.jsonl" --full-h "$V9_RAW/full-h" \
      --model "$MODEL" --output "$V9_RAW/static_features" \
      --instruction-records "$V8/records-merged.jsonl" --filter-filler \
      --rank "$RANK" --world-size "$WORLD_SIZE" --device "$DEVICE" \
      --image-grid-thw 1 20 32 --prompt-mode instruct
    ;;
  pca-fit)
    # Reads the 4096-d Key64 store (hardlinked into V9_COORD) and writes the
    # coordinate tree's own PCA. --fit-selection task_balanced is load-bearing:
    # the default first-N selector drew all 2,000 states from click-button-v1.
    mkdir -p "$V9_COORD"
    "$QWEN_PY" -m experiments.state_tokenizer.key64_pca fit \
      --records "$V8/full-721.jsonl" --features "$V9_COORD/static_features" \
      --prefix key64-static --output "$V9_COORD/key64-static-pca.npz" \
      --device "$DEVICE" --fit-states "$PCA_FIT_STATES" \
      --fit-selection "$PCA_FIT_SELECTION"
    ;;
  pca-transform)
    "$QWEN_PY" -m experiments.state_tokenizer.key64_pca transform \
      --features "$V9_COORD/static_features" --pca "$V9_COORD/key64-static-pca.npz" \
      --prefix key64-static --rank "$RANK" --world-size "$WORLD_SIZE" \
      --device "$DEVICE"
    ;;
  normalization)
    # Normalization does not exist yet at this point; check features vs pca only.
    require_coord_binding \
      "features=$V9_COORD/static_features" \
      "pca=$V9_COORD/key64-static-pca.npz"
    "$JAX_PY" -m experiments.state_tokenizer.fit_normalization \
      --records "$V8/full-721.jsonl" --features "$V9_COORD/static_features" \
      --pca "$V9_COORD/key64-static-pca.npz" \
      --output "$V9_COORD/key64-static-pca-normalization.npz"
    ;;
  a1)
    coord_preflight
    mkdir -p "$(dirname "$A1_RESULT")"
    "$JAX_PY" -u -m experiments.state_tokenizer.a1_continuous_bottleneck \
      --records "$V8/full-721.jsonl" --features "$V9_COORD/static_features" \
      --normalization "$V9_COORD/key64-static-pca-normalization.npz" \
      --output "$A1_RESULT" --label "$A1_LABEL" \
      --num-e-tokens 64 --e-dim 512 \
      --stage-steps 2000 3000 5000 \
      --stage-learning-rates 1e-3 3e-4 1e-4 \
      --batch-size 32 --eval-batch-size 250 --eval-every 250 \
      --patience-evals 0 --gate-profile report_only \
      --platform gpu --device-index 0 --matmul-precision highest
    ;;
  a2)
    coord_preflight
    mkdir -p "$(dirname "$A2_RESULT")"
    "$JAX_PY" -u -m experiments.state_tokenizer.a2_categorical_bottleneck \
      --records "$V8/full-721.jsonl" --features "$V9_COORD/static_features" \
      --normalization "$V9_COORD/key64-static-pca-normalization.npz" \
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
    coord_preflight
    mkdir -p "$BRIDGE"
    BRIDGE_CACHE_ARGS=(
      --records "$V8/full-721.jsonl" --features "$V9_COORD/static_features"
      --normalization "$V9_COORD/key64-static-pca-normalization.npz"
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
    coord_preflight
    "$QWEN_PY" -m experiments.state_tokenizer.train_reader_bridge \
      --cache "$BRIDGE/train-cache-fused-observation.npz" --model "$MODEL" \
      --mode layer16 --representation "$BRIDGE_REPRESENTATION" \
      --pca "$V9_COORD/key64-static-pca.npz" \
      --normalization "$V9_COORD/key64-static-pca-normalization.npz" \
      --output "$BRIDGE/layer16-connector.pt" --device "$DEVICE"
    ;;
  coord-check)
    # Standalone binding audit; run it after any pipeline change.
    coord_preflight
    ;;
  wma-smoke)
    coord_preflight
    : "${WMA_DATASET:?set WMA_DATASET to the WorldMemArena data root}"
    # The adapter reads the head from the environment, not from a flag.
    export RESIDUALMEM_RETRIEVAL_HEAD="${RESIDUALMEM_RETRIEVAL_HEAD:-$BRIDGE/retrieval-head-fused-observation.pt}"
    # -1 selects the final (cumulative) checkpoint. Index 0 covers only the first
    # few sessions and yields too few observations to judge anything.
    "$QWEN_PY" -u -m experiments.state_tokenizer.worldmemarena_retrieval_smoke \
      --dataset "$WMA_DATASET" \
      --sample-id "${WMA_SAMPLE_ID:-web_01}" \
      --checkpoint-index "${WMA_CHECKPOINT_INDEX:--1}" \
      --qwen35-model "$MODEL" \
      --pca "$V9_COORD/key64-static-pca.npz" \
      --normalization "$V9_COORD/key64-static-pca-normalization.npz" \
      --tokenizer-device "$DEVICE" \
      --output "$BRIDGE/worldmemarena-${WMA_SAMPLE_ID:-web_01}-smoke.json"
    ;;
  help|*)
    echo "usage: $0 {modality-lengths|extract-full-h|rebuild-static|pca-fit|pca-transform|normalization|a1|a2|bridge-cache|retrieval-head|input-reader|layer16-reader|coord-check|wma-smoke} [rank] [world-size] [device]"
    echo
    echo "coordinate trees:"
    echo "  V9_RAW   = $V9_RAW"
    echo "             (modality-lengths, full-h, 4096-d Key64 store)"
    echo "  V9_COORD = $V9_COORD"
    echo "             (official PCA + normalization + projections)"
    echo "  BRIDGE   = $BRIDGE"
    ;;
esac
