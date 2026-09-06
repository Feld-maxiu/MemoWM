#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=/home/cbd/project/residual-mem
PY9="$RUN_ROOT/.venv-ama-vllm-qwen35/bin/python"
PY32="$RUN_ROOT/.venv-ama-embedding-cu124/bin/python"
MODEL9=/data1/models/Qwen3.5-9B
MODEL32=/data1/models/Qwen3-32B
PAIR_FILE="$RUN_ROOT/outputs/ama_latent_memory/syqa/qformer-pairs-teacher-complete.npz"
STORE_DIR="$RUN_ROOT/outputs/ama_latent_memory/syqa/qformer-store-teacher-complete"
TEACHER_DIR="$RUN_ROOT/outputs/ama_latent_memory/syqa/teacher-fused-complete"
TRUNK_CACHE=/data1/datasets/residual-mem/syqa-trunk-layer16-teacher-complete
QFORMER_DIR="$RUN_ROOT/outputs/ama_latent_memory/syqa/qformer"
POST_ROOT="$RUN_ROOT/outputs/ama_latent_memory/syqa/post-qformer"
CANDIDATES_DIR="$POST_ROOT/candidates"
BRIDGE_DIR="$POST_ROOT/bridge"
LOG_DIR="$RUN_ROOT/logs/ama_latent_memory/syqa/post-qformer"

mkdir -p "$CANDIDATES_DIR" "$BRIDGE_DIR" "$LOG_DIR"
cd "$RUN_ROOT"
export PYTHONPATH="$RUN_ROOT:$RUN_ROOT/xt_ama_adapter"
export TOKENIZERS_PARALLELISM=false

echo "[post] waiting for QFormer completion $(date --iso-8601=seconds)"
while [[ ! -f "$QFORMER_DIR/FORMAL_QFORMER_TRAINING_COMPLETE" ]]; do
  if ! tmux has-session -t syqa-formal 2>/dev/null; then
    echo "[post] syqa-formal disappeared before completion" >&2
    exit 1
  fi
  sleep 30
done
echo "[post] QFormer completion observed $(date --iso-8601=seconds)"

NAMES=(obs0.5-ce obs0.5-gap obs1.0-ce obs1.0-gap)
CKPTS=(
  "$QFORMER_DIR/qformer-K32-syqa-v1-obs0.5.pt"
  "$QFORMER_DIR/qformer-K32-syqa-v1-obs0.5.gapbest.pt"
  "$QFORMER_DIR/qformer-K32-syqa-v1-obs1.0.pt"
  "$QFORMER_DIR/qformer-K32-syqa-v1-obs1.0.gapbest.pt"
)
TRAIN_REPORTS=(
  "$QFORMER_DIR/qformer-K32-syqa-v1-obs0.5.json"
  "$QFORMER_DIR/qformer-K32-syqa-v1-obs0.5.json"
  "$QFORMER_DIR/qformer-K32-syqa-v1-obs1.0.json"
  "$QFORMER_DIR/qformer-K32-syqa-v1-obs1.0.json"
)

run_formal_gate() {
  local index=$1
  local gpu=$2
  local name=${NAMES[$index]}
  local checkpoint=${CKPTS[$index]}
  local training_report=${TRAIN_REPORTS[$index]}
  local directory="$CANDIDATES_DIR/$name"
  mkdir -p "$directory"
  if [[ -f "$directory/GATE_AUDITED" ]]; then
    return 0
  fi
  echo "[post] gate $name on GPU$gpu $(date --iso-8601=seconds)"
  set +e
  CUDA_VISIBLE_DEVICES=$gpu "$PY9" -u -m \
      experiments.state_tokenizer.observation_kl_precheck \
      --xbar-dir "$STORE_DIR" --checkpoint "$checkpoint" --model "$MODEL9" \
      --queries 32 --probe P4 --observations 48 --max-new-tokens 96 --seed 35 \
      --device cuda:0 --output "$directory/p4.json" \
      > "$LOG_DIR/$name-p4.log" 2>&1
  p4_status=$?
  set -e
  if [[ ! -f "$directory/p4.json" ]]; then
    touch "$directory/AUDIT_FAILED"
    echo "[post] $name P4 audit did not produce a report (status=$p4_status)" >&2
    return 0
  fi
  if ! CUDA_VISIBLE_DEVICES=$gpu "$PY9" -u -m \
      experiments.state_tokenizer.evaluate_qformer_qa_controls \
      --pairs "$PAIR_FILE" --xbar-dir "$STORE_DIR" --checkpoint "$checkpoint" \
      --model "$MODEL9" --output "$directory/qa-controls.json" \
      --queries 32 --qformer-layers 4 --observations 96 --seed 35 --device cuda:0 \
      > "$LOG_DIR/$name-qa-controls.log" 2>&1; then
    touch "$directory/AUDIT_FAILED"
    echo "[post] $name QA-control audit failed" >&2
    return 0
  fi
  if "$PY9" -u -m experiments.state_tokenizer.freeze_qformer_candidate \
      --checkpoint "$checkpoint" --training-report "$training_report" \
      --p4-report "$directory/p4.json" --qa-controls "$directory/qa-controls.json" \
      --output "$directory/frozen.pt" \
      > "$LOG_DIR/$name-freeze.log" 2>&1; then
    touch "$directory/GATE_PASSED"
  else
    echo "[post] $name formal gate rejected; materializing exploratory artifact" >&2
    "$PY9" -u -m experiments.state_tokenizer.freeze_qformer_candidate \
      --checkpoint "$checkpoint" --training-report "$training_report" \
      --p4-report "$directory/p4.json" --qa-controls "$directory/qa-controls.json" \
      --exploratory --output "$directory/frozen.pt" \
      > "$LOG_DIR/$name-freeze-exploratory.log" 2>&1
    touch "$directory/GATE_EXPLORATORY"
  fi
  touch "$directory/GATE_AUDITED"
  echo "[post] gate $name audited $(date --iso-8601=seconds)"
}

# Three physical GPUs, one worker per card.  The fourth candidate starts only
# after the first wave, so GPU3 is never visible to this pipeline.
run_formal_gate 0 0 & gate0=$!
run_formal_gate 1 1 & gate1=$!
run_formal_gate 2 2 & gate2=$!
wait "$gate0"
wait "$gate1"
wait "$gate2"
run_formal_gate 3 0

audited=0
for directory in "$CANDIDATES_DIR"/*; do
  [[ -d "$directory" && -f "$directory/GATE_AUDITED" ]] || continue
  audited=$((audited + 1))
  name=$(basename "$directory")
  echo "[post] cache/retrieval $name $(date --iso-8601=seconds)"
  if [[ ! -f "$directory/cache.npz" ]]; then
    cache_failed=0
    cache_pids=()
    for rank in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$rank "$PY9" -u -m \
        experiments.state_tokenizer.build_qformer_bridge_cache \
        --xbar-dir "$STORE_DIR" --teacher-dir "$TEACHER_DIR" \
        --checkpoint "$directory/frozen.pt" --model "$MODEL9" \
        --trunk-cache "$TRUNK_CACHE" --output "$directory/cache.rank${rank}.npz" \
        --queries 32 --qformer-layers 4 --qk-norm --device cuda:0 \
        --validation-fraction 0.2 --seed 35 --rank "$rank" --world-size 3 \
        > "$LOG_DIR/$name-cache-rank${rank}.log" 2>&1 &
      cache_pids+=("$!")
    done
    for pid in "${cache_pids[@]}"; do
      wait "$pid" || cache_failed=1
    done
    if [[ $cache_failed -ne 0 ]]; then
      touch "$directory/RETRIEVAL_FAILED"
      echo "[post] $name cache build failed" >&2
      continue
    fi
    "$PY9" -u -m experiments.state_tokenizer.merge_qformer_bridge_cache \
      --shards "$directory/cache.rank0.npz" "$directory/cache.rank1.npz" \
      "$directory/cache.rank2.npz" --output "$directory/cache.npz" \
      > "$LOG_DIR/$name-cache-merge.log" 2>&1
  fi
  if [[ ! -f "$directory/head.pt" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PY9" -u -m \
      experiments.state_tokenizer.train_retrieval_bridge \
      --cache "$directory/cache.npz" --output "$directory/head.pt" \
      --device cuda:0 --max-steps 10000 --batch-size 64 --eval-batch-size 128 \
      --eval-every 250 --patience-evals 12 --learning-rate 3e-4 \
      --weight-decay 1e-4 --temperature 0.05 --clip-norm 5.0 --seed 35 \
      > "$LOG_DIR/$name-head-train.log" 2>&1
  fi
  "$PY9" -u -m experiments.state_tokenizer.head_recall \
    --head "$directory/head.pt" --cache "$directory/cache.npz" \
    --split validation --device cpu --output "$directory/head-recall.json" \
    > "$LOG_DIR/$name-head-recall.log" 2>&1
done

if [[ $audited -eq 0 ]]; then
  touch "$POST_ROOT/BLOCKED_NO_QFORMER_AUDIT"
  echo "[post] no QFormer candidate completed P4/QA audits" >&2
  exit 1
fi

if ! "$PY9" -u -m experiments.state_tokenizer.select_syqa_bridge_candidate \
    --candidates-dir "$CANDIDATES_DIR" --output "$POST_ROOT/selection.json" \
    > "$LOG_DIR/candidate-selection.log" 2>&1; then
  touch "$POST_ROOT/BLOCKED_NO_COMPLETE_CANDIDATE"
  echo "[post] no candidate completed all selection audits" >&2
  exit 1
fi

readarray -t SELECTED < <("$PY9" -c \
  'import json,sys; x=json.load(open(sys.argv[1]))["selected"]; print(x["name"]); print(x["artifact"]); print(x["cache"]); print(x["head"])' \
  "$POST_ROOT/selection.json")
SELECTED_NAME=${SELECTED[0]}
SELECTED_QFORMER=${SELECTED[1]}
SELECTED_CACHE=${SELECTED[2]}
SELECTED_HEAD=${SELECTED[3]}
echo "[post] selected $SELECTED_NAME $(date --iso-8601=seconds)"

BRIDGE_DATA="$BRIDGE_DIR/syqa-matched-k32.npz"
if [[ ! -f "$BRIDGE_DATA" ]]; then
  "$PY9" -u -m experiments.state_tokenizer.build_molmoweb_qwen32_bridge_dataset \
    --cache "$SELECTED_CACHE" --pairs "$PAIR_FILE" --head "$SELECTED_HEAD" \
    --output "$BRIDGE_DATA" > "$LOG_DIR/bridge-dataset.log" 2>&1
fi

# Required one-step baseline smoke from the manual.
if [[ ! -f "$BRIDGE_DIR/smoke-required.pt" ]]; then
  CUDA_VISIBLE_DEVICES=0,1,2 "$PY32" -u -m \
    experiments.state_tokenizer.train_qwen32_bridge \
    --dataset "$BRIDGE_DATA" --model "$MODEL32" \
    --output "$BRIDGE_DIR/smoke-required.pt" --distill-weight 0 \
    --micro-batch-size 1 --accumulate 8 --max-steps 1 --eval-every 1 \
    --eval-limit 2 --rms-calibrate-to-reader --gradient-checkpointing \
    > "$LOG_DIR/bridge-smoke-required.log" 2>&1
fi

# Benchmark mathematically equivalent effective-batch-8 layouts.  Fixed model
# load/hash overhead is shared by every run; all benchmark artifacts are kept
# separate and are never used to initialize the formal Bridge.
BENCHMARK="$BRIDGE_DIR/microbatch-benchmark.tsv"
if [[ ! -f "$BENCHMARK" ]]; then
  : > "$BENCHMARK"
  for micro in 1 2 4; do
    accum=$((8 / micro))
    start=$(date +%s)
    if CUDA_VISIBLE_DEVICES=0,1,2 "$PY32" -u -m \
        experiments.state_tokenizer.train_qwen32_bridge \
        --dataset "$BRIDGE_DATA" --model "$MODEL32" \
        --output "$BRIDGE_DIR/smoke-m${micro}-a${accum}.pt" --distill-weight 0 \
        --micro-batch-size "$micro" --accumulate "$accum" \
        --max-steps 3 --eval-every 3 --eval-limit 2 \
        --rms-calibrate-to-reader --gradient-checkpointing \
        > "$LOG_DIR/bridge-smoke-m${micro}-a${accum}.log" 2>&1; then
      elapsed=$(( $(date +%s) - start ))
      printf '%s\t%s\t%s\t%s\n' "$micro" "$accum" "$elapsed" pass >> "$BENCHMARK"
    else
      elapsed=$(( $(date +%s) - start ))
      printf '%s\t%s\t%s\t%s\n' "$micro" "$accum" "$elapsed" fail >> "$BENCHMARK"
    fi
  done
fi

fastest=$(awk -F '\t' '$4 == "pass" {print $0}' "$BENCHMARK" | sort -t $'\t' -k3,3n | head -n 1)
if [[ -z "$fastest" ]]; then
  touch "$POST_ROOT/BLOCKED_BRIDGE_SMOKE"
  echo "[post] every equivalent bridge micro-batch smoke failed" >&2
  exit 1
fi
IFS=$'\t' read -r BEST_MICRO BEST_ACCUMULATE BEST_SECONDS _ <<< "$fastest"
echo "[post] bridge layout micro=$BEST_MICRO accumulate=$BEST_ACCUMULATE benchmark=${BEST_SECONDS}s"

TEACHER_CACHE="$BRIDGE_DIR/qwen32-teacher-top128"
CUDA_VISIBLE_DEVICES=0,1,2 "$PY32" -u -m \
  experiments.state_tokenizer.build_qwen32_teacher_cache \
  --dataset "$BRIDGE_DATA" --model "$MODEL32" --output "$TEACHER_CACHE" \
  --top-k 128 --min-coverage 0.99 --max-memory-gib 34 --enable-thinking \
  --resume --seed 35 > "$LOG_DIR/qwen32-teacher-cache.log" 2>&1

FORMAL_BRIDGE="$BRIDGE_DIR/qwen32-input-k32-rms.pt"
echo "[post] formal Bridge start $(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES=0,1,2 "$PY32" -u -m \
  experiments.state_tokenizer.train_qwen32_bridge \
  --dataset "$BRIDGE_DATA" --teacher-cache "$TEACHER_CACHE" \
  --model "$MODEL32" --output "$FORMAL_BRIDGE" \
  --distill-weight 0.3 --micro-batch-size "$BEST_MICRO" \
  --accumulate "$BEST_ACCUMULATE" --max-steps 3000 --eval-every 250 \
  --patience-evals 8 --learning-rate 1e-4 --weight-decay 1e-4 \
  --clip-norm 5.0 --rms-calibrate-to-reader --gradient-checkpointing \
  --seed 35 --enable-thinking > "$LOG_DIR/qwen32-bridge-formal.log" 2>&1

touch "$BRIDGE_DIR/FORMAL_BRIDGE_TRAINING_COMPLETE"
echo "[post] formal Bridge complete $(date --iso-8601=seconds)"
