#!/usr/bin/env bash
# Restartable launcher for one v9 Full-H extraction shard.
set -u

rank=$1
gpu=$2
world_size=${3:-6}
max_attempts=${V9_EXTRACT_MAX_ATTEMPTS:-20}

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/outputs/state_tokenizer/v9-instruct"
LOG="$OUT/logs/extract-full-h-rank$(printf '%02d' "$rank").log"
PID_DIR="$OUT/pids"
STATUS_DIR="$OUT/status"
mkdir -p "$(dirname "$LOG")" "$PID_DIR" "$STATUS_DIR"
echo $$ > "$PID_DIR/extract-full-h-rank$(printf '%02d' "$rank").pid"
printf 'running\n' > "$STATUS_DIR/extract-full-h-rank$(printf '%02d' "$rank").status"

# Each worker sees only its assigned GPU, so the child always uses cuda:0.
export CUDA_VISIBLE_DEVICES="$gpu"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

for ((attempt=1; attempt<=max_attempts; attempt++)); do
  echo "=== rank=$rank gpu=$gpu attempt=$attempt $(date -Is) ===" >> "$LOG"
  "$ROOT/scripts_v9_instruct.sh" extract-full-h "$rank" "$world_size" cuda:0 \
    >> "$LOG" 2>&1
  status=$?
  if [ "$status" -eq 0 ]; then
    echo "=== rank=$rank complete $(date -Is) ===" >> "$LOG"
    printf 'complete\n' > "$STATUS_DIR/extract-full-h-rank$(printf '%02d' "$rank").status"
    exit 0
  fi
  echo "=== rank=$rank exit=$status; retrying in 15s $(date -Is) ===" >> "$LOG"
  sleep 15
done

echo "=== rank=$rank gave up after $max_attempts attempts $(date -Is) ===" >> "$LOG"
printf 'failed\n' > "$STATUS_DIR/extract-full-h-rank$(printf '%02d' "$rank").status"
exit 1
