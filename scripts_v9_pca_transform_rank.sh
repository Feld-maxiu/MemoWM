#!/usr/bin/env bash
# Restartable launcher for one v9 Static-Key64 PCA transform shard.
set -u

rank=$1
gpu=$2
world_size=${3:-6}
max_attempts=${V9_PCA_TRANSFORM_MAX_ATTEMPTS:-20}

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${V9_ROOT:-$ROOT/outputs/state_tokenizer/v9-instruct}"
LOG="$OUT/logs/pca-transform-rank$(printf '%02d' "$rank").log"
PID_DIR="$OUT/pids"
STATUS_DIR="$OUT/status"
mkdir -p "$(dirname "$LOG")" "$PID_DIR" "$STATUS_DIR"
echo $$ > "$PID_DIR/pca-transform-rank$(printf '%02d' "$rank").pid"
printf 'running\n' > "$STATUS_DIR/pca-transform-rank$(printf '%02d' "$rank").status"

export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

for ((attempt=1; attempt<=max_attempts; attempt++)); do
  echo "=== rank=$rank gpu=$gpu attempt=$attempt $(date -Is) ===" >> "$LOG"
  "$ROOT/scripts_v9_instruct.sh" pca-transform "$rank" "$world_size" cuda:0 \
    >> "$LOG" 2>&1
  exit_code=$?
  if [ "$exit_code" -eq 0 ]; then
    echo "=== rank=$rank complete $(date -Is) ===" >> "$LOG"
    printf 'complete\n' > "$STATUS_DIR/pca-transform-rank$(printf '%02d' "$rank").status"
    exit 0
  fi
  echo "=== rank=$rank exit=$exit_code; retrying in 15s $(date -Is) ===" >> "$LOG"
  sleep 15
done

echo "=== rank=$rank gave up after $max_attempts attempts $(date -Is) ===" >> "$LOG"
printf 'failed\n' > "$STATUS_DIR/pca-transform-rank$(printf '%02d' "$rank").status"
exit 1
