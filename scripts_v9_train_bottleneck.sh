#!/usr/bin/env bash
# Restartable process wrapper for the sequential v9 A1/A2 training stages.
set -u

phase=$1
gpu=${2:-0}
max_attempts=${V9_BOTTLENECK_MAX_ATTEMPTS:-6}

case "$phase" in
  a1|a2) ;;
  *) echo "usage: $0 {a1|a2} [gpu]" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="${V9_ROOT:-$ROOT/outputs/state_tokenizer/v9-instruct}"
LOG="$OUT/logs/train-${phase}.log"
PID_DIR="$OUT/pids"
STATUS_DIR="$OUT/status"
mkdir -p "$(dirname "$LOG")" "$PID_DIR" "$STATUS_DIR"
echo $$ > "$PID_DIR/train-${phase}.pid"
printf 'running\n' > "$STATUS_DIR/train-${phase}.status"

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONFAULTHANDLER=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest

for ((attempt=1; attempt<=max_attempts; attempt++)); do
  echo "=== phase=$phase gpu=$gpu attempt=$attempt $(date -Is) ===" >> "$LOG"
  "$ROOT/scripts_v9_instruct.sh" "$phase" >> "$LOG" 2>&1
  exit_code=$?
  if [ "$exit_code" -eq 0 ]; then
    echo "=== phase=$phase complete $(date -Is) ===" >> "$LOG"
    printf 'complete\n' > "$STATUS_DIR/train-${phase}.status"
    exit 0
  fi
  echo "=== phase=$phase exit=$exit_code; retrying in 30s $(date -Is) ===" >> "$LOG"
  sleep 30
done

echo "=== phase=$phase gave up after $max_attempts attempts $(date -Is) ===" >> "$LOG"
printf 'failed\n' > "$STATUS_DIR/train-${phase}.status"
exit 1
