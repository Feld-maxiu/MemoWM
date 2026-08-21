#!/usr/bin/env bash
# Keep the deterministic BrowserGym lane plan filled up to a measured cap.
set -euo pipefail

R="$(cd "$(dirname "$0")" && pwd)"
MAX_RUNNING="${V8_MAX_RUNNING:-64}"
POLL_SECONDS="${V8_POLL_SECONDS:-30}"
TARGET_STATES="${V8_TARGET_STATES:-100000}"
LOG="$R/outputs/state_tokenizer/v8-lane-scheduler.log"
PID="$R/outputs/state_tokenizer/v8-lane-scheduler.pid"

mkdir -p "$R/outputs/state_tokenizer"
echo $$ > "$PID"
exec >> "$LOG" 2>&1

echo "=== scheduler start $(date -Is) max_running=$MAX_RUNNING target_states=$TARGET_STATES poll_seconds=$POLL_SECONDS ==="
export PYTHONPATH="$R"
exec "$R/browsergym-venv/bin/python" \
  -m experiments.state_tokenizer.launch_browsergym_lanes \
  --max-running "$MAX_RUNNING" \
  --target-states "$TARGET_STATES" \
  --watch \
  --poll-seconds "$POLL_SECONDS"
