#!/usr/bin/env bash
# One restartable deterministic BrowserGym episode lane.
set -u

task_index=$1
lane=$2
lanes=$3
target_episodes=$4
R="$(cd "$(dirname "$0")" && pwd)"
shard=$(printf 'lane-t%02d-l%02dof%02d' "$task_index" "$lane" "$lanes")
OUT="$R/outputs/state_tokenizer/v8-lanes"
LOG="$R/outputs/state_tokenizer/v8-lane-logs/$shard.log"
PID_DIR="$R/outputs/state_tokenizer/v8-lane-pids"
mkdir -p "$(dirname "$LOG")" "$PID_DIR"
echo $$ > "$PID_DIR/$shard.pid"

export PLAYWRIGHT_BROWSERS_PATH="$R/browsergym-venv/browsers"
export LD_LIBRARY_PATH="$R/browsergym-venv/syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export MINIWOB_URL="file://$R/third_party/miniwob-plusplus/miniwob/html/miniwob/"
export PYTHONPATH="$R"
pre_observation_delay="${BROWSERGYM_PRE_OBSERVATION_DELAY:-0.5}"
reuse_args=()
if [[ "${BROWSERGYM_REUSE_BROWSER:-0}" == "1" ]]; then
  reuse_args+=(--reuse-browser)
fi

episode_start=$((task_index + 12 * lane))
episode_stride=$((12 * lanes))
for ((attempt=1; attempt<=100; attempt++)); do
  echo "=== attempt $attempt $(date -Is) target_episodes=$target_episodes ===" >> "$LOG"
  "$R/browsergym-venv/bin/python" -m experiments.state_tokenizer.collect_browsergym \
    --output "$OUT" --worker-id 0 --num-workers 1 \
    --assigned-task-index "$task_index" --episode-start "$episode_start" \
    --episode-stride "$episode_stride" --target-episodes "$target_episodes" \
    --shard-name "$shard" --max-steps 7 --random-action-prob 0.5 \
    --pre-observation-delay "$pre_observation_delay" \
    "${reuse_args[@]}" \
    --resume --log-every 25 --log-level INFO >> "$LOG" 2>&1
  status=$?
  if [ "$status" -eq 0 ]; then
    echo "=== $shard complete $(date -Is) ===" >> "$LOG"
    exit 0
  fi
  echo "=== $shard exit=$status, retrying $(date -Is) ===" >> "$LOG"
  sleep 5
done
echo "=== $shard gave up after 100 attempts ===" >> "$LOG"
exit 1
