#!/bin/bash
# One self-healing worker for the v8 collection.
#
# Two failure modes made the first two attempts useless, and this script exists
# to survive both:
#
#   * The session container is rebuilt, wiping /tmp and ~/.cache/ms-playwright.
#     The browser binary vanishes mid-run and every later reset fails. Pointing
#     PLAYWRIGHT_BROWSERS_PATH at NAS keeps it; the system .so files are frozen
#     alongside it, so a rebuilt container needs no apt.
#   * A worker dies. --resume continues from the shard on disk, so a crash costs
#     the last episode rather than the run.
#
# Completion is judged by exit code, not by the manifest: the collector writes a
# manifest even when it gives up on a task, and reports that in aborted_tasks.
set -u
w=$1
R="$(cd "$(dirname "$0")" && pwd)"
export PLAYWRIGHT_BROWSERS_PATH="$R/browsergym-venv/browsers"
export LD_LIBRARY_PATH="$R/browsergym-venv/syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export MINIWOB_URL="file://$R/third_party/miniwob-plusplus/miniwob/html/miniwob/"
export PYTHONPATH="$R"
OUT="$R/outputs/state_tokenizer/v8"
LOG="$R/outputs/state_tokenizer/v8-logs/w$w.log"
mkdir -p "$(dirname "$LOG")"

for attempt in $(seq 1 100); do
  echo "=== attempt $attempt $(date -Is) ===" >> "$LOG"
  "$R/browsergym-venv/bin/python" -m experiments.state_tokenizer.collect_browsergym \
    --output "$OUT" --target-states 100000 --max-steps 7 --random-action-prob 0.5 \
    --num-workers 12 --worker-id "$w" --resume --log-level INFO >> "$LOG" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "=== worker $w complete $(date -Is) ===" >> "$LOG"
    exit 0
  fi
  echo "=== worker $w exit=$status, retrying $(date -Is) ===" >> "$LOG"
  sleep 15
done
echo "=== worker $w gave up after 100 attempts ===" >> "$LOG"
exit 1
