#!/bin/bash
# Watchdog for the overnight A2 run.
#
# A2 prints nothing until it finishes, so "still alive" and "died an hour ago"
# look identical from the outside. This relaunches it if the process is gone and
# the result file was never written, and records every attempt with a timestamp
# so the morning check can tell one clean run from three crashes.
#
# It cannot survive the session container being rebuilt -- that is what wiped
# /tmp and the Playwright browsers mid-collection earlier. If the morning shows
# no watchdog and no result, that is the cause; the artifacts on NAS are intact
# and A2 can simply be relaunched.
set -u
R=/root/nas/users/luzheng/workspace/ssh/czs/ResidualMem
JX=/root/nas/users/luzheng/workspace/enter/envs/ResidualMem/bin/python3.11
V8=$R/outputs/state_tokenizer/v8
RESULT=$R/outputs/a2/v8-m32-gw
LOG=$R/outputs/state_tokenizer/v8-logs/a2.log
WATCH=$R/outputs/state_tokenizer/v8-logs/a2-watchdog.log

cd "$R"
for attempt in $(seq 1 6); do
  if [ -f "$RESULT" ]; then
    echo "$(date -Is) result present, watchdog exiting" >> "$WATCH"
    exit 0
  fi
  if pgrep -f "a2_categorical_bottleneck" > /dev/null; then
    sleep 120
    continue
  fi
  echo "$(date -Is) attempt $attempt: no process and no result, launching" >> "$WATCH"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 PYTHONFAULTHANDLER=1 \
    $JX -u -m experiments.state_tokenizer.a2_categorical_bottleneck \
    --records $V8/full-721.jsonl --features $V8/static_features \
    --normalization $V8/key64-static-pca-normalization.npz \
    --init-a1-checkpoint $R/outputs/a1/v8.npz --a1-reference $R/outputs/a1/v8 \
    --num-subspaces 32 --group-weights '1,2,0.5,0.5' \
    --output "$RESULT" --label a2-v8-m32-gw >> "$LOG" 2>&1
  echo "$(date -Is) attempt $attempt exited status=$?" >> "$WATCH"
  sleep 30
done
echo "$(date -Is) watchdog gave up after 6 attempts" >> "$WATCH"
