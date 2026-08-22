#!/usr/bin/env bash
# Start/stop the local OpenAI-compatible endpoint that backs the WorldMemArena
# answer and judge stages.
#
# Control is by pidfile, deliberately. Matching on `ps | grep` is unsafe here:
# any command line that starts or stops this server contains the module path,
# so the pattern finds the caller as well as the server, and the usual
# bracket trick does not help because the pattern then appears twice on the
# same line. That mistake killed the calling shell three times before this
# script existed.
#
#   bash scripts_local_llm_server.sh start|stop|status
#
# GPUS accepts repeats -- "4,4,5,5" puts two replicas on each of two cards.
# A 9B in bf16 needs about 23 GB with activations, so two per 72 GB card is
# comfortable and doubles the concurrency the evaluator can actually use.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$ROOT/outputs/logs/eval/server.pid"
PY="${QWEN_PY:-/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python}"
mkdir -p "$ROOT/outputs/logs/eval"

case "${1:-start}" in
  stop)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      PID="$(cat "$PIDFILE")"
      # Kill the whole process group, not just the parent. In multi-worker mode
      # the parent spawns one child per replica; killing the parent alone leaves
      # them running and holding ~23 GB of GPU memory each, so the next start
      # OOMs. `setsid` at launch makes the parent a group leader, which is what
      # makes the negative-pid form work.
      kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
      for _ in $(seq 1 20); do
        kill -0 "$PID" 2>/dev/null || break
        sleep 0.5
      done
      kill -KILL -- "-$PID" 2>/dev/null || true
      echo "stopped pid=$PID (process group)"
    else
      echo "not running"
    fi
    rm -f "$PIDFILE"
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "running pid=$(cat "$PIDFILE")"
    else
      echo "not running"
    fi
    ;;
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running pid=$(cat "$PIDFILE")"
      exit 0
    fi
    LOG="$ROOT/outputs/logs/eval/server-$(date +%Y%m%d-%H%M%S).log"
    ln -sfn "$(basename "$LOG")" "$ROOT/outputs/logs/eval/server-latest.log"
    cd "$ROOT"
    PYTHONPATH="$ROOT" setsid nohup "$PY" -u \
      -m experiments.state_tokenizer.local_openai_server \
      --model "${MODEL:-models/Qwen3.5-9B}" \
      --gpus "${GPUS:-2,2,3,3,4,4,5,5,6,6,7,7}" \
      --port "${PORT:-8017}" \
      > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    echo "started pid=$(cat "$PIDFILE") log=$LOG"
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
