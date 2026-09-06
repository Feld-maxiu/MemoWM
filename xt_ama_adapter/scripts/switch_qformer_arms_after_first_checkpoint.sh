#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"

log_dir="outputs/ama_latent_memory/qformer/logs"
ckpt05="outputs/ama_latent_memory/qformer/qformer-K32-webchain-h500-v1-scratch-obs0.5.pt"
ckpt10="outputs/ama_latent_memory/qformer/qformer-K32-webchain-h500-v1-scratch-obs1.0.pt"

checkpoint_ready() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  .venv-ama-vllm-qwen35/bin/python -c \
    'import sys, torch; p=torch.load(sys.argv[1], map_location="cpu", weights_only=False); assert p["metadata"]["best_step"] == 250' \
    "$path"
}

# Loading verifies the closing ZIP records and metadata, so a partially
# written 315 MB torch artifact can never trigger termination of stage 1.
until checkpoint_ready "$ckpt05" && checkpoint_ready "$ckpt10"; do
  sleep 30
done

stop_arm() {
  local pid_file="$1"
  local parent
  parent="$(cat "$pid_file")"
  local child
  child="$(pgrep -P "$parent" | head -n 1 || true)"
  if [[ -n "$child" ]]; then
    kill -TERM "$child"
    for _ in $(seq 1 30); do
      kill -0 "$child" 2>/dev/null || break
      sleep 1
    done
  fi
}

stop_arm "$log_dir/qformer-K32-webchain-h500-v1-scratch-obs0.5.pid"
stop_arm "$log_dir/qformer-K32-webchain-h500-v1-scratch-obs1.0.pid"

exec bash xt_ama_adapter/scripts/run_webchain_h500_passval_cached_continuation.sh
