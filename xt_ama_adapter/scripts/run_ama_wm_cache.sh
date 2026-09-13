#!/usr/bin/env bash
# AMA-Bench WM eval-supply tail: wait for the 5 encode shards, then requantize
# with the fitted webworld c64 codebook, build the discrete WM cache, and run
# the copy/source/marginal baselines. Never fits a codebook.
set -eu
RUN=/home/cbd/project/residual-mem
AMA="$RUN/outputs/wm_train/ama-v1"
PY=$RUN/.venv-ama-embedding-cu124/bin/python
for k in 0 1 2 3 4; do
  while ! grep -q "wrote " "$RUN/logs/wm_train/ama-encode-$k.log" 2>/dev/null; do
    sleep 20
  done
done
echo "[ama] all 5 encode shards done $(date --iso-8601=seconds)"

cd "$RUN"
export LD_LIBRARY_PATH=$(echo /data2/xrz/miniforge3/envs/internnav/lib/python3.9/site-packages/nvidia/*/lib | tr ' ' ':')
PYTHONPATH="$RUN" "$PY" -u -m xt_ama_adapter.scripts.quantize_states_pq \
  --states "$AMA/ama-states-0.npz" --states "$AMA/ama-states-1.npz" \
  --states "$AMA/ama-states-2.npz" --states "$AMA/ama-states-3.npz" \
  --states "$AMA/ama-states-4.npz" \
  --pq "$RUN/outputs/wm_train/webworld-v2/pq-h32-c64.npz" \
  --output "$AMA/ama-pq-c64-v2codebook.npz" \
  > "$RUN/logs/wm_train/ama-quantize.log" 2>&1
echo "[ama] requantize done $(date --iso-8601=seconds)"

PYTHONPATH="$RUN" "$PY" -m experiments.world_model.cache_web \
  --codes "$AMA/ama-pq-c64-v2codebook.npz" \
  --records "$AMA/records.jsonl" \
  --output "$AMA/cache-h4" --max-history 4 --include-splits train validation \
  > "$RUN/logs/wm_train/ama-cache.log" 2>&1
echo "[ama] cache done $(date --iso-8601=seconds)"

PYTHONPATH="$RUN" "$PY" -m experiments.world_model.baselines \
  --cache "$AMA/cache-h4" \
  --output "$AMA/baselines-h4" \
  > "$RUN/logs/wm_train/ama-baselines.log" 2>&1
echo "[ama] baselines done $(date --iso-8601=seconds)"
