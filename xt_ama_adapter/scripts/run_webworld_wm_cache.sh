#!/usr/bin/env bash
# WebWorld (real a11y) WM data-supply tail: wait for the 4 encode shards,
# then fit PQ codes and build the discrete WM cache (cache_web contract).
set -u
RUN=/home/cbd/project/residual-mem
WW="$RUN/outputs/wm_train/webworld-v1"
PY=$RUN/.venv-ama-embedding-cu124/bin/python
mkdir -p "$RUN/logs/wm_train"
for k in 0 1 2 3; do
  while ! grep -q "wrote " "$RUN/logs/wm_train/webworld-encode-$k.log" 2>/dev/null; do
    sleep 20
  done
done
echo "[webworld] all 4 encode shards done $(date --iso-8601=seconds)"

cd "$RUN"
PYTHONPATH=. nohup "$PY" -u -m experiments.state_tokenizer.qformer_pq \
  --states "$WW/states-0.npz" --states "$WW/states-1.npz" \
  --states "$WW/states-2.npz" --states "$WW/states-3.npz" \
  --records "$WW/records.jsonl" \
  --output "$WW/pq-h32-c256.npz" --problem-batch 128 --fit-states 6000 \
  > "$RUN/logs/wm_train/webworld-pq.log" 2>&1
echo "[webworld] pq done $(date --iso-8601=seconds)"

PYTHONPATH=. "$PY" -m experiments.world_model.cache_web \
  --codes "$WW/pq-h32-c256.npz" \
  --records "$WW/records.jsonl" \
  --output "$WW/cache" --include-splits train validation \
  > "$RUN/logs/wm_train/webworld-cache.log" 2>&1
echo "[webworld] cache done $(date --iso-8601=seconds)"
