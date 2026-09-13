#!/usr/bin/env bash
# webworld-v2 tail: wait for the 3 encode shards, then PQ(C=64) + cache(max_history 32)
set -u
RUN=/home/cbd/project/residual-mem
PY=$RUN/.venv-ama-embedding-cu124/bin/python
OUT=$RUN/outputs/wm_train/webworld-v2
LOG=$RUN/logs/wm_train
for g in 0 1 2; do
  while ! grep -q "wrote " "$LOG/webworld2-encode-$g.log" 2>/dev/null; do
    sleep 60
  done
done
echo "[v2-tail] encode done $(date --iso-8601=seconds)"
PYTHONPATH=$RUN "$PY" -u -m experiments.state_tokenizer.qformer_pq \
  --states "$OUT/states-0.npz" --states "$OUT/states-1.npz" --states "$OUT/states-2.npz" \
  --records "$OUT/records-all.jsonl" \
  --output "$OUT/pq-h32-c64.npz" --num-categories 64 \
  --problem-batch 128 --fit-states 20000 \
  > "$LOG/webworld2-pq.log" 2>&1 || { echo "[v2-tail] PQ failed"; exit 1; }
echo "[v2-tail] PQ done $(date --iso-8601=seconds)"
PYTHONPATH=$RUN "$PY" -m experiments.world_model.cache_web \
  --codes "$OUT/pq-h32-c64.npz" \
  --records "$OUT/records-all.jsonl" \
  --output "$OUT/cache" --max-history 32 --include-splits train validation \
  > "$LOG/webworld2-cache.log" 2>&1 || { echo "[v2-tail] cache failed"; exit 1; }
echo "[v2-tail] cache done $(date --iso-8601=seconds)"
