#!/usr/bin/env bash
# webworld-v2: full-scale WebWorldData -> records -> 8-GPU encode -> PQ(C=64)
# -> cache(max_history 32). Scaled-up rebuild of run_webworld_wm_cache.sh after
# the 19k-transition run starved the copy gate (see WMA-era recipe: 495k
# transitions, max_history 32).
set -u
RUN=/home/cbd/project/residual-mem
PY=$RUN/.venv-ama-embedding-cu124/bin/python
SRC=/data1/datasets/xt_ama_adapter/webworld/full
OUT=$RUN/outputs/wm_train/webworld-v2
LOG=$RUN/logs/wm_train
QFORMER=$RUN/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce
# Qwen3.5 model classes need transformers>=5 (sglang env); the residual-mem venv's
# transformers 4.51 lacks them, so encoding runs on the sglang python.
PY_ENC=$RUN/.venv-ama-vllm-qwen35/bin/python
mkdir -p "$OUT" "$LOG"

echo "[v2] stage 1: records per chunk $(date --iso-8601=seconds)"
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
  PYTHONPATH=$RUN nohup "$PY" -u "$RUN/xt_ama_adapter/scripts/build_webworld_wm_records.py" \
    --slice "$SRC/chunk_$i.jsonl" --output-dir "$OUT/chunks/chunk_$i" \
    --row-offset $((i * 10000000)) \
    > "$LOG/webworld2-records-$i.log" 2>&1 &
done
wait
echo "[v2] stage 1 done $(date --iso-8601=seconds)"

echo "[v2] stage 2: merge states (dedupe, first wins)"
"$PY" - "$OUT" << 'PYEOF'
import json, glob, sys
out = sys.argv[1]
seen, order = set(), []
for path in sorted(glob.glob(out + "/chunks/chunk_*/states.jsonl")):
    with open(path) as handle:
        for line in handle:
            sid = json.loads(line)["state_id"]
            if sid not in seen:
                seen.add(sid)
                order.append(line if line.endswith("\n") else line + "\n")
with open(out + "/states-all.jsonl", "w") as handle:
    handle.writelines(order)
print("unique states:", len(order))
PYEOF
cat "$OUT"/chunks/chunk_*/records.jsonl > "$OUT/records-all.jsonl"

echo "[v2] stage 3: 8-GPU encode $(date --iso-8601=seconds)"
for g in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$RUN/xt_ama_adapter nohup "$PY_ENC" -u "$RUN/xt_ama_adapter/scripts/encode_wm_text_states.py" \
    --states-jsonl "$OUT/states-all.jsonl" \
    --residualmem-root "$RUN" \
    --qwen35-model /data1/models/Qwen3.5-9B \
    --qformer "$QFORMER/frozen.pt" \
    --retrieval-head "$QFORMER/head.pt" \
    --output "$OUT/states-$g.npz" --device cuda:0 --batch-size 4 \
    --shard-index $g --shard-count 6 \
    > "$LOG/webworld2-encode-$g.log" 2>&1 &
done
wait
for g in 0 1 2 3 4 5; do
  [ -s "$OUT/states-$g.npz" ] || { echo "[v2] encode shard $g missing output, aborting"; exit 1; }
done
echo "[v2] stage 3 done $(date --iso-8601=seconds)"

echo "[v2] stage 4: PQ fit C=64 $(date --iso-8601=seconds)"
PYTHONPATH=$RUN "$PY" -u -m experiments.state_tokenizer.qformer_pq \
  --states "$OUT/states-0.npz" --states "$OUT/states-1.npz" \
  --states "$OUT/states-2.npz" --states "$OUT/states-3.npz" \
  --states "$OUT/states-4.npz" --states "$OUT/states-5.npz" \
  --records "$OUT/records-all.jsonl" \
  --output "$OUT/pq-h32-c64.npz" --num-categories 64 \
  --problem-batch 128 --fit-states 20000 \
  > "$LOG/webworld2-pq.log" 2>&1 || exit 1
echo "[v2] stage 4 done $(date --iso-8601=seconds)"

echo "[v2] stage 5: cache max_history=32 $(date --iso-8601=seconds)"
PYTHONPATH=$RUN "$PY" -m experiments.world_model.cache_web \
  --codes "$OUT/pq-h32-c64.npz" \
  --records "$OUT/records-all.jsonl" \
  --output "$OUT/cache" --max-history 32 --include-splits train validation \
  > "$LOG/webworld2-cache.log" 2>&1 || exit 1
echo "[v2] all done $(date --iso-8601=seconds)"
