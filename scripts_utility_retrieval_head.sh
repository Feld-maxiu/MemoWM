#!/usr/bin/env bash
# Build the full/gated retrieval cache and train only its standalone head.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DEV="cuda:${1:-0}"
D="${QFORMER_BRIDGE_DIR:-$ROOT/outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar}"
DATA="${RESIDUALMEM_DATA_DIR:-$ROOT/../data/molmoweb-pilot}"
ARM="K32e-obs0.5-gapbest"
BASE="$D/cache-$ARM.npz"
CACHE="$D/cache-$ARM-utility-gated.npz"
HEAD="$D/head-$ARM-utility-gated.pt"

for path in \
  "$BASE" \
  "$ROOT/gate/codes.npz" \
  "$ROOT/gate/posteriors.npz" \
  "$ROOT/gate/mask-lambda0.0010.npz" \
  "$ROOT/gate/records.jsonl" \
  "$DATA/pq-full/opq-shared-mix10-M32-C64.npz"; do
  test -f "$path" || { echo "missing artifact: $path" >&2; exit 1; }
done

set +u
source /mnt/data/public_tools/miniconda3/etc/profile.d/conda.sh
conda activate "${QFORMER_CONDA_ENV:-qwen-vl}"
set -u

cd "$ROOT"
if [[ -f "$HEAD" ]]; then
  echo "refusing to overwrite selected locked head: $HEAD" >&2
  echo "move to a fresh output tree before reproducing from scratch" >&2
  exit 2
fi
if [[ ! -f "$CACHE" ]]; then
  python -u -m experiments.state_tokenizer.build_utility_retrieval_cache \
    --cache "$BASE" \
    --codes gate/codes.npz \
    --codebook "$DATA/pq-full/opq-shared-mix10-M32-C64.npz" \
    --posteriors gate/posteriors.npz \
    --mask gate/mask-lambda0.0010.npz \
    --records gate/records.jsonl \
    --label train \
    --output "$CACHE"
fi

python -u -m experiments.state_tokenizer.train_retrieval_bridge \
  --cache "$CACHE" \
  --representation utility_gated \
  --consistency-weight "${UTILITY_KEY_CONSISTENCY_WEIGHT:-0.1}" \
  --output "$HEAD" \
  --device "$DEV"

python -u -m experiments.state_tokenizer.head_recall \
  --head "$HEAD" \
  --cache "$CACHE" \
  --split validation \
  --view gated \
  --device cpu \
  --output "$D/head-$ARM-utility-gated-recall.json"
