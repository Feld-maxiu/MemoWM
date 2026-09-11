#!/usr/bin/env bash
# The plan is prepared separately; the released mask and WMA reader stay intact.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
GATE_PYTHON="${GATE_PYTHON:-/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
GATE_RUN=gate/two-stage-v1
mkdir -p "$GATE_RUN/logs"
if [[ -e "$GATE_RUN/mask-two-stage-lambda0.0010.npz" ]]; then
  echo "Exported pilot already exists; refusing to rerun." >&2
  exit 1
fi
pids=()
for rank in 0 1 2 3; do
  "$GATE_PYTHON" -u -m experiments.utility_gate.two_stage_mask label \
    --plan "$GATE_RUN/plan.json" --model models/Qwen3.5-9B \
    --output "$GATE_RUN/labels" --device "cuda:${rank}" \
    --rank "$rank" --world-size 4 --chunk 6 --memory-fraction 0.55 \
    > "$GATE_RUN/logs/rank-${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ "$status" != 0 ]]; then
  echo "A labeling shard failed. Inspect logs before resuming." >&2
  exit 1
fi
"$GATE_PYTHON" -m experiments.utility_gate.two_stage_mask export \
  --plan "$GATE_RUN/plan.json" --labels "$GATE_RUN/labels" \
  --test-posteriors gate/posteriors-testset.npz \
  --output "$GATE_RUN/mask-two-stage-lambda0.0010.npz"
