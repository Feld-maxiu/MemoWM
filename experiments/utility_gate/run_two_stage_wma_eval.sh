#!/usr/bin/env bash
# Paired WMA evaluation: replace only the utility mask and rebuilt codec cache.
# Do not overwrite the release, retrain any head, or alter the WMA adapter.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
WMA_EVAL_REPO="$PWD"
WMA_EVAL_ROOT="$WMA_EVAL_REPO/../WorldMemArena"
WMA_EVAL_PYTHON="${WMA_EVAL_PYTHON:-/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python}"
WMA_EVAL_BRIDGE="$WMA_EVAL_REPO/outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar"
WMA_EVAL_OUTPUT="$WMA_EVAL_REPO/gate/two-stage-v1/wma-qa"
export PYTHONPATH="$WMA_EVAL_REPO:$WMA_EVAL_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export RESIDUALMEM_ROOT="$WMA_EVAL_REPO"
export RESIDUALMEM_QWEN35_MODEL="$WMA_EVAL_REPO/models/Qwen3.5-9B"
export RESIDUALMEM_QFORMER="$WMA_EVAL_BRIDGE/qformer-K32e-obs0.5.gapbest.pt"
export RESIDUALMEM_QFORMER_QUERIES=32 RESIDUALMEM_QFORMER_LAYERS=4
export RESIDUALMEM_RETRIEVAL_HEAD="$WMA_EVAL_BRIDGE/head-K32e-obs0.5-gapbest-utility-gated.pt"
export RESIDUALMEM_CODEC_RECONSTRUCTIONS="$WMA_EVAL_REPO/gate/two-stage-v1/wma-web-closedloop-two-stage-lambda0.0010.npz"
export QWEN_VL_EMBED_LOCAL=1 LLM_MAX_CONCURRENT=24
export OPENAI_MODEL_JUDGE=Qwen3.5-9B OPENAI_API_KEY_JUDGE=EMPTY JUDGE_TEMPERATURE=0
unset ANCHOR_RECORDS RESIDUALMEM_WMA_ANCHOR_RECORDS RESIDUALMEM_INPUT_CONNECTOR
unset RESIDUALMEM_L16_CONNECTOR RESIDUALMEM_NO_READER RESIDUALMEM_QFORMER_SELF_ATTN
test -f "$RESIDUALMEM_CODEC_RECONSTRUCTIONS"
test -f "$RESIDUALMEM_QFORMER"
test -f "$RESIDUALMEM_RETRIEVAL_HEAD"
for rank in 0 1; do
  if [[ -e "$WMA_EVAL_OUTPUT/shard-${rank}" ]]; then
    echo "WMA shard directory already exists; inspect/resume it explicitly instead of overwriting." >&2
    exit 1
  fi
done
mkdir -p "$WMA_EVAL_OUTPUT/logs"
"$WMA_EVAL_PYTHON" - <<'PY'
import hashlib, json, os
from pathlib import Path
import numpy as np
from residualmem.benchmarks.wma_codec_reconstruction import load_wma_codec_reconstruction_cache
repo = Path(os.environ['RESIDUALMEM_ROOT'])
codec_path = Path(os.environ['RESIDUALMEM_CODEC_RECONSTRUCTIONS'])
load_wma_codec_reconstruction_cache(codec_path)
mask_path = repo / 'gate/two-stage-v1/mask-two-stage-lambda0.0010.npz'
with np.load(codec_path, allow_pickle=False) as data:
    meta = json.loads(str(data['metadata']))
assert (meta['states'], meta['transitions'], meta['initial_all_send_states']) == (956, 817, 139)
assert meta['lambda'] == 0.001
assert meta['mask_sha256'] == hashlib.sha256(mask_path.read_bytes()).hexdigest()
def fingerprint(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
config = {
    'protocol': 'wma-two-stage-utility-qa-v1',
    'baseline': 'ResidualMem-Instruct-Xbar-Input-RAG',
    'qformer': fingerprint(os.environ['RESIDUALMEM_QFORMER']),
    'retrieval_head': fingerprint(os.environ['RESIDUALMEM_RETRIEVAL_HEAD']),
    'codec': fingerprint(codec_path), 'mask': fingerprint(mask_path),
    'codec_metadata': meta,
    'anchor': False, 'reader_enable_thinking': False,
    'reader_max_new_tokens': 512, 'reader_do_sample': False, 'top_k': 10,
    'judge_model': 'Qwen3.5-9B', 'judge_temperature': 0,
    'judge_endpoints': ['http://127.0.0.1:8023/v1', 'http://127.0.0.1:8024/v1'],
    'reader_devices': ['cuda:7', 'cuda:1'], 'embedding_devices': ['cuda:7', 'cuda:7'],
    'reference': str((repo / '../WorldMemArena/exp_results/repro-route-a-20260907-final').resolve()),
    'expected_samples': 27, 'expected_qa': 1459,
    'changed_component': 'utility vector and its closed-loop codec reconstruction only',
}
(repo / 'gate/two-stage-v1/wma-qa/run_config.json').write_text(json.dumps(config, indent=2) + '\n')
print('Validated upgraded codec:', meta['closed_loop_gated_rate_bits'], 'bits/transition', flush=True)
PY
pids=()
devices=(7 1)
for rank in 0 1; do
  QWEN_EMBED_DEVICE="cuda:7" \
  RESIDUALMEM_DEVICE="cuda:${devices[$rank]}" RESIDUALMEM_HEAD_DEVICE="cuda:${devices[$rank]}" \
  OPENAI_BASE_URL_JUDGE="http://127.0.0.1:$((8023 + rank))/v1" \
  "$WMA_EVAL_PYTHON" -u -m experiments.state_tokenizer.run_wma_codec_eval_shard \
    --wma-root "$WMA_EVAL_ROOT" --shard-index "$rank" --num-shards 2 --workers 24 \
    --output "$WMA_EVAL_OUTPUT/shard-${rank}" \
    > "$WMA_EVAL_OUTPUT/logs/shard-${rank}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if [[ "$status" != 0 ]]; then
  echo "At least one WMA shard failed. Keep partial checkpoints; inspect logs." >&2
  exit 1
fi
"$WMA_EVAL_PYTHON" -m experiments.state_tokenizer.merge_wma_eval_shards \
  --wma-root "$WMA_EVAL_ROOT" --input "$WMA_EVAL_OUTPUT/shard-0" \
  --input "$WMA_EVAL_OUTPUT/shard-1" --output "$WMA_EVAL_OUTPUT/final"
