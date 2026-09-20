# WMA-ResidualMem 复现手册

> 链路：**观察 → Q-Former → OPQ 量化 → 世界模型条件编码 → e·V 效用门控
> → 闭环重建 → 检索 → Reader**。在 WorldMemArena Web 上，codec 把每状态
> 6144 bit 的定宽记忆压到 4333.4 bit；端到端 Reader 评测为
> **QA-C 0.6018（878/1459）**。方法与结论见 `技术报告_ResidualMem.md`。

---

## 1. 部署目录与环境

把 ResidualMem 与带有 ResidualMem adapter 的 WMA 评测器放在同一目录：

```text
workspace/
├── residual-mem/       # 本仓库、发布产物、codec 与评测脚本
└── WorldMemArena/      # 发布页指定的 ResidualMem-compatible WMA checkout
    └── WorldMemArena/  # Hugging Face 下载的官方数据集
```

WMA 评测器必须是发布页锁定的兼容版本：以官方 WorldMemArena 为基础，额外注册
`ResidualMem-Instruct-Xbar-Input-RAG` adapter 并支持 `--subcategory`。未经适配的
upstream checkout 不包含该 baseline，不能直接执行本文命令。

```bash
WORK=/path/to/workspace
REPO=$WORK/residual-mem
WMA=$WORK/WorldMemArena

cd "$WMA"
cp .env.example .env
```

ResidualMem 使用两个隔离环境：

```bash
cd "$REPO"

# codec / world model
python -m venv .venv-jax
.venv-jax/bin/python -m pip install -r requirements.txt

# Q-Former / retrieval / Reader / WMA
conda create -n residualmem-wma python=3.10 -y
conda activate residualmem-wma
python -m pip install -r "$WMA/requirements.txt" tenacity
# 再安装发布页锁定的 PyTorch、Transformers 与 Qwen3.5 依赖版本
```

本文结果验证过的 Torch 环境为 PyTorch 2.7.1、Transformers 5.14.1、
SentenceTransformers 6.0.0、`qwen-vl-utils` 0.0.14；正式发布的 environment lock
优先于这里的版本摘要。

统一变量：

```bash
cd "$REPO"
JX=$REPO/.venv-jax/bin/python
PY=/path/to/miniconda/envs/residualmem-wma/bin/python
D=$REPO/outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar
DATA=/path/to/residualmem-artifacts/molmoweb-pilot
export RESIDUALMEM_DATA="$DATA"
export PYTHONPATH="$REPO:$WMA" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TRITON_CACHE_DIR=/path/to/a/writable/triton-cache
```

在 Torch 环境中下载官方 WMA 数据并检查兼容 evaluator：

```bash
cd "$WMA"
huggingface-cli download LCZZZZ/WorldMemArena --repo-type dataset \
  --local-dir ./WorldMemArena

"$PY" -c "from eval_framework.memory_adapters.residualmem_instruct_adapter import ResidualMemInstructAdapter"
"$PY" -m eval_framework.cli --help | rg -- '--subcategory'
test "$(find WorldMemArena/agent/gui/web -maxdepth 1 -name '*.json' | wc -l)" -eq 27
```

| 谁用哪个 | |
|---|---|
| `$JX` | 世界模型（cache）、OPQ、门控码率、manifest |
| `$PY` | Q-Former、reader、检索头 |

☠️ 两个解释器互相看不见是预期行为；不要在一个解释器中同时 import 两套运行时。
☠️ **OMP 线程必须限制**，否则 GPU 推理慢 5 倍。☠️ `TRITON_CACHE_DIR` 必须指向
空间充足的可写目录。

---

## 2. 发布产物

从 Hugging Face 下载（`feldmatthew/WMA-ResidualMem`）并按下列位置摆放：

```bash
cd "$REPO"
huggingface-cli download feldmatthew/WMA-ResidualMem --local-dir hf
mkdir -p "$D"
cp hf/weights/qformer-K32e-obs0.5.gapbest.pt \
   hf/weights/head-K32e-obs0.5-gapbest-utility-gated.pt "$D/"
cp hf/weights/world_model-best.pkl "$REPO/run/best.pkl"
cp hf/codec/opq-shared-mix10-M32-C64.npz "$DATA/pq-full/"
cp hf/codec/ev-vectors.npz "$REPO/gate/"
cp hf/configs/web_h16_C64_full.yaml "$REPO/configs/world_model/"

# 哈希校验：所有 runtime 项均为 ok 才继续
"$JX" -m residualmem.manifest
```

| 角色 | 路径 |
|---|---|
| Q-Former | `$D/qformer-K32e-obs0.5.gapbest.pt`（step 6000） |
| 检索头 | `$D/head-K32e-obs0.5-gapbest-utility-gated.pt` |
| OPQ 码本 C=64 | `$DATA/pq-full/opq-shared-mix10-M32-C64.npz` |
| 世界模型 | `run/best.pkl`（60k 步） |
| e·V 效用门控 | `gate/ev-vectors.npz`（判决规则 `e_{t,j}·V_j ≥ λ·H_{t,j}`，λ=5.798041e-03） |

全流程评测生成两个派生产物：

| 角色 | 路径 |
|---|---|
| 闭环门控重建 | `testset/wma-web-closedloop-ev-v1.npz` |
| 端到端 QA aggregate | `../WorldMemArena/exp_results/repro-ev-20260919-final/aggregate_metrics.json` |

☠️ **这个仓库最危险的一处**：`pq/` 与 `pq-full/` 下有**完全同名**的
`opq-shared-mix10-M32-C64.npz`，是同一配置拟合的两次。只有 `pq-full/` 那本与世界模型
训练数据内嵌的解码器逐位一致。用错另一本会产出合法的 `(32,32) uint8` 码、缓存能建、
模型能打分——**只是所有数字差约 972 bit，且全程无任何报错**。
**只有 manifest 的哈希校验能拦住**——改名字或删文件都不行，因为出问题的正是「同名」这件事。

---

## 3. 复现评测

数据流（从官方 WMA Web JSON 开始，不依赖预生成的 `testset/`）：

```text
WMA Web JSON
  → Q-Former xbar → 冻结 OPQ code → WM + e·V 门控闭环重建
  → WMA session memory → 检索 top-10 → soft-token Reader → WMA Judge
```

### 3.1 配置 Reader 与 Judge

Reader 与 Judge 均为本地 Qwen3.5-9B（`$REPO/models/Qwen3.5-9B`），不需要任何
OpenAI API。Reader 由评测命令里的 `QWEN35_MODEL` 指定；Judge 用仓库自带的
OpenAI-compatible 本地 server（占用 4 张 GPU 做双副本，示例用 2/3 两卡，避开
评测要用的 0/1）：

```bash
cd "$REPO"
QWEN_PY="$PY" MODEL="$REPO/models/Qwen3.5-9B" GPUS=2,2,3,3 PORT=8017 \
  bash scripts_local_llm_server.sh start
curl -fsS http://127.0.0.1:8017/v1/models
```

然后编辑 `$WMA/.env`，把 Judge 指向该 server：

```bash
OPENAI_API_KEY_JUDGE=local
OPENAI_BASE_URL_JUDGE=http://127.0.0.1:8017/v1
OPENAI_MODEL_JUDGE=<`/v1/models` 返回的第一项>
```

### 3.2 从官方 WMA Web 数据构建 codec 输入

```bash
cd "$REPO"
CUDA_VISIBLE_DEVICES=0 "$JX" scripts_build_wm_testset.py \
  --dataset-repo "$WMA/WorldMemArena" \
  --checkpoint "$D/qformer-K32e-obs0.5.gapbest.pt" \
  --codebook "$DATA/pq-full/opq-shared-mix10-M32-C64.npz" \
  --output testset --jax-python "$JX" --torch-python "$PY" --device cuda:0
```

该命令顺序执行 `convert_wma → Q-Former encode → OPQ apply → cache_web`。完成后检查：

```bash
$PY - <<'PY'
import json
p = json.load(open("testset/provenance.json"))
assert p["dataset_repo"]["samples"] == 27
assert p["counts"] == {"states": 956, "transitions": 817, "episodes": 139}
print(p["counts"])
PY
```

### 3.3 物化闭环 WM + e·V 门控重建

```bash
CODEC=$REPO/testset/wma-web-closedloop-ev-v1.npz
CUDA_VISIBLE_DEVICES=0 "$JX" -m \
  experiments.state_tokenizer.materialize_wma_codec_reconstructions \
  --cache testset/cache --states testset/states.npz --records testset/records.jsonl \
  --codebook "$DATA/pq-full/opq-shared-mix10-M32-C64.npz" \
  --config configs/world_model/web_h16_C64_full.yaml \
  --checkpoint run/best.pkl --mask gate/ev-vectors.npz \
  --split validation --platform gpu --device-index 0 --output "$CODEC"
```

episode 初态发送完整 OPQ 码；后续状态以已经重建的历史计算 WM posterior，满足
`e_{t,j}·V_j < λ·H_{t,j}` 的位置由 posterior argmax 填充，并把新重建继续写入下一步
历史。metadata 必须为 956 个状态、817 条转移、139 个 all-send 初态、
`lambda=5.798040571718835e-03`、闭环码率 `4333.363682216191` bit、keep fraction
`0.8356960488`、R² `0.7636862392`。

### 3.4 先跑一条 WMA smoke

```bash
cd "$REPO"
CODEC=$REPO/testset/wma-web-closedloop-ev-v1.npz
unset ANCHOR_RECORDS RESIDUALMEM_WMA_ANCHOR_RECORDS
QFORMER_PYTHON="$PY" \
WMA_ROOT="$WMA" QFORMER_BRIDGE_DIR="$D" \
QWEN35_MODEL="$REPO/models/Qwen3.5-9B" \
CODEC_RECONSTRUCTIONS="$CODEC" \
RETRIEVAL_HEAD_ARM=K32e-obs0.5-gapbest-utility-gated \
OUTPUT_ARM=K32e-obs0.5-gapbest-closedloop-smoke \
WORKERS=4 SMOKE=1 \
bash scripts_qformer_official_eval.sh \
  K32e-obs0.5-gapbest qformer-K32e-obs0.5.gapbest.pt 0
```

确认输出中 baseline 为 `ResidualMem-Instruct-Xbar-Input-RAG`，codec 路径非空，且每条
eval 都没有 `error`。若 QA-C 异常接近 1.0，先检查 Judge 配置，不要继续正式运行。

### 3.5 正式评测：双卡分片

正式冻结结果使用两张 GPU 对 27 个 Web 样本做奇偶分片。先在两个终端共享以下环境：

```bash
CODEC=$REPO/testset/wma-web-closedloop-ev-v1.npz
export PYTHONPATH="$REPO:$WMA"
export RESIDUALMEM_ROOT="$REPO"
export RESIDUALMEM_QWEN35_MODEL="$REPO/models/Qwen3.5-9B"
export RESIDUALMEM_QFORMER="$D/qformer-K32e-obs0.5.gapbest.pt"
export RESIDUALMEM_QFORMER_QUERIES=32 RESIDUALMEM_QFORMER_LAYERS=4
export RESIDUALMEM_RETRIEVAL_HEAD="$D/head-K32e-obs0.5-gapbest-utility-gated.pt"
export RESIDUALMEM_CODEC_RECONSTRUCTIONS="$CODEC"
export QWEN_VL_EMBED_LOCAL=1 LLM_MAX_CONCURRENT=24
unset RESIDUALMEM_WMA_ANCHOR_RECORDS
```

两个终端分别执行：

```bash
# GPU 0：web_01, web_03, ..., web_27
QWEN_EMBED_DEVICE=cuda:0 RESIDUALMEM_DEVICE=cuda:0 RESIDUALMEM_HEAD_DEVICE=cuda:0 \
"$PY" -u -m experiments.state_tokenizer.run_wma_codec_eval_shard \
  --wma-root "$WMA" --shard-index 0 --num-shards 2 --workers 24 \
  --output "$WMA/exp_results/repro-ev-20260919-shard0"

# GPU 1：web_02, web_04, ..., web_26
QWEN_EMBED_DEVICE=cuda:1 RESIDUALMEM_DEVICE=cuda:1 RESIDUALMEM_HEAD_DEVICE=cuda:1 \
"$PY" -u -m experiments.state_tokenizer.run_wma_codec_eval_shard \
  --wma-root "$WMA" --shard-index 1 --num-shards 2 --workers 24 \
  --output "$WMA/exp_results/repro-ev-20260919-shard1"
```

两路完成后合并：

```bash
"$PY" -u -m experiments.state_tokenizer.merge_wma_eval_shards \
  --wma-root "$WMA" \
  --input "$WMA/exp_results/repro-ev-20260919-shard0" \
  --input "$WMA/exp_results/repro-ev-20260919-shard1" \
  --output "$WMA/exp_results/repro-ev-20260919-final"
```

合并器硬校验两片不重叠、样本数为 27、有效 QA 数为 1459，且每条 eval 无 error。
最终读取 `repro-ev-20260919-final/aggregate_metrics.json`。

---

## 4. 要复现出的结果

| 指标 | 目标值 |
|---|---:|
| **QA-C** | **0.6018**（878/1459） |
| QA-H（幻觉） | 0.1892（276/1459） |
| QA-O（遗漏） | 0.2090（305/1459） |
| RC hit rate | 0.6764（3486/4914） |
| Recall@1 / @5 / @10 | 0.2242 / 0.5328 / 0.6988 |
| 闭环码率 / keep fraction | 4333.36 bit / 0.8357（压缩比 1.418×） |

该结果使用 OPQ 重建的初态和「已重建历史 → WM 后验 → e·V 门控 → WM argmax
填充」的递归后续状态，再经双视图检索头与 soft-token Reader（Qwen3.5-9B）。
`CODEC_RECONSTRUCTIONS` 使 adapter 按官方 `image_id/state_id` 读取 `gated_xbar`，
路径 fail-closed：键缺失、shape 不匹配或记录重复时直接报错，不会静默退回 full-xbar。

---

## 5. 复现硬约束

| | |
|---|---|
| **Web 评测集不进训练** | 27 个 `agent/gui/web` 样本不得进入任何组件的梯度训练；WM 只用它们选点 |
| **Q-Former shape 必须一致** | 编码、gap 与 QA 都使用 32 queries / 4 layers；多个 CLI 默认是 16 queries |
| **先做单样本 QA smoke** | Judge 配置缺失时，历史评测路径曾回退到 gold answer，QA-C 接近 1.0 应立即停止检查 |
| **OPQ 码本哈希** | 只能使用 `pq-full/` 下 sha256 为 `bcc4c393…` 的码本；`pq/` 下同名文件属于另一次拟合 |
| **WM cache 历史长度** | cache 保存 32 步，模型只用 16 步；评测必须将 `cache.max_history` 设为 config 中的 16 |
| **zsh 不做词分割** | `L="--lambdas 0.0009 0.0010"` 后 `$L` 会被当成**单个参数**；参数写死，或走 `xargs` |
| **`nohup` 扛不住 SIGTERM** | 工具超时会杀掉整个进程组，长任务用 `setsid` |

---

## 6. 相关文档

| | |
|---|---|
| `技术报告_ResidualMem.md` | 方法、全部实测结果 |
| `UTILITY_GATE.md` | e·V 效用门控：推导、闭环 |
