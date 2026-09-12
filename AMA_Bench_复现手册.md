# residual-mem · AMA-Bench 复现手册

更新时间：2026-09-12。本手册讲解如何在 AMA-Bench WEB 子集上复现 residual-mem 的全部结果：
latent memory/Q-Former 问答链路（最高 **38.33%**）与 Q-Former 码空间上的世界模型及任务效用门控
（**3773.7 bits/transition**、两阶段门控 **34.4% keep 下近乎无损**）。所有命令均在本仓库实际执行过，
产物路径与 sha256 见 §6 自检清单。

复现有两条主线，相互独立、可分别复现：

| 主线 | 内容 | 最终指标 |
|---|---|---|
| A. QA 部署链路 | Q-Former → 检索 → Bridge → Qwen3-32B Reader → LLM-as-Judge | 38.33%（360 题） |
| B. WM / 门控链路 | Q-Former 状态 → OPQ v2 码 → 世界模型 → 任务效用门控 | 3773.7 bits/transition；34.4% keep 下 \|ΔNLL\| 0.047 bit |

两条主线共用同一套 Q-Former 编码底座（`syqa/post-qformer/candidates/obs1.0-ce`），但**工作表征不同**：
A 线发送 raw Q-Former latents（bf16），B 线在 OPQ v2 离散码空间上工作，二者不可混用。

---

## 1. 环境与资源准备

### 1.1 仓库与全局设置

```bash
cd /home/cbd/project/residual-mem
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
```

**PYTHONPATH 规则（重要）**：仓库存在嵌套包 `xt_ama_adapter/xt_ama_adapter/`（adapter 正则包），
PYTHONPATH 写法按命令族二选一，写错会 ModuleNotFoundError：

| 命令族 | PYTHONPATH | 原因 |
|---|---|---|
| `python -m xt_ama_adapter.scripts.*`（§3.1 的 records/encode/quantize） | `PYTHONPATH="$PWD"` | 后缀 `:xt_ama_adapter` 会让内层包遮蔽外层，`xt_ama_adapter.scripts` 变得不可导入 |
| `python -m experiments.*` 且依赖 adapter 库（§2 QA 链路、§4 效用门控） | `PYTHONPATH=".:xt_ama_adapter"` | 需要 `import xt_ama_adapter.qwen32_bridge` 等，走内层正则包 |
| `python -m experiments.world_model.*`、`dump_wm_posteriors` | `PYTHONPATH="$PWD"` 即可 | 不 import adapter 库 |

### 1.2 Python 环境

| 环境 | Python | 用途 |
|---|---|---|
| `.venv-ama-vllm-qwen35` | `.venv-ama-vllm-qwen35/bin/python` | Q-Former 训练、伪 Web observation 生成、teacher/trunk cache、Judge vLLM 服务 |
| `.venv-ama-embedding-cu124` | `.venv-ama-embedding-cu124/bin/python` | Qwen3-32B Bridge/Reader、AMA Judge 客户端、世界模型（JAX 0.4.30）、效用门控 |

世界模型相关命令（JAX）必须携带库路径前缀，否则 DNN 初始化失败：

```bash
export LD_LIBRARY_PATH=$(echo /data2/xrz/miniforge3/envs/internnav/lib/python3.9/site-packages/nvidia/*/lib | tr ' ' ':')
```

### 1.3 基础模型权重

| 用途 | 模型 | 路径 |
|---|---|---|
| Q-Former trunk、QA teacher、伪 Web 生成 | Qwen3.5-9B | `/data1/models/Qwen3.5-9B` |
| 检索 query encoder | Qwen3-VL-Embedding-8B | `/data1/models/Qwen3-VL-Embedding-8B` |
| Bridge reader、答案生成、AMA Judge | Qwen3-32B | `/data1/models/Qwen3-32B` |

### 1.4 数据与基准

| 内容 | 路径 |
|---|---|
| AMA-Bench 代码与数据 | `third_party/AMA-Bench/`（评测文件 `dataset/test/open_end_qa_set.jsonl`） |
| HumanTrajs 原始数据 | `/data1/datasets/xt_ama_adapter/filtered/humantrajs-semantic-steps.jsonl`（486 轨迹 / 7,870 step） |
| SyQA 原始数据 | `/data1/datasets/xt_ama_adapter/filtered/molmoweb-syntheticqa-10k/` |
| 评测口径 | WEB 开放式 QA 子集，排除 episode 184（`--skip-episode-ids 184`），最终 **30 episodes / 360 题** |

### 1.5 GPU 与显存约定

- Qwen3-32B 以 fp32 加载时约 128 GB，通过 `device_map=auto` 摊到多卡；用
  `RESIDUALMEM_READER_MAX_MEMORY="34,37,34,66,63,36"` 按卡给出 GiB 上限（配合
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`）。
- 答案生成阶段 32B reader 双卡（`CUDA_VISIBLE_DEVICES=4,5`，`--max-memory-gib 34`）。
- Judge vLLM：3×Qwen3-32B 单卡 DP（GPU 3-5，`--max-model-len 8192`，port 8057）。
- 效用测量与世界模型训练/推理一律 **fp32**。

---

## 2. 主线 A：QA 部署链路（24.17% → 27.50% → 38.33%）

### 2.1 训练数据构建

三套数据共用同一 Q-Former 训练框架（脚本均在 `xt_ama_adapter/scripts/`）：

| 数据源 | 流水线脚本 | 产物 |
|---|---|---|
| HumanTrajs | `prepare_humantrajs_training_data.py` → `generate_pseudo_web_observation_vllm.py` → `generate_ama_state_qa_vllm.py` → `merge_dedupe_qa.py` → `build_humantrajs_qformer_inputs.py` | `outputs/ama_latent_memory/manifests/humantrajs-post-action-v1.jsonl`（6,750 状态）、`outputs/ama_latent_memory/qa/humantrajs-pseudo-web-v1.verified.jsonl`（3,371 QA） |
| WebChain 混合 | `build_webchain_qformer_inputs.py` → `merge_qformer_training_sources.py` → `make_pass_only_validation_pairs.py` | `outputs/ama_latent_memory/qformer/mixed-web-v1.pairs.npz`（1,740 pairs） |
| SyQA | 直接从 MolmoWeb SyntheticQA 构建 pairs | `outputs/ama_latent_memory/syqa/qformer-pairs-teacher-complete.npz`（11,944 QA / 6,645 observation） |

数据要点：时序对齐 `memory_i = action_i + sanitized(other_obs_i) + screenshot_(i+1)`；QA 经截图证据 +
伪文本证据双重过滤；instruction/action thought/未来帧一律不进入输入；train/val/test 按 trajectory 切分
且互斥；官方 AMA test 不进入任何训练或选点。

### 2.2 Q-Former 训练与冻结候选

训练入口：`experiments/state_tokenizer/train_qformer_joint`（启动脚本
`xt_ama_adapter/scripts/run_syqa_qformer_formal.sh`）。结构：4 层、hidden 1024、8 heads、**32 query × 512 维**。

```text
L_QFormer = CE_gold + 0.3×KL_q + w_obs×KL_obs + 1.0×InfoNCE_same_session(T=0.05)
```

- 每套数据跑 `w_obs=0.5/1.0` 两臂，最大 9,000 steps，lr 1e-4，grad-accum 4，seed 35。
- 选点标准：`validation answer CE`（正式链路用）与 `P4 observation gap`（诊断用），两者不可混用。
- 冻结候选（`run_syqa_post_qformer_bridge.sh` 前置步骤，`experiments/state_tokenizer/freeze_qformer_candidate.py`）：

```text
outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/
├── frozen.pt   # SyQA obs1.0 CE-best @7500，sha256 80c4e098…
└── head.pt     # 检索头，sha256 a6c1f451…
```

### 2.3 Bridge 训练（三个版本）

| Bridge | 训练数据 | 启动脚本 | 用途 |
|---|---|---|---|
| `qwen32-input-k32-rms.pt` | SyQA pairs | `run_syqa_post_qformer_bridge.sh` | 24.17% 基线 |
| `qwen32-input-k32-rms-delta-margin-total1000.pt`（sha `de13a5cd…`） | SyQA pairs + hidden distill + matched-vs-shuffled margin | `run_qwen32_delta_margin_total1000.sh` | 27.50% |
| `qwen32-input-k32-rms-top10.pt`（sha `30d3d9c2…`） | `bridge/humantrajs-80c4-retrieved-top10.npz`（HumanTrajs 检索 top-10 扩充） | 同 formal 配方（日志 `logs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-bridge-formal-top10.log`） | **38% 链路** |

Bridge 损失：`L = CE_gold + 0.30×KL_teacher`（delta-margin 版另加 `0.10×hidden_distill +
0.20×max(0, 0.10 + CE_matched − CE_shuffled)`），输出 soft token 做 RMS calibration 对齐 Qwen3-32B embedding。

### 2.4 官方评测三步（cache → answer → judge）

**第一步：状态 cache**（每 episode 一份 latents + step texts，`run_ama_web_latent cache-shard`）：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=".:xt_ama_adapter" \
  .venv-ama-vllm-qwen35/bin/python -u -m experiments.state_tokenizer.run_ama_web_latent cache-shard \
  --test-file third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl \
  --residualmem-root "$PWD" \
  --qwen35-model /data1/models/Qwen3.5-9B \
  --qformer  outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt \
  --retrieval-head outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt \
  --query-model /data1/models/Qwen3-VL-Embedding-8B \
  --output-dir outputs/ama_latent_memory/ama_eval/ablation-arms/cache-v2 \
  --device cuda:0 --shard-index 0 --shard-count 2 --skip-episode-ids 184
# shard 0/1 并行跑满 30 episodes；已复现目录即 cache-v2，可跳过
```

**第二步：答案生成**。基线臂（24.17%/27.50%）用 `run_ama_web_latent_formal.sh` /
`run_ama_web_delta_margin_total1000.sh`；**38% 配置**（latent+anchor，element 锚，k=8，RRF 混合检索）：

```bash
CUDA_VISIBLE_DEVICES=4,5 PYTHONPATH=".:xt_ama_adapter" \
  .venv-ama-embedding-cu124/bin/python -u -m experiments.state_tokenizer.run_ama_web_latent answer \
  --test-file third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl \
  --cache-dir outputs/ama_latent_memory/ama_eval/ablation-arms/cache-v2 \
  --qformer  outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt \
  --retrieval-head outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt \
  --bridge   outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-top10.pt \
  --reader-model /data1/models/Qwen3-32B \
  --checkpoint      outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10-anchor/element/k8-hybrid/answer-checkpoint.jsonl \
  --answers-output  outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10-anchor/element/k8-hybrid/answers.jsonl \
  --memory-mode latent+anchor --anchor-style element \
  --top-k-latent 8 --top-k 8 --retrieval hybrid \
  --enable-thinking true --max-model-len 32000 --max-new-tokens 8192 \
  --batch-size 4 --max-memory-gib 34 --skip-episode-ids 184
```

要点：检索为 RRF 混合（semantic + lexical BM25），**top-8**；latent 行 + element 锚文本行同时进 prompt；
`enable_thinking=True`。

**第三步：Judge**（官方 `evaluate.py` + 本地 vLLM Qwen3-32B judge）：

```bash
JDIR=element/k8-hybrid bash xt_ama_adapter/scripts/run_ama_anchor_element_judge.sh
# judge 配置：xt_ama_adapter/configs/ama_web_judge_gpu3.yaml（GPU 3-5，port 8057）
# 结果：outputs/.../element/k8-hybrid/evaluation.json
```

---

## 3. 主线 B：世界模型（OPQ v2 码空间）

### 3.1 AMA 状态编码与 v2 码本重量化

```bash
PY=.venv-ama-embedding-cu124/bin/python
AMA=outputs/wm_train/ama-v1

# (1) 从 AMA-Bench 数据构建 records/states（只取 train+validation split，test 不进入）
PYTHONPATH="$PWD" $PY -m xt_ama_adapter.scripts.build_ama_wm_records \
  --dataset third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl \
  --output-dir $AMA --include-splits train validation
# → 208 episodes / 14,944 状态（10,314 唯一）/ records.jsonl + states.jsonl

# (2) 状态编码（Qwen3.5-9B trunk layer-16 + Q-Former），5 个 shard 并行
for k in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$k PYTHONPATH="$PWD" $PY -m xt_ama_adapter.scripts.encode_wm_text_states \
    --states-jsonl $AMA/states.jsonl --residualmem-root "$PWD" \
    --qwen35-model /data1/models/Qwen3.5-9B \
    --qformer outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt \
    --retrieval-head outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt \
    --output $AMA/ama-states-$k.npz --shard-index $k --shard-count 5 &
done; wait

# (3) 用 WebWorld-v2 拟合好的 OPQ 码本重量化（不重新拟合，保证同一码空间）
PYTHONPATH="$PWD" $PY -m xt_ama_adapter.scripts.quantize_states_pq \
  --states $AMA/ama-states-0.npz --states $AMA/ama-states-1.npz --states $AMA/ama-states-2.npz \
  --states $AMA/ama-states-3.npz --states $AMA/ama-states-4.npz \
  --pq outputs/wm_train/webworld-v2/pq-h32-c64.npz \
  --output $AMA/ama-pq-c64-v2codebook.npz

# (4) 离散 WM 缓存 + 免训练基线
PYTHONPATH="$PWD" $PY -m experiments.world_model.cache_web \
  --codes $AMA/ama-pq-c64-v2codebook.npz --records $AMA/records.jsonl \
  --output $AMA/cache-h4 --max-history 4 --include-splits train validation
PYTHONPATH="$PWD" $PY -m experiments.world_model.baselines \
  --cache $AMA/cache-h4 --output $AMA/baselines-h4
```

码空间：32 slot × 32 子空间，C=64，定宽 **6144 bit/状态**。权威码表
`ama-pq-c64-v2codebook.npz`（含 rotation，携带 `codes/all` + `state_ids/all`）。
注：早期 `ama-pq-c64.npz` 为 v1 码表，与 WM 的 v2 码空间重合率仅 1.56%（随机水平），已作废——
2026-09-11 之前基于它的全部 |U| 标签与 λ 曲线臂均已重跑。

### 3.2 单文件训练数据（发布形态）

```bash
PYTHONPATH="$PWD" $PY -m experiments.world_model.build_amabench_wm_train \
  --output outputs/release/amabench_wm_train.npz
```

- 合并 WebWorldData v2（train+validation，反遗忘参照）与 **AMA train split only**；
  AMA validation/test 一律排除，评测走官方 ama-bench。
- **条数**：共 268,799 transitions = WebWorld-v2 256,034（train 230,774 + validation 25,260）
  + AMA train 12,765；batch 32 内 16+16 混采（`--mix-ratio 0.5`，读 npz 内 `t_corpus_ids`）。
- 产物 `outputs/release/amabench_wm_train.npz`（约 536 MB）+ `amabench_wm_train.sha256`；
  离线第三方复现可直接用 `outputs/release/wm_training_bundle_v1.tar.gz`
  （含双语料 cache、码本 provenance、README）。

### 3.3 世界模型训练（混合语料 v2amamix）

```bash
export LD_LIBRARY_PATH=$(echo /data2/xrz/miniforge3/envs/internnav/lib/python3.9/site-packages/nvidia/*/lib | tr ' ' ':')
PYTHONPATH="$PWD" $PY -u -m experiments.world_model.train \
  --cache outputs/release/amabench_wm_train.npz \
  --config configs/world_model/webworld_v2_c64.yaml \
  --variant full --seed 0 \
  --init-from outputs/wm_train/webworld-v2/runs/full_seed0_c64_v2fix/best.pkl \
  --mix-ratio 0.5 \
  --selection-cache outputs/wm_train/ama-v1/cache-h4 \
  --batch-size 32 --learning-rate 1e-4 --max-steps 40000 \
  --output outputs/wm_train/webworld-v2/runs/full_seed0_c64_v2amamix_repro
```

- 混合采样：每个 batch（32）内 WebWorld : AMA = 16 : 16（`--mix-ratio 0.5`，读 npz 内
  `t_corpus_ids`）。
- warm-start：init 自 `v2fix best`；`--selection-cache` 指向 AMA val（model selection 用，
  非 test）。
- 仓库内原始 run 为 `full_seed0_c64_v2amamix`（直接用 cache pair：
  `--cache outputs/wm_train/webworld-v2/cache-h8 --mix-cache outputs/wm_train/ama-v1/cache-h4`），
  单文件复现 run 为 `full_seed0_c64_v2amamix_repro`，两者指标一致（§5.2）。

### 3.4 锚定评测与后验导出

```bash
# 锚定评测（AMA val，1,971 transitions / 28 episodes）
PYTHONPATH="$PWD" $PY -m experiments.world_model.final_eval \
  --checkpoint outputs/wm_train/webworld-v2/runs/full_seed0_c64_v2amamix/best.pkl \
  --config configs/world_model/webworld_v2_c64.yaml \
  --cache outputs/wm_train/ama-v1/cache-h4 \
  --output outputs/utility_gate_ama/final_eval/report.json

# WM 后验导出（门控输入；产物 .json 内含 verification 块，对锚定 best_selection 校验）
PYTHONPATH="$PWD" $PY -m experiments.utility_gate.dump_wm_posteriors \
  --cache outputs/wm_train/ama-v1/cache-h4 \
  --config configs/world_model/webworld_v2_c64.yaml \
  --split validation \
  --checkpoint outputs/wm_train/webworld-v2/runs/full_seed0_c64_v2amamix/best.pkl \
  --output outputs/utility_gate_ama/posteriors-val.npz
# train split 同法 → posteriors-train.npz
```

---

## 4. 主线 B 续：任务效用门控（两阶段 mask）

门控定义：位置 j 的任务效用 `U_j = [L_ans(q,a|x̂^(-j)) − L_ans(q,a|x̂^full)] / ln2`（量化空间内差分，
OPQ 自身误差在差分中抵消）；keep 规则 `max(|U|₁, U_cond) ≥ λ·H_j`。**λ 工作点 = 3.584e-04**，
效用向量（1024 维常量）随模型发布，H_j 解码端自算，mask 开销为零。

### 4.1 stage-1 |U| 反事实标注（fp32 硬约束）

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5 \
  RESIDUALMEM_READER_MAX_MEMORY="34,37,34,66,63,36" \
  RESIDUALMEM_FORCE_EFFICIENT_ATTN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=".:xt_ama_adapter" \
  .venv-ama-embedding-cu124/bin/python -m experiments.utility_gate.label_official \
  --pairs outputs/utility_gate_ama/pairs-official.jsonl \
  --posteriors outputs/utility_gate_ama/posteriors-train.npz --split train \
  --prompt latent+anchor --anchor-k 8 --anchor-style element \
  --bridge outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-top10.pt \
  --dtype float32 --positions 16 --max-rows 333 \
  --output outputs/utility_gate_ama/labels-fit-anchor-v2.npz
```

协议要点：与部署同 bridge（sha 锁定）、同 prompt 口径（latent+anchor / element / k8）；null 对照
（同 batch 嵌入两次，误差 >1e-4 bit 即报错）；每行带 dtype 审计。dtype 实测（smoke）：
fp32 \|ΔNLL\| median 0.0008 bit（信号量级，可用）；bf16 median 0.15 bit、argmax 改变 33%（噪声 ≈
信号 ×200，不可用）。

### 4.2 两阶段包络（fit-mask → prepare → label → export）

```bash
TS=outputs/utility_gate_ama/two-stage-v1
PY=.venv-ama-embedding-cu124/bin/python

# (1) stage-1 初始 mask
PYTHONPATH=".:xt_ama_adapter" $PY -m experiments.utility_gate.two_stage_mask_ama fit-mask \
  --labels outputs/utility_gate_ama/labels-fit-anchor-v2.npz \
  --lambda-value 3.584e-04 --output $TS/mask-stage1-lambda3.584e-04.json

# (2) 第二遍标注计划（train 状态 × 每状态 1 题）
PYTHONPATH=".:xt_ama_adapter" $PY -m experiments.utility_gate.two_stage_mask_ama prepare \
  --mask $TS/mask-stage1-lambda3.584e-04.json \
  --max-states 96 --max-questions-per-state 1 \
  --bridge outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-top10.pt \
  --output $TS/plan-1q.json

# (3) 条件效用标注（完整 drop 背景中逐位置恢复；world_size=3 并行，rank=0/1/2）
CUDA_VISIBLE_DEVICES=0,1,2 RESIDUALMEM_READER_MAX_MEMORY="34,37,34,66,63,36" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=".:xt_ama_adapter" \
  $PY -m experiments.utility_gate.two_stage_mask_ama label \
  --plan $TS/plan-1q.json --rank 0 --world-size 3 --max-prompt-tokens 1000 \
  --output $TS/labels/rank-0

# (4) 导出包络 mask
PYTHONPATH=".:xt_ama_adapter" $PY -m experiments.utility_gate.two_stage_mask_ama export \
  --plan $TS/plan-1q.json --labels $TS/labels \
  --val-posteriors outputs/utility_gate_ama/posteriors-val.npz \
  --output $TS/mask-two-stage.npz
```

### 4.3 开环验证（λ 曲线 + 对照臂）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 RESIDUALMEM_READER_MAX_MEMORY="34,37,34,66,63,36" \
  RESIDUALMEM_FORCE_EFFICIENT_ATTN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=".:xt_ama_adapter" \
  .venv-ama-embedding-cu124/bin/python -u -m experiments.utility_gate.curve_official \
  --fit-labels outputs/utility_gate_ama/labels-fit-anchor-v2.npz \
  --pairs outputs/utility_gate_ama/pairs-official.jsonl \
  --posteriors outputs/utility_gate_ama/posteriors-val.npz \
  --utility-override outputs/utility_gate_ama/two-stage-v1/mask-two-stage.npz \
  --bridge outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-top10.pt \
  --lambdas 3.584e-04 --max-rows 100 --prompt latent+anchor \
  --arms random displacement rate \
  --output outputs/utility_gate_ama/curve-two-stage-val.json
```

- `--utility-override` 接入两阶段包络向量；不带该参数时从 fit 标签拟合 stage-1 |U|。
- `--arms` 选对照臂（random / displacement / rate，等预算配对）；赶时间可 `--arms none`。
- 码率轴在全部 val 状态上统计（`val_states_with_posterior = 1203`），质量轴只在带官方问题的
  100 行上测——两轴刻意不同集（报告 §0 口径）。

---

## 5. 实验结果汇总

### 5.1 QA 链路（官方 360 题，Qwen3-32B LLM-as-Judge）

消融臂（同一 reader/bridge/cache）：

| 臂 | 配置 | accuracy |
|---|---|---:|
| web-latent-formal | 旧 exploratory bridge，k=1 | 24.17% |
| latent-only-top10 / k5 | rank-10 bridge，semantic k5 | 25.56% |
| latent-only-top10-anchor / element / k8 | latent+anchor，semantic k8 | 38.06% |
| **latent-only-top10-anchor / element / k8-hybrid** | **+ RRF 混合检索** | **38.33%** |

Delta-margin Bridge 复测（SyQA 链路，top_k=1）：

| Bridge | 正确数 | 准确率 |
|---|---:|---:|
| 基础（CE + 0.30×KL） | 87/360 | 24.17% |
| + hidden distill + CE margin | 99/360 | 27.50% |

按题型（基础 → delta-margin）：A 25.62%→26.45%，B 32.22%→37.78%，C 16.67%→21.11%，D 20.34%→23.73%。
逐题配对：31 对→错、19 对→错。

### 5.2 世界模型（AMA val：1,971 transitions / 28 episodes）

| 方法 | bits/transition | 相对 WM |
|---:|---:|---:|
| marginal | 4571.23 | +797.5 |
| copy（前码复制） | 4060.49 | +286.8 |
| source（源表回退） | 3813.88 | +40.1 |
| **WM（v2amamix）** | **3773.73** | — |

- WM 众数命中率 **38.61%**（train 36.71%）；定宽 6144 bit → 压缩比 **1.63×**。
- 逐轴：kept 轴（38.35%）WM **1.08** bit/轴（source 1.20 / copy 1.40）；changed 轴（61.65%）WM
  **5.31**（source 5.33 / copy 5.56）。
- 单文件复现 run（`v2amamix_repro`）：3795.22 bits/transition，code accuracy 37.49%——与原 run 一致量级。
- 锚定：`final_eval` 3773.73；`run.json best_selection` 3773.95；后验 dump 与之相差 5.0e-5 bit（容差内）。

### 5.3 任务效用门控（λ = 3.584e-04，latent+anchor 口径，fp32）

stage-1 曲线（100 val 官方问题；\|ΔNLL\| 为答案 NLL 变化，bit）：

| λ | keep | 码率 | \|ΔNLL\| | random 臂 | rate 臂 |
|---:|---:|---:|---:|---:|---:|
| 3.475e-05 | 34.0% | 1360 | 0.338 | 0.588 | 0.392 |
| 3.584e-04 | 9.5% | 389 | 0.415 | 0.781 | 0.737 |
| 8.625e-04 | 5.5% | 222 | 0.531 | 0.805 | 0.757 |

两阶段包络（最终方案）开环验证：

| 方法 | keep | 码率 | \|ΔNLL\| 均值 | 中位 |
|---|---:|---:|---:|---:|
| 全发 | 100% | 3868.6 | 0 | 0 |
| stage-1 gate（同 λ） | 9.5% | 389 | 0.415 | — |
| **两阶段 gate（同 λ）** | **34.4%** | **1325.6** | **0.047** | **0.017** |
| OPQ 量化自身（参照） | — | — | 0.148 | — |

结论：同 λ 下 \|ΔNLL\| 0.415 → 0.047（8.8×），低于 OPQ 量化自身噪声，码率仅为全发的 34%
（压缩 4.25×）；等预算（34%）下比孤立反事实 gate 好 7.2×。两阶段 rescue 在 train 语料上救回
24.8% 初始 drop（keep 17.2% → 37.6%），mask agreement 79.5%，零新增 drop。

未决：两阶段 gate 的对照臂（random / displacement / rate，约 2 GPU·h）未跑；stage-1 曲线中
displacement 臂在 34% 预算处出现 0.372 → 0.024 的异常悬崖，且等预算下 \|ΔNLL\| 0.024 优于两阶段
gate 的 0.047，悬崖成因与该臂可部署性待查。

### 5.4 边界与声明

- 开环 \|ΔNLL\| 的 eval 分布与 |U| fit 分布同为生成式局部 QA（train/val 状态分切）；官方 val 问题
  本地不存在（`third_party/AMA-Bench/dataset/` 仅 test/），泛化到官方问题分布的最终验证留给官方端到端评测。
- 表征 A（raw latents）与表征 B（v2 码空间）不可混用；closed-loop 若把 OPQ 解码态塞进 QA 链路，
  需先重新验证 bridge 兼容性。
- 效用向量（1024 维常量）随模型发布；H_j 解码端自算，mask 开销为零。

### 5.5 结果文件索引

| 结果 | 文件 |
|---|---|
| 38% 评测 | `outputs/ama_latent_memory/ama_eval/ablation-arms/latent-only-top10-anchor/element/k8{,-hybrid}/evaluation.json` |
| 27.50% 评测 | `outputs/ama_latent_memory/ama_eval/web-latent-delta-margin-total1000/evaluation.json` |
| 24.17% 评测 | `outputs/ama_latent_memory/ama_eval/web-latent-formal/evaluation.json` |
| WM 锚定评测 | `outputs/utility_gate_ama/final_eval/report.json`；`outputs/wm_train/webworld-v2/runs/full_seed0_c64_v2amamix/run.json` |
| WM 基线 | `outputs/wm_train/ama-v1/baselines-h4/baseline.json` |
| stage-1 曲线 | `outputs/utility_gate_ama/curve-v2.npz` + `.report.json` |
| 两阶段开环 | `outputs/utility_gate_ama/curve-two-stage-val.json.npz` + `.report.json` |
| 两阶段 mask | `outputs/utility_gate_ama/two-stage-v1/mask-two-stage.npz`（+ `.json` 元数据） |

---

## 6. 复现自检清单（sha256 锚定）

复现后逐项比对（前 16 位即可）：

| 产物 | sha256 前 16 位 | 校验位置 |
|---|---|---|
| Q-Former frozen.pt | `80c4e09851ad91b6` | cache-v2 episode 元数据 `qformer_sha256` |
| 检索头 head.pt | `a6c1f451e20651d2` | cache-v2 episode 元数据 `retrieval_head_sha256` |
| 部署 bridge（top10） | `30d3d9c28d068cac` | `answer-checkpoint.jsonl` 的 `bridge_sha256` |
| delta-margin bridge | `de13a5cdc6668949` | 对 `outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-delta-margin-total1000.pt` 现算 |
| 权威码表 ama-pq-c64-v2codebook.npz | （随 cache manifest） | `cache_manifest_sha256 = c6bb6dfa…`（WM cache-h4） |
| 单文件训练数据 amabench_wm_train.npz | `f46a72b5d827326c`（manifest） | `v2amamix_repro/run.json` |
| 两阶段计划 plan-1q.json | `ab1535a063946159` | `mask-two-stage.json` 的 `plan_sha256` |

行为自检：

- [ ] 评测集 = 30 episodes / 360 题（episode 184 已排除），test split 未进入任何训练/选点；
- [ ] AMA WM 训练数据 = train split only（validation 只用于 model selection 与开环评测）；
- [ ] 效用标注 null 对照全程 0.0 bit，dtype 为 fp32；
- [ ] 38% 链路逐题记录含 `bridge_sha256 = 30d3d9c2…`、`memory_mode = latent+anchor`、k=8；
- [ ] Judge 使用官方 `evaluate.py`，API key 非空（避免静默返回 gold answer）。
