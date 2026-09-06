# AMA 测评实验资源与环境说明

更新时间：2026-09-02

本文档用于说明 AMA-WEB latent memory/Q-Former 实验中模型权重、训练数据、中间产物和运行环境的位置。路径以当前机器目录结构为准，仓库根目录为：

```text
/home/cbd/project/residual-mem
```

## 1. 模型权重位置

### 1.1 基础模型

| 用途 | 模型 | 路径 |
|---|---|---|
| Q-Former trunk、QA teacher、伪 Web observation 生成 | Qwen3.5-9B | `/data1/models/Qwen3.5-9B` |
| Qwen3.5 base 备份/对照 | Qwen3.5-9B-Base | `/data1/models/Qwen3.5-9B-Base` |
| Q-Former fused/query teacher、检索 query encoder | Qwen3-VL-Embedding-8B | `/data1/models/Qwen3-VL-Embedding-8B` |
| 32B Bridge reader、答案生成、AMA Judge | Qwen3-32B | `/data1/models/Qwen3-32B` |

### 1.2 当前实验 checkpoint

#### HumanTrajs Q-Former

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/qformer/
├── qformer-K32-humantrajs-obs0.5.pt
├── qformer-K32-humantrajs-obs0.5.gapbest.pt
├── qformer-K32-humantrajs-obs1.0.pt
├── qformer-K32-humantrajs-obs1.0.gapbest.pt
└── qformer-K32-humantrajs-obs{0.5,1.0}.json
```

`obs0.5/obs1.0` 表示 `KL_obs` loss 的权重；普通 `.pt` 为 validation CE 选点，`.gapbest.pt` 为 P4 observation gap 选点。

#### WebChain Q-Former

scratch 初始 checkpoint：

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/qformer/
├── qformer-K32-webchain-h500-v1-scratch-obs0.5.pt
├── qformer-K32-webchain-h500-v1-scratch-obs0.5.gapbest.pt
├── qformer-K32-webchain-h500-v1-scratch-obs1.0.pt
└── qformer-K32-webchain-h500-v1-scratch-obs1.0.gapbest.pt
```

pass-only validation continuation checkpoint：

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/qformer/
├── qformer-K32-webchain-h500-passval-v1-cached-from250-obs0.5.pt
├── qformer-K32-webchain-h500-passval-v1-cached-from250-obs0.5.gapbest.pt
├── qformer-K32-webchain-h500-passval-v1-cached-from250-obs1.0.pt
└── qformer-K32-webchain-h500-passval-v1-cached-from250-obs1.0.gapbest.pt
```

当前 WebChain 训练只形成阶段性结果，使用前应结合对应日志和 validation 口径，不要直接视为完整收敛模型。

#### SyQA Q-Former

原始训练 checkpoint：

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/syqa/qformer/
├── qformer-K32-syqa-v1-obs0.5.pt
├── qformer-K32-syqa-v1-obs0.5.gapbest.pt
├── qformer-K32-syqa-v1-obs1.0.pt
└── qformer-K32-syqa-v1-obs1.0.gapbest.pt
```

正式 AMA-WEB 评测使用的冻结 Q-Former 和 retrieval head：

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/
├── frozen.pt
└── head.pt
```

#### SyQA Bridge

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/syqa/post-qformer/bridge/
├── qwen32-input-k32-rms.pt
├── qwen32-input-k32-rms-semantic.pt
├── qwen32-input-k32-rms-delta-margin-pilot250.pt
└── qwen32-input-k32-rms-delta-margin-total1000.pt
```

正式 AMA-WEB 结果对应：

- 旧 Bridge：`qwen32-input-k32-rms.pt`；
- Delta-margin Bridge：`qwen32-input-k32-rms-delta-margin-total1000.pt`；
- Delta-margin Bridge SHA256：`de13a5cdc66689496dd7a03d8f9313ad4ea4e6fb69bdbd78592a6541654eafa2`。

## 2. 训练数据位置

### 2.1 HumanTrajs 人类采集数据

原始/过滤后数据：

```text
/data1/datasets/xt_ama_adapter/raw/humantrajs/
/data1/datasets/xt_ama_adapter/filtered/humantrajs-semantic-steps.jsonl
/data1/datasets/xt_ama_adapter/filtered/humantrajs-semantic-images/
```

当前使用的 HumanTrajs 处理产物：

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/manifests/
├── humantrajs-post-action-v1.jsonl
├── humantrajs-post-action-v1.report.json
├── humantrajs-post-action-v1.rejections.jsonl
├── humantrajs-post-action-v1.alignment-audit.jsonl
├── humantrajs-pseudo-web-v1.jsonl
└── humantrajs-pseudo-web-v1.report.json

/home/cbd/project/residual-mem/outputs/ama_latent_memory/qa/
├── humantrajs-local-v2.verified.jsonl
├── humantrajs-pseudo-web-v1.verified.jsonl
└── humantrajs-pseudo-web-v1.verified.report.json
```

进入 Q-Former 的最终输入：

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/qformer/
├── humantrajs-pseudo-web-v1.pairs.npz
├── humantrajs-pseudo-web-v1.inputs.report.json
├── humantrajs-pseudo-web-v1.store/
├── humantrajs-pseudo-web-v1.trunk-layer16/
├── humantrajs-pseudo-web-v1.observation-teacher/
└── humantrajs-pseudo-web-v1.fused-teacher/
```

规模：3,371 条最终验证 QA，覆盖 472 条 trajectory；训练/验证/测试划分为 2,802/342/227。

### 2.2 WebChain 数据

WebChain 的当前可复现训练输入和处理产物位于：

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/webchain/
├── webchain-pass-review-v1.store/
├── webchain-pass-review-v1.trunk-layer16/
├── webchain-pass-review-v1.observation-teacher/
├── webchain-pass-review-v1.fused-teacher/
└── logs/

/home/cbd/project/residual-mem/outputs/ama_latent_memory/qformer/
├── webchain-h500-v1.pairs.npz
├── webchain-h500-v1.report.json
├── webchain-h500-v1.store/
├── webchain-h500-passval-v1.pairs.npz
├── webchain-h500-passval-v1.report.json
├── mixed-web-v1.pairs.npz
├── mixed-web-v1.report.json
└── mixed-web-v1.store/
```

实际混合训练输入由 **500 条 HumanTrajs train 子集 + 1,240 条 WebChain QA** 组成，总计 1,740 pairs。WebChain 的原始外部目录没有在当前训练 report 中固化为单一绝对路径，因此交付和复现以以上 `pairs.npz`、`store`、teacher cache 和 report 为准；如需重新从原始 WebChain 数据构建，应同时保留原始 WebChain source path 和对应 manifest。

训练日志：

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/qformer/logs/
├── qformer-K32-webchain-h500-v1-scratch-obs0.5.log
├── qformer-K32-webchain-h500-v1-scratch-obs1.0.log
├── qformer-K32-webchain-h500-passval-v1-cached-from250-obs0.5.log
└── qformer-K32-webchain-h500-passval-v1-cached-from250-obs1.0.log
```

### 2.3 SyQA 数据

SyQA 原始/过滤后数据：

```text
/data1/datasets/residual-mem/molmoweb-syntheticqa-modelscope/
/data1/datasets/xt_ama_adapter/raw/syntheticqa/
/data1/datasets/xt_ama_adapter/filtered/molmoweb-syntheticqa-10k/
/data1/datasets/xt_ama_adapter/filtered/molmoweb-syntheticqa-10k/images/
```

SyQA Q-Former 训练产物：

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/syqa/
├── qformer-pairs-teacher-complete.npz
├── qformer-store-teacher-complete/
├── teacher-observation/
├── teacher-fused-complete/
├── qformer/
└── post-qformer/
```

其中 trunk layer-16 cache 位于：

```text
/data1/datasets/residual-mem/syqa-trunk-layer16-teacher-complete/
```

当前 SyQA 训练输入包含 11,944 个 QA pairs、6,645 个 observations；正式 Q-Former 训练使用完整 teacher 可用子集，AMA-Bench 官方 test 未进入训练。

## 3. AMA-Bench 测试数据与结果位置

AMA-Bench 代码和测试数据：

```text
/home/cbd/project/residual-mem/third_party/AMA-Bench/
/home/cbd/project/residual-mem/third_party/AMA-Bench/dataset/test/open_end_qa_set.jsonl
```

AMA-WEB latent memory 正式结果：

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/ama_eval/
├── web-latent-formal/
│   ├── answers.jsonl
│   ├── answer-checkpoint.jsonl
│   ├── evaluation.json
│   └── cache/
└── web-latent-delta-margin-total1000/
```

对应日志：

```text
/home/cbd/project/residual-mem/logs/ama_latent_memory/ama_eval/
├── web-latent-formal/
└── web-latent-delta-margin-total1000/
```

## 4. 运行环境位置

### 4.1 Qwen3.5-9B / Q-Former / vLLM 环境

```text
/home/cbd/project/residual-mem/.venv-ama-vllm-qwen35/
```

用于：

- Q-Former 训练；
- HumanTrajs 伪 Web observation 生成和验证；
- SyQA teacher/trunk cache；
- Qwen3.5-9B vLLM 服务；
- AMA 评测中的缓存阶段和部分 Judge 服务。

Python：

```text
/home/cbd/project/residual-mem/.venv-ama-vllm-qwen35/bin/python
```

历史 Q-Former 手册还记录了兼容的 conda 环境：

```bash
source /mnt/data/public_tools/miniconda3/etc/profile.d/conda.sh
conda activate qwen-vl
```

实际运行脚本优先使用仓库内 `.venv-ama-vllm-qwen35` 的 Python。

### 4.2 Qwen3-32B / Bridge / Embedding 环境

```text
/home/cbd/project/residual-mem/.venv-ama-embedding-cu124/
```

用于：

- Qwen3-32B Bridge 训练；
- Qwen3-32B Reader；
- Qwen3-VL-Embedding-8B query encoder；
- AMA Judge 客户端和 delta-margin 评测。

Python：

```text
/home/cbd/project/residual-mem/.venv-ama-embedding-cu124/bin/python
```

### 4.3 其他仓库环境

```text
/home/cbd/project/residual-mem/.venv-ama-vllm/
/home/cbd/project/residual-mem/.venv-modelscope/
/home/cbd/project/residual-mem/.venv-msqa/
/home/cbd/project/residual-mem/miniconda3/
```

这些环境用于历史 vLLM、ModelScope、SyntheticQA 或辅助数据处理流程；不建议在没有核对依赖的情况下替换正式训练环境。

依赖清单：

```text
/home/cbd/project/residual-mem/requirements.txt
/home/cbd/project/residual-mem/requirements-residualmem.txt
/home/cbd/project/residual-mem/environment.yml
```

## 5. 复现时的工作目录和基本设置

```bash
cd /home/cbd/project/residual-mem
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
```

GPU 分工原则：

- GPU 0–3：训练、离线 cache 和评估；
- GPU 4–7：历史上用于本地 LLM 服务和官方评测，不应与训练任务混用；
- Bridge/32B 任务根据脚本使用 GPU 0–2；
- 运行中的 vLLM 服务先检查端口和进程，不要直接使用 `pkill -f`。

主要启动脚本：

```text
/home/cbd/project/residual-mem/xt_ama_adapter/scripts/run_webchain_h500_qformer_scratch_arms.sh
/home/cbd/project/residual-mem/xt_ama_adapter/scripts/run_webchain_h500_passval_cached_continuation.sh
/home/cbd/project/residual-mem/xt_ama_adapter/scripts/run_syqa_qformer_formal.sh
/home/cbd/project/residual-mem/xt_ama_adapter/scripts/run_syqa_post_qformer_bridge.sh
/home/cbd/project/residual-mem/xt_ama_adapter/scripts/run_ama_web_latent_formal.sh
/home/cbd/project/residual-mem/xt_ama_adapter/scripts/run_ama_web_delta_margin_total1000.sh
/home/cbd/project/residual-mem/xt_ama_adapter/scripts/run_ama_web_delta_margin_total1000_judge.sh
```

## 6. 复现前检查清单

- 确认使用的 Q-Former、retrieval head、Bridge 和 Reader 路径与实验记录一致；
- 确认 Q-Former checkpoint 的 `qk_norm` metadata 与运行时一致；
- 确认 WebChain 使用的是普通 validation 还是 pass-only validation；
- 确认 AMA-Bench test 数据没有被写入训练 manifest 或 checkpoint 选择流程；
- 确认 Judge API key 非空，避免评测脚本静默返回 gold answer；
- 先用 smoke 样本检查生成结果，再运行完整 360 条 AMA-WEB 评测。

