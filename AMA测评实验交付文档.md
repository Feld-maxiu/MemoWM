# AMA 测评实验交付文档

更新时间：2026-09-02

## 1. 交付结论

本轮围绕 AMA-WEB latent memory/Q-Former 链路完成了三类训练数据实验：

1. **HumanTrajs（人工采集轨迹）**：完成了时序对齐、伪 Web observation 构建、QA 证据复筛和 Q-Former 训练；数据质量和防泄漏流程最完整，适合作为 AMA-WEB 的主要域内训练数据。
2. **WebChain**：将 WebChain 的网页 AXTree/QA 与 500 条 HumanTrajs 样本混合，用于验证真实网页结构数据对 Q-Former 的帮助；当前只有阶段性训练结果，未形成可用于正式 AMA 结论的完整收敛 run。
3. **SyQA（MolmoWeb SyntheticQA）**：完成了大规模 Q-Former 训练，并以其冻结 checkpoint 作为后续 Bridge 和 AMA-WEB 正式评测的基础。旧 Bridge 在 360 道 AMA-WEB QA 上达到 **24.17%**，加入 delta-margin loss 的 Bridge 复测达到 **27.50%**，绝对提升 **3.33 个百分点**。

三类实验使用相同的核心 Q-Former 训练框架：冻结 Qwen3.5-9B trunk/teacher，仅训练约 78.9M 参数的 StateQFormer 及训练期连接层；Q-Former 输出固定为 **32 个 latent slot，每个 512 维**。

## 2. 统一模型与训练目标

### 2.1 模型链路

```text
网页截图/AXTree/伪 Web 文本
        ↓
冻结 Qwen3.5-9B trunk（第 16 层 hidden state）
        ↓
StateQFormer：4 层、hidden 1024、8 heads、32 个 query
        ↓
32 × 512 维 xbar latent memory
        ↓
retrieval head / Qwen3-32B Bridge
        ↓
冻结 Qwen3-32B Reader
        ↓
AMA-Bench LLM-as-Judge
```

Q-Former 训练时开启 `qk_norm`，关闭 self-attention 和 microbatch 丢弃；学习率为 `1e-4`，gradient accumulation 为 4，seed 为 35。训练过程中没有发生 skipped step 或 dropped microbatch。

### 2.2 Q-Former loss function

三次 Q-Former 实验采用以下总损失：

```text
L_QFormer = CE_gold
          + 0.3 × KL_q
          + w_obs × KL_obs
          + 1.0 × InfoNCE_same_session
```

其中：

| 损失项 | 作用 | 使用方式 |
|---|---|---|
| `CE_gold` | 让 latent + question 能生成正确答案 | 对 gold answer 做 teacher-forced cross entropy |
| `KL_q` | 让 latent reader 的答案分布接近完整 observation teacher | 权重固定为 0.3 |
| `KL_obs` | 保留网页状态/屏幕 observation 的信息 | `w_obs=0.5` 或 `1.0`，作为两条受控臂 |
| `InfoNCE_same_session` | 区分同一 session 内不同 step，服务后续检索 | 权重 1.0，temperature 0.05 |

训练和选点同时记录两类指标：

- validation answer CE：越低越好；
- held-out P4 observation gap：`KL_mismatched - KL_matched`，越高越好。gap ≤ 0 表示 latent 没有表现出可验证的 observation 区分能力。

### 2.3 Bridge loss

SyQA checkpoint 后续接入 Qwen3-32B Bridge。基础 Bridge 训练目标为：

```text
L_bridge = CE_gold + 0.30 × KL_teacher
```

在最终 delta-margin Bridge 复测中增加了：

```text
+ 0.10 × hidden_distill
+ 0.20 × max(0, 0.10 + CE_matched - CE_shuffled)
```

其中 `hidden_distill` 对齐 Qwen3-32B 第 16/32/48 层的 hidden teacher；margin 项要求 matched memory 的答案 CE 至少比 shuffled memory 好 0.10。Bridge 采用 RMS calibration，使输出 soft token 的 RMS 与 Qwen3-32B reader embedding 对齐。

## 3. 实验一：HumanTrajs 人工采集数据

### 3.1 数据来源

原始数据来自：`/data1/datasets/xt_ama_adapter/filtered/humantrajs-semantic-steps.jsonl`。

原始规模为 **486 条 trajectory、7,870 个 source step**。源数据中截图语义为动作执行前状态，因此采用以下时序对齐：

```text
memory_i = action_i + sanitized(other_obs_i) + screenshot_(i+1)
```

删除了 terminal action、无连续下一帧、缺失/损坏/低信息截图，并移除 action thought、最终回答和 `send_msg_to_user` 等潜在泄漏信息。最终得到：

| 项目 | 数量 |
|---|---:|
| 保留状态 | 6,750 |
| 拒绝状态 | 1,120 |
| 保留率 | 85.77% |
| train / validation / test 状态 | 5,563 / 744 / 443 |
| trajectory 切分 | 393 / 49 / 40 |

由于 HumanTrajs 没有真实 DOM/AXTree，使用冻结 Qwen3.5-9B-VL 依据截图、URL 和页面标题生成 `vl-pseudo-web-observation-v1`。6,525 张唯一截图全部成功转写，并回填到 6,750/6,750 个 memory 状态。

### 3.2 QA 构建和过滤

QA 生成器不读取 instruction、action thought 或未来帧。生成的 QA 先经过截图证据验证，再经过伪 Web 文本证据复筛；最终只有 `pseudo_web_text_verified` 的 QA 进入训练。

| 阶段 | 数量 |
|---|---:|
| 截图初筛 QA | 4,780 |
| 伪文本复筛保留 | 3,371 |
| 伪文本复筛通过率 | 70.52% |
| train / validation / test QA | 2,802 / 342 / 227 |
| 覆盖 trajectory | 472 |

最终 train/validation/test 之间 trajectory 无交集，保留 QA 的跨 split 截图哈希无交集。需要保留的限制是：伪 Web observation 是视觉模型转写文本，不等同于真实 AXTree；生成器和 verifier 均使用 Qwen3.5-9B-VL，存在一定教师自洽性偏差。

### 3.3 训练方式与结果

使用 32-query、4-layer StateQFormer，从零训练两条 `w_obs` 对照臂，最大 9,000 steps；两个实验均为同一 seed、同一数据和同一训练配置。

| 训练臂 | 最优 validation answer CE | CE 最优 step | 最优 P4 gap | gap 最优 step |
|---|---:|---:|---:|---:|
| `w_obs=0.5` | **2.6574** | 2,000 | 0.0385 | 3,750 |
| `w_obs=1.0` | **2.6256** | 2,750 | **0.0543** | 4,750 |

结果表明，`w_obs=1.0` 在本实验上同时得到更低的 validation CE 和更高的 observation gap，因此其 gap-best checkpoint 被作为后续候选之一。但该实验主要验证训练数据流水线和域内训练可行性，当前没有在 HumanTrajs 训练臂上完成同口径 AMA-Bench 正式 test，因此不能直接把上述 CE/gap 外推成 AMA 准确率。

## 4. 实验二：WebChain 数据

### 4.1 数据来源与构建

WebChain 使用网页 trajectory 中的 AXTree 页面结构和配套 QA。AXTree 被清洗为 query-independent 的页面文本块，保留 role、name、value、aria-label、placeholder、title、alt、href 等可用属性；instruction 被排除，不进入训练输入。

为增加训练稳定性，实际输入数据由两部分组成：

| 来源 | 数量 | 质量状态 |
|---|---:|---|
| HumanTrajs train 子集 | 500 | `VERIFIED` |
| WebChain | 1,240 | `PASS` 310、`REVIEW` 930 |
| 合计 | **1,740** | 混合 Web 训练集 |

数据按 trajectory/session 组织，最终训练/验证/测试 pair 数为 **1,482 / 127 / 131**；用于重新建立严格验证集的 pass-only 子集为 **31 个 validation observation / 31 个 trajectory**，其余 96 条原 validation 被降为 holdout，避免用质量未通过的样本选点。

### 4.2 训练方式与阶段性结果

WebChain 同样使用 32-query、4-layer StateQFormer，训练目标仍为：

```text
L_QFormer = CE_gold + 0.3 × KL_q + w_obs × KL_obs
          + 1.0 × InfoNCE_same_session
```

训练先从零启动两条 `w_obs=0.5/1.0` 臂，但 scratch 日志只完整记录到 **250 step**，随后基于 checkpoint 切换到 pass-only validation 继续训练。pass-only continuation 日志目前记录到 **1,250 step**，没有形成完整 9,000-step 的可比最终结果。

在已记录的最后 step：

| 训练臂 | step | validation CE | P4 gap | 解释 |
|---|---:|---:|---:|---|
| `w_obs=0.5` | 1,250 | 1.3420 | 0.0005 | gap 仅略高于 0，证据较弱 |
| `w_obs=1.0` | 1,250 | 1.3619 | 0.0065 | gap 为正，但仍很小 |

因此 WebChain 实验当前可以交付的结论是：数据构建、AXTree 清洗、混合训练和 pass-only validation 流程已跑通；但现有日志不足以支持“WebChain 训练后模型稳定提升”或三类数据的最终准确率排序。下一步若需要正式比较，应固定同一验证集、完成完整预算并重新跑 retrieval gate 与 AMA-WEB test。

## 5. 实验三：SyQA 数据

### 5.1 数据来源

SyQA 即 MolmoWeb SyntheticQA，来源是网页截图到问题/答案的合成 QA 数据。为适配当前 text observation reader，实验使用 Qwen3.5-9B-VL 生成视觉转写 observation，并基于 QA/截图证据构建 Q-Former pair。

最终 Q-Former 训练数据元信息如下：

| 项目 | 数量 |
|---|---:|
| observation | 6,645 |
| QA pairs | 11,944 |
| train / validation observation | 5,107 / 825 |
| teacher 完整可用 subset | 2,829 samples |
| session 数 | 988 |
| 官方 AMA test 是否进入训练 | 否 |

SyQA 训练时 instruction 和 action 均不进入 Q-Former 输入；问题只作为 answer CE/KL 的 query，observation 由独立的视觉/文本 teacher 提供。官方 AMA-Bench test 不参与训练、标签生成或 checkpoint 选择。

### 5.2 训练方式与 Q-Former 结果

从零训练两条严格同配置的 `w_obs=0.5/1.0` 臂，最大 9,000 steps。两条臂的最终 metadata 如下：

| 训练臂 | 最优 validation CE | CE 最优 step | 最优 P4 gap | gap 最优 step |
|---|---:|---:|---:|---:|
| `w_obs=0.5` | **2.3050** | 7,000 | **0.3003** | 8,500 |
| `w_obs=1.0` | **2.2523** | 7,500 | 0.2757 | 5,750 |

`w_obs=1.0` 在 validation answer CE 上更好，故正式 AMA-WEB 链路采用其 CE-best Q-Former candidate，并冻结 QFormer 与 retrieval head。这里的 gap-best 和 CE-best 是两个不同选点标准，不应混用。

### 5.3 Bridge 与 AMA-WEB 正式评测

正式 AMA-WEB 数据来自 `AMA-Bench/dataset/test/open_end_qa_set.jsonl` 的 WEB 开放式 QA 子集：原始 31 条 trajectory、372 条 QA；按实验要求排除 episode 184 后，最终评测 **30 条 trajectory、360 条 QA**，每条有效 trajectory 12 道题。

推理链路中 Q-Former、retrieval head、Bridge、Qwen3-32B Reader 和 Judge 全部冻结；使用 matched memory、`top_k=1`，Judge 为 Qwen3-32B LLM-as-Judge，分数为二值正确/错误。

| Bridge 版本 | Loss/训练变化 | 正确数 | 准确率 |
|---|---|---:|---:|
| `qwen32-input-k32-rms.pt` | `CE_gold + 0.30×KL_teacher` | 87/360 | **24.17%** |
| `qwen32-input-k32-rms-delta-margin-total1000.pt` | 增加 hidden distill 和 matched-vs-shuffled CE margin | 99/360 | **27.50%** |

按 QA 类型的结果：

| 类型 | 旧 Bridge | Delta-margin Bridge |
|---|---:|---:|
| A | 31/121（25.62%） | 32/121（26.45%） |
| B | 29/90（32.22%） | 34/90（37.78%） |
| C | 15/90（16.67%） | 19/90（21.11%） |
| D | 12/59（20.34%） | 14/59（23.73%） |
| **总体** | **87/360（24.17%）** | **99/360（27.50%）** |

逐题配对结果为：31 题从错变对、19 题从对变错、68 题均正确、242 题均错误。因此 delta-margin 方向在本次固定 360 题复测中得到支持，但这是单次 test run，尚不足以证明提升在不同 seed、不同 QA 集合上稳定。

## 6. 三类实验对比与边界

| 数据源 | 主要用途 | 当前最可信结果 | 当前边界 |
|---|---|---|---|
| HumanTrajs | AMA-WEB 域内 warm-up/训练 | 3,371 条证据复筛 QA；`w_obs=1.0` CE 2.6256、gap 0.0543 | 尚未完成同口径 AMA 正式 test |
| WebChain | AXTree/结构化网页数据混合训练 | 1,240 条 WebChain QA + 500 条 HumanTrajs；已跑通 pass-only validation | 只到 1,250 step 的阶段性结果，不能下最终排序结论 |
| SyQA | 视觉/OCR warm-up，并作为 Bridge/AMA 链路基础 | Q-Former CE 2.2523；AMA-WEB 24.17% → 27.50% | SyQA 与 AMA-WEB 存在视觉/文本输入域差异，最终准确率只代表固定配置 |

需要避免的表述：

- 不能把伪 Web observation 称为真实 DOM/AXTree；
- 不能把 Q-Former validation CE 或 P4 gap 直接称为 AMA 准确率；
- 不能仅凭 SyQA 的 24.17%/27.50% 归因 Q-Former 的净增益，因为当前正式 AMA-WEB 还缺少 question-only、shuffled-memory 和 no-latent-memory 的同口径对照；
- 不能把 WebChain 当前 1,250 step 的中间日志写成完整训练结果。

## 7. 关键产物与复现入口

- HumanTrajs 数据流水线：[HUMANTRAJS_训练数据流水线.md](xt_ama_adapter/HUMANTRAJS_训练数据流水线.md)
- Q-Former/Bridge 方案：[QFORMER_32B_BRIDGE_技术方案.md](xt_ama_adapter/QFORMER_32B_BRIDGE_技术方案.md)
- AMA-WEB 正式记录：[AMA_WEB_QA_实验记录_2026-09-01.md](xt_ama_adapter/AMA_WEB_QA_实验记录_2026-09-01.md)
- HumanTrajs manifest：[humantrajs-post-action-v1.report.json](outputs/ama_latent_memory/manifests/humantrajs-post-action-v1.report.json)
- HumanTrajs pseudo Web QA：[humantrajs-pseudo-web-v1.verified.report.json](outputs/ama_latent_memory/qa/humantrajs-pseudo-web-v1.verified.report.json)
- WebChain mixed dataset：[webchain-h500-v1.report.json](outputs/ama_latent_memory/qformer/webchain-h500-v1.report.json)
- SyQA Q-Former 日志：[qformer-K32-syqa-v1-obs0.5.log](../logs/ama_latent_memory/syqa/qformer-K32-syqa-v1-obs0.5.log)、[qformer-K32-syqa-v1-obs1.0.log](../logs/ama_latent_memory/syqa/qformer-K32-syqa-v1-obs1.0.log)
- AMA-WEB 结果记录：[AMA_WEB_QA_实验记录_2026-09-01.md](xt_ama_adapter/AMA_WEB_QA_实验记录_2026-09-01.md)
