# AMA Latent Memory：QFormer 与 Qwen3-32B Bridge 技术方案

> 状态：设计稿，尚未实现。最后核对：2026-08-28。
>
> 本文约定最终 Reader 为本地 Qwen3-32B，因此 bridge 的目标维度是
> **512 → 5120**，不是 5210。AMA-Bench 本身不要求使用 32B；5120 来自
> `Qwen3-32B/config.json` 的 `hidden_size`。

## 1. 结论与实验边界

当前工作区只有 QFormer 的网络定义、训练代码和历史实验报告，没有可访问的
QFormer checkpoint。不能把随机初始化的 `StateQFormer` 当作已有 tokenizer 使用。
本实验需要依次完成：

1. 从头训练新版 QFormer；
2. 冻结 QFormer，预计算各 observation 的 `xbar`；
3. 训练与该 QFormer 严格绑定的 retrieval head；
4. 冻结 QFormer、retrieval head 和 Qwen3-32B，仅训练 `512 → 5120` bridge；
5. 在 AMA-Bench 上评测 latent memory，并与已有 official embedding RAG 结果比较。

正式 latent 路径不把检索到的原始 step 文本交给 Reader。原始文本可以作为训练
监督、teacher context、审计信息和错误分析材料，但推理时 Reader 只接收 soft-token
memory 与问题 token embedding。

```text
AMA step（action + observation）
        │
        ▼
冻结 Qwen3.5-9B，第 16 层完整序列 H_t（T×4096）
        │
        ▼
训练后冻结的 QFormer（K=32）
        │
        ├── xbar（32×512）── retrieval head ──► 4096-D 检索向量
        │                                      │
        │                              与问题向量做 top-k
        │                                      │
        └──────────── 选中的 top-k xbar ◄──────┘
                               │
                               ▼
                    bridge（512→5120）
                               │
                               ▼
              [latent soft tokens, question embeddings]
                               │
                               ▼
                    Qwen3-32B inputs_embeds
```

这里有两个需要隔离的实验变量：

- 状态编码器是否能把 observation 压成可用的 `xbar`；
- Qwen3-32B 是否能通过 bridge 解码该 `xbar`。

因此第一版必须分阶段训练，不做 QFormer + bridge + 32B 的端到端联合训练。

## 2. 当前资产盘点

### 2.1 已存在

- `residualmem/latent/qformer.py`：可训练的 `StateQFormer`；
- `experiments/state_tokenizer/train_qformer_joint.py`：QFormer 联合训练器；
- Qwen3.5-9B trunk/reader 所需代码；
- `/data1/models/Qwen3-32B`，hidden size 为 5120；
- AMA step 适配、向量检索与 trace 数据结构；
- MolmoWeb SyntheticQA（下文简称 SyQA）；
- 486 条 HumanTrajs trajectory，共 7,870 个源 step；时序清洗后得到 6,750 个动作后状态；
- 6,525 个唯一截图的 `vl-pseudo-web-observation-v1` 缓存，覆盖全部 6,750 个状态；
- 3,371 条经截图初筛和伪文本证据复筛的 HumanTrajs local QA。

### 2.2 当前缺失

- 任意可加载的 QFormer `.pt/.pth` checkpoint；
- 新版 QFormer 对应的 xbar cache；
- 与新版 QFormer 严格绑定的 retrieval-head checkpoint；
- SyQA 到 QFormer 训练格式的稳定构建器（HumanTrajs manifest 已完成）；
- `512 → 5120` bridge、训练器和 checkpoint protocol；
- Qwen3-32B `inputs_embeds` 推理入口；
- AMA-Bench runtime 注册和端到端评测入口。

`outputs/` 被 Git 忽略。任何训练产物都必须有独立的产物清单、hash 与备份策略，
不能假设换一台机器或重新 checkout 后仍然存在。

## 3. 新版 QFormer 冻结规格

第一版固定以下结构，不在同一轮实验中扫结构超参：

| 字段 | 固定值 |
|---|---:|
| trunk | Qwen3.5-9B，第 16 层 |
| input dimension | 4096 |
| queries / slots | 32 |
| QFormer hidden | 1024 |
| heads | 8 |
| layers | 4 |
| output dimension | 512 |
| self-attention | false |
| QK-normalization | true |
| query-conditioned write | 禁止 |

QFormer 写入必须保持 query-independent：同一个 trajectory step 只编码一次，后续所有
问题复用同一份 `xbar`。否则系统不再是可缓存的记忆，而是逐问题重新编码 observation。

### 3.1 QK-normalization 是模型结构的一部分

`qk_norm=true` 不新增 parameter，因此不会改变 `state_dict` 的 key。用
`strict=True` 加载 checkpoint 也无法识别开关错误，但打开与关闭时前向结果不同。
因此必须把它写入 artifact metadata，并参与完整 artifact hash。

最低 metadata 要求：

```json
{
  "protocol": "qwen35_qformer_state_tokenizer_v2",
  "queries": 32,
  "input_dim": 4096,
  "output_dim": 512,
  "hidden": 1024,
  "heads": 8,
  "layers": 4,
  "self_attention": false,
  "qk_norm": true,
  "trunk_model": "Qwen3.5-9B",
  "trunk_layer": 16,
  "training_data_manifest_sha256": "...",
  "state_dict_sha256": "...",
  "architecture_sha256": "..."
}
```

旧协议 `qwen35_qformer_state_tokenizer_v1` 没有完整表达前向语义。新版产物应升级
protocol，不能仅依靠文件名里的 `K32` 或人工记忆判断配置。

## 4. 数据评估与使用方式

### 4.1 SyQA 当前数据质量

原始 materialized SyQA：

- 10,000 张截图，图片均存在；
- 每张截图 5 个 QA，共 50,000 个 QA；
- OCR 27,420（54.84%）；
- affordance 12,935（25.87%）；
- summarization 9,645（19.29%）；
- 答案长度中位数 7 词，P95 为 27 词。

原始数据可以作为视觉状态压缩和 Reader bridge 的预训练数据，但当前 semantic-filter
产物不能直接作为正式训练集：

- 过滤器只处理了每张图的 `messages[0]`，其余 40,000 个 QA 没有审核；
- 当前保留 4,143 条，保留率 41.43%；
- 其中 4,047 条是 OCR，占 97.68%，任务分布严重失衡；
- 5,857 条被拒绝，至少 4,634 条是 JSON 解析失败，不等价于内容不合格；
- 过滤结果的 `sample_id` 全为空，并丢失 website、URL、question type 和 QA index；
- 10,000 张图只来自 12 个网站，allrecipes 占 79.73%；
- 只有 1,568 个唯一 URL，9,445 行来自重复 URL；
- 全部 50,000 个问题中有 12,806 个 normalized duplicate question。

因此必须重新展开和过滤 SyQA：

1. 将每张图的 5 个 messages 全部展开；
2. 主键使用 `(image_id, qa_index)`；
3. 保留 `website/url/question_type/question_form`；
4. 使用可靠 structured output，或把解析失败与语义拒绝分开记录；
5. 对同一截图的多个 QA 统一分配 split；
6. 按 URL/页面族分组切分，禁止按 QA 行随机切分；
7. 对 allrecipes 下采样或使用按网站加权采样；
8. 分别报告 OCR、affordance、summarization 指标。

### 4.2 HumanTrajs 与 AMA 域对齐

SyQA 是“真实截图 → 截图 QA”，而当前 AMA adapter 的主要输入是 `action + textual
observation`，截图缺失时使用白图。只用 SyQA 会造成明显的视觉域和输入格式错配。

HumanTrajs 更适合承担域内训练：它具有 instruction、action、step observation、截图和
trajectory 顺序。当前流水线已将 `action_i` 对齐到 `screenshot_{i+1}`，并删除 terminal、
低信息动作后帧和 action thought。7,870 个源 step 最终得到 6,750 个可用 memory 状态，
按 trajectory 切分为 train 5,563 / validation 744 / test 443。

由于源数据没有真实 DOM/AXTree，Qwen3.5-9B-VL 将 6,525 个唯一截图转写为
`vl-pseudo-web-observation-v1`，回填覆盖 6,750/6,750 状态。该文本长度 p50 为 1,357 字符、
p95 为 2,238、最大 4,029；它是视觉转写文本，不得声称为真实 AXTree。

截图 verifier 通过的 4,780 条 local QA 又经过只看伪文本的证据复筛，最终保留 3,371 条
（70.52%）：train 2,802 / validation 342 / test 227。最终 472 条 trajectory 在 split 间
零交叉，截图哈希也零跨 split。训练时只读取
`verification_status=pseudo_web_text_verified` 的最终文件。

推荐的数据职责：

| 数据 | QFormer | Bridge | 最终评测 |
|---|---|---|---|
| 修复后的 SyQA | 视觉/OCR warm-up | 视觉 latent warm-up | 仅域外诊断 |
| HumanTrajs pseudo Web | AMA-WEB 主要域内训练 | AMA-WEB 主要域内训练 | trajectory-held-out validation |
| AMA-Bench test | 禁止 | 禁止 | 唯一正式测试 |

不得用 AMA test 的问题、答案、gold step、evidence 标注或其派生 teacher 输出训练任何模块。
如 AMA 提供官方 train split，也必须按 episode 切分并记录版本；不能默认 test 可以参与拟合。

## 5. 阶段 A：训练 QFormer

### 5.1 冻结与可训练模块

冻结：

- Qwen3.5-9B trunk；
- 用作 teacher/reader 的语言模型参数。

训练：

- `StateQFormer`；
- 训练期 Qwen3.5 connector；
- 联合 semantic/retrieval head（如果启用语义目标）。

训练期 Qwen3.5 connector 只是为 QFormer 提供可微监督。最终 Qwen3-32B 不复用它；
32B 使用后续单独训练的 5120 维 bridge。

### 5.2 目标函数

沿用并修正现有训练器的目标：

```text
L_QFormer = CE_gold
          + w_q * KL_question
          + w_o * KL_observation
          + lambda_sem * InfoNCE_same_group
```

- `CE_gold`：latent + question 对 gold answer 的 teacher-forced CE；
- `KL_question`：冻结 teacher 读取完整 observation 后的答案分布；
- `KL_observation`：约束 latent 保留具体屏幕/状态信息；
- `InfoNCE`：保持同一 trajectory 内不同 step 可区分，并服务后续检索。

对 SyQA，如果没有可靠 DOM/AXTree 或 textual observation，不得把空字符串伪装成完整
teacher context。可只使用 screenshot-grounded CE/视觉 teacher，或先生成带来源标记的
OCR/页面描述。HumanTrajs 当前可使用带明确协议标记的伪 Web observation，但实验报告必须
单列其教师自洽性偏差。所有数据源的损失可用性必须在 manifest 中明确记录。

### 5.3 训练稳定性要求

- 必须开启 `qk_norm`；
- `drop_microbatches=false`；
- 对梯度先做 finite 检查，再进行 clipping 和 optimizer step；
- 跳步数应接近 0；
- 固定 seed、validation split 和 validation rows；
- CE-best 与 gap-best 分别存盘；
- checkpoint 保存完整架构 metadata，而不是只保存 slot 数。

### 5.4 QFormer 验收闸

至少同时满足：

1. 训练过程没有持续 non-finite gradient，学习率没有长期冻结在地板；
2. held-out observation gap `mismatched_KL - matched_KL > 0`；
3. `matched_beats_mismatched` 达到预注册阈值，第一版使用 0.90；
4. 单独训练的 retrieval head 在 trajectory/session 内 R@1 显著高于随机；
5. QA validation CE 优于 zero-latent 和 shuffled-latent 对照；
6. 人工检查 latent 没有依赖问题文本，写侧保持 query-independent。

正式 checkpoint 只从通过上述闸的候选中选择。不得因为文件名是 `K32e` 就默认健康。

## 6. 阶段 B：生成冻结 xbar 与训练 retrieval head

QFormer 通过验收后立即冻结。之后重新、一次性生成：

- SyQA xbar cache；
- HumanTrajs xbar cache；
- AMA ingest 时使用的在线 encoder 配置；
- 与该 xbar 坐标严格绑定的 retrieval-head cache。

cache 每行必须携带：

```text
sample_id / trajectory_id / step_index / split
xbar[32,512] / valid[32]
QFormer artifact hash
输入 wire-format protocol
可选的原始训练监督引用
```

retrieval head checkpoint 必须记录它训练所用 cache 的 QFormer artifact hash；不能只通过
文件名含有 `qformer` 判断匹配。旧 K16 cache、旧 head 和旧 connector 全部禁止复用。

## 7. 阶段 C：训练 512 → 5120 bridge

### 7.1 Bridge 基线结构

第一版沿用 soft-token connector 的基本形状，但建立新 protocol：

```text
LayerNorm(512)
  → Linear(512, 5120)
  → GELU
  → Linear(5120, 5120)
  → RMSNorm(5120)
  + slot_embedding[32, 5120]
  + memory_rank_embedding[top_k, 5120]
```

`slot_embedding` 区分同一个 xbar 内的 32 个 query slot；
`memory_rank_embedding` 区分第 1、2、… 个检索 step。两者不能混为一个 rank。

如果数据量无法支撑约 29M 参数的基线 MLP，可预注册一个较小的 bottleneck 消融，例如
`512 → 2048 → 5120`。不能看到验证结果后再任意更换结构。

### 7.2 Reader 输入

对每个问题：

```text
latent = bridge(top-k xbar)              # B × (top_k*32) × 5120
question = embed(question_input_ids)      # B × N × 5120
inputs_embeds = concat(latent, question)
attention_mask = concat(latent_valid, question_mask)
labels = -100 on latent/question prompt; gold token ids on answer span
```

必须使用与正式 AMA Reader 相同的 system prompt、chat template、答案格式、thinking 设置、
position IDs 和 generation stopping 条件。训练与推理 prompt 不一致会让 bridge 指标失去意义。

### 7.3 冻结与梯度

冻结：

- QFormer；
- retrieval head；
- Qwen3-32B 全部参数。

仅优化 bridge。梯度仍需穿过冻结的 Qwen3-32B 返回 bridge，因此不能用切断 autograd 的
`inference_mode` 包住 student forward。应禁用 32B 参数梯度、关闭 KV cache，并根据硬件
采用 model parallel、activation checkpointing 或较短序列。

### 7.4 Bridge 损失

基础目标：

```text
L_bridge = CE_gold + w_distill * KL_teacher
```

- `CE_gold` 始终可用；
- 只有 teacher 能读取可靠原始 observation 文本时才启用 KL；
- SyQA 只有截图而 Qwen3-32B 是 text-only 时，不能让 teacher 读空 context；
- teacher 可使用可靠 OCR/页面描述，但必须标注其来源并单独做消融。

第一版不更新 QFormer。未来只有在分阶段基线稳定后，才能新增“小学习率解冻 QFormer”的
联合训练消融。

### 7.5 Bridge 语义保持增强目标

#### 7.5.1 动机与边界

`CE_gold` 和答案分布 KL 只通过最终答案间接监督 bridge；Reader embedding RMS 对齐只控制
数值尺度。这三者都不能单独证明 `xbar[512]` 中的样本特异信息在 bridge 输出的 5120 维
内容表示中仍然存在。

当前 `qwen32-input-k32-rms.pt` 的 validation 结果为：

| arm | answer CE |
|---|---:|
| matched | 2.4727 |
| shuffled | 2.5172 |
| zero | 4.9498 |
| question-only | 4.9883 |

`shuffled - matched = 0.0445`，说明 Reader 明显使用了 latent token，但对正确和错误记忆的
区分仍然较弱。这与“bridge 没有充分保留 `xbar` 的样本特异语义”一致，但不能据此排除
QFormer 表示或 retrieval 数据本身区分度不足。因此在重训前必须先做第 7.5.6 节的冻结
probe 诊断。

不能直接使用以下逐维约束：

```text
bridge_output[..., :512] == xbar
```

QFormer 的 512 维坐标与 Qwen3-32B input embedding 的 5120 维坐标没有天然逐维对应。强制把
QFormer 坐标写入 Reader 的任意前 512 维，可能保持数值，却破坏 Reader 可读性。增强目标应
同时约束“信息可恢复”“关系几何保持”和“冻结 Reader 实际感知到的语义”。

#### 7.5.2 内容分支接口

bridge 训练接口新增可选返回值：

```python
soft_tokens, valid_mask, content_states = bridge(
    xbar, valid, return_content=True
)
```

`content_states` 必须取自加入 `slot_embedding` 和 `memory_rank_embedding` 之前、经过
`output_norm` 的 `B × top_k × 32 × 5120` 内容分支。语义保持 loss 不能直接作用于混入位置
embedding 的最终 soft token，否则位置参数可能在不保留 observation 内容的情况下满足
辅助目标。Reader 正式输入仍使用加入位置参数并完成 Reader-RMS 校准后的 `soft_tokens`。

#### 7.5.3 可恢复性损失

增加一个只在训练时存在的线性语义 decoder：

```text
content_states[5120]
  → LayerNorm(5120)
  → Dropout(0.1) + Gaussian noise
  → Linear(5120, 512)
  → reconstructed_xbar[512]
```

对有效 slot 定义：

```text
x_target = non_affine_LayerNorm(xbar)
L_rec = mean_valid(
    1 - cosine(reconstructed_xbar, x_target)
    + 0.25 * SmoothL1(reconstructed_xbar, x_target)
)
```

Gaussian noise 的标准差第一版设为对应 content RMS 的 `0.01`。单层 decoder、dropout 和噪声
用于降低 bridge 把信息藏入 Reader 几乎不可见的小幅维度、再由复杂 decoder 解码的风险。
decoder 与 bridge 一起训练，但不保存到推理 artifact，也不增加推理成本。

`L_rec` 只能证明输出包含 `xbar` 信息，不能证明冻结 Reader 会使用这些信息，所以不能单独
作为正式增强方案。

#### 7.5.4 关系几何损失

分别对输入 slot 和内容分支做 L2 归一化，并比较它们的 cosine Gram matrix：

```text
G_x = normalize(xbar) @ normalize(xbar).T
G_y = normalize(content_states) @ normalize(content_states).T
L_rel = SmoothL1(mask_off_diagonal(G_y), mask_off_diagonal(G_x))
```

该目标不要求 512 和 5120 逐维对应，但要求原来相似的 slot 映射后仍相似、原来不同的 slot
映射后仍可区分。对角线恒为 1，应从 loss 中排除。第一版同时计算：

1. 每个 memory 内 32 个 slot 的 intra-memory relation；
2. 每条 trajectory 有效 slot 均值的 inter-example relation。

如果 Reader micro-batch 太小，inter-example relation 可额外抽取 16 条 `xbar`，仅执行便宜的
bridge forward，不经过冻结 32B。该辅助 batch 不改变 Reader CE 的 effective batch size。

#### 7.5.5 Reader hidden-state 语义蒸馏

这是语义保持增强的主约束。对具有可靠原始 observation 文本、独立 verifier 通过且属于
train split 的样本，让冻结 Qwen3-32B teacher 读取 oracle observation text + question，缓存
问题末尾 token 在第 16、32、48 层的 5120 维 hidden anchor。Student 使用 bridge latent +
同一 question，并在相同问题 token 位置提取 hidden state：

```text
L_hid = mean(layer in {16, 32, 48})(
    1 - cosine(student_question_hidden[layer],
               teacher_question_hidden[layer])
)
```

Teacher 与 Student 使用同一个冻结 Reader，hidden width 相同，不新增投影层。只缓存三个
question-boundary anchor，不缓存整段序列，以控制磁盘和内存开销。Student 侧使用 forward
hook 只保留对应层和 token 的张量，梯度必须能够从这些 hidden state 返回 bridge。

没有可靠 observation text 的 SyQA 样本不得伪造 text teacher，也不得使用 AMA test 的
trajectory、问题、答案或其派生 hidden state。此类样本只使用 `CE_gold + L_rec + L_rel`，并在
metadata 中记录各类 loss 的有效样本数。

#### 7.5.6 训练前冻结 probe

正式重训前，先冻结当前 QFormer 和当前 bridge，在 train split 的 `content_states` 上训练一个
`5120 → 512` 单层 decoder，并只在 grouped validation 上报告：

- reconstruction cosine；
- normalized MSE；
- linear CKA；
- content covariance 的有效秩；
- matched 与 shuffled content distance。

probe 只能读取 train split 拟合参数；validation 和 AMA test 不得参与拟合。若 held-out probe
已经能高质量恢复 `xbar`，则不能再把 matched/shuffled gap 小简单归因于 bridge 信息丢失，
应优先检查 QFormer、retrieval 和 Reader 是否忽略该信息。

#### 7.5.7 总损失与权重

第一版预注册：

```text
L_bridge_semantic =
    CE_gold
    + 0.30 * KL_teacher
    + 0.10 * L_hid
    + 0.05 * L_rec
    + 0.05 * L_rel
```

三个新增 loss 的权重在前 200 optimizer steps 从 0 线性 warm up 到目标值。Reader embedding
RMS 继续使用硬校准，不作为可学习语义 loss。若某 batch 没有 hidden teacher，`L_hid` 记为
不可用而不是 0 分样本，并按有效样本数单独聚合。

不得仅根据各项 loss 的原始数值临时更改权重。若 smoke test 显示某项梯度比 `CE_gold` 的
bridge 梯度大 10 倍以上，停止训练并在正式运行前重新预注册权重；不得看到 AMA 结果后再
修改。

建议新增 CLI（尚未实现）：

```text
--hidden-distill-weight 0.10
--semantic-reconstruction-weight 0.05
--semantic-relation-weight 0.05
--semantic-warmup-steps 200
--semantic-decoder-dropout 0.10
--semantic-noise-ratio 0.01
```

checkpoint metadata 必须额外保存以上参数、三项 loss 的 train/validation 统计、hidden teacher
cache hash、teacher layer 列表、decoder probe 指标和语义有效样本数。推理 artifact 不包含
decoder 参数。

#### 7.5.8 消融与候选选择

正式实验必须从相同 seed、相同初始化、相同数据顺序和相同步数训练以下候选：

| 候选 | CE/KL | `L_rec` | `L_rel` | `L_hid` |
|---|---:|---:|---:|---:|
| Baseline | ✓ |  |  |  |
| SP-A | ✓ | ✓ |  |  |
| SP-B | ✓ | ✓ | ✓ |  |
| SP-Full | ✓ | ✓ | ✓ | ✓ |

允许先从当前 checkpoint 继续训练 300–500 step 做稳定性 smoke test，但该结果不得与从零训练
的 baseline 作正式优劣结论。正式候选必须从相同初始化重训，且只根据 grouped validation
选择。AMA-WEB 已是观察过的 test，不得用于选择候选或调 loss 权重；锁定最终候选后只做
一次报告性复测。

### 7.6 Bridge 验收闸

必须报告以下受控对照：

- matched latent；
- shuffled latent；
- zero latent；
- question-only；
- oracle original-text context（只作上界，不是正式 latent 路径）。

通过标准至少包括：

1. matched 的 validation CE 明显优于 shuffled/zero latent；
2. matched 的 QA 正确率显著优于 question-only；
3. `shuffled CE - matched CE` 应从当前 0.0445 提高到预注册目标 0.10，且 matched CE 相比
   同初始化 baseline 的恶化不得超过 1%；
4. 在 OCR、affordance、summarization 和 HumanTrajs 上分别报告结果；
5. validation 按 URL/trajectory 分组，无同页面泄漏；
6. SP 候选的 held-out reconstruction cosine 第一版目标为不低于 0.90，并报告 CKA 和有效秩；
7. checkpoint metadata 绑定 QFormer hash、Reader model hash、数据 manifest、prompt hash 和
   hidden teacher cache hash；
8. matched/shuffled 提升必须同时反映在 QA 指标上，不能只用辅助 loss 或 probe 指标过闸。

## 8. 阶段 D：AMA Latent Memory 推理

每个 episode 的写入过程：

1. 用稳定 wire format 适配每个 AMA step；
2. 冻结 Qwen3.5 trunk + 冻结 QFormer 编码一次；
3. 保存 `(episode_id, step_index, xbar, valid, retrieval_vector)`；
4. 同一个 step 的 xbar 对该 episode 的所有问题复用。

每个问题的读取过程：

1. Qwen3-VL-Embedding 编码问题；
2. retrieval head 对所有 step 排序；
3. 选择 top-k 的 xbar；
4. 按确定顺序拼接 soft tokens；
5. 与问题 embeddings 拼接后送入 Qwen3-32B；
6. 保存检索排名、step id、分数、latent hash 和生成答案用于审计。

正式 Reader 不接收 top-k step 的原始文本。若为了调试提供 text fallback，必须使用不同的
baseline 名称和结果目录，防止与 latent 指标混淆。

## 9. AMA 比较设计

目标比较至少包括：

| 实验臂 | 检索 | Reader memory 输入 |
|---|---|---|
| Official embedding RAG | official embedding | 原始文本 |
| Latent memory | 新 QFormer + matched head | bridge soft tokens |
| Question-only | 无 | 无 memory |
| Shuffled-latent control | 新 QFormer + 故意错配 | 错误 soft tokens |

Official embedding RAG 与 latent memory 同时改变了检索器和 Reader memory 形式，因此不能仅靠
两臂差异归因“latent 优于文本”。若要做因果归因，还需增加“同一 QFormer 检索结果 + 原始文本
Reader”的诊断臂；该臂不是正式目标路径，但能拆开 retrieval 与 representation 的贡献。

所有臂必须使用：

- 同一 AMA split 和问题集合；
- 相同 top-k；
- 相同 Qwen3-32B prompt、generation 参数和判分器；
- 独立结果目录；
- 明确的失败计数，禁止缺 API key 时返回 gold answer 的静默 fallback。

## 10. 需要修改或新增的代码

### 10.1 更新现有代码

- `xt_ama_adapter/xt_ama_adapter/runtime.py`
  - `QFormerRuntimeConfig` 增加 `qk_norm`；
  - 构造 `StateQFormer` 时传递完整 metadata；
  - validator 校验完整架构和 QFormer/head hash 绑定；
  - 默认 K 从 16 改为 32，但不允许缺 metadata 时猜测。
- `residualmem/latent/qformer_runtime.py`
  - 支持并强制校验 `qk_norm`；
  - 从 checkpoint metadata 构造网络。
- `experiments/state_tokenizer/build_qformer_bridge_cache.py`
  - 增加 `--qk-norm` 或完全移除人工架构参数，统一从 checkpoint 读取；
  - cache 写入完整 artifact hash。
- `experiments/state_tokenizer/train_qformer_joint.py`
  - 保存完整架构 metadata；
  - architecture hash 覆盖无参数开关；
  - 新 protocol 与旧产物隔离。
- `xt_ama_adapter/scripts/semantic_filter_syntheticqa_vllm.py`
  - 展开全部 messages；
  - structured output；
  - 保留原始 metadata 与拒绝原因。

### 10.2 新增模块

- SyQA/HumanTrajs → QFormer 训练 manifest 构建器；
- HumanTrajs step QA 质量审核与 grouped split 工具；
- `Qwen32InputSoftTokenBridge`；
- bridge cache/pairs 构建器；
- bridge 训练和验证脚本；
- bridge artifact validator；
- Qwen3-32B direct-input Reader；
- AMA latent-memory runner；
- matched/shuffled/zero/question-only 对照测试。

## 11. 产物目录建议

```text
outputs/ama_latent_memory/
  manifests/
    syqa-v2.json
    humantrajs-stepqa-v1.json
  qformer/
    qformer-k32-qknorm-cebest.pt
    qformer-k32-qknorm-gapbest.pt
    training-report.json
  caches/
    syqa-xbar.npz
    humantrajs-xbar.npz
  retrieval/
    head-qformer-k32.pt
    recall-report.json
  bridge/
    qwen32-input-k32.pt
    training-report.json
  ama_eval/
    latent-memory/
    official-embedding-rag/
    question-only/
    shuffled-latent/
```

每个 `.pt` 必须有同名 JSON report，并记录 Git commit、命令、环境、模型路径/hash、数据
manifest hash、随机种子、选择指标与 best step。

## 12. 开训顺序与停止条件

按以下顺序执行，前一阶段未过闸时不进入下一阶段：

1. 按 `HUMANTRAJS_训练数据流水线.md` 修复时序对齐和泄漏，并只保留独立验证通过的 QA；
2. 固定 grouped train/validation manifest；
3. 修复 QFormer artifact protocol 和所有 `qk_norm` 加载路径；
4. 训练 K32/QK-norm QFormer；
5. 完成 gap、QA、retrieval 三类验收；
6. 冻结 QFormer，重新生成全部 xbar cache；
7. 训练并验收 matched retrieval head；
8. 实现 Qwen3-32B direct-input 路径并先做 shape/gradient smoke test；
9. 对当前 bridge 做冻结 semantic probe，区分 bridge 信息丢失与上游表示问题；
10. 按第 7.5.8 节从相同初始化训练 baseline 和语义保持候选；
11. 通过 matched/shuffled/zero、semantic probe 和 QA validation 闸；
12. 锁定代码和 checkpoint，再运行 AMA 正式评测。

以下任一情况应停止正式训练并先修数据或代码：

- checkpoint 缺少完整架构 metadata；
- cache 的 QFormer hash 与 runtime checkpoint 不一致；
- QK-norm 开关依赖人工记忆；
- validation 按 QA 行随机切分；
- SyQA 仍只包含第一条 message；
- bridge matched 与 shuffled latent 无明显差异；
- AMA test 数据进入任何训练或模型选择流程；
- Reader 实际接收了检索 step 原文，却仍被记作 latent-memory 实验。

## 13. HumanTrajs 正式实现与命令（2026-08-29）

当前实现新增以下可执行阶段，并保持本方案的闸门顺序：

- `evaluate_qformer_qa_controls.py`：固定 validation observations 上执行
  matched / 同轨迹 shuffled / zero / question-only QFormer QA 对照；
- `freeze_qformer_candidate.py`：只有 48 条 P4、96 条 QA control、梯度与丢批检查通过后，
  才补齐架构 metadata 并生成新的 frozen artifact；
- `build_qwen32_bridge_dataset.py`：使用与 frozen QFormer hash 绑定的 retrieval head 做真实
  trajectory 内 top-k，不强塞 gold step；
- `build_qwen32_teacher_cache.py`：32B teacher 读取经过 verifier 的伪 Web observation，
  只缓存 train split 的 answer-span top-k 分布；
- `train_qwen32_bridge.py`：冻结 32B 全部参数，仅训练 K32 bridge，并在 validation 上同时报告
  matched / shuffled / zero / question-only CE。

Bridge 阶段固定只暴露物理 GPU 0、1、2：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 .venv-ama-embedding-cu124/bin/python -u -m \
  experiments.state_tokenizer.build_qwen32_teacher_cache \
  --dataset outputs/ama_latent_memory/bridge/humantrajs-retrieved-top4.npz \
  --model /data1/models/Qwen3-32B \
  --output outputs/ama_latent_memory/bridge/qwen32-teacher-top128

CUDA_VISIBLE_DEVICES=0,1,2 .venv-ama-embedding-cu124/bin/python -u -m \
  experiments.state_tokenizer.train_qwen32_bridge \
  --dataset outputs/ama_latent_memory/bridge/humantrajs-retrieved-top4.npz \
  --teacher-cache outputs/ama_latent_memory/bridge/qwen32-teacher-top128 \
  --model /data1/models/Qwen3-32B \
  --output outputs/ama_latent_memory/bridge/qwen32-input-k32.pt \
  --distill-weight 0.3 --accumulate 8 --max-steps 3000
```

脚本会再次检查 `CUDA_VISIBLE_DEVICES` 必须精确等于 `0,1,2`；不会访问 GPU 3–7。
正式训练前先用同一命令的 `--max-steps 1 --eval-every 1 --eval-limit 2
--distill-weight 0` 做 shape、三卡反传和“仅 Bridge 有梯度”smoke。smoke 输出不得作为正式
checkpoint，正式 run 仍从随机初始化重新开始。
