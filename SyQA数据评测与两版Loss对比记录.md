# SyQA 数据评测、推断与两版 Loss 对比记录

更新时间：2026-09-02

## 1. 实验目的

本次实验使用 SyQA（MolmoWeb SyntheticQA）训练 Q-Former，并将冻结的 Q-Former latent memory 接入 Qwen3-32B，验证以下问题：

1. SyQA 训练出的 32-slot latent 是否能够保留网页状态信息；
2. 512 维 Q-Former latent 经过 Bridge 映射到 Qwen3-32B 的 5120 维输入后，Reader 是否真正使用了 memory；
3. 基础 Bridge loss 与后续增强 loss 的差异，是否能反映到 AMA-WEB 正式准确率上。

本记录区分三种结果：

- Q-Former validation：用于判断 latent 表示是否训练成功；
- Bridge validation/diagnostic：用于判断 Reader 是否使用 matched memory、是否区分 shuffled memory；
- AMA-WEB inference/test：在固定 360 道题上的最终推理结果。

## 2. SyQA Q-Former 训练

### 2.1 数据

SyQA 训练数据包含：

| 项目 | 数量 |
|---|---:|
| observations | 6,645 |
| QA pairs | 11,944 |
| train observations | 5,107 |
| validation observations | 825 |
| session 数 | 988 |
| teacher 完整可用样本 | 2,829 |

官方 AMA-Bench test 没有进入训练、伪 observation 生成、QA 标签生成或 checkpoint 选择流程。

SyQA 本身主要是截图到 QA 的数据，而正式 reader 是 text-only 的 Qwen3-32B，因此实验先用 Qwen3.5-9B-VL 生成带协议标记的视觉转写 observation。该 observation 用于训练期 teacher 和 observation distillation，但不能当作真实 DOM/AXTree。

### 2.2 Q-Former 配置

- Qwen3.5-9B trunk：冻结，使用第 16 层 hidden state；
- StateQFormer：4 层、hidden size 1024、8 heads；
- latent：32 个 query，每个 512 维；
- 训练参数：约 78.9M；
- learning rate：`1e-4`；
- gradient accumulation：4；
- seed：35；
- `qk_norm=true`；
- self-attention：关闭；
- `w_obs`：分别测试 0.5 和 1.0。

### 2.3 Q-Former loss

两条 SyQA Q-Former 训练臂使用相同的总目标，只改变 `KL_obs` 权重：

```text
L_QFormer = CE_gold
          + 0.3 × KL_q
          + w_obs × KL_obs
          + 1.0 × InfoNCE_same_session
```

含义如下：

| 项目 | 含义 |
|---|---|
| `CE_gold` | latent + question 生成 gold answer 的 teacher-forced cross entropy，直接保证 QA 能力 |
| `KL_q` | latent reader 的答案分布逼近完整 observation teacher 的答案分布，权重 0.3 |
| `KL_obs` | 约束 latent 保留当前网页状态/observation 信息，权重为 0.5 或 1.0 |
| `InfoNCE_same_session` | 区分同一 session 中不同 step 的 latent，权重 1.0、temperature 0.05 |

### 2.4 Q-Former validation 结果

| Q-Former 臂 | 最优 validation answer CE | CE 选点 step | 最优 P4 gap | gap 选点 step |
|---|---:|---:|---:|---:|
| `w_obs=0.5` | 2.3050 | 7,000 | 0.3003 | 8,500 |
| `w_obs=1.0` | **2.2523** | 7,500 | 0.2757 | 5,750 |

P4 gap 定义为：

```text
gap = KL_mismatched - KL_matched
```

gap 为正说明 matched latent 比错配 latent 更能保留对应 observation 的信息。最终 AMA-WEB 链路使用的是 `w_obs=1.0` 的 CE-best candidate：

```text
outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt
outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt
```

## 3. 第一次：基础 Bridge 训练与 AMA-WEB 推断

### 3.1 Bridge 结构和训练数据

Q-Former 输出为 `32 × 512`，Qwen3-32B 的输入 embedding 为 5120 维，因此单独训练一个 512→5120 Bridge。Q-Former、retrieval head 和 Qwen3-32B Reader 全部冻结，仅更新 Bridge。

Bridge 训练数据：

```text
outputs/ama_latent_memory/syqa/post-qformer/bridge/syqa-matched-k32.npz
```

teacher cache：

```text
outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-teacher-top128
```

基础 Bridge checkpoint：

```text
outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms.pt
```

### 3.2 第一次 loss

第一次 Bridge 使用基础目标：

```text
L_bridge_v1 = CE_gold + 0.30 × KL_teacher
```

具体含义：

- `CE_gold`：Bridge 输出的 soft token 加 question 后，Qwen3-32B 生成 gold answer 的 token-level CE；
- `KL_teacher`：student（Bridge latent + question）的答案分布，逼近 teacher（可靠原始文本 observation + question）的答案分布；
- `RMS calibration`：把 Bridge 输出的 token RMS 对齐到 Qwen3-32B input embedding 的 RMS，但这是硬校准，不是额外的可学习 loss。

第一次没有加入：

- hidden-state distillation；
- matched/shuffled hard-negative margin；
- xbar 内容重建 loss；
- xbar 关系几何 loss。

因此第一次 loss 对 latent 的样本特异信息只有间接约束：只要最终答案分布和 gold answer 变好，Bridge 就可能找到某种对 Reader 有效、但不一定充分保留 xbar 语义的映射。

### 3.3 第一次 Bridge validation

在 validation diagnostic 上记录到：

| 输入条件 | Answer CE |
|---|---:|
| matched memory | **2.4727** |
| shuffled memory | 2.5172 |
| zero memory | 4.9498 |
| question-only | 4.9883 |

关键差值：

```text
shuffled - matched = 0.0445
```

这说明：

1. memory 不是完全没进入 Reader，因为 matched CE 明显低于 zero memory/question-only；
2. Reader 能感知 matched 与 shuffled 的差异，但差值只有 0.0445，memory 的样本特异信息仍然较弱；
3. zero memory 和 question-only 的 CE 接近，说明仅靠问题本身无法达到 matched memory 的效果。

### 3.4 第一次 AMA-WEB 推断

固定使用：

- 30 条有效 trajectory；
- 360 道 QA；
- episode 184 整条排除；
- matched memory；
- retrieval `top_k=1`；
- Qwen3-32B Reader；
- Qwen3-32B LLM-as-Judge。

结果：

```text
87 / 360 = 24.17%
```

按 QA 类型：

| 类型 | 正确数 | 准确率 |
|---|---:|---:|
| A | 31/121 | 25.62% |
| B | 29/90 | 32.22% |
| C | 15/90 | 16.67% |
| D | 12/59 | 20.34% |

这个结果只代表 `SyQA Q-Former + 基础 Bridge + matched top_k=1`，不能单独解释为 Q-Former 的净提升，因为当时没有相同 Reader、相同生成参数和相同 Judge 下的 question-only、shuffled-memory、no-latent-memory 正式对照。

## 4. 后续 validation 与推断分析

基础 Bridge 的 validation 暴露出一个问题：

```text
matched CE < shuffled CE，但差距较小
```

因此后续分析不再只看 matched CE，而是同时看：

- matched answer CE；
- shuffled answer CE；
- `shuffled - matched` 的 memory discrimination gap；
- zero memory 和 question-only 的 CE；
- Qwen3-32B 多层 hidden representation 的相似度；
- Bridge 输出内容对 xbar 的可恢复性和关系结构保持。

这里的逻辑是：

1. `matched < zero/question-only`：说明 memory 有用；
2. `shuffled > matched`：说明 memory 不是任意 memory，而是需要匹配当前样本；
3. hidden similarity 高：说明 Bridge 输出在 Reader 内部表征空间中接近 teacher；
4. xbar reconstruction/关系保持较好：说明 Bridge 没有只优化答案 CE，而是保留了更多样本特异信息。

## 5. 中间版本：semantic Bridge validation

在最终 delta-margin 版本之前，曾测试加入语义保持目标的 `qwen32-input-k32-rms-semantic.pt`。该版本使用：

```text
L_bridge_semantic = CE_gold
                  + 0.30 × KL_teacher
                  + 0.10 × hidden_distill
                  + 0.05 × L_reconstruction
                  + 0.05 × L_relation
```

其中：

- `hidden_distill`：对齐 Qwen3-32B 第 16、32、48 层 hidden；
- `L_reconstruction`：从 Bridge content 重建 xbar，使用 cosine + Smooth L1；
- `L_relation`：保持样本内/样本间的关系几何。

在 step 3000 的 validation diagnostic：

| 指标 | semantic Bridge |
|---|---:|
| matched answer CE | 2.4729 |
| shuffled answer CE | 2.5168 |
| zero answer CE | 4.9498 |
| question-only answer CE | 4.9883 |
| hidden/content cosine | 0.9839 |
| relation loss | 0.1287 |

semantic Bridge 的 answer CE 与基础 Bridge 接近，但它提供了额外的内容保持诊断；这说明仅增加语义重建项还没有明显改变 AMA 答案能力，因此后续进一步加入了直接针对 matched/shuffled 区分的 margin loss。

## 6. 第二次：Delta-margin Bridge 训练

### 6.1 第二次 loss

第二次从 semantic Bridge checkpoint 初始化，使用 2048 条训练样本、总计 1000 steps（pilot 250 steps + continuation 750 steps），学习率降为 `2e-5`。

最终 loss 为：

```text
L_bridge_v2 = CE_gold
            + 0.30 × KL_teacher
            + 0.10 × L_hidden_delta
            + 0.20 × L_margin
```

其中：

#### 1. Answer CE

```text
L_CE = CE(student_answer, gold_answer)
```

保留基础 Bridge 对最终答案的直接监督。

#### 2. Teacher answer KL

```text
L_KL = KL(teacher_answer_distribution || student_answer_distribution)
```

继续让 Bridge latent 的 Reader 输出接近完整 observation teacher，权重仍为 0.30。

#### 3. Hidden delta distillation

第二次不是简单要求 student hidden 逐维复制 teacher hidden，而是对 hidden difference 做 whitening 后再对齐：

```text
delta_teacher = hidden_teacher - hidden_question_only
delta_student = hidden_student - hidden_question_only
L_hidden_delta = cosine_loss(whiten(delta_student), whiten(delta_teacher))
```

使用 Qwen3-32B 第 16/32/48 层 hidden，并设置 `hidden_whiten_floor_ratio=0.10`。减去 question-only hidden 的目的，是尽量把监督集中在“memory 相对于问题额外带来的信息”，而不是让 Bridge 只学习问题本身的共同表征。

#### 4. Matched/shuffled margin loss

对同一道题同时计算 matched 和 shuffled memory 的答案 CE：

```text
L_margin = max(0, margin + CE_matched - CE_shuffled)
```

本次设置：

```text
margin = 0.10
weight = 0.20
```

它直接要求：

```text
CE_shuffled - CE_matched >= 0.10
```

因此相较于第一版只要求 matched CE 变小，第二版额外要求“正确 memory 必须比错误 memory 更有用”。这正是两版 loss 的核心区别。

### 6.2 两版 loss 对比

| 维度 | 第一版基础 Bridge | 第二版 delta-margin Bridge |
|---|---|---|
| 答案监督 | `CE_gold` | 保留 |
| teacher 答案分布 | `0.30×KL_teacher` | 保留 |
| Reader hidden | 无 | `0.10×hidden_delta` |
| matched/shuffled 区分 | 只作为诊断 | `0.20×margin loss` 直接训练 |
| xbar 语义重建 | 无 | 通过 hidden delta 间接约束；最终版本未启用 semantic decoder reconstruction |
| 训练初始化 | random/base Bridge | 从 semantic Bridge 初始化 |
| 学习率 | `1e-4` | `2e-5` |
| 训练预算 | 3,000 steps | 1,000 steps，总计含 pilot 250 steps |
| 主要目标 | 让 Reader 能使用 latent | 让 Reader 使用正确且样本匹配的 latent |

第一版更像“让 Bridge 学会把 latent 变成可读 soft token”；第二版进一步要求“这些 soft token 保留的内容必须能区分正确 memory 和 shuffled memory”。

## 7. 第二次 validation 结果

Delta-margin 训练日志在 step 1000 的结果为：

| 指标 | step 1000 |
|---|---:|
| matched answer CE | **2.4465** |
| shuffled answer CE | 2.5103 |
| zero answer CE | 4.9498 |
| question-only answer CE | 4.9883 |
| hidden distill loss | 0.9402 |
| margin loss | 0.0573 |

对应 discrimination gap：

```text
2.5103 - 2.4465 = 0.0638
```

与第一次记录的 `0.0445` 相比，matched/shuffled 差异扩大，且 matched CE 降低。这说明第二版 loss 在 validation diagnostic 上同时改善了：

- 答案生成能力；
- 对正确 memory 的依赖；
- 对 shuffled memory 的区分能力。

但这个 validation 结论仍然不是 AMA test 结论，因为它是在 Bridge 训练数据/固定 validation protocol 上测量的，不能替代最终 360 题评测。

## 8. 第二次 AMA-WEB 推断结果

保持第一次的 AMA-WEB 数据、Q-Former、retrieval head、Reader、`matched + top_k=1` 和 Judge 口径不变，只替换 Bridge：

```text
outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-delta-margin-total1000.pt
```

结果：

```text
99 / 360 = 27.50%
```

相对第一版：

```text
24.17% → 27.50%
绝对提升：+3.33 个百分点
相对提升：+13.79%
```

逐题配对结果：

- 31 题：错 → 对；
- 19 题：对 → 错；
- 68 题：两版均正确；
- 242 题：两版均错误。

这说明第二版并非简单地把所有答案都推向正确，而是发生了有方向的逐题变化；净变化为 `31-19=12` 道题，对应总体从 87 道正确增加到 99 道正确。

## 9. 最终推断

目前证据支持以下较谨慎的结论：

1. **SyQA 训练出的 Q-Former 能产生可用的网页 latent 表示。** Q-Former validation 的 answer CE 和 P4 gap 均为正向，且 zero/question-only 对照明显更差。
2. **基础 Bridge 已经能把 latent 注入 Qwen3-32B。** matched CE 低于 zero/question-only，说明 latent 确实进入 Reader 并被使用。
3. **基础 Bridge 对样本特异信息的保留还不够强。** matched 与 shuffled 的 validation 差距只有 0.0445，说明“有 memory”和“使用正确 memory”之间仍有差异。
4. **delta-margin loss 针对性补足了 matched/shuffled 区分目标。** validation 中 discrimination gap 从 0.0445 增加到约 0.0638，AMA-WEB 单次测试准确率同步从 24.17% 增至 27.50%。
5. **第二版提升方向得到支持，但尚不能称为稳定增益。** 当前只有一批 360 题、单个 seed 和单次 Judge 评测；还需要 question-only、shuffled-memory、no-latent-memory 及多 seed 复测，才能估计 Q-Former/Bridge 的独立贡献和方差。

## 10. 关键日志和结果位置

Q-Former：

```text
/home/cbd/project/residual-mem/logs/ama_latent_memory/syqa/qformer-K32-syqa-v1-obs0.5.log
/home/cbd/project/residual-mem/logs/ama_latent_memory/syqa/qformer-K32-syqa-v1-obs1.0.log
```

第一版 Bridge：

```text
/home/cbd/project/residual-mem/logs/ama_latent_memory/syqa/post-qformer/qwen32-bridge-formal.log
/home/cbd/project/residual-mem/outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms.pt
```

semantic Bridge：

```text
/home/cbd/project/residual-mem/logs/ama_latent_memory/syqa/post-qformer/qwen32-bridge-semantic-formal.log
/home/cbd/project/residual-mem/outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-semantic.pt
```

第二版 delta-margin Bridge：

```text
/home/cbd/project/residual-mem/logs/ama_latent_memory/syqa/post-qformer/qwen32-delta-margin-pilot250.log
/home/cbd/project/residual-mem/logs/ama_latent_memory/syqa/post-qformer/qwen32-delta-margin-total1000.log
/home/cbd/project/residual-mem/outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-delta-margin-total1000.pt
```

AMA-WEB 推断结果：

```text
/home/cbd/project/residual-mem/outputs/ama_latent_memory/ama_eval/web-latent-formal/evaluation.json
/home/cbd/project/residual-mem/outputs/ama_latent_memory/ama_eval/web-latent-delta-margin-total1000/
```

