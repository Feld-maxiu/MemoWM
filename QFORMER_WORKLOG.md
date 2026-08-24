# Q-Former 状态 tokenizer 工作日志

> 承接 `WMA_RAG_WORKLOG.md`（记录到 Phase 3 reader connector 为止）。
> 结论沉淀在 `WORLDMEMARENA_TOKENIZER_RAG.md`；这里记过程与实现。
>
> 日期：2026-08-23

---

## 1. 起因

**Stage C 没达标，退路条款被触发。** 全量 233 题，按 text 是否答对拆开做配对 McNemar：

| 条件 | Correct | Halluc | Omission |
|---|---:|---:|---:|
| text（上界） | 0.2961 | 0.1030 | 0.5966 |
| union（重建 CE） | 0.2275 | 0.1588 | 0.6137 |
| qa-distill（answer CE + KL） | 0.1717 | 0.0815 | 0.7468 |

```
text 能答对子集 n= 69:  只union对= 4   只qa-distill对= 8   p=0.3877  ← 不显著
text 答不对子集 n=164:  只union对=21   只qa-distill对= 4   p=0.0009  ← 显著
```

判据（掉 < 5 点）：union −6.9，qa-distill −12.4，**都不达标**。
两个目标函数完全不同的 connector 在保真度子集上统计无法区分，
说明约束在**它们共享的部分**——固定的 64 槽池化表示。

**但那次对比是脏的（我的 bug）。** `reduction="batchmean"` 在 `(1, L, V)` 上除以第一维 1，
是对答案位置**求和**而非求平均，实测旧/新 = **17.00× = answer_len**。
`--distill-weight 1.0` 实际是 10–30 倍权重，
所以「ℒ_Distill 无效」**予以撤回**。同批修掉的另外三处：
seed 在模块构造之后（两个文件）、验证集取索引序前 N 行、协议串本地重复声明。
**四个的共同根因：仓库里没有任何测试碰过任何训练脚本。**

### 固定池化实测（956 条 web 观察）

```
组          槽数   实际有效   两两余弦   能量占比
image        32       32      0.5877     74.3%   ← 塌缩主因
detail       16       16      0.6920     19.2%
context      16        3      0.8741      6.4%   ← 13 槽结构性为零
```

单条观察 995 token（880 图像 + 114 文本）。图像走 22×40 patch 网格，
`adaptive_avg_pool2d` 压到 4×8（`key_pooling.py:270-274`），**每槽平均掉 27.5 个 token**。
不同网页顶上都是 chrome、中间内容区、大片白底——按位置平均本身就在抹掉页面身份。
文本侧反而槽位过剩且 13 槽恒零：**预算分配和信息量是反的**。

### 因果注意力让三样东西静默失效

1. 图像槽对序列化器结构性免疫（看不到排在后面的文本）；
2. context 槽恒定 3/16 有效（合成 AXTree 每条只产 3 个节点）；
3. **观察 prompt 完全惰性**——序列是 `[image] → DOM → instruction`，
   图像和 DOM 都看不到指令，而指令自己的 token 因 `PROMPT_SLOTS = 0` 被整段丢弃。
   `fixed_prompt.py:12-15` 那句「保留可见文本、输入值、控件类型、选中/聚焦/启用状态及空间关系」
   **对 64 个槽的因果影响精确为零**。

第 3 条决定了损失设计。

---

## 2. 设计

```
H_t (1129×4096, 冻结主干第16层)      ← 全部位置，含 29 个指令 token
   ↓  P_ρ：64 个可学习查询 × [cross-attn → self-attn → FFN] × L
xbar (64×512)  →  connector  →  64 soft token  →  冻结 reader
```

**不是「学习式投影」，是三件事同时被修掉**：查询按内容寻址而非按位置平均；
交叉注意力不是因果的，每个查询同时看图像、文本和指令；槽位预算由内容决定。

**两条不可动的约束**：

1. **写侧必须与 query 无关**——WMA adapter 在 ingest 时编码一次、缓存 latent、
   此后每个问题复用。query 条件化会作废整个存储模型。问题只进 reader 的 prompt。
2. **$K_x=64$ 锁死**——`_check_latents` 硬拒非 `(64,512)`，A2 位置表 64 槽，
   connector 的 `rank_embedding` 是 `(64,4096)`。
   **本轮拿不到压缩比主张**，收益必须全部来自质量。

### 损失只有两项

$\mathcal L_{\text{answer}}$（金标答案 CE，问题在 reader prompt 里）
$+\ w\cdot\mathcal L_{\text{distill}}$（冻结生成器读原文轨迹的 KL，按答案位置求平均）。

**不加 §5.2 式 (9a) 的四个锚点**（sem/ocr/vis/state），三条理由：

1. 那句 prompt 已经在要求四个子空间，它不起作用只因为因果注意力挡住了；
   **交叉注意力一上，prompt 自己就开始工作**；
2. §5.4 警告的是 tokenizer 与 **WM** 共训（WM 有动机让残差虚假变小），
   reader 没有这个动机，两项损失都在**奖励**保留信息；
3. $\mathcal L_{\text{ocr}}$ 正是让 connector「答完接着复述观察」的那个目标。

**顺带买到**：不要 $\mathcal L_{\text{state}}$ ⇒ 不要 memory_points ⇒
**tokenizer 训练不接触 WMA 的任何标注**，原本「效果 vs 主张干净度」的取舍消失。

**已知代价**：$\mathcal L_{\text{distill}}$ 的教师读 `fused_text`（约 89 token 字幕），
**不含截图空间细节**，图像内容只被 QA 恰好问到的部分约束。
对策是把 §5.4 的指标做成**监控而非损失**（见 §3.5）。

---

## 3. 实现

### 3.1 `residualmem/latent/qformer.py`

**`StateQFormer`** — 端口自 JAX `residualmem/latent/tokenizer.py:33-89` 的 `resample()`，
**补上了 latents 之间的 self-attention**（JAX 版是纯 cross-attention，
查询之间无法分工，也无法避免彼此重复）。

```
forward(hidden_states, modality_ids, positions, key_mask) -> (B, 64, 512)

context = LayerNorm(Linear(4096→H)(H_t))          # H_t 先 .to(权重 dtype)，bf16→fp32
        + modality_embedding(modality_ids)         # nn.Embedding(4, H)，零初始化
        + FourierPositionEncoding(positions)       # 固定 buffer，非参数
context = context * key_mask[..., None]            # padding 归零，非仅 mask

latents = queries.expand(B, -1, -1)                # nn.Parameter(64, H), std=0.02
for block in blocks:                               # 前置归一化残差
    latents += cross_out(SDPA(cross_q(LN(latents)), cross_k(context), cross_v(context),
                              attn_mask=key_mask[:, None, None, :]))   # (B,1,1,T) 广播
    latents += self_out(SDPA(self_q(LN(latents)), self_k(...), self_v(...)))
    latents += FFN(LN(latents))                    # Linear(H→4H) → GELU(tanh) → Linear(4H→H)
return LayerNorm(Linear(H→512, bias=False)(latents))
```

`key_mask` 只用于抵消同批不同长度的右填充——与池化路径不同，
**单条观察内没有「无效输入 token」的概念**（池化路径里 detail/context 的空槽是真的空）。

参数量（`hidden=1024, heads=8, layers=4`）：

| 组件 | 参数 |
|---|---:|
| `blocks`（4 × 16.788M） | 67.154M |
| `input_projection` (4096→1024) | 4.195M |
| `output_projection` (1024→512) | 0.524M |
| `queries` (64×1024) | 0.066M |
| 其余（模态嵌入、两处 LayerNorm） | 0.007M |
| **StateQFormer 合计** | **71.946M** |

每个 block 16.788M = cross 4.194M + self 4.194M + FFN 8.394M + 三处 LayerNorm。
`layers=6` 时 105.52M。

**`QFormerStateReader`** — P_ρ + connector 打包成**单一产物**。
理由：resampler 的输出空间只对与它一起训练的 connector 有意义，
而这个仓库已经为「两半坐标系在同名文件下漂移」付过一次代价（见 `pca_binding`）。

```python
def encode(self, ...) -> (xbar, valid)   # 写侧，query 无关，valid 恒 True
def forward(self, ...) -> (soft, xbar, valid)
```

**`qformer_hash`** — 对排序后的 `state_dict` 逐张量哈希，替代 `pca_sha256` 溯源链。

### 3.2 `experiments/state_tokenizer/trunk_states.py`

主干跑到第 16 层，返回**整条序列**而非 64 个池化槽。
不是给 `encode` 加一个 flag——那个 dataclass 是全 numpy 的，所有 eval 消费者都依赖它。

```python
inputs, truncated, kept, _ = prepare_inputs(processor, image, dom, OBSERVATION_PROMPT,
                                            max_length, "instruct")   # 与 encode 完全一致
if truncated: raise                      # 静默截断会改变 H_t 而调用方看不见
indices = modality_indices(...)          # (image, dom, instruction) 三组索引
hook = layers[15].register_forward_hook(lambda ...: raise _StopAtLayer)
with torch.no_grad():                    # ← 不是 inference_mode
    try: model.model(**inputs, use_cache=False)
    except _StopAtLayer: pass
```

返回 `TrunkStates(hidden (T,4096) bf16, modality_ids (T,), positions (T,) ∈[0,1])`。
`modality_labels` 把**每个位置**都标上：`IMAGE=0 / DOM=1 / INSTRUCTION=2 / WRAPPER=3`——
`WRAPPER` 是 chat 模板加四个 DOM/指令标记 token，`modality_indices` 有意排除它们，
但整段丢掉会让序列出现空洞。实测一条观察：`image=880 dom=167 instruction=29 wrapper=53`。

`collate(batch)` 右填充到批内最长，padding 位置在**每个张量里都归零**且 mask 为 False。

### 3.3 `experiments/state_tokenizer/reader_losses.py`

两项损失的**唯一实现**，`train_reader_qa` 与 `train_qformer_joint` 共用——
两份拷贝正是 KL 被写错的原因。

```python
inputs = cat(memory_embeds, question_embeds, answer_embeds)     # 记忆 | 指令+问题 | 答案
labels = cat(-100 × memory, -100 × question, answer_ids)        # 只在答案上算 CE
student = run(latent, valid)                                    # 记忆 = 64 soft token
teacher = run(embed(text_ids), ones)  # no_grad                 # 记忆 = fused_text 原文
span = slice(-answer_len - 1, -1)                               # 两条序列末端对齐
kl = F.kl_div(log_softmax(student.logits[:, span]).reshape(answer_len, -1),
              log_softmax(teacher.logits[:, span]).reshape(answer_len, -1),
              reduction="batchmean", log_target=True)           # ← reshape 是修复本身
return student.loss + weight * kl
```

`INSTRUCTION` 与 `Qwen35LatentReader.answer` 推理时构造的**逐字相同**——
训练在另一种形状上，正是第一版 connector「答完接着复述观察」的成因。

### 3.4 数据侧

**`build_qformer_qa_pairs.py`** — 存**引用**而非 blob。
`build_reader_qa_pairs` 把 64×512 的 xbar 内联，每个问题一份，
9,041 对 / 3,182 条观察 = **1.22 GB**；这里存 `(sample_id, record_index)`，**33.4 MB**，
每条观察只存一份。计数完全一致（7776 训练 / 1265 验证 / 154 样本），说明 join 未变。

依赖抽取持久化了 `synthetic_axtree`：`wma_extract_xbar.py` 原本只存 `synthetic_axtree_chars`，
而重跑主干需要树本身。文件里已有同样理由的先例（`fused_text` 存下来而不是事后重建）。
缺该字段的旧抽取在此**直接报错**，不静默产出空树。
重抽后验证 **xbar 逐位相同**、`fused_text` 相同、新存树长度与旧 `chars` 字段一致。

**`ObservationStore`** — `(sample_id, index)` → 主干输入 + 该观察的**固定池化 xbar**（`m11` arm），
后者是监控的配对基线。records 按样本缓存，pooled xbar 按需惰性加载。

### 3.5 `experiments/state_tokenizer/train_qformer_joint.py`

**微批 1 + 梯度累积，而非真批量**（偏离批准方案，已说明）：
student 的记忆是 64 个 soft token，teacher 的是变长文本，
真批量要在两种不同 padding 上对齐答案跨度——正是 KL 那个 bug 所属的类别。
累积复用已验证的单样本前向，代价约 7%。

```python
for step in 1..max_steps:
    optimizer.zero_grad()
    for _ in range(accumulate):                       # 默认 4
        row = rng.choice(train_rows)
        (loss_for(row, distill_weight) / accumulate).backward()
    clip_grad_norm_(joint.parameters(), 5.0);  optimizer.step()   # AdamW lr 1e-4
```

`loss_for` 一步：`ObservationStore` 取记录 → 开图 → `trunk_states`（no_grad）
→ `collate([·])` → `joint(...)` → `answer_ce_and_distill_kl`。

**eval**（每 100 步，96 行固定随机子集，`weight=0`）同时算三条监控，
对比对象是**同一批观察**的固定池化 xbar：

| 指标（均在跨观察去均值后） | 抓什么 |
|---|---|
| `pairwise_cosine` | 状态之间是否相互区分 |
| `effective_rank` | 变化是否困在低维子空间 |
| `mean_to_deviation` | 信息量相对常量偏移是否过小 |

任一劣于固定池化即置 `collapse_regressed`，届时定点加回**对应的那一个**锚点。
早停：patience 8 次 eval，按 val answer CE。
产物用 `save_bridge` 存单一 `QFormerStateReader`，metadata 带 `qformer_sha256`。

### 3.6 测试（25 项，全新增）

| 文件 | 锁住的性质 |
|---|---|
| `test_qformer.py`（10） | 形状契约；**指令段改变输出**（当前管线证明不具备的那条）；padding 不可达输出；梯度到查询；确定性（可冻结）；模态嵌入零初始化；哈希随权重移动 |
| `test_trunk_states.py`（5） | **`inference_mode` 中毒 vs `no_grad`**；模态标签无空洞；collate 填充与掩码 |
| `test_reader_losses.py`（5） | **KL 是每位置平均而非求和**；权重只缩放 KL；空记忆禁用 KL；梯度到 latent |
| `test_qformer_monitors.py`（5） | **常量偏移不得被读成塌缩**；两种失效模式各自的指标；尺度不变 |

**三处对抗性自检**（避免写出空过的测试）：
padding 测试不 mask 时差 1.35、mask 后 1.31e-06（判别力 6 个数量级）；
指令段 mask 掉后差异**精确为 0**，证明差异确实来自那一段；
KL 测试自带守卫「若求和与求平均在此重合则本测试无判别力」。

---

## 4. 过程中被纠正的三次

**桩模型太简单。** 最初 `head(inputs_embeds)` 是逐位置函数，没有上下文混合，
记忆影响不到答案位置的 logits，于是 KL 恒 0、梯度到不了 latent——两条测试因此失败。
是桩的问题不是被测代码的问题，加上因果累积均值后通过。

**监控函数口径错了，差点被当成灾难性结果。** step 400 时报 `COLLAPSE-REGRESSED`：
`cos 1.0000/0.5504`，但**有效秩 30.5 与之矛盾**（真塌缩应趋近 1）。
载入 checkpoint 在 40 条验证观察上直接量：

| | 未去均值余弦 | **去均值余弦** | ‖均值‖/‖偏差‖ | 有效秩 |
|---|---:|---:|---:|---:|
| Q-Former | +1.0000 | **+0.0389** | **251.50** | 14.7 |
| 固定池化 | +0.5739 | −0.0225 | 1.20 | 28.8 |

`spread()` 算有效秩时去了均值、算余弦时没有，而固定池化的 xbar 因 group×channel 标准化
本来就近零均值——**两个表示在未去均值的余弦下根本不可比**。去均值后两边都接近正交，
**没有塌缩**。但同一次测量挖出一个真实的病：**‖均值‖/‖偏差‖ = 251.5**，
有信息的部分只占范数的 0.4%。下游 A1 与 connector 都以 `LayerNorm(512)` 开头会吃掉大部分，
故先作为独立指标观察，不改架构。
**三条训练没有重启**——`spread()` 不参与任何损失；但已产出日志与 checkpoint metadata 里的
`monitors` 是旧口径，**不能引用**。

**有效秩测不出「状态都一样」。** 写测试时才发现：去均值后有效秩是**尺度不变**的，
20 个近乎相同的状态加各向同性微噪声，去均值后就是全秩（18.6）。
那是 `mean_to_deviation` 的职责。测试因此拆成两条，各管一种失效模式。

### 环境上的两个坑（都是我的）

`/tmp` 被我自己的冒烟 checkpoint（364 MB + 140 MB）填满，
torch 在 **import 期**要往临时目录写生成的模块模板，于是两个配置死于
`OSError: [Errno 28]`——报错位置离原因很远。修法：删掉自己的文件 + `TMPDIR` 指向工作区。
另外 **zsh 默认不做词分割**，`for cfg in "A 1 4 1.0"; do set -- $cfg` 在 bash 下能拆成四个词，
zsh 下不拆，于是启动命令带着 `--device cuda:` 和空的层数/权重。

---

## 5. 官方管线三臂对比（n=1459，`agent/arena/web` 全量）

前两个自建离线对照各自都测出了我自己引入的混淆（prompt 外壳、思维链吃掉 token 预算），
所以改走 `python -m eval_framework.cli` ——官方检索、官方 reader prompt、
官方 `mm_mode: image`、官方判题。我方只提供 adapter 读的那几个产物。

### 主表：六列有区分度的

| | QA-C↑ | QA-H↓ | QA-O↓ | RC↑ | F1↑ | BLEU-1↑ |
|---|---:|---:|---:|---:|---:|---:|
| Raw-Fused 官方对照（v6） | 0.5415 | 0.2132 | 0.2454 | 0.6540 | 0.3010 | 0.1568 |
| **ResidualMem，观察行交原文（v3）** | **0.5764** | 0.1885 | **0.2351** | **0.6858** | **0.3295** | **0.1836** |
| ResidualMem，观察行交 latent（v4） | 0.5007 | **0.1332** | 0.3661 | 0.6851 | 0.2418 | 0.1793 |

### 两条臂的真实配置（我一度归因错，见下）

| | 检索 | **答题时观察行以什么形式进 prompt** | answer prompt tokens |
|---|---|---|---:|
| v3 | 并集头（latent 检索） | **原文**——运行时 connector 尚不存在 | 2319（中位） |
| v4 | 并集头（同一个头） | **64 个 soft token** | 0（原生答题绕过框架计量） |
| v6 | 官方 Qwen3-VL-Embedding | 原文 + 最多 5 张截图 | 2850 |

**两条臂用同一个检索头，唯一差别是观察行交原文还是交 latent。**

### 记忆侧六列：三臂几乎完全相同

```
              Recall    Corr   Hallu   Irrel  Update  IntRej
对照          0.8890  0.8707  0.0015  0.1279  0.5475  0.0000
v4            0.8871  0.8786  0.0015  0.1199  0.5398  0.0000
v3            0.8912  0.8704  0.0029  0.1267  0.5464  0.0000
```

它们测的是 **session 阶段的记忆抽取**，不经过 tokenizer / 检索头 / connector。
**报成 12 列对比是充数**，主表只留上面六列。`IntRej` 三臂皆 0，是退化指标。

### 两条结论，强度不同，不能混报

**① 「latent 检索 + 原文生成」净正，全六列赢官方对照。**
在双方都用原文生成的条件下，我们的检索头比官方 Qwen3-VL-Embedding 检索得更好
（QA-C +3.5、RC +3.2），**而且用更少的上下文**（2319 vs 2850 prompt token）。

**② 「latent 替代原文」净负。**
```
v3 交原文    QA-C 0.5764   QA-H 0.1885   QA-O 0.2351
v4 交 latent QA-C 0.5007   QA-H 0.1332   QA-O 0.3661
```
Correct 掉 **7.6 点**，幻觉降 5.5 点，弃答涨 13.1 点。
**这正是 Phase 3 一直想测而没测准的那个数**，方向与 Stage C 一致（latent 不如它替代的文本忠实），
现在是官方口径、全量题目。

> ⚠️ **一次归因错误，已收回。** 我最初把 v3→v4 的 7.6 点读成「修 KL bug 的代价」。
> 但 v3 完成于 2026-08-23 08:12:49，而最早的 connector 产物是 08:56:12——**晚 44 分钟**，
> 运行时它根本不存在。token 用量证实（v3 中位 2319 = 读原文，v4 为 0 = 原生 latent 答题）。
> v3 里没有被修的那个东西，掉分与 KL 无关。

### 检索头：Q-Former 的状态可检索，但明显更差

`head_recall.py` 用 `same_sample_rank_metrics`（同 session 内排名，即 benchmark 的真实任务）：

| 头 | 槽 | 同 sample 内 R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|---:|
| 固定池化（WMA 语料） | 64 | **0.3429** | 0.6333 | 0.7726 | 0.4825 |
| 固定池化（并集语料） | 64 | 0.3060 | 0.5917 | 0.7405 | 0.4391 |
| **Q-Former K=16** | 16 | **0.1286** | 0.3750 | 0.5393 | 0.2584 |
| *chance* | | *0.0369* | | | |

**硬闸未触发**（3.5× chance，状态确实可检索），但比固定池化差 2.7 倍。
差距里有三项还没分开：槽数 16 vs 64、K16-nosa 本身撞上限没训完、
**以及 Q-Former 的训练目标里从来没有检索信号**（InfoNCE 只作用在冻结状态之上的头）。
K16-long / K64-long 正在分离前两项。

两处顺带更正：

- **并集头在 WMA 域内反而比 WMA 专用头差**（0.3060 vs 0.3429），
  它用域内精度换了跨域鲁棒性。
- 我曾说全局 R@k「会好看而实际废」——**说过头了**。实测全局（840-way）比同 sample 内
  （39-way）**更低**（0.2940 vs 0.3429），干扰项更多而已。
  同 sample 内该做主判据的正当理由是**它就是 benchmark 的任务**，不是它更难。

### 出表时必须写明

1. **不与论文 Table 2 并排**：答题+判题都是本地 Qwen3.5-9B（**同一模型既答题又判题**），
   只跑 `agent/arena/web` 27 个样本（论文 461）。只做三臂横向比较；最终数字换 GPT-5.4 重判。
2. 记忆侧六列三臂不变，不作为对比证据。
3. **带图行 ours 3.4% vs control 15.6%**——我们的观察行**确实带 `image_path`**
   （`residualmem_instruct_adapter.py:212`，不是漏设），是抢不到 top-10。
   `mm_mode` 对齐修不了这个不对等，QA-C 的差里含此混淆。

---

## 6. 当前状态

**端到端已验证**（真 9B）：`H_t (1129,4096) bf16` → `xbar (1,64,512)` → `soft (1,64,4096)`，
`loss=9.6824 CE=7.6435 KL=2.0389`，查询梯度 `|g|=1.18e+03` 有限。

**在跑**（8 卡）：

| GPU | 任务 | 可训练 | 进度 |
|---|---|---:|---|
| 0 | connector 重训（Phase 0 基线） | 19.1M | val CE 1.9231 @3500 |
| 1 | Q-Former A：layers=4, distill=1.0 | 91.1M | val CE 2.1232 @500 |
| 2 | Q-Former B：layers=6, distill=1.0 | 124.7M | val CE 2.0966 @500 |
| 3 | Q-Former C：layers=4, distill=0.3 | 91.1M | val CE 1.9476 @700 |
| 4–7 | LLM 服务（8 副本） | — | 供 Stage C 与官方评测 |

C 已接近基线，而它是从零学表示的。**但这些 val CE 之间不能严格比**：
基线用 `reader-qa-pairs` 的 128 行，Q-Former 用 `qformer-qa-pairs` 的 96 行，
样本级划分同源但行子集与大小不同。有效对比仍然只有 Stage C。

**要盯的**：有效秩在三条上都持续下降（固定池化 64.7，现在 26–30）。这是真正的坍塌信号。

---

## 7. 待办 / 门槛

| 闸 | 判据 |
|---|---|
| Gate 1（Stage C） | text 能答对子集上 latent 的 Correct 显著高于 Phase 0 基线（配对 McNemar p<0.05）。**需先扩大 n**——现在只有 69 |
| Gate 2（官方评测） | `agent/arena/web` 全量，R@10 与 `correct_ratio` 对比 v3 |
| 坍塌监控 | 三条指标任一劣于固定池化即判失败 |

**Phase 2 下游重训**：冻结 P_ρ → 写标准化产物 → 重训 A1 → 重训 A2 → 重建 L16 restorer。
协议串需 bump 到 `_v2`（现在**不编码维度**，改了只会得到 size-mismatch traceback）。

⚠️ **对批准方案的一处修正**：方案写「L16 baseline 约 10 行就能保住」。
更准确的说法是 `Layer16Restorer` 用的是 `components.transpose()`，
**只在正交时等于伪逆**；学习式投影一般不正交，需额外存最小二乘伪逆并给 restorer 加可选 buffer。

**比特预算口径**：64 个查询恒有效，那 64 个 mask bit 变成常量，
16,448 应改口径为 **16,384 + 64（常量掩码）** 并说明。
