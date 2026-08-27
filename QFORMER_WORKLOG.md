# Q-Former 状态 tokenizer 工作日志

> 承接 `WMA_RAG_WORKLOG.md`（记到 Phase 3 reader connector）。结论沉淀在
> `WORLDMEMARENA_TOKENIZER_RAG.md`；这里记过程、实现与被推翻的判断。
>
> 2026-08-23 起，末次更新 2026-08-25

---

## 0. 当前主结果

**n=1459，`agent/gui/web` 全 27 样本，官方 `eval_framework.cli`，本地 Qwen3.5-9B 答题+判题。**

| | QA-C↑ | QA-H↓ | QA-O↓ |
|---|---:|---:|---:|
| v6 Raw-Fused 官方对照 | 0.5415 | 0.2132 | 0.2454 |
| v4 固定池化 + latent（**旧 prompt，作废**） | 0.5007 | 0.1332 | 0.3661 |
| v7 Q-Former + latent（**旧 prompt，作废**） | 0.5086 | 0.1311 | 0.3603 |
| v3 固定池化检索 + **原文** | 0.5764 | 0.1885 | 0.2351 |
| **v9 Q-Former + latent（对齐 prompt）** | **0.5949** | 0.1857 | **0.2193** |

配对 McNemar（n=1457）：

```
v9 vs v7   278/151   p=8.9e-10  ***   仅对齐 prompt 就 +8.6 分
v9 vs v6   154/ 76   p=3.0e-07  ***   显著打败官方 RAG
v9 vs v3    78/ 50   p=0.0167   *     latent 显著打败原文
v3 vs v6   126/ 76   p=0.00053  ***
v7 vs v4    58/ 45   p=0.237    ns    ← 整晚检索工作，端到端不显著
```

**16 个 soft token 的 latent 记忆，QA-C 显著高于官方 Raw-Fused RAG，也显著高于把原文交给 reader。**

⚠️ 不与论文 Table 2 并排：只跑 27/461 样本，且**同一个本地模型既答题又判题**。
只作三臂横向比较，最终数字换 GPT-5.4 重判。

---

## 1. 起因

Stage C 未达标（掉 <5 点的判据：union −6.9、qa-distill −12.4）。两个目标函数完全不同的
connector 在保真度子集上统计无法区分 → 约束在**它们共享的部分**：固定的 64 槽池化表示。

**固定池化实测**（956 条 web 观察）：

```
组         槽数  实际有效  两两余弦  能量占比
image       32     32     0.5877    74.3%   ← 塌缩主因
detail      16     16     0.6920    19.2%
context     16      3     0.8741     6.4%   ← 13 槽结构性为零
```

图像走 22×40 patch 网格，`adaptive_avg_pool2d` 压到 4×8，**每槽平均掉 27.5 个 token**；
不同网页顶上都是 chrome、中间内容区、大片白底——按位置平均本身就在抹掉页面身份。
文本侧反而槽位过剩。**预算分配和信息量是反的。**

**因果注意力让三样东西静默失效**：图像槽对序列化器结构性免疫；context 槽恒定 3/16 有效；
**观察 prompt 完全惰性**——序列是 `[image] → DOM → instruction`，图像和 DOM 看不到指令，
指令自己的 token 因 `PROMPT_SLOTS = 0` 被整段丢弃。`fixed_prompt.py:12-15` 那句
「保留可见文本、输入值、控件类型、选中/聚焦/启用状态及空间关系」**对 64 个槽的因果影响精确为零**。

第 3 条决定了损失设计。

---

## 2. 设计

```
H_t (1129×4096, 冻结主干第16层)   ← 全部位置，含 29 个指令 token
   ↓  P_ρ：K 个可学习查询 × [cross-attn (→ self-attn) → FFN] × L
xbar (K×512)  →  connector  →  K 个 soft token  →  冻结 reader
```

**不是「学习式投影」，是三件事同时被修掉**：查询按内容寻址而非按位置平均；
交叉注意力不是因果的，每个查询同时看图像、文本和指令；槽位预算由内容决定。

**约束 1（不可动）**：写侧必须与 query 无关——adapter 在 ingest 时编码一次、缓存 latent、
此后每个问题复用。query 条件化会作废整个存储模型。

~~**约束 2：K_x=64 锁死**~~ — **已放开**。当时的理由（`_check_latents` 硬拒非 `(64,512)`、
`rank_embedding` 是 `(64,4096)`）已参数化。**现行 K_x=16**，压缩比主张因此可得。

**损失只有两项**：$\mathcal L_{\text{answer}}$（金标答案 CE）+ $w\cdot\mathcal L_{\text{distill}}$
（冻结生成器读原文的 KL，按答案位置求平均）。不加 §5.2 式 (9a) 的四个锚点：那句 prompt 已经在
要求四个子空间，交叉注意力一上它自己就开始工作；§5.4 警告的是与 **WM** 共训（WM 有动机让残差
虚假变小），reader 没有；$\mathcal L_{\text{ocr}}$ 正是让 connector「答完接着复述观察」的目标。

**顺带买到**：不要 $\mathcal L_{\text{state}}$ ⇒ 不要 memory_points ⇒
**tokenizer 训练不接触 WMA 的任何标注**。

---

## 3. 实现

| 文件 | 要点 |
|---|---|
| `residualmem/latent/qformer.py` | `StateQFormer`（端口自 JAX `resample()`，self-attention 可选且**默认关**）+ `QFormerStateReader`（P_ρ + connector + 检索头打成**单一产物**）+ `qformer_hash` 溯源 |
| `experiments/state_tokenizer/trunk_states.py` | 主干到第 16 层返回**整条序列**；`modality_labels` 逐位置标 `IMAGE/DOM/INSTRUCTION/WRAPPER`；**`no_grad` 而非 `inference_mode`**（后者永久毒化张量，autograd 不可用） |
| `experiments/state_tokenizer/reader_losses.py` | 两项损失的**唯一实现**，两个训练器共用——两份拷贝正是 KL 被写错的原因 |
| `experiments/state_tokenizer/train_qformer_joint.py` | 联合训练；QA 路径微批 1 + 累积，ℒ_sem 独立成步 |
| `experiments/state_tokenizer/head_recall.py` | 检索闸：**同 session 内** R@1/5/10 + session 内有效秩（按 n−1 天花板归一） |
| `residualmem/latent/qformer_runtime.py` | `QFormerInstructTokenizer`，与 `FrozenV9InstructTokenizer` 同 `encode()` 契约；**拒绝没有检索头的 checkpoint** |

**参数量**（`hidden=1024, heads=8, layers=4`）：StateQFormer 71.9M，联合模块 78.8M 可训练。

**关键实现细节**：

```python
# reader_losses：reshape 就是修复本身（batchmean 在 (1,L,V) 上是对位置求和，实测 17.00× = answer_len）
kl = F.kl_div(log_softmax(student.logits[:, span]).reshape(answer_len, -1),
              log_softmax(teacher.logits[:, span]).reshape(answer_len, -1),
              reduction="batchmean", log_target=True)

# qformer：padding 归零而非仅 mask，「padding 不可达输出」是模块性质而非调用方责任
context = context * key_mask[..., None]
```

**测试**：`tests/state_tokenizer` 131 项 + `tests/residualmem/test_hybrid_reader.py` 9 项。
三处对抗性自检（避免写出空过的测试）：padding 测试不 mask 时差 1.35、mask 后 1.31e-06；
指令段 mask 掉后差异**精确为 0**；KL 测试自带守卫「若求和与求平均在此重合则本测试无判别力」。

---

## 4. 被推翻的判断（按代价排）

**① 「latent 比原文差 7 分」——完全是我的 prompt bug 造出来的。**
`Qwen35LatentReader.answer` 用的是我自己写的 20 词 prompt + 裸 tokenizer，而非官方的
`_ANSWER_SYSTEM_PROMPT`（7 条指令 + 5 步 approach + JSON 格式）。差异直接对应观察到的失败形态：

| 官方 | 我的 | 后果 |
|---|---|---|
| 7. **Only if** the memories truly contain **NO** information relevant… | "If absent, say exactly…" | **QA-O 0.3603 → 对齐后 0.2193** |
| 5. 问图像时返回 `image_id` | **完全没提** | 图像题型被静默阉割 |
| 6. 15 词以内 / JSON / think step by step | 无 | 生成行为不同 |
| chat 模板 | 裸 tokenizer | 同一模型行为差别很大 |
| （API 侧 thinking **关**） | 默认**开**，96 token 全花在 `Thinking Process:` 前言上 | 一个答案都产不出 |

**唯一的真实约束是 latent 不能走 chat-completions API（soft token 是嵌入，API 收文本），
这只解释「为什么本地跑」，不解释「为什么换 prompt」。**
对齐后 QA-C **0.5086 → 0.5949**。v4/v7 的 QA 数字与官方臂不同口径，**已在表里标作废**。

**② 「有效秩是可检索性的先行指标」——错了三次，不再用。**
- 把 41.7 当分数念：那是**跨网站** 96 条观察上测的，benchmark 只问 session 内
- 「方差被网站身份方向主导（23 样本秩 7.2）」：无天花板归一的随手测量，**真值相反**——
  Q-Former 在 session 内保留池化的 66%，跨 session 只保留 24%，丢掉的**恰恰是**网站身份
- 决定性反例：A 臂与 B 臂 session 内有效秩**都是 21.5**，而 R@1 差 50%
  → **秩只能证伪塌缩，不能预测可检索性**

**③ 「联合头是唯一适配这套坐标的头」——反了。** 两条臂上单独头都赢
（A 0.1321 vs 0.1036，B 0.2131 vs 0.1988）。联合头只有 3 个负样本、还在追一个移动的表示；
单独头有 64 个负样本、拟合冻结表示。**联合头是雕刻刀，单独头是量尺**——它的价值是送进 xbar
的梯度，不是它自己排名多准。adapter 默认已改为显式配置的头优先。

**④ 监控函数口径错，差点被当成灾难。** `spread()` 算有效秩去了均值、算余弦没去，
而池化 xbar 因 group×channel 标准化本来就近零均值——**两个表示在未去均值余弦下不可比**。
去均值后都接近正交，**没有塌缩**；但挖出一个真病：`mean_to_deviation = 251.5`，
有信息的部分只占范数 0.4%（ℒ_sem 后降到 1.2–1.4）。
旧 checkpoint metadata 里的 `monitors` 是旧口径，**不可引用**。

**⑤ 有效秩测不出「状态都一样」。** 去均值后它是尺度不变的，20 个近乎相同的状态加各向同性
微噪声就是全秩。那是 `mean_to_deviation` 的职责，测试因此拆成两条。

**⑥ 桩模型太简单。** `head(inputs_embeds)` 逐位置 → KL 恒 0、梯度到不了 latent。
是桩的问题，加因果累积均值后通过。

### 操作上的坑（都是我的）

- **在评测跑着的时候改它正在用的代码文件**，中途的 `SyntaxError` 让 v8 的 10 个样本失败。
  **规则：跑评测期间不碰它的代码。**
- `/tmp` 被自己的冒烟 checkpoint 填满，torch **import 期**写临时模块模板失败，
  `OSError: Errno 28` 的报错位置离原因很远。修法：`TMPDIR` 指向工作区。
- **zsh 不做词分割**，`set -- $cfg` 在 bash 下拆四个词、zsh 下不拆，启动命令带着 `--device cuda:`。
- `nohup` 扛不住 Bash 工具超时（SIGTERM），要 `setsid` + 让启动命令立刻返回。
- `pkill -f` 会匹配到自己的 shell，用 PID。

---

## 5. ℒ_sem 重做：同 session InfoNCE

### 5.1 诊断

`head_recall.py`（840 行 / 31 session，held-out）：Q-Former session 内 R@1 **0.1464** 对
固定池化 **0.3429**。四个候选原因，三个被数据排除：

| 候选 | 判定 |
|---|---|
| 泄漏（31 个验证 session 有 15 个 Q-Former 训练时见过） | **排除**。干净 session 上差距反而更大（0.1382 vs 0.3618） |
| 容量（16 槽 vs 64 槽） | 当时判**排除**（8192 维 ≫ 40 个方向）——**后来收回**：那只证明不受*秩*限制，不证明不受*信息量*限制 |
| 检索头目标错配 | **排除**。池化头面对同一目标、同一缓存、同样负样本，拿到 0.3618 |
| **两个 tokenizer 在优化相反的东西** | **保留** |

**PCA 的目标函数就是可区分性**（最大化保留方差）；**Q-Former 的目标是任务充分性**。
而这两者部分对立：分开同 session 第 3 轮和第 7 轮的是滚动位置、光标、哪一行高亮——
**恰恰是一个好答题者应当归一化掉的偶然状态**。同一购物车页两轮「车里有什么」答案相同，
**答题损失因此奖励把它们编码成同一个东西**。

**机制**：池化的 32 个图像槽固定在 4×8 网格上，「同站点不同状态」直接表现为「第 17 槽变了」——
**差异有固定地址**。Q-Former 的查询是内容寻址的，同站点两轮取到相似内容、输出就相似。
**让 resampler 稳健的置换不变性，正是抹掉 session 内差异的那个性质。**

### 5.2 旧 ℒ_sem 为什么等于没有

```python
soft, xbar, valid = joint(*collate([states]))     # batch 恒为 1
sem_loss = (1.0 - F.cosine_similarity(joint.project(xbar, valid), F.normalize(target))).mean()
```

**B=1 时 logits 是 1×1，单 logit 对标签 0 的交叉熵恒等于 0**——即便当初写了 InfoNCE 也是恒零。
只剩余弦项教「指向你的教师」，而教师彼此就很像：该投影对自己教师余弦 **0.8199**、对别人
**0.6491**，只差 0.17，靠编码共享分量即可满足，R@1 = **0.045**（已发布头 0.84）。
且累积循环里 4 个 xbar 各自 `.backward()` 后图当场释放，**从不共存**。

### 5.3 新目标与单变量约束

$$\mathcal L = \mathcal L_{\text{QA}} + \lambda_{\text{sem}}\cdot\mathcal L_{\text{same-session InfoNCE}}$$

**QA 路径一个字节没动**（`P(o)=1/N` 然后 `q~Q(o)`、微批 1、逐条 backward）。
ℒ_sem 另起一路、**独立采样**：先抽一个 session，再抽它的 B=4 个观察。

三个刻意设计：**批取自单个 session**（131 个训练 session 合格，128 个能填满 B=4）；
**独立于 QA 采样**——否则改善无法在 *InfoNCE 起作用 / session 相关的 QA 梯度起作用 / 两者交互*
之间归因；**自己重跑一次 Q-Former 前向而非 `retain_graph`**——`soft` 要反传穿过冻结的 9B，
跨累积保图等于持有 B 份 9B 激活图，重跑 resampler 只有 79M。实测 **+0.90 s/步（+25% 墙钟）**。

`symmetric_infonce` **直接 import** `train_retrieval_bridge` 的那个函数（非抄写），温度同为 0.05。

### 5.4 两条臂（唯一差别是 `--sem-mode`）

相同：`--queries 16 --qformer-layers 4 --accumulate 4 --distill-weight 0.3 --learning-rate 1e-4
--sem-weight 1.0 --max-steps 9000 --eval-every 250 --validation-observations 96
--patience-evals 8 --seed 35`，不传 `--self-attention`。

| | A 对照 | B 处理 |
|---|---|---|
| `--sem-mode` | `cosine`（QA 微批内，B=1 无负样本） | `same-session`（独立抽 4 条 + InfoNCE） |
| best CE | **1.7290** @6250 | 1.7643 @6000 |
| best 处 rank | 42.0 / 67.4 | **50.3** / 67.4 |

**顺带的可复现性验证**：A 对 K16-long（1.7285 @6750，rank 41.7）**CE 差 0.0005、秩差 0.3**。

### 5.5 检索结果（干净 session，16 个 Q-Former 未见过的）

| | R@1 | R@5 | MRR | session 内秩 |
|---|---|---|---|---|
| A 余弦对照 | 0.1336 | 0.3479 | 0.2500 | 21.5 (60.0%) |
| **B InfoNCE** | **0.2005** | **0.4401** | **0.3189** | 21.5 (59.4%) |
| 固定池化 | 0.3618 | 0.6659 | 0.5023 | — |

**R@1 +50%（相对），噪声量级 ±0.005（A 是 K16-long 的复现）。**
**但 session 内有效秩两条臂完全相同**——InfoNCE 没有增加方向，
**它把已有的方向对齐到了教师上**。见 §4②。

### 5.6 已废弃：K=64

两次都发散——`lr 1e-4` 在 500–800 步 NaN，降到 `3e-5` 在 3250 步后仍 NaN
（`xbar contains non-finite values`）。**连一次干净收敛都拿不到**，且槽数对照被学习率混淆。不再补。

### 5.7 免费负样本（已实现，未训）

`--sem-extra-negatives`（默认 −1 = 全取）。student 必须过 Q-Former 前向，
**teacher 是冻结、预计算、已缓存的**（`TeacherStore` 一次载入整个 session 的数组），
把两者数量绑死在 `sem_batch` 上没有任何理由。实测负样本 **3.0 → 19.9，墙钟成本为零**。
严格是原目标的超集（teacher→student 方向和余弦项逐字不变）。
**暂不训**：R@1 已测到端到端不显著（v7 vs v4 p=0.237），先确认检索能传导。

---

## 6. 模式 1：teacher 换成完整原始观察

### 6.1 为什么：原来的蒸馏项在教「复现你已经拥有的文本」

`L_q` 的 teacher 读 `fused_text`——约 89 token 的字幕，无截图。而那段字幕
**逐字嵌在 student 自己的 AXTree 里**（`<ref=2 tag=textarea value="This is a Windows 10 desktop..."/>`）。
**teacher 的全部输入是 student 输入的子集。** student 唯一多出的是截图像素，而 KL 对它只字未提。

模式 1 补上缺的那一半：teacher 读 `[截图, AXTree, 探针]` 自由生成，student 只有 latent + 同一探针，
在同一段上对齐。**跨度由 teacher 自己的生成定，全程不出现 benchmark 标注。**

探针族（全英文，P1–P3 训练、**P4 留出**）定义在 `observation_kl_precheck.py:46-52`。
训练时每个微批从 P1–P3 抽一个；P4 只在 eval 用。

### 6.2 预计算：为什么必须，以及一个被明确推翻的旧决定

带视觉的 teacher 前向约 1172 token，每微批做一遍训练慢 3–4 倍。但 teacher 冻结、贪心、
输入固定 ⇒ 结果确定 ⇒ 只算一次。

⚠️ `train_reader_qa.py:19-22` 明文拒绝过缓存 teacher logits（"cheaper than the storage and
simpler than a top-k approximation"）。**那对 89 token 无视觉的文本 teacher 成立，对读截图的
不成立。前提变了，不是随意翻案。**

```
vocab_size 248320
全量     每条 48 MB   合计 0.2 TB   ❌
top-128  每条 98 KB   合计 1.6 GB   ✅（3182 观察 × 4 探针 + 7776 pair）
```

**top-k 在这个 KL 方向上有原则**：每项由教师概率加权，截断丢掉的是权重最小的部分，
残余质量即误差上界。**实测 70 万个位置：中位保留 0.999987，最小 0.73。**

### 6.3 阶段 C 的闸：通过

两个 checkpoint **同为 step 1250**、同一批 48 条观察、同探针 P4——唯一变量是 `w_o`：

| | 匹配 KL | 错配 KL | **gap** | 匹配更低 |
|---|---|---|---|---|
| `w_o=0`（对照） | 0.7919 | 0.7944 | **+0.0024** | **56.2%**（≈抛硬币） |
| `w_o=0.5` | **0.6746** | 0.7491 | **+0.0745** | 77.1% |
| 固定池化 64 槽 | 0.7248 | 0.8859 | +0.1610 | 89.6% |

**gap 是对照的 31 倍。** 对照的 56.2% 说明：**没有 obs 目标时，latent 里几乎没有屏幕特异信息。**

另一个数：`w_o=0.5` 的**匹配 KL 0.6746 低于固定池化的 0.7248**——32 槽复现教师行为**比 64 槽更准**，
差的是**区分度**。

⚠️ 训练日志里的 gap 只用 24 条观察，噪声大（step 1250 报 +0.1272，48 条重测是 +0.0745）。
**只当趋势看，出表一律用 48 条口径。**

### 6.4 checkpoint 选择：一个我指出过却没修的缺陷

早停与存盘都按 val CE，而这个实验的目标指标是 gap。`w_o=0.5` 上 CE 最低在 1250、
gap 最高在 1500，**保下的是前者**。已改成**两份都存**
（`<name>.pt` 选 CE、`<name>.gapbest.pt` 选 gap，metadata 带 `selected_by`），早停仍看 CE
以保持与既往 run 可比。

### 6.5 非有限梯度：八个假设，八个被证伪

**共同形态**：死前每次 eval 损失都正常，然后突然 `xbar contains non-finite values`；
`xbar` 出 Q-Former 时已非有限，说明**权重在上一步就被坏梯度污染**，traceback 一直指向症状。

**查明的机制**（这个是确定的）：

```
某步梯度出 inf
  → clip_grad_norm_ 的缩放系数 = max_norm / (inf + 1e-6) = 0
  → inf × 0 = NaN                    ← 裁剪本身制造了 NaN
  → optimizer.step() 把 NaN 写进权重
  → 下一次前向 xbar 非有限
```

**被证伪的八个**：

| | 假设 | 证伪它的测量 |
|---|---|---|
| ① | 槽数 K=64 不稳 | 两种 sem 各跑 1200 步都不炸 |
| ② | ℒ_sem 的形式 | 同上 |
| ③ | 无 warmup / 学习率 | `warmup=500` 反而在 1053 步炸，`warmup=0` 同 seed 跑过 800 步 |
| ④ | 前向 KL 无界 | 加 `LOGPROB_FLOOR` 后仍炸（但那时护栏放在归一化之后，近乎无效） |
| ⑤ | obs 项梯度被 4× 放大 | **真 bug，已修**；`w=0.5` 因此干净，但 0.8/1.0 仍触发 |
| ⑥ | 检索头 `F.normalize` 分母塌陷 | 触发时 `headmin = 12.01`，健康 |
| ⑦ | 9B 反传的 bf16 数值 | **`L_sem` 根本不过 9B**（`trunk_states` 是 `no_grad`） |
| ⑧ | 主干精度 | fp32 跳步 148 次、bf16 221 次，**fp32 触发更早**（723 vs 850） |

**⑤ 是唯一的真 bug**：`obs_step` 在累积循环内被调 `accumulate` 次，每次做全权重 backward，
所以 `--obs-weight 1.0` 实际是 4.0。它解释了触发时机与权重的严格单调
（1.0→532 步、0.8→756、0.5→不触发）和 `w=2.0` 的 CE 崩到 2.71。

**⑥ 那次否定差点是错的**：我的 hook 只记 `norm(dim=-1).min()`，
而**一行 inf 会让 min 报告健康的那些**。补记 max 与 finite 后仍是健康的，结论才站得住。

**三个损失项都能是首次触发点**（干净计数：`L_obs` 5 次、`L_sem` 15 次）。
唯一共享的可训练部分是 Q-Former 自身。**随机、聚集、与权重弱相关**——
同 seed 同配置一次在 447 步炸、下一次跑过 450 步无事。

**已停止追根因。** 两道护栏让它不再致命：

```python
grad_norm = clip_grad_norm_(...)
if not isfinite(grad_norm):
    skipped += 1
    lr = max(lr * 0.5, lr0 * min_lr_fraction)   # 逃逸：不降的话权重不动，永远出不来
    continue                                     # 跳过更新，NaN 链被堵
```

⚠️ **代价是实验拿不到收敛值。** `w_o=1.0` 跳步 286 次后学习率压到 1e-06 下界：

```
step 1750   CE 1.9527   gap +0.0833   rank 22.9   ← 此后 lr 到底
step 2000   CE 1.9627   gap +0.0816   rank 24.0
step 4000   CE 1.9573   gap +0.0859   rank 24.0   ← rank 四次 eval 一位小数不动
```

前 1750 步 gap 涨 0.053，后 2250 步只涨 0.004，**速度差 60 倍**。
**这批数字是「1750 步的模型」，不是收敛的模型。**
`GradScaler` 的完整逻辑还有「连续 N 步不跳就把 lr 涨回去」，我只实现了一半。

### 6.6 权重扫描（48 条口径，与固定池化同口径）

| 臂 | 匹配 KL | 错配 KL | gap | 匹配更低 |
|---|---|---|---|---|
| 对照 `w_o=0` | 0.7919 | 0.7944 | +0.0024 | 56.2%（≈抛硬币） |
| `w_o=0.8`（跳步 401，**没训成**） | 0.6532 | 0.6669 | +0.0137 | 81.2% |
| `w_o=1.0` CE-best @1750 | 0.6473 | 0.6959 | +0.0487 | 87.5% |
| `w_o=0.5` | 0.6353 | 0.6899 | +0.0546 | 89.6% |
| **`w_o=1.0` gap-best @4000** | 0.6501 | 0.7157 | **+0.0655** | **91.7%** |
| 固定池化 K=64 | 0.7248 | 0.8859 | +0.1610 | 89.6% |

**匹配 KL 全部约 0.64–0.65，低于固定池化的 0.7248**——32 槽复现教师行为**比 64 槽更准**，
差的是**区分度**（gap 只有它的 41%）。

⚠️ **`w_o=1.0` 的 gap-best 可疑**：它选自学习率已到底的冻结区间，
2000–4000 步 gap 在 0.0816–0.0859 之间随机游走，**取最大值就是取噪声**。
训练日志（24 条）说两者差 0.004，48 条重测说差 35%——**至少一个是噪声**。

**存两份 checkpoint 的收益比预期小**：六个里只有这一对有实质差别。

---

## 7. 当前状态（2026-08-25 15:00）

**在跑**：v8（Q-Former 检索 + 原文 + 官方答题）补跑我搞坏的 10 个样本。
它出来后拆开最后一层：`v8 vs v3` = 检索管线值多少；`v9 vs v8` = latent 相对原文值多少。

**产物**：

| | 数字 |
|---|---|
| `qformer-K16-semnce.pt` | best CE 1.7643 @6000，rank 50.3 |
| `qformer-K16-semcos.pt` | best CE 1.7290 @6250，rank 42.0（对照） |
| `retrieval-head-qformer-semnce.pt` | best val loss 3.3802 @1250 |
| `qformer-semnce-bridge-cache.npz` | 4379 行 / 156 样本，840 validation |
| **`exp_results/v9-qformer-latent-aligned/`** | **QA-C 0.5949** |

**防漂移**：官方 prompt 文本放在 `residualmem/latent/instruct_bridge.py`
（保持 `residual-mem` 不依赖 WorldMemArena），adapter 构造时调 `_assert_official_prompts_match()`
**与 `eval_framework.cli` 逐字比对，不一致直接报错**。

**出表时必须写明**：

1. 只跑 `agent/gui/web` 27 样本（论文 461），**同一本地模型既答题又判题**，最终换 GPT-5.4 重判
2. 记忆侧 6 列（Recall/Corr/Hallu/Irrel/Update/IntRej）**三臂差 <0.01**——
   它们测的是 session 阶段的记忆抽取，不经过被改动的组件，**报成 12 列是充数**
3. 我们的观察行**确实带 `image_path`**（`residualmem_instruct_adapter.py`），
   带图行 v9 vs v3/v6 的差是**抢不到 top-10**，不是漏设
4. v9 是**三组件 + 两配方**的打包改动，非单变量
5. Q-Former 只在 WMA 非 web 的 QA 上训过，**它在 BrowserGym 上的状态是分布外的**

---

## 8. 待办

| | |
|---|---|
| **v8 收尾** | 补跑 10 个 → 拆开「检索管线 vs latent」 |
| **v9 复跑固定池化臂** | 现有 v3/v4 是旧 prompt，要在对齐 prompt 下重跑才能和 v9 单变量对比 |
| λ_sem 扫描 | 旋钮已验证有效，3 个点能说清「再推能追平」还是「到顶了」 |
| 免费负样本重训 | 代码就绪；等确认检索能传导到 QA |
| MolmoWeb 接入 | HumanTrajs（36K 轨迹 × 20.8 步）是唯一对症的同 session 负样本源；**但无 AXTree、无 QA、183 GB，且教师向量的分布问题未解** |
| GPT-5.4 重判 | 现在答题与判题同模型 |

**Phase 2 下游重训**：冻结 P_ρ → 写标准化产物 → 重训 A1 → 重训 A2 → 重建 L16 restorer。
协议串需 bump 到 `_v2`（现在**不编码维度**，改了只会得到 size-mismatch traceback）。

⚠️ `Layer16Restorer` 用的是 `components.transpose()`，**只在正交时等于伪逆**；
学习式投影一般不正交，需额外存最小二乘伪逆并给 restorer 加可选 buffer。
