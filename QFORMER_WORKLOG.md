# Q-Former 状态 tokenizer 工作日志

> 承接 `WMA_RAG_WORKLOG.md`。那份记录到 Phase 3（reader connector）为止；
> 这份从 Stage C 的阴性结果开始。结论沉淀在 `WORLDMEMARENA_TOKENIZER_RAG.md`，
> 这里记过程，**包括我自己造成的错误**。
>
> 日期：2026-08-23

---

## 1. Stage C：阴性，触发了预先写好的退出条件

Phase 3 的判据是「latent 相对 text 的 Correct 下降 < 5 个点」。全量 233 题：

| 条件 | Correct | Halluc | Omission |
|---|---:|---:|---:|
| text（上界） | 0.2961 | 0.1030 | 0.5966 |
| union（重建 CE） | 0.2275 | 0.1588 | 0.6137 |
| qa-distill（answer CE + KL） | 0.1717 | 0.0815 | 0.7468 |

总表会误导——`text` 自己只有 29.6%，说明**大多数被检索到的观察行本来就不含答案**。
按 text 是否答对拆开，并做配对 McNemar（同样 233 行，两个 connector 逐行对齐）：

```
text 能答对子集 n= 69:  只union对= 4   只qa-distill对= 8   p=0.3877
text 答不对子集 n=164:  只union对=21   只qa-distill对= 4   p=0.0009
```

两条读数：

1. **新版那个「+5.8 点」不成立**（p=0.39）。真正考保真度的子集只有 n=69，
   4 vs 8 的差异远达不到显著。
2. **显著的是另一头**：旧 connector 在**记忆答不上来的行**上产出金标答案的次数是新版的 5 倍。
   结合它整体幻觉更高（0.1588 vs 0.0815），最可能的解释是它在编、偶尔编中——
   与「重建 CE 让它复述观察」的诊断一致。但无法完全排除「它捞到了 text 条件漏掉的东西」。

**判据结论**：union −6.9 点，qa-distill −12.4 点，**都不达标**。
计划写的退路是「先试 KL 蒸馏；仍不行则 latent 注入路线搁置」。

---

## 2. 但那次对比是脏的——我的 bug

`train_reader_qa.py` 的蒸馏项：

```python
kl = F.kl_div(student_logits, teacher_logits, reduction="batchmean", log_target=True)
```

`batchmean` 除以 `input.shape[0]`。张量是 `(1, answer_len, vocab)`，**第一维是 1**，
所以这是对 answer_len 个位置**求和**而不是求平均。数值验证：

```
旧(1,L,V) batchmean = 17.1523   ← 对 17 个位置求和
新(L,V)   batchmean =  1.0090   ← 每位置平均
旧/新 = 17.00x  (= answer_len)
```

**`--distill-weight 1.0` 实际是 `answer_len ×`（10–30 倍）的权重。**
那一版 connector 是在蒸馏项压过 CE 一个数量级的情况下训出来的，
所以「ℒ_Distill 无效」这个阴性结论**不成立**。

修复后重训，两项都在下降（旧版 KL 在 0.88–34.68 之间无趋势乱跳）：

```
新版  step  500  val CE 2.1026  KL 0.6845
      step 1000  val CE 2.0606  KL 0.3737
      step 1500  val CE 1.9906  KL 0.3617
```

⚠️ **但新旧的 val CE 不可直接比**：我在修 bug 的同时把验证集从「索引序前 128 行」
改成了随机抽 128 行，验证集本身变了。有效对比只有 Stage C（同题、同裁判）。

### 同批修掉的另外三处

| 位置 | 问题 |
|---|---|
| `train_reader_qa.py:154` | `torch.manual_seed` 在 connector 构造**之后**，初始化不可复现 |
| `train_reader_bridge.py:133` | 同一个 bug |
| `train_reader_qa.py:193` | 验证集取索引序前 128 行 = 最早排序的那几个 held-out 样本，不是该 split 的代表 |
| `connector_fidelity.py:41` | 本地重复声明 `READER_BRIDGE_PROTOCOL` 字面量，协议串一旦 bump 会静默分叉 |

**这三个加上 KL 那个，共同的根因是：仓库里没有任何测试碰过任何训练脚本。**
`WMA_RAG_WORKLOG.md` 记录的两个 reader bug 也是同样原因活下来的。

---

## 3. 诊断：瓶颈在共享的那一部分

两个目标函数**完全不同**的 connector，在保真度子集上统计无法区分（37.7% vs 43.5%，p=0.39）。
两条不同的路走到同一个数，约束通常在它们共享的部分——**固定的 64 槽池化表示**。

### 实测（956 条 web 观察）

```
组          槽数   实际有效    两两余弦    能量占比
全部 64      64       51       0.6261     100.0%
image        32       32       0.5877      74.3%   ← 塌缩主因
detail       16       16       0.6920      19.2%
context      16        3       0.8741       6.4%   ← 13 槽结构性为零
```

单条观察 995 token（880 图像 + 114 文本）。图像走 **22×40** 的 patch 网格，
`F.adaptive_avg_pool2d` 压到 **4×8**（`key_pooling.py:270-274`），**每槽平均掉 27.5 个 token**。

两个不同网页顶上都是浏览器 chrome、中间内容区、大片白底——
**按空间位置平均这个算子本身就在抹掉页面身份**。
而文本侧槽位过剩（114 token 进 32 槽）且 13 槽恒零：**预算分配和信息量是反的**。

### 因果注意力让三样东西静默失效

1. **图像槽对序列化器结构性免疫**——图像 token 看不到排在它后面的文本。
2. **context 槽恒定 3/16 有效**——合成 AXTree 每条观察只产出 3 个节点。
3. **观察 prompt 完全惰性。** 序列是 `[image] → DOM → instruction`
   （`extract_qwen.py:52-56, 79-85`）。图像和 DOM 都看不到指令；
   指令自己的 token 因 `PROMPT_SLOTS = 0` 被整段丢弃（`static_key_pooling.py:479`）。

   那句 `fixed_prompt.py:12-15`：

   > 请忠实表示当前页面状态，保留可见文本、输入值、控件类型、选中/聚焦/启用状态及空间关系。

   **对 64 个槽的因果影响精确为零。** 它写进了输入、花了 29 个 token、什么都没条件化。

第 3 条决定了损失设计（见 §5）。

---

## 4. 设计

```
H_t (1129×4096, 冻结主干第16层)   ← 全部位置，含 29 个指令 token
   ↓
P_ρ  64 个可学习查询 → [cross-attn to H_t + latents 间 self-attn + FFN] × L
   ↓
Linear(→512) + per-slot LayerNorm
   ↓
xbar (64×512) → connector → 64 soft token → 冻结 reader
```

**不是「学习式投影」，是三件事同时被修掉**：查询按内容寻址而非按空间位置平均；
交叉注意力不是因果的，每个查询同时看图像、文本和指令；槽位预算由内容决定。

### 两条不可动的约束

1. **写侧必须与 query 无关。** WMA adapter 在 ingest 时编码一次、算一次检索向量、
   把 latent 缓存进 `_latent_by_memory_id`，此后每个问题复用。
   query 条件化会作废整个存储模型。**问题只进 reader 的 prompt。**
2. **$K_x = 64$ 锁死。** `_check_latents` 硬拒非 `(64,512)`；A2 的位置表是 64 槽；
   connector 的 `rank_embedding` 是 `(64,4096)`。
   **因此本轮拿不到压缩比主张**，收益必须全部来自质量。

---

## 5. 损失只有两项，锚点被砍掉了

$\mathcal L_{\text{answer}}$（金标答案 CE，问题在 reader prompt 里）
$+\ \mathcal L_{\text{distill}}$（冻结生成器读原文的轨迹做 KL，按答案位置求平均）。

**不加 §5.2 式 (9a) 的四个锚点**（sem / ocr / vis / state）：

1. **那句 prompt 已经在要求四个子空间**，它不起作用只是因为因果注意力挡住了。
   **交叉注意力一上，prompt 自己就开始工作**——锚点想买的东西，prompt 免费给。
2. §5.4 警告的是 tokenizer 与 **WM** 共训（WM 有动机让残差虚假变小）。
   reader 没有这个动机，两项损失都在**奖励**保留信息。
3. $\mathcal L_{\text{ocr}}$ 就是让 connector「答完接着复述观察」的那个目标。

**顺带买到的**：不需要 $\mathcal L_{\text{state}}$ ⇒ 不需要 memory_points ⇒
**tokenizer 训练不接触 WMA 的任何标注**。原本「效果 vs 主张干净度」的取舍直接消失。

**已知代价**：$\mathcal L_{\text{distill}}$ 的教师读的是 `fused_text`（约 89 token 字幕），
**不含截图空间细节**，图像内容只被 QA 恰好问到的部分约束。
对策是把 §5.4 的指标做成**监控而非损失**——每次 eval 报**同一批观察上**
学习式 xbar 与固定池化 xbar 的两两余弦、有效秩（配对对比）。
任一劣于固定池化即判失败，届时定点加回**对应的那一个**锚点。

---

## 6. 实现

| 文件 | 内容 |
|---|---|
| `residualmem/latent/qformer.py` | `StateQFormer`（端口自 JAX `tokenizer.resample`，**补了 latents 自注意力**）、`QFormerStateReader`（P_ρ+connector 单一产物）、`qformer_hash`（替代 `pca_sha256` 溯源） |
| `experiments/state_tokenizer/trunk_states.py` | 主干跑到第 16 层返回**整条序列**；模态标签覆盖全部位置（含 wrapper） |
| `experiments/state_tokenizer/reader_losses.py` | 两项损失的**唯一实现**，两个 trainer 共用 |
| `experiments/state_tokenizer/build_qformer_qa_pairs.py` | pairs 存**引用**而非 blob：1.22 GB → 33.4 MB |
| `experiments/state_tokenizer/train_qformer_joint.py` | 联合训练 + 坍塌监控 |

### 几处值得记的决定

**`no_grad` 而非 `inference_mode`。** `frozen_v8_runtime.py:164` 用的是 `inference_mode`，
它产出的是 inference tensor，**对 autograd 永久中毒**——不是 detach，是任何记录梯度的算子都会抛
`RuntimeError: Inference tensors cannot be saved for backward`，且传染给所有下游张量。
这是插入可训练模块的硬阻塞。已写成测试钉住（`test_inference_mode_poisons_the_graph_but_no_grad_does_not`）。

**微批 1 + 梯度累积，而非真批量。**（偏离批准方案，已向用户说明。）
student 的记忆是 64 个 soft token，teacher 的是变长文本，真批量要在两种不同 padding 上
对齐答案跨度——正是 KL 那个 bug 所属的类别。累积复用已验证的单样本前向，代价约 7%。

**持久化 `synthetic_axtree`。** 抽取原本只存字符数，而重跑主干需要树本身。
文件里已有同样理由的先例（`fused_text` 存下来而不是事后重建，避免两份观察集漂移）。
重抽后**验证 xbar 逐位相同**，确认改动没有动到别的东西。

### 测试（20 项，全新增）

- `test_qformer.py`（10）：形状契约；**指令段改变输出**（当前管线证明不具备的那条性质）；
  padding 不可达输出；梯度到查询；确定性；模态嵌入零初始化；hash 随权重移动。
- `test_trunk_states.py`（5）：`inference_mode` 中毒 vs `no_grad`；模态标签无空洞；collate 掩码。
- `test_reader_losses.py`（5）：**KL 是每位置平均而非求和**（今天这个 bug 的回归）；
  权重只缩放 KL；空记忆禁用 KL；梯度到 latent。

**两处对抗性自检**（避免写出空过的测试）：
- padding 测试：不 mask 时差 1.35，mask 后差 1.31e-06——判别力 6 个数量级。
- 指令段测试：把该段 mask 掉后差异**精确为 0**，证明差异确实来自那一段。
- KL 测试自带守卫：「若求和与求平均在此重合则本测试无判别力」。

**桩模型的一次修正**：最初 `head(inputs_embeds)` 是逐位置函数，没有上下文混合，
记忆影响不到答案位置的 logits，导致 KL 恒 0、梯度到不了 latent——两条测试因此失败。
是桩太简单，不是被测代码的问题；加上因果累积均值后通过。

---

## 7. 当前状态

**端到端已打通**（真 9B，GPU 3）：
```
H_t (1129, 4096) bf16   模态 image=880 dom=167 instruction=29 wrapper=53
91.1M 可训练  →  xbar (1,64,512)  →  soft (1,64,4096)
loss=9.6824  CE=7.6435  KL=2.0389
查询梯度 |g|=1.18e+03  有限
```

**在跑**（8 卡）：

| GPU | 任务 | 可训练参数 |
|---|---|---|
| 0 | connector 重训（Phase 0 干净基线） | 19.1M |
| 1 | Q-Former A：layers=4, distill=1.0 | 91.1M |
| 2 | Q-Former B：layers=6, distill=1.0 | 124.7M |
| 3 | Q-Former C：layers=4, distill=0.3 | 91.1M |
| 4–7 | LLM 服务（8 副本，供 Stage C 与官方评测） | — |

单次约 5.7 小时（3000 步 × 累积 4 = 12,000 样本 @ ~1.55 s，加 30 次 eval 约 29 分钟），
patience 8 次 eval（= 3200 样本）可能提前停。

**初始化时监控会报 `COLLAPSE-REGRESSED`**（余弦 0.9992 vs 固定池化 0.5917）——
随机查询在相似上下文上产出几乎相同的状态，这在 init 时是预期的。
有意义的是收敛后是否仍然回归。

### 启动时踩的两个坑（都是我的）

1. **`/tmp` 被我自己填满。** 两个冒烟 checkpoint 各 364 MB / 140 MB 写进了 `/tmp`，
   而 `/tmp` 只有 30 G 且已用 28 G。torch 在 import 期要往临时目录写一个生成的模块模板
   （`torch/distributed/nn/jit/instantiator.py`），于是 A、B 两个配置在**导入阶段**就死于
   `OSError: [Errno 28] No space left on device`——报错位置离原因很远。
   修法：删掉我自己产生的冒烟文件，并把 `TMPDIR` 指到工作区（那边 6.1 TB）。
   `/tmp/pytest-of-luzheng`（1.6 G）经用户确认后删除。
2. **zsh 不做词分割。** 重启脚本里用 `for cfg in "A 1 4 1.0"; do set -- $cfg` 传参，
   bash 会把 `$cfg` 拆成四个词，**zsh 默认不拆**，于是 `$2 $3 $4` 全空，
   启动的命令带着 `--device cuda:` 和空的层数/权重。改为显式写死每个配置。

---

## 8. 待办 / 门槛

| 闸 | 判据 |
|---|---|
| Gate 1（Stage C） | text 能答对子集上，latent 的 Correct 显著高于 Phase 0 基线（配对 McNemar p<0.05）。**需先扩大 n**——现在只有 69 |
| Gate 2（官方评测） | `agent/arena/web` 全量，R@10 与 `correct_ratio` 对比 v3 |
| 坍塌监控 | 两两余弦 / 有效秩任一劣于固定池化即判失败 |

**下游重训**（Phase 2）：冻结 P_ρ → 写标准化产物 → 重训 A1 → 重训 A2 → 重建 L16 restorer。
协议串需 bump 到 `_v2`（现在协议串**不编码维度**，改了维度只会得到 size-mismatch traceback）。

⚠️ **对批准方案的一处修正**：方案里写「L16 那族 baseline 约 10 行改动就能保住」。
更准确的说法是：`Layer16Restorer` 用的是 `components.transpose()`，
**这只在 components 正交时等于伪逆**。学习式投影一般不正交，
所以需要额外存一份最小二乘伪逆并给 restorer 加可选 buffer——仍然不大，但不是「转置一下」。

**比特预算口径**：64 个查询恒有效，那 64 个 mask bit 变成常量。
16,448 应改口径为 **16,384 + 64（常量掩码）** 并说明。
