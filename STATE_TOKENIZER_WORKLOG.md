# ResidualMem State Tokenizer 工作日志

当前冻结状态：**v8 / AXTree / `(32,16,16,0)` / 10 万状态**。

> 2026-08-20 恢复说明：v8 本文与结果定义保持不变；跨 WorldMemArena
> 的 v9 Qwen3.5-Instruct wrapper、fused-observation retrieval/reader bridge
> 已恢复在 `QWEN35_INSTRUCT_WORLDMEMARENA.md`。模型权重与训练输出不属于
> 本次两日前源码备份，需按该文档重跑。
本文件记录「做了什么、为什么、哪些量具骗过我」，不是 API 文档。

---

## 0. 数值与评测协议（对 A1、A2、temporal WM 强制生效）

A1 的重建误差对 matmul 精度的敏感度比 A0 高一个数量级（A0 改善 2%，A1 改善 20%），且同一 checkpoint 在 TF32 下换批次切分有约 1% 波动。这个量级已与后续要测的容量差异可比，因此所有静态验证实验采用下列协议。

| 项 | 规定 | 实现方式 |
|---|---|---|
| matmul precision | `highest`（真 fp32，禁用 TF32） | A1 runner 默认，写入结果 `numerics` |
| model activations | FP32 | 模型内部统一 `jnp.float32` |
| metric SSE accumulation | FP64 host 累加 | `_host_group_sse`，batch 内与跨 batch 均 FP64 |
| evaluation batch size | 固定并随结果记录 | `--eval-batch-size` |
| 评测设备 | 固定单一 device 并记录 | `--platform/--device-index` |
| 评测切分 | 确定性行选择与批次切分 | task round-robin + 顺序切片 |

FP64 累加使指标与批次切分**完全无关**：`eval-batch-size` 取 64 与取 2 的 MSE 在相对误差 `1e-12` 内一致（由 `test_metric_accumulation_is_batch_split_invariant` 锁定）。

任何报告的误差必须同时给出精度、batch size 与 device，缺一不可视为可比。

**环境分裂是硬约束**：抽取管线只有 torch（`MemCompiler`），bottleneck 只有 jax（`ResidualMem`），采集只有 playwright（`browsergym-venv`）。三者互不可导入。跨环境共享的常量必须放在无第三方依赖的模块里（见 §6.1）。

---

## 1. 最终方法

Qwen 的输入是**三个模态**，不是只有文本：

$$
\underbrace{\text{screenshot } 498\times321}_{\text{vision tokens}}
\;+\;
\underbrace{\text{AXTree} \rightarrow \text{compact 线格式}}_{\texttt{<dom>}\ldots\texttt{</dom>}}
\;+\;
\underbrace{\text{固定观察提示}}_{\texttt{<instruction>}\ldots}
$$

$$
\rightarrow \text{Qwen3.5 layer-16 Full-H}
\rightarrow \text{Static Key64}
\rightarrow \text{PCA512}
\rightarrow \text{group×channel normalization}
$$

`modality_indices` 按标记切出 (image, dom, instruction) 三段隐状态，Static Key64 分别池化：**截图占 32 槽（64 槽的一半）**，AXTree 占 detail 16 + context 16，固定提示占 0（已回收，见 §3）。

送入 World Model 的标准化状态为 $\bar x_t\in\mathbb R^{64\times512}$。

符号固定：$x_t$ 原始 Static PCA 状态；$\bar x_t$ 固定标准化状态；$y_t\in\{0..255\}^{64\times32}$ A2 的离散码；$z_t$ **保留给** World Model 的 grouped categorical stochastic latent，不得混用。

---

## 2. 为什么从 compact DOM 换成 AXTree

v7 的语义门控暴露出一个结构性上界，不是调参能解决的。绑定探针总体 0.7935，分解后字面进入 raw 槽时 0.8345、未进入时 0.6873；剩余部分需要 `ref`/`parent` 这类结构信息，而 Static Key64 **按设计**把它们排除在 raw detail 之外。纯 DOM 路线上「指令优先 + 加槽」最多把绑定抬到约 0.834。

AXTree 把绑定从「跨节点引用」变成「字面相邻」：`checkbox 'Nb'` 是同一行，角色与名字在同一个 span 里。实测（12 任务）：

| | compact DOM | AXTree |
|---|---:|---:|
| 目标→元素解析规则 | 三级回退（`t` 节点 → 共享 parent 的可点击兄弟 → 可点击 parent） | 一条：目标 == 可点击节点的 accessible name |
| 解析率 | 97.1% | **100%**（116/116、60/60，全部有 bid 且可点击） |
| token 数（12 任务均值） | 481 | **118**（4.1×） |

核对过：**没有任何 probe 标签依赖几何**（static 全是 role 存在性，dynamic 全是交互态），而几何+样式占 compact DOM token 的 66.3%，故全部丢弃。

### 2.1 复用线格式，池化链不改

`parse_dom_spans` 是通用的逐行 `<key=value/>` 解析器。把 AXTree 序列化成同一格式、并把 role 映射到**现有 tag 词表**，则 `key_pooling`、`v2_data`、`static_key_pooling` 全部无需改动：

```
<ref=21 parent=20 tag=input_checkbox text="Nb" value="true" flags=0,1,0,1/>
```

`ref` ← browsergym bid；`parent` ← 最近的有 bid 的祖先；`text` ← accessible name；`value` ← textbox 的值或 checkbox 的 checked；`flags` ← `[focused, tampered, 0, is_leaf]`（`v2_data` 硬性要求 4 位）。

两处 AXTree 特有的处理：

- **StaticText 去重**：AXTree 在控件上挂 name，同时又有一个重复该 name 的 StaticText 子节点。二者都保留会让候选翻倍、预算减半。只有当文本未被同级/父级控件的 name 覆盖时才保留（`scroll-text-2`/`copy-paste` 的正文正是靠这条留下的）。
- **`tampered` 由 Python 侧记录**：BrowserGym 的 `remove_human_display` 主动 `removeEventListener` 掉了设置该位的监听器，MiniWoB 的 `wob_ref` 也因 `getDOMInfo()` 从不被调用而永不赋值。我们自己的动作是页面唯一的交互来源，故在采集侧记录动作过的 bid 即可，精确且无失败模式。

---

## 3. Slot 布局

| 分组 | 槽数 | 处理方式 |
|---|---:|---|
| Screenshot | 32 | 按真实 Qwen merged vision grid 二维平均池化；BrowserGym 498×321 → grid `(1,20,32)` → merged `10×16` |
| DOM-detail | **16** | 保留精确 value、控件 text、option/label text 等 UI literal |
| DOM-context | 16 | 对完整节点求均值，包含已进入 detail 的 token |
| Fixed prompt | **0** | 已回收给 detail |
| 合计 | 64 | 输出 `64×4096` |

**prompt 槽为何归零**：它装的是每个状态**逐字节相同**的固定观察提示。跨状态余弦相似度中位 0.214，而 detail 是 0.002，相差两个数量级——这就是「几乎不携带每状态信息」的样子。

**detail 槽为何 12→16**：预算是绑定能否成立的瓶颈。`click-checkboxes` 一条指令点名至多 5 个目标、每个 1–5 token，12 槽即使配合指令优先仍漏掉约 19%。

> 我曾以「AXTree 把 token 压掉 4.1× 后预算不再是瓶颈」为由推迟这两项改动，**并且用同一个错误假设推迟了两次**。预算按**被选中字面的 token 数**计，与文本总长无关：标签仍是那些 5-token 随机串，数量也没变。实测两项各自贡献 +16.2pp 和 +9.2pp。

**图像网格必须显式传**：`rebuild_static_key64 --image-grid-thw 1 20 32`。默认值 `1 20 14` 对应 v7 的 160×210 截图，用错会让 32 个 image 槽静默错位池化（`rebuild` 会校验 `merged_image_hw`，是唯一拦得住它的地方）。

---

## 4. Static selector

固定 DOM 语义，不使用 IDF 或数字特殊打分：

- 语义组 `current_state/actions/choices/auxiliary`；
- 加权轮转 `current_state, actions, current_state, choices, current_state, auxiliary`；
- **组内**按「文本是否出现在任务指令中」分层（§4.1）；
- 同一字符 span 重复出现只保留最高优先级候选，最终再做 token-index 去重；
- `flags/ref/parent/id/classes` 不占用 raw detail；
- Qwen BPE 长度 `≤8` 的短 span 必须完整保留，预算不足时跳过而非截断；
- 短 span 处理完后若仍有预算，长 span 顺序池化到剩余槽；
- invalid slots 补零并保存显式 valid mask。

### 4.1 指令优先

排序只在**组内**进行。跨组轮转是「无论任务问什么，状态都描述了 actions 和 choices」的保证，跨组重排等于拿通用性换任务贴合度。组内重排则是把该组自己的份额花在指令真正点名的控件上。

解码一个真实状态可以直接看到问题：12 个槽里 **3 个花在干扰项 `XKYa` 上**，而 3 个指令目标被挤掉。

计费规则：task instruction 视为 environment-provided side information，对所有模型免费且相同，但 Total Episodic Rate 须单列其成本。

**`--instruction-records` 必须传采集清单**：特征清单里的 `instruction` 是任务无关的固定观察提示（这是 leakage 控制的设计），传错会让候选按错误文本排序而**静默不生效**。

### 4.2 AXTree 曾静默丢掉 52% 的目标

`static_candidates` 的分组规则要求 `text` 挂在 `CHOICE_TAGS`/`ACTION_TAGS` 节点上，或是父节点属于这两类的 `t` 节点。compact DOM 里标签在 `label` 下的 `t` 节点上 → 归入 `choices`；**AXTree 把 name 挂到 `input_checkbox` 自身，没有任何规则接得住 → `group is None`，候选在进入抢槽之前就被整个丢弃**。

`choices` 组候选数从 v7 的 640 变成 **0**。已勾选的 checkbox 因被更早的 `associated_with_checked` 分支救下，故障因此表现为「部分丢失」而非全丢，更难察觉。

修复是一行：`text` 挂在 `CHECKABLE_TAGS` 上时同样归入 `choices`。对 v7 逐项无影响（v7 的 `input_checkbox` 本就没有 `text` 属性）。

---

## 5. 数据（v8）

BrowserGym MiniWoB，12 个任务，10 万状态。

- **任务内容与 Farama 版一致**：BrowserGym 钉在 `7fd85d7`，我们本地在 `eb59fed`，12 个任务 HTML 与 `html/core/` **逐字节无差异**。「非官方移植」指 gym 封装层，不是任务内容。
- 采集：`--max-steps 7 --random-action-prob 0.5`，脚本级联 + ε 混合策略，与 v7 逐条对齐，使表示差异可归因于观察而非状态分布。实测 scripted 50,181 / random 49,815。
- 划分：7:2:1 任务分层（70018/20011/9979），每任务 train 比例 0.694–0.705，**零 episode 跨划分、零旧 train 泄漏到 held-out**。
- 每状态另存 `dom_control`（`flatten_dom_to_str`）与裁剪版 `axtree_raw`（5.3KB/状态，10 万约 0.53GB）。前者使 AXTree-vs-DOM 可在**同一环境内**受控对照；后者使序列化规则的修改变成「重新池化 9 分钟」而非「重采一整轮」。

### 5.1 采集的两个静默失败

**worker 数不得为 5 的倍数。** episode 索引是 `worker_id + k·num_workers`，而 `split_for_episode` 按 `index % 10` 分桶。当 `num_workers` 是 5 的倍数时，每个 worker 只能触及极少数桶，整个 worker——因而整个任务——落进单一划分。下游不会报错，划分文件只是悄悄不再覆盖全部 12 个任务。已加参数校验。

**浏览器消失后会无限空转。** 会话容器重建会清掉 `/tmp` 与 `~/.cache/ms-playwright`。浏览器二进制一旦消失，`env.reset` 永久失败（BrowserGym 的 reset 会先 `self.context.close()`，context 已是 None），而 per-episode 的 `try/except` 把异常吞掉后继续递增索引——**空转 707,739 次，12 个进程满负荷，记录数一条不增**。已加 `--max-consecutive-failures`（默认 20）+ 非零退出码，外层脚本按退出码而非清单文件判断完成。

对策：Chromium 装到 NAS（`browsergym-venv/browsers`），84 个系统 `.so` 固化到 `browsergym-venv/syslibs`，启动脚本设 `PLAYWRIGHT_BROWSERS_PATH` + `LD_LIBRARY_PATH`。采集加 `--resume`：追加模式，按 `(task, episode_index)` 续，**丢弃每个任务的最后一个 episode 重采**（episode 整体 flush，中途被杀会留下不完整的一个）。

> Ubuntu 24.04 的浏览器依赖包名带 `t64` 后缀（`libasound2t64` 等），用旧名会得到 "no installation candidate"。

---

## 6. PCA512 与 Normalization

- PCA 与 normalization **只在 train split 拟合**（`key64_pca fit` 与 `fit_normalization` 均硬性过滤）；
- PCA：2 万 train 状态，4096 → 512；
- normalization：group×channel，70,018 train 状态。三个真实组的 scale 分别为 image 0.151–4.054、detail 0.278–6.288、context 0.074–2.166，比值 85.3。

零宽 prompt 组允许存在但 sigma 被钳到下限，是永不被索引的哑值；加载器只对「有槽却无有效数据」的组报错。

### 6.1 layout 被硬编码在五处，全部静默失效过

prompt 槽回收后，`fit_normalization`、`rebuild_static_key64` 的清单、`a0`、`a1`、两个 bottleneck 的默认值**全都继续按 `(32,12,16,4)` 切分**。归一化把 detail 的后 4 个槽当成 prompt 统计，报告了 4890 个「有效 prompt 槽」——而一个异常都没抛。

现在 layout 的单一来源是 `experiments/state_tokenizer/slot_layout.py`（**无第三方依赖**，torch 与 jax 两侧都能 import；`key_pooling` 再导出它以保持既有引用不变）。`residualmem` 不得 import `experiments`，故两侧一致性由 `tests/state_tokenizer/test_slot_layout.py` 断言，该测试刻意不 import torch 或 jax，两个解释器下都能跑。

---

## 7. A1 / A2

架构沿用 A0 论证出的**地址与内容解耦**：

$$K_s=W_K k_s^{slot},\qquad V_{t,s}=W_V\bar x_{t,s}.$$

$k_s^{slot}$ 只依赖 slot index（「从哪里读」），$\bar x_{t,s}$ 只携带当前状态（「读到什么」）；64 个 learned decoder queries 对应 64 个输出 slots。相同 valid mask 下 attention routing 与状态内容无关，invalid keys 被 mask 屏蔽。不使用 group embedding、source-position embedding、输入投影 $W_x$ 或 query self-attention。

训练目标是所有 valid slots、所有 512 通道的自然平均 MSE；group 指标只报告，不做等权 loss。

### 7.1 v8 结果

| | 配置 | 压缩比 | validation R² |
|---|---|---:|---:|
| A1 | 16 e-token × 512 | 4.0× | 0.8166 |
| **A1（在用）** | **64 e-token × 512** | **1.0×** | **0.99891** |
| **A2** | M=32 子空间，C=256，`--group-weights 1,2,0.5,0.5` | — | **0.8511** |

A2 每状态 16,448 bit（2048 codes × 8 bit + 64 mask bit）。13 项数值门控全过；码本健康：**零坍缩子空间**，活跃类别中位 252/256，perplexity 中位 166.8，dominant share 中位 3.9%。

按组的量化退化 `rho_disc`：**context 50.9 < detail 100.1 < image 217.5**。detail 因 2× 权重被压在 image 的一半以下，加权生效。

**A2 未收敛，是被步数上限截断的**：验证 MSE 单调降到最后一步（step 70000 即最优点），最后 5000 步只降 0.0013，外推再跑一倍步数约到 R² 0.865。无过拟合迹象（验证 MSE 从未回升），码本 perplexity 仍在缓慢上升。

**当前 A1 不是瓶颈**：64×512 → 64×512 是同维自编码器，R²=0.9989 即恒等映射。因此「A1 作为量化前的连续上界」目前是空命题，全部压缩都发生在 A2 的量化。见 §10 待办。

### 7.2 A1 里藏着一个已在 A2 修过的 XLA 缺陷

```python
@jax.jit
def loss_and_grad(current, index):
    return jax.value_and_grad(loss_fn)(
        current, train.xbar_device[index], train.valid_device[index])   # 闭包取批
```

`train.xbar_device` 通过闭包进入 jit，使**整个 train 划分成为编译图的操作数**。7,013 个状态时是 0.92GB 可以编译；70,018 个是 9.18GB，XLA 在 `backend_compile` 段错误，**且无任何 Python 层报错**——日志里只有 fault handler 的栈。

修法：批次作为**参数**传入，gather 在 jit 之外（`take_batch`）。A2 早已如此，A1 漏了。

### 7.3 A1 与 A2 的 e-token 数必须匹配

A1 默认 `--num-e-tokens 16`，A2 默认 64。不匹配时 `load_a1_warm_start` 报 `encoder/e_queries has shape (16,512), expected (64,512)`。A2 用 `--init-a1-checkpoint`（不是 `--init-checkpoint`，后者是 A2 自身的续训入口）。

---

## 8. 门控结果（v8 vs v7）

判据由两项**同时**构成：目标**字面存在性**与 target→element **绑定**，只有前者改善而后者不动视为失败——那说明只优化了 bag of words，agent 仍然不知道该点哪个控件。

### 8.1 字面存在性（免模型）

`target_slot_coverage`：指令目标的字符是否被 raw detail 槽完整覆盖。不依赖任何词表或模型。

| 配置 | 覆盖率 |
|---|---:|
| v7（compact DOM，12 槽） | 0.7214 |
| v8 AXTree 初版 | 0.3423 |
| + `choices` 分组修复 | 0.7032 |
| + 指令优先 | 0.8655 |
| **+ 16 槽（回收 prompt）** | **0.9578** |

`click-option` **1668/1668 满分**；`click-checkboxes` 3489/3716。

### 8.2 绑定

`binding_probe`：$P(\text{ref}\mid x_t, T, \text{target literal})$。

| | v7 (DOM) | v8 (AXTree) |
|---|---:|---:|
| 条件探针 | 0.7935 ± 0.030 | 0.7476 ± 0.048 |
| `literal_ablated`（字面清零对照） | 0.6463 | **0.2877** |
| `random_guess` | 0.1115 | 0.1103 |
| 相对对照的增量 | +0.147 | **+0.460** |
| **对照校正后** | 0.416 | **0.646** |

**raw 数字 v8 略低，但这个比较是误导性的。** v7 的探针不给字面输入也能答对 64.6%，说明那 0.7935 里的绝大部分不是绑定能力，而是「从状态本身猜哪些 ref 通常是目标」。v8 去掉字面后掉到 0.2877（接近随机 0.1103），说明其 0.7476 几乎全部来自真正读懂了字面与元素的对应。

`dom_oracle` 按构造恒为 1.0，是**上界与标签合理性检查**，不是 baseline。

### 8.3 重建保真度（frozen probe，探针只在 clean 上训练）

| 指标 | clean | A1 | A2 | R_disc(A2\|A1) |
|---|---:|---:|---:|---:|
| **value exact** | 0.9963 | 0.9965 | **0.9357** | 0.9390 |
| value position | 0.9991 | 0.9992 | 0.9849 | 0.9841 |
| dynamic AP | 0.9812 | 0.9808 | 0.9655 | 0.9747 |
| task accuracy | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

v7 累计 M32 的 value exact 为 0.7890，v8 为 **0.9357**。

**结论：本轮通过。** 字面 0.7214→0.9578，绑定（对照校正）0.416→0.646，两项都改善且无退化项。

---

## 9. 量具失效清单

本节记录**测量本身骗过我**的情形。这类问题比代码 bug 更贵：它们不报错，只给出一个看起来合理的数字。

**9.1 `key64-static-detail-ranges` 存的是 token 索引，不是字符偏移。** 拿它切原文得到 `'0'`、`' '`、`'t'` 这类单字符，覆盖率算出 0.0066。必须先经 `dom_token_offsets` 换算。

**9.2 raw literal 槽逐 token 发槽。** 一个多 token 目标（`s6WcI`）横跨数个槽，逐槽比较会把它判成缺席。必须先把覆盖的字符位置并成掩码再判断。曾据此错误宣称「70% 的 detail 槽装碎片」，按 span 合并后实为 7.7%。

**9.3 `binding_probe` 的 `MAX_REF = 24` 是按 MiniWoB ref 范围硬编码的。** v7 的 ref 最大 17，BrowserGym 的 bid 到 36，**48.3% 的可点击元素被静默丢弃**，样本从 5384 掉到 2741。探针在残缺候选集上训练，得到 probe 0.5195 / ablated 0.5217——读起来像「绑定完全失效」，实为标签被截断。现已从数据推导 `max_ref`，且截断直接报错。

**9.4 `overlap_words` 在 10 万规模下失效。** episode-target 桶（prevalence < 0.05）里 17 个标签 prevalence 恰为 0（autocomplete 的国家名进了 train 词表却没出现在 validation，零正例的 AP 是退化的），其余是 `cancel`/`ok`/`yes`/`and` 这类虚词。**真正的目标标签是每 episode 随机生成的字符串，永远不可能进入在 train 上拟合的词表。** v7 时代 `prevalence < 0.05` 这个启发式能凑效属于偶然（3000 个 validation 状态、词表更小，`bg`/`jt` 碰巧复现）。字面存在性改用 §8.1 的免词表量具。

**9.5 探针的对照组必须与探针一起训练。** `literal_ablated` 是重新训一个字面清零的探针，不是把已训探针的输入置零——后者是分布外扰动，测到的下降分不清是「信息缺失」还是「输入异常」。

**9.6 目标提取的语法边界。** `choose-list` 的指令是 `"Select Ertha from the list..."`，粗糙正则会把 `from`/`the`/`list` 当成目标；`form-sequence` 的目标是滑块值与序数位置（`"the 3rd checkbox"`），根本不是元素标签。绑定量具只适用于 `click-checkboxes` 与 `click-option`。

**9.7 `dom.find` 会先命中别处子串。** 判断目标是否在文本中必须遍历所有出现位置。曾据此错误宣称「60.5% 的目标缺席」，实为 22.4%。

---

## 10. 待办

**A1 是否应当真正压缩。** 当前 64×512 → 64×512 的 A1 是恒等映射，「连续上界」是空命题；整条链的压缩全部由 A2 的量化承担。已有两个数据点（1×→0.9989，4×→0.8166），缺中间点。这个实验决定的不是「换个配置」，而是 **A1 这一阶段在管线里是否站得住**——若 A1@64 确为恒等，则 A2 直接量化归一化后的 PCA 应给出相同结果，那样 A1 就是纯粹的死重。建议同时跑 A2-without-A1 作为对照。

**A2 延长训练。** 当前是被步数截断的下界，非能力上限。是否值得取决于 value exact 0.9357 是否够用。

**同环境 AXTree-vs-DOM 受控对照。** `dom_control` 已在采集时存好，只差一次抽取。这是把「AXTree 更好」从跨环境比较升级为受控消融的唯一途径（v7/v8 的图像模态本就不可比：截图 160×210 vs 498×321）。

**World Model。** $p(y_{t+1}\mid y_{\le t}, u_{\le t})$。注意：终止步的记录也带 `action`（那个导致终止的动作），但其后继状态从未被记录，**构造转移必须按 `(episode_id, step+1)` 存在来 join**，只看 `action` 非空会高估转移数。

---

## 11. 实现与产物

**layout 单一来源**

- `experiments/state_tokenizer/slot_layout.py`（无第三方依赖，两侧共享）

**采集（browsergym-venv）**

- `experiments/state_tokenizer/collect_browsergym.py`（v7 的 `collect_miniwob.py` 保留不动）
- `scripts_v8_collect.sh`（自愈循环 + 持久化浏览器路径）

**表示（MemCompiler / torch）**

- `common.py`（`compact_axtree`、`axtree_rows`、`derive_probe_labels_axtree`）
- `key_pooling.py` / `static_key_pooling.py` / `rebuild_static_key64.py`
- `merge_records.py`（写 `global_index` = 合并清单行号，抽取链的唯一约定）
- `build_split_721.py`（`--extracted` 现为可选，供全新数据线使用）
- `fit_normalization.py` / `key64_pca.py`
- `filler_vocabulary.py`（**放在 tokenizer 侧**，因 `residualmem/encoders/__init__` 会传递引入 jax）

**瓶颈（ResidualMem / jax）**

- `residualmem/world_model/continuous_bottleneck.py` / `categorical_bottleneck.py`
- `experiments/state_tokenizer/a1_continuous_bottleneck.py` / `a2_categorical_bottleneck.py`

**评测**

- `target_slot_coverage.py`（免词表的字面存在性，§8.1）
- `binding_probe.py`（target→element 绑定，§8.2）
- `semantic_eval.py` / `slot_probe.py` / `reconstruct_store.py`
- `tests/run_tests.py`（pytest 替身；**放在仓库内**，因 `/tmp` 不跨会话存活）

**数据**

```
outputs/state_tokenizer/v8/
  records-merged.jsonl      10 万采集记录（含 axtree_raw / dom_control）
  full-721.jsonl            7:2:1 划分 + global_index + 固定观察提示
  full-721-v2.jsonl         + v2 probe 标签
  features/                 78G  extract_qwen 基础特征
  full-h/                   420G Full-H（**进槽规则定稿前保留**）
  static_features/          55G  Static Key64 + PCA
  key64-static-pca.npz / key64-static-pca-normalization.npz
outputs/a1/v8 + v8.npz
outputs/a2/v8-m32-gw + v8-m32-gw.npz
outputs/semantic_eval/v8_gate.json / v8_binding_clean.json / v8_target_raw_coverage_d16.json
```

保留 Full-H 的理由：本轮三次修改（分组规则、指令优先、layout）**每次都只需重新池化 9 分钟**，而重跑抽取要 1 小时。

**GPU 与 CPU 的选择**：`rebuild_static_key64` 在 CPU 上 0.39 states/s，GPU 上 72.6 states/s（**185×**）。10 万状态从 7 小时降到 9 分钟。

**`kernels` 未安装**：`extract_qwen`/`extract_fixed_prompt` 必须传 `--no-use-kernels`，否则 transformers 5.8.1 直接拒绝启动。
