# WorldMemArena tokenizer-RAG 交接文档

> 更新：2026-08-22。WMA-RAG 轨道的单一入口。
> 权威来源：协议见 `TOKENIZER_WM_HANDOVER.md` §9/§10；tokenizer 训练史见
> `STATE_TOKENIZER_WORKLOG.md`；WM 线见 `WORLD_MODEL_WORKLOG.md` §0/§25。
> 本文件只保留**当前在做的事**和**会绊倒人的东西**，历史过程一律不留。

---

## 0. 现状

**病因已定位到 PCA 基，不是 Qwen/Key64 表示本身。**

WMA smoke 显示 Xbar 的 observation row 在 100 个 top-10 槽位中命中 **0** 条
（Raw-Fused 命中 8），14-way Recall@1 ≈ 随机；域内同一个 head 是 500-way R@1 0.840。

P1 查明 `xbar` 大幅出域、head 表示塌缩；P3.0 进一步把偏移定位到 **PCA 之前**：
原始 Key64 在 WMA 上**完全健康**（范数 34.11 vs 34.21、有效秩 112 vs 121），
但 **42% 的方差落在 512 维 PCA 子空间之外**（域内 8%），inverse-PCA 的
$R^2$ 只有 **0.43**（域内 0.92）。干预验证：**在含 WMA 的语料上重拟合 PCA，
WMA $R^2$ 回到 0.83，域内只掉 2 个点**。

且 effective rank 仅 112–121 ≪ 512 —— **维度够用，是基的朝向错了**。

三条已被否掉的修法：只重训 head（问题在输入不在拟合）；朴素分辨率对齐
（降采样让出域加剧 3.5 倍）；重做整个表示（Key64 没坏）。详见 §3，下一步见 §8。

全量 web benchmark 仍冻结。

---

## 1. 两条实验线

| 线 | 状态 |
|---|---|
| **WMA-RAG**（当前主线） | 域内全过；跨域失败。P1+P3.0 已定位病因为 **PCA 基朝向**（§3.8），修法见 §8 |
| **WM 线**（v8 离散世界模型） | 已收官：validation **7,100.76** bits/transition（3 seeds），赢 source 基线 1,891.53，相对固定宽 16,448 压缩 2.32×。test 仍锁定（需 5 变体 × 3 种子 = 15 个正式 run）。详见 WM worklog，**本文件不再复述** |

四个预注册主实验：`ResidualMem-Instruct-{Xbar,A2}-{Input,L16}-RAG`。检索侧共享
同一个 retrieval head，Input/L16 只影响 reader 路径——**所以这两个轴在检索指标上
没有差别**。当前只有 Xbar 的检索侧跑通；两个 reader connector 从未训练
（磁盘上不存在 `input-connector.pt` / `layer16-connector.pt`，跑 answer 会
直接 RuntimeError）。

---

## 2. 检索协议（判据在这里）

```text
raw full round text ─────────────── Qwen3-VL document encoder ─┐
                                                               ├─ global cosine top-10
screenshot + user + caption ─┬─ Raw-Fused Qwen3-VL encoder ────┤
                             └─ v9 tokenizer → Xbar/A2 → head ─┘
query ─────────────────────────── official Qwen3-VL query encoder
```

- 每个非空 round 恰好两行；空观察 round 仅一行。不做 round dedup。
- **assistant 的观察/计划/Action JSON 绝不进入 tokenizer**——它们是 policy 输出，
  不是 `x_t` 输入。`observation_from_turn` 有 guard，不要绕过。
- caption 被 loader 内联进 user text，适配器必须先移除副本。
- WMA 无 AXTree：零训练适配把 user 文本 + caption 序列化成合法 synthetic AXTree。
- **判据只认 observation row 命中与 paired cosine。** 总体 Recall@10 和
  overlap 0.91 是假象——两边有 49 条**完全相同**的 full-round text row 在掩盖。

实现位置：`residualmem/benchmarks/worldmemarena_tokenizer.py`（序列化）、
`residualmem/latent/frozen_v8_runtime.py`（frozen runtime）、
`residualmem/latent/instruct_bridge.py`（head/reader）、
`eval_framework/memory_adapters/{qwen_embed_adapter,residualmem_instruct_adapter}.py`。

---

## 3. P1 偏移归因（已完成，2026-08-22）

### 3.0 结论

**失效在上游：`xbar` 大幅出域，retrieval head 在 WMA 上发生表示塌缩。**
模态归因（视觉 vs 文本）**不可报**——不是因为算不出来，而是因为几乎没有信号可归因。

⚠️ §3.1–3.7 全部测在 `xbar` 上，**只能说"上游"，说不清是哪一级上游**。
**§3.8 才把它定位到 PCA 基**（Key64 本身健康），§3.9 用干预验证了修法。
读结论请直接看 §3.8/§3.9，本节是通往那里的路径。

三条互相印证的证据：

1. **D0**：WMA 的 `xbar` 相对训练支持域，MMD² 0.86（同分布标尺 −0.0004）、
   最近邻距离 14.60（标尺 3.37）、3σ 尾部质量 2.7×。
2. **表示塌缩**：956 个 head 输出的**两两余弦中位 0.8202**，域内是 0.3418。
   所有 WMA observation 被映射到几乎同一个方向 —— 这直接解释了 R@1 为何是随机：
   **embedding 全都一样就无法排序**。
3. **信号量级**：完整表示相对空对照只提供 0.021 的 cosine 增量
   （0.1937 vs 0.1725），而域内是 0.8711。交互项 |I_VT| = 0.0159 与
   φ_V = 0.0146 同量级 → `share_is_reportable = false`。

**因此不得写「视觉 X%、文本 Z%」。** 正确表述是：两模态强耦合且总信号量接近噪声，
无法单独归因。

### 3.1 两层诊断设计

现有证据全部来自 head 的 4096 维输出侧，**无法区分**「tokenizer/PCA 已经把状态编坏」
与「`xbar` 尚可、只是 head 不在这片区域泛化」。两者修法代价差一个数量级。

- **D0 latent coverage**：`xbar` 是否出域。纯 numpy，无 GPU/teacher/head。
- **D1–D4**：head 输出侧按槽位组与模态分解。

入口：`experiments/state_tokenizer/wma_{extract_xbar,encode_teacher,latent_coverage,shift_attribution}.py`

### 3.2 D0：latent coverage（27 样本 / 956 observation）

`xbar` 本身就是训练坐标下的 z 分数（`GroupChannelNormalizer` 已做 `(x-μ)/σ`，
μ/σ 只在 train 上拟合），所以 `P(|xbar|>3)` 直接是超出训练 3σ 的尾部质量。
脚本会断言域内 train 的 z 均值≈0、标准差≈1（实测 −0.00011 / 1.00014，成立）。

| 统计量 | train | validation（标尺） | **WMA m11** |
|---|---:|---:|---:|
| P(\|z\|>3) | 0.00650 | 0.00639 | **0.01786** |
| MMD²（pooled） | — | **−0.00042** ≈ 0 | **0.8598** |
| 最近邻距离中位数 | — | 3.37 | **14.60** |

### 3.3 D2 模态 2×2（Shapley）

| | 真实截图 V | 空白白图 |
|---|---|---|
| synthetic-AXTree 文本 T | m11 = **0.1937** | m01 |
| 空 AXTree | m10 | m00 = **0.1725** |

**不能用加性分解**：head 非线性，`f(V,T) ≠ f(V,0)+f(0,T)-f(0,0)`。用 Shapley：

$$\phi_V=\tfrac12[(m_{10}-m_{00})+(m_{11}-m_{01})],\quad
\phi_T=\tfrac12[(m_{01}-m_{00})+(m_{11}-m_{10})]$$

效率性 $\phi_V+\phi_T=m_{11}-m_{00}$ 恒成立（写成 assert，实测残差 7.5e-09；
7/7 单元测试通过）。实测 φ_V = 0.0146、φ_T = 0.0066、grand = 0.0211。

**交互项 $I_{VT}=m_{11}-m_{10}-m_{01}+m_{00}$ 必须同时报出**：实测
|I_VT| = 0.0159，与 φ_V 同量级 → 两模态强耦合，**份额不可报**。
脚本用 `share_is_reportable` 字段把这个判断固化，不依赖读者自觉。

### 3.4 D1 槽位组（3-player Shapley）

image / detail / context 三组的 φ **全部为负**（−0.0149 / −0.0295 / −0.0290），
grand = −0.0734。即：把真实内容喂进去，比喂零向量**更差**
（零输入下 head 输出常量，对 teacher 的 cosine 是 0.2670，高于真实表示的 0.1937）。
这是塌缩的另一个侧面，不是"某个组坏了"。

### 3.5 D3 分辨率对照

WMA 1280×720（16:9）vs 训练域 498×321（1.55:1）。**直接 resize 会同时改变分辨率
和压扁几何**，故设两个臂隔离二者：

| 臂 | P(\|z\|>3) | paired cosine |
|---|---:|---:|
| m11 原生 1280×720 | 0.01786 | 0.1937 |
| m11_resampled（直接拉伸） | 0.06171 | 0.2118 |
| **m11_letterbox（保宽高比）** | **0.06464** | 0.2137 |

**两个重采样臂几乎一致**，所以与宽高比畸变无关：**把 WMA 截图降到训练分辨率会让
latent 出域程度加剧 3.5 倍**。合理解释是 MiniWoB 页面元素大而简单，真实网页文字
密集，降采样后文字全糊。cosine 的微小回升（0.19→0.21）发生在塌缩区间内，不构成改善。

**结论：朴素的分辨率对齐不是解法，而是有害的**——这否掉了 P3(a) 的主要手段之一。

⚠️ 本表测在 head 输出上（paired cosine），而 head 已塌缩，诊断力弱。
**已在 image 组 xbar 上复测，结论不变且更强，见 §3.11。**

### 3.6 WMA 数据的结构性事实

全部 **956 个 observation / 27 个 sample**（`web_01..web_27`，磁盘上就是 27 个；
早先"28"的说法是把 `images/` 子目录数进去了）：

- **全部有截图**（956/0）→ §9.8 的「有/无截图」分桶**没有变异，做不了**。
- **user_text 全部为空**（956 条全是 0 字符）→ 所谓「文本」模态**完全是 caption**；
  synthetic AXTree 恒为一行，valid 槽位恒为 49（32 image + 17 文本）。
- 因此 `instruction=observation.user_text or None` 恒为 `None`，
  **指令优先排序在 WMA 上从未生效**——而域内它贡献了 +16.2pp 的目标字面覆盖率。
  这是一条此前未记录的结构性域差异。
- caption 字符数中位 368（339–582）。

### 3.7 全量检索表现

per-sample paired cosine 只在 **0.1532–0.2433**（中位 0.1919）之间，**没有任何一个
样本可用**。27 个样本里 13 个的 same-sample R@1 高于随机（≈ 掷硬币），
R@1 中位 0.04 vs 随机 0.02。

### 3.8 P3.0：偏移定位到 PCA 之前（决定性）

§3.1–3.7 全部跑在 `xbar` 上，即 **PCA 与 normalization 之后**，因此无法区分两种
修复成本差一个数量级的诊断：**(A)** Qwen/Key64 表示本身不跨域，还是 **(B)** Key64
有信息、只是 PCA 基只张成了 MiniWoB 世界。P3.0 在原始 64×4096 Key64 上直接测。

入口 `wma_pre_pca_localization.py`。域内 H 从 `static_features/*/key64-static-bf16.npy`
直接读（两棵坐标树硬链接共享），WMA 侧由 `wma_extract_xbar.py --save-key64` 产出。

**自检**：PCA 产物记录 EVR 累计 0.9186，故 train 的 $r_\perp$ 必须 ≈ 0.0814。
实测 0.0822 ✅。对不上则加载/中心化/投影算子有错，其余数字一概不可信。

**① out-of-subspace 残差** $r_\perp=\|(H-\mu)-(H-\mu)PP^\top\|^2/\|H-\mu\|^2$：

| | train | validation | **WMA** | 倍数 |
|---|---:|---:|---:|---:|
| all | 0.0822 | 0.0816 | **0.4223** | 5.2× |
| image / detail / context | 0.083 / 0.091 / 0.040 | 0.083 / 0.089 / 0.040 | 0.384 / 0.474 / **0.352** | 4.6× / 5.3× / **8.8×** |

**WMA 有 42% 的中心化方差落在 512 维子空间之外**，域内只有 8%。

**② inverse-PCA fidelity**（最接近 memory 目标的指标）：

| | train | validation | **WMA** |
|---|---:|---:|---:|
| $\cos(H,\hat H)$ | 0.9930 | 0.9931 | 0.9659 |
| **$R^2(H,\hat H)$** | 0.9178 | 0.9184 | **0.4274** |

512 维码只保住 WMA 原始 Qwen state 的 **43%** 方差。⚠️ 注意 cosine 仍有 0.966——
它被共享的均值方向主导，**只看 cosine 会得出「保真度还行」的错误结论**，判据用 $R^2$。

**③ H 层面分布（决定性判别）**：

| | train | validation | **WMA** |
|---|---:|---:|---:|
| norm 中位 | 34.21 | 34.19 | **34.11** |
| effective rank | 121.1 | 116.3 | **111.9** |
| 两两余弦中位 | 0.854 | 0.848 | 0.881 |
| MMD²（随机投影后） | — | **0.0002**（标尺） | **0.1267** |
| NN 距离中位 | — | 6.20 | 13.87 |

**Key64 在 WMA 上完全健康**——范数几乎相同、有效秩相当、两两余弦相当。
下游那个表示塌缩（head 输出两两余弦 0.82 vs 域内 0.34）**不是从 H 开始的**。

**④ PCA 放大了差距**：MMD² 从 pre-PCA 的 0.1267 到 post-PCA 的 0.8598，**放大 6.8×**。

**结论：(B) 成立。** 且 **effective rank 只有 112–121，远小于 512**——表示本身低秩，
**512 维完全够用，问题纯粹是基的朝向错了**。这把修法从"扩大训练域重做表示"
收窄成"在含 WMA 的语料上重拟合 PCA"。

### 3.9 干预验证：重拟合 PCA 是否足够

`wma_pca_refit_probe.py`。三种拟合语料，语料行数固定 20,000 以控制数据量变量；
**WMA 按 sample 划分 fit/held-out**（13/14）——同一 sample 内的 observation 来自
连续 round、高度相关，按 observation 划分会泄漏并报出不存在的恢复。

| 拟合语料 | 域内 $r_\perp$ | 域内 $R^2$ | **WMA $r_\perp$** | **WMA $R^2$** |
|---|---:|---:|---:|---:|
| train_only（现行配方） | 0.0866 | 0.9134 | 0.4290 | 0.4183 |
| **train_plus_wma** | 0.1008 | 0.8934 | **0.1541** | **0.8322** |
| wma_only | 0.3582 | 0.5600 | 0.1397 | 0.8595 |

- `train_only` 复现 P3.0 的官方 PCA 数字（0.429/0.418 vs 0.422/0.427）→ harness 有效。
- **混合语料几乎完全恢复 WMA**（$R^2$ 0.418→0.832），**域内只掉 2 个点**。
- `wma_only` 与 `train_only` 近乎对称 → 两域确实需要不同方向，原失衡是**语料构成
  的产物**，非表示的固有性质。

⚠️ **边界**：本探针只证明 $r_\perp$/$R^2$ 可恢复，**未证明重拟合后 retrieval head
就能工作**——head 必须在新坐标上重训，下游塌缩是否随之解决是独立问题。
且 WMA 侧只用了 13 个 web sample、占语料一半，真正重拟合时的混合比例与对其它
WMA 域（非 web）的泛化仍需另定。

---

### 3.10 P3.2：serializer 格式对齐（阴性结果，已冻结）

**问题**：喂给 tokenizer 的 synthetic AXTree 不是训练域的线格式。

| | 训练域（`compact_axtree`） | v1 synthetic |
|---|---|---|
| 行 | `<ref=13 parent=0 tag=button text="no" flags=0,0,0,1/>` | `<node ref="wma_caption_0" tag="textarea" value="..." />` |
| 单条观察 | 14 节点 / 7 候选 | **1 节点 / 1 候选** |
| 分组 | `choices` 6 / `actions` 1 | **全部 `current_state`** |
| 候选长度中位 | **5** 字符（UI 字面） | **395** 字符（散文段落） |

另外 v1 的 `parent=0` **悬空**（无 ref=0 节点），而训练域总以
`<ref=0 parent=0 tag=root .../>` 打头。

**用固定标尺筛 6 个变体**（当前 BrowserGym-only PCA；head 已塌缩，
downstream cosine 对上游改动无诊断力）。299 个 observation：

| 变体 | $r_\perp$ | $R^2$ | MMD² | NN 中位 |
|---|---:|---:|---:|---:|
| **标尺：域内 validation** | **0.0816** | **0.9184** | **0.0001** | **6.131** |
| **aligned（已冻结）** | **0.4121** | 0.4726 | 0.1110 | **13.675** |
| canon+noinstr | 0.4201 | 0.4597 | 0.1140 | 13.721 |
| v1 基线 | 0.4205 | 0.4223 | 0.1262 | 13.879 |
| canonical | 0.4210 | 0.4217 | 0.1283 | 13.858 |
| canon+split+noins | 0.4261 | 0.4610 | 0.1076 | 14.022 |
| canon+split | 0.4304 | 0.4299 | 0.1130 | 14.169 |

⚠️ **$R^2$ 与 MMD² 跨变体不可直接比较**——各变体 valid 槽位数不同（49/50/51），
分母不同。$r_\perp$ 是逐槽比值再聚合，是这里唯一稳健的判据。

**结论：格式不是病因。** 五个变体把 $r_\perp$ 最多动了 2%，而要缩的差距是
0.42 → 0.08。**偏移在观察的内容，不在它的写法**，只有重拟合基能解决。
`split_captions` 反而更差（0.4304），已排除。

**冻结的样式**（`DEFAULT_AXTREE_STYLE`，2026-08-22）：
`canonical_rows + root_node + explicit_no_instruction`。它在每个稳健指标上
最好或持平、无一项变差，且原则上更干净。v1 保留为 `LEGACY_V1_AXTREE_STYLE`
以复现历史数字。

```
<ref=0 parent=0 tag=root flags=0,0,0,1/>
<ref=1 parent=0 tag=textarea value="<no_instruction>" flags=0,0,0,1/>
<ref=2 parent=0 tag=textarea value="A Google Forms job application form is..." flags=0,0,0,1/>
```

**三处故意不对齐**：

1. **字面必须放 `value=` 不能放 `text=`。** `static_candidates` 只对
   ACTION/CHOICE/CHECKABLE 标签处理 `text`，`tag=textarea text="..."` 匹配不上
   任何规则 → **候选 0 个，caption 在进入槽位竞争前被整个静默丢弃**
   （实测验证；正是 §4.2 当年丢掉 52% 目标的同一个坑）。已写进
   `test_wma_serializer.py` 锁死。
2. **root 不带 text**：训练域 root 携带任务标题，WMA 观察里没有标题，
   编一个是造假不是对齐。
3. **层级是平的**：WMA 没有真实父子结构，全部 `parent=0`（现在能解析到 root）。

⚠️ **副作用**：空观察兜底从 1 节点变成 2 节点，故 m00/m10 两个消融臂的基线改变，
§3.2–§3.7 的数字对应 v1 样式。冻结样式的重测见 §3.11。

---

### 3.11 冻结样式复测：残余偏移由 image 槽位主导

用冻结 serializer 重抽 27 个 web 样本（956 observation）后复跑 §3.8。域内三列
一个数字都没变（serializer 只动 WMA 侧，符合预期），WMA 侧聚合看似小幅改善：

| WMA 全部槽位 | v1 | 冻结 | Δ |
|---|---:|---:|---:|
| $r_\perp$ | 0.4223 | 0.4135 | −0.009 |
| $R^2$ | 0.4274 | 0.4755 | **+0.048** |
| MMD²（PCA 前） | 0.1267 | 0.1041 | −18% |
| effective rank | 111.9 | **73.5** | **−38.4** |
| 占用槽位 | 46,844 | 48,756 | **+1,912** |

**但 +1,912 = 956 × 2**：root 行与 `<no_instruction>` 行给每个观察各加了 2 个
**内容恒定**的槽位。恒定向量单一方向、极易重建，会同时抬高平均 $R^2$、压低
有效秩、把方差挤进头部方向。**所以聚合层面的改善里有稀释成分，不能直接当作对齐。**

按槽位组拆开才干净（in-domain validation 作标尺）：

| 组 | 槽位数 v1→冻结 | $r_\perp$ v1 | $r_\perp$ 冻结 | $R^2$ v1 | $R^2$ 冻结 | 域内 $r_\perp$ |
|---|---|---:|---:|---:|---:|---:|
| **image** | 30,592 → **30,592** | 0.3841 | **0.3841** | 0.3841 | 0.3841 | 0.0827 |
| **detail** | 15,296 → **15,296** | 0.4743 | **0.4689** | 0.2361 | **0.3616** | 0.0892 |
| context | 956 → 2,868 | 0.3521 | 0.1481 | **−1.6977** | 0.5313 | 0.0398 |

三个结论：

1. **image 槽位一个数字都没变**（4 位小数完全相同）。原因是**因果注意力**：图像
   token 在序列里排在 DOM 之前，其 layer-16 隐状态看不见 DOM。**任何 serializer
   改动在机制上都不可能影响 image 槽位**——这不是本次实验的偶然结果，是结构性的。
2. **detail 的改善是真的**：槽位数完全相同（无稀释），$R^2$ 仍 0.2361 → 0.3616。
   canonical 行格式 + 前置的 root/`<no_instruction>` 确实把 caption 的表示推近训练分布。
3. **context 的 $R^2$ = −1.6977 是个病灶**：负值意味着 inverse-PCA 重建**比直接用
   均值还差**。v1 每观察只有 1 个 context 槽位且完全跑出子空间；冻结样式后
   $r_\perp$ 0.352 → 0.148（域内 0.040），是三组里最接近训练的。

**因此聚合 $r_\perp$ 那 2% 的改善主要来自 context 组的重新加权，而非普遍对齐。**
§3.10「格式不是病因」的结论不变，且理由更准确了。

**残余偏移由 image 主导**：image 占全部槽位的 **66%**，$r_\perp$ 0.3841 对域内
0.0827，而 serializer 永远碰不到它。这决定了 §8 的优先级——重拟合必须先修
**视觉子空间**。

**D3 复测（在上游指标上，结论不变且更强）**：§3.5 原本测在已塌缩的 head 输出上，
诊断力弱。改测 image 组 xbar（每观察恒 32 槽，无稀释）：

| 臂 | z_std | P(\|z\|>3) | 槽范数中位 |
|---|---:|---:|---:|
| 训练基准 | 1.000 | 0.0061 | 21.42 |
| m11 原生 1280×720 | 1.267 | 0.0255 | 27.82 |
| m11_resampled（缩到 498×321） | **1.778** | **0.0925** | **39.41** |
| m11_letterbox（保宽高比） | **1.807** | **0.0970** | **40.06** |

降采样把 image 槽位推得**更远**：尾部 2.5% → 9.3%（训练的 15 倍），范数 +42%。
letterbox ≈ resample → 与宽高比无关。**§3.5 的判决在上游复现，可放心引用。**
（`--extra-arm` 只算 marginals 不算 MMD/NN，是脚本的小缺口；3.6 倍的尾部差距
不是边缘判断，未为此改代码。）

WMA 原生 image 槽位的偏移方向是 **范数偏大 + 方差偏大 + 尾部偏厚**（27.82 / 1.267 /
0.0255 对 21.42 / 1.000 / 0.0061），即被"放大"而非被"打散"——与 detail 组恰好相反
（detail 的 z_std 0.911 < 1、尾部 0.0070 低于训练，是被"压扁"）。

产物：`wma-pre-pca-localization-frozen.json`、`wma-latent-coverage-frozen.json`、
`wma-latent-coverage-frozen-arms.json`。

---

## 4. 关键数字

### 4.1 WMA 跨域 smoke（web_01 final，25 sessions / 49 text rows / 14 obs / 10 questions）

| 指标 | Raw-Fused | Xbar |
|---|---:|---:|
| overall evidence Recall@10 | 0.765 | 0.725 |
| **top-10 observation 命中（100 槽位）** | **8** | **0** |
| 14-state paired cosine | — | 0.1836 |
| 14-way R@1 | — | 0.0714（随机 0.0714） |

✅ **provenance 缺口已关闭（8-22）**：v1 报告不记录 PCA hash 且 `.tmp/` 与 shell
history 被会话容器重建清空，一度无法判定用的是新坐标还是作废坐标。P1 用**官方
PCA** 重编码 web_01 后**逐位复现** 0.1836 / 0.0714 / 0.2958，证明 smoke 当时用的
就是官方坐标，结论有效。报告格式已升到 `..._smoke_v2`，新增 `coordinates` 块
记录 pca/normalization/head 的路径与 SHA256，杜绝再次丢失。

### 4.2 坐标错配的定量标尺

`coordinate_provenance_check.py`，域内 500 条 validation：

| 臂 | paired cosine | 500-way R@1 | MRR |
|---|---:|---:|---:|
| 正确坐标（cache 自带 / 从新树重建，逐位相同） | **0.8711** | 0.840 | 0.8946 |
| 同一批状态，作废 PCA 坐标 | **0.3919** | 0.004（随机 0.002） | 0.0138 |
| 参照：WMA 跨域 smoke | 0.1836 | 0.0714 | 0.2958 |

两个用途：(1) **排除了「0/100 是 PCA 混用造成的」**——坐标全错也只到 0.3919，
WMA 比它还低；(2) 给 D0/D1 一个「已知的坏」当标尺。

### 4.3 P2 重训（2026-08-22，绑定官方 PCA）

新产物：`outputs/a1/v9-instruct-pca20k-balanced{,.npz}`、
`outputs/a2/v9-instruct-pca20k-balanced-m32-gw{,.npz}`。A1 10k 步约 10 分钟，
A2 70k 步约 37 分钟，八项数值门禁全过。

| 指标 | 新（官方 PCA） | 旧（作废 PCA） |
|---|---:|---:|
| A1 validation R² | 0.99557 | 0.99868 |
| A2 validation R² | 0.83198 | 0.86329 |
| A2 `delta_disc` / `rho_disc` | 0.16323 / 37.93 | 0.13541 / 103.33 |
| collapsed subspaces | 0 | 0 |
| perplexity median | 170.20 | 165.50 |

**读法（防误读为「重训把结果做差了」）**：

- **R² 变低是预期的，两列不可直接比较。** 旧 PCA 的 2,000 个拟合状态全部来自
  `click-button-v1`，投影分布窄、重建容易；新 PCA 跨 12 任务 task-balanced，
  是更难也更诚实的目标。A1 与 A2 同向变化正符合「目标变难」而非「实现变差」。
- **`rho_disc` 103→38 不是改善。** 它是量化增量相对连续基线的比值，而新 A1 的
  基线误差本身更大，分母被污染。绝对量化损伤 `delta_disc` 实际**升高**了
  （0.1354→0.1632）。**不得据此宣称量化质量提升。**
- `best_selection_step` 仍等于末步 → **A2 依旧被步数上限截断、未收敛**，是下界
  不是能力上限。

### 4.4 tokenizer 域内门禁（v8）

目标字面进 raw 槽 0.7214→0.9578；绑定探针（对照校正）0.416→0.646；
A2 value exact 0.9357；A2 码率 16,448 bit/state（2,048 codes × 8 bit + 64 mask）。

---

## 5. 产物与坐标树

### 5.1 ⚠️ 两棵同名坐标树（最容易出错的地方）

`outputs/state_tokenizer/` 下两棵目录结构与文件名**完全相同**的树：

| 树 | `key64-static-pca.npz` SHA256 | 状态 |
|---|---|---|
| `v9-instruct-pca20k-balanced/` | `f98c517c…f6efb2a` | ✅ **唯一有效坐标**（20k task-balanced，EVR 0.9186） |
| `v9-instruct/` | `6d2df9b5…f001a26` | ❌ 作废（2,000-state first-N，全来自 `click-button`） |

作废树**不能删**——420 GiB 的 Full-H 就在里面。原始 4096 维 Key64 特征在两棵树间
**硬链接共享**（实占 55G + 6.2G，不是 110G），只有 PCA 投影各自独立。

取错坐标**不报错**，只静默给出错误结果（量级见 §4.2）。现已有防线：

- `experiments/state_tokenizer/pca_binding.py`：读回各产物自带的 provenance
  （`static_features/*/key64-static-pca-artifact.sha256`、normalization npz 里的
  `pca_sha256`）做三方比对，并比对官方 hash。9/9 单元测试，torch/jax 双环境通过。
- `scripts_v9_instruct.sh` 的消费坐标 stage 会先跑 `coord_preflight`；
  `./scripts_v9_instruct.sh coord-check` 可单独审计；重拟合 PCA 用
  `EXPECTED_PCA_SHA256` 覆盖。
- `a2_categorical_bottleneck.py` 的 `_assert_a1_coordinates`：A1/A2 同为 64×512，
  几何检查发现不了坐标错配。守卫在 `build_split` 之前触发，不浪费加载。

### 5.2 主要产物

| 产物 | 路径 |
|---|---|
| canonical v8（100,008 states，7:2:1 = 70,018/20,011/9,979） | `outputs/state_tokenizer/v8` → `v8-recovered` |
| Full-H（~420 GiB） | `outputs/state_tokenizer/v9-instruct/full-h` |
| 官方 PCA + normalization + 投影 | `outputs/state_tokenizer/v9-instruct-pca20k-balanced/` |
| A1 / A2（官方坐标） | `outputs/a1/v9-instruct-pca20k-balanced{,.npz}`、`outputs/a2/v9-instruct-pca20k-balanced-m32-gw{,.npz}` |
| teacher cache（5,000 train + 500 val） | `outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/train-cache-fused-observation.npz` |
| retrieval head（best step 5,000，域内 R@1/5/10 = 0.840/0.956/0.986） | `…/retrieval-head-fused-observation.pt` |
| WMA xbar（27 样本 × 6 臂） | `…/wma-xbar/` |
| WMA teacher | `…/wma-teacher/` |

---

## 6. 环境与入口

| 环境 | 位置 | 用途 |
|---|---|---|
| JAX | `.venv-jax`（jax 0.4.33 + ptxas 12.8.93） | A1/A2、WM、bridge |
| 采集 | `browsergym-venv` | BrowserGym/Playwright（Ubuntu 24.04 冻结栈，勿在 22.04 跑） |
| torch | `/mnt/data/public_tools/miniconda3/envs/qwen-vl` | Qwen 抽取、Qwen3-VL teacher |
| 权重 | `models/Qwen3.5-9B`、`models/Qwen3-VL-Embedding-8B` | 均已落盘 |

`scripts_v9_instruct.sh <stage>`：modality-lengths / extract-full-h /
rebuild-static / pca-fit / pca-transform / normalization / a1 / a2 /
bridge-cache / retrieval-head / input-reader / layer16-reader /
**coord-check** / **wma-smoke**。

⚠️ 脚本用两个坐标树变量，**不要混**：`V9_RAW`（原始抽取，Full-H 只在这里）
与 `V9_COORD`（官方 PCA + normalization + 投影）。

JAX 前置：`export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1`

测试：**没有 pytest**，用 `tests/run_tests.py` 替身。注意它**不支持
`@pytest.fixture`**，用了 fixture 的测试文件会在 import 期就失败（预先存在，
`test_a1/a2_*` 两个文件如此）。

⚠️ 本机三个路径是**同一块盘**，可互换：`/home/luzheng/workspace/iclr/czs`
≡ `/mnt/data/users/luzheng/workspace/iclr/czs` ≡ `/mnt/workspace/users/luzheng/...`
（inode 相同）。根 README 与 `experiments/world_model/README.md` 里的
`/root/nas/...` 路径已全部失效。

---

## 7. 已知坑

### 7.1 工程

- `extract_qwen` / `extract_fixed_prompt` 必须 `--no-use-kernels`（transformers 5.8.1）。
- `rebuild_static_key64` 必须 `--image-grid-thw 1 20 32`（默认值是 v7 旧网格，
  用错**静默**错位池化）。
- `--instruction-records` 必须传**采集清单**（传错则指令优先排序静默失效）。
- A1 的 `--num-e-tokens` 必须与 A2 匹配（都用 64）；A2 用 `--init-a1-checkpoint`
  而非 `--init-checkpoint`（后者是 A2 自己的 resume 入口）。
- JAX 长任务 `PYTHONFAULTHANDLER=1` 不要省（XLA 段错误无 Python 报错）；
  A1/A2 运行中日志 0 字节正常，用 `ps -o etime,pcpu` 判断。
- 采集 `--num-workers` 不得为 5 的倍数（episode 分桶会塌缩）。
- **`pgrep -f` / `pkill -f` 会匹配发起 shell 自身**，本轮已被这个坑骗到两次
  （一次误判进程还在，一次 kill 没生效）。判断存活用
  `ps -eo pid,args | awk '/[p]attern/'`，按 pid 操作。
- **本机 shell 是 zsh，不做无引号变量的词分割**。`for x in $LIST` 会把整个字符串
  当成一项——曾据此误读一次测试结果。用数组或显式 `bash <<'EOS'`。
- 会话容器重建清 `/tmp` 与 `~/.cache`：日志写仓库、长任务加 `--resume`。
- 跨环境共享常量只能放无第三方依赖模块（`slot_layout.py`、`pca_binding.py`）。

### 7.2 量具（对 P1 直接相关）

- `key64-static-detail-ranges` 是 **token 索引**不是字符偏移。
- raw literal 槽逐 token 发槽：多 token 目标须按 span 合并掩码再判存在性。
- 探针的 `literal_ablated` 对照必须**重新训练**，不能对已训探针置零输入。
- `dom.find` 判存在须遍历所有出现位置。
- 绑定量具只适用 `click-checkboxes` / `click-option`。
- MAX_REF 硬编码曾静默丢弃 48.3% 候选——**任何按范围硬编码的量具先查截断率**。

### 7.3 协议

- A2 的 2,048 个 code **全部计费**（与 valid mask 无关）。
- PCA/normalization 只在 train split 拟合；追加数据必须复用旧坐标。
- `global_index` = 合并清单行号：新增数据若排序插到中间，55G 特征库整体错位
  （追加须让新 id 排在最后）。
- 两个仓库**各有一个 `fused_observation_text`**，实现不同。实测在 WMA web 上
  逐字相同（因为 user_text 恒空，两条 strip 路径都是 no-op），但这是数据的巧合
  不是保证。teacher 编码读抽取产物里存的 `fused_text`，不二次推导。
- `QwenVLEmbedAdapter.__init__` 若无 base URL 会**eager 加载 16 GB
  SentenceTransformer**。只要 xbar 不要检索时，覆盖 `_encode_docs` **和**
  `_encode_query`（两条路径都要，QA 阶段走后者）并声明一个不可达 base URL。
- ⚠️ `WorldMemArena` 是上游 benchmark 仓库（origin = github.com/UCSB-AI/WorldMemArena），
  **禁止推送**；`eval_framework/baselines/Qwen3-VL-Embedding-8B/weights/` 是 16 GB
  权重目录，切勿 `git add -A`。

---

## 8. 接下来

**P1 / P3.0 / P3.2 已完成（§3）。** 病因定位到 PCA 基的朝向，Qwen/Key64 表示本身
健康。已否掉四条修法：只重训 head、朴素分辨率对齐（§3.5 + §3.11 两次独立验证）、
重做整个表示、serializer 格式对齐（§3.10/§3.11，阴性）。

**残余偏移由 image 槽位主导**（66% 的槽位，$r_\perp$ 0.3841 对域内 0.0827，且
serializer 在机制上碰不到）。重拟合的首要目标是**视觉子空间**，判据也应以
image 组的 $r_\perp$ 为主，而非聚合值。

### 8.1 gate 纪律（本轮定下，务必遵守）

**不要一口气跑完 PCA → A1 → A2 → head。** A1/A2 对「mixed PCA 能否解除 Xbar
retrieval collapse」这个问题完全无关：若 $R^2$ 修好了但 head 两两余弦仍是 0.80，
说明 coverage 已修复而 **retrieval mapping 有独立的跨域问题**，此时重训 A2 是纯浪费。

```
serializer 冻结 ✅ → mixed PCA → normalizer → Xbar
   → bridge-cache（仅 Xbar 侧，teacher 复用）→ retrieval head
   → ★ WMA-web Xbar retrieval GATE ★
        ├── FAIL → 停，不训练 A1/A2，转查 retrieval mapping
        └── PASS → A1 → A2 → A2 retrieval → reader connector
```

**依赖纪律**：serializer 必须在最终 PCA **之前**冻结（已完成，§3.10）。任何
serializer 改动都会改变 $H_{16}$，已拟合的 PCA 就不再是同一个坐标系。

### 8.2 拟合语料构成（已抽取）

拟合/评测分离不可妥协：**评测域 `agent/gui/web` 全部 27 个 sample 一条都不进拟合**，
否则测的是记忆不是泛化。

| 子类（loader 字面值） | 样本 | 截图分辨率 | 入拟合 |
|---|---:|---|---|
| `agent/vab/webarena-lite` | 39 | 1280×720 | ✅ |
| `agent/arena/image_edit` | 30 | 1280×720 | ✅ |
| `agent/arena/word_docs` | 25 | 1280×720 | ✅ |
| `agent/arena/excel` | 24 | 1280×720 | ✅ |
| `agent/arena/file_mgmt` | 19 | 1280×720 | ✅ |
| `agent/vab/mobile` | 19 | 1440×3200（中位 2.59 MP） | ✅ 待探针裁定 |
| ~~`agent/vab/css`~~ | ~~20~~ | **最高 1280×37015** | ❌ 见下 |
| ~~`agent/arena/web`~~ | ~~27~~ | 1280×720 | ❌ **评测域** |
| **合计入拟合** | **156** | | |

⚠️ loader 的子类字面值与磁盘目录**不一致**（磁盘是 `agent/gui/*`，loader 是
`agent/vab/*` 与 `agent/arena/*`，且 `webarena-lite` 是连字符）。按目录名写过滤器
会静默拿到空语料；脚本已在不匹配时抛 KeyError 并列出可用值。

**为什么剔除 `css`**：它的截图是**整页滚动长图**，最高 1280×37015、单图约
15,800 个视觉 token，241 张里有 143 张超过 7,000 token，最高的几张**根本塞不进
`max_length=8192`**（`prepare_inputs` 会抛错，实测在 `css_02` 崩掉）。两个理由：

1. 只留能塞进去的那些，是**按页面高度做的有偏子采样**——悄悄改变了"css"在语料里
   的含义，属于 §7 反对的静默截断。
2. 更重要：**§3.11 显示残余偏移由 image 主导**，而 css 的视觉体制（6,000–15,800
   token 的整页截图）与评测域的 880-token 视口截图根本不是一类。把它塞进去等于
   让 PCA 把方差花在 `web` 从不访问的方向上——**正是我们要修的那个失败模式，
   只是换了个方向**。

`mobile` 保留但待定：它能塞进去（最高 4,500 token），仍是视口截图，但 2.59 MP
对评测域的 0.92 MP 仍有差距。**抽取是可逆的，拟合才是承诺**——由 §8.3 的探针在
留出的 27 个 web 上按子类裁定。

### 8.3 P3.1 重拟合：混合比例已实测，工作点 25%

**已完成（8-22）**，入口 `wma_pca_mixture_sweep.py`。与 §3.9 旧探针的区别在协议：
旧的把同一个目录对半切（同分布内插），新的**拟合语料全是非 web 子类（156 sample）、
评测是 27 个 web 且一条没进拟合**——真正的跨任务泛化。旧协议数字
（`wma_pca_refit_probe_v1`）被 §3.9 引用，故未原地改语义。

| WMA 占比 | 域内 $r_\perp$ | 域内 $R^2$ | **WEB image $r_\perp$** | WEB all | WEB $R^2$ |
|---|---:|---:|---:|---:|---:|
| 0.00（现行） | 0.0867 | 0.9133 | 0.3889 | 0.4199 | 0.4677 |
| **0.25（选定）** | 0.0938 | 0.9051 | **0.1264** | 0.1194 | 0.8618 |
| 0.50 | 0.1004 | 0.8949 | 0.1212 | 0.1101 | 0.8813 |
| 0.75 | 0.1125 | 0.8755 | 0.1179 | 0.1040 | 0.8930 |
| 1.00 | **0.3185** | **0.6219** | 0.1102 | 0.0949 | 0.9041 |

**25% 拿到全部可得收益的 94%，只付出全部代价的 2.8%**（image $r_\perp$ 的总改善空间
0.2787，25% 已吃掉 0.2625；域内 $R^2$ 的总代价 0.2914，25% 只付 0.0082）。
**纯 WMA 会让域内崩掉**（$R^2$ 0.913→0.622），所以不是越多越好。
主判据用 image 组的 $r_\perp$ 而非聚合值——§3.11 已说明聚合会被 context 组的重新加权带偏。

**子类留一消融**（在 0.50 处，固定总行数 20,000）：拿掉任一子类，
WEB image $r_\perp$ 最多变动 0.003（0.1200–0.1237 对基准 0.1212）。
**没有哪个子类承重，也没有哪个有害**；`mobile`（2.59 MP，评测域 0.92 MP）拿掉后
0.1211 与全留几乎相同 → **六个子类全留**。

⚠️ **边界**：这只证明**表示层面**的 coverage 可恢复，**未证明** retrieval head
随之不再塌缩——head 必须在新坐标上重训，那是独立问题，即下面的 GATE。

落地时：重拟合后必须重走 normalization → bridge cache → retrieval head，
`EXPECTED_PCA_SHA256` 需同步更新，**所有历史码字与数字作废**。
`pca-transform` 约 30–60 分钟（100,008 states 重投影，I/O 为主）。

**GATE 判据（预注册，跑之前定死）**：

| 判据 | 现状 | PASS 阈值 |
|---|---:|---|
| **① WMA head 输出两两余弦**（主判据） | 0.8202（域内 0.3418） | **< 0.60** |
| ② top-10 中 observation 命中 | 0/100（Raw-Fused 8/100） | **> 0** |
| ③ paired cosine 相对空对照增量 | +0.021（0.1937 vs 0.1725） | 显著高于空对照 |
| ④ same-sample R@1 | 中位 0.04（随机 0.02） | 明显高于随机 |
| ⑤ 域内 500-way R@1 不退化 | 0.840 | **≥ 0.80** |

**FAIL（①不达标）→ 停，不训练 A1/A2。**

**P4 reader connector（Input/L16）。** 检索侧通过后做。定义见 handover §9.7。

**P5（可选、独立轨道）WM 线续作。** 扩数据（幂律外推 2× ≈ +599）或补
5 变体 × 3 种子解锁 test。重采可复现性已验证。

**遗留待办**（非紧急）：A1 恒等映射质疑（64×512→64×512 近恒等，可能是死重，
建议 A2-without-A1 对照）；AXTree-vs-DOM 同环境受控对照（`dom_control` 已存，
只差一次抽取）。

---

## 9. 本轮（2026-08-22）改了什么

代码与产物变更清单，供审阅与回溯：

**新增**

| 文件 | 作用 |
|---|---|
| `experiments/state_tokenizer/pca_binding.py` | 三方坐标绑定校验（features ↔ normalization ↔ pca），读回各产物自带的 provenance；无 torch 依赖，两个环境都能用 |
| `experiments/state_tokenizer/coordinate_provenance_check.py` | 坐标错配的定量标尺（§4.2），也是复现 0.8711/0.3919 的入口 |
| `experiments/state_tokenizer/wma_extract_xbar.py` | WMA observation 的 `xbar` 抽取，6 个消融臂；**包装 tokenizer 而非重实现 session 遍历**，observation 集合在构造上与检索路径一致 |
| `experiments/state_tokenizer/wma_encode_teacher.py` | WMA teacher 编码，模型只加载一次 |
| `experiments/state_tokenizer/wma_latent_coverage.py` | D0 |
| `experiments/state_tokenizer/wma_shift_attribution.py` | D1–D4，Shapley 分解 |
| `experiments/state_tokenizer/wma_pre_pca_localization.py` | P3.0：PCA 之前的偏移定位（r⊥ / inverse-PCA fidelity / H 层面分布 / effective rank）；纯 numpy，两环境可跑 |
| `experiments/state_tokenizer/wma_pca_refit_probe.py` | P3.0 的干预验证：三种拟合语料的留出评估，WMA 按 sample 划分防泄漏 |
| `tests/state_tokenizer/test_pca_binding.py` | 9 项，torch/jax 双环境通过 |
| `tests/state_tokenizer/test_wma_shapley.py` | 7 项，含效率性恒等式与闭式公式比对 |

**修改**

- `scripts_v9_instruct.sh`：拆 `V9_RAW` / `V9_COORD` 两棵坐标树；消费坐标的
  stage 加 `coord_preflight`；`A1_RESULT`/`A2_RESULT` 默认值改为
  `-pca20k-balanced` 命名以免覆盖旧产物；新增 `coord-check` 与 `wma-smoke` stage
  （后者文档早已声称存在但实际没有）。
- `a2_categorical_bottleneck.py`：新增 `_assert_a1_coordinates`，在 `build_split`
  之前拦截 A1/A2 坐标错配。
- `frozen_v8_runtime.py`：`encode` 新增 `validate` / `image` / `return_key64` 三个
  仅供诊断使用的参数（空对照要绕过校验，分辨率对照要替换图像来源，P3.0 要 PCA
  之前的 64×4096），metadata 增记 `image_size`。
- `worldmemarena_retrieval_smoke.py`：报告升到 `..._smoke_v2`，新增 `coordinates`
  块记录 pca/normalization/head 的 SHA256。
- `WorldMemArena/README_DATASET.md`：四个环境变量路径全部失效已修正，
  补坐标陷阱说明与复现命令。

**订正的错误记载**

- 交接文档原写「A1 绑定新 PCA hash」是错的，实际绑的是作废 PCA
  （`pca_sha256 = 6d2df9b5…`）。这直接导致 P2 的定义有误，已更正为"先 a1 再 a2"。
- 「`agent/gui/web` 有 28 个 sample」是错的，实际 27 个。

---

## 10. 文档导航

| 文件 | 定位 | 可信度 |
|---|---|---|
| 本文件 | WMA-RAG 单一入口（**结论**） | ✅ 以本版为准 |
| `WMA_RAG_WORKLOG.md` | WMA-RAG 线的**过程记录**：做了什么、为什么这么定、踩了什么坑 | ✅ 权威 |
| `TOKENIZER_WM_HANDOVER.md` | 移植手册：数据契约、cache 格式、§9 WMA-RAG 协议、§10 v8 重采记录 | ✅ 权威 |
| `STATE_TOKENIZER_WORKLOG.md` | tokenizer 训练史（§9 量具失效、§10 待办） | ✅ 权威 |
| `WORLD_MODEL_WORKLOG.md` | WM 线（§0 结案表、§25 正式化、§22.2 噪声底、§23.7 秩筛查） | ✅ 权威 |
| `技术报告_ResidualMem.md` | 愿景与系统设计（H1–H4、公式、benchmark、风险 §18） | ⚠️ 头部 8-12 注记过时 |
| `README.md` | 方法概览 + 工程坑 | ⚠️ NAS 路径失效，坑有效 |
| `refine-logs/*`、`experiments/world_model/README.md` 的 "Current decision" | 停在 8-12「M1 失败即停」 | ❌ 已被 WM worklog §25 推翻 |

WorldMemArena 侧：`README_DATASET.md`（含坐标陷阱说明）、`README_ResidualMem.md`。
