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

**P1 + P3.0 已完成（§3）。** 病因定位到 PCA 基的朝向，Qwen/Key64 表示本身健康。
已否掉三条修法：只重训 head、朴素分辨率对齐、重做整个表示。

**P3.1 重拟合 PCA（主路，已有干预证据支持）。** §3.9 显示混合语料能把 WMA 的
$R^2$ 从 0.42 拉到 0.83 而域内只掉 2 个点。落地要定的事：

1. **混合比例**——探针用的是 1:1 且只有 13 个 web sample，需要扫比例并确认对
   非 web 的 WMA 域（embodied / mobile / excel 等）泛化。
2. **重拟合后必须重走** normalization → A1 → A2 → bridge cache → retrieval head。
   §5.1 的坐标绑定门禁会拦住任何混用，但要记得 `EXPECTED_PCA_SHA256` 需同步更新，
   且**所有历史码字与数字作废**。
3. **验证点**：重训 head 后下游塌缩是否解决（head 输出两两余弦能否从 0.82 降回
   0.34 量级）——这是 §3.9 明确**未**证明的部分。

**P3.2 serializer 格式对齐（便宜，可与 P3.1 并行）。** 评价指标**不能再看 head
cosine**——head 已塌缩成近似常量，downstream 指标对 upstream 改动的诊断力极弱。
应看 $r_\perp$ / MMD(xbar) / NN 距离 / $P(|z|>3)$。候选改动做成可单独开关的变体
以便单独归因：统一 observation prompt skeleton；无指令时显式写固定
`<no_instruction>` 而非走另一条模板分支；synthetic AXTree 用与训练相同的 canonical
serialization；caption 包裹/分隔符/special token 格式一致；不 resize 截图。
§3.6 显示 WMA 的 64 槽里只有 17 个携带文本、恒为一行 caption、指令优先排序从未
生效——这几处格式差异嫌疑很大。

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
| 本文件 | WMA-RAG 单一入口 | ✅ 以本版为准 |
| `TOKENIZER_WM_HANDOVER.md` | 移植手册：数据契约、cache 格式、§9 WMA-RAG 协议、§10 v8 重采记录 | ✅ 权威 |
| `STATE_TOKENIZER_WORKLOG.md` | tokenizer 训练史（§9 量具失效、§10 待办） | ✅ 权威 |
| `WORLD_MODEL_WORKLOG.md` | WM 线（§0 结案表、§25 正式化、§22.2 噪声底、§23.7 秩筛查） | ✅ 权威 |
| `技术报告_ResidualMem.md` | 愿景与系统设计（H1–H4、公式、benchmark、风险 §18） | ⚠️ 头部 8-12 注记过时 |
| `README.md` | 方法概览 + 工程坑 | ⚠️ NAS 路径失效，坑有效 |
| `refine-logs/*`、`experiments/world_model/README.md` 的 "Current decision" | 停在 8-12「M1 失败即停」 | ❌ 已被 WM worklog §25 推翻 |

WorldMemArena 侧：`README_DATASET.md`（含坐标陷阱说明）、`README_ResidualMem.md`。
