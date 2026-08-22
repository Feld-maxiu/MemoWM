# WMA-RAG 线工作日志

WorldMemArena 检索/QA 线的施工记录。世界模型线（bits/transition）见
`WORLD_MODEL_WORKLOG.md`；tokenizer 本体的历史见 `STATE_TOKENIZER_WORKLOG.md`；
交接与坐标树规矩见 `TOKENIZER_WM_HANDOVER.md`；诊断结论的正式版见
`WORLDMEMARENA_TOKENIZER_RAG.md`（本文件是过程记录，那里是结论）。

---

## 2026-08-22

这一天做完了三件事：把 serializer 定稿并证明格式不是病因；把残余偏移进一步定位到
image 槽位；用真正的跨任务留出集确定了混合 PCA 的工作点。另外顺手修了两个会污染
判断的缺陷。

### 1. P3.2 serializer 对齐（阴性结果，已冻结）

**动机**：喂给 tokenizer 的 synthetic AXTree 不是训练域的线格式。训练是
`<ref=13 parent=0 tag=button text="no" flags=0,0,0,1/>`，v1 是
`<node ref="wma_caption_0" tag="textarea" value="..." />`，一次差五处：`node` 前缀、
带引号的非数字 ref、无 `parent`、无 `flags`、闭合前多一个空格。且 v1 的 `parent=0`
**悬空**（没有 ref=0 节点）。

**做法**：把每条规则做成可单独开关的 `AxtreeStyle` 标志位，用**当前 BrowserGym-only
PCA 作固定标尺**筛 6 个变体。不能用 head cosine 判——head 已塌缩成近似常量，
downstream 对 upstream 改动的诊断力极弱。

299 个 observation 上的结果：

| 变体 | $r_\perp$ | $R^2$ | MMD² | NN 中位 |
|---|---:|---:|---:|---:|
| 标尺：域内 validation | **0.0816** | 0.9184 | 0.0001 | 6.131 |
| **aligned（已冻结）** | **0.4121** | 0.4726 | 0.1110 | **13.675** |
| canon+noinstr | 0.4201 | 0.4597 | 0.1140 | 13.721 |
| v1 基线 | 0.4205 | 0.4223 | 0.1262 | 13.879 |
| canonical | 0.4210 | 0.4217 | 0.1283 | 13.858 |
| canon+split+noins | 0.4261 | 0.4610 | 0.1076 | 14.022 |
| canon+split | 0.4304 | 0.4299 | 0.1130 | 14.169 |

**结论：格式不是病因。** 五个变体把 $r_\perp$ 最多动了 2%，而要缩的差距是 0.42 → 0.08。
`split_captions` 反而更差，已排除。冻结样式是
`canonical_rows + root_node + explicit_no_instruction`，它在每个稳健指标上最好或持平、
无一项变差。v1 保留为 `LEGACY_V1_AXTREE_STYLE` 以复现历史数字。

**三处故意不对齐**，第一处是硬约束：

1. **字面必须放 `value=` 不能放 `text=`。** `static_candidates` 只对
   ACTION/CHOICE/CHECKABLE 标签处理 `text`，`tag=textarea text="..."` 匹配不上任何规则
   → **候选 0 个**，caption 在进入槽位竞争前被整个静默丢弃（实测验证）。这正是 §4.2
   当年丢掉 52% 目标的同一个坑，已连同反事实一起写进 `test_wma_serializer.py` 锁死，
   防止后来人"顺手对齐"时踩雷。
2. **root 不带 text**：训练域 root 携带任务标题，WMA 观察里没有标题，编一个是造假。
3. **层级是平的**：WMA 没有真实父子结构。

**为什么必须在拟合 PCA 之前冻结**：任何 serializer 改动都会改变 $H_{16}$，
已拟合的 PCA 就不再是同一个坐标系。这条依赖关系写进了流程。

### 2. 冻结样式复测 → 残余偏移定位到 image 槽位

用冻结 serializer 重抽 27 个 web 样本（956 observation）后复跑 P3.0。
聚合看似小幅改善（$r_\perp$ 0.4223→0.4135，$R^2$ 0.4274→**0.4755**，MMD² −18%），
但**占用槽位从 46,844 涨到 48,756，恰好 +1,912 = 956×2**——root 行与
`<no_instruction>` 行给每个观察各加了 2 个**内容恒定**的槽位。恒定向量极易重建，
会同时抬高平均 $R^2$、压低有效秩（111.9→73.5）、把方差挤进头部方向。
**所以聚合改善里有稀释成分，不能直接当作对齐。**

按槽位组拆开才干净：

| 组 | 槽位数 v1→冻结 | $r_\perp$ v1 | $r_\perp$ 冻结 | $R^2$ v1 | $R^2$ 冻结 | 域内 $r_\perp$ |
|---|---|---:|---:|---:|---:|---:|
| **image** | 30,592 → **30,592** | 0.3841 | **0.3841** | 0.3841 | 0.3841 | 0.0827 |
| **detail** | 15,296 → **15,296** | 0.4743 | **0.4689** | 0.2361 | **0.3616** | 0.0892 |
| context | 956 → 2,868 | 0.3521 | 0.1481 | **−1.6977** | 0.5313 | 0.0398 |

三个发现：

1. **image 槽位一个数字都没变**（4 位小数完全相同）。原因是**因果注意力**：图像 token
   在序列里排在 DOM 之前，其 layer-16 隐状态看不见 DOM。**任何 serializer 改动在机制上
   都不可能影响 image 槽位**——这不是偶然，是结构性的。
2. **detail 的改善是真的**：槽位数完全相同（无稀释），$R^2$ 仍 0.2361 → 0.3616。
3. **context 的 $R^2$ = −1.6977 是个病灶**：负值意味着 inverse-PCA 重建**比直接用均值
   还差**。冻结样式后降到 $r_\perp$ 0.148，是三组里最接近训练的。

**因此聚合那 2% 主要来自 context 组的重新加权，而非普遍对齐**，"格式不是病因"的结论
不变、理由更准。**残余偏移由 image 主导**：占全部槽位 66%，$r_\perp$ 0.3841 对域内
0.0827，而 serializer 永远碰不到它。这决定了后续优先级。

### 3. D3 复测：缩放有害，在上游指标上复现

§3.5 原本测在已塌缩的 head 输出上，诊断力弱。改测 image 组 xbar（每观察恒 32 槽，
无稀释）：

| 臂 | z_std | P(\|z\|>3) | 槽范数中位 |
|---|---:|---:|---:|
| 训练基准 | 1.000 | 0.0061 | 21.42 |
| m11 原生 1280×720 | 1.267 | 0.0255 | 27.82 |
| m11_resampled（缩到 498×321） | **1.778** | **0.0925** | **39.41** |
| m11_letterbox（保宽高比） | **1.807** | **0.0970** | **40.06** |

降采样把 image 槽位推得**更远**：尾部 2.5% → 9.3%（训练的 15 倍），范数 +42%。
letterbox ≈ resample → 与宽高比无关。**判决在上游复现，可放心引用。**

WMA 原生 image 槽位的偏移方向是**范数偏大 + 方差偏大 + 尾部偏厚**，即被"放大"；
detail 组恰好相反（z_std 0.911 < 1、尾部低于训练），是被"压扁"。

### 4. 一个此前没注意到的结构事实：池化输入基数差 5.5×

`full-h` 的 `modality-lengths` 显示域内截图**恒定产生 160 个 image token**
（`--image-grid-thw 1 20 32`），而 WMA web 1280×720 产生 **880 个**。
静态池化不管有多少 token，一律 `adaptive_avg_pool2d` 压进 32 个 image 槽位：

| | image token | 每槽平均 token 数 |
|---|---:|---:|
| 域内训练 | 160 | **5.0** |
| WMA web | 880 | **27.5** |

**池化算子在 WMA 上处于一个它从未见过的输入基数区间。** 这不是编码器的问题，也不是
PCA 的问题，是手写池化本身不迁移。⚠️ 但要注意：单纯的平均会让范数**缩小**，而实测
是范数**增大**，所以基数不匹配是一个值得命名的结构性风险，**不能断言它是主因**，
内容差异同时在起作用。要证伪需要 token 计数对照（把 880 先池化到 160 再走同样规则），
已列为待办、暂缓。

### 5. 抽取拟合语料：两次崩溃，两个真实问题

**崩溃一：`css` 子类塞不进 max_length。**
`ValueError: image + instruction alone exceed max_length=8192`。查下来 css 的截图是
**整页滚动长图**，最高 **1280×37015**（约 15,800 个视觉 token），241 张里 143 张超过
7,000 token。其余所有子类和评测域 `web` 完全一致，都是 1280×720（880 token）。

**决策：整个剔除 css，而不是只留能塞进去的那些。** 后者是按页面高度做的**有偏子采样**，
悄悄改变了"css"在语料里的含义；而且 §2 刚测出残余偏移由 image 主导，css 的视觉体制
与评测域根本不是一类，塞进去等于让 PCA 把方差花在 `web` 从不访问的方向上——
**正是我们要修的那个失败模式，只是换了个方向**。
`mobile` 保留待定（最高 4,500 token，仍是视口截图）：**抽取是可逆的，拟合才是承诺**。

**崩溃二：101 个未拉取的 git-LFS 指针。**
`PIL.UnidentifiedImageError`，文件是 131 字节的 ASCII 指针而非 PNG。全量扫描：
**评测域 `web` 956/956 全部干净**（P1/P3.0 的结论不受影响），问题只在 `word_docs`
（737 张里 101 张，涉及 5/25 个样本）。网络可达，`git lfs pull --include=...`
补齐（只 fetch 不 push），复查 `agent/gui` 全量 5,755 张现已全部为真 PNG。

最终拟合语料 **156 样本**（webarena_lite 39 / image_edit 30 / word_docs 25 / excel 24 /
mobile 19 / file_mgmt 19），评测集 27 个 web，**一条都不进拟合**。
早先那次跑留下的 `css_01.npz` 已移入 `_excluded/` 并附说明（未删除）。

⚠️ loader 的子类字面值与磁盘目录**不一致**（磁盘 `agent/gui/*`，loader
`agent/vab/*` 与 `agent/arena/*`，且 `webarena-lite` 是连字符）；CLI 评测又是第三套
（`agent/arena/web`，`datasets/worldmemarena.py:55` 的规范化映射）。按目录名写过滤器
会静默拿到空语料。

### 6. 混合 PCA 比例 sweep：工作点定在 25%

新脚本 `wma_pca_mixture_sweep.py`。与旧探针 `wma_pca_refit_probe.py` 的区别是协议：
旧的把同一个目录对半切（同分布内插），新的**拟合语料全是非 web 子类、评测是 27 个 web
且一条没进拟合**（跨任务泛化）。旧协议的数字被 §3.9 引用，故不原地改语义。

| WMA 占比 | 域内 $r_\perp$ | 域内 $R^2$ | **WEB image $r_\perp$** | WEB all | WEB $R^2$ |
|---|---:|---:|---:|---:|---:|
| 0.00（现行） | 0.0867 | 0.9133 | 0.3889 | 0.4199 | 0.4677 |
| **0.25** | 0.0938 | 0.9051 | **0.1264** | 0.1194 | 0.8618 |
| 0.50 | 0.1004 | 0.8949 | 0.1212 | 0.1101 | 0.8813 |
| 0.75 | 0.1125 | 0.8755 | 0.1179 | 0.1040 | 0.8930 |
| 1.00 | **0.3185** | **0.6219** | 0.1102 | 0.0949 | 0.9041 |

**25% 拿到全部可得收益的 94%，只付出全部代价的 2.8%**
（image $r_\perp$ 总空间 0.2787，25% 吃掉 0.2625；域内 $R^2$ 总代价 0.2914，25% 付 0.0082）。
纯 WMA 会让域内崩掉，**不是越多越好**。

子类留一消融（在 0.50 处，固定总行数）：拿掉任一子类，WEB image $r_\perp$ 最多变动
0.003（0.1200–0.1237 对基准 0.1212）。**没有哪个子类承重，也没有哪个有害**；
`mobile` 拿掉后 0.1211 与全留几乎相同 → **六个全留**。

⚠️ 这只证明**表示层面**的 coverage 可恢复，**未证明** retrieval head 随之不再塌缩——
head 必须在新坐标上重训，那是独立问题。

### 7. 顺手修的两个缺陷

- **`frozen_encoder_gate.py:45`** 用死布局 `(32,12,16,4)` **覆盖了 checkpoint 自带的
  布局**，导致该门禁的分组指标切在错误槽位区间上（`all/*` 不受影响）。这是**门禁本身
  的正确性缺陷**，而后续要重度依赖门禁，故先修。改为从 `slot_layout.KEY64_LAYOUT` 取。
- **`residualmem/world_model/slot_reconstruction.py:18`** 的默认值同样作废。修法是
  复用同包 `continuous_bottleneck.DEFAULT_GROUP_SIZES` 而非再抄一份字面值——
  **重复字面值正是当年那个 bug 的根因**（`residualmem/` 是库、`experiments/` 是驱动，
  库不该反向依赖 `experiments/slot_layout.py`，所以不能直接 import 它）。

剩余的 `(32,12,16,4)` 命中全是解释历史 bug 的注释，或**故意**用非默认布局验证 config
能接受任意布局的测试夹具，均正确。

---

## 方向变更（2026-08-22 晚）

读了 *One Token per Multimodal Evidence: Latent Memory for Resource-Constrained QA*
(arXiv:2606.10572) 后，决定把 tokenizer 的压缩层从「无监督 PCA」改为「学习式投影」。
完整计划见 plan 文件，此处只记要点与理由。

**为什么这个方向站得住**：技术报告 §5.2 **本来就规定** tokenizer 是 Perceiver
Resampler / Query Transformer（$x_t=P_\rho(H_t)$）。盘上的「手写 Key64 + PCA」是对论文
自身设计的偏离。§5.4 的护栏（冻结主干、固定 probe 监督、辅助恢复目标、不让 WM 梯度
回流）与该论文的三项损失几乎重合。**这是回到 §5.2，不是背离主张。**

**测到的五个失败，与该工作的设计差异一一对应**：

| 已测事实 | 缺失的东西 |
|---|---|
| 42% 方差出子空间，inverse-PCA $R^2$ 0.4755 vs 0.9184 | PCA 最大化方差而非检索可分性，且基由语料决定 |
| 残余偏移由 image 主导，serializer 碰不到 | 全链**没有任何视觉监督** |
| head **已有 InfoNCE** 却仍塌缩（0.8202 vs 域内 0.3418） | 负样本全在域内；目标空间由外部 8B 教师固定 |
| 池化输入基数 5.5× | 手写池化非计数不变 |
| 指令优先排序在 WMA 从未生效 | 手写规则假定 BrowserGym schema |

**已定范围**：档 2 —— 学习式投影替换 PCA512（含检索侧改造），**不动 Key64 池化**。
**WM 线分叉冻结**在当前 tokenizer，7,100.76 bits/transition（3 seeds）原样保留。

**关键可行性**：token 级 layer-16 H **已在盘上**
（`outputs/state_tokenizer/v9-instruct/full-h/`，300 GB，`tokens-bf16.npy` +
`offsets.npy` + `modality-lengths.npy`），所以学习式模块**不需要重跑 9B 编码器**。

**架构选择**：`z = xbar_pca(x) + f_θ(x)`，`f_θ` 输出层零初始化 →
初始化时 `z ≡ xbar` **逐位相同**，下游全不变，基线在 step 0 精确复现，
任何改善都可归因且**不可能比基线更差**。这条会写成单元测试锁死。

### 评价体系的两处修正（用户提出，已采纳）

**其一，$r_\perp$ 只留给 PCA。** 它依赖固定线性子空间 $S_{PCA}$ 及其正交补；
`z=f(x)` 不对应任何固定 512 维线性子空间，要求它"image $r_\perp$ 优于 mixed PCA"
是把两个不同几何模型塞进同一把尺子。learned projection 改报 **Recon distortion /
Probe retention / Retrieval Recall@K**。

好消息：**probe 设施全是现成的**。记录里每条都带 `probe` 字段（11 个 UI 状态谓词
`STATE_LABELS` + `visible_words`），外加 `task`（12 类各 8,334 条）与 `action`。
`evaluate_probes.py` 的 `make_gate(reference, candidate)` 正是为这个而写；
`binding_probe.py` 专抓「字面还在但不知道属于哪个元素」——学习式投影恰好可能出这种问题。

**其二，$\mathcal L_{vis}$ 先池化再约束**：`v_z = MLP(masked_mean(z[image slots]))`，
$\mathcal L_{vis} = 1-\cos(v_z, v_{teacher})$。逐槽回归同一个全局 embedding 会把 32 个
image slot 拉成同质表示。这也更贴近论文原式 $\|v_i-\text{MLP}(z_i)\|^2$——那里的
$z_i$ 本就是已池化的单 token。

### GATE 改用官方 benchmark 指标（用户提出，已采纳）

cosine 类降为机制诊断——它们只说明"塌缩修没修"，不说明"检索修没修"。
但查框架代码后发现，**用户点名的 Evidence Recall@K 恰好是最不灵敏的那个**：

`qa.py:124-159` 的第 3 条匹配路径从 gold id `mp_S08_1` 解析出 `S08`，此后**任何**来自
S08 的检索行都算命中、不看内容；而 embedding 适配器铸的是合成 id `r00001`，
精确 ID 匹配从不触发。**我们与 Raw-Fused 共用同样的 49 条 `full_round_text` 行且来自
同样的会话**，两臂的 `recall_at` 会被这条路径拉平。

→ 它仍是必报的论文指标，但**主判据改用 `retrieval_coverage.hit_rate`**
（LLM 逐条读检索文本判覆盖，基于内容），另补两个灵敏度切片：只看 `image_id` 类 gold、
去掉共享文本行的 observation-only 消融。

**其它三条量具性质**（详见 plan）：框架的 nDCG 是非标准实现（rank 1 与 rank 2 折扣都
是 1.0），框架内可比但不可与文献并列；judge 失败**静默降级为 Omission**，必须同时记录
`num_valid`；`hit_rate` 看的是全部检索项、不截断到 top_k。

☠️ **一个会造成灾难性假阳性的陷阱**（`cli.py:435-437`）：
```python
if not target.api_key:
    return q.gold_answer          # 静默返回标准答案
```
**`OPENAI_API_KEY` 未设时每个答案都是 gold answer，`correct_ratio` → ~1.0 且无任何
警告。** 每次评测前必须验证管线确实在调模型。

**一个意外收获**：`qa_runner.py:244` 是
`adapter.answer(...) if _has_native_answer(adapter) else answer_fn(q, retrieval)`，
而 `native_answer = self._reader is not None`。**没有 reader 时框架自动走文本问答路径**
——「latent 检索 + 原文生成」这个 arm **今天零改动就能跑**，QA 判据不必等 reader
connector（`answer()` 里的 raise 实际不可达）。这正是该论文仓库 Mistral launcher 的做法，
也正好隔离检索失败与生成失败。

---

## 当前状态

**已完成**：P3.2 serializer 冻结；冻结样式复测与 image 主导的定位；D3 上游复测；
LFS 修复；拟合语料 156 样本抽取；混合比例 sweep（工作点 25%）；两处槽位布局缺陷；
**0c 判别实验（结论推翻了原先的病因排序，见下）**。

---

## 0c：只重训 head —— 病因不在 PCA 基，在训练分布

**做法**：`wma_encode_teacher.py` 补出 156 个非 web 样本的 teacher embedding
（4,379 个 observation），新写 `wma_build_bridge_cache.py` 装配成
`train_retrieval_bridge.load_cache` 认的格式（**按 sample 切分**，31 个留出样本），
然后用**完全不变**的 `train_retrieval_bridge.py` 训一个 head，在 27 个 web 上测。
**web 一条都没进训练。**

装配时抓到一个隐患：`wma_encode_teacher.py` **不做 L2 归一化**（范数 0.9965–1.0039），
而域内的 `build_instruct_bridge_cache.py` 做——训练器的 InfoNCE 与 cosine 两项都假定
单位向量。守卫拦下并归一化了，否则会静默地训歪。

### 结果

| 27 个 web 样本 / 956 obs（按观察数加权） | 域内训练 head（现役） | **WMA 非 web 训练 head** |
|---|---:|---:|
| paired cosine m11 | 0.1937 | **0.5852** |
| 空对照 m00 | 0.1725 | 0.1360 |
| **信号增量 m11−m00** | 0.0211 | **0.4491**（21×） |
| 同样本 R@1（随机 0.0282） | 0.0439（1.6×） | **0.2542（9.0×）** |
| R@5 | 0.2029 | **0.5575** |
| R@10 | 0.3598 | **0.7029** |
| MRR | 0.1427 | **0.3999** |
| R@1 高于随机的样本数 | 13/27 | **27/27** |

**塌缩指标（预注册 gate 主判据，阈值 < 0.60）**：

| head | WMA web 两两余弦 | 域内 validation |
|---|---:|---:|
| 域内训练（现役） | **0.8275** | 0.3418 |
| WMA 非 web 训练 | **0.4052** ✅ | 0.0592 |

### 结论：原先的病因排序错了

**当前的 PCA 坐标并没有把 WMA 信息毁掉。** 同一个 4.7M head、同一组 PCA 基、
同样的 xbar，只把训练分布换成 WMA，检索就从随机水平变成 9 倍于随机，塌缩指标
从 0.83 掉到 0.41。而且这是**跨任务迁移**——训练只见过
excel/file_mgmt/image_edit/mobile/webarena_lite/word_docs。

对照 §6 的 sweep：混合 PCA 把 image $r_\perp$ 从 0.389 压到 0.126，那是真实的表示改善；
但 0c 说明**它不是检索的瓶颈约束**。§3.8 当年从「42% 方差出子空间」推出「基错了」是
对的，从「所以检索失败是因为基错了」则是**过度推断**——两件事都真，但因果链不是那条。

**两条不能忽略的边界**：

1. 这个 head **只**在 WMA 上训过，域内表现未验证（域内两两余弦 0.0592 反而比现役的
   0.3418 更散，很可能已退化）。Phase 1 真正的配置是**并集**训练，且域内不得退化
   （gate ⑤：域内 500-way R@1 ≥ 0.80）。
2. 上表是**自制的同样本 N-way 自检索诊断，不是官方 WorldMemArena 指标**。
   官方 `retrieval_coverage.hit_rate` / `recall_at` / `answer_matching` 仍要靠 0a。

**对计划的影响**：Phase 1（训练分布跨域化）的优先级上升为主路且预期收益已被验证；
Phase 2（学习式投影）从「必要」降为「可能锦上添花」，其价值需要在 Phase 1 的基线上
重新论证——若 Phase 1 已经打平 Raw-Fused，Phase 2 的成本就要重新权衡。

---

## 下一步

1. **0a** —— 产出官方 benchmark 锚点（仓库里**没有任何 checked-in 基线数字**）。
   规模 27 samples / 1,459 questions，全量约 5,050 次 judge + 1,459 次生成调用，
   先试点估成本。跑前做 API-key 前置检查（`cli.py:435` 会静默返回 gold answer）。
2. **Phase 1** —— 并集 cache（域内 5,500 + WMA 非 web 4,379）训练，
   `MultiQueryRetrievalHead` 与现有 head 对照，域内不得退化，走官方指标 gate。
3. Phase 2 的取舍在 Phase 1 结果出来后重定。

---

## 官方评测基础设施（2026-08-22 晚）

官方仓库**只为 embedding 服务提供启动脚本**（`run_qwen_vl_embed_vllm.sh`），
answer/judge 模型预期指向外部 OpenAI 兼容 API。本机 vLLM 不可用
（唯一装了 vllm 0.11.0 的 env 是 torch 2.11，而 0.11.0 钉死 torch 2.8，
扩展加载报 undefined c10::cuda symbol），且**不该改动别人的共享 conda 环境**，
所以新写 `experiments/state_tokenizer/local_openai_server.py` 顶这一格
（标准库 + transformers，不新增依赖）。

**踩过的三个坑，都已修**：

1. **线程加副本无效**：12 个线程副本与 4 个的吞吐完全一样（0.71 req/s）——
   `generate()` 的自回归循环在 Python 层，全卡在 **GIL** 上。改成**多进程 +
   `SO_REUSEPORT`** 后 24 路并发 4.59 req/s，**6.5×**。
2. **keep-alive 废掉了负载均衡**：`SO_REUSEPORT` 均衡的是**连接**不是请求，
   而 OpenAI SDK 持有连接池，一条连接固定绑到某个 worker，导致堆积方 300 秒超时、
   服务端 `BrokenPipeError`。改为每响应后 `Connection: close`，让内核逐请求重新分配，
   24 路 5.32 req/s 且零超时。
3. **停服务只杀了父进程**：多进程模式下子进程存活并各占约 23 GB 显存，
   下次启动 OOM。`setsid` 使父进程成为进程组长，改用 `kill -- -PID` 杀整组。
   入口是 `scripts_local_llm_server.sh`（**用 pidfile，不用 `ps|grep`**——
   后者在本场景必然自匹配，本会话已因此误杀自己的 shell 三次）。

**Qwen3.5-9B 默认带 thinking 前导**，judge 要 JSON 标签时会把预算烧在推理上、
返回不可解析的文本（框架随后静默降级为 `Omission`）。chat template 支持
`enable_thinking=False`；框架本身也对 DeepSeek judge 做同样的事
（注释："saves tokens, faster"），故默认关闭，`-think` 后缀可开回。

⚠️ **本轮评测的重大局限（必须写进任何引用）**：

| | 论文 Table 2 | 本轮 |
|---|---|---|
| 回答模型 | **GPT-5.4-nano**（统一骨干） | Qwen3.5-9B |
| judge | **GPT-5.4-mini** | Qwen3.5-9B ——**与回答模型相同，即自我评判** |
| 范围 | 461 样本 / 24,258 QA，Agentic + Lifelong 聚合 | `agent/gui/web` |
| embedding 服务 | vLLM | SentenceTransformer 本地回退（两臂一致） |

**所以本轮数字不得与论文 Table 2 并列。** 自我评判是我配 `.env` 时图省事引入的
方法论缺陷，有已知的自我偏好偏差。有效的只是**同一次运行内的两臂对比**。
计划是本地迭代、定稿后用 `--eval-only` 从已落盘的 `pipeline_*.jsonl` 换官方模型重判
（不必重跑管线）。

---

## 官方指标下的第一次两臂对比（web_01，n=53）

| | Raw-Fused | ResidualMem（现役 head） | Δ |
|---|---:|---:|---:|
| 记忆 Recall / Corr / Irrel | 0.8743 / 0.9200 / 0.0800 | **完全相同** | 0 |
| RC (hit_rate) | 0.8301 | 0.8364 | +0.006 |
| Recall@1 / @5 / @10 | 0.374 / 0.718 / 0.806 | 0.374 / 0.721 / 0.801 | ~0 |
| nDCG@10 | 0.5715 | 0.5873 | +0.016 |
| QA-C / QA-H / QA-O | 0.7925 / 0.0755 / 0.1321 | 0.7736 / 0.1132 / 0.1132 | 1–2 题 |
| answer tokens/题 | 2,213 | 2,280 | **+67** |

**记忆写入侧逐位相同**——代码层面就注定：两臂的 `memory_delta` 是同一批文本。

`notmention_when_retrieved_ratio` 与 `omission_ratio` **完全相等**（0.1321），
即**每一次 omission 都发生在 gold 已被检索到之后**：剩余误差的三分之二在生成侧。

**按行类型分解**（用官方 `_ranking_metrics`，仅 web_01）：

| | R@10 完整 | R@10 去掉观察行 | R@10 只留观察行 | 观察行占 top-10 |
|---|---:|---:|---:|---:|
| Raw-Fused | 0.8063 | **0.8063** | 0.4158 | 10.2% |
| ResidualMem | 0.8012 | **0.8012** | **0.0000** | **0.0%** |

我们的观察行**从未进入 top-10，贡献精确为零**——塌缩的直接后果。
⚠️ 但**不能**据此断言「这个 benchmark 下观察行都不产生分数」：n=1 样本，
Raw-Fused 那一列同样需要全量才能判断。全量 27 样本正在跑。

---

## Phase 1：并集 head（已完成，两条门槛均通过）

`merge_bridge_caches.py` 把域内 5,500 行与 WMA 非 web 4,379 行合成 9,879 行
（**各自保留原有 train/validation 划分**，使域内 validation 仍是现役 head 当年
测的那 500 条，可比）。用**未改动的** `train_retrieval_bridge.py` 训练。

合并时发现并堵掉一条捷径：**两域的 teacher 范数系统性不同**
（域内 0.9964–1.0039，3,871/5,500 行偏离单位；WMA 精确 1.0）。
InfoNCE 完全可以拿范数当作区分域的旁路信号，与内容无关。已统一归一化。

| head | 域内 R@1 | 域内 cos | 域内两两 | **WMA 两两** | **WMA R@1/随机** | 高于随机 |
|---|---:|---:|---:|---:|---:|---:|
| 域内训练（现役） | 0.8400 | 0.8711 | 0.3418 | **0.8275** | 1.56× | 13/27 |
| WMA-only（0c） | **0.1920** | 0.1008 | 0.0592 | 0.4052 | 9.00× | 27/27 |
| **并集** | **0.8420** | **0.8821** | 0.3829 | **0.4627** | **7.37×** | 26/27 |

- 预注册门槛：域内 500-way R@1 ≥ 0.80 ✅、WMA 塌缩 < 0.60 ✅。
- **域内不但没退化，还略好**（R@1 +0.002，cos +0.011）。
- 并集拿到 WMA-only 约 82% 的检索收益而不牺牲域内；paired cosine 0.6035 甚至
  高于 WMA-only 的 0.5852，说明两域数据互补而非冲突。
- 顺带证实 0c 那个 head 的域内确实废了（R@1 0.1920），**单域训练必然牺牲另一边**。

⚠️ 这些都是**表示层面**的指标。它是否能推动**官方**指标，必须由全量三臂对比回答
（Raw-Fused / ResidualMem-现役 / ResidualMem-并集）。

---

## 待办

1. 全量 27 样本三臂对比，按行类型重做上面的分解（n=1,459 才有资格下结论）。
2. Phase 3 reader connector：把回答从「原文交还」换成「latent 注入」。
   现在 2,213 tok/题 两臂几乎相同，**「低成本」这条腿目前是负的**（我们多 3%）。
3. 定稿后用官方模型（GPT-5.4-nano / GPT-5.4-mini）`--eval-only` 重判。

**未提交**：`local_openai_server.py`、`scripts_local_llm_server.sh`、
`wma_build_bridge_cache.py`、`merge_bridge_caches.py`、本文件的本节。

**未提交**：serializer 冻结相关改动、`wma_serializer_screen.py`、
`wma_pca_mixture_sweep.py`、`wma_build_bridge_cache.py`、两处布局修复、
`test_wma_serializer.py` 的 docstring 更正、`WORLDMEMARENA_TOKENIZER_RAG.md` 的
§3.11/§8/§8.3 改写、本文件。
