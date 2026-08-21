# 状态 Tokenizer + World Model 技术手册（跨 benchmark 移植版）

> 面向要在**别的 benchmark** 上跑这套东西的人。本文假设你不熟悉本仓库，但熟悉
> 深度学习工程。重点在于：**哪些是通用的、哪些是和 MiniWoB 绑死的、接口边界在哪。**
>
> 实验过程与结论见 `WORLD_MODEL_WORKLOG.md`（很长，可只读 §0 和 §24）。
> 本文只讲「怎么训、接口是什么」。

---

## 0. 一句话概括

整条流水线做的是**状态压缩 + 条件预测**两件事：

1. **Tokenizer**：把一个高维观测（图像 + 文本）压成固定形状的离散码
   `y ∈ {0..255}^{64×32}` —— 64 个 latent token，每个 32 个 subspace，每个 subspace
   一个 8-bit 码字。**共 2048 个码字 = 16,448 bit**（含 64 bit 的有效性掩码）。
2. **World Model**：给定 `(历史状态, 动作, 任务)`，预测下一个状态的这 2048 个码字，
   指标是 **held-out 预测码长（bits/transition）**，越低越好。

关键点：tokenizer **训练一次就冻结**，之后所有 WM 实验都只读它的输出。

WorldMemArena web 的跨 benchmark 接入不改变这里的 v8/WM 协议；适配层、
v9 Instruct prompt 与 Raw-Fused 对照见本文 §9，v8 数据重采协议与 2026-08
恢复重建记录见本文 §10，当前状态与关键数字速查见
`WORLDMEMARENA_TOKENIZER_RAG.md`。特别注意 assistant plan/action 是动作
输出，不是当前观察，禁止进入 tokenizer detail slot。
这样 WM 的所有比较都在同一个离散空间里，数字可比。

---

## 1. 目录结构

```
experiments/
  state_tokenizer/     # 第一段：采集 + 特征 + tokenizer 训练
  world_model/         # 第二段：cache 构建 + WM 训练 + 统计
    dev/               # 诊断脚本（不进正式流程，见 §4.6）
residualmem/
  world_model/
    continuous_bottleneck.py   # A1 模型定义
    categorical_bottleneck.py  # A2 模型定义（离散化在这里）
  encoders/normalization.py
configs/world_model/
  v8_discrete.yaml     # 基础配置
  v8_frozen.yaml       # 冻结的最终架构 + 预算
tests/                 # 有回归测试，改代码后请跑
```

---

## 2. 数据契约（**移植时最重要的一节**）

### 2.1 采集记录（record）的字段

采集器每个状态写一行 JSON。下游真正读的字段（见
`experiments/world_model/cache.py::_minimal_records`，70-104 行）：

| 字段 | 类型 | 必需 | 谁读它 |
|---|---|---|---|
| `state_id` | str | **是** | 全局唯一键；`merge_records` 按它排序 |
| `global_index` | int | **是** | 必须等于合并清单里的**行号**，特征库按它索引 |
| `task` | str | **是** | 任务 id（决定 `task_ids`） |
| `episode_id` | str | **是** | 同一 episode 的状态必须同 id |
| `step` | int | **是** | episode 内从 **0** 开始、**连续**、不重复 |
| `split` | str | **是** | `train` / `validation` / `test`，**整个 episode 同一个 split** |
| `action` | dict\|null | **是** | `record[t]["action"]` 是**导致 t+1 的动作**；最后一个状态可为 null |
| `dom` | str | 是* | 状态文本；也是动作 target 语义的唯一来源 |
| `screenshot` | str | 是* | 相对路径，指向 PNG |
| `instruction` | str | 是* | 任务指令文本 |

\* 后三个是**给 tokenizer 用的观测内容**。如果你的 benchmark 观测形态不同
（比如机器人的关节角 + 点云），这三个字段就是你要替换的东西，见 §5。

> **坑 1**：`global_index` 是「合并后按 `state_id` 排序的行号」，不是任意 ID。
> 55G 特征库按它索引，所以**新增数据如果排序插到中间，整个特征库会错位**。
> 若要增量追加，让新 `state_id` 排在所有旧 id 之后（本仓库用 `v9` 前缀，因为
> `'v' > 't'`）。
>
> **坑 2**：`step` 必须从 0 开始且连续。`build_transition_archive` 会检查并抛错。

### 2.2 transition 的构造规则

`cache.py::build_transition_archive`（123-278 行）：

- 只有同一 episode 内**确实存在 `step+1`** 时，当前 step 才产生一个 transition。
  终止动作之后没有后继状态，**不能**仅凭 `action != null` 造 transition。
- 因此有恒等式：**`transitions = states − episodes`**。
  （本仓库实测：100,008 − 43,751 = 56,257，逐个吻合。）
- 历史窗口 `MAX_HISTORY = 7`，**左填充**：不足 7 步时前面补 `-1` 索引，
  动作补 PAD 哨兵（`ACTION_TYPE_IDS["PAD"]`、`TAG_PAD_ID`、`REF_PAD_ID`）。

> **移植提示**：如果你的 episode 很短（本仓库平均只有 2.29 个状态/episode），
> 每个状态只能换来 0.56 个 transition。**transition 数才是训练数据量**，
> 采集时按 transition 而不是按状态估算预算。

### 2.3 cache 格式 —— 真正的接口边界

**如果你能产出这个目录，`train.py` 就能直接跑，完全不需要用我们的采集器。**

`outputs/world_model/v8/cache/`：

| 文件 | 形状 / dtype | 含义 |
|---|---|---|
| `codes.npy` | `uint8[N_states, 64, 32]` | tokenizer 输出的离散码 |
| `valid.npy` | `bool[N_states, 64]` | 每个 latent token 是否有效 |
| `global_indices.npy` | `int64[N_states]` | 必须等于 `arange(N)` |
| `transitions.npz` | 见下 | 17 个数组 |
| `manifest.json` | — | 协议、计数、SHA256 |

`transitions.npz` 的 17 个数组（`T` = transition 数，`H` = 7）：

| 数组 | 形状 / dtype | 含义 |
|---|---|---|
| `history_indices` | `int64[T, H]` | 指向 `codes` 的行；`-1` 表示填充 |
| `target_indices` | `int64[T]` | 被预测状态在 `codes` 中的行 |
| `task_ids` | `uint8[T]` | 任务编号（`sorted(task slugs)` 的位置） |
| `episode_ids` | `int32[T]` | episode 编号（`sorted(episode names)` 的位置） |
| `steps` | `uint8[T]` | episode 内步号 |
| `split_ids` | `uint8[T]` | 0/1/2 = train/validation/test |
| `action_types` | `uint8[T, H]` | 动作类型 id |
| `action_tags` | `uint8[T, H]` | 动作作用元素的类别 id |
| `action_refs` | `uint8[T, H]` | 动作作用元素的编号 |
| `action_payloads` | `uint8[T, H, 40]` | 动作内容的 UTF-8 字节 |
| `action_lengths` | `uint8[T, H]` | payload 实际字节数 |
| `action_policies` | `uint8[T, H]` | 采集策略（scripted/random），仅用于分层统计 |
| `action_targets` | `uint8[T, H, 40]` | 目标元素文本字节（**当前未启用**，见 §7） |
| `action_target_lengths` | `uint8[T, H]` | 同上 |
| `structural_action_bits` | `uint16[T]` | 动作账单：仅结构 |
| `semantic_action_bits` | `uint16[T]` | 动作账单：结构 + target |
| `full_action_bits` | `uint16[T]` | 动作账单：结构 + target + payload |

`manifest.json` 必需键：`protocol`、`format_version`、`layout`、`state_counts`、
`transition_counts`、`episodes`、`episode_names`、`tasks`、`policy_names`、
`artifact_sha256`、`source_sha256`。`FrozenCache.__init__` 会校验形状与协议。

> **计数门禁**：`cache.py:53-54` 硬编码了本项目的 states/transitions 期望值。
> 换数据集时用 `--no-v8-count-gate`，或加一组你自己的常量。

---

## 3. 第一段：状态 Tokenizer

### 3.1 特征抽取

用 **Qwen3.5-9B** 的第 16 层 hidden states 作为观测编码器（不微调，纯前向）。
输入是「截图 + 状态文本 + 指令」拼成的多模态 prompt。

- `extract_qwen.py`：跑主 prompt，产出每个状态的 hidden states + modality 长度。
  **必须加 `--no-use-kernels`**（本环境没有 `kernels` 包，transformers 5.8.1 会拒绝启动）。
- `extract_fixed_prompt.py`：跑一个**固定的、与任务无关的**观测 prompt。
  「fixed prompt」的意义是给出一个跨状态恒定的参照带，用来检测哪些槽位其实
  不携带状态信息 —— 实测它的跨状态余弦相似度中位数 0.214，而 detail 带是 0.002，
  差两个数量级，所以**后来 prompt 槽位被回收给了 detail**（见下）。

### 3.2 Key64 槽位结构

`experiments/state_tokenizer/slot_layout.py`：

```python
IMAGE_SLOTS   = 32
DETAIL_SLOTS  = 16   # 原本 12，把 prompt 的 4 个回收过来了
CONTEXT_SLOTS = 16
PROMPT_SLOTS  = 0    # 已废弃：几乎不携带每状态信息
KEY64_LAYOUT  = (32, 16, 16, 0)   # 合计 64
GROUP_NAMES   = ("image", "detail", "context", "prompt")
```

`rebuild_static_key64.py` 把 Qwen 的变长 hidden states **自适应平均池化**成这 64 个槽位
（`common.py::adaptive_average_pool`，保序、短输入时相邻槽位重叠）。

> **坑 3**：`--image-grid-thw 1 20 32` 是**必填**的。默认值 `1 20 14` 是旧版
> 160×210 截图的网格，用错了会**静默地**错误池化那 32 个 image 槽位，不报错。
> 这个参数必须和你截图的实际分辨率对应。

> **注意**：`common.py::SLOT_LAYOUTS = {64: (40,20,4)}` 是**更早版本**的布局，
> 与 `KEY64_LAYOUT` 不是一回事，别用混。

### 3.3 归一化与 PCA

- `key64_pca.py fit` → 512 维 PCA（**只在 train split 上拟合**）
- `key64_pca.py transform` → 应用
- `fit_normalization.py` → 每 group × channel 的 mean/std（**同样只在 train 上**）

> **移植时**：这两个都必须只在你的 train split 上拟合，否则评测集信息泄漏。
> 如果是在已有数据上追加，**必须复用**旧的 PCA/normalizer，否则旧状态的码字会变，
> 历史数字全部作废。

### 3.4 A1：连续瓶颈

`experiments/state_tokenizer/a1_continuous_bottleneck.py`

做的事：`xbar_t → e_t (N × d_e) → xbar_hat_t`，一个**确定性**的连续瓶颈。
**没有** categorical latent、prior、KL、transition、action、时序 —— 每一行状态独立。
目标函数只有一个：**带 mask 的重建 MSE**。

关键参数（默认值）：

```
--num-e-tokens 16      # v8 实际用 64
--e-dim 512
--num-heads 8
--ffn-hidden 1024
--stage-steps 2000 3000 5000        # 分阶段学习率
--batch-size 32
--weight-decay 1e-4  --clip-norm 10.0
```

产物 `outputs/a1/v8.npz`（encoder/decoder 权重）。有一组质量门禁
（`--total-r2-gate 0.999` 等），不过就会写失败报告并退出。

### 3.5 A2：离散瓶颈（真正产生码字的地方）

`experiments/state_tokenizer/a2_categorical_bottleneck.py`

**A1 送连续 `r_t`，A2 把它离散化，其余一律不变**：encoder/decoder 直接继承 A1 的
checkpoint。方案是 **PQ-8（product quantization）+ 直通估计器**：

```
r (N, d_e)  --split-->  (N, M, d_sub)     M = num_subspaces
每个 subspace 独立选一个 category  ->  codes ∈ {0..255}^(N, M)
前向：hard gather（比特精确）
反向：straight-through, st = probs + sg(hard - probs)
```

见 `residualmem/world_model/categorical_bottleneck.py::quantize`（329 行）。
注意它的前向是**精确的 gather**，不是 soft mixture —— soft 路径只做诊断用，
因为一个完整的概率单纯形会构成远超 `log2(C)` bit 的连续侧信道。

**刻意没有的东西**（原文如此）：prior、KL、commitment loss、codebook loss、
usage balancing、entropy 正则、Gumbel 噪声、任务损失、transition、action、history。
**唯一目标就是带 mask 的重建 MSE。**

关键参数：

```
--num-e-tokens 64  --e-dim 512
--num-subspaces 32       # v8 用 32（产物名 v8-m32-gw 的 m32）
--num-categories 256
--kmeans-iterations 25   # codebook 用 train split 上的 per-subspace K-means 初始化
--calibrate-temperature  # 默认开；把 softmax 温度标到 max-prob ≈ 0.8
--max-steps 70000  --batch-size 32
--codebook-learning-rate 3e-4
--backbone-learning-rate 3e-5      # backbone 学得慢十倍
```

> **为什么这里不做小样本 overfit 检查**：4096 bit 的码远大于索引 2000 个训练状态所需的
> 11 bit，训练集重建可以靠「记住这是第几个状态」解决。容量结论只能看 held-out。

### 3.6 公开接口

```python
# residualmem/world_model/categorical_bottleneck.py
initialize_params(config, seed=0) -> dict
quantize(params, r, config, *, soft=False, return_diagnostics=False)
    # -> (e_tilde, codes)   codes 形状 (..., N, M)
embed_codes(params, codes, config)
reconstruct(params, xbar, valid_mask, config, *, return_codes=False)
calibrate_temperature(...)
code_histogram(codes, config) / code_health(histogram, config)

# residualmem/world_model/continuous_bottleneck.py
encode(...) / decode(...) / reconstruct(...)
masked_mse(prediction, target, valid_mask)
masked_group_mse(...)   # 按 image/detail/context 分组看
```

`cache.py::build_cache`（349-372 行）就是把这些串起来产出码字的地方：

```python
x_t, valid = store.load(indices)                 # 特征库
xbar = normalizer.normalize(x_t, valid)          # 冻结的 normalizer
_, batch_codes = Q.reconstruct(params, ..., config, return_codes=True)
codes[start:stop] = batch_codes.astype(np.uint8)
```

---

## 4. 第二段：World Model

### 4.1 预测对象与计费口径（**先读这个，否则数字会看错**）

- 预测 `p(y_{t+1} | y_{t-6..t}, u_{t-6..t}, T)`，**2048 个码字全部计费**，
  即使某些 latent token 无效也照收 —— 这是「固定宽度」的口径。
- 指标是 **held-out 预测码长**，不是两部分 MDL：
  **模型参数一律不计费**（神经网络权重不计，统计基线的计数表同样不计）。
  所以神经模型和统计基线口径一致，可以直接比。**写报告时必须显式声明这一点。**
- 需要传输的侧信息才计费：动作 59.39 bit + 任务 1.57 bit = **60.97 bit/transition**。
- **history 不计费** —— 过去的状态在解码端本来就有。这条不对称很重要：
  只要 history 不是负收益就该用。

### 4.2 架构

`experiments/world_model/model.py`（528 行，纯 JAX）

```
输入序列 = 7 个时刻 × 64 个 latent token = 448 个位置
注意力  = block-causal：同一时刻内双向，未来时刻不可见
每个位置的输入 = 码字嵌入 + 有效性摘要 + 动作嵌入 + 任务 + 时间 + 槽位   （相加，d_model=256）
```

- **码字嵌入** `code_embedding[64, 32, 256, 8]`：每个 (token, subspace, category)
  一个 8 维向量，32 × 8 = 256 = `d_model`。
  **硬约束：`d_model == num_subspaces × code_embedding_dim`**（`model.py:63` 会检查）。
  这也是为什么 `d_model` 从来没被单独扫过 —— 动它会同时动嵌入维度。
- **动作编码**：结构字段（type/tag/ref）各 32 维嵌入 + payload 字节走一个 **byte-GRU**
  （≤40 字节，字节嵌入 16 维，隐层 64 维），拼接后线性投影到 `d_model`，
  再广播到该时刻的 64 个 token 上。
- **backbone**：标准 pre-LN Transformer。冻结配置为 **12 层 / d=256 / 8 头 /
  MLP 4096 / dropout 0.3**，33,244,832 参数。
- **两个输出头**：
  - mask 头：最后时刻均值池化 → 64 个 logit（哪些 slot 有效）
  - code 头：**权重共享（tied）** —— 最后时刻的隐状态切成 32 份 8 维，
    直接与 `code_embedding` 点积（`model.py:449`），没有独立的输出矩阵。

### 4.3 copy gate（**最关键的一个组件，别删**）

```
p(c') = π · 1[c'=c] + (1 − π) · softmax(logits)
```

`π` 是每个 (token, subspace) 由隐状态算出的 sigmoid。对数域用 `logaddexp` 混合，
恒等分支用 **−1e30 而不是 −inf**（保住梯度通路）。

**为什么必须有**：tied 的输出头把 logits 分解成 `h · e_{c'}`，因此对任意上下文
**code logits 矩阵的秩 ≤ 8**，而「保持当前码字」需要的是满秩的单位核。
实测：纯 rank-8 要付 451.1 bit，加了 gate 只剩 **63.3 bit** —— gate 吸收了 86% 的秩损失。
端到端训练测得它值 **+434.76 bit**，是本项目最重要的单个结构改动。

代价只有 18,432 个参数（`64×32×8 + 64×32`）。

### 4.4 训练

```bash
JX=<venv>/bin/python
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false

# 冒烟：128 条 transition 的记忆测试（会检查 NLL 下降 ≥ 90%）
$JX -m experiments.world_model.train \
  --cache outputs/world_model/v8/cache \
  --config configs/world_model/v8_frozen.yaml \
  --variant full --seed 0 --overfit-transitions 128 \
  --output <out>/overfit128

# 正式训练
$JX -m experiments.world_model.train \
  --cache outputs/world_model/v8/cache \
  --config configs/world_model/v8_frozen.yaml \
  --variant full --seed 0 --device-index 0 \
  --baseline-json outputs/world_model/v8/baselines/validation/baseline.json \
  --output <out>/full_seed0
```

> **坑 4**：overfit 模式下 `train=not overfit`，即 **dropout 和 weight decay 自动关闭**
> （`train.py:163,178`）。如果这个门禁没过，**先怀疑步数不够**，不要怀疑 dropout ——
> cosine 学习率按 `max_steps` 退火，把 `max_steps` 改小会让学习率提前退到 0。
> 本仓库实测：3,000 步只掉 38%，20,000 步掉到 **100%**（14367 → 0.27）。

产物：`best.pkl` / `last.pkl` / `metrics.jsonl` / `per_transition.npz` /
`per_episode.jsonl` / `run.json` / `resolved.yaml`。

### 4.5 三个统计基线

`experiments/world_model/baselines.py`，全部在 train 上拟合、在 held-out 上评测：

| 基线 | 含义 | v8 validation（code bits） |
|---|---|---:|
| task marginal | 只知任务 | 10,628.26 |
| copy-aware | 知当前状态，编码「保持/改变」 | 9,562.38 |
| **source-conditioned Markov** | 按当前码字查 train 统计表 | **8,988.52** |

第三个是真正的对手。**新 benchmark 上第一件事就是把这三个算出来** ——
它们很便宜（CPU 几十秒），而且直接告诉你任务有多少可压缩性。

### 4.6 `dev/` 目录是什么

诊断脚本，**不进正式流程**。它们在 train 内部再切 80/20（`d_residual.py::episode_split`），
所有架构搜索都在这个 train-dev 上做，从不看 validation。
`train.py` 有 `--allow-dev-variant` 开关和 `dev_run` 标记，`statistics.py` 会拒绝
带该标记的产物混进正式统计。**移植时建议保留这个围栏习惯。**

---

## 5. 与 benchmark 的耦合点

### 5.1 动作 schema —— 耦合最深，**必须重写**

`experiments/world_model/schema.py` 假设动作是**网页 DOM 操作**：

```python
ACTION_TYPE_IDS = {"CLICK":0, "FILL":1, "SELECT_OPTION":2, "PAD":3, "MASK":4, "UNK":5}
TAG_IDS = {"button","input_checkbox","input_radio","input_text","label","option","select","UNK"}
REF: 0..63          # AXTree 元素编号，硬上界 64
payload: ≤40 字节 UTF-8
```

还有一堆 AXTree 文本解析（`_ROW_RE`、`parse_dom_nodes`、`extract_target_text`）和
`SELECT_OPTION` 的父节点校正逻辑 —— 这些对非网页环境**完全没有意义**。

**换成别的 benchmark 要做什么：**

模型侧真正需要的只有四样东西（见 `model.py::_action_embedding`，244 行）：

| 抽象 | 当前实现 | 你的 benchmark 可以是 |
|---|---|---|
| **类型** 小基数离散 | CLICK/FILL/SELECT | 技能 id / 按键 id / 动作模式 |
| **类别** 小基数离散 | 元素 tag | 物体类别 / 关节组 |
| **引用** 中基数离散 | 元素 ref 0..63 | 物体实例 id / 目标格子 |
| **载荷** 变长字节 | 输入文本 UTF-8 | 任意可序列化内容 |

- 如果你的动作是**连续向量**（如机器人关节增量），最省事的做法是把 payload 通道
  换成一个小 MLP 直接吃向量，其余三路填常数；或者离散化后塞进 type/tag/ref。
  改动点只在 `_action_embedding`，**backbone 完全不用动**。
- 如果动作是**纯离散按键**，只用 type 一路即可，其余填 PAD。

**计费同步改**：`schema.py::action_side_information_bits`（256 行）现在写死
`structural = 2 + 3 + 6`（类型 2 bit + tag 3 bit + ref 6 bit），payload 是
`6 + 8 × 字节数`。你换了动作空间，**这个序列化预算必须跟着重算**，否则
「总 bits = 观测 + 账单」的比较就不成立。配置里的
`structural_bill_bits / payload_length_bill_bits / task_bill_bits_per_episode` 也要改。

### 5.2 其余耦合点

| 位置 | 耦合内容 | 处理 |
|---|---|---|
| `common.py::PILOT_TASKS` | 12 个 MiniWoB 任务名 | 换成你的任务列表 |
| `collect_browsergym.py` | Playwright/BrowserGym 采集 | 整个换掉，只要产出 §2.1 的字段 |
| `cache.py:53-54` | states/transitions 计数门禁 | 换成你的数值或用 `--no-v8-count-gate` |
| `split_for_episode` | `episode_index % 10` → 6/2/2 | 按需改；**整个 episode 必须同 split** |
| `slot_layout.py` | 32/16/16/0 的模态划分 | 按你的观测模态重新分配（合计仍需 64，或同步改 `NUM_LATENT_TOKENS`） |

### 5.3 完全通用的部分（不用动）

- `model.py` —— 除了 `_action_embedding` 那一小段，backbone / copy gate /
  `codelength_bits` 对 benchmark 无任何假设
- `train.py` / `config.py` —— 给定合法 cache 即可跑
- `residualmem/world_model/{continuous,categorical}_bottleneck.py` —— 纯张量运算
- `baselines.py` —— 只依赖码字和 task id
- `statistics.py` 的 `paired_episode_bootstrap` —— 通用的按 episode 配对自助法

---

## 6. 移植的最小路径

1. **先算三个统计基线**。只需要 `codes.npy` + `transitions.npz`。
   如果 source-conditioned Markov 已经非常接近固定宽度，说明你的状态几乎不可压，
   先别急着训 WM。
2. **确定动作抽象**，按 §5.1 的四元组映射，同时把计费公式改对。
3. **产出 cache 格式**（§2.3）。这是接口边界，产出了就能跑 `train.py`。
4. **跑 overfit 门禁**（`--overfit-transitions 128`，20k 步）。
   不过就是实现有问题，别往下走。
5. **在 train 内部切 80/20 做所有架构搜索**，validation 留到最后一次性用。
6. 最后 3 个种子 + 固定预算 + 事先指定 `last.pkl`，一次性评测。

---

## 7. 已经踩过的坑（**强烈建议读**，能省很多时间）

| 结论 | 数值 | 说明 |
|---|---|---|
| **噪声底约 110 bit** | 同配置三次跑出 7431.95 / 7470.89 / 7542.08 | step-0 评测逐位相同 ⇒ 差异全来自训练期 GPU 非确定性。**不可靠换估计量消除**，只能多种子取均值。**任何单种子、小于 ~110 bit 的差异都不能当结论。** |
| **容量与正则化会混淆** | 15M 以上在 dropout 0.1 下全部落进噪声带 | 一度据此写下「容量封顶」，是**错的**：开 dropout 0.3 后 33M 的格子反超 208 bit。扫容量**必须同时扫正则化**。 |
| copy gate | **+434.76** | 见 §4.3，必须有 |
| payload | **+526.93** | 迄今最大单一信息源 |
| history（7 步） | 仅 +40.16（0.7σ） | 状态**完全可观测**，过程接近 Markov，所以历史几乎没用。保留只因为它不计费。 |
| source prior + 神经 residual | **−385.24（有害）** | 短预算下领先，跑满预算后反超并退化。**别用短预算排架构。** |
| 加宽 `code_embedding_dim` | 最多 +28.1 | 有 copy gate 之后输出头不是瓶颈 |
| target_text 通道 | 观测无收益，账单 +60.81 | 代码还在（`use_target_channel`）但**默认关闭**，且 `train.py` 不传该字段会崩 |
| 数据量 | 右端边际 **+297 bit**，是 source 对照（+74）的 4.02× | 数据是真正的瓶颈；容量不是 |
| **跨 run 比最小值时评测点数必须相同** | — | 否则「在平坦盆地里多抽几次最小值」会产生向下偏差。曾因此在 30% 进度上误判正则化无效。 |

---

## 8. 环境

三个互不兼容的 venv（**别混用**）：

| 阶段 | 解释器 | 关键依赖 |
|---|---|---|
| 采集 | `browsergym-venv` | playwright / gymnasium / browsergym；**刻意不装 torch、jax** |
| 特征抽取 | Mem_compiler 环境 | torch / transformers |
| tokenizer + WM | `.venv-jax`（原名 residualmem-jax-venv） | jax 0.4.33 / optax；**Blackwell sm_120 需 ptxas ≥ 12.8.93** |

```bash
JX=<repo>/../residualmem-jax-venv/bin/python
export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false
$JX tests/run_tests.py tests/world_model/test_*.py     # 改代码后请跑
```

已知失败：`test_fp64_evaluation_is_batch_partition_invariant`（jax 0.10.2 下的
既有问题，与业务代码无关）。其余应全过。

> **环境警告**：本仓库的采集环境是在 **Ubuntu 24.04 / glibc 2.39** 上冻结的
> （`browsergym-venv/syslibs` 里有 84 个 .so）。在 22.04 / glibc 2.35 上**跑不起来**，
> 而且失败方式很隐蔽：即使勉强启动，Chromium 的字体渲染会变，截图有 5–16% 的像素差异。
> 由于 64 个槽位里有 32 个是 image 槽位，这会让新采数据和旧数据**分布不一致**。
> **换机器采集前，务必先用相同 seed 重采几条旧 episode，逐字节比对
> `dom` / `axtree_raw` / 截图像素。**

---

## 9. WorldMemArena RAG 适配协议（v9 Instruct）

> 原独立文件 `QWEN35_INSTRUCT_WORLDMEMARENA.md`，2026-08-22 并入。
> 本节固定 ResidualMem v9 在 WorldMemArena web 子集上的输入、训练与评测协议。
> 核心问题：把当前观察压缩为 `xbar_t` / A2 后，能否在相同 RAG 检索配置下接近
> 未压缩的 fused observation。当前状态与关键数字速查见
> `WORLDMEMARENA_TOKENIZER_RAG.md`。

### 9.1 三类不可混淆的内容

| 内容 | tokenizer / fused observation | 完整 round 文本行 |
|---|---:|---|
| 当前 screenshot | 是 | attachment 元数据 |
| user 原始文本 | 是 | 是 |
| image caption | 是 | 是 |
| assistant 的观察/计划（`I can see...`） | **否** | 是 |
| assistant Action JSON | **否** | 是 |

assistant 是基于当前观察产生的 policy/action 输出，不是 `x_t` 的输入。数据 loader
会把 caption 再内联到 user text；适配器必须先移除这份副本，保证 caption 在
observation 中只出现一次。

WorldMemArena 没有 BrowserGym AXTree。零训练适配将 user 文本与 caption 序列化为
合法的 synthetic AXTree；截图仍走视觉 token。该映射仅复用 tokenizer 既有输入槽，
不使用 benchmark 标签。

### 9.2 严格对照

**Raw-Fused**（`Qwen3-VL-Embedding-8B-FusedObs-RAG`）：每个有观察的 round 产生
两条 document row —— (1) 完整 round 文本（user + assistant）；(2) screenshot +
genuine user text + caption 在**同一条** Qwen3-VL document 请求中的 fused vector。
空观察 round 只产生第 1 条。fused 编码失败必须报错，禁止静默降级为 caption-only。

**ResidualMem** 四个主实验：

- `ResidualMem-Instruct-Xbar-Input-RAG`
- `ResidualMem-Instruct-A2-Input-RAG`
- `ResidualMem-Instruct-Xbar-L16-RAG`
- `ResidualMem-Instruct-A2-L16-RAG`

保持相同的完整 round 文本行；第 2 条由共享 retrieval head 将 `xbar_t`（或 A2
重建的 `xbar_hat_t`）映射到归一化 4096 维向量。

所有方法使用官方 Qwen3-VL query encoder、row-level cosine、`top_k=10`、不做
round dedup（同一 round 两行可能同时命中）。官方 `Qwen3-VL-Embedding-8B` 的
完整文本向量 + 独立 image 向量模式保留为论文复现参考，**不是压缩质量的控制组**。

### 9.3 v9 tokenizer 与产物

v9 = v8 Base，仅 prompt wrapper 换成 `Qwen3.5-9B-Instruct` 原生 chat template；
DOM/instruction marker、Static Key64 `(32,16,16,0)`、PCA512、train-only
group/channel normalization、A1/A2 语义全部不变。

```text
outputs/state_tokenizer/v9-instruct-pca20k-balanced/key64-static-pca.npz
outputs/state_tokenizer/v9-instruct-pca20k-balanced/key64-static-pca-normalization.npz
outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/train-cache-fused-observation.npz
outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/retrieval-head-fused-observation.pt
```

2026-08-21 用户决定：本轮只评估连续 `xbar`，不重跑 A1/A2。旧 A2 checkpoint 绑定
纠错前的旧 PCA 坐标（见 §10.3），不能与本节 20k-balanced PCA 混用——既不进入
cache，也不进入 retrieval loss。

### 9.4 A1 / A2 正式重训配置

train split 优化、validation 选 best checkpoint、全程不读 test。A1 固定 64 个
512 维连续 token、1.0× 标量宽度、10,000 步固定预算：

```text
stage steps       = 2,000 / 3,000 / 5,000
learning rates    = 1e-3 / 3e-4 / 1e-4
batch             = 32
eval every        = 250
matmul precision  = highest
patience          = 0（固定预算）
```

A2 从本轮 A1 checkpoint 初始化，冻结 v8 主配置：PQ、64 tokens、M=32、C=256、
`group_weights=(1,2,0.5,0.5)`、K-means 25 iterations、70,000 steps；codebook LR
`3e-4`、backbone LR `3e-5`、batch 32、每 500 steps validation、patience 0。

K-means **不抽样**：70,018 个 train state 全部参与 2,048 个 `(token, subspace)`
问题。latent 以 FP32 存 host memory，GPU 每次只驻留 64 个问题（避免一次性物化
约 9.18 GB 及转置副本导致 OOM，见 §10.3）。随机数协议
`per_problem_fold_in_v1`（seed 0 + 全局 problem id）：调整 resident problem batch
只改显存峰值，不改每个问题的初始化与最终中心。

两阶段均 FP32 activation + `highest` matmul precision + FP64 host metric 累加。
统一入口：

```bash
./scripts_v9_instruct.sh a1
./scripts_v9_instruct.sh a2
```

正式产物：`outputs/a1/v9-instruct{,.npz}`、`outputs/a2/v9-instruct-m32-gw{,.npz}`。

### 9.5 Retrieval head 训练

训练数据只来自 BrowserGym/MiniWoB++：默认 5,000 train + 500 validation，按 task
平衡抽样；WorldMemArena 记录、QA 和答案均不参与训练。

Teacher 是同一 BrowserGym observation 的官方 Qwen3-VL document embedding：

- image：BrowserGym screenshot；
- text：真实 task instruction + AXTree DOM；
- image/text 必须位于同一个 document 请求；
- 输出为归一化 4096 维 `teacher_fused_embedding`。

Student 是 `MaskedAttentionRetrievalHead`，只读连续 `xbar`：

```text
LayerNorm(512)
  -> masked scalar attention over 64 slots
  -> weighted pooling
  -> Linear(512,1024) + GELU + Linear(1024,4096)
  -> L2 normalize
```

Loss 为 symmetric InfoNCE + cosine。代码向后兼容 `representation=both` 的
Xbar/A2 共享 head（该模式另有 `0.1 × consistency`），本轮产物 metadata 为
`representation=xbar`，不含 `a2_xbar`。

缓存协议锁（旧 screenshot-only `teacher_embedding` cache 会被拒绝）：

```text
qwen35_instruct_bridge_cache_v2
qwen3_vl_fused_observation_v1
browsergym_task_plus_dom_v1
```

```bash
python -m experiments.state_tokenizer.build_instruct_bridge_cache \
  --records outputs/state_tokenizer/v8/full-721.jsonl \
  --features outputs/state_tokenizer/v9-instruct-pca20k-balanced/static_features \
  --normalization outputs/state_tokenizer/v9-instruct-pca20k-balanced/key64-static-pca-normalization.npz \
  --representation xbar \
  --teacher-backend sentence_transformers \
  --teacher-model-path models/Qwen3-VL-Embedding-8B \
  --teacher-num-gpus 8 \
  --output outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/train-cache-fused-observation.npz

python -m experiments.state_tokenizer.train_retrieval_bridge \
  --cache outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/train-cache-fused-observation.npz \
  --representation xbar \
  --output outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/retrieval-head-fused-observation.pt
```

### 9.6 2026-08-21 正式 Xbar 结果（MiniWoB 域内）

- PCA：20,000 train states，12 task 各 1,666–1,667；解释方差 `0.9185917377`，
  SHA256 `f98c517c...f6efb2a`。
- cache：5,000 train + 500 validation，只有 `xbar/valid/teacher_fused_embedding`；
  全部 finite、invalid slots 精确为零、5,500 索引唯一。
- retrieval head：best step 5,000，validation loss `0.6449992`、paired cosine
  `0.8711105`；500-way Recall@1/5/10 = `0.840/0.956/0.986`，MRR `0.8946`。

⚠️ 这是 **MiniWoB/BrowserGym 域内** validation，不能替代 WMA 跨域检验（§9.8）。

### 9.7 Reader connector

- **Input connector**：`LN(512) -> Linear(4096) -> GELU -> Linear(4096) -> RMSNorm`，
  另有零初始化 rank embedding；把每个 latent slot 作为 Qwen 输入 soft token。
- **Layer-16 connector**：先严格逆 normalization/PCA 恢复到 4096 维 layer-16
  状态，再叠加零初始化的 `Linear(512,4096)` residual adapter；推理时在第 17 层
  入口替换 prefix states。

Reader 必须用 Qwen3.5-Instruct 原生生成答案。若 connector 未加载，适配器不得
假装运行 native reader；当前实现会明确报错或走未启用 native-answer 的外部
answer stage。

### 9.8 分布偏移与判据（当前主要风险）

训练域是 MiniWoB++ 的规则化小网页（498×321），评测域是 1280×720 的真实 Chrome
轨迹，caption 更长、视觉熵和页面复杂度更高。**这是实验的主要风险，不是实现
细节。**必须同时报告：

- Raw-Fused vs Xbar：连续 tokenizer 的跨域损失；
- Xbar vs A2：离散化额外损失；
- retrieval RC/Recall 与最终 QA 指标；
- WMA web 内按「有截图/无截图、caption 长度、session 长度」分桶结果。

成功判据不是绝对分数复现论文全表，而是在完全相同的两行建库与 top-10 协议下
Residual 与 Raw-Fused 接近；若 Raw-Fused 自身偏低，应先排查 Qwen server/数据
协议，不能归因给 tokenizer。

**web_01 最终 checkpoint smoke**（25 sessions / 49 full-round rows / 14
observations / 10 questions）：

| 指标 | Raw-Fused | Xbar |
|---|---:|---:|
| evidence Recall@10 | 0.765 | 0.725 |
| NDCG@10 | 0.5429 | 0.5529 |
| top-10 observation rows（100 槽位合计） | 8 | **0** |

总体 top-10 overlap `0.91`、top-1 agreement `1.00` 是两侧共有 49 条完全相同
full-round text row 的**掩盖效应**。只看 14 条 observation：paired cosine
`0.1836`，14-way Recall@1/5/10 = `0.0714/0.5714/0.8571`（R@1 ≈ 随机 1/14），
MRR `0.2958`。首个 checkpoint（5 sessions / 2 obs）同样 observation row 从未
命中（paired cosine 0.2174）。

**结论：当前零训练 synthetic-AXTree 适配未通过跨域压缩门禁**；不得用总体
top-10 overlap 宣称压缩成功，修正域适配前不得扩到全量 web benchmark。逐问证据：

```text
outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/
  worldmemarena-web01-checkpoint0-10q.json
  worldmemarena-web01-final-checkpoint-10q.json
```

---

## 10. v8 数据重采协议与 2026-08 恢复重建记录

> 原独立文件 `RECOVERY_WORKLOG_20260820.md`，2026-08-22 并入精简。
> 只保留可复用的工程协议、门禁数字与 bugfix；8-20 的恢复步骤（权重下载、
> 源码找回）已全部完成，过程叙述不再保留。

背景：两日前备份只有源码，`outputs/` 全部丢失，100,008 states 无法逆向恢复，
必须重采且与旧数据逐字节等价。环境固化：BrowserGym 0.14.3 + Playwright 1.44
Chromium 固化在仓库目录；MiniWoB++ 检出官方固定 commit `7fd85d71`。

### 10.1 lane 并行采集协议（不改变数据内容的加速）

原 12-worker 协议：task `t` 的 episode 序列 `episode_index = t + 12k`，env seed
= `seed + t×1,000,000 + episode_index`。实测吞吐仅 ~75 states/min（BrowserGym
0.14.3 每个 episode 冷启动 task 与 chat 两个 Chromium），外推 2–3 天。

- **lane 划分**：只按 `k mod L` 分 lane，`lane l: episode_index = t + 12l + n·(12L)`。
  任一 episode 的 id、seed、动作 RNG、split、页面内容全部不变。
- **assembler 硬校验**（`assemble_browsergym_lanes.py`）：episode lattice 无缺口、
  step 从 0 连续、seed/split/state_id 全匹配、无重复/半截 episode；与旧数据的
  重叠记录与截图**逐 SHA256 比对**；最后按冻结顺序精确保留每任务最早 8,334
  states。
- **并发教训**：228 lanes 一次性全启 → 全部阻塞在首次 `env.reset()`，系统
  load ≈ 250、大量 SwiftShader GPU-process（Chromium 软件渲染子进程，与 CUDA
  无关）。正确做法是**有上限的分批调度**（64 active lanes），lattice 与最终
  数据完全不变，按实测吞吐调档而非按 CPU 核数盲拉。
- **`pre_observation_delay` 不可降为 0**：0.5→0 秒在 12 task / 37 records 中
  产生 8 条 AXTree 语义 sidecar 差异 + 5 张截图差异，门禁失败。生产保持 0.5 秒。
- **browser reuse**（opt-in）：保留 0.5s 观察时序，每个 episode 仍新建 incognito
  context，只复用两个 Chromium 进程。门禁 114/114 截图 SHA256 一致，tokenizer/WM
  输入字段全等；稳态吞吐 ~4.2k–4.4k states/min（消除冷启动后的数量级提升），
  0 retry、0 episode failure。
- **AXTree nodeId 波动**：少量 `axtree_raw` 仅在 CDP `nodeId/parentId/childIds`
  数值上不同；完全不复用 browser 的对照同样出现 → 是 Chrome 跨进程固有的内部
  编号波动，不是 reuse 改变状态。assembler 只对这三类 id 做按节点顺序的规范化
  比较，拓扑与 role/name/value/browsergym_id 仍须完全一致；有回归测试保证语义
  变化会被拒绝。

### 10.2 canonical 装配与审计（2026-08-21 完成）

- 228/228 deterministic lanes 全部完成：128,364 候选 states / 53,112 完整
  episodes，collector/retry/failure 均为 0；精确裁剪到 100,008（12 task × 8,334）。
- 与两日前 partial reference 的 **6,792 重叠 states 全量审计**：6,792/6,792
  规范化 nodeId 后语义完全相同；6,770 截图逐字节相同；剩余 22 张差异仅
  scrollbar/resize-handle raster（最坏 112/159,858 像素 ≈ 0.07%，全图 uint8
  MAE ≤ 0.023），fresh-browser 对照同样存在。assembly 门禁因此显式放宽为
  changed fraction `0.001` / MAE `0.03`；**代码默认仍是 0/0 严格模式，不得
  静默放宽**。
- 产物结构：`outputs/state_tokenizer/v8-recovered` 为实体；canonical
  `outputs/state_tokenizer/v8` 是指向它的**符号链接**（防脚本误读旧数据）；
  partial reference 保留在 `v8-reference-partial-20260821` 供审计。
- `records-merged.jsonl` / `full-721.jsonl` / `extend-100008.jsonl` 均 100,008
  条，state id 唯一、index 连续、截图缺失 0；7/2/1 split = train 70,018 /
  validation 20,011 / test 9,979，episode 跨 split 0、old-train leakage 0；
  merged manifest SHA256
  `c61ee90c87684ec3873174849300a7443351d71b71301ff4718fec7936e45865`。

### 10.3 v9 Instruct 重建要点与 bugfix

- **modality fast path**：固定 498×321 截图的 image token 恒为 160（12 task
  首尾 24 条实测）；DOM/instruction 长度用 text-only tokenizer 与完整 processor
  逐项相等。先用一张真实图片标定 image token，其余走 fast path，超长记录自动
  回退完整 processor。100,008/100,008 覆盖，fallback 0、truncation 0。
- **legacy Key64 layout bug**：恢复代码残留旧 `(32,28,4)`；冻结语义布局
  (32,16,16,0) 对应的 Key64 modality layout 应为 `(32,32,0)`。`ragged_store.py`
  改为从冻结 semantic layout 自动推导；categorical bottleneck 测试的 detail
  slice `32:44` 同步修正为 `32:48`。
- **Full-H 抽取**：6 ranks × 16,668，约 12.3 states/s/rank、每卡 ~18.7 GiB，
  总占用 ~420 GiB；`done.npy == key64-done.npy` 且前缀连续。
- **PCA first-N 纠错**：恢复期生成的 2,000-state PCA 实为 first-N selector
  （全部来自 `click-button-v1`），并非 task-balanced。新增确定性
  `task_balanced` selector + 回归测试；正式 PCA 见 §9.6。旧 artifact 保留供
  审计但禁止作为正式坐标。
- **A2 K-means OOM 修复**：旧实现把 `(70018,64,512)` latent 留在 GPU 再整体
  转置为 2,048 个 PQ 问题（额外连续分配 9.18 GB），K-means 前即 OOM。修复后
  全量 latent 走 host FP32 + GPU 驻留 64 问题 + `per_problem_fold_in_v1` 种子
  协议，并有 blocked-vs-monolithic 精确等价测试。
- **teacher 编码后端**：memcompiler 环境的 vLLM 0.11.0 与其 Torch CUDA ABI 不
  匹配（模型加载前退出、不占 GPU）。改用官方 SentenceTransformer 6.0 本地
  encoder；其原生 spawn pool 无法 pickle checkpoint 动态函数 → 改为每 GPU 一
  个独立 deterministic shard process。
- **A2 作废时间线**：8-21 17:42 起训的 A2 绑定纠错前的旧 PCA（19:31 才发现并
  重训 20k-balanced），该轮 A2 坐标失效，不入 cache、不入 loss（见 §9.3）；
  绑定新 PCA 的重训待做。
