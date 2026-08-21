# Qwen3.5-Instruct × WorldMemArena：训练与对照协议

本文固定 ResidualMem v9 在 WorldMemArena web 子集上的输入、训练和评测协议。核心目标是回答：将当前观察压缩为 `xbar_t` / A2 后，是否能在相同 RAG 检索配置下接近未压缩的 fused observation。

## 1. 不可混淆的三类内容

一个 web round 的字段分为：

| 内容 | tokenizer / fused observation | 完整 round 文本行 |
|---|---:|---:|
| 当前 screenshot | 是 | attachment 元数据 |
| user 原始文本 | 是 | 是 |
| image caption | 是 | 是 |
| assistant 的观察/计划（`I can see...`） | **否** | 是 |
| assistant Action JSON | **否** | 是 |

assistant 是基于当前观察产生的 policy/action 输出，不是 `x_t` 的输入。数据 loader 会把 caption 再内联到 user text；适配器必须先移除这份副本，保证 caption 在 observation 中只出现一次。

WorldMemArena 没有 BrowserGym AXTree。零训练适配将 user 文本与 caption 序列化为合法的 synthetic AXTree；截图仍走视觉 token。这个映射仅复用 tokenizer 的既有输入槽，不使用 benchmark 标签。

## 2. 严格对照

### Raw-Fused

名称：`Qwen3-VL-Embedding-8B-FusedObs-RAG`

每个有观察的 round 产生两条 document row：

1. 完整 round 文本（user + assistant）；
2. screenshot + genuine user text + caption 在同一条 Qwen3-VL document 请求中的 fused vector。

空观察 round 只产生第 1 条。fused 编码失败必须报错，禁止静默降级为 caption-only。

### ResidualMem

四个主实验：

- `ResidualMem-Instruct-Xbar-Input-RAG`
- `ResidualMem-Instruct-A2-Input-RAG`
- `ResidualMem-Instruct-Xbar-L16-RAG`
- `ResidualMem-Instruct-A2-L16-RAG`

它们保持相同的完整 round 文本行；第 2 条由共享 retrieval head 将 `xbar_t` 或 A2 重建的 `xbar_hat_t` 映射到归一化 4096 维向量。

所有方法使用官方 Qwen3-VL query encoder、全局 row-level cosine、`top_k=10`，不做 round 去重。因此同一 round 的两条 row 可能同时命中，也可能只命中一条。官方 `Qwen3-VL-Embedding-8B`（完整文本向量 + 独立 image 向量）保留为论文复现参考，但不是压缩质量的严格控制组。

## 3. v9 tokenizer

v9 使用本地 `Qwen3.5-9B-Instruct` 第 16 层。相对 v8 Base，仅 prompt wrapper 改为原生 chat template；DOM/instruction marker、Static Key64 `(32,16,16,0)`、PCA512、train-only group/channel normalization、A1/A2 均保持同一语义。

主要产物：

```text
outputs/state_tokenizer/v9-instruct-pca20k-balanced/key64-static-pca.npz
outputs/state_tokenizer/v9-instruct-pca20k-balanced/key64-static-pca-normalization.npz
outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/train-cache-fused-observation.npz
outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/retrieval-head-fused-observation.pt
```

2026-08-21 当前实验按用户决定只评估连续 `xbar`，不重跑 A1/A2。旧 A2 checkpoint
绑定旧 PCA 坐标，不能与本节的 20k-balanced PCA 混用；它既不进入 cache，也不进入
retrieval loss。

### 3.1 A1 / A2 正式重训配置

旧 v8 checkpoint 的结构可以用来核对配置，但其 metadata 绑定 Base 模型的 PCA、
normalization 和 records hash，不能直接当作 v9 结果。v9 从 seed 0 重新训练；train split
用于优化，validation 用于选取 best checkpoint，全程不读取 test split。

A1 固定为 64 个 512 维连续 token，保持 1.0× 标量宽度，训练 10,000 steps：

```text
stage steps       = 2,000 / 3,000 / 5,000
learning rates    = 1e-3 / 3e-4 / 1e-4
batch             = 32
eval every        = 250
matmul precision  = highest
patience          = 0（固定预算）
```

A2 从本轮 A1 checkpoint 初始化，使用冻结的 v8 主配置：PQ、64 tokens、M=32、C=256，
`group_weights=(1,2,0.5,0.5)`，K-means 25 iterations，训练 70,000 steps；codebook LR
`3e-4`、backbone LR `3e-5`、batch 32、每 500 steps validation、patience 0。prompt 权重虽
保留为 0.5，但冻结布局的 prompt 槽数为 0，因此不贡献 loss。

K-means **不抽样**：70,018 个 train state 全部参与 2,048 个 `(token, subspace)` 问题。
为避免一次性在 GPU 物化约 9.18 GB latent 及另一份同尺寸转置副本，latent 以 FP32 放在
host memory，GPU 每次只处理 64 个独立问题。随机数协议为
`per_problem_fold_in_v1`（seed 0 + 全局 problem id），因此调整 resident problem batch 只
改变显存峰值，不改变每个问题的初始化或最终中心。该内存调度记录在结果 JSON 的
`init.kmeans` 中。

两阶段均使用 FP32 activation、`highest` matmul precision 和 FP64 host metric 累加。
统一入口为：

```bash
./scripts_v9_instruct.sh a1
./scripts_v9_instruct.sh a2
```

正式产物分别为 `outputs/a1/v9-instruct{,.npz}` 和
`outputs/a2/v9-instruct-m32-gw{,.npz}`。

## 4. Retrieval head 训练

训练数据只来自 BrowserGym/MiniWoB++：默认 5,000 train + 500 validation，按 task 平衡抽样；WorldMemArena 记录、QA 和答案均不参与训练。

Teacher 是同一 BrowserGym observation 的官方 Qwen3-VL document embedding：

- image：BrowserGym screenshot；
- text：真实 task instruction + AXTree DOM；
- image/text 必须位于同一个 document 请求；
- 输出为归一化 4096 维 `teacher_fused_embedding`。

当前主实验的 Student 是 `MaskedAttentionRetrievalHead`，只读取连续 `xbar`：

```text
LayerNorm(512)
  -> masked scalar attention over 64 slots
  -> weighted pooling
  -> Linear(512,1024) + GELU + Linear(1024,4096)
  -> L2 normalize
```

目标为 Xbar 的 symmetric InfoNCE + cosine。代码仍向后兼容 `representation=both` 的
Xbar/A2 共享 head（该模式另有 `0.1 × consistency`），但本轮产物 metadata 明确为
`representation=xbar`，不含 `a2_xbar`。

缓存协议锁：

```text
qwen35_instruct_bridge_cache_v2
qwen3_vl_fused_observation_v1
browsergym_task_plus_dom_v1
```

旧的 screenshot-only `teacher_embedding` cache 会被拒绝。

示例：

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

### 4.1 2026-08-21 正式 Xbar 结果

- PCA：20,000 个 train state，12 task 各 1,666–1,667；解释方差
  `0.9185917377`，SHA256 `f98c517c...f6efb2a`。此前恢复出的 2,000-state PCA 实际是
  first-N 且全部来自 `click-button`，不是 task-balanced，已禁止作为正式坐标。
- cache：5,000 train + 500 validation，只有 `xbar/valid/teacher_fused_embedding`，
  所有值 finite、invalid slots 精确为零、索引 5,500/5,500 唯一。
- retrieval head：best step 5,000，validation loss `0.6449992`、paired cosine
  `0.8711105`；500-way Recall@1/5/10 为 `0.840/0.956/0.986`，MRR `0.8946`。

这些是 **MiniWoB/BrowserGym 域内 validation**，不能替代 WMA 跨域检验。

## 5. Reader connector

Input connector：`LN(512) -> Linear(4096) -> GELU -> Linear(4096) -> RMSNorm`，另有零初始化 rank embedding。它把每个 latent slot 作为 Qwen 输入 soft token。

Layer-16 connector：先严格逆 normalization/PCA 恢复到 4096 维 layer-16 状态，再叠加零初始化的 `Linear(512,4096)` residual adapter；推理时在第 17 层入口替换 prefix states。

Reader 必须用 Qwen3.5-Instruct 原生生成答案。若 connector 未加载，适配器不得假装运行 native reader；当前实现会明确报错或走未启用 native-answer 的外部 answer stage。

## 6. 分布偏移与判据

训练域是 MiniWoB++ 的规则化小网页，评测域是 1280×720 的真实 Chrome 轨迹，caption 更长、视觉熵和页面复杂度更高。这是实验的主要风险，而不是实现细节。必须同时报告：

- Raw-Fused vs Xbar：连续 tokenizer 的跨域损失；
- Xbar vs A2：离散化额外损失；
- retrieval RC/Recall 与最终 QA 指标；
- WMA web 内按“有截图/无截图、caption 长度、session 长度”分桶结果。

成功判据不是绝对分数复现论文全表，而是在完全相同的两行建库与 top-10 协议下，Residual 与 Raw-Fused 接近；若 Raw-Fused 自身偏低，应先排查 Qwen server/数据协议，不能归因给 tokenizer。

### 6.1 WMA web smoke 的当前结论

`web_01` 最终 checkpoint、完整 25 sessions、14 个 screenshot observation、前 10 个
问题的严格对照结果：

| 指标 | Raw-Fused | Xbar |
|---|---:|---:|
| evidence Recall@10 | 0.765 | 0.725 |
| NDCG@10 | 0.5429 | 0.5529 |
| top-10 observation rows（100 个槽位合计） | 8 | 0 |

两行检索的 top-10 overlap 为 0.91、top-1 agreement 为 1.00，但这是因为两侧共有
49 条完全相同的 full-round text row。只看 14 条 observation，Raw fused 与 Xbar 的
paired cosine 为 `0.1836`，14-way Recall@1/5/10 为 `0.0714/0.5714/0.8571`，MRR
`0.2958`。Recall@1 接近随机 `1/14`，因此当前零训练 synthetic-AXTree 适配**没有通过
跨域压缩门禁**；不能用总体 top-10 overlap 宣称压缩成功，也不应在修正前直接扩到全量
web benchmark。

逐问证据与 top-10 rows 保存在：

```text
outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/
  worldmemarena-web01-checkpoint0-10q.json
  worldmemarena-web01-final-checkpoint-10q.json
```
