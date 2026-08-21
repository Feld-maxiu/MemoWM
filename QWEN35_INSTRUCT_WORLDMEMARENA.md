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
outputs/state_tokenizer/v9-instruct/key64-static-pca.npz
outputs/state_tokenizer/v9-instruct/key64-static-pca-normalization.npz
outputs/a2/v9-instruct-m32-gw.npz
outputs/instruct_bridge/v9-instruct/train-cache-fused-observation.npz
outputs/instruct_bridge/v9-instruct/retrieval-head-fused-observation.pt
outputs/instruct_bridge/v9-instruct/input-connector.pt
outputs/instruct_bridge/v9-instruct/layer16-connector.pt
```

已删除的模型权重和上述训练产物不能由源码恢复，需要重新生成。

## 4. Retrieval head 训练

训练数据只来自 BrowserGym/MiniWoB++：默认 5,000 train + 500 validation，按 task 平衡抽样；WorldMemArena 记录、QA 和答案均不参与训练。

Teacher 是同一 BrowserGym observation 的官方 Qwen3-VL document embedding：

- image：BrowserGym screenshot；
- text：真实 task instruction + AXTree DOM；
- image/text 必须位于同一个 document 请求；
- 输出为归一化 4096 维 `teacher_fused_embedding`。

Student 是同一个 `MaskedAttentionRetrievalHead`，分别读取 `xbar` 和 A2 `xbar_hat`：

```text
LayerNorm(512)
  -> masked scalar attention over 64 slots
  -> weighted pooling
  -> Linear(512,1024) + GELU + Linear(1024,4096)
  -> L2 normalize
```

目标为 Xbar/A2 各自的 symmetric InfoNCE + cosine，并加 `0.1 ×` 两分支 consistency。共享 head 是刻意的：比较对象是表示损失，不允许给 A2 单独增加检索容量。

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
  --features outputs/state_tokenizer/v9-instruct/static_features \
  --normalization outputs/state_tokenizer/v9-instruct/key64-static-pca-normalization.npz \
  --a2-checkpoint outputs/a2/v9-instruct-m32-gw.npz \
  --output outputs/instruct_bridge/v9-instruct/train-cache-fused-observation.npz

python -m experiments.state_tokenizer.train_retrieval_bridge \
  --cache outputs/instruct_bridge/v9-instruct/train-cache-fused-observation.npz \
  --output outputs/instruct_bridge/v9-instruct/retrieval-head-fused-observation.pt
```

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

