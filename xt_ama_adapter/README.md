# AMA → x_t 数据适配器

这个独立包将 AMA-Bench trajectory 中的单个 `action + observation` 转成
稳定、query-independent 的 x_t 输入记录，并包含与既有 QFormer/retrieval
head、Qwen3-VL 查询向量的运行时接口。它不修改 AMA-Bench 源文件。

每条输出记录保留 `episode_id`、`step_index` 与原始 evidence 文本；后续
`xt_rag` 检索器可以用该映射将 top-k 结果交回 AMA Reader。

运行测试（从 `scb/` 根目录）：

```bash
PYTHONPATH=xt_ama_adapter:AMA-Bench AMA-Bench/.venv/bin/python -m unittest discover -s xt_ama_adapter/tests -v
```

测试会读取本地 `AMA-Bench/dataset/test/open_end_qa_set.jsonl` 的第一个
episode，但不会调用模型或 GPU。

## x_t 检索运行时

运行时必须在已有的 `qwen-vl` 环境中执行，并显式传入以下同坐标系的模型
与权重。默认设备是 CPU；不会自动使用 GPU：

- Qwen3.5-9B：AMA step 的冻结 backbone；
- `qformer-K16-nosa.pt`：16×512 的 QFormer x_t；
- `retrieval-head-qformer-wma.pt`：与上述 QFormer cache 匹配的 4096 维
  retrieval head；
- Qwen3-VL-Embedding-8B：AMA 问题的 4096 维查询向量。

`QFormerRuntimeConfig`、`QFormerXTDocumentEncoder`、`QwenVLQueryEncoder` 和
`XTRetrievalIndex` 构成完整的“step 向量 → 问题向量 → top-k 原始 step 文本”接口。
首次运行前可调用 `validate_artifact_pair(config)`；它只读 checkpoint、不会
加载 9B 模型到 GPU，并会拒绝 PCA/QFormer 或不同 slot 数的坐标混用。

## 当前边界：Qwen3-32B latent Reader 尚未实现

本包当前只实现 AMA→x_t 的数据/检索层，不提供 AMA-Bench 的运行时注册，也不将
检索结果回映为文本给 Reader。目标实验要求把 top-k 的 `(xbar, valid)` 经一个
**尚待训练**的 `512 → 5120` bridge 直接注入本地 Qwen3-32B 的
`inputs_embeds`。现有 Qwen3.5 connector 的输出维度是 4096，不能复用。

后续应在本包中单独加入 bridge 训练、checkpoint 验证和 Qwen3-32B direct-input
推理入口；AMA-Bench 保持从官方仓库直接获取且不作源代码修改。
