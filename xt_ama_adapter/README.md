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

## Qwen3-32B latent Reader（已实现）

Qwen3-32B latent Reader 已实现：`qwen32_bridge.Qwen32LatentReader` 将检索命中的
top-k `(xbar, valid)` 经训练好的 bridge（512→5120，
`syqa/post-qformer/bridge/qwen32-input-k32-rms-top10.pt`）注入本地 Qwen3-32B 的
`inputs_embeds`。评测入口为 `experiments/state_tokenizer/run_ama_web_latent.py`，
支持 `latent-only` / `latent+anchor`（verbatim 元素锚，保精确 ID/文本）/
`text-only` / `matched` 等 memory mode，以及 `semantic` / `lexical` / `hybrid` /
`augment` 检索通道。AMA-Bench 数据位于 `third_party/AMA-Bench`（不入库，需自行放置）。

bridge 训练/选型代码在 `experiments/state_tokenizer/`（`train_qwen32_bridge.py`、
`freeze_qformer_candidate.py`、`select_syqa_bridge_candidate.py` 等），与
AMA-Bench 官方代码保持分离；AMA-Bench 从官方仓库直接获取且不作源代码修改。
