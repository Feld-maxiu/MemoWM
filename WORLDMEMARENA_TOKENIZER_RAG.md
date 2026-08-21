# WorldMemArena tokenizer-RAG 对照速查

权威定义见 `QWEN35_INSTRUCT_WORLDMEMARENA.md`。本文件只记录实现映射。

```text
raw full round text ─────────────── Qwen3-VL document encoder ─┐
                                                              ├─ global cosine top-10
screenshot + user + caption ─┬─ Raw-Fused Qwen3-VL encoder ───┤
                              └─ v9 tokenizer -> Xbar/A2 -> head ┘
query ─────────────────────────── official Qwen3-VL query encoder
```

每个非空 round 恰好两行；assistant 只在 full-round text 行，绝不进入 observation/latent 行。空观察仅一行。检索可能同时命中同一 round 的两行，不做 round dedup。

实现位置：

- ResidualMem observation serializer：`residualmem/benchmarks/worldmemarena_tokenizer.py`
- frozen runtime：`residualmem/latent/frozen_v8_runtime.py`
- retrieval/reader bridge：`residualmem/latent/instruct_bridge.py`
- WMA raw encoder：`eval_framework/memory_adapters/qwen_embed_adapter.py`
- WMA Residual adapter：`eval_framework/memory_adapters/residualmem_instruct_adapter.py`

