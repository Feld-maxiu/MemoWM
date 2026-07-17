# ResidualMem v0.3：Exact Segment + Progressive Reader

本文是当前代码的实现契约。`ResidualMem_v0.2_极简可实现版.md` 仅保留为历史设计记录；v0.3 不读取 v0.2 memory 或 checkpoint，也不提供迁移器。

## 1. 核心选择

ResidualMem v0.3 不再为状态字段配置 `weighted`、权重或 lambda。进入规范状态的字段全部精确重建，`FieldSpec` 只描述名字、类型、值域和 optional 属性。换言之，代码没有逐字段 `must` 开关，因为“属于 schema”本身就等价于 must。

当前 Crafter 规范状态仍严格限定为 40 个字段：`pos_x/pos_y`、16 个 inventory 字段和 22 个 achievement 字段。当前版本不伪造语义地图、空间关系或观测中不存在的实体关系。DreamerV3/RL 主体代码保留，便于后续把 Reader 或 memory 接回 RL policy。

## 2. 世界模型与精确残差

世界模型是两层 GRU，默认 hidden size 为 512。一步预测只读取“上一重建状态 + 当前动作 + dt”，循环状态由 decoder 自己维护，不能读取未来真值。每个 Segment 都从精确 anchor 和零 GRU hidden 开始，因此可以独立打开。

训练使用 teacher forcing，但训练窗口严格按 codec 的 Segment 边界切分并重置 hidden。v0.3 不再进行多 lambda closed-loop fine-tuning：精确模式中每一步经过残差修正后的重建状态就是真实状态，多 lambda 既没有定义，也不会产生不同的 decoder-visible 输入。

编码时比较预测状态和真实状态，只要任意 schema 字段不同就写入该步 residual。解码时先运行 WM 默认预测，再应用 residual，因而所有规范字段逐步精确。文件 magic 为 `RSMEMV03`，每个 Segment 有 CRC 和末状态 hash。

## 3. Segment 事件索引

检索单位是 Segment。每个 Segment 只有一个 dense embedding，它只做语义兜底；主要召回来自 SQLite `.rmi` 中的结构化倒排索引。

结构化索引仅保存五类事实：

- 执行过的动作；
- 发生变化的字段名；
- 出现过的 literal；
- Segment 的起止时间；
- 从当前规范字段和动作中可确定的实体。

索引故意不保存关系、字段 old/new 值、精确事件步位置或完整状态。它负责找到可能相关的 Segment，而不是替代 memory stream。具体值和精确时间只能通过打开 Segment 后重建得到。

Segment document 由上述五类事实确定性生成。外部 embedding provider 对 document 编码后，以 `.npy` 矩阵导入。索引将每个归一化向量按 Segment 独立量化为 int8 + scale，并记录 provider 的 model id 和维度；查询向量必须来自相同 model id。代码只定义 `EmbeddingProvider` 协议，不绑定或安装具体模型。

## 4. 结构化优先、dense 补位

`QueryPlan` 可以包含 actions、changed fields、literals、entities、time range 和第一次展示所需的字段。候选排序遵循：

1. 对每个结构化条件做精确倒排匹配；
2. 按结构化覆盖率、时间和 Segment id 确定性排序；
3. 结构化候选先占据 top-k；
4. 若仍有空位，才按 dense cosine score 补齐。

因此 dense 相似度不会把一个已有结构化证据的 Segment 挤出候选首位。默认 `top_k=8`。

## 5. Progressive Reader

`ReaderPolicy` 是抽象策略边界，当前没有具体 LLM backend。一次查询的控制流如下：

1. Reader 将自然语言查询转成 `QueryPlan`，同时选择第一屏需要看的字段；
2. 引擎检索 Segment，并自动打开首个候选；
3. 只把该 Segment 的精确 anchor 中所选字段发给 Reader；
4. 信息不足时 Reader 调用 `EXPAND(fields, max_steps)`；
5. decoder 从当前 GRU hidden 向前运行，到“下一个 residual”或 step cap 中较早者停止，默认 cap 为 8；
6. 当前步到停止步的所有中间状态都会发送给 Reader，但只投影 Reader 请求的字段；
7. Reader 还可以 `REVEAL` 已缓存区间，或 `SWITCH` 到另一个候选 Segment；
8. `ANSWER` 必须引用实际展示过的 `(segment_id, step range, fields)`，否则引擎拒绝答案。

字段选择只减少发给 Reader 的证据，不改变 WM。WM 始终重建完整规范状态，因为后续 hidden 和 residual 解码都依赖完整闭环状态。Segment session 是前向状态机，持有当前完整状态、独立 GRU hidden、action cursor、lazy residual cursor、literal dictionary 和已经重建的缓存。

## 6. 评估与成本

codec 的正确性指标是 exact field accuracy 和 exact state accuracy，二者在合法 v0.3 stream 上都必须为 1。检索/Reader 实验的主结果应报告 accuracy-cost curve，而不只报告最终正确率。每条 query trace 至少记录：候选来源、工具调用、打开的 Segment 数、WM steps、应用的 residual 数、发送给 Reader 的 UTF-8 evidence bytes、index bytes、memory bytes 和端到端延迟。

建议消融包括 structured-only、dense-only、eager full-Segment、persistence WM、no-action WM。所有实验固定并记录 seed，答案正确性必须对真实 query label 评估，不能用检索分数或模型自评代替 ground truth。

## 7. CLI 边界

- `encode/decode`：精确 v3 memory stream；
- `export-index-docs`：导出确定性 Segment documents；
- `build-index`：导入预计算 embedding 并建立 `.rmi`；
- `retrieve`：执行结构化优先候选召回；
- `replay-query`：用 JSON scripted Reader 复现实验 trace；
- `sanity`：小规模 Crafter、GRU 与 codec 确定性检查。

真实 LLM Reader 和真实 embedding provider 留给后续实验接入；抽象接口、证据边界和成本核算已经固定。
