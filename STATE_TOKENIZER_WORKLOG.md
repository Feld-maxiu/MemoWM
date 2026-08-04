# State Tokenizer Pilot 工作日志

## 实验约束

- 冻结 Qwen3.5-9B-Base，第 16 层状态，不携带跨时间步 KV cache。
- 表示只比较 `H → Y64`、`H → Y32`、`Y64 → PCA512`。
- 如果 64 槽 probe gate 失败，立即停止；不得自动训练 learned queries。
- 大型产物写入已被 `.gitignore` 覆盖的 `outputs/state_tokenizer/`。
- 完成后删除下载碎片、未完成 shard、临时 checkpoint 和 PCA 工作矩阵；保留可复现实验所需的数据 manifest、最终表示、PCA artifact、metrics 与日志。

## 2026-08-03：环境审计与启动

- 当前工作目录：`/mnt/data/users/luzheng/workspace/iclr/czs/ResidualMem`。
- 当前机器：8× NVIDIA RTX PRO 5000 72GB Blackwell；启动时 GPU 0/1 分别约有 67GB/45GB 可用，其余卡已有约 49–52GB 占用。
- 采用共享只读环境 `/mnt/data/public_tools/miniconda3/envs/qwen-vl`：PyTorch 2.7.1+cu128、Transformers 5.14.1，已确认沙箱外 CUDA 可识别 SM120。
- MiniWoB 使用已部署的 `../miniwob-plusplus/.venv`、Chrome 151 与匹配的 ChromeDriver。
- Qwen3.5-9B-Base 已下载并校验为完整 4-shard checkpoint，位于 `outputs/state_tokenizer/models/Qwen3.5-9B-Base/`（约 19GB）。
- ResidualMem worktree 原本存在大量用户修改和未跟踪的 latent 实现；本实验新增独立 `experiments/state_tokenizer/`，不覆盖这些文件。

## 实现记录

- 新增确定性 64/32 槽布局、adaptive average pooling、DOM 序列化和 probe 标签提取公共模块。
- 新增可并行 MiniWoB collector：按任务分配 worker、按 episode 划分 60/20/20、保存 screenshot/DOM/instruction 与仅用于 probe 的浏览器属性 sidecar。
- 新增 Qwen layer-16 early-stop extractor；一次前向同时生成 Y64、Y32 与 H 的固定 mean/max probe 摘要，完整变长 H 不落盘。
- 新增 GPU 线性 probe、Y64 硬停止 gate、条件式 PCA512 拟合/转换，以及 feature shard 完整性校验。
- learned queries 不属于当前实现范围。

## 2026-08-03：小样验证

- 单元测试：`tests/state_tokenizer/test_common.py` 共 4 项通过；全部实验脚本通过 `py_compile`。最终环境未安装 pytest，因此使用等价的逐函数 runner 复验 4 项测试。
- 真实 MiniWoB：`click-checkboxes-v1` 采集 4 个连续状态成功，sidecar 能识别 unchecked → checked 的状态变化。
- Qwen Base checkpoint 没有 chat template；已改用其原生 `<|vision_start|><|image_pad|><|vision_end|>` 输入协议，并通过 processor token 展开验证。
- 4 个真实状态端到端提取成功：图像 token 70 个，DOM token 446–453 个，instruction token 9–10 个；序列总长 570–576，无截断。
- GPU 0 上 early-stop 到第 16 层的稳态提取速度约 1.08 states/s。Transformers 报告缺少 `fla`/`causal-conv1d` 快路径，当前使用官方 torch fallback；本轮不修改共享环境。

## 2026-08-03：正式数据采集与特征抽取

- 使用 4 个独立 Chrome/MiniWoB worker 收集 10,008 个状态；12 个任务各 834 个状态。
- episode 级划分为 train/validation/test = 6,013/1,996/1,999；共 5,235 个 episode，同一 episode 未跨 split。
- 合并 manifest SHA256：`1b92a077610f0009870465fb877c0a25bfe1eee9e68c5c2af84062417b1822e9`。
- 检查所有 screenshot 路径存在，抽检 100 张 PNG 可解码；采集日志无 episode 或 sidecar 异常。
- GPU 2–7 被既有 6-way SGLang 服务占用（每卡仅余约 20–24GB），未抢占；正式抽取均分到尚有安全显存余量的 GPU 0/1。
- 两个 extractor 各负责 5,004 条记录；启动后的稳态速度约 5.0 states/s/GPU。
- 正式抽取耗时约 980 秒；序列长度 282–1,337，图像/DOM/instruction token 范围分别为 70、161–1,210、7–33；0 条截断，数值与索引覆盖审计通过。
- 首次启动 probe 时发现任务分类头把标签数组宽度误当成类别数，尚未产生指标即触发 CUDA assert；已修正为 `max(label)+1` 并加入类别范围检查，随后重跑成功。
- Y64 相对 H 的硬门通过：state retention 1.0000、text retention 0.9206、mean retention 0.9603、task accuracy drop 0.0000。

## 2026-08-03：最终 pilot 结果

所有 probe 都使用相同的简单线性头与相同 split。为处理变长 H，参考输入是 image/DOM/instruction 各自 mean+max 拼接的 `6×4096` 摘要；Y64/Y32 也按对应模态槽做相同 mean+max 聚合，因此结果只支持“当前 probe 套件未检测到明显损失”，不等价于无损重建完整 H。

| 表示 | 每状态形状 | task acc | state mAP | BoW mAP | 对参考的 state/text retention | 结果 |
|---|---:|---:|---:|---:|---:|---|
| H probe 摘要 | `6×4096` | 1.0000 | 1.0000 | 0.4658 | — | reference |
| Y64 | `64×4096` | 1.0000 | 1.0000 | 0.4340 | H→Y64: 1.0000 / 0.9206 | pass |
| Y32 | `32×4096` | 0.9990 | 1.0000 | 0.4778 | Y64→Y32: 1.0000 / 1.1595 | pass |
| Y64→PCA512 | `64×512` | 0.9985 | 1.0000 | 0.5085 | Y64→PCA: 1.0000 / 1.2804 | pass |

- PCA 使用 5,000 个 train 状态（320,000 个槽向量）拟合，512 维解释方差比为 0.9297。
- retention 大于 1 来自候选 probe 的测试 AP 高于参考 probe，属于独立优化/正则化差异；只能解读为本 pilot 未测到损失，不能声称压缩增加了信息。
- 按存储量，BF16 的 Y64/Y32/PCA512 分别约为 512/256/64 KiB 每状态。就本 pilot 而言，`Y64→PCA512` 在最小体积下仍通过全部 gate，适合作为下一阶段 WM 的首选输入；Y32 可保留为更少状态 token 的替代基线。

## 清理与最终审计

- 删除约 1.5GB 可重建的 `probe-inputs/` 缓存、4 份已合并的 worker JSONL、两个 `/tmp` sanity 目录、下载缓存元数据与 Python `__pycache__`。
- 保留最终 manifest/截图、Qwen checkpoint、H probe 摘要、Y64、Y32、PCA512 表示、PCA artifact、所有结果 JSON 和成功运行日志。
- 最终审计重新验证 manifest SHA256、10,008 条索引完整覆盖、两个 extraction/PCA done mask、PCA component 形状 `(4096, 512)`，以及三个 gate 均为 pass。
- 正式采集/抽取/PCA/probe 日志中未发现 traceback、OOM 或 episode 错误。

## 待执行与结果

- [x] collector 单元测试和真实 MiniWoB 小样本。
- [x] Qwen 多模态第 16 层抽取与 64/32 槽落盘小样。
- [x] 64 槽 probe gate（通过）。
- [x] 仅在 gate 通过后：32 槽 probe 与 PCA512（均通过）。
- [x] 中间产物清理与完成审计。

---

## 2026-08-03：v2 Full-H / Slot-Aware 修正实验启动

### 修正原因

- v1 的 `H probe 摘要` 只保留三种模态的 mean/max，不能作为完整 `H_t` oracle；自本节起统一改称 `H_global_summary`，其既有 gate 仅视为历史诊断。
- v1 probe 在预测前再次把 Y64/Y32 聚合成模态级 mean/max，无法检查局部 slot、slot 顺序或具体元素状态。
- 静态 role 和 BoW 容易受任务模板与 instruction 泄漏；v2 主 gate 只使用任务内变化的动态状态和随机 textbox value，DOM-only BoW 改为非阻塞诊断。

### 锁定协议

- 从现有 10,008 条 manifest 选择 train/validation/test = 2,000/500/1,000；每个 episode 最多一个状态。
- 临时提取完整 layer-16 image/DOM/instruction token，使用 ragged BF16 缓存；Y64/Y32/PCA512 复用现有结果。
- reader 固定为 `input→256 Linear→LayerNorm→单 learned query/一层 4-head cross-attention→Linear heads`。
- 使用模态内归一化位置 `p=(i+0.5)/N_modality`；Full-H token 与 pooled slot 都落在 `(0,1)`。
- probe seeds 为 0/1/2。Full-H oracle 不合格则报告 `INCONCLUSIVE_PROBE`；Y64 主 gate 失败则立即停止，禁止自动训练 learned queries。

### 已完成实现与数据审计

- 新增 compact DOM v2 parser 与动态标签：逐元素 checkbox/radio 状态、逐 textbox nonempty/focused、interactive tampered、随机 `state-#####` value。
- 标签只从实际送入 Qwen 的 DOM 字符串派生，不使用浏览器 sidecar。当前数据不支持的 disabled、动态 dialog、具体 option 与 success/failure 显式标记为 unsupported。
- 新增 episode-unique subset 选择器及测试；初次真实构建发现部分多步任务的独立 episode 少于等额任务配额，因此改用带 episode 容量上限的任务 water-filling，未放宽 episode 唯一性。
- 第一版状态代表等概率采样使 `checkbox_0_checked` 在 test 中只有 17 个负例。为避免降低 gate 阈值，已锁定多步 episode 采用 50% 初始态、50% 均匀动作后状态的采样方式，显式平衡未改变/已改变状态。
- 最终 subset 已固定为 3,500 个互不重复 episode，train/validation/test = 2,000/500/1,000；SHA256 为 `d58aaee36ae9a11b6cdb7b989a6de19ea4663d7cacbeaa19486f6bb376d55983`。
- 最终 subset 有 7 个满足三 split 正负例阈值的动态标签：`checkbox_0_checked`、`has_random_value`、`interactive_tampered`、`textbox_0_focused`、`textbox_0_nonempty`、`textbox_1_focused`、`textbox_1_nonempty`。其余逐元素标签保留在逐标签表中，但不计入主 gate。
- 新增 ragged Full-H store/extractor、H→Y64/Y32 BF16 精确对齐检查、统一 slot-aware reader、task-ID/task+step/instruction-only 泄漏 baseline 和三 seed 分阶段 gate 汇总器。
- 新增 v2 单元测试后共 10 项通过；所有新增脚本通过 `py_compile`。归一化位置和 reader padding invariance 均有独立测试。
- 正式 GPU 启动前复查：GPU 0/1 分别空闲约 66.8/45.1 GiB；GPU 2–7 各已有约 48–52 GiB 占用，故只将 GPU 0/1 判定为空闲安全卡，不抢占其余服务。
- 4-state Full-H 检查初次在 CPU pooling 上发现 Y32 BF16 不一致（最大绝对差 0.09375）；定位为 CPU/CUDA reduction 顺序差异。改用与旧缓存生成时相同的 CUDA reduction 后，Y64 与 Y32 均逐位一致，BF16 最大绝对误差均为 0.0。
- 统一 reader 在 64 个 train 状态上做 overfit sanity：动态状态 AP 与五位随机 value accuracy 均达到 1.0，证明单 query reader、归一化位置、loss 与评测链路能访问并学习 slot 内容。
- 正式 Full-H 已在 GPU 0/1 双卡启动，各处理 1,750 个状态；启动后稳态约 4.3–4.5 states/s/GPU。

### v2 待执行

- [x] 固定最终 3,500-state subset 与标签覆盖表。
- [x] 实现并单元验证 ragged Full-H extractor/store。
- [x] 实现 normalized-position slot-aware reader、泄漏 baselines 与三 seed gate。
- [x] 4-state Full-H→Y64/Y32 对齐及 reader overfit sanity。
- [ ] Full-H qualification。
- [ ] Y64 动态状态/value 主 gate；失败即停。
- [ ] 仅在 Y64 通过后运行 Y32/PCA512。
- [ ] 删除 Full-H 临时缓存并完成最终审计。

### 运行时修正

- 首次启动 Full-H/instruction-only probe 时，两个任务尚未完成首个 epoch 就各占用了上百个 CPU 线程。定位到 ragged reader 每次把磁盘 BF16 先展开为 FP32，且 instruction-only 为了取十几个 instruction token 会先读取整个 image/DOM 序列。
- 中止了这两个尚未产出 checkpoint/metrics 的 seed-0 任务；改为 copy-on-write memmap 的 BF16 零转换 view、按 modality offset 直接切 instruction，并将每个 probe 的 CPU threads 限制为 4。11 项测试复验通过后从 seed 0 重跑。
- 修正后 Full-H 与 instruction-only 都能在几十秒内完成多个 epoch，GPU 不再因无效 FP32 转换长期空等。该修正不改变任何 token 数值、reader 结构、标签、split 或 gate。

### Full-H 正式提取与资格检验

- 双卡正式提取完成 3,500/3,500 个状态，GPU 0/1 各 1,750 个；墙钟约 384 秒，速率分别为 4.555/4.555 states/s。
- 两个 shard 共 1,949,352 个 token，临时 BF16 Full-H 逻辑大小约 14.872 GiB。随机抽检 32 个状态，CUDA reduction 后重建的 Y64/Y32 与旧缓存均 BF16 逐位一致，最大绝对误差 0.0。
- 完成 Full-H、instruction-only、task-ID-only、task+step 四组 reader，每组只重训 probe seeds 0/1/2；Qwen 只提取一次。

| 表示 / baseline | 动态 task-macro AP | 随机 value accuracy | DOM-only BoW AP | 说明 |
|---|---:|---:|---:|---|
| Full-H slot-aware | 0.9783 ± 0.0019 | 0.8858 ± 0.0324 | 0.3605 ± 0.0032 | oracle candidate |
| instruction-only | 0.9609 ± 0.0046 | 0.1947 ± 0.0190 | 0.3156 ± 0.0045 | 最强动态/value 泄漏 baseline |
| task+step | 0.8250 ± 0.0025 | 0.1869 ± 0.0021 | 0.3096 ± 0.0009 | step 泄漏明显 |
| task-ID-only | 0.4501 ± 0.0000 | 0.1794 ± 0.0017 | 0.2585 ± 0.0349 | template baseline |

- Full-H 有 7 个可评测动态标签，满足“至少 6 个”的资格要求。
- 随机 value：Full-H 0.8858，相对最强 leakage baseline 0.1947 提升 0.6911，超过 +0.10 要求。
- 动态状态：Full-H 0.9783，相对最强 leakage baseline（instruction-only 0.9609）只提升 0.0175，未达到 +0.10 要求。逐标签上 `checkbox_0_checked` 的 Full-H/instruction-only AP 都为 1.0，`has_random_value` 也几乎都为 1.0，说明这些动态 presence/checked 标签仍被 instruction/episode 采样模式强烈泄漏。
- 因而 Full-H oracle 资格检验结果为 **`INCONCLUSIVE_PROBE`**。这不是 Y64 失败，也不能推出池化损失大；它表示当前动态 probe 无法把页面状态信息与 instruction/采样相关性充分分离。
- 按预先锁定的 hard-stop 协议，本轮没有运行 Y64、Y32 或 PCA512 的 v2 probe，没有计算 retention，也没有训练 learned queries。DOM-only BoW 始终只是诊断指标，不参与停止判断。
- 机器可读结果：`outputs/state_tokenizer/metrics/v2/aggregate-qualification.json`；逐标签表：`outputs/state_tokenizer/metrics/v2/per-label-qualification.csv`。

### v2 最终状态

- [x] Full-H qualification 已完成，结果 `INCONCLUSIVE_PROBE`。
- [x] 依协议在 Y64 前停止，未越过 gate。
- [x] 删除约 15 GiB 可重建 Full-H、12 个 probe checkpoint 与两个 sanity 临时目录；这些缓存不可直接恢复，但可由保留的 subset、Qwen checkpoint 和脚本重建。
- [x] 最终审计：11 项测试通过；8 个新增/修改脚本通过 `py_compile`；12 份正式 result JSON 齐全；subset SHA256 复核一致；v2 日志无 traceback/OOM/RuntimeError；聚合状态复核为 `INCONCLUSIVE_PROBE`。

---

## 2026-08-03：v2.1 Value-only Compression Retention

### 修订理由与协议

- 动态二值 probe 的 leakage 不再阻塞干净的随机五位 value 实验。上一轮 Full-H value accuracy 为 0.8858，而 instruction-only 仅为 0.1947，已经证明 Full-H 能访问页面中的具体随机内容。
- 新实验只在存在 `state-#####` 的样本上优化五个十分类 value heads；train/validation/test 分别为 590/133/240。动态状态、静态 role、task 与 BoW heads 仍保持同一 reader 定义，但不参与 loss 或 checkpoint 选择。
- 表示固定为 Full-H、instruction-only、Y64、Y32、PCA512（代码名 `x64`），reader 结构、hidden width、单 query、归一化位置、optimizer、split 和 seeds 0/1/2 完全相同。
- 主指标沿用此前的五个位置平均 accuracy `macro_position_accuracy`，并新增完整五位字符串全对的 `exact_value_accuracy` 作为更严格诊断。
- instruction-only 是唯一 leakage baseline。主 retention 同时报告均值比值和 seed-paired 均值/标准差：`(M_compressed-M_instruction)/(M_full_h-M_instruction)`。

### 实现状态

- `slot_probe.py` 新增 `--objective value_only`，自动过滤有效 value 样本、只计算 digit CE、只用 validation value accuracy 早停，并记录三个 split 的有效样本数。
- 新增 `aggregate_value_only.py`，计算 Y64/Y32/PCA512 的 macro-position 与 exact-value leakage-adjusted retention，并输出 JSON/CSV。
- 新增 retention 单元测试；当前 state-tokenizer 测试总数为 12，全部通过。
- GPU 预检结果与前一轮一致：GPU 0/1 是仅有的安全卡；GPU 2–7 各有约 48–52 GiB 常驻占用，不抢占。

### 待执行

- [x] value-only reader smoke sanity：Y64、64 个 value 样本、8 epochs，loss 从 2.415 降至 2.166，训练集位置准确率从 0.150 升至 0.250；确认过滤、digit-only loss、早停与 exact-value 指标链路工作正常。
- [x] 双卡重建临时 Full-H：GPU 0/1 各 1,750 states，墙钟约 381 秒、4.60/4.59 states/s；随机 32-state CUDA 对齐检查中 Y64/Y32 的 BF16 最大误差均为 0.0。
- [x] 五种表示各 3 seeds value-only probe；每个结果均为 train/validation/test = 590/133/240 个有效 value，最多 60 epochs、patience 8。
- [x] 汇总绝对 value 指标与 leakage-adjusted retention。
- [x] 清理临时 Full-H/checkpoint 并完成最终审计：删除约 15 GiB 可重建 Full-H、15 个 probe checkpoint、smoke 目录和 Python cache；保留 15 份 result JSON、聚合 JSON、CSV 与全部日志。

### Value-only 结果

| 表示 | 五位置平均 accuracy | 五位整串 accuracy | `R_value`（均值比值） | seed-paired `R_value` |
|---|---:|---:|---:|---:|
| Full-H | 0.8806 ± 0.0206 | 0.6417 ± 0.0579 | 1.0000 | 1.0000 |
| instruction-only | 0.1828 ± 0.0054 | 0.0000 ± 0.0000 | 0.0000 | 0.0000 |
| Y64 | 0.6436 ± 0.0064 | 0.3597 ± 0.0064 | **0.6604** | 0.6610 ± 0.0249 |
| Y32 | 0.6336 ± 0.0146 | 0.3472 ± 0.0024 | **0.6461** | 0.6469 ± 0.0348 |
| Y64→PCA512 | 0.6978 ± 0.0179 | 0.3681 ± 0.0168 | **0.7381** | 0.7379 ± 0.0072 |

- `R_value` 主值按三 seed 均值代入 `(M_compressed-M_instruction)/(M_full_h-M_instruction)`；同时报告同 seed 配对后比值的均值和样本标准差，两种算法结论一致。
- 更严格的整串 exact-value retention 分别为 Y64 0.5606、Y32 0.5411、PCA512 0.5736；排序与主指标相同。
- 结论很明确：64 槽和 32 槽都保留了显著的页面随机 value 信息，但在这个单-query reader 下只保留约 65%–66% 的 leakage-adjusted Full-H 能力，没有达到此前设想的 0.80 retention。
- PCA512 在 Y64 上表现更好，主 retention 为 0.7381，但仍低于 0.80。它不可能增加原表示的信息；更合理的解释是 PCA 去噪、4096→512 降维改善了有限数据下 reader 的优化/正则化。
- Y64 与 Y32 差异很小（0.6604 vs 0.6461），当前主要损失更像来自固定分段池化本身，而不是 64→32 的额外槽数下降；PCA 则部分恢复了线性可读性。
- 机器可读聚合：`outputs/state_tokenizer/metrics/v2/value_only/aggregate-value-only.json`；简表：`outputs/state_tokenizer/metrics/v2/value_only/value-only-summary.csv`。
- 最终审计：12 项测试通过；8 个 value-only 相关脚本通过 `py_compile`；15 份结果的 protocol、seeds、590/133/240 样本数和 reader 配置一致；三项 retention 从原始 seed metrics 独立复算到 1e-12 一致；subset SHA256 仍为 `d58aaee36ae9a11b6cdb7b989a6de19ea4663d7cacbeaa19486f6bb376d55983`；正式日志无 traceback/OOM/RuntimeError/ValueError。

---

## 2026-08-03：v3 Fixed-Prompt Dynamic-State Revalidation

### 目的与锁定协议

- v2 动态状态实验中 instruction-only 的 task-macro AP 达到 0.9609，证明原任务 instruction 对 checked/focused/nonempty 存在严重泄漏；因此旧动态结果不能用于判断池化表示的信息保真度。
- 本轮不是仅在 probe 端屏蔽 instruction，而是在送入 Qwen 前把全部 3,500 个状态的任务 instruction 统一替换为以下任务无关 observation prompt：`请忠实表示当前页面状态，保留可见文本、输入值、控件类型、选中/聚焦/启用状态及空间关系。`
- 因 prompt 会改变 Qwen hidden state，本轮重新提取 layer-16 Full-H、Y64、Y32；PCA512 也只在 fixed-prompt 的 2,000 个 train 状态上重新拟合，不复用旧 PCA。
- probe 仍为同一个 256 hidden、单 learned query、一层 4-head cross-attention reader；位置仍使用模态内归一化坐标。没有为动态状态重新设计更复杂的 probe。
- loss 与 validation early stopping 只使用 9 个逐元素标签：checkbox 0/1/2 checked、textbox 0/1/2 nonempty、textbox 0/1/2 focused。满足既有三 split 覆盖阈值的 5 个标签构成主 task-macro AP，其余标签仍逐项报告但不混入主值。
- Full-H、Y64、Y32、PCA512，以及 fixed-prompt-token-only、task-ID-only、task+step 均训练 seeds 0/1/2。压缩 retention 使用不读取页面内容的最强 metadata leakage baseline（task-ID/task+step）校正，并同时保留逐标签绝对 AP 与 prevalence。

### 实现与冒烟测试

- 新增固定提示词 manifest builder、Full-H/Y64/Y32 联合 extractor、fixed-prompt PCA fit/transform、动态状态专用 objective 与三 seed 汇总器。
- fixed-prompt subset 保持 train/validation/test = 2,000/500/1,000、3,500 个 episode 全部唯一；原 subset SHA256 为 `d58aaee36ae9a11b6cdb7b989a6de19ea4663d7cacbeaa19486f6bb376d55983`，fixed-prompt manifest SHA256 为 `2febd4675a3c0b9da9473d82f804c32349197cc3957faca58b2dadbdd85e90c1`，且 `unique_instructions=1`。
- 相关脚本全部通过 `py_compile`；state-tokenizer 测试增至 15 项并全部通过。
- 4-state 真实 Qwen 冒烟测试确认固定提示词编码为 29 个 instruction token；从新 Full-H 重新池化的 Y64/Y32 与联合 extractor 输出均 BF16 逐位一致，最大绝对误差为 0.0。
- GPU 预检显示 GPU 0/1 分别约有 66.8/45.1 GiB 余量；GPU 2–7 各有约 49–52 GiB 常驻占用。本轮只并行使用 GPU 0/1，各负责 1,750 个状态，不抢占其余服务。
- 运行中确认 fixed-prompt-token-only 不能再解释为“instruction 语义泄漏”：Qwen 是因果模型，位于 image/DOM 之后的 29 个固定 prompt token hidden states 已被前文页面内容上下文化。它作为一个短状态表示继续报告，但不纳入 leakage baseline；真正不看页面内容的 task-ID/task+step 才用于泄漏校正。

### 当前进度

- [x] 固定 observation prompt manifest 与协议审计。
- [x] 动态状态专用 loss/metric、PCA 与汇总实现。
- [x] 15 项测试和 4-state Full-H→Y64/Y32 逐位一致性检查。
- [x] 双卡正式提取 3,500 个 fixed-prompt 状态。
- [x] 在 fixed-prompt train split 重拟合并变换 PCA512。
- [x] 七种表示/baseline 各 3 seeds 动态状态 probe。
- [x] 聚合逐标签 AP、泄漏校正 retention、最终审计与缓存清理。

### 正式运行与结果

- 双卡提取完成 3,500/3,500 个状态；GPU 0/1 各 1,750 个，墙钟分别为 384.08/383.48 秒，速率 4.556/4.563 states/s。固定 prompt 的 instruction 长度在所有状态中均为 29 token。
- 两个 Full-H shard 合计 2,000,798 个 token，逻辑 BF16 大小 15.265 GiB。随机抽检 32 个正式状态，Full-H 重新池化所得 Y64/Y32 与联合 extractor 保存值均 BF16 逐位一致，最大绝对误差 0.0。
- PCA512 仅用 fixed-prompt train split 的 2,000 状态（128,000 个 Y64 slot）拟合，累计解释方差为 0.93144；随后成功变换全部 3,500 状态。
- 七种表示/metadata baseline 各完成 seeds 0/1/2，共 21 个 probe。主指标是 5 个覆盖合格标签的 task-macro AP：`checkbox_0_checked`、`textbox_0_nonempty`、`textbox_1_nonempty`、`textbox_0_focused`、`textbox_1_focused`。

| 表示 / baseline | 动态状态 task-macro AP | 相对 task+step 的 leakage-adjusted retention |
|---|---:|---:|
| Full-H | 0.9881 ± 0.0132 | 1.0000 |
| fixed-prompt token hidden states | 0.9987 ± 0.0013 | 仅作短状态表示诊断 |
| task-ID-only | 0.4102 ± 0.0000 | metadata baseline |
| task+step | 0.7651 ± 0.0057 | 最强 metadata leakage baseline |
| Y64 | **0.9904 ± 0.0036** | **1.0104**（seed-paired 1.0136 ± 0.0672） |
| Y32 | 0.9798 ± 0.0036 | **0.9627**（seed-paired 0.9659 ± 0.0676） |
| Y64→PCA512 | 0.9770 ± 0.0056 | **0.9500**（seed-paired 0.9516 ± 0.0470） |

- `checkbox_0_checked` 在 Full-H/Y64/Y32/PCA 和 task+step 上 AP 都为 1.0，说明它仍被 episode 采样中的 task+step 完全预测，不能单独作为页面状态读取证据。泄漏校正中该标签的 oracle/candidate margin 都为 0，因此不会人为抬高 retention；主要有效证据来自 4 个 textbox nonempty/focused 标签。
- 四个 textbox 标签的逐标签趋势一致：Y64 AP 为 0.9927/0.9868/0.9928/0.9799，Y32 为 0.9969/0.9452/0.9967/0.9602，PCA512 为 0.9928/0.9571/0.9895/0.9454；顺序依次为 textbox-0 nonempty、textbox-1 nonempty、textbox-0 focused、textbox-1 focused。
- Y64 的 AP/retention 略高于 Full-H 不代表增加了信息；这是有限样本、不同输入长度和优化/正则化造成的 reader 可读性差异。合理结论是：在这个单-query reader 和 fixed-prompt 动态状态诊断下，Y64 未检测到相对 Full-H 的可测损失；32 槽和 PCA512 分别保留约 96% 和 95% 的泄漏校正能力。
- fixed-prompt token hidden states 的高 AP 也不表示任务 instruction 泄漏复发。它们位于 image/DOM 之后，已通过因果 attention 被页面前文上下文化；因此它是一个 29-token 的隐式状态摘要，而不是只含固定字符串语义的 baseline。

### 最终审计与清理

- fixed-prompt manifest 复核为 3,500 条、3,500 个 unique episode、`unique_instructions=1`，SHA256=`2febd4675a3c0b9da9473d82f804c32349197cc3957faca58b2dadbdd85e90c1`。
- 21 份 result JSON 的 protocol、representation、seed、train/validation/test=2,000/500/1,000、单 query reader 配置与 `dynamic_state_only` objective 全部一致；v3 正式日志未发现 traceback、OOM、RuntimeError、ValueError、AssertionError 或 NaN。
- 新增 metadata leakage 解释测试后，state-tokenizer 测试总数为 16，全部通过；所有 v3 脚本通过 `py_compile`。
- 已删除约 15.3 GiB 可重建 Full-H token 文件、21 个 probe checkpoint 和 4-state 临时目录；这些缓存不可直接恢复，但可由保留的 fixed-prompt manifest、Qwen checkpoint 与脚本重建。保留 Y64、Y32、PCA512、PCA artifact、21 份 result JSON、聚合 JSON、逐标签 CSV 与全部运行日志。
- 机器可读聚合：`outputs/state_tokenizer/metrics/v3/fixed_dynamic/aggregate-fixed-dynamic.json`；逐标签表：`outputs/state_tokenizer/metrics/v3/fixed_dynamic/fixed-dynamic-per-label.csv`。

---

## 2026-08-03：v3.1 Fixed-Prompt Exact Random-Value Revalidation

### 目的与协议

- 使用与 v3 动态状态实验完全相同的任务无关 observation prompt，在 Qwen 输入端替换原 task instruction：`请忠实表示当前页面状态，保留可见文本、输入值、控件类型、选中/聚焦/启用状态及空间关系。`
- 从既有 3,500-state fixed-prompt manifest 精确筛出含 `state-#####` 五位随机 value 的 963 个状态；train/validation/test=590/133/240，963 个 episode 全部唯一，`unique_instructions=1`，subset SHA256=`84bfaa9fb393db105db35a68316aaaf930b33281be1a4c5b761a0987a56a3bce`。
- 只为这 963 个必要状态临时重建 layer-16 Full-H，避免再次产生完整 3,500-state 的约 15 GiB 缓存；Y64/Y32/PCA512 复用 v3 fixed-prompt 正式表示，其中 PCA512 是此前仅在 fixed-prompt 2,000 个 train 状态上重新拟合的版本。
- Full-H、Y64、Y32、PCA512 使用完全相同的 256 hidden、单 learned query、一层 4-head cross-attention reader，seeds=0/1/2；loss 和 validation early stopping 只使用五个 digit 十分类交叉熵。
- 主指标为五个位置平均 accuracy，随机十分类 chance=0.1；严格指标为五位整串全对 accuracy，chance=`10^-5`。retention 统一定义为 `(M_compressed-M_chance)/(M_full_h-M_chance)`，同时报告 `1-retention` 相对损失与绝对 accuracy drop。
- fixed prompt token hidden states 已被 image/DOM 因果上下文化，因此不作为泄漏 baseline；额外运行 task-ID/task+step 只用于验证 metadata 是否接近随机 chance。

### 当前进度

- [x] fixed-prompt value-only subset、protocol 标识与 chance-adjusted 聚合器。
- [x] 19 项 state-tokenizer 测试及相关脚本 `py_compile`。
- [x] 双卡提取 963-state 临时 Full-H 并验证 Y64/Y32 对齐。
- [x] Full-H/Y64/Y32/PCA512 与 metadata baseline 各 3 seeds。
- [x] 汇总精确 value 损失、最终审计与临时缓存清理。

### 正式结果

- 双卡临时 Full-H 提取完成 963/963，GPU 0/1 分别处理 482/481 状态，墙钟 106.92/106.42 秒，速率 4.508/4.520 states/s；合计 466,385 个 token，逻辑 BF16 大小 3.558 GiB。
- 随机抽检 32 个 value 状态，从 Full-H 重建的 Y64/Y32 与 v3 保留表示均 BF16 逐位一致，最大绝对误差 0.0。
- 六组表示/metadata baseline 各完成 seeds 0/1/2，共 18 个 probe。所有主表示训练 60 epochs 上限、patience 8；每个结果都使用 train/validation/test=590/133/240。
- 有限 240-state test split 的 digit marginal 并非完美均衡：train 全局多数类 baseline 为 0.1642，按 task 条件为 0.1800，按 task+step 条件为 0.1842；训练得到的 task+step probe 为 0.1819。因此主 retention 使用 `max(task/task+step, theoretical chance)` 的更保守 baseline；同时保留理论 chance=0.1 校正结果。

| 表示 | 五位置平均 accuracy | 五位整串 accuracy | 保守 `R_position` / 相对损失 | 保守 `R_exact` / 相对损失 |
|---|---:|---:|---:|---:|
| Full-H | 0.8917 ± 0.0242 | 0.6569 ± 0.0428 | 1.0000 / 0% | 1.0000 / 0% |
| task-ID-only | 0.1767 ± 0.0090 | 0.0000 ± 0.0000 | metadata baseline | metadata baseline |
| task+step | 0.1819 ± 0.0005 | 0.0000 ± 0.0000 | 最强 metadata baseline | metadata baseline |
| Y64 | 0.6414 ± 0.0092 | **0.3722 ± 0.0127** | **0.6474 / 35.26%** | **0.5666 / 43.34%** |
| Y32 | 0.6281 ± 0.0133 | 0.3417 ± 0.0042 | **0.6286 / 37.14%** | **0.5201 / 47.99%** |
| Y64→PCA512 | **0.6856 ± 0.0010** | 0.3514 ± 0.0048 | **0.7096 / 29.04%** | **0.5349 / 46.51%** |

- 相对 Full-H 的绝对五位置 accuracy drop：Y64=0.2503、Y32=0.2636、PCA512=0.2061；严格整串 accuracy drop 分别为 0.2847、0.3153、0.3056。
- 仅用理论随机 chance=0.1 校正时，`R_position` 为 Y64=0.6839、Y32=0.6670、PCA512=0.7396；比保守结果略高，但排序和结论不变。
- 固定 prompt 结果与旧 task-instruction value-only 结果非常接近：旧 Full-H/Y64/Y32/PCA512 五位置 accuracy 为 0.8806/0.6436/0.6336/0.6978，新结果为 0.8917/0.6414/0.6281/0.6856。由此确认此前观察到的 random value 损失不是 instruction 泄漏造成的。
- PCA512 在逐位 accuracy 上最好，说明降维/去噪提高了有限数据下单 query reader 的线性可读性；但严格整串 accuracy 为 0.3514，低于 Y64 的 0.3722，因此不能宣称 PCA 保留了更多原始信息。
- 最终结论：固定分段池化已经造成主要精确 value 损失；64→32 槽只带来较小附加损失。按更保守 metadata 校正，Y64/Y32/PCA512 的逐位能力分别损失约 35%/37%/29%，严格整串能力损失约 43%/48%/47%。

### 最终审计与清理

- 963-state manifest SHA256=`84bfaa9fb393db105db35a68316aaaf930b33281be1a4c5b761a0987a56a3bce`；18 份 result JSON 的 fixed-prompt protocol、representation、seed、split 样本数、reader 配置和 `value_only` objective 全部一致。
- v3.1 日志未发现 traceback、OOM、RuntimeError、ValueError、AssertionError 或 NaN；19 项 state-tokenizer 测试全部通过。
- 已删除约 3.56 GiB 可重建临时 Full-H 与 18 个 probe checkpoint；保留 18 份 result JSON、聚合 JSON、CSV、fixed-prompt Y64/Y32/PCA512 与全部日志。
- 机器可读聚合：`outputs/state_tokenizer/metrics/v3/fixed_value/aggregate-fixed-value.json`；简表：`outputs/state_tokenizer/metrics/v3/fixed_value/fixed-value-summary.csv`。

---

## 2026-08-03：v3.2 Fixed-Prompt Token Hidden States 精确 Value

### 协议与实现

- 将 fixed observation prompt 的 29 个 contextual hidden states 作为独立压缩表示，使用与 v3.1 完全相同的 963-state manifest、590/133/240 split、单-query reader、value-only loss、seeds 0/1/2 和 Full-H oracle。
- 新增 `--instruction-only-cache` 紧凑提取模式：Qwen 前向和 layer-16 token 数值不变，但只落盘 29 个 prompt token hidden states，不保存 image/DOM tokens 或重复的 Y64/Y32。
- 4-state 冒烟测试和 963-state 正式缓存审计均通过；每个状态恰好 29×4096 BF16，三种 modality 长度记为 `[0,0,29]`，现有 `RaggedFullHStore`/instruction-only reader 可直接读取。
- 相关测试增至 21 项并全部通过；脚本通过 `py_compile`。

### 提取与压缩规模

- 双卡提取完成 963/963，GPU 0/1 分别处理 482/481 状态，墙钟 97.17/96.53 秒，速率 4.960/4.983 states/s。
- 共缓存 27,927 个 token，逻辑 BF16 大小 0.213 GiB，即每状态 232 KiB。
- 同一 value subset 的 Full-H 平均为 484.30 token/状态、约 3.784 MiB/状态；29-token 表示减少 94.01% token/字节，压缩率约 16.70×。它也比 Y64 的 512 KiB/状态更小约 2.21×。

### 精确随机 Value 结果

| 表示 | 五位置平均 accuracy | 五位整串 accuracy | 保守 `R_position` | 理论 chance 校正 `R_position` | `R_exact` |
|---|---:|---:|---:|---:|---:|
| Full-H | 0.8917 ± 0.0242 | 0.6569 ± 0.0428 | 1.0000 | 1.0000 | 1.0000 |
| task+step baseline | 0.1819 ± 0.0005 | 0.0000 | baseline | — | baseline |
| fixed-prompt 29-token hidden states | **0.1625 ± 0.0051** | **0.0000 ± 0.0000** | **0.0000** | **0.0789** | **0.0000** |

- 三个 seed 的逐位 accuracy 分别为 0.1658/0.1650/0.1567；五位整串全对均为 0。
- 该表示比 Full-H 绝对下降 0.7292 个逐位 accuracy、下降 0.6569 个整串 accuracy。其逐位结果低于 task+step metadata baseline，因此保守 baseline-adjusted raw retention 为 -0.0274，按“无可检测增益”截断为 0；只减理论 digit chance=0.1 时 retention 也仅为 0.0789，即损失 92.11%。
- 这与 v3 动态状态结果形成清楚对照：同样 29 个 contextual prompt tokens 能很好表达 checked/focused/nonempty 等粗粒度属性，却几乎不保留具体五位随机文本。动态 probe 的高分不能外推为状态表示在精确内容上信息充分。
- 因此 fixed-prompt token hidden states 不适合作为需要精确 value 的 State Tokenizer；Y64/Y32/PCA512 虽有明显损失，但仍远强于该 29-token 摘要。

### 审计与清理

- 三份 instruction-only result JSON 的 fixed-prompt protocol、seeds、590/133/240 样本数、reader 配置和 value-only objective 均一致；总结果集现为 21 份 JSON。
- instruction-only 提取和 probe 日志未发现 traceback、OOM、RuntimeError、ValueError、AssertionError 或 NaN。
- 已删除约 218 MiB 正式紧凑缓存、4-state 冒烟目录和 3 个新增 checkpoint；保留三份 result JSON、更新后的聚合 JSON/CSV 与日志。
