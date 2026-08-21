# WorldMemArena tokenizer-RAG 交接文档（换账号接手的单一入口）

> 更新：2026-08-22。本文件自足汇总两条实验线的现状、产物与下一步。
> 权威协议：`QWEN35_INSTRUCT_WORLDMEMARENA.md`（WMA-RAG 轨道）。
> 运行记录：`RECOVERY_WORKLOG_20260820.md`（8-20 恢复 + 8-21 全天重建）。
> WM 线权威结论：`WORLD_MODEL_WORKLOG.md` §0 / §25。

---

## 0. 一段话现状

v8/v9 数据与 tokenizer 链已在 8-21 全部重建完成（100,008 states 重采、Full-H、
Key64、20k-balanced PCA、A1、Xbar retrieval head），域内门禁全部通过；但
WorldMemArena 跨域 smoke 显示 **Xbar 的 observation row 在 100 个 top-10 槽位中
命中 0 条**（Raw 命中 8 条），14-way Recall@1≈随机。跨域压缩门禁失败，全量
web benchmark 已冻结，等 P1 偏移归因。两仓库有大量未提交改动（P0，先做）。

---

## 1. 项目全局一页纸

### 1.1 ResidualMem 是什么

世界模型互补残差记忆，四个核心假设（技术报告 §3.3）：

- **H1 互补性**：只存 WM 无法恢复的修正信息，同任务性能下记忆更省；
- **H2 能力-容量**：WM 越强，固定任务质量所需外部记忆越少；
- **H3 效用**：预测误差不足以决定记忆价值，需任务效用过滤噪声；
- **H4 可变码率**：部分可预测的观察，保存少量修正码优于全存或全丢。

架构六模块：冻结 Qwen3.5 观察编码器 → 状态 tokenizer → 概率 latent WM →
条件残差 codec → 率失真 mask/效用门控 → 外部残差记忆 + 修正解码器。
核心公式：`Reconstructed World = WM Prior + Residual Memory`（条件解码，非向量加）。

### 1.2 两条实验线

| 线 | 目标 | 当前状态 |
|---|---|---|
| **WM 线**（v8 离散 World Model） | 冻结 A2 码上预测下一状态，bits/transition 越低越好 | **§25 正式化完成**：validation 7,100.76（3 seeds，last.pkl），赢 source 1,891.53，总压缩 2.32×。test 仍锁定 |
| **WMA-RAG 线**（当前主线） | 压缩表示 xbar/A2 经 retrieval head 对齐 Qwen3-VL，在 WorldMemArena 上对照 Raw-Fused | 域内通过；**跨域 smoke 失败**，待偏移归因 |

### 1.3 哪些文档结论已过时（勿引用，共 4 处）

| 过时文档 | 过时内容 | 以什么为准 |
|---|---|---|
| `技术报告_ResidualMem.md` 头部注记（2026-08-12） | "M1 失败、按协议停止、未进入消融" | `WORLD_MODEL_WORKLOG.md` §0/§25（第二轮 dev 诊断 + 正式化已推翻） |
| `refine-logs/EXPERIMENT_TRACKER.md` | M2-M4 NOT RUN、停在 M1 失败 | 同上 |
| `refine-logs/IMPLEMENTATION_STATUS.md` | 同上（停在 2026-08-12） | 同上 |
| `experiments/world_model/README.md` 的 "Current decision" | 同上（M1 failed 即停） | 同上（该 README 的 M0-M4 命令入口、cache 格式、测试说明仍有效） |

历史背景：第一轮 M1 full seed0 9,058.17 输给 source 8,990.72 后按预注册停止；
第二轮在 train 内部 80/20 上重做架构搜索（copy gate、容量×dropout 交互、数据
缩放曲线），最终 12L/4096 + dropout 0.3 正式化反超。refine-logs/ 是 8-14 数据
事故时的抢救稿，其数值已被 WORLD_MODEL_WORKLOG 完整吸收。

### 1.4 WM 线关键结论速览（详见 WM worklog §0 结案表、§25）

- **正式化**：validation 11,263 transitions，3 seeds `last.pkl`@300k，
  含全部 side-info **7,100.76** bits/transition（vs source 8,992.29，**赢 1,891.53**；
  vs 固定宽 16,448，**2.32×**）。预注册预测 7091–7106，实测偏低 51（数据增益被低估）。
- **噪声底 ~110 bit（σ≈56）**：同配置三次复现极差 110.13，源自训练期 GPU
  非确定性，**不可靠换估计量消除**，只能多种子取均值。任何单种子 <110 bit
  的差异不得作为结论。
- 已定案：copy gate **+434.76**（必留）、payload **+526.93**（最大单一信息源）、
  history 采纳（不计费）、source prior **−385.24 有害**、target_text 不用
  （账单 +60.81 确定性）、输出头/code dim 已关闭（上限 63.1 bit < 噪声底）。
- **数据是最大杠杆**：缩放曲线右端边际 +297.42（source 对照的 4.02×），
  幂律在实测范围内测不到饱和。
- **test 锁定**：`freeze.py` 硬性要求 5 变体 × 3 种子 = 15 个正式 run 才能解锁；
  目前只有 `full` 的 3 seeds。validation 已被观测 8 次（FINDINGS §9），
  是 replication split，**唯一干净终点是 test**。
- **§25.6 曾否决扩数据**（理由：渲染栈无法复现，截图 5–16% 像素差异）
  ——**该否决已被 8-21 重采推翻**：本机就是原采集环境（Ubuntu 24.04 冻结栈），
  lane 并行 + browser reuse 重采通过 6,792 条重叠审计（6,770 截图逐字节相同，
  22 张仅 scrollbar raster 波动 <0.07%）。扩数据/补变体解锁 test 的杠杆重新打开。

---

## 2. WMA-RAG 检索协议（速查）

```text
raw full round text ─────────────── Qwen3-VL document encoder ─┐
                                                              ├─ global cosine top-10
screenshot + user + caption ─┬─ Raw-Fused Qwen3-VL encoder ───┤
                              └─ v9 tokenizer -> Xbar/A2 -> head ┘
query ─────────────────────────── official Qwen3-VL query encoder
```

- 每个非空 round 恰好两行；**assistant 的观察/计划/Action JSON 绝不进入
  observation/latent 行**（它们是 policy 输出，不是 `x_t` 输入）。
- 空观察 round 仅一行（full-round text）。检索不做 round dedup，同一 round
  两行可能同时命中。
- caption 由 loader 内联到 user text，适配器必须先移除副本，保证 caption 在
  observation 中只出现一次。
- WMA 无 BrowserGym AXTree：零训练适配把 user 文本 + caption 序列化为合法
  synthetic AXTree；截图走视觉 token。
- 所有方法共用官方 query encoder、row-level cosine、top_k=10。
- **总体 Recall@10 被共有 full-round text row 掩盖**（overlap 0.91 是假象），
  判据必须看 observation row 命中与 paired cosine。

实现位置：

- ResidualMem observation serializer：`residualmem/benchmarks/worldmemarena_tokenizer.py`
- frozen runtime：`residualmem/latent/frozen_v8_runtime.py`
- retrieval/reader bridge：`residualmem/latent/instruct_bridge.py`
- WMA raw encoder：`eval_framework/memory_adapters/qwen_embed_adapter.py`
- WMA Residual adapter：`eval_framework/memory_adapters/residualmem_instruct_adapter.py`

四个预注册主实验：`ResidualMem-Instruct-{Xbar,A2}-{Input,L16}-RAG`（共享
retrieval head 相同，reader 路径不同；当前只有 Xbar-Input 的检索侧跑通）。

---

## 3. 状态表（2026-08-22）

| 阶段 | 状态 | 关键数字 |
|---|---|---|
| 环境恢复（.venv-jax / browsergym-venv / qwen-vl 联合环境） | ✅ | JAX 0.4.33 + ptxas 12.8.93（Blackwell sm_120 兼容） |
| 权重落盘（Qwen3.5-9B / Qwen3-VL-Embedding-8B） | ✅ residual-mem 侧 | WMA 侧 `eval_framework/baselines/.../weights/` 下载中 |
| v8 数据重采（lane 并行 + browser reuse） | ✅ | 100,008 states / 43,751+ episodes；重叠审计通过 |
| v9 Full-H 抽取（6 ranks） | ✅ | 100,008/100,008；~420 GiB |
| Static Key64 + PCA + normalization | ✅ | 20k task-balanced PCA，解释方差 0.9186 |
| A1（64×512 连续瓶颈） | ✅ | validation R² 0.99868 |
| A2（M=32, C=256） | ❌ 作废 | 绑定纠错前的 first-N PCA，未与新 PCA 混用；P2 重训 |
| teacher cache（fused observation） | ✅ | 5,000 train + 500 val，518 MiB |
| retrieval head（Xbar） | ✅ | 域内 500-way R@1/5/10 = 0.840/0.956/0.986，MRR 0.8946 |
| WMA 跨域 smoke（web_01 final ckpt） | ❌ 门禁失败 | obs rows 0/100 槽位（Raw 8）；14-way R@1 0.0714≈随机 |
| reader connector（Input/L16） | ⏳ 未训 | 检索侧通过后做（P4） |
| 全量 web benchmark | 🧊 冻结 | 修正域适配前不启动 |
| 两仓库 commit + push gitee | ⏳ P0 | residual-mem ~20 文件；WorldMemArena 6+新文件 |

---

## 4. 产物对照表

### 4.1 数据与特征（residual-mem）

| 产物 | 路径 | 说明 |
|---|---|---|
| canonical v8 | `outputs/state_tokenizer/v8` | → 指向 `v8-recovered` 的符号链接 |
| 重采 lanes | `outputs/state_tokenizer/v8-lanes` | 228 lanes，128,364 候选裁到 100,008 |
| partial reference | `outputs/state_tokenizer/v8-reference-partial-20260821` | 旧 12-worker 中断数据，仅审计用 |
| 划分清单 | `full-721.jsonl` / `records-merged.jsonl` | 7:2:1 = 70,018/20,011/9,979；merged manifest SHA256 `c61ee90c…65865` |
| Full-H | `outputs/state_tokenizer/v9-instruct/full-h`（6 shards） | ~420 GiB |
| Static PCA 特征 | `…/v9-instruct-pca20k-balanced/static_features` | ~55 GiB |
| **正式 PCA** | `…/v9-instruct-pca20k-balanced/key64-static-pca.npz` | SHA256 `f98c517c…f6efb2a`；12 task × 1,666–1,667 train states |
| ⚠️ 作废 PCA | `…/v9-instruct-pca2000/`（2,000-state） | first-N 全部来自 click-button，**禁止作为坐标**，仅审计保留 |
| normalization | `…/key64-static-pca-normalization.npz` | SHA256 `648135a1…7e57` |
| A1 | `outputs/a1/v9-instruct{,.npz}` | R² 0.99868；绑定新 PCA hash |
| ⚠️ 作废 A2 | `outputs/a2/…`（8-21 训练） | 绑定旧 PCA 坐标，不入 cache、不入 loss |

### 4.2 bridge / retrieval

| 产物 | 路径 |
|---|---|
| teacher cache | `outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/train-cache-fused-observation.npz`（xbar + Qwen3-VL fused teacher，5,500 条） |
| retrieval head | `…/retrieval-head-fused-observation.pt`（best step 5,000） |
| WMA smoke 逐问报告 | `…/worldmemarena-web01-{checkpoint0,final-checkpoint}-10q.json` |

### 4.3 WM 线（v8）

| 产物 | 路径 |
|---|---|
| 冻结 cache | `outputs/world_model/v8/cache/`（codes/valid/transitions/manifest，SHA256 齐全） |
| 统计基线 | `outputs/world_model/v8/baselines/validation/baseline.json` |
| M1 诊断 | `outputs/world_model/v8/diagnostics/m1_seed0.json`、`figures/m1/` |
| 第二轮 dev | `outputs/world_model/v8_dev/`（不进正式统计） |
| §25 正式化 | 3 seeds × 300k，`last.pkl`（详见 WM worklog §25；v8_frozen.yaml 头注含预注册预测） |

---

## 5. 关键数字表

### 5.1 WM 线（bits/transition，validation，含全部 side-info）

| 系统 | 数值 |
|---|---:|
| 固定宽保存 | 16,448.00 |
| task marginal | 10,636.92 |
| copy-aware | 9,566.16 |
| source-conditioned Markov | 8,992.29 |
| M1 full（已废弃架构） | 9,115.65 |
| **§25 正式化（3 seeds, last.pkl）** | **7,100.76** |

单种子噪声 σ≈56；数据缩放右端边际 +297.42；补全部 train 免费 +196（已在 §25 兑现）。

### 5.2 tokenizer（v8 门控 / v9 域内）

| 项 | 数值 |
|---|---:|
| 目标字面进 raw 槽（v7→v8） | 0.7214 → 0.9578 |
| 绑定探针（对照校正，v7→v8） | 0.416 → 0.646 |
| A2 value exact | 0.9357 |
| v8 A1 R² / v9 A1 R² | 0.99891 / 0.99868 |
| v8 A2 R²（12 任务域内，未收敛） | 0.8511 |
| A2 码率 | 16,448 bit/state（2,048 codes × 8 bit + 64 mask） |

### 5.3 WMA 跨域 smoke（web_01 final，25 sessions / 49 full-round rows / 14 obs / 10 questions）

| 指标 | Raw-Fused | Xbar |
|---|---:|---:|
| overall evidence Recall@10 | 0.765 | 0.725 |
| NDCG@10 | 0.5429 | 0.5529 |
| top-10 obs rows（100 槽位） | **8** | **0** |
| 14-state paired cosine | — | 0.1836 |
| 14-way Recall@1/5/10 | — | 0.0714/0.5714/0.8571（随机 1/14≈0.071） |
| MRR | — | 0.2958 |

**读法**：overall 接近是 49 条共有 full-round text row 的掩盖效应；
Xbar 的压缩表示在 WMA observation 上跨域失效。

---

## 6. 环境与入口（当前服务器 `/home/luzheng/workspace/iclr/czs/`）

| 环境 | 位置 / 入口 | 用途 |
|---|---|---|
| JAX | `residual-mem/.venv-jax`（jax 0.4.33 + ptxas 12.8.93） | A1/A2、WM、bridge 训练 |
| 采集 | `residual-mem/browsergym-venv` | BrowserGym/Playwright（Ubuntu 24.04 冻结栈，勿在 22.04 跑） |
| torch | qwen-vl 联合环境 | Qwen 抽取、Qwen3-VL teacher |
| 权重 | `residual-mem/models/Qwen3.5-9B`、`models/Qwen3-VL-Embedding-8B` | 均已落盘并过前向门禁 |
| 阶段入口 | `residual-mem/scripts_v9_instruct.sh <stage>` | modality-lengths / extract-full-h / rebuild / pca / normalization / a1 / a2 / bridge-cache / retrieval-head / wma-smoke |

⚠️ 根 README 与 `experiments/world_model/README.md` 里的旧 NAS 路径
（`/root/nas/...`、`/mnt/data/users/...`）**全部失效**，以本节与本机路径为准。

运行前置（采集）：

```sh
R=/home/luzheng/workspace/iclr/czs/residual-mem
export PLAYWRIGHT_BROWSERS_PATH=$R/browsergym-venv/browsers
export LD_LIBRARY_PATH=$R/browsergym-venv/syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
```

运行前置（JAX）：`export PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONFAULTHANDLER=1`

测试：**没有 pytest**，用 `tests/run_tests.py` 替身（支持 parametrize/raises/approx）。

---

## 7. 已知坑（浓缩版；完整版见各 worklog）

### 7.1 工程坑（根 README）

- `extract_qwen` / `extract_fixed_prompt` 必须 `--no-use-kernels`（transformers 5.8.1）。
- `rebuild_static_key64` 必须 `--image-grid-thw 1 20 32`（498×321 截图；默认值是 v7 旧网格，用错**静默**错位池化）。
- `--instruction-records` 必须传**采集清单**（特征清单里的 instruction 是固定观察提示，传错则指令优先排序静默失效）。
- A1 的 `--num-e-tokens` 必须与 A2 匹配（都用 64）；A2 用 `--init-a1-checkpoint`。
- JAX 长任务 `PYTHONFAULTHANDLER=1` 不要省（XLA 段错误无 Python 报错）；A1/A2 运行中日志 0 字节正常，用 `ps -o etime,pcpu` 判断。
- 采集 `--num-workers` 不得为 5 的倍数（episode 分桶会塌缩）。
- `pkill -f` 会匹配发起 shell 自身；等待进程用 `while ps aux | grep -q "[p]attern"`；**启动脚本里不要做基于模式的清理**。
- 会话容器重建清 `/tmp` 与 `~/.cache`：日志写仓库、长任务 `--resume`、`setsid nohup` 也扛不住重建。
- 跨环境共享常量只能放无第三方依赖模块（`slot_layout.py` 是唯一实例）。

### 7.2 量具坑（STATE_TOKENIZER_WORKLOG §9，对 P1 直接相关）

- `key64-static-detail-ranges` 是 **token 索引**不是字符偏移，须先 `dom_token_offsets` 换算。
- raw literal 槽**逐 token 发槽**：多 token 目标须按 span 合并掩码再判存在性。
- 探针的 `literal_ablated` 对照必须**重新训练**，不能对已训探针置零输入。
- `dom.find` 判存在须遍历所有出现位置（子串先命中别处）。
- 绑定量具只适用 `click-checkboxes` / `click-option`（目标提取有语法边界）。
- MAX_REF 硬编码曾静默丢弃 48.3% 候选——任何按范围硬编码的量具先查截断率。

### 7.3 协议坑（WMA-RAG）

- assistant plan/action 禁入 tokenizer（§2）；caption 去重；两行建库不做 dedup。
- A2 的 2,048 个 code **全部计费**（与 valid mask 无关）——WM 线一切码率数字的前提。
- PCA/normalization 只在 train split 拟合；追加数据必须复用旧坐标，否则全部码字与历史数字作废。
- 旧 A2 checkpoint（任何绑定旧 PCA 的产物）不得混入新 cache。
- WMA 总体检索指标被 full-round text row 掩盖，判据只认 observation row / paired cosine。
- `transitions = states − episodes`（100,008 − 43,751 = 56,257）；构造转移必须按 `(episode_id, step+1)` 存在 join，不能只看 action 非空。
- `global_index` = 合并清单行号：新增数据若排序插到中间，55G 特征库整体错位（追加须让新 id 排在最后）。
- 数据事故教训（WM worklog §14）：代码推 Gitee 是唯一救回代码的机制；`outputs/` 被 gitignore，结果 JSON 靠 `write_dev_json` 镜像到同步范围外。

---

## 8. 接下来要做什么

**P0（先做）：提交 + 推送两仓库到 gitee。**
- `residual-mem`：全部修改文件 + 本交接文档；push `git@gitee.com:feld-ceng/residual-mem.git` main。
- `WorldMemArena`：6 修改 + 新文件（residualmem_instruct_adapter、tests、README_DATASET/README_ResidualMem）；**排除** `eval_framework/baselines/Qwen3-VL-Embedding-8B/weights/`（加 .gitignore）；新建 gitee 私有仓库推送。

**P1 偏移归因诊断（零训练，最优先实验）。**
在 14 个 WMA observation 上做槽位分解：Xbar→head 输出与 teacher 的逐 slot 余弦、
image/detail/context 组贡献；按 handover §9.8 分桶（有/无截图、caption 长度、session 长度）；
用同分辨率重采样截图做对照，消除分辨率混杂（WMA 1280×720 vs 训练域 498×321）。
产出：偏移主要来自视觉域还是 synthetic-AXTree 文本域的定量结论 → 决定 P3 走 (a) 还是 (b)。

**P2 重训 A2 绑定新 PCA。**
`./scripts_v9_instruct.sh a2`（70k 步、K-means 全量 70,018 host-latent、64 问题/GPU 批）。
完成后 A2-Input/A2-L16 两个主实验才有合法表示。

**P3 域适配路线（依 P1 结果二选一或并行）。**
(a) 零训练改进：WMA 序列化/分辨率对齐、synthetic-AXTree 规则修正；
(b) 扩大训练域多样性（多分辨率重采、更多任务模板），需重走 PCA→A1→A2 链。

**P4 reader connector（Input/L16）重训。**
native answer 路径缺 connector；检索侧（P1-P3）通过后做。定义见协议 §5。

**P5（可选、独立轨道）WM 线续作。**
重采可复现性已验证（§1.4），两个方向：扩数据（幂律外推 2×≈+599，但新 episode
须与旧同分布——lane 协议保证）；或补 5 变体 × 3 种子解锁 test（唯一干净终点）。

**附：tokenizer 侧遗留待办**（STATE_TOKENIZER_WORKLOG §10，非紧急）：
A1 恒等映射质疑（64×512→64×512 近恒等，A1 可能是死重，建议 A2-without-A1 对照）；
AXTree-vs-DOM 同环境受控对照（`dom_control` 采集时已存，只差一次抽取）。

---

## 9. 文档导航（10 个项目 .md 的定位与可信度）

| 文件 | 定位 | 可信度 |
|---|---|---|
| `WORLDMEMARENA_TOKENIZER_RAG.md` | 本交接文档 | 以本版为准 |
| `WORLD_MODEL_WORKLOG.md` | WM 线权威（§0 结案表、§25 正式化；§22.2 噪声底；§23.7 秩筛查） | ✅ 权威 |
| `STATE_TOKENIZER_WORKLOG.md` | tokenizer 训练史（§9 量具失效、§10 待办） | ✅ 权威 |
| `TOKENIZER_WM_HANDOVER.md` | 跨 benchmark 移植手册：数据契约、cache 格式、坑表 §7、**§9 WMA-RAG 协议、§10 v8 重采/恢复记录**（两份旧文档已并入删除） | ✅ 权威 |
| `技术报告_ResidualMem.md` | 愿景与系统设计（H1-H4、公式、benchmark、风险 §18） | ⚠️ 头部 8-12 注记过时 |
| `README.md` | 方法概览 + 环境与工程坑（旧 NAS 路径已失效） | ⚠️ 路径过时，坑有效 |
| `refine-logs/EXPERIMENT_TRACKER.md` | M0-M4 状态 | ❌ 停在 M1，已被 §25 取代 |
| `refine-logs/IMPLEMENTATION_STATUS.md` | 同上 | ❌ 同上 |
| `refine-logs/FINDINGS_20260814.md` | 8-14 抢救稿（A1 分解/A2 residual/A3 输出头/§8 正式约束/§9 窥视登记） | ⚠️ 数值已被 WM worklog 吸收 |
| `experiments/world_model/README.md` | WM 命令入口与 cache 格式 | ⚠️ "Current decision" 过时，命令仍有效 |

WorldMemArena 仓库侧：`README_DATASET.md`、`README_ResidualMem.md`（8-21 新建）、
`eval_framework/memory_adapters/` 下 adapter 与测试。
