# LongMemEval 复现手册（ResidualMem 闭环门控 / e·V gate）

> 更新：**2026-09-19**。面向外部使用者：克隆官方仓库、下载 LongMemEval 数据与公共模型，
> 再从 Hugging Face 拉取我们的产物，用官方评测接口完成复现。
> 产物仓库：**https://huggingface.co/feldmatthew/WMA-ResidualMem**（`longmemeval/` 目录，
> 全部为 LongMemEval 评测产物；WMA 等其他 benchmark 的产物不在其中）。

## 1. 环境与数据

### 1.1 环境

```bash
git clone <本仓库> residual-mem && cd residual-mem
conda env create -f environment.yml        # 或 pip install -r requirements.txt
```

纯评测只需上述主环境；涉及 jax 的闭环/门控拟合脚本另需 `residual-mem/.venv-jax`
（本手册的评测步骤用不到）。

### 1.2 公共模型（不在产物仓库内）

| 用途 | 模型 | 放置路径 |
|---|---|---|
| Reader / Judge | `Qwen/Qwen3.5-9B` | `models/Qwen3.5-9B/` |
| 文本检索编码器 | `Qwen/Qwen3-Embedding-8B` | `models/Qwen3-Embedding-8B/` |

### 1.3 LongMemEval-V2 数据（web，small tier，240 题）

数据来自官方 benchmark 仓库与官方 HF 数据集（无需任何自制数据）：

```bash
git clone https://github.com/xiaowu0162/LongMemEval-V2.git residual-mem/LongMemEval-V2
cd LongMemEval-V2/data && python download_data.py   # 拉取 HF 数据集 xiaowu0162/longmemeval-v2（~6GB）
```

下载完成后按下述布局就位（评测只读这三个文件，截图图像不需要——本管线消费文本观察）：

```
LongMemEval-V2/data/longmemeval-v2/
├── questions.jsonl                    # 问题（240 道 web 题）
├── haystacks/lme_v2_small.json        # haystack 定义
└── web_small_trajectories.jsonl       # web 观察文本（synthetic axtree）
```

以上即官方 `download_data.py` 的默认输出布局，无需手工整理。

**重要**：这个 checkout 不只提供数据——评测判分直接加载官方 scorer
（`LongMemEval-V2/evaluation/qa_eval_metrics.py`，通过 `run_longmemeval_local.py` 的
`_load_eval_function()` 以官方实现为 source of truth）。harness 在 `residual-mem/LongMemEval-V2`
（兄弟目录）或工作区上一级搜索该 checkout，按上面命令 clone 的位置正好满足。
已验证版本：commit `2cc8c54`（2026-08-09）。

## 2. 下载我们的产物并摆放（一条命令）

从 HF 下载 `longmemeval/setup_artifacts.sh`，在仓库根目录执行——自动下载全部产物
并按官方 harness 期望的相对路径摆放（cache 位置与 gated index 的 `metadata.cache`
严格对齐，abscontract runner 落在仓库根目录）：

```bash
curl -O https://huggingface.co/feldmatthew/WMA-ResidualMem/resolve/main/longmemeval/setup_artifacts.sh
bash setup_artifacts.sh          # 需在 residual-mem 仓库根目录执行；约 1.2GB
```

摆放结果（脚本内容与下表一一对应，可供核对）：

| HF 路径 (`longmemeval/…`) | 仓库内路径 |
|---|---|
| `caches/full-send/` | `.lme-cache/web-small-latents-K32-v5-query-b1/` |
| `caches/gated-ev185/` | `.lme-cache/wm-v1-latents-K32-v5-query-b1-gated-ev185/` |
| `retrieval/text-index-v2-compact.npz` | `outputs/longmemeval/retrieval/` |
| `retrieval/text-index-v2-compact-gated-ev185.npz` | `outputs/longmemeval/retrieval/` |
| `ev-gate/{vectors.npz,calibration.json}` 与 `ev-gate/recons/` | `outputs/longmemeval/wm-v1/gate-lme/ev-v1/`（仅重新施加门控/闭环时需要，纯评测不需要） |
| `world-model/wm-full-seed0/{best.pkl,resolved.yaml}` | `outputs/longmemeval/wm-v1/wm-full-seed0/`（仅闭环需要） |
| `world-model/pq-C64-M32-shared.npz` | `outputs/longmemeval/wm-v1/`（仅闭环需要） |
| `qformer/qformer-K32-webchain-v5-obs0.5.gapbest.pt` | `outputs/longmemeval/qformer/`（仅闭环需要） |

## 3. 评测（官方接口）

两个 cache 各自独立可评；命令完全相同，只换 `LME_CACHE`/`LME_TEXT_INDEX`。

```bash
PY=/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python   # 按实际环境调整

# 1) judge ×2（GPU 0→8023、GPU 2→8024；与评测分片不同卡，单 judge 约 37GB 显存）
mkdir -p outputs/longmemeval/eval/logs
env -u CUDA_VISIBLE_DEVICES OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false "$PY" -B \
  -m experiments.state_tokenizer.local_openai_server --model models/Qwen3.5-9B \
  --gpus 0 --port 8023 --max-batch 8 --max-new-tokens-cap 1536 \
  > outputs/longmemeval/eval/logs/judge-8023.log 2>&1 &
# 第二个 judge：--gpus 2 --port 8024，日志 judge-8024.log
# 就绪判据：curl --fail http://127.0.0.1:8023/v1/models 与 8024 均 200

# 2) 四分片评测（GPU 4,5,6,7；gated cache + abscontract prompt = 42.5% 主结果。
#    LME_READER_SYSTEM_PROMPT 是 launcher 原生支持的 prompt 选择，其余与 stock 发车一致）
RUN=gated-ev185-abs; OUT=outputs/longmemeval/eval/$RUN; LOG=outputs/longmemeval/eval/logs-$RUN
mkdir -p "$OUT" "$LOG"
env -u CUDA_VISIBLE_DEVICES \
  LME_PYTHON="$PY" LME_READER_SYSTEM_PROMPT=abscontract LME_GPUS=4,5,6,7 \
  LME_OUTPUT="$OUT" LME_LOGS="$LOG" \
  LME_CACHE=.lme-cache/wm-v1-latents-K32-v5-query-b1-gated-ev185 \
  LME_TEXT_INDEX=outputs/longmemeval/retrieval/text-index-v2-compact-gated-ev185.npz \
  LME_TEXT_MODEL=models/Qwen3-Embedding-8B \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  bash experiments/state_tokenizer/run_longmemeval_latent_text_rag_eval.sh \
  > "$LOG/driver.log" 2>&1 &

# 3) 结果核查（四片各 60/60，merge 由 launcher 自动完成）
wc -l "$OUT"/results-r[0-3].jsonl && cat "$OUT"/results.summary.json
```

- `LME_READER_SYSTEM_PROMPT` 不设（默认 `longmemeval`）即 stock prompt 路径：gated cache
  复现 89 (37.1%)，full-send 基线（`LME_CACHE=.lme-cache/web-small-latents-K32-v5-query-b1`、
  `LME_TEXT_INDEX=outputs/longmemeval/retrieval/text-index-v2-compact.npz`）复现 96 (40.0%)。
- abscontract 是仓库原生注册的 prompt 变体（`experiments/state_tokenizer/prompts.py`，
  经 `longmemeval_reader.SYSTEM_PROMPTS` 暴露给 `--reader-system-prompt`），字节级 sha256
  `f0fc164a…abf778` 由 `tests/residualmem/test_longmemeval.py` 钉死；每个分片
  `results-rN.config.json` 的 `reader_system_prompt_sha256` 可与之核对。
- 检索配置由脚本固定（`--retrieval-mode text --top-k 6 --no-site-filter --latent-payload
  --text-memory-layout state-snapshots --anchor`）；同一问题集上两种 cache 的检索 hits 完全一致。
- 生成参数：thinking 开启、temperature 0.6、top_p 0.95、top_k 20、max_new_tokens 20000
  （未固定 seed，单次采样）。

## 4. 预期结果

| 配置 | 码率 (bits/状态) | QA 正确（240 题） |
|---|---:|---:|
| **gated-ev185 + abscontract（主结果，§3 命令）** | **2527** | **102 (42.5%)** |
| gated-ev185 + stock prompt | 2527 | 89 (37.1%) |
| full-send 基线 + stock prompt | 3915 | 96 (40.0%) |

- 门控产物 `ev-gate/vectors.npz` + `calibration.json` 的语义：`|U_j| = ē_j · V_j` 精确分解；
  状态相关错误率 `e_{t,j}` 由解码端后验熵查冻结校准表得到；keep 规则 `e_{t,j}·V_j ≥ λ·H_{t,j}`，
  λ=1.85e-04。abscontract 相对 stock 的净增全部来自该拒答前提的题（-abs 题 24/72，
  stock 两臂均为 10-12/72）。
- 复现性：未固定 seed，重跑波动 ±6 题以内；差异在此范围内不下结论，用配对 McNemar。

## 5. 注意事项

- **index–cache 绑定**：`text-index-v2-compact-gated-ev185.npz` 的 `metadata.cache` 指向
  `.lme-cache/wm-v1-latents-K32-v5-query-b1-gated-ev185`；布局校验要求二者一致，按 §2 表摆放即可。
- **码率口径**：按状态计（每状态存一次，与后续问题数无关），不按问题平均。
- **judge 显存隔离**：judge 与评测分片不可同卡；评测进行中不要关闭 judge 端口。
- **中断续跑**：launcher 内置 `--resume`；先确认原进程退出，再用同一输出目录与配置续跑。
