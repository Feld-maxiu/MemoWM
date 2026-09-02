# ResidualMem

ResidualMem 是一个面向 GUI/Web 与交互轨迹的外部记忆方法原型。当前主线使用冻结的 Qwen3.5 中间层状态、Static Key64、PCA512、固定 group×channel normalization，以及地址/内容解耦的 Slot-key routing。

完整实验结论见：

- `WMA_RESIDUAL_复现手册.md`（**从这里开始**：整条链路怎么跑、产物在哪）
- `技术报告_ResidualMem.md`（方法、全部实测、必须声明的事项）
- `UTILITY_GATE.md`（效用门控与闭环）
- `QFORMER_实验手册.md` / `WM_MIXED_复现手册.md`（分段细节与坑）

`WORLD_MODEL_WORKLOG.md`（v8/MiniWoB 线，语料已废弃）与 `WMA_RAG_WORKLOG.md`
（检索线，2026-08-22 冻结）已于 2026-09-02 移除，内容见
`git show d91aaa3:WORLD_MODEL_WORKLOG.md`。

v8 离散 WM 已实现为独立的 `experiments/world_model/` 实验线；正式入口、
计费不变量与 M0--M4 命令见 `experiments/world_model/README.md`。M0 已完成；
M1 的 128-transition 过拟合门通过，但 full seed 0（9,058.17 bits/transition）
未打败 source-conditioned Markov（8,990.72），因此按预注册协议停止 M2--M4。
test 未打开，且不存在 test freeze manifest。

两个入口不要混用：

- `configs/world_model/v8_discrete.yaml`：当前 v8 离散预测证据门；
- `configs/residualmem_web_static_pca.yaml`：v4/旧 layout RSSM legacy 配置，
  不得作为 v8 运行入口。

## 当前方法状态

**冻结状态：v8 / AXTree / `(32,16,16,0)` / 10 万状态。** 该状态属于已作废的固定池化路线；其过程与判据随 `STATE_TOKENIZER_WORKLOG.md` 于 2026-08-27 移出，全文见 `git show aa9e5a7:STATE_TOKENIZER_WORKLOG.md`。当前路线见 `QFORMER_实验手册.md`。

Qwen 的输入是**三个模态**：

```text
screenshot 498×321  +  AXTree→compact 线格式  +  固定观察提示
→ Qwen3.5 layer-16 Full-H
→ Static Key64 (64×4096)
→ PCA512 (64×512, 仅 train 拟合)
→ train-only group×channel normalization
```

Key64 布局：

| Group | Slots | Method |
|---|---:|---|
| Screenshot | 32 | 按真实 merged vision grid 二维平均池化（`10×16`） |
| DOM detail | **16** | 精确 UI literal，组内按指令优先排序 |
| DOM context | 16 | full-node pooling |
| Fixed prompt | **0** | 已回收给 detail |

符号：`x_t` 原始 Static PCA 状态；`xbar_t` 固定标准化状态；`e_t` A1 连续 latent tokens；`y_t ∈ {0..255}^{64×32}` A2 离散码；**`z_t` 保留给后续 RSSM 的 stochastic latent，不得混用**。

### v8 与 v7 的差别

从 compact DOM 换成 BrowserGym AXTree，因为 v7 的绑定能力撞到了结构性上界（`ref`/`parent` 按设计不进 raw detail）。AXTree 把绑定从「跨节点引用」变成「字面相邻」：`checkbox 'Nb'` 是同一行。

门控结果（判据：字面存在性与绑定**两者都必须改善**）：

| | v7 (DOM) | v8 (AXTree) |
|---|---:|---:|
| 目标字面进 raw 槽 | 0.7214 | **0.9578** |
| 绑定（对照校正后） | 0.416 | **0.646** |
| value exact（A2 重建） | 0.7890 | **0.9357** |

绑定的 raw 数字 v8 反而略低（0.7476 vs 0.7935），但 v7 的探针**不给字面输入也能答对 64.6%**——那部分不是绑定能力。v8 去掉字面掉到 0.2877（近随机 0.1103）。

### A1 / A2 现状

| | 配置 | 压缩 | validation R² |
|---|---|---:|---:|
| A1（在用） | 64 e-token × 512 | **1.0×** | 0.99891 |
| A2 | M=32, C=256, `--group-weights 1,2,0.5,0.5` | — | 0.8511 |

**当前 A1 不是瓶颈**——同维自编码器，R²=0.9989 即恒等映射；全部压缩由 A2 量化承担（16,448 bit/state）。A1 率失真曲线（见 WM 日志 §6）显示 64→32 时 detail R² 从 0.998 崩到 0.626，而 image 只掉 0.035，故不建议在 A1 做均匀压缩。

**A2 未收敛**：验证 MSE 单调降到最后一步，是被步数上限截断的，非能力上限。

A0 wiring 结论仍然成立：slot-only key 负责地址，raw normalized value 负责内容；content-dependent routing FAIL。

---

## 工作目录

```text
/root/nas/users/luzheng/workspace/ssh/czs/ResidualMem
```

进入仓库：

```sh
cd /root/nas/users/luzheng/workspace/ssh/czs/ResidualMem
```

所有命令都应从该目录运行，并设置：

```sh
export PYTHONPATH=.
```

---

## 实际使用的环境

**三个环境互不相容，且都不是自包含的。** 不要假设某一个能同时跑采集、抽取和 WM——它们分别只有 playwright、torch、jax，互相 import 会直接 `ModuleNotFoundError`。

| 用途 | 解释器 | 有什么 | 没有什么 |
|---|---|---|---|
| **采集** | `residual-mem/browsergym-venv/bin/python` | playwright、gymnasium、browsergym | torch、jax |
| **抽取 / 池化 / 探针** | `enter/envs/MemCompiler/bin/python3.12` | torch、transformers | jax、pytest |
| **A1 / A2 瓶颈** | `enter/envs/ResidualMem/bin/python3.11` | jax 0.4.33、optax | torch |

跨环境共享的常量必须放在无第三方依赖的模块里。当前唯一一个是 `experiments/state_tokenizer/slot_layout.py`（64 槽布局），`key_pooling` 再导出它。同理 `filler_vocabulary.py` 放在 tokenizer 侧而非 `residualmem/encoders/`，因为后者的 `__init__` 会传递引入 jax。

### 1. 采集（BrowserGym + Playwright）

```sh
R=/mnt/data/users/luzheng/workspace/iclr/czs/residual-mem
export PLAYWRIGHT_BROWSERS_PATH=$R/browsergym-venv/browsers
export LD_LIBRARY_PATH=$R/browsergym-venv/syslibs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export MINIWOB_URL="file://$R/third_party/miniwob-plusplus/miniwob/html/miniwob/"
export PYTHONPATH=$R
$R/browsergym-venv/bin/python -m experiments.state_tokenizer.collect_browsergym \
  --output outputs/state_tokenizer/v9 --target-states 1000 \
  --max-steps 7 --random-action-prob 0.5 --num-workers 6 --worker-id 0 --resume
```

**前两个 export 是必须的，不是优化。** 会话容器重建会清掉 `~/.cache/ms-playwright` 与 apt 装的浏览器库；Chromium 与所需动态库已固化到项目目录。缺了它们，`env.reset` 会永久失败而采集脚本**空转不报错**（已加 `--max-consecutive-failures` 拦截）。

当前宿主是 Ubuntu 22.04，动态库通过 `apt-get download` + `dpkg-deb -x` 解包到
`browsergym-venv/syslibs`，没有修改系统包。若迁移到 Ubuntu 24.04，对应包名会带
`t64` 后缀（如 `libatk1.0-0t64`、`libcups2t64`），不能原样沿用 22.04 包名。

`--num-workers` **不得为 5 的倍数**（脚本会拒绝）：episode 索引是 `worker_id + k·num_workers`，划分按 `index % 10` 分桶，5 的倍数会让整个任务落进单一划分。

生产采集用自愈脚本 `scripts_v8_collect.sh <worker_id>`，它按**退出码**而非清单文件判断完成。

> `miniwob-plusplus/.venv`（Selenium 路线）仅供 v7 及更早复现，v8 起不再使用。

### 2. Qwen 抽取与池化

```sh
MC=/root/nas/users/luzheng/workspace/enter/envs/MemCompiler/bin/python3.12
M=/root/nas/public_ckpt/Qwen3.5-9B          # layer 16，无跨状态 KV cache
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $MC -m experiments.state_tokenizer.extract_qwen \
  --records .../full-721.jsonl --model $M --output .../features \
  --rank 0 --world-size 8 --no-use-kernels
```

**`--no-use-kernels` 是必须的**：`kernels` 包未安装，transformers 5.8.1 会直接拒绝启动。

池化必须显式传图像网格：

```sh
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 $MC -m experiments.state_tokenizer.rebuild_static_key64 \
  --records .../full-721.jsonl --full-h .../full-h --model $M \
  --output .../static_features --device cuda:0 --rank 0 --world-size 8 \
  --image-grid-thw 1 20 32 --filter-filler \
  --instruction-records .../records-merged.jsonl
```

- `--image-grid-thw 1 20 32` 对应 BrowserGym 的 498×321 截图。默认值 `1 20 14` 是 v7 的 160×210，用错会让 32 个 image 槽错位池化（`rebuild` 的 `merged_image_hw` 校验是唯一拦得住它的地方）。
- `--instruction-records` 必须传**采集清单**。特征清单里的 `instruction` 是任务无关的固定观察提示，传错会让指令优先排序**静默失效**。
- **`--device cuda:0`，不要用 CPU**：CPU 0.39 states/s，GPU 72.6 states/s（185×）。10 万状态 7 小时 → 9 分钟。

抽取链的完整顺序：

```
collect → merge_records（写 global_index = 合并清单行号）
        → build_split_721（7:2:1 分层；--extracted 可省略）
        → fixed_prompt（换成固定观察提示）
        → extract_qwen → extract_fixed_prompt（产出 Full-H）
        → rebuild_static_key64 → key64_pca fit/transform → fit_normalization
```

`extract_fixed_prompt` 而非 `extract_full_h`：只有前者写 `fixed-prompt-summary.json`，`rebuild` 会校验它。

恢复版 v9 不再为了取得 token 数额外跑一遍 `extract_qwen`。先运行
`experiments.state_tokenizer.modality_lengths --prompt-mode instruct` 生成只含
`modality-lengths.npy` 的轻量 shard，再把它传给 `extract_fixed_prompt --source-features`。
该预扫描只加载 processor/config、不加载 9B 权重；正式抽取仍会逐条重算长度并在不一致时
立即失败，因此它不降低协议校验强度。可直接使用 `scripts_v9_instruct.sh modality-lengths`。

### 3. A1 / A2 瓶颈（JAX）

```sh
JX=/root/nas/users/luzheng/workspace/enter/envs/ResidualMem/bin/python3.11
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 PYTHONFAULTHANDLER=1 $JX -u -m \
  experiments.state_tokenizer.a1_continuous_bottleneck \
  --records .../full-721.jsonl --features .../static_features \
  --normalization .../key64-static-pca-normalization.npz \
  --num-e-tokens 64 --output outputs/a1/v9
```

- **A1 的 `--num-e-tokens` 必须与 A2 匹配**（A1 默认 16，A2 默认 64，不匹配时 warm start 报 shape 错）。
- A2 用 `--init-a1-checkpoint`（`--init-checkpoint` 是 A2 自身的续训入口）。
- **`PYTHONFAULTHANDLER=1` 不要省**：XLA 编译期段错误不产生任何 Python 报错，没有它日志会是空的。
- 这两个脚本只在结束时输出，运行中日志为 0 字节是正常的，不代表卡死；用 `ps -o etime,pcpu` 判断。

### 4. 测试

```sh
MC=/root/nas/users/luzheng/workspace/enter/envs/MemCompiler/bin/python3.12
JX=/root/nas/users/luzheng/workspace/enter/envs/ResidualMem/bin/python3.11
PYTHONPATH=. $MC tests/run_tests.py tests/state_tokenizer/test_axtree_serialization.py ...
PYTHONPATH=. $JX tests/run_tests.py tests/state_tokenizer/test_slot_layout.py
```

**没有 pytest**。`tests/run_tests.py` 是替身，支持 `parametrize` / `raises` / `approx` / `tmp_path`。它放在**仓库内**而不是 `/tmp`——`/tmp` 不跨会话存活。

按环境分组：`test_collect_resume.py` 需要 browsergym venv；`test_slot_layout.py` 刻意不 import torch 或 jax，两侧都能跑（它断言两处 layout 定义一致，这是唯一的跨环境一致性检查）。

### 5. 长任务与会话容器

`setsid nohup ... &` 起的进程 PPID=1，脱离当前 shell，但**扛不住会话容器重建**——那会清掉 `/tmp` 与 `~/.cache`。已知在 00:04 发生过一次，导致 10 万采集在 25k 处猝死。

对策：日志写 NAS 而非 `/tmp`；长任务用 `--resume`；不要依赖 `/tmp` 里的任何脚本。

`pkill -f <pattern>` 会匹配到发起它的 shell 自身（退出码 144）。用 `ps aux | grep "[p]attern"` 取 PID 再 kill。

**括号写法也救不了「在启动脚本里清理同类进程」。** 一条同时包含
`kill`（按模式匹配）与 `python -m experiments...` 的脚本，其自身命令行必然含有该模式，
`[d]ev\.` 的括号技巧对此无效 —— 括号只防 grep 匹配 grep 自己，防不住脚本匹配脚本自己。
曾因此在训练启动前杀掉整条命令链，四小时零产出且日志为空。
**启动脚本里不要做基于模式的清理**：先单独一条命令确认无残留，再单独一条命令启动。

**同一个坑也会咬等待器。** `while pgrep -f 'dev.d_head_bakeoff'; do sleep 60; done`
永远不会退出 —— 等待器自己的命令行里就含该字符串，`pgrep` 一直匹配得到。曾因此
留下两个空转 7 小时的 shell。等待自己起的进程结束时一律用括号写法：

```sh
while ps aux | grep -q "[d]ev\.d_head_bakeoff"; do sleep 60; done
```

---

## 最终配置

Static PCA + normalization + Slot WM 配置：

```text
configs/residualmem_web_static_pca.yaml
```

Normalization 实现：

```text
residualmem/encoders/normalization.py
```

Slot reconstruction / routing：

```text
residualmem/world_model/slot_reconstruction.py
experiments/state_tokenizer/a0_slot_reconstruction.py
```

A0 Slot-key 核心：

```text
K_s = W_K k_s^slot
V_t,s = W_V xbar_t,s
```

Slot key 只表示地址，Value 只表示当前状态内容。相同 valid mask 下，attention routing 不依赖状态内容。

A1 连续 latent bottleneck：

```text
residualmem/world_model/continuous_bottleneck.py
experiments/state_tokenizer/a1_continuous_bottleneck.py
```

A1 核心（encoder 允许内容相关 Key，decoder 严格使用 latent index 地址）：

```text
K_t,s = W_K LN(xbar_t,s + p_s)      V_t,s = W_V xbar_t,s        # encoder
K_i   = W_K a_i^latent              V_t,i = W_V e_t,i           # decoder
```

A1 没有独立 YAML：`N`、`d_e` 与训练计划由 CLI 指定，完整 resolved config、数据哈希与协议写入结果 JSON 和 checkpoint metadata。checkpoint 协议为 `a1_continuous_bottleneck_v1`，与 A0 checkpoint 互不兼容，不同 `N`/`d_e` 也不做 partial load。

复现 A1-Wide 结构控制（与 A0 同一组 64 states，§0 数值协议）：

```sh
PY=/root/nas/users/luzheng/workspace/enter/envs/ResidualMem/bin/python3.11
PYTHONPATH=. XLA_PYTHON_CLIENT_PREALLOCATE=false "$PY" \
  -m experiments.state_tokenizer.a1_continuous_bottleneck \
  --records outputs/state_tokenizer/v4/collection/fixed-subset.jsonl \
  --features outputs/state_tokenizer/v4/static_features \
  --normalization outputs/state_tokenizer/v4/key64-static-pca-normalization.npz \
  --output outputs/a1_continuous_bottleneck/wide_formal64_fp32.json \
  --num-e-tokens 64 --e-dim 512 --overfit-states 64 \
  --stage-steps 2000 3000 5000 20000 40000 \
  --stage-learning-rates 1e-3 3e-4 1e-4 1e-4 1e-4 \
  --gate-profile structural --platform gpu --device-index 0
```

`--matmul-precision` 默认即为 `highest`，无需显式传。A0 runner 未加该参数，其同协议锚点通过环境变量复现：

```sh
JAX_DEFAULT_MATMUL_PRECISION=highest "$PY" \
  -m experiments.state_tokenizer.a0_slot_reconstruction ... \
  --routing-mode slot_key --value-source raw_xbar --early-stop-mse 0
```

---

## 最终保留的产物

清理后 State Tokenizer 最终目录约 0.319GiB。

### 正式数据与 manifest

```text
outputs/state_tokenizer/v4/collection/
```

### Static PCA features

```text
outputs/state_tokenizer/v4/static_features/worker*/key64-static-pca-bf16.npy
outputs/state_tokenizer/v4/static_features/worker*/key64-static-valid.npy
outputs/state_tokenizer/v4/static_features/worker*/key64-static-positions.npy
outputs/state_tokenizer/v4/static_features/worker*/key64-static-slot-kind.npy
outputs/state_tokenizer/v4/static_features/worker*/key64-static-detail-ranges.npy
outputs/state_tokenizer/v4/static_features/worker*/key64-static-audit.npy
```

### PCA 与 normalization

```text
outputs/state_tokenizer/v4/key64-static-pca.npz
outputs/state_tokenizer/v4/key64-static-pca.summary.json
outputs/state_tokenizer/v4/key64-static-pca-normalization.npz
outputs/state_tokenizer/v4/key64-static-pca-normalization.summary.json
```

### 汇总指标

```text
outputs/state_tokenizer/v4/metrics/key64_summary/
```

### 最终文件清单

```text
outputs/state_tokenizer/v4/final-static-pca-manifest.json
```

该 manifest 记录清理后保留文件的路径、大小和 SHA256。

---

## A0 结果

### Hard diagonal oracle

```text
Natural MSE: 2.62e-5
R²: 0.999974
Denormalized RMSE: 0.00349
All group gates: PASS
```

### Slot-key routing

```text
Natural MSE: 8.27e-5
R²: 0.999918
Denormalized RMSE: 0.00751
Attention top-1 same-slot: 1.0
All group gates: PASS
```

机器可读结果：

```text
outputs/a0_slot_reconstruction_hard/formal64_final.json
outputs/a0_slot_reconstruction_slotkey/formal64_final.json
outputs/a0_slot_reconstruction_slotkey/comparison_all.json
```

---

## A1 结果

### A1-Wide 结构控制（N=64, d_e=512，无标量压缩）

数值协议：matmul precision `highest`、FP32 activations、FP64 host 指标累加、固定 eval batch 与 device（该协议对 jax 侧的 A1/A2 生效，两者均已作废；见 `git show aa9e5a7:STATE_TOKENIZER_WORKLOG.md` §0）。A0 Slot-key 已在同协议下重训作为锚点。

```text
Natural MSE: 5.13e-5          (A0 Slot-key 同协议 6.84e-5 的 0.75×)
R²: 0.999949
Denormalized RMSE: 0.00710    (A0 同协议 0.00653)
Invalid attention / output: 0
All group gates: PASS
Total steps: 70,000
```

分组对照（均为 `highest` 精度）：

| Group | A1-Wide | A0 Slot-key | A1/A0 |
|---|---:|---:|---:|
| Image | 4.75e-5 | 6.69e-5 | 0.71 |
| Detail | 4.95e-5 | 6.97e-5 | 0.71 |
| Context | 6.66e-5 | 6.78e-5 | 0.98 |
| Prompt | 4.82e-5 | 8.09e-5 | 0.60 |
| **Total** | **5.13e-5** | **6.84e-5** | **0.75** |

两个需要注意的细节：A1 在标准化 MSE 上优于 A0，但在 denormalized RMSE 上略差（残差更集中在高方差通道）；A1 的优势不均匀，context 组基本持平（0.98×）。

收敛轨迹（TF32 版，说明训练预算问题）：

| 累计步数 | Total MSE |
|---:|---:|
| 10,000 | 4.99e-4 |
| 30,000 | 1.66e-4 |
| 70,000 | 8.47e-5 |

A0 的 10k 步预算对 A1 不足；10k 步时 loss 仍单调下降，按该预算判定会把优化不足误报成结构缺陷。协议下 70k 步时误差仍在下降，所以 `5.13e-5` 是上界而非收敛下界。

A1 与 A0 的定性差异：A0 Slot-key 的 attention top-1 same-slot=1.0（近 one-hot 逐 slot 复制），而 A1-Wide 的 attention 始终弥散（每个 latent query 有效读取 23.8 个输入 slot，每个输出 slot 混合 36.4 个 latent token），仍然近无损——A1 学到的是分布式编码。

### matmul 精度

JAX 在 NVIDIA GPU 上对 float32 matmul 默认使用 TF32，这是 §0 协议的起因：

| 训练+评测精度 | A0 Slot-key | A1-Wide | A1/A0 |
|---|---:|---:|---:|
| TF32 | 8.27e-5 | 8.47e-5 | 1.02 |
| `highest` | 6.84e-5 | 5.13e-5 | 0.75 |

A0 在真 fp32 下训练改善 17%（只换评测精度仅 2%），A1 改善 39%。**训练精度对 A0 同样是未受控变量**，因此协议要求训练与评测同精度。

机器可读结果：

```text
outputs/a1_continuous_bottleneck/wide_formal64_fp32.json          # 协议内，正式
outputs/a1_continuous_bottleneck/wide_formal64_fp32_attention.npz
outputs/a0_slot_reconstruction_slotkey_fp32/formal64_final.json   # A0 同协议锚点
outputs/a1_continuous_bottleneck/wide_formal64.json               # TF32, 10k 步
outputs/a1_continuous_bottleneck/wide_formal64_cont1.json         # TF32, 30k 步
outputs/a1_continuous_bottleneck/wide_formal64_cont2.json         # TF32, 70k 步
```

尚未运行：任何标量压缩配置（`N<64` 或 `d_e<512`）、1,000-state test、多 seed 稳定性。

---

## A2 结果（PQ-8 grouped categorical）

数据流 `x̄_t → r_t → z_t → ẽ_t → x̂̄_t`。64 个 latent token 各切 8 个 64 维子空间，每子空间独立选 1 个 256-way 码字，codebook `(64,8,256,64)` 同时充当 prototype 与 embedding。前向严格是 gather 出的码字（无连续旁路），确定性 argmax 无 RNG。

码率：`64×8×8 = 4096` bits + 64 mask bits = 520 bytes/state，相对 A1 FP32 latent 张量是 **256× 名义缩减**——tensor footprint 口径，**不是实测 bitrate**（未做熵编码）。

```text
A1 连续基线 (validation): mse 1.256e-2  R² 0.9874
A2 PQ-8    (validation): mse 2.741e-1  R² 0.7254   rho_disc 21.81
```

三个运行（预算 vs 学习率的单变量对照）：

| 运行 | 步数 | val MSE | R² |
|---|---:|---:|---:|
| backbone LR 3e-5 | 70k | 3.025e-1 | 0.6969 |
| backbone LR 3e-5 续训 | 140k | 2.957e-1 | 0.7037 |
| backbone LR 3e-4 | 70k | 2.741e-1 | 0.7254 |

预算翻倍只买到 2.3%，LR 提高 10 倍在一半步数买到 9.4%——不是预算受限，但都已趋平。

分组（A1 R² → A2 R²）：detail `0.998→0.485`、image `0.981→0.681`、context `0.997→0.938`、prompt `0.998→0.944`。detail 组承载精确 UI 文本与短值，损失最严重。

**缺口来源**（三项证据排除常见解释）：码本健康无坍塌（perplexity 125、active 162/256、坍塌子空间 0）；de-dup 后 R² 仅从 0.725 降到 0.720；hard/soft 比仅 1.161（说明 argmax 不是主要额外代价，但因 soft 权重受距离结构约束、且 decoder 对 soft mixture 是 OOD，**不能据此断言码本张成空间不足**）。

量化几何——训练把一个健康的量化器破坏了：

| | train 量化相对误差 | validation | latent rms | codebook rms |
|---|---:|---:|---:|---:|
| 初始化（K-means on 冻结 A1 latent） | 0.142 | 0.218 | 1.503 | 1.532 |
| 联合训练 70k | 0.832 | 0.856 | 2.388 | 1.551 |

该比值应读作 **encoder 输出与量化 prototype 的几何失配程度**（训练健康度指标），不是严格的信息损失比例——联合训练后 `r` 已无固定物理意义，encoder 可自由缩放/旋转/换基。指向的判断是：softmax-ST 的 assignment gradient 不构成有效 commitment。

**尚未回答**：以上都不能证明「PQ-8 本身够不够」。需要 frozen-A1 容量诊断（冻结 encoder 与码本 ⇒ codes 完全固定，只训 decoder）来区分「联合训练破坏几何」与「PQ-8 太激进」。

### 容量诊断结论（推翻上述猜想）

| 运行 | validation R² | 量化误差 |
|---|---:|---:|
| A1 连续基线 | 0.9874 | — |
| D0 零训练 | 0.6063 | 0.218 |
| **D2 decoder-only（收敛）** | **0.6826** | 0.218 |
| 联合训练 70k | **0.7254** | 0.856 |

冻结 encoder（量化误差保持 0.218）并给 decoder 完全自由，只到 R² 0.683——**比放任 encoder 漂移的联合训练更差**。所以联合训练没有破坏什么，`quant/relative_error` 也不能当作「越低越好」的指标（这里有直接反例）。按预注册判据（D2 ≤ 0.70），**瓶颈是量化容量，不是优化或几何**；「换标准 VQ STE + commitment」因此不再是证据支持的方向。

latent 空间率失真（冻结 A1 latent，纯 K-means）：C 从 256→1024 使 train 误差降 4.5 倍（0.143→0.032）而 validation 几乎不动（0.218→0.215）——**码本受限于数据量而非 bit 预算**；8× 码率只把 validation 误差从 0.218 降到 0.123。per-token 码本（每个仅 2000 样本，val 0.218）显著优于 shared 码本（每个 128,000 样本，val 0.442），说明 64 个 latent token 占据显著不同的区域。

机器可读结果：

```text
outputs/a1_continuous_bottleneck/split_wide_fp32.json      # A1 连续基线
outputs/a2_categorical_bottleneck/formal_pq8.json          # backbone 3e-5, 70k
outputs/a2_categorical_bottleneck/formal_pq8_cont.json     # 续训至 140k
outputs/a2_categorical_bottleneck/formal_pq8_lr3e4.json    # backbone 3e-4, 70k
outputs/a2_categorical_bottleneck/initcheck.json           # K-means 与 τ 校准
```

适用范围：validation 为同 12 个任务、同模板的新实例，**不是跨任务泛化**。切分按 episode 对齐，但数据本身含重复初始页面（1.4% 的 validation 状态在 train 中有近乎逐位相同的孪生）。

尚未运行：其他 `(N,M,C)` 容量点、多 seed、test split（全程未打开）。

---

## 测试状态

清理前完成的关键测试：

```text
State tokenizer extended tests: 45/45 passed
Normalization/mask/slot RSSM: 6/6 passed
Legacy RSSM: 4/4 passed
```

按最终清理要求，实验期间新增的 probe/test 脚本已经删除；仓库当前保留的 state-tokenizer tests 为 21/21 通过。

A1 新增测试：

```sh
PY=/root/nas/users/luzheng/workspace/enter/envs/ResidualMem/bin/python3.11
PYTHONPATH=. JAX_PLATFORMS=cpu "$PY" -m pytest \
  tests/residualmem/test_continuous_bottleneck.py \
  tests/state_tokenizer/test_a1_continuous_bottleneck.py \
  tests/residualmem/test_categorical_bottleneck.py \
  tests/state_tokenizer/test_a2_categorical_bottleneck.py -q
```

当前 13/13 + 11/11 + 13/13 + 10/10 通过。注意 `tests/state_tokenizer` 其余测试需要 `torch`，只能在 PyTorch 环境运行；A1/A2 集成测试需要 JAX/Optax，只能在 JAX 环境运行。

全部 ResidualMem tests 曾为 39/40；唯一失败是未修改的 QueryEngine 全零 embedding tie 测试：检索返回 segment1，而测试固定引用 segment0。该失败与 State Tokenizer、normalization、RSSM slot posterior 或 A0 routing 无关。

---

## 运行约束

- 不使用 benchmark test labels 调整 State Tokenizer。
- PCA 与 normalization 只在 train split 拟合。
- Invalid slots 在标准化、attention、输出和 loss 中都必须屏蔽。
- 不永久保存 BF16 `xbar_t`；保留 BF16 `x_t`，训练时在线 FP32 normalization。
- `z_t` 只用于 categorical stochastic latent，不用于命名标准化状态；A1 的连续 latent 一律写作 `e_t`。
- 进入 latent/dynamics 前必须先通过对应 wiring/sanity gate。
- A1 容量之间的比较只在训练预算相同且都接近收敛时才成立（A0 的 10k 步预算对 A1 不足）。
- A1 test split 默认关闭，只有在容量选择与门限冻结后才用 `--evaluate-test` 打开。
- 不要删除 `final-static-pca-manifest.json` 中列出的最终产物。
