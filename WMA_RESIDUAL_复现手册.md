# WMA-ResidualMem 复现手册

> 一条链路：**观察 → Q-Former → OPQ 量化 → 世界模型条件编码 → 效用门控**，在
> WorldMemArena 上把每状态 6144 bit 的定宽记忆压到 4323.8 bit。
>
> 本文只讲**怎么跑**和**东西在哪**。方法与结论见 `技术报告_ResidualMem.md`，
> 门控的推导与全部声明见 `UTILITY_GATE.md`。
>
> 取代了 `WMA_RAG_WORKLOG.md`（检索线，2026-08-22 冻结）与
> `WORLD_MODEL_WORKLOG.md`（v8/MiniWoB 线，语料已废弃）。两者均无可运行命令，
> 完整内容在 git：`git show d91aaa3:WMA_RAG_WORKLOG.md`。

---

## 0. 先决定你要哪一种复现

| | 做什么 | 耗时 |
|---|---|---|
| **A. 只要数字** | 从已冻结的产物重算主表全部数值 | **约 1 分钟** |
| **B. 从原始数据重建** | 抽特征 → 拟合码本 → 训世界模型 → 打标 → 门控 | **数天**（含 3.8 h 八卡编码 + 60k 步训练） |

绝大多数情况要的是 A。B 只在换语料或换 Q-Former 时才需要。

---

## 1. 环境

☠️ **两个解释器互相看不见**，这不是配置疏忽而是刻意的：`.venv-jax` 关掉了
`include-system-site-packages`，所以 `conda activate qwen-vl && python` 里没有 jax。
任何跨环节的编排**只能用 subprocess**，不能 import。

```bash
REPO=/mnt/data/users/luzheng/workspace/iclr/czs/residual-mem && cd $REPO
JX=$REPO/.venv-jax/bin/python                                    # jax 0.4.33 + GPU
PY=/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python     # torch
export PYTHONPATH="$REPO" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TRITON_CACHE_DIR=/mnt/data/users/luzheng/.cache/triton
```

| 谁用哪个 | |
|---|---|
| `$JX` | 世界模型（train/evaluate/cache）、OPQ、后验、门控码率、manifest |
| `$PY` | Q-Former、reader、打标、检索头 |

☠️ **OMP 线程必须限制。** 这台机器 250 核，不限的话线程自旋会把 CPU 抢光，
GPU 推理**慢 5 倍**。
☠️ **`TRITON_CACHE_DIR` 不能留在 `/tmp`**（已 95% 满），torch 会在 import 期
炸 `Errno 28`，报错位置离原因很远。

---

## 2. 产物在哪

**全部由 `configs/system.lock.yaml` 管辖，加载时校验 sha256。** 先跑这个：

```bash
$JX -m residualmem.manifest        # 六项全 ok 才继续
```

| 角色 | 路径 |
|---|---|
| Q-Former | `$D/qformer-K32e-obs0.5.gapbest.pt`（step 6000） |
| 检索头 | `$D/head-K32e-obs0.5-gapbest.pt` |
| 码本 C=64 | `$DATA/pq-full/opq-shared-mix10-M32-C64.npz` |
| 世界模型 | `run/best.pkl`（60k 步；`last.pkl` 是它的硬链接，两者字节相同） |
| 效用掩码 | `gate/mask-lambda0.0010.npz` |

```
D=$REPO/outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar
DATA=/mnt/data/users/luzheng/workspace/iclr/czs/data/molmoweb-pilot
```
`$DATA` 在仓库外。`/home/luzheng/...`、`/mnt/data/users/luzheng/...`、
`/mnt/workspace/users/luzheng/...` 是**同一块盘**，脚本里混用是正常的，别去「修正」。

☠️ **这个仓库最危险的一处**：`pq/` 与 `pq-full/` 下有**完全同名**的
`opq-shared-mix10-M32-C64.npz`，是同一配置拟合的两次。只有 `pq-full/` 那本与世界模型
训练数据内嵌的解码器逐位一致。用错另一本会产出合法的 `(32,32) uint8` 码、缓存能建、
模型能打分——**只是所有数字差约 972 bit，且全程无任何报错**。
**只有 manifest 的哈希校验能拦住**——改名字或删文件都不行，因为出问题的正是「同名」这件事。

（C=16 码本已于 2026-09-02 删除，全线只用 C=64。技术报告 §5.1 的 C=16 行与 `UTILITY_GATE.md` §4.4 的
可证伪实验若要重做，需先用 `$DATA/encoded` 重新拟合。）

被删掉的 30 个历史 checkpoint 的指纹与元信息留在 `CHECKPOINT_INVENTORY.json`。

---

## 3. 路径 A：一条命令重算主表

```bash
$JX scripts_run_system.py --gpu 1
```

依次做：校验六个产物 → 重新导出掩码（自检码率）→ 闭环空对照 → 门控/随机/对抗三臂
→ 写 `system_results.json`。约 53 秒。

**应当得到**：

| 臂 | 码率 | 压缩比 | 答案代价 |
|---|---:|---:|---:|
| 定宽全发 | 6144.00 | 1.000× | — |
| + 世界模型条件编码 | **5182.83** | 1.185× | 0.148 bit（量化自身） |
| + 效用门控 λ=0.0010（开环） | **4311.96** | 1.425× | 0.103 bit |
| + 闭环修正 | **4323.84** | 1.421× | — |

☠️ **5182.83 是硬闸**。它必须复现 `run/run.json` 的
`best_selection.code_bits_per_transition = 5182.83124384419`（容差 1e-3；`code_bits`
存 float32，逐位比对会失败，早稿写「逐位一致」是错的）。对不上说明产物之间已经漂移，
后面的数一个都不能信。

---

## 4. 路径 B：从原始数据重建

四段，**严格按序**。每段的详细坑见括号里的手册。

### 4.1 Q-Former（`$PY`，见 `QFORMER_实验手册.md`）

```bash
bash scripts_qformer_train_arm.sh 0.5 0                 # 约 13 h，单卡 25 GB
bash scripts_qformer_downstream.sh K32e-obs0.5-gapbest qformer-K32e-obs0.5.gapbest.pt 0
```
下游四步一条命令：抽状态缓存 → 训检索头 → 检索闸 → 48 条观察 gap，两臂并行约 26 分钟。

☠️ `--queries` 默认 16，**必须显式传 32**，没有任何代码校验它与 checkpoint 是否匹配。
☠️ 一次训练写**两个** checkpoint：`X.pt` 按 val CE 选，`X.gapbest.pt` 按留出探针 gap 选。
交付的是后者。

### 4.2 量化 + 世界模型（`$JX`，见 `WM_MIXED_复现手册.md`）

```bash
$JX scripts_wm_dataset.py unpack --dataset worldmemarena_wm_train.npz --output ./cache
# 核对：transitions 495527 / validation 0 / fixed_width_bits 6144 / max_history 32

CUDA_VISIBLE_DEVICES=0 $JX -m experiments.world_model.train \
  --cache ./cache --config configs/world_model/web_h16_C64_full.yaml \
  --variant full --seed 0 --output ./run --platform gpu --device-index 0 \
  --dev-fraction 0.05 --max-steps 60000 --min-steps 5000 \
  --eval-every 2500 --patience-steps 12500

CUDA_VISIBLE_DEVICES=2 $JX scripts_build_wm_testset.py \
  --dataset-repo <WorldMemArena>/WorldMemArena \
  --checkpoint $D/qformer-K32e-obs0.5.gapbest.pt \
  --codebook $DATA/pq-full/opq-shared-mix10-M32-C64.npz \
  --output ./testset --jax-python "$JX" --torch-python "$PY" --device cuda:0

CUDA_VISIBLE_DEVICES=2 $JX -m experiments.world_model.evaluate \
  --cache ./testset/cache --checkpoint ./run/best.pkl \
  --split validation --output ./testset/eval
```

☠️ `--output` 目录必须不存在或为空（`train.py:500`），别提前 mkdir 子目录。
☠️ `--resume` 校验 `config_sha256`，改了预算再 resume 会被拒——要换预算就重跑。
☠️ `CUDA_VISIBLE_DEVICES=k` 必须配 `--device-index 0`，jax 只枚举可见设备。
☠️ cache 是 32 步窗口、模型是 16 步，**任何读 cache 的代码都要设
`cache.max_history = config.model.max_history`**，否则 shape 错误的报错位置离原因很远。

### 4.3 效用门控（见 `UTILITY_GATE.md` §9）

建 gate cache → 后验 → 打标 → 导出掩码。命令在那份文档里，此处不复制以免两处漂移。

### 4.4 闭环

```bash
# ☠️ 先跑空对照，gap 必须为 0 才能信后面的数
$JX -m experiments.utility_gate.closed_loop_rate \
  --cache testset/cache --config configs/world_model/web_h16_C64_full.yaml \
  --checkpoint run/best.pkl --mask gate/mask-lambda0.0010.npz \
  --all-send --split validation --output gate/closedloop-null.json
```
全发（$m\equiv1$）时解码端状态等于编码端状态，两趟**必须逐位相同**。这条对照抓到过
一个真错：`history_indices` 是 32 列而 batch 只取最近 16 列，按 32 列索引会把每个状态的
码写进别的状态的时间槽。

---

## 5. 实验结果

全部为一手实测。口径不同的表**不可横向相加**——码率在 817 条转移上、QA 在 1,459 道题上。

### 5.1 码率：世界模型 vs 统计基线（817 条外部转移）

| | bit/转移 | 相对定宽 |
|---|---:|---:|
| 定宽全发 | 6144.00 | 1.000× |
| 任务边缘分布（marginal） | 6381.99 | 0.963× |
| copy-aware | 6051.42 | 1.015× |
| source-conditioned Markov（最强统计基线） | 5420.84 | 1.133× |
| **世界模型（60k 步）** | **5182.83** | **1.185×** |
| **+ 效用门控 λ=0.0010（开环）** | **4311.96** | **1.425×** |
| **+ 闭环修正** | **4323.84** | **1.421×** |

基线由 `baselines/baseline.json` 逐位复现。定宽审计：码 6144 + mask 32 = 6176。
☠️ marginal **劣于**定宽（0.963×）不是 bug——它按任务边缘分布编码，比均匀更差。

### 5.2 码本大小的率失真取舍（pilot 语料）

| C | 定宽 | 最强基线 | 世界模型 | 压缩比 | 重建 R² | 持久率 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4096 | 3395.0 | 3039.2 | **1.348×** | 0.9767 | 25.1% |
| **64** | 6144 | 5431.7 | **5027.1** | 1.222× | 0.9881 | 13.7% |
| 256 | 8192 | 7605.5 | 7052.7 | 1.162× | 0.9909 | 10.1% |

两 seed 均值，seed 间差 20.3 / 30.3 / 53.4 bit。☠️ 运行间噪声底 **≥31 bit**，
小于此的差值不得当作结论。

### 5.3 效用门控

| λ=0.0010 | fit-corpus（3,756 状态） | **WMA web（817，官方评测集）** |
|---|---:|---:|
| 全发 → 门控 | 4631.3 → **3931.4** | 5182.8 → **4312.0** |
| 省下 | 15.1% | **16.8%** |
| 答案 \|ΔNLL\| | 0.078 bit | **0.103 bit** |
| 占量化自身代价（0.148）的 | 53% | **70%** |
| 压缩比 | 1.33× → 1.56× | 1.19× → **1.42×** |

配对检验（同一行、同一 forward）：web 上 vs random **t = −25.5**、vs 位移 −21.4、
vs 码长 −19.6。掩码在 fit-corpus 上拟合、在 web 上评测，**跨域**。

**闭环**（去掉「历史全发」假设）：漂移 +11.88 bit，省 16.80% → **16.57%**。上界是紧的。
☠️ 但等预算随机掩码漂移 11.0–12.5 bit，门控的 11.88 落在其中——**这份稳健不是门控挣来的**。

### 5.4 Q-Former 各臂（27/27 样本，1,459 题）

| | 槽数 | QA-C | 保真 gap | 检索 R@1 |
|---|---:|---:|---:|---:|
| v6 官方 Raw-Fused | — | 0.5415 | — | — |
| v9 Q-Former | 16 | 0.5949 | +0.0976 | 0.1464 |
| v10 固定池化 | 64 | **0.5984** | +0.1610 | **0.3429** |
| **★ K32e obs0.5 gapbest** | 32 | 0.5953 | **+0.1744** | 0.2310 |
| K32e obs1.0 gapbest | 32 | 0.5977 | +0.1408 | 0.2095 |

★ = 选定臂。唯一站得住的主张是：**K=32 的学习式 resampler 用固定池化一半的存储，
在屏幕保真上超过 64 槽固定池化，QA-C 不劣化**。
☠️ **不能写成「答题质量持平」**——K=32 vs v10 的 QA-C 差异 p=0.652，不显著；
而且 16 槽也打平，说明 QA-C 根本没在量存储敏感度。
☠️ Q-Former 在 session 内检索上**稳定输给固定池化**（0.2310 vs 0.3429）。

### 5.5 检索臂官方三臂对比（agent/arena/web，judge 零失败）

| | Raw-Fused | ResMem-现役 | ResMem-并集 |
|---|---:|---:|---:|
| QA-C | 0.5408 | **0.5778** | 0.5668 |
| QA-H（幻觉） | 0.2132 | **0.1720** | 0.1727 |
| RC (hit_rate) | 0.6536 | **0.6914** | 0.6867 |
| Recall@1 / @10 | **0.2414** / 0.6960 | 0.2221 / 0.7002 | 0.2242 / **0.7012** |
| answer tok/题 | 3,228 | 2,252 | **2,205** |

☠️ **两个混淆，引用时必须标注**：

1. **QA 一栏不是同类比较**——`mm_mode` 让 Raw-Fused 最多收到 5 张截图，
   ResidualMem 只有文本（3,228 vs 2,252 tok/题）。**我们 QA-C 更高不能读作方法更好。**
2. **token 少 32% 是同一配置的副产物**，不是表示更紧凑。

**复现可信度**：我们跑出的 Raw-Fused 是 QA-C **54.08** / RC **65.36**，论文 Table 2 报
**51.86** / **73.44**。回答与 judge 模型不同、且只跑了较难的 Agentic 一半，量级站得住。
☠️ 早期试点的 **79.25 是假象**（n=53 + 图片 bug），补图跑满 1,459 题后自行回落。

**按行类型分解**（官方 `_ranking_metrics`，n=1,459）——这是 §6 那条盲区的来源：

| | 完整 R@10 | 去掉观察行 | **观察行净贡献** | 占 top-10 |
|---|---:|---:|---:|---:|
| Raw-Fused | 0.6960 | 0.6505 | **+0.0455** | **14.6%** |
| ResMem-现役 | 0.7002 | 0.7002 | **0.0000** | **0.0%** |
| ResMem-并集 | 0.7012 | 0.6954 | +0.0058 | 3.0%

**现役 head 的 latent 行对检索分数贡献精确为零。** 这条轴的天花板约 4.5 个 R@10 点。

---

## 6. 反复踩到的坑

| | |
|---|---|
| **zsh 不做词分割** | `L="--lambdas 0.0009 0.0010"` 后 `$L` 会被当成**单个参数**；进程静默失败、日志为空。`kill $PIDS` 同理。参数写死，或走 `xargs` |
| **`nohup` 扛不住 SIGTERM** | 工具超时会杀掉整个进程组，长任务用 `setsid` |
| **打分必须 fp32** | bf16 的噪声底 0.186 bit 超过信号中位数的一半，且 delta 与 fp32**符号相反**。历史 bf16 数字不可与本结果并列 |
| **`baselines --output` 是目录** | 且码率嵌在 `bits_per_transition` 下，不在顶层 |
| **别用公共 robotwin 环境** | torch 2.4 在本机 sm_120 上跑不了 |

---

## 7. 这份复现测不到什么

☠️ **QA-C 看不见 latent 的变化。** 报告 §5.6：选定臂上 **90.3% 的题一条 latent 行
都检不到**，两个只差一个变量的臂在那 1,282 题上**逐字相同**。所以
**「量化/门控接进去后 QA-C 不动」不能表述为「无损」**——那是测不到。

有判别力的质量轴是 **\|ΔNLL\|（bit）**，已测、配对 t = −25.5。
裁判标签那条路也算过：2,248 配对下 McNemar 最小可检 1.32 pp，而门控效应约
0.0118 pp，**低 112 倍**，故未采用（`gate/judge-power.json`）。

☠️ **闭环下的稳健不是门控的功劳。** 等预算随机掩码漂移 11.0–12.5 bit，门控的
11.88 落在其中。详见 `UTILITY_GATE.md` §7.2。

---

## 8. 相关文档

| | |
|---|---|
| `技术报告_ResidualMem.md` | 方法、全部实测结果、22 条必须声明的事项 |
| `UTILITY_GATE.md` | 效用门控：推导、闭环、被推翻的判断 |
| `QFORMER_实验手册.md` | Q-Former 训练/评测细节与坑 |
| `WM_MIXED_复现手册.md` | 世界模型训练/评测细节与坑 |
| `CHECKPOINT_INVENTORY.json` | 全部 56 个历史 checkpoint 的指纹（含已删除的 30 个） |
