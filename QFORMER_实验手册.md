# Q-Former 状态压缩桥 · 实验手册

> 面向**接手跑实验的人**。想看排查过程和被推翻的假设，去 `QFORMER_WORKLOG.md`（编年体）。
> 本文只回答三件事：这是什么、怎么跑、哪里会把你坑死。
>
> 最后更新 2026-08-27。所有命令都在本机核对过；带 `file:line` 的地方是我读过代码确认的，没读过的会明说。

---

## 0. 三十秒版本

把一个冻结 9B 读到的一屏观察（截图 + AXTree，约 1100–2800 token）压成 **K 个软 token**，喂给同一个冻结 9B 去答题。要证的是**压缩比**，不是质量：

> **K=32 的学习式 resampler，用固定池化 1/2 的存储，在答题（QA-C）与屏幕保真（gap）两项上持平 64 槽固定池化。**

现在的状态：**QA-C 已经打平，保真 gap 还差一截，而唯一跑完的那条臂是欠训的。** 详见 §7。

---

## 1. 这是什么

### 1.1 数据流

```
截图 + AXTree
   │
   ├─ 冻结 Qwen3.5-9B，取第 16 层隐状态  H (N×4096)        ← trunk，无梯度
   │
   ├─ Q-Former：K 个可学 query 对 H 做 cross-attention      ← 唯一训练的部分 78.9M
   │     4 层 / hidden 1024 / 8 头
   │
   ├─ xbar (K×512)
   │
   ├─ InputSoftTokenConnector → K 个软 token
   │
   └─ 冻结 Qwen3.5-9B 作为 reader，读 [软token, 问题] 出答案
```

**只有中间那一段有梯度。** trunk 和 reader 是同一个冻结的 9B，各前向一次。

### 1.2 四个损失项

```
L = CE_gold  +  w_q·KL_q  +  w_o·KL_obs  +  λ_sem·InfoNCE
```

| 项 | CLI | teacher 输入 | student 输入 | 备注 |
|---|---|---|---|---|
| `CE_gold` | 恒开 | — | `[latent, 问题]` | **用 benchmark 标注**，论文里应作消融不作主张基础 |
| `KL_q` | `--distill-weight 0.3` | `[截图, AXTree, 问题, 金标答案]` | `[latent, 问题, 金标答案]` | 答案跨度上的 KL |
| `KL_obs` | `--obs-weight W` | `[截图, AXTree, probe]` + teacher **自己贪心生成**的 96 token | `[latent, 同 probe, 同续写]` | **全程不碰 benchmark 标注** |
| `InfoNCE` | `--sem-weight 1.0` | — | 同 session 内正负样本 | `--sem-mode same-session` |

`KL_obs`（模式 1）是本轮的新东西，也是整个实验要检验的那一项。

### 1.3 探针 P1–P4

定义在 `observation_kl_precheck.py:46-52`，是**唯一副本**，trainer 直接 import（`train_qformer_joint.py:76`）：

```
训练  P1  Faithfully describe the current screen state: visible text, input values,
          control types, selected/focused/enabled states, and spatial relations.
      P2  List the interactive controls visible on this screen and their states.
      P3  Describe the layout of this screen from top to bottom.
留出  P4  What text is currently visible on this screen?
```

**P4 从不进训练**：`:459` 的 `train_probes = [n for n in sorted(PROBES) if n != args.held_out_probe]` 把它排除，每个累积微批从 P1–P3 里独立抽一个。

⚠️ **P4 与 P1 语义重叠**（「可见文本」是 P1 那句 "visible text" 的子集）。所以它测的是**没见过的问法**，不是没见过的能力。写主张时必须照这个口径写。

### 1.4 两个头条指标

**QA-C**（答题正确率）来自官方 CLI，见 §4.3。

**保真 gap**（nat/token，越大越好）来自 `observation_kl_precheck.py:262-286`：

```
对每条观察 i：
   teacher 读 [截图_i, AXTree_i, probe]，贪心生成续写 C_i
   matched     = KL(teacher_i ‖ student(latent_i,           probe, C_i))
   mismatched  = KL(teacher_i ‖ student(latent_{(i+1)%n},   probe, C_i))
   gap = mean(mismatched) − mean(matched)
```

**差分-中差分**：两边的 prefix 长度不对称（32 槽 vs ~1100 真 token）带来的通用代价在两边完全相同，相减抵消，剩下的才是「这个 latent 是否携带了这一屏的内容」。同时报 `matched_beats_mismatched`（逐观察胜率，50% = 抛硬币）。

---

## 2. 环境

```bash
source /mnt/data/public_tools/miniconda3/etc/profile.d/conda.sh
conda activate qwen-vl
cd /mnt/data/users/luzheng/workspace/iclr/czs/residual-mem
export PYTHONPATH="$PWD"                        # 加载 WMA 数据集时还要加 :$PWD/../WorldMemArena
export TOKENIZERS_PARALLELISM=false
```

**不要用别的 env。** 本文所有命令的 python 都是 `/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python`。

**GPU 分工**（本机 8 卡）：

| 卡 | 用途 |
|---|---|
| 0–3 | 训练 / 预计算 / 各种离线评估 |
| **4–7** | **本地 LLM 服务，评测要用，别碰**（8 副本，端口 8017） |

查服务是否活着：

```bash
curl -s http://127.0.0.1:8017/v1/models      # 应返回 Qwen3.5-9B
cat outputs/logs/eval/server.pid             # 只能通过 pidfile 控制
```

☠️ **别用 `pkill -f` 去杀它**——`scripts_local_llm_server.sh` 的注释记录了它三次把调用方 shell 一起杀掉。

---

## 3. 目录与产物

约定 `D=outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar`。

| 路径 | 大小 | 是什么 | 谁产的 |
|---|---|---|---|
| `$D/wma-xbar-fitcorpus-axtree/` | 574 MB | 156 个样本的 trunk 特征 + `synthetic_axtree` | `wma_extract_xbar.py` |
| `$D/wma-teacher-fitcorpus/` | 70 MB | 检索 teacher 向量 | `wma_encode_teacher.py` |
| `$D/qformer-qa-pairs.npz` | 33 MB | 9041 条 QA 配对 | `build_qformer_qa_pairs.py` |
| `$D/wma-observation-teacher/` | 1.6 GB | P1–P4 续写 + top-128 logits | `build_observation_teacher.py` |
| `$D/qformer-*.pt` | 315 MB | 训好的 Q-Former | `train_qformer_joint.py` |
| `$D/cache-*.npz` | 373 MB | 从 checkpoint 抽的状态缓存 | `build_qformer_bridge_cache.py` |
| `$D/head-*.pt` | 19 MB | 检索头 | `train_retrieval_bridge.py` |

**一次训练写两个 checkpoint**（`train_qformer_joint.py:1378, 1399`）：

- `X.pt` —— 按 **val CE** 选
- `X.gapbest.pt` —— 按**留出探针 gap** 选

元数据里有 `selected_by` 字段标明是哪个。

---

## 4. 从零跑一遍

### 4.1 预计算（一次性，已经做完了）

**这四步已经完成，产物在 §3 的表里。** 只有换数据集或换 AXTree 风格时才需要重跑。依赖顺序是硬的：

```
wma_extract_xbar ──► wma-xbar-fitcorpus-axtree
       ├──► wma_encode_teacher      ──► wma-teacher-fitcorpus
       ├──► build_qformer_qa_pairs  ──► qformer-qa-pairs.npz
       │        └──► build_observation_teacher ──► wma-observation-teacher/
       └──► train_qformer_joint（吃上面全部四个）
```

只有 `build_observation_teacher.py` 支持分片（`--rank i --world-size n` 步长切分，一进程一卡；**本仓库没有 torchrun**）：

```bash
for R in 0 1; do
  DEV=$([ $R -eq 0 ] && echo 1 || echo 3)
  python -u -m experiments.state_tokenizer.build_observation_teacher \
    --xbar-dir "$D/wma-xbar-fitcorpus-axtree" --pairs "$D/qformer-qa-pairs.npz" \
    --model models/Qwen3.5-9B --output "$D/wma-observation-teacher" \
    --topk 128 --max-new-tokens 96 --max-answer-tokens 64 --min-coverage 0.99 \
    --rank $R --world-size 2 --device "cuda:$DEV" &
done; wait
```

各 rank 写同一个输出目录，每样本一个 npz，步长互斥所以安全。

⚠️ `--resume` **默认就是 True**，重跑会静默跳过全部已有文件。要重算得先清目录或传 `--no-resume`。

### 4.2 训练

用现成的 `outputs/logs/p1/run-arm.sh`（`$1` = obs 权重，`$2` = 卡号）：

```bash
nohup bash outputs/logs/p1/run-arm.sh 0.5 0 > outputs/logs/p1/k32d-obs0.5.log 2>&1 &
nohup bash outputs/logs/p1/run-arm.sh 1.0 2 > outputs/logs/p1/k32d-obs1.0.log 2>&1 &
```

展开后的完整命令：

```bash
python -u -m experiments.state_tokenizer.train_qformer_joint \
  --pairs "$D/qformer-qa-pairs.npz" --xbar-dir "$D/wma-xbar-fitcorpus-axtree" \
  --teacher-dir "$D/wma-teacher-fitcorpus" --teacher-cache "$D/wma-observation-teacher" \
  --model models/Qwen3.5-9B --output "$D/qformer-K32d-obs${W}.pt" --device "cuda:$2" \
  --queries 32 --qformer-layers 4 --accumulate 4 \
  --distill-weight 0.3 --distill-teacher observation --obs-weight "$W" \
  --sem-weight 1.0 --sem-mode same-session --sem-batch 4 --sem-extra-negatives 0 \
  --learning-rate 1e-4 --clip-norm 5.0 --seed 35 --no-drop-microbatches \
  --lr-recover-steps 10 --min-lr-fraction 0.25 --max-skipped-steps 500 \
  --held-out-probe P4 --probe-observations 24 \
  --max-steps 9000 --eval-every 250 --validation-observations 96 --patience-evals 8
```

约 13 小时 / 9000 步 / 单卡 25 GB。`python -m ... --help` 有全部 44 个参数。

**关键参数别乱动**（理由见 §6）：

| 参数 | 值 | 为什么 |
|---|---|---|
| `--no-drop-microbatches` | 关 | 开了会杀死训练，见 §6.1 |
| `--min-lr-fraction 0.25` | | 学习率地板，防冻结，见 §6.2 |
| `--lr-recover-steps 10` | | 干净 10 步涨回一档 |
| `--clip-norm 5.0` | | **梯度范数会到 1e15，这是正常的**，见 §6.1 |
| `--seed 35` | | 全部历史 run 都用它，改了就没法比 |

### 4.3 下游评测

训完一条臂之后四步，`{CKPT}` 是 `qformer-K32d-obs0.5.pt` 这类：

**① 抽状态缓存**（`--queries` 默认 16，**必须改成 32**）

```bash
python -u -m experiments.state_tokenizer.build_qformer_bridge_cache \
  --xbar-dir "$D/wma-xbar-fitcorpus-axtree" --teacher-dir "$D/wma-teacher-fitcorpus" \
  --checkpoint "$D/{CKPT}" --model models/Qwen3.5-9B \
  --output "$D/cache-{ARM}.npz" \
  --queries 32 --qformer-layers 4 --device cuda:1 \
  --validation-fraction 0.2 --seed 35
```

**② 训检索头**（其余全走默认）

```bash
python -u -m experiments.state_tokenizer.train_retrieval_bridge \
  --cache "$D/cache-{ARM}.npz" --output "$D/head-{ARM}.pt" --device cuda:1
```

**③ 检索闸**

```bash
python -u -m experiments.state_tokenizer.head_recall \
  --head "$D/head-{ARM}.pt" --cache "$D/cache-{ARM}.npz" \
  --split validation --device cpu \
  --output outputs/logs/p1/hr-{ARM}.json
```

⚠️ **这一步代码里没有通过标准**（`grep -c VERDICT head_recall.py` → 0，无退出码）。判据是**人工比较**：看 **within-sample R@1** 有没有打过固定池化。参考值：固定池化 0.3429（WMA）/ 0.3060（并集）/ 0.3618（干净 session），随机 0.0369。最近一条 K32 臂是 **0.1655**，还差得远。
**别看 global R@1**（`head_recall.py:22-24`）——干扰项更多只是更难，within-sample 才是任务本身。

**④ 保真 gap**（`--queries` 同样默认 16，**必须改**）

```bash
python -u -m experiments.state_tokenizer.observation_kl_precheck \
  --xbar-dir "$D/wma-xbar-fitcorpus-axtree" --checkpoint "$D/{CKPT}" \
  --model models/Qwen3.5-9B --queries 32 \
  --probe P4 --observations 48 --max-new-tokens 96 --seed 35 \
  --device cuda:1 --output outputs/logs/p1/final-{ARM}.json
```

这一步**有**判据（`:323-328`）：`gap <= 0` → 不要训；`matched_beats_mismatched < 0.9` → 先查；否则安全。

⚠️ **报告数字必须用 `--observations 48`。** 训练内联的 24 条噪声太大：同一 checkpoint 在 24 条上读 +0.1272，48 条上读 +0.0745。

**⑤ 官方 QA 评测**

```bash
cd ../WorldMemArena
export PYTHONPATH="$PWD"
export QWEN_EMBED_DEVICE=cuda:0 QWEN_VL_EMBED_LOCAL=1 LLM_MAX_CONCURRENT=48

export RESIDUALMEM_ROOT=/mnt/data/users/luzheng/workspace/iclr/czs/residual-mem
export RESIDUALMEM_QWEN35_MODEL=$RESIDUALMEM_ROOT/models/Qwen3.5-9B
export RESIDUALMEM_QFORMER=$RESIDUALMEM_ROOT/outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/{CKPT}
export RESIDUALMEM_QFORMER_QUERIES=32
export RESIDUALMEM_QFORMER_LAYERS=4
export RESIDUALMEM_RETRIEVAL_HEAD=$RESIDUALMEM_ROOT/outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/head-{ARM}.pt
export RESIDUALMEM_DEVICE=cuda:0
export RESIDUALMEM_HEAD_DEVICE=cuda:0

python -u -m eval_framework.cli \
  --dataset ./WorldMemArena --split all --subcategory agent/arena/web \
  --max-eval-workers 48 \
  --baseline ResidualMem-Instruct-Xbar-Input-RAG \
  --output-dir ./exp_results/{ARM}
```

**Q-Former 是通过环境变量插进去的，不是 CLI 参数**（`residualmem_instruct_adapter.py:105`）。

- `--baseline` 必须是 `ResidualMem-Instruct-Xbar-Input-RAG`。`-A2-` 变体会报错（Q-Former 不产 a2_xbar），`-L16-` 需要另外的 connector。
- **`RESIDUALMEM_INPUT_CONNECTOR` 不能和 `RESIDUALMEM_QFORMER` 同时设**（`:199-204` 会 raise），resampler 自带 connector。
- **`RESIDUALMEM_RETRIEVAL_HEAD` 一定要显式设。** 不设会静默回退到 checkpoint 里那个联合头，实测差一截（0.2131 vs 0.1988）。

出数：每个样本目录下 `aggregate_metrics.json` 的 `question_answering.{correct_ratio, hallucination_ratio, omission_ratio}` = QA-C / QA-H / QA-O。

---

## 5. 读日志

训练每 250 步打一行：

```
[qformer] step  1500  val CE 1.9024  (train CE 0.2961, KL 0.2053, sem 0.0000, obs 0.5229)
          headmin 11.9881  P4 1.4021/1.4459 gap +0.0438
          cos +0.0054/-0.0086  rank 24.4/67.4  mean/dev 1.7/1.2  COLLAPSE-REGRESSED
```

| 字段 | 含义 |
|---|---|
| `val CE` | 验证集答案 CE，**早停和 `.pt` 的选择依据** |
| `(train CE, KL, sem, obs)` | 四个损失项，**存活微批的均值** |
| `headmin` | 检索头输出范数的最小值，监控 `F.normalize` 的分母塌缩 |
| `P4 A/B gap G` | A = 匹配 KL，B = 错配 KL，**G = B − A**（已修，见 §6.3） |
| `cos a/b` `rank a/b` `mean/dev a/b` | **斜杠左边是学习式，右边是固定池化对照** |
| `COLLAPSE-REGRESSED` | 塌缩监控回退了。**几乎每行都有，目前不当作闸** |

`rank 24.4/67.4` = 学习式有效秩 24.4，固定池化 67.4。

跳步会打：

```
[qformer] step  1833  non-finite gradient (inf), update skipped, lr -> 5e-05; drawn=[...]
[qformer] step  1843  10 clean steps, lr -> 1e-04
```

---

## 6. 坑（必读）

这一节是本文最值钱的部分。每一条都真实地烧掉过至少一整轮实验。

### 6.1 ☠️ 梯度范数会到 1e15，**这是正常的，别去掐它**

同一把工具在四个状态下量的逐微批梯度范数（口径已折算到训练所见）：

| 状态 | 中位数 | p90 | >1e3 占比 | 非有限 | 结局 |
|---|---|---|---|---|---|
| 全新初始化 | 157.7 | 403.8 | 0.8% | 0.0% | — |
| **K32b step~1750** | **8.56** | **2.4e15** | **29.5%** | 2.0% | **跑满 9000 步，目前最好模型** |
| K32c w=1.0 step 750 | 6.0e4 | 8.6e12 | 56.8% | 0.0% | 805 步中止 |
| K32c w=0.5 step 1500 | 867 | 4.2e15 | 48.2% | 2.9% | 1746 步中止 |

**这个分布是重尾且固有的。** `clip_grad_norm_(5.0)` 会把整个梯度重新归一化，所以原始范数是 8.56 还是 2.4e15，**产生的更新完全一样**——这就是为什么 K32b 有 29.5% 的微批超过 1e3 却训得好好的。

我曾经加了个「逐微批丢弃超阈值梯度」的守卫（`--drop-microbatches --microbatch-max-norm 1e3`），结果：

- 阈值 1e3 恰好是**随机初始化模型的 p99**（971.4），**从第 1 步起就没有余量**
- 丢掉了每批约三分之一的样本，`buffer/kept` 方差变大 → 权重漂移 → 梯度变大 → 丢更多
- 中位数从 8.56 被推到 6e4，**守卫弄坏了它要保护的权重**
- w=1.0 在 805 步、w=0.5 在 1746 步双双中止

**`--drop-microbatches` 现在默认为 False，别打开。** 正确的做法就是 `clip_grad_norm_` + 非有限跳步，K32b 已经验证过。

### 6.2 ☠️ 学习率衰减会把 run 冻死

K32b w=1.0 的死法：**63 步内连续 halving 7 次撞到 1e-6 地板，从 1911 步起权重不动，剩下 7000 步全废。** 286 次跳步本身只占 3.2%，不是问题；**问题是每次跳步都连带惩罚学习率。**

现在的策略（`--min-lr-fraction 0.25 --lr-recover-steps 10`）：坏一步减半但**最低只到 2.5e-5**，干净 10 步涨回一档。跳步照跳，只是不再把学习率压死。

判断一条 run 有没有中这个坑：`grep 'lr ->' 日志 | tail -1`，如果长期停在 `2.5e-05` 就是在地板上。

### 6.3 训练日志的 gap 曾经是错的（已修）

`train_qformer_joint.py` 的内联 gap 用 `held_out[:-1]` 做匹配均值，而错配集覆盖的是 teacher 1..n-1，**错开一位**，配对失效。被排除的那条观察在每次 eval 里都是同一条离群值（matched ~2.59 对典型 1.41），**把 gap 抬高约 0.05**——和报告值本身一个量级。59 条历史 eval 行，59 条都虚高。

**已修为 `held_out[1:]`，现在打印的 `B − A` 精确等于 gap。**

要点：

- **`observation_kl_precheck.py` 从来是对的**（循环移位，n 对 n）。**主表里的 gap 全部来自它，不受影响。**
- 影响面仅限：日志打印、checkpoint 元数据、**`.gapbest.pt` 选哪一步**。训练本身无梯度参与，`.pt`（CE-best）和早停看 val CE，都不受影响。
- **2026-08-27 之前跑的 run，其 `.gapbest.pt` 是按旧口径选的。** 补救不用重跑：直接用 `observation_kl_precheck.py` 在存下的 checkpoint 上重算。

### 6.4 ☠️ 微批 backward 的缩放，和 L_sem 的相对权重

守卫关闭时，四个微批的梯度**直接累加进同一个 `.grad`**，而 `L_sem` 每步只跑一次、backward 进同一个 `.grad`，**不属于那个累加**。所以微批那一侧必须自带 `1/accumulate`：

```
正确    grad = mean(4 个微批)  +  L_sem 梯度
错误    grad = sum (4 个微批)  +  L_sem 梯度      ← L_sem 相对权重被削到 1/4
```

`clip_grad_norm_` 归一化的是总和，**救不了内部比例失衡**。守卫开启时不需要这个缩放，因为那条路末尾有 `buffer / kept` 还原均值——**两条路径的正确做法相反**，这就是当初写错的原因。代码里现在是 `microbatch_scale`，守卫开为 1.0、关为 `1/accumulate`。

症状：`--sem-weight 1.0` 写在命令行里，实际生效的是 0.25。日志上看是 **`rank` 塌下去**（K32d 第一次尝试 step 250 读到 8.6，同 seed 的 K32b 是 20.2）、`sem` 降不下来、`mean/dev` 乱跳。**L_sem 正是维持表征分散度的那一项**，所以有效秩是最灵敏的探针。

第一次 K32d 尝试因此白跑 1.1 小时，日志留在 `outputs/logs/p1/k32d-obs*.SCALEBUG.log` 作为对照样本。

### 6.5 ☠️ 评测可能静默返回标准答案

`eval_framework/cli.py:436-437`：

```python
target = target_for_baseline(baseline_name)
if not target.api_key:
    return q.gold_answer          # 不报错、不警告
```

**API key 为空时，管线直接把金标答案当作模型输出**，`correct_ratio` 会变成 ~1.0，看起来像重大突破。

当前状态：`eval_framework/.env` 存在且六个键都非空，实测能解析出 `Qwen3.5-9B` / `http://127.0.0.1:8017/v1` / 非空 key，**管线是好的**。但这个坑随时会因为 `.env` 丢失而复现（仓库根目录那个 `.env` 现在就是缺失的）。

**每次评测前用 `--smoke` 跑一个样本，看 `correct_ratio` 是不是可疑地接近 1.0。**

### 6.6 `--queries` 默认是 16

`build_qformer_bridge_cache.py` 和 `observation_kl_precheck.py` 的 `--queries` **都默认 16**。K=32 的 checkpoint 不传 `--queries 32` 会挂或者静默配错。`--qformer-layers` 同理，`--self-attention` 也必须与训练时一致——**没有任何代码校验这个**。

### 6.7 评测跑着的时候别改代码

上一轮因为在评测运行中编辑仓库代码，触发 `SyntaxError`，**废掉了 v8 的 10 个样本**（`QFORMER_WORKLOG.md:168-169`）。

同理，训练跑着的时候改 `train_qformer_joint.py` 不会影响已加载的进程，但会让盘上代码和运行中代码不一致，复现时说不清。

### 6.8 27 个 `agent/gui/web` 样本永远不能进训练

`build_qformer_qa_pairs.py:14-15` 的 `EVAL_SUBCATEGORY = "web"` 把它们整体排除。**它们的 QA、答案、memory_points、evidence 一律不得进入任何训练。** 这是评测集。

### 6.9 其他

- **`--resume` 默认 True**（`build_observation_teacher.py:101`、`wma_extract_xbar.py:313`），重跑会静默跳过。
- **`wma-xbar-fitcorpus`（无 `-axtree` 后缀）不带 `synthetic_axtree`**，用它会在 `build_qformer_bridge_cache.py:80-81` 报错。认准 `wma-xbar-fitcorpus-axtree`。
- **`agent/vab/css` 被故意排除**：整页滚动截图高达 1280×37015，放不进 `max_length`，只留短的会造成高度偏置。
- **`--validation-fraction 0.2 --seed 35` 是承重的**，它定义了 by-sample 的训练/验证切分，改了就和之前所有 head 没法比。
- **同一个本地模型既答题又判题**，这是必须写进表注的 caveat，最终数字要换 GPT-5.4 重判。
- **`enable_thinking=False` 是承重的**（`build_observation_teacher.py:151`）。不关的话贪心解码会把 96 token 全花在推理前言上，生成不出页面描述。

---

## 7. 当前结果，以及它们的可信度

### 7.1 主表（27/27 样本，1459 条 QA）

| | 槽数 | QA-C | QA-H | QA-O | 保真 gap |
|---|---|---|---|---|---|
| v6 官方 Raw-Fused | — | 0.5415 | 0.2132 | 0.2454 | — |
| v9 Q-Former | **16** | 0.5949 | 0.1857 | 0.2193 | +0.0976 |
| v10 固定池化 | **64** | **0.5984** | 0.1864 | **0.2152** | **+0.1610** |
| v12 = K32b obs1.0 CE-best | 32 | 0.5936 | 0.1885 | 0.2180 | +0.0487 |
| v12 = K32b obs1.0 gap-best | 32 | 0.5936 | 0.1851 | 0.2214 | +0.0655 |

配对 McNemar：

```
v12 cebest vs gapbest   14/14   p=1.000      完全无差别
v12 gapbest vs v9               p=0.864      不显著
v12 gapbest vs v10              p=0.371      不显著
v12 gapbest vs v6               p=5.39e-07   ***
```

### 7.2 这些数字能说什么、不能说什么

**能说**：K=32 + 观察损失达到 QA-C 0.594，与 K=16 和 64 槽固定池化统计打平，显著高于官方 RAG 基线。评测本身是干净的。

**不能说**：

1. **不能说观察损失有用。** v12 的 gap（+0.0655）**比不加这个损失的 v9（+0.0976）还低**。模式 1 就是为了推高 gap 才加的。
2. **v12 vs v9 不是受控比较。** v12 best_step 1750 / val CE 1.9527 / 286 次跳步 / 1911 步起冻结；v9 best_step 6000 / val CE 1.7643 / 0 次跳步。**在训练目标本身上就输了一大截**，QA-C 打平是收敛模型和瘫痪模型之间的打平。配对差 95% CI [−0.0092, +0.0065]，跨零。
3. **w_o 作为变量没测成。** 三个权重里只有 w=1.0 跑到 9000 步；w=0.5 在 1592 步中止，w=0.8 停在 best_step 1250 / val CE 2.0424。
4. **Q-Former 不比固定池化更好**，QA-C 打平（p=0.636）。主张是**压缩比**，不是质量。
5. **保真度涨不一定传导到 QA-C。** 先例：session 内 R@1 涨 50%，端到端 p=0.237 不显著。并列报两个指标，**不拿前者代替后者**。
6. **每条臂都是「训过的」对「零样本的官方基线」。** 改损失消除不掉，表注必须写明。

### 7.3 「内部指标涨、QA-C 不动」已经出现三次

这是本项目最稳定的现象，做新实验前先想清楚你要证的到底是哪一个。

---

## 8. 在跑 / 待办

**在跑**（2026-08-27 起，约 13 小时）：

```
K32d w=0.5  → GPU 0   outputs/logs/p1/k32d-obs0.5.log
K32d w=1.0  → GPU 2   outputs/logs/p1/k32d-obs1.0.log
```

守卫关闭、GradScaler 完整逻辑。

**验证点（很重要，已经抓到过一个 bug）**：K32d w=1.0 与 K32b w=1.0 同 seed 同配置，而 K32b 首次跳步在 1833 步，所以 **250–1750 步的七次 eval 应当重现 K32b**：

```
K32b w=1.0  step 250   val CE 2.0684   headmin 12.0433   rank 20.2
```

（`train CE` / `KL` 会差 4 倍，那是汇总口径从"求和"改成"除以 kept"；`gap` 口径也已改，见 §6.3。**看 `val CE` / `headmin` / `rank`。**）

第一次 K32d 尝试在这里就没通过——`rank` 读到 8.6 而不是 20.2，`headmin` 却对得上（12.0431 vs 12.0433，说明初始权重是同一个）。顺着这个差异找到了 §6.4 的缩放 bug。**这个对照点值得每次重写训练循环之后都跑一遍。**

**待办**：

- [ ] `head_recall.py` **从未在 `head-K32b-obs1.0-{cebest,gapbest}.pt` 上跑过**，这道闸是欠着的
- [ ] 重训固定池化 connector，让主表两列的数据处理口径一致
- [ ] 跑一条干净的 `w_o=0` 对照到收敛
- [ ] 最终数字换 GPT-5.4 重判
- [ ] 2026-08-27 前的 `.gapbest.pt` 用 `observation_kl_precheck.py` 重算 gap（§6.3）

**可选提速**（未做）：`latents_for` 每个微批都重跑一次 trunk 前向，而 trunk 是冻结模型对固定输入的确定性函数，9000 步 × 4 微批分摊到 2741 条观察上**每条被重复编码约 13 次**。缓存下来数值完全等价，约 31 GB 磁盘，估计快 1.5–2 倍。需要改代码 + 预计算 + 重启，**不要在信任度要紧的 run 跑着的时候做**。

（`flash-linear-attention` / `causal-conv1d` 没装，Qwen3.5 的门控 delta rule 层在跑纯 PyTorch 回退。装上会更快，但**换核就换数值**，会毁掉与历史 run 的逐位对照。不推荐。）

---

## 附录 A：从已删文档抢救出来的两件事

2026-08-27 删掉了 `STATE_TOKENIZER_WORKLOG.md` / `TOKENIZER_WM_HANDOVER.md` /
`WORLDMEMARENA_TOKENIZER_RAG.md`——它们记录的方法（A1/A2 bottleneck、world model、
固定池化时代的检索协议、jax 侧的数值协议）已经作废。**全文仍在 git history 里**：

```bash
git show aa9e5a7:STATE_TOKENIZER_WORKLOG.md
git show aa9e5a7:TOKENIZER_WM_HANDOVER.md
git show aa9e5a7:WORLDMEMARENA_TOKENIZER_RAG.md
```

下面两件事当时还在服役，所以抄过来。

### A.1 PCA / normalization 基底的拟合口径

`wma_extract_xbar.py:309-310` 把 `--pca` 和 `--normalization` 声明为 **required**，
当前训练输入 `wma-xbar-fitcorpus-axtree` 就是用它们抽的：

```
outputs/state_tokenizer/v9-instruct-pca20k-balanced/key64-static-pca.npz               8.5 MB
outputs/state_tokenizer/v9-instruct-pca20k-balanced/key64-static-pca-normalization.npz  27 KB
```

拟合口径（换数据集或换 AXTree 风格需要重拟合时照这个来）：

- **只在 train split 上拟合**——`key64_pca fit` 与 `fit_normalization` 都硬性过滤，不是约定
- **PCA**：2 万 train 状态，4096 → 512
- **normalization**：group×channel，70,018 train 状态。三个真实组的 scale 分别为
  image 0.151–4.054、detail 0.278–6.288、context 0.074–2.166，**比值 85.3**
- 零宽 prompt 组允许存在但 sigma 被钳到下限，是永不被索引的哑值；加载器只对
  「有槽却无有效数据」的组报错

⚠️ **换了 `--axtree-style` 就换了 H16，一个风格下拟合的 PCA 不是另一个风格的合法基底**
（`wma_extract_xbar.py:321-322`）。

### A.2 slot layout 的单一来源

layout 曾经被硬编码在**五处**：`fit_normalization`、`rebuild_static_key64` 的清单、
`a0`、`a1`、两个 bottleneck 的默认值。prompt 槽回收之后它们**全都继续按
`(32,12,16,4)` 切分**，归一化把 detail 的后 4 个槽当成 prompt 统计，报告了 4890 个
「有效 prompt 槽」——**一个异常都没抛**。

现在单一来源是 `experiments/state_tokenizer/slot_layout.py`（**无第三方依赖**，torch 与
jax 两侧都能 import；`key_pooling` 再导出它以保持既有引用）。目前仍有 12 个文件依赖它。

`residualmem` 不得 import `experiments`，所以两侧一致性由
`tests/state_tokenizer/test_slot_layout.py` 断言——**该测试刻意不 import torch 或 jax**，
两个解释器下都能跑。

（相关的硬约束：抽取管线只有 torch、bottleneck 只有 jax、采集只有 playwright，三者互不可
导入。跨环境共享的常量必须放在无第三方依赖的模块里。）
