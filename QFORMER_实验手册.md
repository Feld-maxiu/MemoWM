# Q-Former 状态压缩桥 · 实验手册

> 最终版的设计、结果、跑法。历史排查过程已压缩成 §5 的一张表，细节在 git：
> `git show c7b9687:QFORMER_WORKLOG.md`（456 行编年体）。
>
> 最后更新 2026-09-06。

---

## 0. 三十秒版本

把一个冻结 9B 读到的一屏观察（截图 + AXTree，约 1100–2800 token）压成 **K=32 个软 token**，喂给同一个冻结 9B 去答题。

**选定产物**：`$D/qformer-K32e-obs0.5.gapbest.pt`（step 6000）+
`$D/head-K32e-obs0.5-gapbest-utility-gated.pt`（检索头 step 1750，$\alpha=0.1$）。

能站住的主张只有一条：

> **K=32 的学习式 resampler，用固定池化 1/2 的存储，在屏幕保真（gap +0.1744 vs +0.1610）上超过 64 槽固定池化；QA-C 不劣化。**

---

## 1. 最终设计

### 1.1 结构

```
截图 + AXTree
   │
   ├─ 冻结 Qwen3.5-9B 第 16 层隐状态  H (N×4096)         ← trunk，no_grad
   │
   ├─ StateQFormer：32 个可学 query 对 H 做 cross-attention
   │     4 层 / hidden 1024 / 8 头 / QK-norm 开 / self-attn 关   ← 唯一训练的 78.9M
   │
   ├─ xbar (32×512)
   │
   ├─ InputSoftTokenConnector → 32 个软 token
   │
   └─ 冻结 Qwen3.5-9B 作为 reader，读 [软token, 问题] 出答案
```

trunk 和 reader 是同一个冻结的 9B，各前向一次。**只有中间那一段有梯度。**

三个不可动的设计点：

- **写侧必须与 query 无关**——adapter 在 ingest 时编码一次、缓存 latent，此后每个问题复用。query 条件化会作废整个存储模型。
- **cross-attention 不是因果的**，每个查询同时看图像、文本和指令。固定池化的因果注意力让观察 prompt 完全惰性（图像和 DOM 看不到指令，指令 token 因 `PROMPT_SLOTS = 0` 被整段丢弃），**那句 prompt 对 64 个槽的因果影响精确为零**。
- **QK-norm 必开**。参数无关的 RMS norm 加在 q/k 点积前，消除了 19 个数量级的梯度重尾（§4.1）。

### 1.2 损失

$$\mathcal L = \text{CE}_{\text{gold}} + 0.3\cdot\text{KL}_q + 0.5\cdot\text{KL}_{\text{obs}} + 1.0\cdot\text{InfoNCE}$$

| 项 | teacher 输入 | student 输入 | 用标注？ |
|---|---|---|---|
| `CE_gold` | — | `[latent, 问题]` | **是**，论文里应作消融不作主张基础 |
| `KL_q` | `[截图, AXTree, 问题, 金标答案]` | `[latent, 问题, 金标答案]` | 是（答案跨度） |
| `KL_obs` | `[截图, AXTree, probe]` + teacher **自己贪心生成**的 96 token | `[latent, 同 probe, 同续写]` | **否** |
| `InfoNCE` | 同 session 内正负样本，温度 0.05 | | 否 |

**两个 teacher 都读完整原始观察。** 旧口径的 `KL_q` teacher 只读约 89 token 的 `fused_text` 字幕，而那段字幕**逐字嵌在 student 自己的 AXTree 里**——teacher 的全部输入是 student 输入的子集，那个项在教「复现你已经原样拥有的文本」。

teacher 输出**预计算成 top-128**（`build_observation_teacher.py`）。截断在这个 KL 方向上有原则：每项由教师概率加权，丢掉的是权重最小的部分，残余质量即误差上界。**实测 70 万个位置：中位保留 0.999987，最小 0.73。** 全量存储要 0.2 TB，top-128 只要 1.6 GB。

**utility gate 接入后只重训独立检索头，不重训 Q-Former。** 令
$x_f=D(z^+)$ 为全码 OPQ 重建，
$x_m=D(m\odot z^+ +(1-m)\odot\hat z_{\rm WM})$ 为门控重建，检索头 $h$
使用：

$$
\mathcal L_{\rm head}=
\tfrac12\mathcal L_{\rm ret}(h(x_f),t)+
\tfrac12\mathcal L_{\rm ret}(h(x_m),t)+
\alpha\left[1-\cos\left(h(x_m),\operatorname{sg}(h(x_f))\right)\right].
$$

默认 $\alpha=0.1$。两路都保留原有 symmetric InfoNCE + teacher cosine，
因此一致性项不是单独优化，不存在“所有 key 一起塌缩”这个自由解。full 分支只在一致性项中
stop-gradient；它自己的检索损失仍正常回传。Q-Former、OPQ、WM 和 utility artifact 全部冻结，
否则会破坏已有码本、posterior 与 utility 位置坐标。

**冻结决定（2026-09-06）**：固定 $\alpha=0.1$，不再扫权重；固定 best step 1750。
相同 full/gated cache 上，新 head 的检索目标 3.30336（旧 head 3.31884），gated
validation R@1/R@10 为 0.2274/0.6560。该项定位为压缩感知的接口正则，不单独作效果贡献。

### 1.3 探针 P1–P4

定义在 `observation_kl_precheck.py:46-52`（**唯一副本**，trainer 直接 import）：

```
训练  P1  Faithfully describe the current screen state: visible text, input values,
          control types, selected/focused/enabled states, and spatial relations.
      P2  List the interactive controls visible on this screen and their states.
      P3  Describe the layout of this screen from top to bottom.
留出  P4  What text is currently visible on this screen?
```

每个累积微批从 P1–P3 独立抽一个；**P4 从不进训练**（`:459` 显式排除），只在 eval 用。

⚠️ **P4 与 P1 语义重叠**（"visible text" 是 P1 的子集）。它测的是**没见过的问法**，不是没见过的能力，主张要照这个口径写。

### 1.4 保真 gap 怎么算

```
对每条观察 i：
   teacher 读 [截图_i, AXTree_i, P4]，贪心生成续写 C_i
   matched     = KL(teacher_i ‖ student(latent_i,           P4, C_i))
   mismatched  = KL(teacher_i ‖ student(latent_{(i+1)%n},   P4, C_i))
   gap = mean(mismatched) − mean(matched)          单位 nat/token，越大越好
```

**差分-中差分**：prefix 长度不对称（32 槽 vs ~1100 真 token）带来的通用代价在两边完全相同，相减抵消，剩下的才是「这个 latent 是否携带了这一屏的内容」。同时报 `matched_beats_mismatched`（逐观察胜率，50% = 抛硬币）。

**这是目前唯一有判别力的指标**——各臂 0.0976 / 0.1408 / 0.1610 / 0.1744 分得开。

---

## 2. 效果

### 2.1 主表（27/27 样本，1459 条 QA）

| | 槽数 | QA-C | QA-H | QA-O | 保真 gap |
|---|---|---|---|---|---|
| v6 官方 Raw-Fused | — | 0.5415 | 0.2132 | 0.2454 | — |
| v9 Q-Former | 16 | 0.5949 | 0.1857 | 0.2193 | +0.0976 |
| v10 固定池化 | **64** | **0.5984** | 0.1864 | **0.2152** | +0.1610 |
| v12 K32b obs1.0 CE-best | 32 | 0.5936 | 0.1885 | 0.2180 | +0.0487 |
| v12 K32b obs1.0 gap-best | 32 | 0.5936 | 0.1851 | 0.2214 | +0.0655 |
| **★ K32e obs0.5 gapbest** | **32** | 0.5953 | 0.1886 | 0.2160 | **+0.1744** |
| K32e obs1.0 gapbest | 32 | 0.5977 | 0.1844 | 0.2180 | +0.1408 |

★ = 选定臂。QA-C 取顶层 `aggregate_metrics.json` 的 micro 值；gap 是 48 条 / P4 口径。

**K32e 两条臂是严格单变量对照**：只有 `--obs-weight` 不同，其余逐字相同、同数据同 seed。各 7250 步 / 12h42m / **跳步 0 次 / 学习率全程 1e-4 未动**。

| | CE-best 步 | val CE | gapbest 步 | val CE |
|---|---|---|---|---|
| obs0.5 | 5500 | **1.7475** | 6000 | 1.7572 |
| obs1.0 | 5500 | 1.7483 | 6500 | 1.7486 |

配对 McNemar（n=1457，`b` = 前者错后者对）：

```
所有 latent 臂两两                不一致 30–44    p=0.62–1.00   全部 ns
所有 latent 臂 vs v6 官方         不一致 222–230  p≈1e-7~1e-8   全部 ***  （胜负约 152:74）
```

### 2.2 一道没过的闸

**24 条 vs 48 条 gap 的排序会反**。内联 24 条说 obs1.0(+0.3691) ≫ obs0.5(+0.2381)，独立 48 条说 obs0.5(+0.1744) > obs1.0(+0.1408)。**两次测量两次反转，至今没解释掉。** 48 条是权威口径，24 条只能看同一条臂的趋势。

### 2.3 出表时必须写明

1. **只跑 `agent/gui/web` 27 样本**（论文 461），**同一个本地模型既答题又判题**，最终换 GPT-5.4 重判。
2. **gap 列来自按 gap 选出的 checkpoint**，且 48 条抽样的**前 24 条就是选择集**（同 seed 35）。这是在被选择的指标上报告。
3. **$w_o$ 两条臂都要报**。两点扫描全报是正常做法，只报赢的那条是第二层的指标选择。
4. **本轮动了三处**（`KL_obs` 新增 + `KL_q` 换 teacher + 槽数 16→32），**非单变量**，表注逐项列。
5. **top-128 是截断近似**，覆盖质量（中位 0.999987 / 最小 0.73）随结果一起报。
6. **每条臂都是「训过的」对「零样本的官方基线」**，改损失消除不掉。
7. **记忆侧 6 列各臂差 <0.01**——它们测的是 session 阶段的记忆抽取，不经过被改动的组件，**报成 12 列是充数**。
8. **Q-Former 只在 WMA 非 web 的 QA 上训过**，在 BrowserGym 上是分布外的。

---

## 3. 怎么跑

### 3.1 环境

```bash
source /mnt/data/public_tools/miniconda3/etc/profile.d/conda.sh
conda activate qwen-vl                    # 别用其他 env
cd /mnt/data/users/luzheng/workspace/iclr/czs/residual-mem
export PYTHONPATH="$PWD"                  # 加载 WMA 数据集时还要加 :$PWD/../WorldMemArena
export TOKENIZERS_PARALLELISM=false
D=outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar
```

☠️ **`set -euo pipefail` 的脚本必须把 conda 激活段用 `set +u` / `set -u` 包起来**——`activate.d/activate-gcc_linux-64.sh:107` 解引用未设置的 `SYS_SYSROOT`，会在跑第一行 python 之前就杀死脚本。

**GPU 分工**：0–3 训练/离线评估；**4–7 是本地 LLM 服务，评测要用，别碰**（8 副本，端口 8017）。查活：`curl -s http://127.0.0.1:8017/v1/models`。☠️ **别用 `pkill -f` 杀它**，记录在案三次把调用方 shell 一起杀掉，用 pidfile / PID。

### 3.2 产物

| 路径 | 大小 | 谁产的 |
|---|---|---|
| `$D/wma-xbar-fitcorpus-axtree/` | 574 MB | `wma_extract_xbar.py` |
| `$D/wma-teacher-fitcorpus/` | 70 MB | `wma_encode_teacher.py` |
| `$D/qformer-qa-pairs.npz` | 33 MB | `build_qformer_qa_pairs.py` |
| `$D/wma-observation-teacher/` | 1.6 GB | `build_observation_teacher.py` |
| `$D/qformer-*.pt` | 315 MB | `train_qformer_joint.py` |
| `$D/cache-*.npz` / `head-*.pt` | 373 / 19 MB | `build_qformer_bridge_cache.py` / `train_retrieval_bridge.py` |
| `$D/cache-*-utility-gated.npz` | 835 MB | `build_utility_retrieval_cache.py`（full/gated 双视图，不重跑 Q-Former） |

**预计算已全部做完**，只有换数据集或换 AXTree 风格时才重跑。依赖顺序是硬的：`wma_extract_xbar` → {`wma_encode_teacher`, `build_qformer_qa_pairs` → `build_observation_teacher`} → `train_qformer_joint`。只有 `build_observation_teacher` 支持分片（`--rank i --world-size n`，一进程一卡，**本仓库没有 torchrun**）。

**一次训练写两个 checkpoint**：`X.pt` 按 val CE 选，`X.gapbest.pt` 按留出探针 gap 选，metadata 的 `selected_by` 标明是哪个。

### 3.3 训练

```bash
nohup bash scripts_qformer_train_arm.sh 0.5 0 > outputs/logs/p1/k32e-obs0.5.log 2>&1 &
```

展开（`$1` = obs 权重，`$2` = 卡号）：

```bash
python -u -m experiments.state_tokenizer.train_qformer_joint \
  --pairs "$D/qformer-qa-pairs.npz" --xbar-dir "$D/wma-xbar-fitcorpus-axtree" \
  --teacher-dir "$D/wma-teacher-fitcorpus" --teacher-cache "$D/wma-observation-teacher" \
  --model models/Qwen3.5-9B --output "$D/qformer-K32e-obs${W}.pt" --device "cuda:$2" \
  --queries 32 --qformer-layers 4 --accumulate 4 --qk-norm \
  --distill-weight 0.3 --distill-teacher observation --obs-weight "$W" \
  --sem-weight 1.0 --sem-mode same-session --sem-batch 4 --sem-extra-negatives 0 \
  --learning-rate 1e-4 --clip-norm 5.0 --seed 35 --no-drop-microbatches \
  --lr-recover-steps 10 --min-lr-fraction 0.25 --max-skipped-steps 500 \
  --held-out-probe P4 --probe-observations 24 \
  --max-steps 9000 --eval-every 250 --validation-observations 96 --patience-evals 8
```

约 13 小时 / 单卡 25 GB。**别乱动的参数**：`--qk-norm` 开（§4.1）、`--no-drop-microbatches` 保持关（开了会杀死训练）、`--min-lr-fraction 0.25` 防冻结（§4.2）、`--seed 35`（全部历史 run 都用它）。`--sem-extra-negatives 0` 而非 `-1`：免费负样本已实现（3.0 → 19.9 个，墙钟成本为零）但本轮已动三处变量，再加就归因不了。

**日志前缀是 `[qformer]`**，每 250 步一行：

```
[qformer] step 6000  val CE 1.7572  (train CE 0.3709, KL 0.1428, sem 0.6738, obs 0.3354)
          headmin 11.05  P4 1.2367/1.4748 gap +0.2381
          cos -0.0079/-0.0086  rank 37.3/67.4  mean/dev 1.1/1.2  COLLAPSE-REGRESSED
```

斜杠**左边是学习式、右边是固定池化对照**。`P4 A/B gap` 的 A=匹配 KL、B=错配 KL。`COLLAPSE-REGRESSED` 几乎每行都有，**目前不当作闸**。

### 3.4 下游评测

前四步一条命令：

```bash
bash scripts_qformer_downstream.sh K32e-obs0.5-gapbest qformer-K32e-obs0.5.gapbest.pt 0
```

依次是：① `build_qformer_bridge_cache` 抽状态缓存 → ② `train_retrieval_bridge` 训检索头 → ③ `head_recall` 检索闸 → ④ `observation_kl_precheck` 算 48 条 gap。两条臂并行约 26 分钟。

- **①④ 的 `--queries` 默认 16，必须传 32**，没有任何代码校验它与 checkpoint 是否匹配。
- **③ 输出 `head_recall` 诊断，不作为方法结论。**
- **④ 有判据**（`:323-328`）：`gap <= 0` → 不要训；`matched_beats_mismatched < 0.9` → 先查。**报告数字必须用 `--observations 48`。**

第五步官方 QA：

```bash
bash scripts_qformer_official_eval.sh K32e-obs0.5-gapbest qformer-K32e-obs0.5.gapbest.pt 0
```

utility-gated 检索头复现入口：

```bash
bash scripts_utility_retrieval_head.sh 0
```

当前选定 head 已锁定；脚本检测到它存在时会拒绝覆盖。从头复现时，脚本从现有
`cache-K32e-obs0.5-gapbest.npz` 离线构造 full/gated 重建，只训练
`MaskedAttentionRetrievalHead`。无 causal WM posterior 的 episode 初态按 codec 规则
all-send，因此两路重建完全相同。

**Q-Former 通过环境变量插入，不是 CLI 参数**（`residualmem_instruct_adapter.py:113-120`）。脚本已固化四个坑：

- `--baseline` 必须是 `ResidualMem-Instruct-Xbar-Input-RAG`（`-A2-` 会报错，Q-Former 不产 a2_xbar）
- **`RESIDUALMEM_RETRIEVAL_HEAD` 一定要显式设**，不设会静默回退到 checkpoint 里的联合头，实测差一截（0.2131 vs 0.1988）
- `RESIDUALMEM_INPUT_CONNECTOR` **不能**和 `RESIDUALMEM_QFORMER` 同设（`:199-204` 会 raise）
- 加了 checkpoint / head 存在性预检，缺文件立刻退出而不是烧两小时

⚠️ **`WORKERS` 默认 24，不是单条臂时用的 48。** 两条臂各设 48 时 96 路并发把判分服务打爆，实测 OOM 重试 1400/1465 次（重试全成功、无掉题，但这是它跑 2h28m 的原因）。只跑一条臂时 `WORKERS=48 bash scripts_qformer_official_eval.sh ...`。单条约 2.5 小时。

**出数**：顶层 `aggregate_metrics.json` 的 `question_answering.{correct_ratio, hallucination_ratio, omission_ratio}` = QA-C/H/O，**这是 27 样本池化的 micro 值，直接用**。

⚠️ 输出目录有两种布局：`eval_framework.cli` 写 `{out}/sample_results/web_NN/`，历史 `[data150_gpt]` 驱动写 `{out}/{id}_{domain}/{baseline}/`。**两者是同一批 27 个样本**（按问题指纹逐一核对过，web_01↔333 … web_27↔359），跨 run 可比。`--dataset ./WorldMemArena` 是指向 `../WorldMemArena_hf_lfs` 的软链接，也是同一份数据。

---

## 4. 坑

### 4.1 QK-norm：19 个数量级的梯度重尾，以及它带来的新坑

同一份 checkpoint、同样 1000 行、只切换 `--qk-norm`：

```
             非有限        >1e3        median      p90        p99        max
无 QK-norm  20 (2.0%)  295 (29.5%)     8.558   2.374e15  4.144e19  8.406e19
有 QK-norm   0 (0.0%)    0 (0.0%)      1.73    3.363     8.364     14.86
```

定位链条：反向钩子测出梯度到 latent 是 ~1e-3、到 context 是 1e13–inf，**16 个数量级出现在 Q-Former 的四层内部而不是任何损失项里**；范围限于 `blocks.0` 的 `cross_q`/`cross_k`，`cross_v` 从不出现（它的梯度是 `p^T @ grad_out`，`p` 是概率分布，天然有界）——所以爆炸在 softmax 雅可比那一侧。K32e 两条臂各 7250 步**跳步 0 次**，val CE 1.7475/1.7483 是本项目最好的。

☠️ **但它是参数无关的**，`state_dict` 键不变（58 个）。这既是能做同权重受控对比的原因，**也意味着开着它训出来的 checkpoint 能干干净净载入一个没开它的模块，`strict=True` 没有键可缺，此后每个下游数字都走错误的前向算出来，一路没有东西会报错。**

修法是让开关跟着 checkpoint 走（commit `c7b9687`）：`QFormerInstructTokenizer(qk_norm=None)` 从 metadata 读，各脚本的 `--qk-norm` 是 `BooleanOptionalAction` 默认 `None`。官方 adapter 不传这个参数，自动就对。metadata 字段出现前写的 checkpoint 用 `stamp_qk_norm.py` 补盖。

⚠️ **同类隐患未修**：`observation_kl_precheck.py` **没有 `--qformer-layers` 参数**，只能吃默认 `layers=4`。现在恰好一致，但换成 6 层就会静默用 4 层去测。已核对 gap 与 QA-C 两处的 queries/layers/self_attn/qk_norm/layer/hidden/heads 目前**完全一致**。

### 4.2 其余的坑

| | 坑 | 后果 / 对策 |
|---|---|---|
| ☠️ | **评测会静默返回标准答案**（`eval_framework/cli.py:436-437`：`if not target.api_key: return q.gold_answer`） | `correct_ratio` 变 ~1.0 像重大突破。**每次评测前 `--smoke` 跑一个样本看是不是可疑地接近 1.0**。当前 `.env` 六个键都非空，但仓库根目录那个 `.env` 已缺失 |
| ☠️ | **学习率衰减会把 run 冻死** | K32b 在 63 步内 halving 7 次撞到 1e-6 地板，1911 步起权重不动，剩下 7000 步全废。现策略最低只到 2.5e-5 + 干净 10 步涨回。查：`grep 'lr ->' 日志 \| tail -1` |
| ☠️ | **`--drop-microbatches` 别开** | 阈值 1e3 恰是随机初始化的 p99，从第 1 步起没余量；丢掉每批 1/3 → 权重漂移 → 梯度更大 → 丢更多，**守卫弄坏了它要保护的权重** |
| ☠️ | **微批 backward 缩放** | 守卫关时四个微批直接累加，而 `L_sem` 每步只跑一次、不属于那个累加，所以微批侧必须自带 `1/accumulate`，否则 `--sem-weight 1.0` 实际生效 0.25。**守卫开时正好相反**（末尾有 `buffer/kept`），这就是当初写错的原因 |
| ☠️ | **27 个 `agent/gui/web` 样本永远不能进训练** | `build_qformer_qa_pairs.py:14-15` 的 `EVAL_SUBCATEGORY = "web"` 整体排除。QA/答案/memory_points/evidence 一律不得进入任何训练 |
| ☠️ | **评测/训练跑着时别改代码** | 中途 `SyntaxError` 废掉过 v8 的 10 个样本。也意味着新加的 metadata 字段对正在跑的 run 无效——K32e 四个 checkpoint 都得事后补盖 |
| ☠️ | **用 sed 改启动脚本会静默截断命令** | 一次 sed 把反斜杠续行替换成空行，训练带着 argparse 默认值跑起来（16 query 而非 32、accumulate 8 而非 4），**九分钟后才发现**。启动脚本要整文件写出 |
| | **训练日志的 gap 曾经错了一位**（已修） | 旧口径用 `held_out[:-1]` 与错配集错开一位，把 gap 虚抬约 0.05。**`observation_kl_precheck.py` 从来是对的，主表 gap 不受影响**；但 2026-08-27 前的 `.gapbest.pt` 是按旧口径选的步 |
| | `--resume` **默认 True** | `build_observation_teacher.py:101`、`wma_extract_xbar.py:313`，重跑静默跳过 |
| | `wma-xbar-fitcorpus` 无 `-axtree` 后缀的那个**不带 `synthetic_axtree`** | 会在 `build_qformer_bridge_cache.py:80-81` 报错 |
| | `--validation-fraction 0.2 --seed 35` 承重 | 定义 by-sample 切分，改了和之前所有 head 没法比 |
| | `enable_thinking=False` 承重 | 不关的话贪心解码把 96 token 全花在推理前言上 |
| | `agent/vab/css` 被故意排除 | 整页截图 1280×37015 放不进 `max_length`，只留短的会造成高度偏置 |
| | 杂项 | `/tmp` 满会让 torch 在 import 期炸（`Errno 28`，报错位置离原因很远）；zsh 不做词分割，`set -- $cfg` 在两个 shell 下行为不同；`nohup` 扛不住工具超时（SIGTERM），要 `setsid` |

---

## 5. 被推翻的判断

**每一条都曾经是被写进文档的结论。** 细节见 `git show c7b9687:QFORMER_WORKLOG.md`。

| | 曾经认为 | 实际 |
|---|---|---|
| 1 | latent 比原文差 7 分 | **是 prompt bug**。用了自写的 20 词 prompt + 裸 tokenizer 而非官方 `_ANSWER_SYSTEM_PROMPT`，且 thinking 默认开、96 token 全花在前言上。对齐后 QA-C **0.5086 → 0.5949**。v4/v7 的数字已全部作废。现在 adapter 构造时 `_assert_official_prompts_match()` 与官方逐字比对 |
| 2 | 有效秩是可检索性的先行指标 | **错了三次，不再用**。决定性反例：两条臂 session 内有效秩都是 21.5，而 R@1 差 50%。**秩只能证伪塌缩，不能预测可检索性** |
| 3 | 联合头是唯一适配这套坐标的头 | **反了**，单独头两条臂都赢。联合头只有 3 个负样本还在追移动的表示；单独头有 64 个负样本、拟合冻结表示。**联合头是雕刻刀，单独头是量尺** |
| 4 | 旧 ℒ_sem 在工作 | **恒等于 0**。B=1 时 logits 是 1×1，单 logit 对标签 0 的交叉熵恒零。只剩余弦项教「指向你的教师」，而教师彼此就很像（0.8199 vs 0.6491），R@1 只有 0.045。改成 same-session InfoNCE 后 R@1 0.1336 → 0.2005 |
| 5 | 表征塌缩了 | **监控口径错**：`spread()` 算秩去均值、算余弦没去，而池化 xbar 本来就近零均值，两者不可比。去均值后都接近正交。**旧 checkpoint metadata 里的 `monitors` 不可引用** |
| 6 | 非有限梯度是 K/ℒ_sem/warmup/KL无界/精度… | **八个假设八个被证伪**，第九个才是根因（QK-norm，§4.1）。唯一查出的真 bug 是 `obs_step` 在累积循环内被调 4 次，`--obs-weight 1.0` 实际是 4.0 |
| 7 | K=64 不稳 | **归因错了两次**，与槽数无关，根因还是 QK-norm。顺带量到一个仍有效的数：检索头归一化前的范数 16 槽 8.28 / 32 槽 8.32 / **64 槽 4.72（腰斩）**——这是选 K=32 的额外理由 |

**一个仍然成立的正面结果**：`w_o=0` 对照的 gap 只有 **+0.0024**、逐条胜率 56.2%（≈抛硬币），而 `w_o=0.5` 是 +0.0745。**没有 obs 目标时，latent 里几乎没有屏幕特异信息。** 另：各臂匹配 KL 约 0.63–0.67，都低于固定池化的 0.7248——**32 槽复现教师行为比 64 槽更准，差的一直是区分度**。

---

## 6. 待办

**没有在跑的任务。** 按优先级：

| | 事情 | 为什么 |
|---|---|---|
| **1** | **换 seed 跑 96 条重测两臂 gap** | §2.3 第 2 条。「obs0.5 保真更好」目前只有一次独立测量，且测在选择集上 |
| 2 | 重训固定池化 connector | 让主表两列的数据处理口径一致 |
| 3 | 跑一条干净的 `w_o=0` 对照到收敛 | 现有对照只到 step 1250 |
| 4 | 最终数字换 GPT-5.4 重判 | 现在答题与判题同模型 |
| 5 | `observation_kl_precheck` 补 `--qformer-layers` | §4.1 的同类隐患 |
| 6 | 2026-08-27 前的 `.gapbest.pt` 重算 gap | §4.2 |

**明确不做**：MolmoWeb 接入（无 AXTree、无 QA、183 GB，教师向量分布问题未解）；装 `flash-linear-attention`（更快但**换核就换数值**，毁掉与历史 run 的逐位对照）。

**可选提速（未做）**：`latents_for` 每微批都重跑 trunk 前向，而 trunk 是冻结模型对固定输入的确定性函数，**每条观察被重复编码约 13 次**。缓存下来数值完全等价，约 31 GB，估计快 1.5–2 倍。不要在信任度要紧的 run 跑着时做。

---

## 附录：重新拟合 PCA / 改 slot layout 时才需要看

**PCA / normalization 基底**（`wma_extract_xbar.py:309-310` 声明为 required，当前输入就是用它们抽的）：**只在 train split 上拟合**（`key64_pca fit` 与 `fit_normalization` 都硬性过滤，不是约定）；PCA 用 2 万 train 状态 4096 → 512；normalization 是 group×channel、70,018 train 状态，三个真实组 scale 比值 85.3。⚠️ **换了 `--axtree-style` 就换了 H16，一个风格下拟合的 PCA 不是另一个风格的合法基底**。

**slot layout 单一来源**是 `experiments/state_tokenizer/slot_layout.py`（**无第三方依赖**，torch 与 jax 两侧都能 import），目前 12 个文件依赖它。它曾被硬编码在五处，prompt 槽回收后全都继续按 `(32,12,16,4)` 切分，归一化把 detail 的后 4 槽当成 prompt 统计、报告了 4890 个「有效 prompt 槽」，**一个异常都没抛**。两侧一致性由 `tests/state_tokenizer/test_slot_layout.py` 断言，**该测试刻意不 import torch 或 jax**。

（硬约束：抽取管线只有 torch、bottleneck 只有 jax、采集只有 playwright，三者互不可导入。跨环境共享的常量必须放在无第三方依赖的模块里。）

**更早删掉的文档**：`git show aa9e5a7:{STATE_TOKENIZER_WORKLOG,TOKENIZER_WM_HANDOVER,WORLDMEMARENA_TOKENIZER_RAG}.md`
