# 世界模型复现手册

> 训练数据是一个文件;测试集从官方仓库现取现转。
> 最后更新 2026-09-01。结论与解读见 `技术报告_ResidualMem.md`,本文只管**怎么跑**和**怎么验**。

---

## 0. 三十秒版本

在 6,144 bit 的定宽离散码上训一个块因果 Transformer 预测下一步的码,码率即
$R_t^{\mathrm{full}}=\sum_j-\log_2 p_t^j$,压缩比 = 6144 / 码率。

- **训练**:`worldmemarena_wm_train.npz`(train-only,495,527 条转移,无 split_ids 列)
- **测试**:WorldMemArena `agent/gui/web` 全部 27 个样本,**从官方仓库现取现转**(§3.3),817 条转移

跑完应该得到:

| | bits/转移 | 压缩比 |
|---|---:|---:|
| marginal 基线 | 6382.00 | 0.963 |
| copy 基线 | 6051.42 | 1.015 |
| source 基线 | 5420.84 | 1.133 |
| **世界模型** | **见 §4(2026-09-01 复现结论)** | — |


---

## 1. 环境

只有一个解释器能跑训练与评测,而且它不是任何 conda 环境;编码那一步另外需要 torch:

```bash
REPO=<repo 根目录>
JX=$REPO/.venv-jax/bin/python                                  # jax 0.4.33 + GPU
TORCH=/mnt/data/public_tools/miniconda3/envs/qwen-vl/bin/python  # 仅 §3.3 编码用
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export OMP_WAIT_POLICY=PASSIVE
```

☠️ `.venv-jax/bin/python` 是指向 conda `qwen-vl` 解释器的**符号链接**,但 `include-system-site-packages = false`,site-packages 完全独立——这就是 `conda activate qwen-vl && python` 看不到 jax 的原因。仓库 README 里的 `$JX` 指向 `/root/nas/...`,是另一台机器的路径,已失效。

☠️ **OMP 线程数必须限制。** torch 按 `nproc`(本机 250)给每个进程开线程池,多进程时严重超订,OMP 在 barrier 上忙等。实测:默认 GPU 占用 8–21%、load average 761;限制后 GPU 67–76%、load 9.2,**快 5 倍**。

---

## 2. 训练数据

`worldmemarena_wm_train.npz`,373 MB,24 个数组,自洽:

```
codes            (520220, 32, 32) uint8    值域 0..63
valid            (520220, 32) bool
history_indices  (495527, 32) int32        −1 补头，右对齐（尾部是近期）
target_indices   (495527,) int32           被预测状态的行号
episode_ids      (495527,) int32           分组连续步
steps            (495527,) int32           episode 内位置
action_*         11 个动作通道
pq_*             冻结解码器（5.13 MB）
metadata         JSON：协议、形状、schema
```

**没有 `split_ids` 列**:整个文件是 train-only,全部 495,527 条转移都是训练数据。
`scripts_wm_dataset.py unpack` 对无该列的文件按全 train 处理(带该列的旧格式仍兼容)。

**把码解回连续 32×512 状态:**

```python
from experiments.state_tokenizer.qformer_pq import decode
z = np.load("worldmemarena_wm_train.npz", allow_pickle=True)
xbar = decode(z["codes"][rows], z["pq_mean"], z["pq_scale"], z["pq_centroids"],
              bases=z["pq_rotation_bases"], order=z["pq_rotation_order"])
```

实测:用文件内解码器重建,R² = **0.9808**。

---

## 3. 四条命令

### 3.1 解包训练数据

```bash
"$JX" scripts_wm_dataset.py unpack \
  --dataset worldmemarena_wm_train.npz --output ./cache
```

**核对**:`transitions 495527`、`validation 0`、`fixed_width_bits 6144`、`max_history 32`、`num_categories 64`。

### 3.2 训练

```bash
CUDA_VISIBLE_DEVICES=0 "$JX" -m experiments.world_model.train \
  --cache ./cache \
  --config configs/world_model/web_h16_C64_full.yaml \
  --variant full --seed 0 \
  --output ./run --platform gpu --device-index 0 \
  --dev-fraction 0.05 \
  --max-steps 60000 --min-steps 5000 --eval-every 2500 --patience-steps 12500
```
`--dev-fraction 0.05` 从训练集里按 episode 留 5% 用于选点。

☠️ 三个会卡住的地方:

- **`--output` 目录必须不存在或为空**(`train.py:500`)。别提前在里面 `mkdir` 子目录。
- **`--resume` 会校验 `config_sha256`**;改了 `--max-steps` / `--patience-steps` 再 resume 会被拒。这个校验是对的,要换预算就重跑。
- `CUDA_VISIBLE_DEVICES=k` 配 `--device-index 0`。jax 只枚举可见设备,写绝对索引会 IndexError。

### 3.3 从官方仓库构建测试集

```bash
CUDA_VISIBLE_DEVICES=2 "$JX" scripts_build_wm_testset.py \
  --dataset-repo <WorldMemArena 官方仓库>/WorldMemArena \
  --checkpoint  <Q-Former checkpoint> \
  --codebook    <PQ 码本> \
  --output      ./testset \
  --jax-python "$JX" --torch-python "$TORCH" --device cuda:0
```

四步全自动:

```
官方 agent/gui/web/*.json
  → convert_wma        956 records / 139 episodes / 817 转移
  → Q-Former 编码       (956, 32, 512)
  → qformer_pq_apply   套冻结码本，不重拟合
  → cache_web          可评测的 cache
```

写出 `provenance.json`,记录**数据集仓库 remote 与 commit、Q-Former checkpoint 的 sha256、码本的 sha256、各阶段行数**。参考值:

```
remote  https://huggingface.co/datasets/LCZZZZ/WorldMemArena
commit  e2148757921fc7e2d66d8ed899823b763227c341
27 样本 → 956 状态 → 817 转移 → 139 episode
```

转换器切掉了 75 条 `cut_multi_action` 和 37 条 `cut_no_action`——**宁可切断 episode 也不给转移贴错动作标签**,所以 817 < 956−139。

☠️ **码本必须与训练数据内嵌解码器逐位一致,否则评测全废。** 存在两本同名的 `opq-shared-mix10-M32-C64.npz`:

```
pq-full/opq-shared-mix10-M32-C64.npz  sha256 bcc4c393…  ← 正确:与 npz 内嵌 pq_centroids 逐位相同
pq/opq-shared-mix10-M32-C64.npz       sha256 65826835…  ← 另一次拟合,与 npz 平均差 0.40
```

用错码本套码,码整体落错量化格子(逐位置 TV 距离中位 0.335,正确时应中位 0.000/p95 0.061),模型评测会贴着定宽 6144 徘徊且越训越差——数字看似"合理"实则全废。验收标准:npz 的 `pq_centroids` 与码本 `centroids` 逐位相同。正确码本不含 `codes/wma_web` 码行,可直接传,无需剥码行;`pq/` 那本自带一套自己编码的 wma_web 码行,与现编码撞键且逐位不一致,传它会报去重冲突。

### 3.4 评测

```bash
CUDA_VISIBLE_DEVICES=2 "$JX" -m experiments.world_model.evaluate \
  --cache ./testset/cache --config configs/world_model/web_h16_C64_full.yaml \
  --checkpoint ./run/best.pkl --split validation --output ./testset/eval \
  --platform gpu --device-index 0
```

基线数字(§4 表)来自**遗留混合 cache**(817 条 web 行标 validation、其余 train);当前 train-only npz 流程不再产出带 validation 行的 cache,基线复现需另行把测试集记录与训练码合并建 cache(可用 `cache_web` 多次传 `--codes` / `--records`),再:

```bash
"$JX" -m experiments.world_model.baselines \
  --cache <合并 cache> --output ./baselines --fit-split train --eval-split validation
```

☠️ `--output` 是**目录**不是文件。它写 `baseline.json` 进去,码率在 **`bits_per_transition` 这一层之下**;stdout 日志把 copy/marginal/source 放在顶层,读错会 KeyError,而且是在训练跑完之后的结果组装阶段才抛。

---

## 4. 预期数字与容差(2026-09-01 复现结论)

| | bits/转移 | 备注 |
|---|---:|---|
| 训练状态 / 转移(train-only) | 520,220 / 495,527 | npz 实测 |
| 测试状态 / 转移 / episode | 956 / 817 / 139 | 重建实测 |
| marginal 基线 | 6382.00 | 逐位复现 ✓ |
| copy 基线 | 6051.42 | 逐位复现 ✓ |
| source 基线 | 5420.84 | 逐位复现 ✓ |
| 世界模型(遗留混合 cache 协议,dropout 0.3) | 5427.6 | ≈ source,差值在噪声内 |
| 世界模型(本手册协议:train-only npz + 外部 testset,dropout 0.1) | **5082.8** | best=last @60000 步,压缩比 1.185 |


复现结论:

1. 三条基线逐位吻合,数据、码本、评测协议正确。
2. dropout 从 0.3 降到 0.1 后,外部 testset 码率 5977→5182.8(60,000 步),全程单调下降、best=last,选点无乐观偏差;而 dropout 0.3 在同协议下训到 60k 收敛于 ≈5427(持平 source)。正则强度是本语料上的主导超参。
3. 评测协议:训练与选点每 2500 步直接在 `--eval-cache`(外部重建 testset 的 817 条 validation)上进行,训练 cache 内无任何 web 协议行;436/817 字节级副本见结论 3。

---


## 附录:重建训练数据

只在需要改 tokenizer 或量化器时才用得上。

```bash
# 1. Q-Former 编码（$TORCH），8 卡约 3.8 小时，--shard-index i --shard-count 8
#    核对：state_ids 合计与 records 集合相等（不是数量相等，是集合相等）
#          metadata.checkpoint_metadata.qformer_sha256 前 16 位 = c9068916dce44ce0
#          ☠️ 那是 checkpoint 内部记录的 sha，不是文件 sha256（后者是 fb31e5ff…）

# 2. PQ 码本，34 分钟
"$JX" -m experiments.state_tokenizer.qformer_pq \
  --states "xbar.w*.npz" --records ... --output pq.npz \
  --num-subspaces 32 --num-categories 64 \
  --kmeans-iterations 25 --seed 0 --rotation shared \
  --mix-states "wma-fitcorpus.*.npz" --mix-share 0.10
#    核对：R² train 0.9876 / validation 0.9877，empty_clusters 0
#    ☠️ --opq-rounds 保持 0。迭代 8 轮能把 R² 推到 0.9899，但 bigram 5,542 vs 5,541,
#       压缩比一个 bit 不动，而换码本会让 65,536 个码嵌入全部失效、必须重训。

# 3. 建 cache
"$JX" -m experiments.world_model.cache_web \
  --codes pq.npz --codes codes-extra.npz --records ... \
  --output cache --max-history 32 --include-splits train validation
```

☠️ **建 cache 时任务词表长度必须是 1。** 模型的 `task_embedding` 形状是 `(1, 256)`,词表变 2 会在加载时形状不匹配。`num_tasks=1` 时那一行是常数偏置、不携带区分信息。

☠️ **`--max-history 32` 而模型 `max_history: 16`** 是有意的:一个 32 步的 cache 可以服务任何 ≤32 的模型,右对齐保留最后 k 个。`evaluate.py` 必须设 `cache.max_history = config.model.max_history`(`FrozenCache` 默认 `None`,不设就会把完整 32 步窗口喂给模型并报形状错)。
