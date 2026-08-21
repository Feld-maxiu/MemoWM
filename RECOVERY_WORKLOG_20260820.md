# 2026-08-20 两日前备份恢复记录

## 恢复基线

- 备份仓库：`czs/residual-mem`，原始 commit `097c92a`，工作树起初干净。
- benchmark：`czs/WorldMemArena`，数据链接到 `../WorldMemArena_hf_lfs`。
- v8 技术报告、STATE/WORLD MODEL worklog 与 handover 均在备份中。
- 备份没有 `models/`、`outputs/`，所以 checkpoint、PCA、A2、bridge 和历史评测 JSON 不可由源码逆推出。

## 已恢复源码

- Qwen3.5-9B-Instruct chat-template extraction，同时保留 v8 Base 默认模式。
- v9 frozen tokenizer runtime 与 Torch A2 inference（Qwen 环境无需导入 JAX）。
- WorldMemArena web observation serializer；assistant plan/action 强制排除。
- fused-observation teacher cache、共享 retrieval head、Input/L16 reader bridge 训练器。
- Raw-Fused、ObsOnly 和四个 ResidualMem-Instruct WMA adapters。
- native answer routing、canonical subcategory filter、配置注册。
- `scripts_v9_instruct.sh` 分阶段可重入入口。
- 数据集、训练协议、对照协议文档和协议测试。

## 已验证

- WMA loader：461 total samples；27 web samples；2,712 web turns；956 images；missing image = 0。
- qwen-vl 联合环境：WMA CLI dry-run 可启动，`--subcategory agent/arena/web` 正确进入配置。
- ResidualMem 新增测试：8/8 passed。
- WMA 新增测试：5/5 passed（基础环境与补齐依赖后的 qwen-vl 环境均通过）。
- 两仓库 `py_compile/compileall` 与 `git diff --check` 通过。

## 尚不能验证

- 本地没有 `Qwen3.5-9B-Instruct` 和 `Qwen3-VL-Embedding-8B` 权重；8013/8014 当前无服务。
- 没有 v9 PCA/normalization、A2 checkpoint、retrieval head 或 reader connector。
- 因而真实 screenshot -> tokenizer -> retrieval -> QA 的 GPU 端到端结果尚不能运行。
- 当前没有 JAX 训练环境；在线 A2 inference 已用 Torch port 解耦，但重新训练 A1/A2 时仍需按项目 requirements 建 JAX 环境。

## 建议重跑顺序

1. 用 ModelScope 恢复 `models/Qwen3.5-9B` 与 Qwen3-VL embedding 权重。
2. `scripts_v9_instruct.sh extract-full-h`，先用 1 个 rank/少量记录做真实模型门禁。
3. 完成 ranks 0–2；突发事件前的约定是先不要启动 ranks 3–5，待前半验证后再补。
4. rebuild Static Key64 -> PCA -> normalization -> A1/A2。
5. 启动 Qwen3-VL server，生成 fused teacher cache，训练 retrieval head/readers。
6. 先跑 1 个 web sample，核对每个非空 round 两行；再并行跑 27 个 web 样本。

## 2026-08-21 运行时恢复进展

- 用户补建的 `.venv-jax` 已在宿主机实测为 JAX/JAXLIB 0.4.33，能够枚举
  `CudaDevice(id=0)`；沙箱内只见 CPU 是设备隔离，不是环境损坏。
- Qwen3.5-9B 与官方 Qwen3-VL-Embedding-8B 权重已完整落盘并通过真实前向门禁。
- BrowserGym 0.14.3 采集环境和 Playwright 1.44 Chromium 已固化到仓库目录；
  MiniWoB++ 检出官方固定 commit `7fd85d71a4b60325c6585396ec4f48377d049838`。
- 真实 3-state 采集 smoke 通过：截图为 498×321，AXTree/DOM/probe/action 字段齐全；
  临时产物随后删除。
- 修复 `scripts_v9_instruct.sh`：正式 Full-H 改用 `extract_fixed_prompt`，其 source
  length 改由无 forward 的 `modality_lengths` 生成，rebuild 显式固定
  `--image-grid-thw 1 20 32`。
- 真实 1-state Instruct layer-16 smoke 通过：modality lengths `[160,178,29]`，
  Key64 layout `[32,16,16,0]`，merged image grid `[10,16]`，done/finite 全通过；
  临时 Full-H/Key64 产物随后删除。
- 两日前备份没有 100,008-state 源记录和截图，无法逆向恢复。目前按原
  12-worker、seed、7-step、0.5 random-action 协议重新采集；各 worker 使用
  `--resume` 和连续失败熔断，采集完成前不能伪称已重启正式抽取。

## 2026-08-21 高并发采集加速（进行中）

### 为什么不用 GPU

BrowserGym 采集阶段没有模型前向，只包含 Chromium 页面重置/交互、CDP AXTree/DOM
抽取、截图、PNG 编码与 JSON/图片写盘。GPU 渲染既不能加速 AXTree/CDP 主路径，
还可能改变截图像素；8 张 GPU 因此保留给后续 Qwen layer-16 抽取。正确的加速维度
是 250 CPU 核上的 episode 级并行。

### 切换前基线

原 12-worker 采集在 2026-08-21 11:54（启动约 56 分钟）达到
`4,218 / 100,008 = 4.218%`。12/12 collector 与 launcher 均存活，所有 shard
最近 33 秒内都有写入，错误/Traceback/重试日志为空。聚合吞吐约 75 states/min，
但每任务固定 8,334 states，`click-dialog-2` 与 `focus-text` 又是严格单状态 episode，
所以完成时间由这两个最慢任务决定；按**本次恢复环境**的速度线性外推约 2–3 天。
这只是当前重采的估算，不是原实验的历史耗时记录。

这里容易与 2026-08-03 的早期 pilot 混淆：当时 worklog 明确记录的是 Farama/
Selenium 路线、4 个 worker、共 **10,008 states**。当前要恢复的是后来冻结的 v8：
BrowserGym/Playwright、**100,008 states / 43,751 episodes**，数据量是前者约 10 倍，
而 BrowserGym 0.14.3 每次 `reset()` 还会分别启动 task 与 chat 两个 Chromium。
旧文档没有记录 v8 原始采集的完整 wall time，只记录过它在 25k 处因会话容器重建
中断；因此不能声称原来一定也花了 2–3 天。

### 加速协议

原 task `t` 的 episode 序列是 `episode_index = t + 12k`，环境 seed 是
`seed + t×1,000,000 + episode_index`。新方案只按 `k mod L` 分 lane：

```text
lane l: episode_index = t + 12l + n·(12L)
```

这不会改变任一 episode 的 id、seed、动作 RNG、split 或页面内容。每个 lane 完整
flush episode；异常 episode 丢弃并用同一 seed 重试。最终把所有 lanes 按
`(task_index, episode_index, step)` 排序，要求 episode lattice 无缺口、step 从 0
连续、seed/split/state_id 全匹配，然后精确保留每任务最早的 8,334 states，得到
100,008 states。旧 12-worker 数据保留为独立 reference；新 lanes 的重叠记录和截图
必须逐条/逐 SHA256 一致才能接受。

当前计划最多同时运行 228 lanes（不超过 240 的保护上限）：单状态慢任务各 32，
中慢任务 16–24，快速多状态任务 12。非单状态任务的 episode 预算根据前约 150 个
episode 的实测均值加 35% 余量；若 assembly 仍报告某任务不足，只等量增加该任务
所有 lane 的 `target_episodes` 并 `--resume`，不会更改已有 episode。

### 当前实现状态

- `collect_browsergym.py` 正在加入 `assigned_task_index / episode_start /
  episode_stride / target_episodes / shard_name`，legacy 默认路径保持不变。
- 新增 `assemble_browsergym_lanes.py`：负责 gap/duplicate/partial-step/seed/split/image/
  overlap 硬校验与精确裁剪。
- 新增 `scripts_v8_collect_lane.sh` 与 `launch_browsergym_lanes.py`：每 lane 独立日志、
  PID、resume、自愈，launch 上限 240。
- 尚未切换生产进程；必须先通过 parser/resume/assembler 单测和真实少量 overlap smoke，
  再安全停止旧 12-worker，避免用未经验证的加速器破坏唯一的恢复数据。

### 切换门禁结果

- parser / custom-stride resume / assembler 单测：`15 passed`。
- 真实 2-lane smoke：lane 0 采 episode `0,24`，lane 1 采 `12,36`；合并后正好
  覆盖原序列前四个 episode。7 条重叠 state 与旧 collector **逐字段完全一致**，
  7 张 screenshot 的 SHA256 也完全一致。
- resume smoke：把 lane 0 的目标从 2 增到 3，恢复逻辑先丢弃最后 episode 再用同一
  seed 重采，最终 episode 为 `0,24,48`，无重复/跳号；4 条 state 与旧记录及截图
  SHA256 再次完全一致。
- 两次临时 lane smoke 目录均已删除。

以上证据说明并行化只改变 episode 的执行顺序，不改变数据内容。下一步允许切换生产：
先记录并 SIGTERM 旧 12 个独立 process group，确认 Chromium 子进程退出；不删除
`outputs/state_tokenizer/v8`，将它保留为 overlap reference。新数据独立写入
`outputs/state_tokenizer/v8-lanes`。

### 生产切换与当前调优状态（2026-08-21 12:34）

- 已按精确 PGID 对旧 12-worker collector 做 `SIGTERM`，并确认对应 BrowserGym/
  Chromium 子进程退出。旧的 `outputs/state_tokenizer/v8` 完整保留（约 143 MB、
  12 个 shard），继续作为逐记录和截图 SHA256 reference，没有覆盖或删除。
- 12:30 首次启动完整 228-lane 计划，数据、日志、PID 分别写入
  `v8-lanes`、`v8-lane-logs`、`v8-lane-pids`。启动器确实拉起 228 个 lane，日志均
  进入 `attempt 1`，没有 traceback、OOM、exit 或 retry。
- 该档位的实测结果并不理想：启动约 4 分钟后仍为 228 个空 JSONL、0 state。
  lane 全部阻塞在首次 `env.reset()`/Chromium 初始化；同时系统 load 接近 250，产生
  大量 SwiftShader GPU-process。这里的 `GPU-process` 是 Chromium 的软件渲染子进程，
  不是 CUDA 采集加速。由此确认“进程数拉满”超过了浏览器启动/CDP 的有效并发点。
- 当前工作是把 launcher 改为**有上限的分批调度器**：保持已经验证过的 228-lane
  episode lattice 和最终数据完全不变，但只维持一批可稳定产出的 active lanes；完成
  后再补下一批。会先在 64 路档位测 5–10 分钟真实 states/min，再根据吞吐与错误率
  向上/向下调，而不是按 CPU 核数盲目并发。
- 接受新并发档位的硬门槛：必须持续产生完整 episode、无 traceback/连续失败，且随机
  抽取的旧数据重叠项继续满足 record 全字段一致、截图 SHA256 一致。稳定约 10 分钟后
  只保留后台任务，结束主动监控，以免无意义占用交互 token。

### 50k 采集决策与 browser 复用门禁（2026-08-21 13:02）

- 64 路、原始 BrowserGym reset 的生产实测：约 7 分钟累计 577 states，稳定均速约
  82 states/min，只比旧 12 路约 75 states/min 快 9%。即时 CPU 仍约 99% idle；原因是
  BrowserGym 0.14.3 每个 episode 都重新启动 task 与 chat 两个 Chromium，并非算力不足。
- 尝试把固定 `pre_observation_delay` 从 0.5 秒降为 0：12 task / 37 records 中出现
  8 条 AXTree 语义 sidecar 差异和 5 张截图差异，**门禁失败，生产保持 0.5 秒**。
- 新增 opt-in browser reuse：保留 0.5 秒观察时序，每个 episode 仍创建新的 task/chat
  incognito context，只复用两个 Chromium 进程。12 task × 3 episodes 的门禁在 45 秒内
  得到 114 records；114/114 截图 SHA256 一致，`dom`、instruction、action、probe 等
  tokenizer/WM 输入字段全部一致。
- 其中 18 条 `axtree_raw` 只在 CDP 的 `nodeId/parentId/childIds` 数值上不同。另跑完全
  **不复用** browser 的 12-task 对照后，同样出现 13 条仅 node id 不同、截图全等，证明
  这是 Chrome 跨进程本来就存在的内部编号波动，不是 reuse 改变了状态。assembler 因此
  只对这三类 id 做按节点顺序的规范化比较，AXTree 拓扑与 role/name/value/browsergym_id
  仍要求完全一致；新增回归测试确保语义变化会被拒绝。
- 用户决定本轮不再补齐冻结 v8 的 100,008 states，而采 **50k**。按每任务同额前缀，
  精确产物是 `ceil(50,000/12) × 12 = 50,004 states`。228 个确定性 lane identity 与
  episode lattice 不变，只把每 lane episode 预算缩放到原计划的一半；最终 assembly
  使用 `--target-states 50000`。
- parser/resume/assembler/50k-plan 回归：`17 passed`；50k dry-run 确认 228 pending、
  0 active、每任务 4,167 states。生产配置采用 browser reuse + 64 active 上限，启动后
  监控实际持续吞吐，再决定是否安全提高 active cap。
- browser reuse 进入稳态后连续两个区间达到约 4.2k、4.4k states/min，0 retry、0
  episode failure，已不是原先估计的“约 2 倍”，而是消除了每 episode 两次 Chromium
  冷启动后的数量级提升。用户随后将目标恢复为完整 v8；13:08 只重启 scheduler，把
  `--target-states` 从 50,000 改回 100,000，64 个 collector 未停、已有 episode 全部
  续用。最终 assembly 目标重新为每任务 8,334、总计 **100,008 states**。

### 采集完成、严格装配与 canonical v8（2026-08-21 13:38–15:24）

- 228/228 deterministic lanes 全部完成，共得到 128,364 个候选 states、53,112 个完整
  episodes；collector/retry/failure 均为 0，所有 BrowserGym/Chromium 生产进程已退出。
- 对两日前 partial reference 的 6,792 个重叠 states 做了全量审计：6,792/6,792 在规范化
  CDP `nodeId/parentId/childIds` 后语义完全相同；6,770 张截图逐字节相同。剩余 22 张只出现在
  task08/task09 的 t=0，差异局限于 scrollbar/resize-handle raster，最坏 112/159,858 像素
  （0.00070062），全图 uint8 MAE 最大 0.02297。fresh-browser 控制也会出现该 Chrome raster
  波动，因此本次 assembly 显式使用 changed fraction `0.001`、MAE `0.03`；代码默认仍为
  两者 `0` 的严格模式，不能静默放宽。
- assembly 从 128,364 个候选中按冻结顺序精确保留 100,008 条，即 12 个 task 各 8,334 条。
  产物位于 `outputs/state_tokenizer/v8-recovered`，截图目录链接到 lanes 的唯一图像副本。
- `records-merged.jsonl`、`full-721.jsonl`、`extend-100008.jsonl` 均为 100,008 条，state id
  唯一、global/final index 连续、截图缺失 0。7/2/1 episode split 为 train 70,018、validation
  20,011、test 9,979，episode 跨 split 为 0，old-train leakage 为 0；merged manifest SHA256
  为 `c61ee90c87684ec3873174849300a7443351d71b71301ff4718fec7936e45865`。
- 原 partial reference 已保存在 `outputs/state_tokenizer/v8-reference-partial-20260821`；canonical
  `outputs/state_tokenizer/v8` 现在是指向 `v8-recovered` 的符号链接，避免后续脚本误读旧数据。
- collector/resume/append/assembler/launcher 相关回归为 `18 passed`，`git diff --check` 通过。

### v9 Instruct modality metadata 与 Full-H 抽取（2026-08-21 15:24 起）

- 原 modality-lengths 实现对每条图片完整调用 multimodal processor，实测约 2 states/s/rank，
  仅为算 offset 却会耗时数小时。验证 12 个 task 首尾共 24 条真实记录后，确认固定 498×321
  screenshot 的 image token 恒为 160；直接对相同 chat-template 文本调用 tokenizer 得到的
  DOM/instruction 长度与完整 processor 逐项相等。因此改为：先用一张真实图片标定 image
  token，随后走 text-only fast path，任何超长记录自动回退完整 processor。
- 6 个 metadata ranks 各 16,668 条已完成，100,008/100,008 全局覆盖；image token 唯一值
  160，总长范围 238–971，full-processor fallback 为 0、truncation 为 0。`subset_rows` 与
  `record_indices` 都严格为 `rank, rank+6, ...`；`global_modality_lengths` 全量检查通过。
- 新增 `scripts_v9_extract_rank.sh`：每个 rank 只暴露一张指定 GPU，保存 PID/status/log，失败时
  使用既有 `done.npy` 做断点续跑。遵守突发事件后的约定，先只启动 rank 0–2，world size
  始终固定为 6；前三个 rank 完成并通过校验后，才按用户指令启动 rank 3–5。
- rank 0–2 首轮真实 Qwen3.5-9B-Instruct layer-16 抽取已稳定运行：每卡约 18.7 GiB，约
  12.0–12.5 states/s；连续观察 650 秒后分别完成 7,467 / 7,452 / 7,400（每 rank 共
  16,668），无 error/retry。`done.npy` 与 `key64-done.npy` 完全相等且完成前缀连续；抽查
  首/中/末已完成 Key64 均 finite、非零，早期 10 条 Key64 valid slots 为 49–59。按当前
  稳态速度每 rank 约 23 分钟完成；10 分钟稳定性门禁通过后保留后台任务并结束主动监控。
- rank 0–2 随后全部成功完成，各为 16,668/16,668；耗时 1,336.7 / 1,339.7 / 1,351.6 秒，
  吞吐 12.470 / 12.442 / 12.332 states/s。summary 均已生成，完成位完全一致，无残留进程，
  GPU 0–2 已释放；前三个 Full-H shards 实际占用约 210 GiB。
- rank 3–5 分别绑定 GPU 3–5 启动。连续观察 675 秒时进度为 7,760 / 7,730 / 7,700，吞吐
  12.195 / 12.127 / 12.093 states/s，每卡约 18.7 GiB、温度 60–64°C，无 error/retry。
  随后的完成位审计为 8,010 / 7,986 / 7,956：`done.npy == key64-done.npy`、已完成前缀
  连续，首/中/末 Key64 抽样均 finite 且非零。第二个 10 分钟稳定性门禁通过后停止主动
  轮询，三个 worker 留在后台继续完成。

### v9 Static-Key64、PCA 与 normalization 完成（2026-08-21 16:47–17:26）

- Full-H rank 3–5 最终也全部完成；六个 rank 合计 100,008/100,008，无缺失、重复或残留
  extractor 进程，Full-H 总占用约 420 GiB。随后六卡重建 Static-Key64，各 rank 均成功
  完成；冻结语义布局为 `(image, detail, context, prompt)=(32,16,16,0)`，Key64 modality
  layout 为 `(32,32,0)`，而不是恢复代码里一度残留的旧 `(32,28,4)`。
- 修复 `ragged_store.py` 的 legacy Key64 layout：现在从冻结 semantic layout 自动推导
  `(32,32,0)`；同步把 categorical bottleneck 测试的旧 detail slice `32:44` 修正为
  `32:48`。相关模块回归为 40/40，通过后总 JAX 回归等价为 80/80。
- PCA 在 train-only、task-balanced 2,000 states 上拟合，SHA256 为
  `6d2df9b56cfb4a2f06f25487d72f7e28b7d42213a8a59b56e00c3909ef001a26`，累计解释方差
  `0.9106246829`。六个 transform rank 合计 100,008/100,008，invalid slots 全为精确零；
  `static_features` 完整产物约 55 GiB。
- train-only group/channel normalization 已完成：artifact SHA256
  `1e20601d73a25c8ff9b93b5808b1a56b3b553ed7fd92b0e9b9e42dd627828b28`，runtime
  normalization hash 为 `d4d460b09158e9376fe75908dcc16eafba51c420051a8c526f460ce0bd519d4e`。
  train states 为 70,018；有效槽计数 image/detail/context/prompt 为
  2,240,576 / 492,624 / 495,647 / 0；只有空 prompt group 的 512 个 channel 按预期 clamp。

### A1 完成与 A2 全量 K-means OOM 修复（2026-08-21 17:42–18:18）

- 正式 A1 使用 `64×512` continuous bottleneck、10,000 固定步，成功生成
  `outputs/a1/v9-instruct{,.npz}`。best step 10,000；validation MSE
  `0.0013232167347897433`、R² `0.9986770169830488`，全部 integrity gates 通过，并且接近
  两日前 v8 的 R² `0.99891`。checkpoint 内 records/PCA/normalization hash 全部绑定本轮
  v9 产物。
- A2 第一次正式启动在 K-means 前发生 OOM：旧实现先把完整
  `(70018,64,512)` latent 留在 GPU，又将其整体转置为 2,048 个 PQ 问题，需要额外连续
  分配 9,177,399,296 bytes；这不是模型训练本身超显存。停止 wrapper 及其自动重试子进程
  后确认 GPU 0 已释放，且没有生成半成品 checkpoint。
- 修复后仍让 **全部 70,018 train states** 参与每个 K-means 问题，不做 subsampling：
  encoder 按 batch 将完整 latent 写入 FP32 host array，GPU 仅驻留 64 个 PQ 问题。
  K-means++ 使用 `seed + global problem id` 的 `per_problem_fold_in_v1` 协议，使 64/32 等
  不同 problem batch 得到相同中心。新增 blocked-vs-monolithic 精确等价测试；A2 完整 CPU
  回归 18/18 通过，`git diff --check` 通过。
- 已删除仅属于失败尝试的 3.3 KiB log、stale PID 和 stale status；没有删除 Full-H、
  Static-Key64、PCA、normalization 或两日前 reference，因为这些仍是恢复审计/重建的唯一
  可用上游，不属于已确认无用的临时文件。
- 18:16 在 GPU 0 重新启动正式 A2。70,018/70,018 host latent 和 2,048/2,048 K-means
  问题均完成，显存约 33.2 GiB（GPU 总计 72 GiB），无 OOM、retry 或 traceback；第 500
  步 validation MSE/R² 为 `0.2293378 / 0.7707027`。连续约 10 分钟稳定性门禁后已到
  21,500/70,000 步，MSE/R² 单调改善到 `0.1593744 / 0.8406538`，code perplexity
  中位数 `154.04`；GPU 0 仍约 33.2 GiB、48°C，进程为 `running`。后台 PID、status 和日志由
  `outputs/state_tokenizer/v9-instruct/{pids,status,logs}/train-a2.*` 管理。

### PCA 恢复纠错与 Xbar-only retrieval（2026-08-21 19:31–20:27）

- 复核旧 worklog 后确认冻结协议是 **20,000 train states**。恢复阶段先前生成的
  2,000-state PCA 并非日志所称的 task-balanced：legacy first-N selector 使 2,000 条
  全部来自 `miniwob/click-button-v1`。该说法已在本节纠正，旧 artifact 保留供审计但
  不再作为正式坐标。
- `key64_pca.py` 新增确定性的 `task_balanced` selector 和回归测试。正式 variant
  `outputs/state_tokenizer/v9-instruct-pca20k-balanced` 在 12 task 各 1,666–1,667 个
  train state 上拟合；PCA SHA256 `f98c517cb23e778b78eb04080ddc01b375a2efe9a42be8992e4a5e8a8f6efb2a`，
  解释方差 `0.9185917377`。六卡 transform 覆盖 100,008/100,008；finite、invalid-zero、
  index 唯一性全量门禁通过。新 normalization SHA256
  `648135a1b065f9fc6c1e87389f105d9e4c2990839c04e6e23dd8a3546e2b7e57`。
- 用户决定本轮不重跑 A1/A2，直接进入连续 Xbar retrieval。旧 A2 checkpoint 的坐标
  hash 与新 PCA 不同，因此没有被混入 cache。bridge builder/trainer 新增向后兼容的
  `representation=xbar`；cache 无 `a2_xbar`，loss 仅为 symmetric InfoNCE + cosine。
- `memcompiler` 环境的 vLLM 0.11.0 与其 Torch CUDA ABI 不匹配，服务在模型加载前退出、
  未占 GPU。随后改用官方 SentenceTransformer 6.0 本地 encoder；其原生 spawn pool 因
  checkpoint 动态函数不可 pickle，改为每 GPU 一个独立 deterministic shard process。
  2-GPU/24-state smoke 和 8-GPU 正式 5,500-state cache 均成功，无残留 worker。
- 正式 cache 位于
  `outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/train-cache-fused-observation.npz`
  （5,000 train + 500 validation，518 MiB，SHA256 `b5226f6d...c027d9`）。Xbar/teacher
  均 finite，invalid slots 精确零，5,500 个 global index 全唯一。
- retrieval head 在 GPU 0 训练，step 5,000 达到最佳 validation loss `0.6449992`、
  paired cosine `0.8711105`；连续 12 次无改善后 step 8,000 early-stop，保存的是 best
  checkpoint。500-way validation Recall@1/5/10=`0.840/0.956/0.986`，MRR=`0.8946`；
  checkpoint SHA256 `92e2c61e...ffeb96`。

### WorldMemArena 跨域 retrieval smoke（2026-08-21 20:23–20:27）

- 新增无 QA judge 的严格 smoke：官方 Qwen3-VL 在 GPU 0，Qwen3.5 tokenizer/head 在
  GPU 1；Raw 与 Xbar 共享完整 round text row、query prompt、global cosine 与 top-10。
- `web_01` 首 checkpoint（5 sessions、2 observations、10 questions）：overall
  Raw/Xbar Recall@10 都为 `0.90`，top-10 overlap `0.89`；但 Xbar observation row 从未
  进入 top-10，2-state paired cosine 只有 `0.2174`。这证明 overall 指标受到共有 text
  row 掩盖，不能当作压缩通过。
- 决定性复核使用最终 checkpoint（25 sessions、49 full-round rows、14 observations、
  10 questions）：Raw/Xbar Recall@10=`0.765/0.725`，NDCG@10=`0.5429/0.5529`，overall
  top-10 overlap=`0.91`、top-1 agreement=`1.00`。然而 100 个 top-10 槽位中 Raw 命中
  8 条 fused observation，Xbar 命中 0 条；14-state paired cosine=`0.1836`，14-way
  Recall@1/5/10=`0.0714/0.5714/0.8571`，MRR=`0.2958`。Recall@1 等于随机量级，确认
  当前 synthetic-AXTree 零训练适配存在显著 MiniWoB→WMA 分布偏移。
- 结论：PCA/head 的域内门禁通过，但 WMA observation 压缩门禁失败。在修正域适配前，
  不应启动全量 web benchmark；共有 full-round row 会把总体 retrieval 指标伪装得很好。
  逐问报告保存在 `outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/` 的两个
  `worldmemarena-*.json` 中。实验结束后 8 张 GPU 均为 0 MiB、无模型/worker 残留。
