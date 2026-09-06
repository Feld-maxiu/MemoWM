# HumanTrajs 训练数据流水线

## 结论

原始 HumanTrajs 可以用于 QFormer 的网页状态预训练，但不能直接作为可靠 QA 监督，也不能
代表完整 AMA-Bench。当前流水线把它限定为 **AMA-WEB warm-up 数据**：先修复时序对齐和
泄漏，再生成局部 QA，由只看 gold 截图的 verifier 初筛，最后将截图转写成伪 Web 文本并
做文本证据复筛。当前没有把未完成的转移/多步 QA 混入训练集。

旧的 `humantrajs-stepqa-gpu*.jsonl` 不进入训练。它们采用了错位截图、读取完整 instruction，
且未经独立证据验证。

## 1. 已确认的源数据语义

源文件同一行的截图是动作执行前状态，而 `other_obs` 更接近动作后的 URL/页面元数据。例如：

```text
step 1 action = goto(github.com)
step 1 screenshot = Incognito 起始页
step 2 screenshot = GitHub 首页
```

因此 v1 固定使用：

```text
memory_i = action_i + sanitized(other_obs_i) + screenshot_{i+1}
```

同时保留 `before_image_path=screenshot_i`，供转移 QA 和人工审计使用。最后一帧、终止动作、
无连续下一帧、缺失/损坏/近乎纯色的动作后截图均不进入 manifest。

## 2. 防泄漏规则

- action 只保留 `action_name`、`action_str`、`action_description`；
- 删除 `action_output.thought`、最终回答、`send_msg_to_user` 等终止动作；
- 超长 `action_description` 中的整页元素值被删除；
- instruction 可留作静态任务元数据，但所有 QA 生成器均禁止读取；
- verifier 只接收 QA 和声明的 gold 截图，不接收 instruction、动作或未来帧；
- 生成标签初始状态一律为 `pending`，只有 verifier 通过后才标记 `verified`；
- 数据按 `trajectory_id` 确定性切分，禁止按 QA 行随机切分。

相同截图在同一轨迹中形成 `equivalent_step_ids`。局部 QA 使用多正例；多步 QA 用
`gold_step_groups` 表示每一组必需证据及其等价正例，避免把完全相同的页面错标成负例。

## 3. 当前产物

```text
outputs/ama_latent_memory/manifests/
  humantrajs-post-action-v1.jsonl
  humantrajs-post-action-v1.report.json
  humantrajs-post-action-v1.rejections.jsonl
  humantrajs-post-action-v1.alignment-audit.jsonl
  humantrajs-pseudo-web-v1.jsonl
  humantrajs-pseudo-web-v1.report.json

outputs/ama_latent_memory/qa/
  humantrajs-local-v2.verified.jsonl
  humantrajs-pseudo-web-v1.verified.jsonl
  humantrajs-pseudo-web-v1.verified.report.json
```

构建命令：

```bash
.venv-ama-vllm-qwen35/bin/python \
  xt_ama_adapter/scripts/prepare_humantrajs_training_data.py \
  --input /data1/datasets/xt_ama_adapter/filtered/humantrajs-semantic-steps.jsonl \
  --output outputs/ama_latent_memory/manifests/humantrajs-post-action-v1.jsonl \
  --report outputs/ama_latent_memory/manifests/humantrajs-post-action-v1.report.json \
  --rejections outputs/ama_latent_memory/manifests/humantrajs-post-action-v1.rejections.jsonl \
  --alignment-audit outputs/ama_latent_memory/manifests/humantrajs-post-action-v1.alignment-audit.jsonl
```

## 4. QA 生成

先生成局部状态 QA：

```bash
.venv-ama-vllm-qwen35/bin/python \
  xt_ama_adapter/scripts/generate_step_qa_vllm.py \
  --input outputs/ama_latent_memory/manifests/humantrajs-post-action-v1.jsonl \
  --model /data1/models/Qwen3.5-9B \
  --mode local --split train \
  --output outputs/ama_latent_memory/qa/humantrajs-local-unverified.jsonl \
  --audit outputs/ama_latent_memory/qa/humantrajs-local-generation-audit.jsonl
```

转移 QA 使用同一脚本加 `--mode transition`。它会同时传入动作前后截图，并排除没有前序
memory step 或动作前帧低信息的记录。

多步 QA：

```bash
.venv-ama-vllm-qwen35/bin/python \
  xt_ama_adapter/scripts/generate_trajectory_qa_vllm.py \
  --input outputs/ama_latent_memory/manifests/humantrajs-post-action-v1.jsonl \
  --model /data1/models/Qwen3.5-9B --split train \
  --output outputs/ama_latent_memory/qa/humantrajs-multistep-unverified.jsonl \
  --audit outputs/ama_latent_memory/qa/humantrajs-multistep-generation-audit.jsonl
```

GPU 并行时，各进程必须使用互不重叠的 `--start/--end` 范围和独立输出文件；完成后再合并，
并按 `(trajectory_id, question, answer)` 去重。validation/test 的 QA 必须分别生成和保存。

## 5. 独立证据验证

```bash
.venv-ama-vllm-qwen35/bin/python \
  xt_ama_adapter/scripts/verify_humantrajs_qa_vllm.py \
  --input outputs/ama_latent_memory/qa/humantrajs-local-unverified.jsonl \
  --model /data1/models/Qwen3.5-9B \
  --output outputs/ama_latent_memory/qa/humantrajs-local-verified.jsonl \
  --audit outputs/ama_latent_memory/qa/humantrajs-local-verification-audit.jsonl
```

验收条件：`keep=true`、`answer_visible=true`、`gold_steps_sufficient=true` 且 score ≥ 4。
模型验证后还要按轨迹分层人工抽查至少 200 条；标签错误率必须低于 5%，否则修改 prompt 或
过滤规则并整批重跑，不能只手工修抽查样本。

## 6. 如何进入 QFormer 训练

- 所有清洗后的状态都可用于 observation reconstruction/KL/对比学习；
- 只有 `verification_status=verified` 的 QA 才能进入 QA CE 和 retrieval loss；
- local QA 的 `gold_step_groups` 只有一个必需证据组；
- transition/multistep QA 有两个及以上必需证据组，评估时要同时报告 group recall；
- train/validation/test 之间不得共享 trajectory；
- 先跑 AMA-WEB proof，不把网页数据上的提升外推到 TEXT2SQL、SOFTWARE、GAME、
  EMBODIED_AI 或 OPENWORLD_QA。

## 7. 停止条件

出现以下任何一项时，不启动 QFormer 正式训练：

- QA 仍读取完整 instruction 或 action thought；
- 使用 `screenshot_i` 监督 `action_i` 的动作后状态；
- 未经 verifier 的生成 QA 混入训练；
- validation/test 按 QA 行切分；
- 人工抽查错误率 ≥ 5%；
- 把 AMA 官方 test episode 用于生成训练标签或模型选择。

## 8. VL 伪 Web observation 协议

HumanTrajs 没有保存真实 DOM/AXTree。为匹配 AMA-WEB adapter 的文本输入，本实验采用
`vl-pseudo-web-observation-v1`：Qwen3.5-9B-VL 只读取当前截图、记录 URL 和页面标题，输出
有长度上限的 `page_title`、`visible_text`、`interactive_elements` 和 `page_summary`。
生成器看不到 instruction、action、thought、QA 或未来帧。

```bash
CUDA_VISIBLE_DEVICES=0 .venv-ama-vllm-qwen35/bin/python \
  xt_ama_adapter/scripts/generate_pseudo_web_observation_vllm.py \
  --input outputs/ama_latent_memory/manifests/humantrajs-post-action-v1.jsonl \
  --model /data1/models/Qwen3.5-9B --batch-size 32 \
  --output outputs/ama_latent_memory/pseudo_observations/pseudo.part0.jsonl \
  --audit outputs/ama_latent_memory/pseudo_observations/pseudo.part0.audit.jsonl
```

完成后用 `materialize_pseudo_web_manifest.py` 将唯一截图缓存回填到 memory manifest。截图
verifier 已通过的 QA 还不能直接用于文本训练，必须再运行
`verify_qa_against_pseudo_observation_vllm.py`。文本 verifier 只接收 `text_observation`、问题
和候选答案，必须给出一段原文连续证据；代码会再次验证证据确实是 observation 的子串。
validation/test 中跨 split 重复的截图直接剔除。

最终训练只读取同时满足以下条件的 QA：

- `qa_input_protocol=vl-pseudo-web-observation-v1`；
- `verification_status=pseudo_web_text_verified`；
- verifier 的 `supported=true`、`answer_entailment=true`、score ≥ 4；
- `evidence_quote` 通过确定性原文子串检查。

该协议是视觉转写文本，不是真实 AXTree，也不包含 DOM 层级、不可见属性或完整页面内容。
Qwen3.5-9B-VL 同时担任生成器和 verifier 会带来自洽性偏差，所以正式报告必须使用上述协议名，
并把真实 AMA-WEB test 结果与训练数据的自动通过率分开报告。

### 8.1 本次实际结果

- 6,525 个唯一截图全部生成成功；首轮 6,457，68 个长度截断样本提高到 2,048 tokens 后
  全部重试成功；
- 回填 6,750/6,750 个 memory 状态，无缺失；train 5,563 / validation 744 / test 443；
- 伪文本字符长度 p50=1,357，p95=2,238，max=4,029；标题和可见文本均无空值；
- 截图初筛 QA 4,780 条，纯文本证据复筛保留 3,371 条（70.52%）；
- 最终 QA：train 2,802 / validation 342 / test 227，覆盖 472 条 trajectory；
- split 间 trajectory 交集为 0，最终 QA 的跨 split 截图哈希为 0；
- 最终训练 JSONL 已移除 `instruction`，并写入 `instruction_excluded_from_training=true`；
- 拒绝 1,409 条：证据不是 observation 连续子串 965，文本不充分支持 441，verifier JSON
  无效 2，eval 跨 split 截图 1。

最终训练 QA 文件的 SHA256 为
`a45ba994b80f3196b4c476432e197ec07d245581aa87788a47fba280e836b793`；完整伪 Web manifest
的 SHA256 为 `1f67d22364a834ecd466499827c9b0321c05c3b668261403d574d883ab388b59`。
