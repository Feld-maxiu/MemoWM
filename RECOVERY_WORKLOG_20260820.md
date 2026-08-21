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

