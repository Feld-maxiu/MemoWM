# ResidualMem 技术报告

> 基于世界模型条件编码与任务效用门控的长期智能体记忆
> 文档版本：1.2
> 实现快照：2026-09-07

## 摘要

ResidualMem 将长期记忆表示为相对于世界模型预测的任务相关修正，而不是逐时刻保存完整观察。当前系统实现了以下链路：

```text
观察前端（WMA：截图 + 文本；AMA-WEB：AXTree）
  → 冻结 Qwen3.5-9B 中间层状态
  → 32 槽 Q-Former 状态 tokenizer
  → semantic latent
      ├→ OPQ 离散码（32 × 32，C=64）
      │   → 块因果 Transformer 世界模型
      │   → 任务效用门控与连续状态重建
      ├→ 压缩感知检索头 → memory key
      └→ Reader Bridge → soft tokens → frozen Reader
  → [可选] 精确符号 anchor → Reader token embeddings
```

在 WorldMemArena Web 外部评测集的 817 条转移上，每状态定宽表示为 6,144 bit。世界模型条件编码将其降到 5,182.83 bit；效用门控在闭环条件下降到 4,323.84 bit，即相对定宽减少 29.63%，相对世界模型全发码率减少 16.57%，压缩比为 1.421×。门控造成的答案绝对 NLL 变化为 0.103 bit；作为参照，OPQ 量化自身造成 0.148 bit 的答案绝对 NLL 变化。

从官方 WMA Web JSON 重新构建 Q-Former 状态、OPQ 码、闭环 WM+utility 重建并执行检索、soft-token Reader 和 Judge 后，完整端到端链路在 1,459 道题上得到 QA-C 0.5984（873/1459）。full-xbar Reader 参照为 0.6004（876/1459）。

检索侧使用 full/gated 双视图训练独立检索头。在完全冻结 Q-Former、OPQ、世界模型和 utility artifact 的条件下，选定模型在 gated validation 上达到 R@1 0.2274、R@10 0.6560、MRR 0.3697；相同数据上的旧检索头分别为 0.2226、0.6488、0.3622。该损失用于约束压缩前后的检索接口一致性。

本文只描述方法、实际实现、实验协议、结果与产物接口。WMA 全链路的运行命令和环境配置统一维护在 `WMA_RESIDUAL_复现手册.md`；效用门控见 `UTILITY_GATE.md`，AMA-WEB 的复现环境与命令见 `AMA_Bench_复现手册.md`。

---

## 1. 问题定义

### 1.1 长时程交互

将智能体轨迹写为

$$
\tau=(o_0,u_0,o_1,u_1,\ldots,o_T),
$$

其中 $o_t$ 是由评测前端提供的观察：WMA 前端以截图及配套文本为主，AMA-WEB 前端以序列化 AXTree 为主；$u_t$ 是在下一观察到达前已知的动作及其参数。后续查询记为 $q$，答案记为 $y$。

ResidualMem 的写入端必须与查询无关：

$$
x_t=E_\rho(o_t),
$$

同一状态只编码和存储一次，可被任意后续查询复用。训练问题在 Q-Former 学习阶段作为观察读出探针，并在 utility 学习阶段定义信息价值；在线写入时，问题不作为 Q-Former 输入，也不改变单条记忆的编码结果。

### 1.2 优化目标

记忆系统的目标是在存储预算约束下保持任务性能：

$$
\min_\psi\ \mathbb E_{\tau,q}
\left[
\mathcal L_{\rm task}(W_\theta,M_\psi(\tau),q)
+\beta\,\operatorname{Bits}(M_\psi(\tau))
\right].
$$

ResidualMem 将状态信息分为两部分：

- 世界模型能够从历史和动作恢复的条件先验；
- 为恢复当前状态而需要写入外部记忆的修正码。

因此系统的核心关系是

$$
\text{reconstructed state}
=\operatorname{Decode}(\text{WM prior},\text{transmitted correction}).
$$

这里的“残差”是相对于先验分布的离散修正，不是连续向量减法。

### 1.3 设计约束

系统遵循三个约束。

1. **因果性**：世界模型只能条件于已到达的观察码、已知动作和任务标识。
2. **编码/解码对称性**：门控判据只能依赖双方都可得到的量；否则需要额外发送逐状态 mask。
3. **任务相关性**：预测困难不直接等于值得保存。发送决策同时考虑任务效用与世界模型不确定性。

---

## 2. 系统流程

### 2.1 写入路径

对每个观察 $o_t$，系统执行：

1. 冻结的 Qwen3.5-9B 读取截图、AXTree 和固定观察提示，输出第 16 层隐状态 $H_t\in\mathbb R^{N_t\times4096}$。
2. StateQFormer 将可变长 $H_t$ 重采样为

   $$
   x_t\in\mathbb R^{32\times512}.
   $$

3. 冻结 OPQ 编码器产生

   $$
   z_t^+\in\{0,\ldots,63\}^{32\times32}.
   $$

4. 世界模型根据解码端可见历史 $\tilde z_{<t}$ 和动作 $u_{<t}$，输出 1,024 个类别分布

   $$
   p_{t,j}(c)=p_\theta(z_{t,j}=c\mid \tilde z_{<t},u_{<t}).
   $$

5. utility gate 计算每个位置的发送决定 $m_{t,j}\in\{0,1\}$。保留位置发送真实码；丢弃位置由解码端填入世界模型众数。
6. 重建状态 $\tilde x_t$ 进入检索头并形成记忆 key；被发送的离散码构成语义状态内容。
7. 可选精确符号旁路在同一次写入中生成 $a_t=A_\eta(o_t)$，并按 observation identity 与语义状态绑定。AXTree 前端抽取元素字段，截图前端使用轻量 OCR 抽取少量精确值；$a_t$ 不进入 OPQ、世界模型或检索头。

首状态不存在因果先验，当前协议对该状态执行全发。

### 2.2 解码与状态推进

解码码字定义为

$$
\tilde z_{t,j}=
\begin{cases}
z_{t,j}^+, & m_{t,j}=1,\\
\arg\max_c p_{t,j}(c), & m_{t,j}=0.
\end{cases}
$$

连续状态由同一个冻结 OPQ 解码器恢复：

$$
\tilde x_t=D_{\rm OPQ}(\tilde z_t).
$$

$\tilde z_t$ 随后进入解码端历史。后续世界模型分布始终条件于已经重建的历史，而不是不可见的真实历史，这构成闭环评测协议。

### 2.3 读取路径

当前读取路径包含检索和生成两部分：

1. `MaskedAttentionRetrievalHead` 将 $\tilde x_t$ 投影到 4,096 维归一化 key；
2. 查询由 Qwen3-VL-Embedding-8B 编码为归一化向量；
3. 在 session 内对全部记忆行计算余弦相似度并取 top-10；
4. 被命中的 latent 行经 Reader Bridge 转成 soft token；启用精确符号旁路时，相同行的 $a_t$ 经冻结 Reader 的 tokenizer 和 input embedding 转成普通 token embedding，并与 soft token、问题模板沿序列维拼接后通过 `inputs_embeds` 输入 Reader。

WMA Reader 参考实验使用 float32 `xbar` 作为在线池内容；码率主结果由完整 codec、utility gate 和闭环协议计算，压缩感知检索结果由 full/gated 重建 cache 计算。三类数值分别描述 Reader、codec 和检索接口，不跨协议相乘。

---

## 3. 组件实现

### 3.1 Q-Former 状态 tokenizer

#### 结构

选定 tokenizer 的配置如下：

| 配置项 | 值 |
|---|---:|
| trunk | 冻结 Qwen3.5-9B，第 16 层 |
| 输入宽度 | 4,096 |
| learned queries | 32 |
| Q-Former hidden | 1,024 |
| 层数 / 头数 | 4 / 8 |
| 输出形状 | $32\times512$ |
| QK normalization | 开启 |
| query self-attention | 关闭 |
| checkpoint 内参数量 | 78,872,576 |

每层执行 pre-norm cross-attention 和 FFN。query 同时访问完整截图、AXTree 和观察提示对应的隐状态，因此输出槽按内容寻址，而不是按固定空间网格池化。所有 32 个输出槽均有效。

checkpoint 同时包含 Q-Former、soft-token connector 和联合训练时的 retrieval head。当前部署使用同一 checkpoint 内的 connector，但检索使用单独冻结的 utility-gated head。

#### 多粒度观察读出目标

Q-Former 的训练目标统一定义为多粒度观察读出监督。训练查询集合由两部分组成：

$$
\mathcal Q_{\rm obs}
=
\mathcal Q_{\rm holistic}
\cup
\mathcal Q_{\rm focal}.
$$

$\mathcal Q_{\rm holistic}$ 包含询问可见内容、页面布局、输入值和控件状态的整体观察探针；$\mathcal Q_{\rm focal}$ 包含由 task QA 提供的重点细节问题。后者同样以当前观察为依据，但对任务相关、信息密度较高的局部状态提供更密集的监督。因此，两类问题的区别是观察读出的粒度与采样重点，而不是“状态建模”和“下游任务”两个彼此独立的目标。

对于任意观察读出问题 $r$，完整观察 teacher 和 latent student 分别定义

$$
p_T(\cdot\mid o,r)
$$

与

$$
p_S(\cdot\mid E_\rho(o),r).
$$

统一的观察读出目标可写为

$$
\mathcal L_{\rm readout}
=
\mathbb E_{r\sim\mathcal Q_{\rm obs}}
D\!\left(
p_T(\cdot\mid o,r),
p_S(\cdot\mid E_\rho(o),r)
\right).
$$

这里的 $D$ 表示与读出类型对应的分布差异；具体实现由下面的整体状态蒸馏、重点细节监督和跨观察辨识三部分展开。

在实际实现中，整体状态问题由 teacher 自生成续写上的分布蒸馏监督；重点细节问题同时使用金标答案交叉熵和完整观察 teacher 的答案分布蒸馏。再加入跨观察辨识目标后，总损失写为

$$
\mathcal L_{\rm QF}
=
\mathcal L_{\rm holistic}
+
\mathcal L_{\rm focal}
+
\mathcal L_{\rm discrimination},
$$

其中

$$
\mathcal L_{\rm holistic}
=
0.5\mathcal L_{\rm KL,obs},
$$

$$
\mathcal L_{\rm focal}
=
\mathcal L_{\rm CE}
+
0.3\mathcal L_{\rm KL,q},
$$

$$
\mathcal L_{\rm discrimination}
=
1.0\mathcal L_{\rm sem}.
$$

各实现项的含义为：

- $\mathcal L_{\rm KL,obs}$：基于整体观察探针和 teacher 自生成续写的状态读出蒸馏；
- $\mathcal L_{\rm CE}$：重点细节问题上基于金标答案的 teacher-forced cross entropy；
- $\mathcal L_{\rm KL,q}$：重点细节问题上完整观察 teacher 与 latent student 的答案分布蒸馏；
- $\mathcal L_{\rm sem}$：使用同 session 负样本的 symmetric InfoNCE，温度为 0.05，用于保持不同观察之间的可辨识性。

task QA 在这里是训练期的 observation readout probes，而不是 Q-Former 的输入条件。无论训练时使用哪一种读出问题，写入端始终只计算

$$
x=E_\rho(o),
$$

而不计算依赖问题的 $E_\rho(o,q)$。因此，一个观察只产生一份 query-independent latent，并可被不同后续问题复用。

观察 teacher 的分布离线存成 top-128。70 万个被缓存位置的教师概率质量中位数为 0.999987。训练使用 P1–P3 三种整体观察探针；P4 不进入梯度训练，只用于 checkpoint 选择和保真评测。

选定 checkpoint 为 `qformer-K32e-obs0.5.gapbest.pt`，对应 step 6000、validation answer CE 1.7572，并按观察保真 gap 选择。

#### 可选精确符号旁路

Q-Former latent 是默认状态表示。对于要求逐字符保真的输入，写入端可并行生成与查询无关的稀疏符号序列

$$
a_t=A_\eta(o_t).
$$

$A_\eta$ 随输入模态适配：AMA-WEB 从 AXTree 保留元素 ID、role 和可见名称；截图输入使用轻量 OCR，仅过滤并保存页面索引、日期时间、URL、带标签计数和带类型数值等精确值。该序列按 observation identity 与 $x_t$ 绑定，不进入 OPQ、世界模型或检索 key。

检索仍完全由重建语义状态 $\tilde x_t$ 完成。仅当第 $t$ 条 latent 被命中时，Reader 才将其 soft token 与该条记忆的原生 token embedding 拼接：

$$
H_t^{\rm mem}
=
\left[C_\phi(\tilde x_t);E_R(a_t)\right].
$$

anchor 是可选的离散 memory payload；启用时按实际符号码长独立计费，不作为共享模型参数或检索索引计费。

### 3.2 OPQ 离散状态

Q-Former 状态先按槽和通道标准化，再经过一个共享的 $512\times512$ 正交旋转；旋转后的通道按方差均衡顺序切为 32 个 16 维子空间。每个槽、每个子空间独立执行 $C=64$ 的 k-means 量化：

$$
z_{t,s,g}^+
=\arg\min_{c\in\{0,\ldots,63\}}
\left\|r_{t,s,g}-e_{s,g,c}\right\|_2^2,
$$

其中 $s\in[1,32]$ 为槽，$g\in[1,32]$ 为子空间。

当前码本参数为：

| 配置项 | 值 |
|---|---:|
| 槽数 | 32 |
| 每槽子空间数 | 32 |
| 子空间维度 | 16 |
| 类别数 $C$ | 64 |
| 总离散位置 | 1,024 |
| 定宽码率 | $32\times32\times6=6,144$ bit |
| k-means 轮数 / seed | 25 / 0 |
| 拟合状态数 | 520,110 |
| 空簇数 | 0 |

当前全量码本的重建指标为 train $R^2=0.98762$、validation $R^2=0.98767$。世界模型训练数据内嵌解码器上的总体重建 $R^2$ 为 0.9808。

OPQ 在拟合后冻结。世界模型、utility 位置定义和检索双视图都以该码本的 1,024 个离散坐标为共同接口。

### 3.3 离散世界模型

世界模型的任务不是直接回归连续状态，而是预测下一状态 1,024 个离散位置的条件分布。给定最多 16 步的重建历史和与转移对齐的动作，模型计算

$$
p_\theta(z_t\mid \tilde z_{<t},u_{<t})
=
\prod_{s=1}^{32}
\prod_{g=1}^{32}
p_\theta(z_{t,s,g}\mid \tilde z_{<t},u_{<t}).
$$

所有位置共享同一个时序 Transformer 上下文，但在输出层并行产生类别分布，不在当前状态的 1,024 个位置之间再引入自回归顺序。

#### 输入表示

历史离散状态张量的形状为

$$
\tilde Z_{<t}
\in
\{0,\ldots,63\}^{H\times32\times32},
\qquad H\le16.
$$

码嵌入参数按槽、子空间和类别分别定义：

$$
E^{\rm code}
\in
\mathbb R^{32\times32\times64\times8}.
$$

在历史时间 $k$ 的槽 $s$ 上，32 个子空间各查找一个 8 维 embedding，再按子空间顺序拼接为 256 维状态 token：

$$
e^{\rm state}_{k,s}
=
\operatorname{Concat}_{g=1}^{32}
E^{\rm code}_{s,g}(\tilde z_{k,s,g})
\in
\mathbb R^{256}.
$$

每个 token 的最终输入由六部分相加：

$$
h^{(0)}_{k,s}
=
e^{\rm state}_{k,s}
+e^{\rm valid}_{k}
+e^{\rm action}_{k}
+e^{\rm task}
+e^{\rm time}_{k}
+e^{\rm slot}_{s}.
$$

其中：

- $e^{\rm valid}_{k}$ 由该状态的 32 维槽有效向量线性投影得到，并广播到该时间步的全部槽；当前 Q-Former 的 32 个槽恒定有效，但该通道保留在模型协议中；
- $e^{\rm task}$ 是任务标识 embedding；当前锁定 cache 中只有一个任务标识；
- $e^{\rm time}_{k}$ 表示 16 步窗口内的相对时间位置；
- $e^{\rm slot}_{s}$ 区分同一时间块内的 32 个状态槽；
- 不存在的历史时间使用 `state_pad` 和 `action_pad` 表示，并在 attention 中屏蔽。

训练样本以状态转移为单位构造。预测 $z_t^+$ 时，窗口中保存的是

$$
(z_{t-H}^+,u_{t-H}),\ldots,(z_{t-1}^+,u_{t-1}),
$$

其中 $u_{t-1}$ 是从最后一个历史状态到目标状态 $z_t^+$ 的动作。短于 16 步的 episode 前缀在左侧填充，因此最后一个输入时间块始终对应 $z_{t-1}^+$ 及其出边动作，模型输出始终监督到紧邻的下一状态。

动作输入不使用自然语言序列直接拼接，而是先编码为一个 256 维条件向量：

$$
e^{\rm action}_{k}
=
W_a\operatorname{Concat}
\left(
e^{\rm type}_{k},
e^x_k,
e^y_k,
e^{\rm delta}_k,
e^{\rm payload}_k
\right).
$$

动作类型共有 17 类。绝对坐标 $x/y$ 分别量化为 64 桶，缺失坐标使用额外的 mask 类别；滚动等相对位移通过带符号对数变换后映射为 32 维；最长 40 byte 的 payload 经 16 维 byte embedding 和 64 维 GRU 编码。当前配置不使用 target 文本通道。

#### Block-causal Transformer

16 个时间步与每步 32 个槽被展平为最长 512 个 token。attention mask 只在时间维上保持因果性：

$$
M\big((k,s),(l,r)\big)=1
\iff
l\le k.
$$

因此，一个槽可以访问当前时间块内的全部 32 个槽以及此前所有有效时间块，但不能访问未来时间块。这种 block-causal 结构同时保留了状态内部的跨槽交互和轨迹级因果性。

模型配置为：

| 配置项 | 值 |
|---|---:|
| 历史窗口 | 16 步 |
| Transformer 层数 / 头数 | 12 / 8 |
| $d_{\rm model}$ | 256 |
| MLP 宽度 | 4,096 |
| dropout | 0.1 |
| code embedding | 每子空间 8 维 |
| copy gate | 开启 |
| 参数量 | 29,092,096 |
| 训练步数 | 60,000 |

每个 Transformer block 使用 pre-norm multi-head self-attention、残差连接和 GELU MLP。模型完成 12 层计算后，只取最后一个有效时间块的 32 个 hidden states 预测下一状态；下文将这组 next-state prediction features 记为 $h_{t,s}$。

#### 离散输出与 copy mixture

每个 256 维槽状态被重新解释为 32 个 8 维子空间向量，并与输入 code embedding 权重绑定：

$$
\ell_{t,s,g,c}
=
\left\langle
h_{t,s,g},
E^{\rm code}_{s,g}(c)
\right\rangle
+b_{s,g,c}.
$$

除普通类别 logits 外，每个位置还预测一个复制概率

$$
\pi_{t,s,g}
=
\sigma\left(a_{s,g}^{\mathsf T}h_{t,s,g}+d_{s,g}\right).
$$

最终分布是“复制上一状态对应码”和“重新预测类别”的概率混合：

$$
p_{t,s,g}(c)
=
\pi_{t,s,g}\mathbf 1(c=\tilde z_{t-1,s,g})
+
(1-\pi_{t,s,g})
\operatorname{Softmax}(\ell_{t,s,g})_c.
$$

copy mixture 为稳定不变的位置提供短码，同时保留对任意新码的完整支持。模型另有一个槽有效性 head：先对最后时间块的 32 个 hidden states 求均值，再一次性预测目标状态的 32 个槽是否有效，

$$
q_t
=
\sigma\left(
W_v\frac1{32}\sum_{s=1}^{32}h_{t,s}+b_v
\right)
\in(0,1)^{32}.
$$

令 $v_{t,s}\in\{0,1\}$ 为目标槽有效标记，$q_{t,s}$ 为上述 head 的预测概率，则实现中的世界模型目标为

$$
\mathcal L_{\rm valid}
=
-\sum_{s=1}^{32}
\left(
v_{t,s}\log q_{t,s}
+(1-v_{t,s})\log(1-q_{t,s})
\right),
$$

$$
\mathcal L_{\rm code}
=
-\sum_{s=1}^{32}
\sum_{g=1}^{32}
\log p_{t,s,g}(z^+_{t,s,g}),
$$

$$
\mathcal L_{\rm WM}
=
\frac{\mathcal L_{\rm valid}+\mathcal L_{\rm code}}{\ln2}.
$$

该目标直接以 bit 表示条件码长。当前 Q-Former 的 32 个槽恒定有效，且实现不以 invalid 标记移除 code loss，因此主结果中的 1,024 个码位置全部监督和计费；有效性项在外部评测上只有 $2.53\times10^{-5}$ bit/转移。utility mask 与槽有效性 head 是两个不同机制：前者选择是否发送真实码，后者描述表示中是否存在该观察槽。

世界模型训练使用真实历史码；在全发协议下，解码端历史与真实历史相同。在 utility gate 的闭环评测中，模型输入改为此前已经由“真实发送码 + 世界模型众数填补”得到的重建历史，从而显式计入误差传播。

### 3.4 任务效用门控

任务效用门控在离散状态的每个码位置决定是否发送真实 OPQ 码。它不把“难预测”本身当作重要性，而是显式比较省略该码对任务读出的影响与发送该码的预期成本。该设计把 task-dependent 的 reader 测量放在离线阶段，把在线决策限制为编码端和解码端都可复现的查表与后验计算。

#### 离线单位置反事实标注

令样本 $i$ 的完整 OPQ 码为 $z_i^+$，对应的离线观察读出问题和金标答案为 $(q_i,a_i)$。世界模型只根据在该状态之前双方共享的重建历史和动作产生后验：

$$
p_{i,j}(c)=p_\theta(z_{i,j}=c\mid \tilde z_{i,<t},u_{i,<t}),
$$

并给出位置 $j$ 的众数填充值

$$
\hat z_{i,j}=\arg\max_c p_{i,j}(c).
$$

第一阶段对选中的位置 $j$ 构造单位置反事实：只把该位置替换为世界模型众数，其余位置均保持真实码，

$$
z_i^{(-j)}
=
(z_{i,1}^+,\ldots,z_{i,j-1}^+,\hat z_{i,j},z_{i,j+1}^+,\ldots,z_{i,1024}^+).
$$

完整条件和反事实条件分别解码为

$$
\hat x_i^{\rm full}=D_{\rm OPQ}(z_i^+),
\qquad
\hat x_i^{(-j)}=D_{\rm OPQ}(z_i^{(-j)}).
$$

两个重建均经过冻结的 Reader Bridge 和 Reader。以金标答案全部 token 的 teacher-forced NLL 作为任务损失：

$$
\mathcal L_{\rm ans}(q_i,a_i\mid x)
=
-\sum_{r=1}^{|a_i|}
\log p_\psi(a_{i,r}\mid a_{i,<r},q_i,x).
$$

位置级带符号效用定义为

$$
U_{i,j}
=
\frac{
\mathcal L_{\rm ans}(q_i,a_i\mid \hat x_i^{(-j)})
-\mathcal L_{\rm ans}(q_i,a_i\mid \hat x_i^{\rm full})
}{\ln2}.
$$

除以 $\ln2$ 将 NLL 统一为 bit。反事实的参考必须是 $D_{\rm OPQ}(z_i^+)$，而不是量化前连续 Q-Former 状态：这样两条分支共享相同的 OPQ 误差，损失差只归因于位置 $j$ 由真实码替换为可由解码端生成的填充值。

标注在 float32 下完成。同一状态—问题行的完整参考和所有反事实变体使用同一冻结 reader、相同问题模板、相同答案 token 上限和相同 batch 几何；每个前向还包含与参考相同的 null control，用来拒绝数值漂移。位置采样按状态—问题行分片，并优先补齐 1,024 个坐标的覆盖；这使反事实代价可控，同时避免只把标签集中在少量 OPQ 槽上。

#### 任务效用分解：错误概率 × 错误伤害

令 $\mathcal A_j$ 是所有对位置 $j$ 完成单位置反事实标注的状态—问题样本，部署 utility 取绝对损失变化的样本均值：

$$
|U_j|
=
\frac{1}{|\mathcal A_j|}
\sum_{i\in\mathcal A_j}|U_{i,j}|.
$$

该均值可以精确分解为错误概率与错误伤害两个因子。当世界模型众数恰好等于真实码（$\hat z_{i,j}=z_{i,j}^+$）时，反事实替换是恒等操作，$|U_{i,j}|=0$；只有众数预测错误的样本才贡献非零项。因此

$$
|U_j|
=
\bar e_j\,V_j,
\qquad
\bar e_j
=
\Pr\left[\hat z_{i,j}\neq z_{i,j}^+\right],
\qquad
V_j
=
\mathbb E\left[\,|U_{i,j}|\;\middle|\;\hat z_{i,j}\neq z_{i,j}^+\,\right],
$$

其中 $\bar e_j$ 是位置 $j$ 在校准状态上的历史错误率，$V_j$ 是该位置一旦预测错误时对任务读出的平均伤害。这是恒等式而非近似；两个因子在位置间近乎不相关，说明它们度量不同的失效模式——$\bar e_j$ 刻画世界模型在该坐标上的可靠性，$V_j$ 刻画错误后果的任务严重性。把两者压缩成一个与状态无关的固定向量，无法区分“世界模型在这里很少出错”与“出错了也不伤”两类低效用位置；本实现保留两个因子，只对第一项做状态化，第二项 $V_j$ 仍是随模型锁定的全局常量。

绝对值聚合的理由不变：$U_{i,j}$ 的符号只表示单次替换让金标答案的 NLL 上升还是下降。带符号均值会让不同状态与问题中的正负效应相互抵消——完整分支已包含 OPQ 量化误差，某个真实码可能对特定问题冗余或偶然误导，替为众数后 NLL 反而下降。门控不利用这类偶然改善奖励丢码，而是把任意方向的任务条件分布变化都视为失真。

#### 状态相关错误率的校准

错误率状态化不引入任何学习组件：世界模型后验本身携带当前状态的可预测性。部署时以后验统计量（后验熵 $H_{t,j}$ 或众数置信 $p^{\max}_{t,j}=\max_c p_{t,j}(c)$）作为 carrier 分箱，箱内取校准状态上的经验错误率：

$$
e_{t,j}
=
\widehat{\Pr}\left[\hat z_j\neq z_j^+\;\middle|\;\mathrm{carrier}_{t,j}\in\mathcal B_k\right],
\qquad
U_{t,j}
=
e_{t,j}\,V_j.
$$

校准表按 0.25 bit 宽的熵箱构建，另存按众数置信分箱的备用表；空箱回退到全局 $\bar e_j$。carrier 的选择在留出状态上完成：比较两种 carrier 的 $U_{t,j}$ 对重建保真度随 keep 率变化的曲线，取对目标操作点拟合更优者；错误率本身在训练状态上估计、留出状态上验证。使用校准后的经验错误率而非原始 $1-p^{\max}_{t,j}$，是因为世界模型并非完美校准，历史统计修正了高置信区间的残余错误率。

在线编码与解码端只加载 $V_j$、校准表和 $\lambda$，不运行 reader，也不把当前查询作为输入；单条 query-independent memory state 仍可被后续不同查询复用，代价是 $V_j$ 与离线标注问题分布绑定。迁移到新任务域时，保持算法、码本和世界模型接口不变，只需在该域的读出问题上重新估计 $V_j$ 与校准表。

#### 率失真门控的推导

世界模型在状态 $t$、位置 $j$ 上的后验熵为

$$
H_{t,j}
=
-\sum_{c=0}^{63}p_{t,j}(c)\log_2p_{t,j}(c).
$$

令 $m_{t,j}=1$ 表示发送真实码，$m_{t,j}=0$ 表示由解码端使用世界模型众数填补。以状态化效用 $U_{t,j}=e_{t,j}V_j$ 作为省略的预期失真，以世界模型后验熵近似发送该位置的预期码长：

$$
\mathcal J_t(m_t)
=
\sum_{j=1}^{1024}
\left(
(1-m_{t,j})\,e_{t,j}V_j
+\lambda m_{t,j}H_{t,j}
\right).
$$

该目标对各位置可分。比较 $m_{t,j}=0$ 与 $m_{t,j}=1$ 的代价，可以直接得到

$$
\boxed{
m_{t,j}=1
\iff
e_{t,j}V_j\ge\lambda H_{t,j}
}.
$$

决策语义是“不发送的预期任务损失 vs 发送的存储成本”：左侧是当前状态该位置被预测错的概率乘以错误后的平均伤害，右侧是该位置预期码长乘以预算系数。超参数 $\lambda$ 是锁定 artifact 的一部分：增大它会提高码率惩罚并减少发送位置，减小它会偏向保存任务相关变化。消融变体使用无熵载体的规则 $e_{t,j}V_j\ge\lambda$，用于分离熵项的贡献。

门控使用熵而不是已经观察到的真实码长

$$
r_{t,j}=-\log_2p_{t,j}(z_{t,j}^+),
$$

是为了维持编码端与解码端的判定对称性。编码端知道 $z_{t,j}^+$，但解码端在收到该码之前并不知道它，故不能独立复现依赖 $r_{t,j}$ 的 mask；采用该规则必须额外发送 mask。相反，$H_{t,j}$ 与 $e_{t,j}$ 只依赖双方共享的世界模型后验和锁定校准表，双方可由相同的已重建历史、动作和锁定模型独立计算。协议只发布 $V_j$、校准表、$\lambda$ 和锁定模型，不传逐状态 mask；实际计费仍使用被发送真码的 $r_{t,j}$，而不是用熵替代码长。

轨迹初态没有可用的因果世界模型后验，按协议全部发送。从第二个可预测状态开始，编码端和解码端都以此前已经重建的历史计算后验和 mask；这也是闭环评测能够逐步复现门控决策的条件。

### 3.5 压缩感知检索头

检索头把压缩后的 memory payload 映射为检索地址。其训练原则是：同一观察的 full OPQ 重建与 utility-gated 重建允许在低效用细节上不同，但二者应指向相同的语义地址。门控改变的是被存储的内容精度，不应任意改变该条记忆在检索空间中的身份。

#### 双视图训练数据

对同一个冻结 OPQ 码 $z^+$，训练 cache 同时构造完整视图与部署视图：

$$
x_f=D_{\rm OPQ}(z^+),
$$

$$
x_m=D_{\rm OPQ}(m\odot z^++(1-m)\odot\hat z_{\rm WM}).
$$

$x_f$ 是所有位置均使用真实码的稳定参考；$x_m$ 严格使用锁定世界模型后验、$|U|$ 和 $\lambda=0.0010$ 生成，是在线部署时真正进入检索头的输入。cache 含 4,379 个状态，其中 3,539 个用于训练、840 个用于验证；3,612 个状态具有世界模型后验，767 个初态按协议全发。对具有后验的状态，平均保留比例为 0.85544，full 与 gated 之间实际改变的码位置比例为 0.095657。

每一对视图共享同一个冻结 Qwen3-VL 融合观察 teacher embedding $t_i\in\mathbb R^{4096}$。teacher 读取完整观察并只用于离线监督，不进入在线存储或检索路径。

#### 检索头结构

`MaskedAttentionRetrievalHead` 的输入为 $32\times512$ 的重建 Q-Former 状态。先对每个槽独立做 LayerNorm，再由一个共享的标量 attention scorer 计算槽权重：

$$
a_{i,s}
=
\frac{
\exp(w^{\mathsf T}\operatorname{LN}(x_{i,s}))
}{
\sum_{r=1}^{32}\exp(w^{\mathsf T}\operatorname{LN}(x_{i,r}))
},
$$

$$
v_i=\sum_{s=1}^{32}a_{i,s}\operatorname{LN}(x_{i,s}).
$$

池化向量经过两层 MLP 投影到 teacher 空间并执行 L2 normalization：

```text
32 × 512 重建状态
  → slot LayerNorm
  → learned scalar attention pooling
  → 512 维 pooled state
  → Linear(512,1024) + GELU + Linear(1024,4096)
  → L2 normalization
```

检索头共有 4,725,248 个参数，输出为

$$
u_i=h(x_i)
=
\frac{W_2\operatorname{GELU}(W_1v_i+b_1)+b_2}
{\left\|W_2\operatorname{GELU}(W_1v_i+b_1)+b_2\right\|_2}.
$$

#### 检索监督与一致性约束

在大小为 $B$ 的 batch 内，检索 head 输出与归一化 teacher embedding 的相似度矩阵为

$$
S_{ab}=\frac{h(x_a)^{\mathsf T}t_b}{\tau},
\qquad \tau=0.05.
$$

同一行的 head 输出与 teacher embedding 构成正样本，其余 batch 行为负样本。symmetric InfoNCE 同时训练“memory 找 teacher”和“teacher 找 memory”两个方向：

$$
\mathcal L_{\rm NCE}^{\rm sym}
=
\frac12
\left(
\operatorname{CE}(S,I_B)
+\operatorname{CE}(S^{\mathsf T},I_B)
\right).
$$

单视图检索目标再加入逐样本 teacher cosine 约束：

$$
\mathcal L_{\rm ret}(h(x),t)
=\mathcal L_{\rm NCE}^{\rm sym}(h(x),t)
+\frac1B\sum_{i=1}^{B}
\left(1-\cos(h(x_i),t_i)\right).
$$

full/gated 双视图目标为

$$
\mathcal L_{\rm head}
=\tfrac12\mathcal L_{\rm ret}(h(x_f),t)
+\tfrac12\mathcal L_{\rm ret}(h(x_m),t)
+\frac{\alpha}{B}\sum_{i=1}^{B}
\left(
1-\cos\left(h(x_{m,i}),\operatorname{sg}(h(x_{f,i}))\right)
\right),
$$

其中 $\alpha=0.1$。最后一项把 gated key 拉向同一观察的 full key，`sg` 使 full key 在该项中充当单向参考；full 分支仍通过自己的 $\mathcal L_{\rm ret}$ 正常更新。

不能只训练 full/gated 一致性项，因为常量 embedding 也能使两条分支完全相同，形成表示塌缩。当前目标让两个视图分别对齐同一个冻结 teacher 并承担 batch 内排序任务，再以较小权重约束二者的一致性。因此它同时保留语义可辨识性和对 utility masking 的不变性。

独立 head 从随机初始化开始训练，使用 AdamW、batch size 64、学习率 $3\times10^{-4}$、weight decay $10^{-4}$、梯度范数上限 5.0 和 seed 35。每 250 step 在 validation split 上计算完整双视图目标，按最低 validation loss 保存 checkpoint；当前选定模型位于 step 1750。验证阶段 batch size 为 128。

该阶段只更新检索头，不重训 Q-Former。原因是目标是校准“冻结 memory 表示到检索空间”的接口：Q-Former、OPQ、世界模型和 utility artifact 已共同定义了离散坐标、预测后验与门控决策。若同步更新 Q-Former，现有 OPQ 码本、世界模型和 utility 标签都将对应旧坐标系，需要整条链重新拟合。冻结上游也使检索结果的变化可以归因于检索头的双视图适配，而不是 memory representation 本身发生漂移。

---

## 4. 码率与质量指标

### 4.1 世界模型全发码率

若发送全部 1,024 个真实码，理想条件码率为

$$
R_t^{\rm full}
=\sum_{j=1}^{1024}-\log_2p_{t,j}(z_{t,j}^+).
$$

该值衡量世界模型作为熵模型时，对完整离散状态的无损条件编码成本。当前报告使用理想码长，不包含具体 rANS 文件格式的常数开销。

### 4.2 门控码率

给定发送向量 $m_t$，状态内容码率为

$$
R_t^{\rm gate}
=\sum_{j=1}^{1024}m_{t,j}
\left[-\log_2p_{t,j}(z_{t,j}^+)\right].
$$

开环码率使用真实历史产生后验；闭环码率使用此前门控重建的 $\tilde z_{<t}$ 产生后验。闭环数值是实际因果解码协议对应的主指标。

相对定宽表示的压缩比定义为

$$
\operatorname{CR}=\frac{6144}{\mathbb E_t[R_t]}.
$$

### 4.3 质量指标

报告使用三类质量指标：

- **重建 $R^2$**：连续 Q-Former 状态与 OPQ/门控重建之间的解释方差；
- **答案 $|\Delta\mathrm{NLL}|$**：原始条件与反事实条件之间答案 NLL 的绝对变化，单位为 bit；
- **检索 R@K / MRR**：检索 head 在同 session 候选中的排序能力。

答案指标使用绝对值，因为任务相关性关心记忆扰动的影响幅度，而不是一次扰动偶然使 teacher-forced NLL 上升或下降的方向。

### 4.4 计费范围

第 6.1 节主码率表使用 semantic-only 配置，统计观察离散码，不包含检索 key、索引结构和共享模型参数。WMA 外部评测集上的动作通道独立计费为 40.175 bit/转移；该值在计算完整 episodic rate 时与观察码率相加。共享 Q-Former、OPQ、世界模型、utility vector、Reader Bridge 和检索头属于系统级 artifact，其具体版本由 manifest 管理。

若启用精确符号旁路，anchor 作为与 observation 绑定的状态 payload 按实际符号码长加入 episodic rate；Reader 运行时通过共享嵌入表产生的 token embedding 不重复计费。

---

## 5. 数据与评测协议

### 5.1 训练数据

| 阶段 | 数据量 | 用途 |
|---|---:|---|
| Q-Former / reader | WMA fit-corpus 观察与 QA | 状态压缩、答案蒸馏、观察蒸馏 |
| OPQ 拟合 | 520,110 states | 冻结离散坐标系 |
| 世界模型 | 520,220 states / 495,527 transitions | 离散下一状态建模 |
| utility labels | 86,400 位置级反事实标签 | 估计全局 $|U_j|$ |
| retrieval-head cache | 4,379 states | full/gated 双视图检索训练 |
| AMA Q-Former | 6,645 observations / 11,944 SyQA pairs | 视觉与精确文本状态读出 |
| AMA Qwen3-32B Bridge | 冻结 AMA Q-Former cache | $512\rightarrow5120$ soft-token 接口训练 |

世界模型训练文件是 train-only；训练时从 episode 维度留出 5% 用于过程选点。当前选定模型在 60,000 步达到 best=last。

### 5.2 外部评测

主码率评测来自 WorldMemArena `agent/gui/web` 的 27 个样本，经轨迹转换后包含 956 个状态；最终 evaluation split 含 817 条转移、70 个评测 episode。该集合不进入世界模型的梯度训练，也不参与 utility vector 拟合；世界模型训练期间每 2,500 步在该 evaluation cache 上测量码率并选择 checkpoint，当前锁定运行的 best 与 60,000 步 last 为同一组权重。WMA 默认配置不启用精确值 anchor 分路。

Q-Former 下游表使用 27/27 个样本、1,459 道 QA。观察保真 gap 使用 48 个观察和不参与梯度训练的 P4 问法，其中前 24 个观察同时用于 gap-best checkpoint 选择。utility 质量评测在 Web 中有 QA 的 527 个状态、1,140 道问题上进行；码率仍对全部 817 条状态转移取平均。

检索头验证集包含 840 行、31 个 session。R@K 先在每个 session 内计算，再按行数加权汇总。

AMA-WEB 外部读取评测来自 `AMA-Bench/dataset/test/open_end_qa_set.jsonl` 的 WEB 开放式 QA 子集。评测排除 episode 184 后包含 30 条 trajectory、360 道 QA，每条 trajectory 含 12 道题。AMA-Bench test 不进入 Q-Former、检索头或 Bridge 的训练和 checkpoint 选择；生成阶段冻结 Qwen3.5-9B trunk、SyQA Q-Former、检索头、Qwen3-32B Bridge 与 Qwen3-32B Reader，答案由同一 Qwen3-32B Judge 按二值正确性评分。

### 5.3 对照方法

码率实验使用以下对照：

- **fixed width**：每个位置固定 6 bit，总计 6,144 bit；
- **marginal**：每个位置独立的任务边缘类别分布；
- **copy-aware**：先预测是否复制上一状态对应码，再预测变化后的目标码；
- **source-conditioned Markov**：按上一状态码条件化的一阶转移表；
- **random / displacement / rate masks**：与 utility gate 匹配预算的发送位置对照。

---

## 6. 实验结果

### 6.1 完整压缩链主结果

外部 WMA Web 评测集，817 条转移：

| 表示 / 编码方式 | bit/转移 | 压缩比 | 相对定宽节省 | 相对 WM 全发节省 |
|---|---:|---:|---:|---:|
| 定宽全发 | 6,144.00 | 1.000× | 0.00% | — |
| 世界模型条件编码，全发 | 5,182.83 | 1.185× | 15.64% | 0.00% |
| + 第一阶段 utility gate，开环 | 4,311.96 | 1.425× | 29.82% | 16.80% |
| + 第一阶段 utility gate，闭环 | 4,323.84 | 1.421× | 29.63% | 16.57% |
| + 两阶段 utility gate，开环 | 4,497.87 | 1.366× | 26.79% | 13.22% |
| **+ 两阶段 utility gate，闭环** | **4,509.77** | **1.362×** | **26.60%** | **12.99%** |

当前结果采用两阶段门控。其闭环表示相对定宽减少 1,634.23 bit/转移，即 26.60%；开环与闭环相差 11.90 bit。第二阶段只会恢复初始被丢弃、且在联合丢码背景下仍有任务影响的位置，因此以一部分码率换取对多位置交互的显式校准。

对应的任务质量测量为：

| 变换 | 答案 $|\Delta\mathrm{NLL}|$ |
|---|---:|
| 原始 Q-Former 状态 → full OPQ 重建 | 0.148 bit |
| full OPQ 重建 → 第一阶段 utility-gated 重建 | 0.103 bit |

两项分别测量量化和门控的增量影响，不将其直接相加作为端到端答案损失。

### 6.2 世界模型与统计基线

相同 817 条外部转移：

| 模型 | bit/转移 | 压缩比 |
|---|---:|---:|
| marginal | 6,381.99 | 0.963× |
| copy-aware | 6,051.42 | 1.015× |
| source-conditioned Markov | 5,420.84 | 1.133× |
| **12 层世界模型** | **5,182.83** | **1.185×** |

世界模型比最强统计基线减少 238.01 bit/转移，说明 16 步状态历史与动作条件提供了超过一阶码转移表的信息。

### 6.3 第一阶段 Utility gate 的率—质量曲线

WMA Web 外部评测集：

| $\lambda$ | 平均保留 | 码率 | 相对全发节省 | 答案 $|\Delta\mathrm{NLL}|$ | 压缩比 |
|---:|---:|---:|---:|---:|---:|
| 仅 OPQ，全发 | 100.0% | 5,182.8 | 0.0% | — | 1.19× |
| 0.0009 | 85.5% | 4,413.8 | 14.8% | 0.090 | 1.39× |
| **0.0010** | **83.6%** | **4,312.0** | **16.8%** | **0.103** | **1.42×** |
| 0.0011 | 81.3% | 4,188.6 | 19.2% | 0.118 | 1.47× |
| 0.0012 | 79.3% | 4,084.6 | 21.2% | 0.137 | 1.50× |

在 fit-corpus 上，$\lambda=0.0010$ 将码率从 4,631.3 降到 3,931.4 bit，节省 15.1%；重建 $R^2$ 从 0.9793 降到 0.9671，答案 $|\Delta\mathrm{NLL}|$ 为 0.078 bit。

等预算质量对照显示 utility 位置选择优于仅使用几何位移、码长或随机选择：

| 语料 | utility vs random | utility vs displacement | utility vs rate |
|---|---:|---:|---:|
| fit-corpus，paired $t$ | −18.6 | −15.6 | −16.7 |
| WMA Web，paired $t$ | **−25.5** | −21.4 | −19.6 |

负值表示 utility gate 的答案绝对 NLL 变化更小。逐位置码率与带符号效用的相关系数接近 0，而码率与 $|U|$ 的 Spearman 相关为 0.46–0.51；因此码率能描述影响规模，但不能替代任务效用决定发送位置。

### 6.4 闭环稳定性

| 评测臂 | 开环码率 | 闭环码率 | 漂移 |
|---|---:|---:|---:|
| 第一阶段 utility gate | 4,311.96 | 4,323.84 | +11.88 |
| 两阶段 utility gate | 4,497.87 | 4,509.77 | +11.90 |
| random，seed 35 | 4,325.89 | 4,337.09 | +11.20 |
| anti-utility | 4,347.62 | 4,381.70 | +34.08 |

四个等预算随机种子的漂移范围为 11.04–12.48 bit，第一阶段 utility gate 位于该范围内；anti-utility 的漂移为 34.08 bit。两阶段校准后的开环—闭环差为 11.90 bit。全发空对照的开环与闭环差为 0.00 bit，并实际回写了 41,472 个历史位置。

将全部历史替换为世界模型众数时，全发码率从 5,182.83 增至 6,509.29 bit；只污染门控丢弃的 16.4% 位置时，同口径全发码率增量为 56.64 bit，低于线性外推的 217.5 bit。当前系统在稀疏历史替换下呈次线性漂移。

### 6.5 Q-Former 多粒度状态读出

27/27 个样本、1,459 道 QA。QA-C/H/O 衡量重点细节读出，保真 gap 衡量整体观察读出对当前屏幕的辨识度；gap 越高表示 matched latent 相对 mismatched latent 保留了更多当前观察信息：

| 方法 | 槽数 | QA-C | QA-H | QA-O | 保真 gap |
|---|---:|---:|---:|---:|---:|
| Raw-Fused | — | 0.5415 | 0.2132 | 0.2454 | — |
| Q-Former K=16 | 16 | 0.5949 | 0.1857 | 0.2193 | +0.0976 |
| 固定池化 | 64 | **0.5984** | 0.1864 | **0.2152** | +0.1610 |
| Q-Former K=32，$w_{obs}=0.5$，原检索头 | 32 | 0.5953 | 0.1886 | 0.2160 | **+0.1744** |
| Q-Former K=32，$w_{obs}=1.0$ | 32 | 0.5977 | **0.1844** | 0.2180 | +0.1408 |
| **full-xbar Reader 参照：K=32，$w_{obs}=0.5$，双视图检索头，无 anchor** | **32** | **0.6004** | 0.1851 | **0.2145** | **+0.1744** |

选定的 K=32 表示使用固定池化一半的槽数，QA-C 与 64 槽固定池化接近，同时获得更高的观察保真 gap。这表明同一份 query-independent latent 能够同时支持整体状态读出和重点细节读出。该参照保持 Q-Former 不变，只将检索头替换为第 6.8 节的 full/gated 双视图训练版本；在 27/27 个样本的 1,459 道题上答对 876 题，QA-C 为 0.6004。它读取全精度 `xbar`；完整 codec 结果见第 6.9 节。

### 6.6 OPQ 率—失真结果

当前全量系统固定使用 $C=64$。以下为同一早期语料协议上的类别数消融，用于展示码本容量对重建和条件码率的影响：

| $C$ | 定宽 bit | 世界模型 bit | 压缩比 | validation $R^2$ | WMA $R^2$ | 码持久率 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 4,096 | 3,039.2 | **1.348×** | 0.9767 | 0.9713 | 25.1% |
| **64** | **6,144** | **5,027.1** | 1.222× | 0.9881 | 0.9855 | 13.7% |
| 256 | 8,192 | 7,052.7 | 1.162× | **0.9909** | **0.9891** | 10.1% |

$C$ 增大时重建质量提高，但定宽成本增长，且码持久率降低。$C=64$ 是当前系统在重建精度、训练规模和存储率之间采用的操作点。全量重拟合后，当前 $C=64$ validation $R^2$ 为 0.98767。

### 6.7 世界模型规模与结构分析

在 $C=256$ 的嵌套训练子集上：

| 拟合转移数 | bit/转移 |
|---:|---:|
| 3,656 | 7,416.6 |
| 7,598 | 7,368.6 |
| 15,162 | 7,198.4 |
| 31,508 | 7,049.8 |

四点对数线性拟合为每翻倍减少 123.0 bit，$R^2=0.955$。该结果说明在对应数据范围内，离散世界模型的条件码率仍随数据量改善。

历史建模消融为：

| 模型 | dev bit/转移 | 上下文 | 每步耗时 |
|---|---:|---|---:|
| **block-causal Transformer** | **7,013.6** | 16 步 | 172 ms |
| block-causal Transformer | 7,080.5 | 7 步 | 63 ms |
| copy baseline | 7,605.5 | 1 步复制统计 | — |
| recurrent GRU | 7,746.5–7,752.1 | 递归压缩历史 | 454 ms |

连续状态预测分析得到 persist $R^2=0.1437$、逐槽 ridge $R^2=0.4622$、12 层 Transformer $R^2=0.4588$、MLP $R^2=0.2811$。这表明当前状态空间的大部分可预测结构接近局部线性，世界模型的主要作用是把该结构转换为离散条件概率和可计费码长。

### 6.8 压缩感知检索头

在同一个 utility-gated cache 的 840 行 validation 上：

| 检索头 / 输入视图 | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| 原检索头 / full | 0.2226 | 0.5119 | 0.6488 | 0.3623 |
| 原检索头 / gated | 0.2226 | 0.5131 | 0.6488 | 0.3622 |
| **双视图检索头 / full** | **0.2286** | **0.5262** | 0.6548 | **0.3706** |
| **双视图检索头 / gated** | 0.2274 | **0.5262** | **0.6560** | 0.3697 |

按新目标在同一 validation cache 上重算，总损失从 3.31884 降到 3.30336。选定 checkpoint 为 step 1750，$\alpha=0.1$。full 与 gated 两个视图的 R@K 接近，且 gated 视图相对原检索头有小幅提升，因此该 head 被固定为当前检索接口。

### 6.9 WMA 两阶段闭环 codec 端到端 Reader 结果

最终系统从官方 WMA Web JSON 重新生成 Q-Former 状态和 OPQ 码，使用已经重建的历史递归计算 WM posterior，经 utility gate 填补后再进入双视图检索头与 soft-token Reader。WMA 默认不启用 anchor 分路。

| 指标 | 闭环 WM + utility |
|---|---:|
| 样本 / 有效 QA | 27 / 1,459 |
| QA-C | **0.6004（876/1459）** |
| QA-H | 0.1864（272/1459） |
| QA-O | 0.2132（311/1459） |
| retrieval coverage | 0.6881（3556/4914） |
| Recall@1 / @5 / @10 | 0.2242 / 0.5328 / 0.6988 |
| 闭环码率 / keep fraction | 4,509.77 bit / 0.8719 |

完整 codec 与第 6.5 节 full-xbar Reader 参照均答对 876 题，QA-C 同为 0.6004。最终结果位于 `gate/two-stage-v1/wma-qa/final/aggregate_metrics.json`。

### 6.10 AMA-WEB AXTree Reader 结果

AMA-WEB 评测使用 query-independent AXTree observation memory、冻结检索头和 Qwen3-32B Reader。该 Reader 协议直接读取连续 $x_t$，用于评估检索和生成接口，不与第 6.1 节的 semantic codec 码率合并。下表保持 30 条 trajectory、360 道 QA、matched memory、`top_k=1`、生成参数和 Judge 不变，只替换 $512\rightarrow5120$ Reader Bridge：

| Bridge | 训练目标 | 正确数 | 准确率 |
|---|---|---:|---:|
| RMS Bridge | $\mathcal L_{\rm CE}+0.30\mathcal L_{\rm KL}$ | 87/360 | 24.17% |
| **Delta-margin Bridge** | 基础目标 + hidden distillation + matched/shuffled CE margin | **99/360** | **27.50%** |

Delta-margin Bridge 的新增目标为

$$
\mathcal L_{\rm bridge}
=
\mathcal L_{\rm CE}
+0.30\mathcal L_{\rm KL}
+0.10\mathcal L_{\rm hidden}
+0.20\max\left(0,0.10+\mathcal L_{\rm CE}^{\rm matched}-\mathcal L_{\rm CE}^{\rm shuffled}\right).
$$

$\mathcal L_{\rm hidden}$ 在 Qwen3-32B 第 16、32、48 层对齐完整观察 teacher 和 latent student 的 question-boundary hidden state；margin 项要求 matched memory 的答案 CE 至少比 shuffled memory 低 0.10。两版 Bridge 的逐题配对变化为 31 题由错变对、19 题由对变错、68 题均正确、242 题均错误，净增 12 题，即绝对提升 3.33 个百分点。

---

## 7. 产物与代码接口

### 7.1 冻结产物

WMA codec 使用 `configs/system.lock.yaml` 作为机器可读产物清单。它锁定以下六个角色并在加载时校验 SHA-256：

| 角色 | 选定产物 |
|---|---|
| Q-Former | `outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/qformer-K32e-obs0.5.gapbest.pt` |
| retrieval head | `outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/head-K32e-obs0.5-gapbest-utility-gated.pt` |
| retrieval cache | `outputs/instruct_bridge/v9-instruct-pca20k-balanced-xbar/cache-K32e-obs0.5-gapbest-utility-gated.npz` |
| OPQ codebook | `${DATA}/pq-full/opq-shared-mix10-M32-C64.npz` |
| world model | `run/best.pkl` |
| utility artifact | `gate/mask-lambda0.0010.npz` |

默认部署只校验五个运行时角色；`retrieval cache` 是训练追溯产物，不参与推理：

```bash
cd /mnt/data/users/luzheng/workspace/iclr/czs/residual-mem
./.venv-jax/bin/python -m residualmem.manifest
./.venv-jax/bin/python -m residualmem.manifest --include-training
```

第一条命令用于发布产物部署，五个运行时角色均须返回 `ok`；第二条用于从头训练后的
完整审计，并额外要求 retrieval cache 返回 `ok`。

AMA-WEB Reader 链使用独立、相互绑定的冻结产物：

| 角色 | 选定产物 |
|---|---|
| Q-Former | `outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/frozen.pt` |
| retrieval head | `outputs/ama_latent_memory/syqa/post-qformer/candidates/obs1.0-ce/head.pt` |
| Reader Bridge | `outputs/ama_latent_memory/syqa/post-qformer/bridge/qwen32-input-k32-rms-delta-margin-total1000.pt` |
| Reader / Judge | `Qwen3-32B` |

AMA cache metadata 记录 Q-Former 和 retrieval-head SHA-256；Bridge metadata 继续绑定这两个 hash、Reader 模型 hash、prompt hash、槽数和最大 memory rank。加载阶段逐项检查，从而防止不同 latent 坐标系、检索头或 Reader prompt 被组合到同一次评测。

### 7.2 关键源码

| 模块 | 代码位置 |
|---|---|
| Q-Former | `residualmem/latent/qformer.py` |
| reader connector / retrieval head | `residualmem/latent/instruct_bridge.py` |
| Q-Former 训练 | `experiments/state_tokenizer/train_qformer_joint.py` |
| OPQ 编解码 | `experiments/state_tokenizer/qformer_pq.py` |
| 世界模型 | `experiments/world_model/model.py` |
| utility 定义与 mask 导出 | `experiments/utility_gate/` |
| 闭环评测 | `experiments/utility_gate/closed_loop_rate.py` |
| full/gated retrieval cache | `experiments/state_tokenizer/build_utility_retrieval_cache.py` |
| 双视图检索训练 | `experiments/state_tokenizer/train_retrieval_bridge.py` |
| WMA 可选 OCR anchor | `residualmem/benchmarks/wma_exact_value_anchor.py`、`residualmem/benchmarks/rapidocr_backend.py` |
| AMA trajectory / AXTree 适配 | `xt_ama_adapter/xt_ama_adapter/adapter.py` |
| AXTree 清洗与精确 anchor | `xt_ama_adapter/xt_ama_adapter/axtree_clean.py` |
| Qwen3-32B Bridge / 混合 Reader | `xt_ama_adapter/xt_ama_adapter/qwen32_bridge.py` |
| AMA cache、检索与生成 | `experiments/state_tokenizer/run_ama_web_latent.py` |
| 系统结果汇总 | `scripts_run_system.py` |

### 7.3 组件依赖

组件之间存在以下固定依赖：

$$
\text{Q-Former}
\rightarrow\text{OPQ}
\rightarrow\text{world model}
\rightarrow\text{utility artifact}
\rightarrow\text{retrieval cache/head}.
$$

更换 Q-Former 会改变连续状态坐标；更换 OPQ 会改变全部离散码坐标；这两类变化都要求重新生成其后的世界模型 posterior、utility 和检索双视图。当前 utility-gated 检索训练只修改最后的 retrieval head，不修改任何上游组件。

可选 anchor 与语义链仅通过 observation identity 和 retrieval rank 对齐，不改变 OPQ、世界模型及其下游 artifact。

### 7.4 结果复现入口

从冻结产物重算系统主表：

```bash
./.venv-jax/bin/python scripts_run_system.py --gpu 1
```

预期核心数值为：

```text
full WM rate       5182.831 bit/transition
utility open-loop  4311.963 bit/transition
utility closed     4323.840 bit/transition
closed compression 1.42096×
```

重训压缩感知检索头的入口为：

```bash
bash scripts_utility_retrieval_head.sh 0
```

该入口从现有 Q-Former cache 离线构造 full/gated OPQ 重建，并只训练 `MaskedAttentionRetrievalHead`。

AMA-WEB 的冻结 Q-Former/Bridge 评测入口为：

```bash
bash xt_ama_adapter/scripts/run_ama_web_delta_margin_total1000.sh
```

---

## 8. 实验口径

技术结论使用以下实验口径：

- 状态表示是 Qwen3.5-9B 第 16 层经 K=32 Q-Former 得到的 $32\times512$ latent；
- 离散表示固定为 32 槽、每槽 32 个子空间、$C=64$；
- 世界模型使用 16 步 Web 轨迹上下文和当前动作协议；
- utility vector 针对当前问题分布估计，迁移到新的任务问题分布时需要重新估计；
- 码率是 latent 的理想条件码长，不代表原始截图的可逆像素压缩率；
- WMA Reader 表、semantic codec 表和 full/gated 检索表分别使用各自冻结协议，不跨表组合数值；
- AMA-WEB 使用 AXTree observation、Qwen3-32B Reader 和二值 LLM-as-Judge；

在上述口径内，实验结果表明：世界模型条件概率能够压缩完整离散状态；任务效用门控能够在较小答案扰动下进一步减少发送码；full/gated 双视图训练能够保持并小幅改善门控重建后的检索指标；delta-margin Bridge 能够改善 Qwen3-32B 对 latent memory 的读取。
