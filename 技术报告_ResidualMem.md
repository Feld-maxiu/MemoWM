# ResidualMem：基于世界模型互补残差编码的长期智能体记忆


> 技术报告草案  
> 版本：v0.1  
> 日期：2026-06-10

## 摘要

长时程智能体会持续接收文本、图像、GUI、工具反馈和环境状态等多模态观察。现有长期记忆方法通常保存完整观察、历史摘要、检索片段或由重要性评分筛选出的事件。这类方法主要从观察本身判断“什么值得记住”，却很少考虑智能体已有世界模型已经能够恢复多少信息。结果是，外部记忆不断重复保存世界模型已经掌握的环境规律和常规状态转移，产生大量冗余。

本报告提出 **ResidualMem**：一种与世界模型互补的长期记忆框架。其核心观点是，世界模型负责生成在当前状态和动作条件下可预测的世界，外部记忆只保存使该预测恢复为真实观察所必需、且对未来任务有用的修正信息。形式上，概率世界模型在观察到来前给出潜在状态先验，观察编码器在看到真实观察后形成后验；ResidualMem不直接保存完整观察或完整后验，而是学习一个条件残差编码，将后验相对先验的必要创新量编码为可变长度离散码。若世界模型能够可靠恢复观察，则无需写入；若观察包含外生事件、具体随机结果或模型例外，则系统仅保存相应修正。

ResidualMem采用“冻结的Qwen3.5多模态观察编码器、统一概率latent world model、条件残差codec、任务效用门控、外部残差存储和状态修正解码器”的总体架构。训练目标采用任务相关的率失真准则，在状态重建、未来任务表现与实际记忆码长之间进行优化。该项目拟验证三个核心命题：第一，相同任务性能下，条件残差记忆比全量存储、摘要和surprise-gated memory使用更少存储；第二，随着世界模型预测能力增强，达到固定任务性能所需的外部记忆容量下降；第三，仅依赖预测误差会保存随机噪声，而结合未来任务效用能够得到更有效的记忆。

> **实现状态（2026-08-12）**：v8 离散 World Model 证据门的 M0/M1 已完成。
> full seed 0 在 validation 上达到 9,058.17 bits/transition，优于 copy-aware
> 基线 9,564.59，但未优于更严格的 source-conditioned Markov 8,990.72。
> 因而实验按预注册协议停止，未进入消融、闭环、test 或条件残差 codec；本报告
> 以下系统设计仍是待验证方案，不能解读为端到端 ResidualMem 已成立。

---

## 1. 研究动机

### 1.1 现有长期记忆的冗余问题

设智能体已经知道如下环境规律：

- 点击有效的“提交”按钮通常会使任务进入已提交状态；
- 文件移动成功后，其位置通常变为目标目录；
- 未受到干预的对象属性通常保持不变；
- 用户的稳定偏好通常会在相邻会话中持续；
- 机器人执行抓取动作后，物体通常随夹爪移动。

若每次发生这些常规事件时，系统仍然保存完整截图、完整对话或完整状态，就会在外部记忆中重复编码世界模型已经掌握的规律。随着交互长度增长，这种冗余会导致：

1. 存储量近似随轨迹长度线性增长；
2. 检索候选持续增加，相关记忆被大量常规事件稀释；
3. 相似观察被重复保存；
4. 长期上下文和推理成本持续增加；
5. 记忆系统与世界模型各自保存同一部分信息，职责重叠。

ResidualMem希望改变记忆的参照系：一条观察是否值得保存，不仅取决于它自身包含什么，还取决于**在给定当前世界模型后，还剩下多少不可恢复的信息**。

### 1.2 核心直觉

ResidualMem采用以下分工：

> 世界模型保存一般规律，外部记忆保存具体实现、外生信息和模型例外。

世界模型负责：

- 常见状态转移；
- 状态持久性；
- 动作的通常后果；
- 已经内化的跨任务环境规律；
- 对未来状态的概率分布。

外部记忆负责：

- 当前世界的初始条件；
- 无法从历史推导出的外生事件；
- 随机过程已经实现的具体结果；
- 世界演化偏离模型预测的例外；
- 对未来任务有用但尚未被模型参数内化的信息。

因此，完整世界状态由两部分共同恢复：

$$
\text{Reconstructed World}
=
\text{World-Model Prior}
+
\text{Residual Memory}.
\tag{1}
$$

这里的“加法”不是简单向量相加，而是条件解码：世界模型提供默认分布，残差码将该分布修正为与实际观察一致的后验状态。

### 1.3 比“预测错误就保存”更严格的定义

ResidualMem不等同于：

$$
\text{Prediction Error}>\tau
\quad\Longrightarrow\quad
\text{Store the Full Observation}.
\tag{2}
$$

式（2）存在三个问题。

第一，预测错误不代表信息有用。视觉噪声、无关背景变化和随机措辞都可能产生较大误差，但未必影响未来任务。

第二，世界模型可能正确预测了一个高熵分布，却不知道这次实际实现了哪个结果。例如模型知道掷硬币正反各半，但若未来需要回答本次结果，仍然必须保存实际结果。

第三，保存完整观察没有利用模型已经预测正确的部分。即使一张GUI截图只有一个按钮状态异常，完整保存截图仍然包含大量冗余。

ResidualMem追求的是：

> 保存能够使世界模型恢复真实状态的最小、任务相关修正码，而不是保存所有高surprise观察。

---

## 2. 与已有工作的关系

预测误差、创新量和选择性记忆并非全新概念。与ResidualMem最接近的工作包括：

- **B'MOJO**：使用Innovation Selection，将难以由状态空间模型预测的token加入eidetic memory；
- **Nemori / What Deserves Memory**：将“可预测意味着冗余”用于LLM Agent的记忆蒸馏；
- **Worth Remembering**：在V-JEPA 2语义视觉空间中使用Bayesian surprise选择机器人情景记忆；
- **D-MEM**：同时使用surprise和长期效用决定是否触发长期记忆更新；
- **Curious Replay**：优先回放世界模型预测不好的经验；
- **预测编码与生成式记忆研究**：使用生成模型恢复可预测内容，以有限情景记忆保存独特信息。

因此，ResidualMem不能仅以“高预测误差才写入”为贡献。它需要与上述工作形成以下区别：

| 方法类别 | 主要操作 | 保存对象 | 是否优化实际码率 | 是否由WM参与解码 |
|---|---|---|---|---|
| 全量记忆 | 全部写入 | 完整观察/轨迹 | 否 | 否 |
| 重要性或效用门控 | 二元选择 | 完整事件 | 通常否 | 通常否 |
| Surprise gate | 预测错误则写入 | 完整观察/事件 | 通常否 | 部分 |
| B'MOJO类创新选择 | 选择难预测token | 原始token | 间接 | 是 |
| **ResidualMem** | 条件可变码率编码 | 最小任务相关修正码 | **是** | **是** |

ResidualMem拟强调的独立研究点是：

1. **记忆内容由世界模型条件化定义**，不是观察自身的独立摘要；
2. **保存粒度是可变码率残差**，不是“整条事件存或不存”的二元选择；
3. **读取时必须由WM先验与残差共同解码**，记忆本身不是完整事实副本；
4. **统一核算内容码、索引、时间、锚点和元数据的实际存储成本**；
5. **系统性验证WM能力与外部记忆容量之间的关系**。

---

## 3. 问题设定

### 3.1 长时程多模态交互

考虑一条交互轨迹：

$$
\tau=(o_0,u_0,o_1,u_1,\ldots,o_T),
\tag{3}
$$

其中：

- $o_t$：时刻$t$的多模态观察，可包含文本、图像、GUI、网页、文档、工具反馈或环境画面；
- $u_t$：在$o_{t+1}$到来之前已经已知的动作、干预或时间条件；
- $s_t$：环境真实状态，通常不可直接访问；
- $q$：后续记忆查询或任务目标；
- $y$：正确答案、目标状态或任务结果。

这里必须严格遵守因果时序。$u_t$只能包含产生$o_{t+1}$之前解码端已经知道的信息，例如Agent动作、已给定指令、时间间隔和领域标识：

$$
u_t=(a_t,\text{known instruction}_t,\Delta t,\text{domain}).
\tag{4}
$$

用户刚刚说出的内容、外部评审结果、环境随机事件等事后才知道的信息属于$o_{t+1}$，不能先抽取成“广义事件”再输入先验，否则会把待预测内容泄漏给世界模型。对于没有显式动作的跨会话数据，$u_t$只能使用时间间隔、会话边界和事前已知任务条件，世界模型主要预测状态持久性和变化分布；真实外生事件由观察后验与Residual Memory吸收。

此外，条件编码假设解码端拥有$u_t$。实验必须明确动作日志是免费side information还是记忆系统的一部分：若推理时不能从环境或Agent执行记录中重新获得动作序列，其编码成本也必须计入总记忆预算。

### 3.2 目标

给定一个冻结或版本固定的世界模型$W_\theta$、记忆策略$\pi_\psi$和总记忆预算$B$，学习由轨迹在线产生的外部记忆$M_\psi(\tau)$，使其在轨迹和未来任务分布上的期望损失最小：

$$
\psi^*
=
\arg\min_{\psi:\operatorname{Bits}(M_\psi(\tau))\leq B}
\mathbb E_{\tau,q}
\left[
\mathcal L_{\mathrm{task}}
\big(W_\theta,M_\psi(\tau),q\big)
\right].
\tag{5}
$$

等价地，可以采用拉格朗日形式：

$$
\min_\psi
\mathbb E_{\tau,q}
\left[
\mathcal L_{\mathrm{task}}
\big(W_\theta,M_\psi(\tau),q\big)
+
\beta\,\operatorname{Bits}\big(M_\psi(\tau)\big)
\right],
\tag{6}
$$

其中$\beta$控制任务质量与记忆成本之间的权衡。

### 3.3 核心研究假设

**H1：互补性假设。** 与保存完整观察相比，只保存世界模型无法恢复的修正信息，可以在相同任务性能下显著降低记忆容量。

**H2：能力—容量假设。** 在表示空间、任务和评测协议固定时，世界模型预测能力越强，达到固定任务性能所需的外部记忆容量越小：

$$
\operatorname{Quality}(W)\uparrow
\quad\Longrightarrow\quad
\operatorname{Bits}(M)\downarrow
\quad
\text{at fixed task performance}.
\tag{7}
$$

**H3：效用假设。** 预测误差本身不足以决定记忆价值；任务相关效用能够过滤随机噪声和无关意外。

**H4：可变码率假设。** 对部分可预测的观察，保存少量修正码优于保存完整观察或完全丢弃。

---

## 4. 总体架构

ResidualMem采用统一概率latent world model路线，由六个核心模块组成：

1. 冻结的多模态观察编码器；
2. 状态tokenizer；
3. 概率latent world model；
4. 条件残差codec；
5. 率失真发送mask、可选任务效用蒸馏与外部记忆；
6. 修正解码器与任务读取器。

系统内部必须区分两条状态路径：

- **编码端观察状态**：$z_t^+$，由真实观察后验得到，只用于决定需要发送什么残差；
- **解码端可重建状态**：$\tilde z_t,\tilde h_t$，只由WM先验、已计费残差和已知side information得到。

只有解码端状态可以推进下一时刻的先验。被门控丢弃的信息不得继续隐藏在RSSM确定性状态、KV cache或未计费工作记忆中。否则系统会出现“外部记忆没有保存，但内部hidden state仍然记住”的容量泄漏。

整体流程如下：

```text
上一重建状态 + 动作/事件
              |
              v
      Probabilistic Latent WM
              |
       状态先验 p_t
              |
      +-------+--------------------+
      |                            |
      |                      真实观察 o_t
      |                            |
      |                    Qwen3.5 Encoder
      |                            |
      |                       观察状态 x_t
      |                            |
      +--------> Posterior q_t <---+
                      |
              Conditional Codec
                      |
              候选残差码 c_t
                      |
                Utility Gate
                      |
            External Residual Memory
                      |
         WM Prior + Retrieved Residuals
                      |
              Corrected State
                      |
             QA / Agent / Prediction
```

---

## 5. 多模态观察编码

### 5.1 主编码器选择

主方案使用冻结的 **Qwen3.5-9B-Base** 作为统一多模态观察编码器。选择Base版本的原因是：

- 支持文本与图像输入；
- 适合研究性微调和表征提取；
- 可以读取内部hidden states，而非必须生成自然语言描述；
- 9B规模兼顾表征能力与批量预编码成本；
- 可使用2B、4B和9B版本进行编码器规模消融。

对每一步观察独立编码：

$$
H_t=E_{\mathrm{Qwen3.5}}(o_t).
\tag{8}
$$

关键限制是：

- 编码器不能接收完整历史；
- 每个时间步重置KV cache和循环状态；
- 跨session完全清空VLM上下文；
- 历史信息只能通过固定容量WM状态或Residual Memory传播。

否则，长上下文VLM本身会成为未计费的长期记忆，破坏实验有效性。

状态tokenizer只能在独立训练数据或训练划分上学习，并在训练WM和评测记忆前冻结。不得使用benchmark测试问题、测试证据标注或测试轨迹调节其表示。

### 5.2 状态tokenizer

Qwen3.5会输出数量可变的文本和视觉token。使用Perceiver Resampler或Query Transformer将其压缩为固定数量状态token：

$$
x_t=P_\rho(H_t)
\in\mathbb R^{K_x\times d_x}.
\tag{9}
$$

状态token应同时保留：

- 实体及属性；
- 对象和界面元素关系；
- 当前任务进度；
- 动作反馈；
- 重要视觉结构；
- 时间和来源信息。

仅用可学习Perceiver输出作为“真实状态”仍不够，因为它可能主动舍弃难预测但有用的信息。主方案应把$x_t$定义为一个训练后冻结的混合目标：

$$
x_t=
\left(
x_t^{\mathrm{sem}},
x_t^{\mathrm{ocr}},
x_t^{\mathrm{vis}},
x_t^{\mathrm{state}}
\right),
\tag{9a}
$$

其中：

- $x_t^{\mathrm{sem}}$：冻结Qwen3.5中间层的语义token；
- $x_t^{\mathrm{ocr}}$：页面、文档和GUI中的精确文本token；
- $x_t^{\mathrm{vis}}$：冻结视觉塔或DINO/V-JEPA特征；
- $x_t^{\mathrm{state}}$：由独立教师或环境信号得到的实体、属性、关系和动作结果。

ResidualMem可以对不同子空间采用不同失真权重，但不能只优化一个可自由漂移的learned latent。

对于最终固定长度PCA状态，符号严格区分为：$x_t$表示原始PCA状态，$\bar x_t$表示只用训练划分统计得到的固定group×channel标准化状态，$z_t$只表示后续WM的分组categorical随机latent。令$m_{t,s}$为slot有效性，则：

$$
\bar x_{t,s,j}
=
m_{t,s}
\frac{x_{t,s,j}-\mu_{g(s),j}}
{\max(\sigma_{g(s),j},\sigma_{\min})}.
\tag{9b}
$$

标准化后必须再次乘$m_{t,s}$，确保原始零padding不会因减均值变成非零向量。$\mu$与$\sigma$只能由训练划分的valid slots估计，训练、评测与部署使用同一冻结artifact；不得对单个状态独立LayerNorm并作为重建target。

为了避免最后一层hidden state过度面向语言生成，可以融合视觉塔输出与中间多模态层：

$$
x_t=P_\rho
\left(
H_t^{\mathrm{vision}},
H_t^{\mathrm{middle}}
\right).
\tag{10}
$$

### 5.3 可选视觉辅助编码器

统一方案以Qwen3.5为主。对于细粒度视觉动态，可增加辅助特征，但不改变统一状态空间：

- GUI、文档和静态图片：DINOv3特征；
- 视频、Crafter和具身任务：V-JEPA 2/2.1特征；
- 检索键：Qwen多模态embedding模型。

辅助特征先经adapter映射，再由同一状态tokenizer融合。它们是可选增强，不应成为主结论成立的必要条件。

### 5.4 防止表示坍塌

观察编码器原则上冻结。若状态tokenizer与WM共同训练，模型可能主动丢失难预测信息，使残差虚假变小。为避免这种退化，需要：

1. 冻结Qwen3.5主干；
2. 使用固定的语义、视觉和状态probe监督状态tokenizer；
3. 对关键实体、属性、关系和动作结果加入辅助恢复目标；
4. 不允许WM梯度更新目标编码器；
5. 在独立数据上验证状态token对下游任务的充分性。

---

## 6. 统一概率Latent World Model

### 6.1 结构选择

主世界模型采用 **Transformer-RSSM**，结合：

- Transformer对长程事件和多实体关系的建模能力；
- RSSM对确定性历史状态与随机潜变量的分离；
- 显式先验—后验结构；
- 对不确定未来的概率建模。

确定性动力学状态为：

$$
\tilde h_t
=
f_\theta
(\tilde h_{t-1},\tilde z_{t-1},u_{t-1},d),
\tag{11}
$$

其中$d$是领域token，例如Web、Mobile、Crafter、Text或Embodied。

在未观察$o_t$之前，世界模型给出先验：

$$
p_t
=
p_\theta(z_t\mid \tilde h_t).
\tag{12}
$$

看到真实的标准化观察状态$\bar x_t$后，后验通过mask-aware slot attention给出：

$$
q_t
=
q_\phi(z_t\mid \tilde h_t,\bar x_t,m_t).
\tag{13}
$$

invalid slots在attention logits中被屏蔽，且$\bar x_t$中的invalid位置保持精确零。状态解码器重建标准化状态：

$$
\hat{\bar x}_t=D_\omega(\tilde h_t,z_t).
\tag{14}
$$

重建损失只在valid slots上计算，并可先在image/detail/context/prompt各组内求masked mean，再按固定组权重聚合。需要回到原PCA空间时，使用冻结的$\mu,\sigma$对$\hat{\bar x}_t$做可逆反标准化并再次应用$m_t$。

### 6.2 离散概率latent

建议使用分组categorical latent：

$$
z_t=(z_t^1,\ldots,z_t^N),
\qquad
z_t^j\in\{1,\ldots,C\}.
\tag{15}
$$

优点包括：

- 便于形成离散记忆码；
- 可以输出每组状态的概率；
- 便于熵编码和实际bit统计；
- 可按组决定哪些修正值得保存；
- 可分析不同latent组承载的状态类型。

训练可使用straight-through estimator、Gumbel-Softmax或Dreamer式离散随机状态。

分组latent只有在信息相对解耦时才适合独立丢弃。需要加入total-correlation约束、group dropout、slot specialization或实体/区域辅助监督，避免同一事实分散在多个组中，使单组效用和码率失去意义。

### 6.3 世界模型预测对象

WM主要预测Qwen3.5定义的语义/视觉latent，而非完整像素。训练目标包括：

- 下一步状态token；
- 动作结果；
- 关键实体属性；
- GUI元素状态；
- 任务进度；
- 多步未来latent；
- 不确定性与概率校准。

对于Web、GUI和具身领域，可使用现有领域WM进行蒸馏或初始化：

- Web：WebWorld；
- Mobile GUI：gWorld；
- Embodied：V-JEPA 2-AC或JEPA-WM；
- Minecraft：MineWorld。

这些模型不作为统一核心，而是提供领域转移知识。最终所有领域均映射到同一个概率latent WM接口。这里的“统一”应理解为统一状态接口、先验—后验形式和训练目标，而不必强迫所有领域共享完全相同的动力学参数。更稳妥的实现是共享骨干加domain adapter或mixture-of-experts transition heads。

### 6.4 世界模型的能力边界

世界模型不应被要求预测无法由历史决定的外生事实。例如：

- 用户突然透露姓名；
- 用户搬到新城市；
- 论文收到不可提前确定的审稿结果；
- 随机环境事件的具体实现。

WM应输出合理分布或“不确定”状态。真实结果到来后，后验收缩到实际状态，所产生的信息需要由Residual Memory保存。

因此，“可预测”不是指模型必须猜中所有未来，而是指：

> 在给定已有状态和动作后，模型对观察分布提供了多少可复用的信息。

---

## 7. 条件残差编码

### 7.1 创新量

先验与后验的差异可用KL散度度量：

$$
I_t
=
D_{\mathrm{KL}}(q_t\|p_t).
\tag{16}
$$

$I_t$是观察相对世界模型带来的理论创新量，可用于：

- surprise分析；
- posterior collapse诊断；
- WM能力比较；
- 候选写入优先级；
- 估计信息更新强度。

但需要强调：

> KL不应直接被当作系统实际存储bit。

只有在特定bits-back编码条件下，KL才与净码率具有严格对应。论文应分别报告理论创新量和真实编码后的存储大小。

### 7.2 条件latent编码

观察后验产生编码端离散目标状态：

$$
z_t^+
=
\operatorname{Quantize}
\!\left[
q_\phi(z_t\mid\tilde h_t,x_t)
\right].
\tag{17}
$$

训练时可使用straight-through categorical sample，部署和实际编码时使用确定性量化。不能笼统地“量化一个分布$q_t$”，因为单个离散码无法重建完整后验分布；真正被编码的是本次观察对应的离散状态实现$z_t^+$。

世界模型根据解码端已经拥有的状态给出条件概率：

$$
p_\theta(z_t^+\mid\tilde h_t).
\tag{18}
$$

若完整编码$z_t^+$，其条件码长近似为：

$$
R_t^{\mathrm{full}}
=
\sum_j-\log_2
p_t^j(z_t^{+,j}).
\tag{19}
$$

这天然体现“WM越准，码长越短”：

- 模型高概率预测正确：只需极少bit；
- 模型预测部分正确：只编码偏离部分；
- 模型完全不知道：需要较长码；
- 模型预测高熵随机分布：仍需保存实际实现结果所需的信息。

### 7.3 可丢弃的分组残差

为了让记忆容量服务于任务，而非无损保存所有细节，为每个latent组学习发送掩码：

$$
m_t^j\in\{0,1\}.
\tag{20}
$$

- $m_t^j=1$：保存该组实际码；
- $m_t^j=0$：不保存，读取时由WM先验估计。

残差码为：

$$
c_t=\{(j,z_t^{+,j}):m_t^j=1\}.
\tag{21}
$$

可实际同步解码的码率为：

$$
R(c_t)
=
\underbrace{
-\log_2p_\eta(m_t\mid\tilde h_t)
}_{\text{mask cost}}
+
\sum_jm_t^j
\left[
-\log_2
p_t^j(z_t^{+,j})
\right].
\tag{22}
$$

主方案令分组prior在给定$\tilde h_t$后条件独立，使编码器和解码器可以直接复现每组概率。若后续改用跨组自回归熵模型，条件中只能使用解码端已经重建的前缀$\tilde z_t^{<j}$，不能使用被省略的真实前缀$z_t^{+,<j}$。实际系统还需计入流终止、对齐和分段头开销。

### 7.4 修正解码

读取时，未发送的latent组由先验恢复，已发送的组使用残差码覆盖或修正：

$$
\tilde z_t^j=
\begin{cases}
z_t^{+,j}, & m_t^j=1,\\
\operatorname{Estimate}
\!\left[
p_t^j(z_t^j)
\right], & m_t^j=0.
\end{cases}
\tag{23}
$$

然后恢复状态：

$$
\tilde x_t=D_\omega(\tilde h_t,\tilde z_t).
\tag{24}
$$

这里的“Residual”并非简单计算$x_t-\hat x_t$，而是对**先验无法可靠决定的状态变量进行条件编码**。完成当前步后，下一步只能从$\tilde z_t$推进：

$$
\tilde h_{t+1}
=
f_\theta(\tilde h_t,\tilde z_t,u_t,d).
\tag{24a}
$$

$z_t^+$不得直接进入下一步持久状态；否则未发送的信息会绕过码率约束。

---

## 8. 任务效用门控

主方法不应依赖逐条精确计算的counterfactual utility label。更稳妥的训练方式是让发送mask直接接受式（42）和式（45）的率失真梯度，端到端学习哪些组值得发送；本节的$\Delta_t$与value head主要用于分析、蒸馏和固定预算下的长期淘汰，而不是ResidualMem成立的必要条件。

### 8.1 为什么需要效用

如果只按照式（16）保存高KL观察，系统会偏向保存：

- 视觉噪声；
- 罕见措辞；
- 无关背景变化；
- 随机但不会再次被使用的细节。

因此，ResidualMem需要估计保存某段残差对未来任务的反事实收益。

### 8.2 反事实效用

对候选残差$c_t$，分别在**解码端闭环轨迹**中计算保留和删除后的未来损失：

$$
\Delta_t
=
\mathcal L_{\mathrm{future/task}}(c_t=\varnothing)
-
\mathcal L_{\mathrm{future/task}}(c_t).
\tag{25}
$$

效用可包含：

- 后续状态重建改善；
- 未来latent预测改善；
- 记忆QA准确率改善；
- Agent任务成功率改善；
- 动态更新与冲突判断改善。

考虑存储成本后的净价值为：

$$
V_t
=
\Delta_t-\beta\,\operatorname{Bits}(c_t).
\tag{26}
$$

当$V_t>0$时，残差值得保存。

### 8.3 在线价值预测

训练时可以访问未来轨迹并近似计算$\Delta_t$，推理时不能。因此可选地训练一个value head：

$$
\hat\Delta_t
=
V_\psi
(p_t,q_t,c_t,g_t,\text{uncertainty}).
\tag{27}
$$

训练目标：

$$
\mathcal L_{\mathrm{value}}
=
\left(\hat\Delta_t-\Delta_t\right)^2,
\tag{28}
$$

其中$g_t$只能是写入时已经知道的任务目标，不能包含未来测试问题。若长期记忆面向未知查询，效用监督应来自训练分布上的未来状态恢复、合成查询和多任务目标，而不能直接使用benchmark测试QA。也可将效用离散为写入/跳过/压缩等级，使用分类或排序损失。

### 8.4 分组效用

效用不应只在整条观察级别计算。对每个残差组$j$估计：

$$
\Delta_t^j
=
\mathcal L(c_t^j\text{ omitted})
-
\mathcal L(c_t^j\text{ retained}).
\tag{29}
$$

然后按照效用密度分配容量：

$$
\rho_t^j
=
\frac{\Delta_t^j}{\operatorname{Bits}(c_t^j)}.
\tag{30}
$$

这样，一次观察可以只保存其中少数有用变化，而不是整体写入或整体丢弃。

需要注意，多个残差组和跨时间记忆之间可能存在互补关系，单项$\Delta_t^j$并不保证可加。式（30）只能作为候选排序信号，不能宣称是全局最优分配。训练时应随机mask残差集合估计上下文相关价值；淘汰后需要更新受影响条目的价值，或直接对segment级保留策略进行预算约束优化。

---

## 9. 外部残差记忆

### 9.1 记忆条目

每条记忆可写为：

$$
m_t=(k_t,c_t,\tau_t,u_t,v_t,\nu_t),
\tag{31}
$$

其中：

- $k_t$：检索键；
- $c_t$：条件残差码；
- $\tau_t$：时间与session位置；
- $u_t$：解码该残差所需、且无法从外部获得的动作side information；
- $v_t$：效用估计；
- $\nu_t$：所依赖的WM版本和编码器版本。

所有字段都必须计入存储成本：

$$
\operatorname{Bits}(M)
=
\sum_t
\left[
R(k_t)+R(c_t)+R(\tau_t)+R(u_t)+R(v_t)+R(\nu_t)
\right]
+
R(\text{anchors}).
\tag{32}
$$

否则模型可能把完整信息藏在检索键或自然语言metadata中。

### 9.2 原始证据指针

证据指针容易形成评测漏洞。若原始图片和文本始终免费可访问，则系统只需保存一个ID，不能反映真实记忆成本。

因此应采用以下协议之一：

1. 原始证据不可在推理时访问；
2. 若允许访问，原始证据本身的存储大小计入总预算；
3. 只保存用于审计的不可读取哈希，不作为回答依据；
4. 单独报告“latent-only”和“latent+raw-anchor”两种设置。

### 9.3 固定预算淘汰

当总记忆超过预算$B$时，可以将效用密度作为淘汰启发式：

$$
m^*
=
\arg\min_{m_i\in M}
\frac{\hat\Delta_i}{\operatorname{Bits}(m_i)}.
\tag{33}
$$

该规则不是全局最优解。淘汰后应重新评估依赖关系和上下文价值，避免删除某个锚点或中间状态后使后续残差无法解码。更稳妥的基本淘汰单位是“独立锚点起始的完整segment”，而不是任意单条条件码。

---

## 10. 顺序解码与随机访问

### 10.1 条件残差的依赖问题

残差$c_t$是在世界模型先验$p_t$条件下编码的。要正确解码$c_t$，必须重建生成$p_t$时的历史状态。因此，ResidualMem天然更接近预测视频编码，而不是彼此独立的向量文档。

若从$t=0$开始顺序回放：

$$
\tilde z_0
\xrightarrow{u_0,W}
p_1
\xrightarrow{c_1}
\tilde z_1
\xrightarrow{u_1,W}
\cdots
\xrightarrow{c_t}
\tilde z_t,
\tag{34}
$$

可以准确重建整条状态轨迹，但长历史查询成本较高。

### 10.2 锚点机制

为支持随机访问，每隔$K$步或在高价值边界处保存独立可解码锚点：

$$
A_r=(b_r,\tau_r).
\tag{35}
$$

其中$b_r$是完整量化状态或intra-coded keyframe，确定性状态由学习到的初始化器恢复：

$$
(\tilde h_r,\tilde z_r)
=
\operatorname{Init}_\xi(b_r).
\tag{35a}
$$

不能直接把未量化浮点$h_r$作为免费锚点，因为它可能携带任意历史信息。$b_r$及其索引必须全部计费。

查询时：

1. 检索距离目标时间最近的前置锚点；
2. 从锚点开始运行WM；
3. 顺序应用该区间内的残差码；
4. 得到目标状态或证据片段。

锚点间隔决定：

- 存储成本；
- 查询延迟；
- 长程误差累积；
- 随机访问能力。

它本身应作为率失真优化的一部分，而不是免费组件。

### 10.3 检索方式

查询$q$首先生成：

$$
r_q=G(q,\text{current state}).
\tag{36}
$$

检索器返回相关时间区间、实体和锚点，而不是直接返回完整答案：

$$
\mathcal S
=
\operatorname{Retrieve}(r_q,\{k_i,\tau_i\}).
\tag{37}
$$

随后由WM与残差共同重建相关状态。由于条件码存在因果依赖，从锚点到目标时间之间的动作side information和所有被保留的前置残差必须按顺序回放；检索器不能任意跳过“看似与查询无关”的中间残差，否则后续先验可能改变。实体索引负责定位segment，真正解码的单位是因果闭合的连续片段。这样可以保留ResidualMem的核心要求：回答来自“模型先验＋现实修正”，而不是把外部记忆退化为普通RAG文本库。

---

## 11. 训练方法

### 11.1 阶段0：数据标准化

将不同benchmark和环境统一为：

$$
(o_t,u_t,o_{t+1},\text{state labels},\text{future queries}).
\tag{38}
$$

需要区分：

- 可预测的动作条件变化；
- 状态持久性；
- 外生事件；
- 随机结果；
- 观察噪声；
- 未来任务相关标签。

对于没有显式动作的数据，只能使用事前已知的会话边界、时间间隔和任务条件。不能从当前观察中抽取事件语义后再作为同一步先验条件，也不能伪造可预测的因果转移。

### 11.2 阶段1：训练状态tokenizer

冻结Qwen3.5主干，训练Perceiver Resampler和状态probe。监督只能来自训练轨迹、环境状态或冻结教师生成的规范状态，不能来自benchmark测试QA。目标包括：

$$
\mathcal L_{\mathrm{state}}
=
\lambda_e\mathcal L_{\mathrm{entity}}
+
\lambda_a\mathcal L_{\mathrm{attribute}}
+
\lambda_r\mathcal L_{\mathrm{relation}}
+
\lambda_v\mathcal L_{\mathrm{visual}}
+
\lambda_f\mathcal L_{\mathrm{feedback}}.
\tag{39}
$$

该阶段保证$x_t$能够表达真实状态，而不是仅保留容易预测的信息。训练结束后冻结状态tokenizer，并在独立验证集上检查精确OCR、实体属性、视觉关系和未来未知查询的恢复率；若这些probe不合格，后续低码率结果不具备解释力。

### 11.3 阶段2：预训练概率世界模型

训练先验、后验和状态解码器：

$$
\mathcal L_{\mathrm{WM}}
=
\mathcal L_{\mathrm{recon}}
+
\beta_{\mathrm{KL}}
D_{\mathrm{KL}}(q_t\|p_t)
+
\lambda_{\mathrm{multi}}
\mathcal L_{\mathrm{multi-step}}
+
\lambda_{\mathrm{cal}}
\mathcal L_{\mathrm{calibration}}.
\tag{40}
$$

其中：

- $\mathcal L_{\mathrm{recon}}$：从后验latent恢复状态token和结构化状态；
- $\mathcal L_{\mathrm{multi-step}}$：闭环预测多个时间跨度；
- $\mathcal L_{\mathrm{calibration}}$：保证预测概率与实际错误率匹配。

训练先使用teacher forcing学习基本动力学，再逐步切换到解码端闭环状态：

$$
\tilde h_t
=
f_\theta
(\tilde h_{t-1},\tilde z_{t-1},u_{t-1}),
\tag{41}
$$

避免只在真实上一状态输入下表现良好、闭环运行时迅速漂移。

### 11.4 阶段3：冻结WM，训练条件残差codec

冻结观察编码器和WM，训练：

- 分组残差选择器；
- 条件熵模型；
- 修正解码器。

采用任务相关率失真目标：

$$
\mathcal L_{\mathrm{codec}}
=
\mathcal D_{\mathrm{state}}(x_t,\tilde x_t)
+
\alpha\mathcal D_{\mathrm{future}}
+
\gamma\mathcal L_{\mathrm{task}}
+
\beta R(c_t).
\tag{42}
$$

其中：

$$
\mathcal D_{\mathrm{state}}
=
\lambda_sD_{\mathrm{semantic}}
+
\lambda_vD_{\mathrm{visual}}
+
\lambda_rD_{\mathrm{relation}}.
\tag{43}
$$

训练阶段使用可微的交叉熵作为码率代理，并通过straight-through、Gumbel或策略梯度训练离散mask；真正的range/arithmetic coder只在验证和测试时生成字节流并报告实际bit。冻结WM的原因是防止WM与codec形成投机协作，例如WM故意输出含有样本ID的隐藏信息，或故意降低先验质量把责任转移给记忆。

### 11.5 阶段4：可选的任务效用蒸馏

对候选残差及其分组进行counterfactual ablation：

```text
保留残差 -> 运行未来状态重建 / QA / Agent任务
删除残差 -> 运行相同任务
两者损失差 -> 监督效用
```

发送mask在阶段3已经由率失真目标端到端学习。本阶段仅在需要在线淘汰或解释性分析时，训练value head预测式（25）和式（29）的效用。所有counterfactual分支都必须从相同锚点开始，并用各自重建状态闭环滚动，不能在删除残差后继续使用真实后验状态。为降低计算，可以：

- 随机mask残差组；
- 使用Shapley近似；
- 训练低成本价值代理；
- 对高surprise样本增加评估频率；
- 以未来query覆盖范围限制counterfactual窗口。

### 11.6 阶段5：训练检索与随机访问

使用benchmark中的证据标注、时间范围和实体关系训练检索器：

$$
\mathcal L_{\mathrm{retrieval}}
=
\mathcal L_{\mathrm{contrastive}}
+
\lambda_t\mathcal L_{\mathrm{temporal}}
+
\lambda_s\mathcal L_{\mathrm{segment}}.
\tag{44}
$$

训练目标不是找到答案文本，而是定位：

- 需要哪个锚点；
- 哪段残差链；
- 哪些实体和状态变量；
- 哪个时间范围。

### 11.7 阶段6：固定预算联合微调

最终目标为：

$$
\mathcal L_{\mathrm{total}}
=
\mathcal L_{\mathrm{task}}
+
\lambda_1\mathcal D_{\mathrm{state}}
+
\lambda_2\mathcal L_{\mathrm{future}}
+
\lambda_3\mathcal L_{\mathrm{retrieval}}
+
\lambda_4\mathcal L_{\mathrm{value}}
+
\beta\operatorname{Bits}(M).
\tag{45}
$$

联合微调时建议：

- Qwen3.5保持冻结；
- WM在最终生成任何持久残差前冻结；若联合更新WM，则每次更新后必须重新生成训练码流；
- 主要训练codec、utility gate、retriever和correction decoder；
- 使用多个预算$B$训练一组rate-performance operating points；
- 每个batch显式执行容量约束和淘汰。

---

## 12. 推理流程

### 12.1 记忆写入

对每个新观察：

1. Qwen3.5独立编码观察，得到$x_t$；
2. WM在看不到$x_t$时产生$p_t$；
3. 后验模型结合$x_t$得到$q_t$；
4. 条件codec生成不同码率的候选$c_t$；
5. 率失真mask policy选择发送哪些残差组；若部署固定预算淘汰器，再由可选value head提供排序；
6. 对选中的残差组执行实际熵编码并写入；
7. 编码端可临时使用$z_t^+$计算码，但持久动力学必须使用$\tilde z_t$推进；
8. 若允许固定长度的工作记忆，其容量、保留时长和跨session清空规则必须单独声明并保持所有方法一致。

当前时刻的Agent policy可以读取刚到达的真实观察$x_t$完成即时反应；这里限制的是跨步、跨会话保留的信息必须经过$\tilde z_t$或显式计费的工作记忆，不能把当前观察永久藏入未受约束的循环状态。

### 12.2 记忆读取

收到查询或任务时：

1. 检索相关实体、时间区间和锚点；
2. 从独立可解码锚点恢复$\tilde h_r,\tilde z_r$；
3. 按时间回放完整动作side information和因果闭合残差segment；
4. 恢复相关状态；
5. 将修正状态交给QA reader或Agent policy；
6. 输出答案、证据或动作。

### 12.3 跨会话

跨session时：

- 清空VLM上下文；
- 清空未计费的临时cache；
- 保留固定参数WM；
- 保留受预算约束的Residual Memory；
- 可保留一个计费的session anchor；
- 下一session通过WM和残差重建必要状态。

---

## 13. 记忆内容的语义解释

ResidualMem最终应自然保存三类信息：

### 13.1 初始条件

世界模型掌握规律，但不知道当前具体世界：

- 用户姓名；
- 当前居住地；
- 项目使用的框架；
- 房间初始布局；
- 当前文件和任务状态。

这些信息不一定高surprise，却无法由一般规律推导，因此需要编码。

### 13.2 外生创新

无法由历史决定的变化：

- 用户突然改变偏好；
- 论文收到外部评审结果；
- 新任务目标被提出；
- 外部对象被其他主体移动。

### 13.3 模型例外

世界实际演化偏离模型：

- 点击按钮后出现异常弹窗；
- 提交动作因支付失败而未完成；
- 文件移动后仍出现在原目录；
- 机器人抓取失败；
- 特定网站存在与通用流程不同的“坑”。

这三类内容可概括为：

$$
\text{Residual Memory}
=
\text{Initial Conditions}
+
\text{Innovations}
+
\text{Model Exceptions}.
\tag{46}
$$

---

## 14. Benchmark与实验设计

### 14.1 核心benchmark

#### WorldMemArena

用途：

- 作为多模态长期记忆主榜；
- 测试Dynamic Update、Memory Conflict、Temporal Reasoning、Visual Update和Agentic Execution；
- 比较最终QA与记忆生命周期表现。

注意：

- Agentic Execution最适合验证动作条件残差；
- Lifelong Evolution包含大量外生事实，更适合验证初始条件和外生创新；
- 仅依赖最终QA不能证明WM不可替代，必须同时报告码率和机制指标。
- 官方评测还涉及memory writing、maintenance和retrieval等阶段；latent memory需要增加只用于评测的export/probe head，将内部状态解码为memory points和证据ID。该head的输出不能反馈到实际记忆存储。

#### LongMemEval-V2

用途：

- 测试Web Agent长期经验；
- Dynamic State Tracking；
- Workflow Knowledge；
- Environment Gotchas；
- Premise Awareness；
- 查询延迟。

其中Environment Gotchas与ResidualMem的“模型例外”高度匹配。

LongMemEval-V2官方定位为test-only benchmark。它应作为冻结系统的评测集，不能用于训练状态tokenizer、WM、utility gate或检索器。若官方harness要求返回文本或图像上下文，需要增加benchmark adapter，将重建状态解码为规范证据，再交给固定reader；同时报告原生latent reader结果和harness兼容结果。

#### EMemBench

用途：

- Jericho文本游戏和Crafter视觉环境；
- 可访问环境ground-truth信号；
- 可根据Agent自身轨迹生成可验证QA；
- 可控地改变WM强度、记忆预算和轨迹长度；
- 验证“WM越强，记忆越省”的因果关系。

EMemBench应作为机制验证的核心benchmark。训练轨迹应由独立环境种子或训练关卡采集，测试关卡和测试Agent轨迹保持隔离。

#### STATE-Bench

用途：

- Travel、Customer Support和Shopping Assistant任务；
- 使用Agent Learning Track的历史轨迹和retrieval hook；
- 测试记忆是否能让Agent从经验中获得可复用改进；
- 报告任务成功率、稳定性、成本和用户体验。

### 14.2 补充benchmark

- **MemGUI-Bench**：移动GUI跨时间、跨空间和跨应用记忆；
- **MEMTRACK**：Slack、Linear和Git交错事件中的效率与冗余；
- **AMA-Bench**：长轨迹和固定容量压力测试；
- **Mem-Gallery**：多模态会话、更新、冲突与拒答；
- **Ego4D Episodic Memory**：长视频中的历史事件定位；
- **WorldArena**：可用于单独评价具身WM能力，但不是长期记忆主榜。

### 14.3 推荐实验组合

考虑训练成本和论证闭环，主论文更合理的优先级是：

1. EMemBench：受控机制与核心因果实验；
2. WorldMemArena：综合多模态长期记忆主榜；
3. LongMemEval-V2：Web环境经验与例外；
4. STATE-Bench或MemGUI-Bench二选一作为Agent执行补充。

同时全面覆盖五个benchmark会使统一WM的数据适配和运行成本过高，STATE-Bench与MemGUI-Bench更适合作为扩展结果。

---

## 15. 评价指标

### 15.1 任务指标

- QA Accuracy；
- Agent Task Success；
- pass@1 / pass@k；
- Dynamic Update Accuracy；
- Interference Rejection；
- Temporal and Spatial Reasoning；
- Visual Memory Accuracy；
- State Reconstruction Accuracy。

### 15.2 记忆成本

真实记忆成本应包含：

- 残差内容码；
- mask；
- 检索键；
- 时间戳；
- 动作/事件metadata；
- 锚点；
- 索引结构；
- 若可访问，原始证据。
- 无法从环境重新获得的动作或事件side information。

定义：

$$
\text{Memory Rate}
=
\frac{\operatorname{Bits}(M)}
{\operatorname{Bits}(\text{raw trajectory})}.
\tag{47}
$$

分母必须采用固定序列化协议，例如UTF-8文本、指定质量的PNG/JPEG或官方原始文件字节，不能由不同方法自行选择。主rate-distortion图还应使用相同冻结latent目标的Store-All-Latent作为分母，以排除不同感知编码器带来的压缩差异。

应同时报告两种口径：

- **Conditional Observation Rate**：假设动作日志由环境免费提供，只计算观察残差；
- **Total Episodic Rate**：动作、事件、索引、锚点和残差全部计费。

此外，论文主张的是外部记忆节省。若比较不同大小的WM，还应补充总描述长度：

$$
\operatorname{TotalCost}
=
\operatorname{Bits}(W)
+
\sum_n\operatorname{Bits}(M_n),
\tag{47a}
$$

并说明共享WM参数在多少用户或轨迹上摊销，避免用无限增大的参数模型换取较小外部记忆却仍宣称“整体更省”。

### 15.3 互补性指标

$$
\text{Complementarity Gain}
=
J(WM+M)-J(WM).
\tag{48}
$$

$$
\text{Memory Saving@Quality}
=
1-
\frac{B_{\mathrm{ResidualMem}}}
{B_{\mathrm{baseline}}}.
\tag{49}
$$

其中$B$是在达到相同任务质量时所需的记忆bit。

### 15.4 机制指标

- Prior prediction accuracy；
- Posterior reconstruction accuracy；
- KL innovation；
- 实际条件码长；
- 每步写入概率；
- 每条记忆平均码率；
- surprise与utility的相关性；
- 残差删除后的性能下降；
- 查询需要回放的平均步数；
- 锚点数量与随机访问延迟。

### 15.5 最关键图表

1. Task Performance vs. Memory Bits；
2. Required Memory Bits@Fixed Quality vs. WM Quality；
3. Trajectory Length vs. Memory Growth；
4. Surprise vs. Utility散点图；
5. Anchor Interval vs. Storage/Latency；
6. 不同信息类型的平均残差码率；
7. 不同预算下的写入内容可视化。

---

## 16. Baseline与消融

### 16.1 Baseline

- Full History / Store-All；
- Long Context；
- Vector RAG；
- Summary Memory；
- LRU / Reservoir Sampling；
- LLM Importance Scoring；
- Surprise-Gated Full Observation；
- Nemori；
- B'MOJO式Innovation Selection；
- D-MEM式Surprise+Utility Gate；
- 固定间隔情景记忆；
- 只保存结构化状态变化；
- WM-only，无外部记忆。

压缩效率的主比较必须在同一个冻结目标空间$x_t$、同一个量化精度和相同metadata规则下进行。至少增加：

- Store-All-Latent：逐步保存全部$z_t^+$；
- Unconditional Codec：不使用WM先验独立编码$z_t^+$；
- Temporal Codec：只使用上一状态，不使用动作条件WM；
- Surprise-Gated Full-Latent：高surprise时保存完整$z_t^+$；
- ResidualMem：按WM先验进行部分条件编码。

原始文本RAG、摘要和长上下文属于系统级应用baseline，但其字节数与latent codec不能直接视为严格同口径结果，应单独报告。

### 16.2 核心消融

1. 无WM，仅训练残差自编码器；
2. 无utility，只按surprise写入；
3. 无可变码率，只做整条观察二元写入；
4. 无条件编码，独立编码每个观察；
5. 不计metadata和索引成本；
6. 无锚点，只顺序回放；
7. 不同latent类型：高斯、categorical、VQ；
8. 不同编码器：Qwen3.5-2B/4B/9B；
9. 不同WM规模、数据量和动作条件；
10. WM冻结与联合训练；
11. 单步WM与多步WM；
12. 只重建状态与加入未来任务损失。

---

## 17. 如何验证“WM越强，记忆越省”

这是整篇论文最重要、也最容易产生混淆的实验。

### 17.1 控制变量

必须固定：

- 观察编码器；
- latent维度；
- residual codec；
- utility gate容量；
- 训练轨迹；
- 下游reader；
- 记忆预算计算方式。

只改变WM能力，例如：

- 模型参数规模；
- WM预训练数据量；
- 是否使用动作；
- 单步或多步预测；
- 是否接受WebWorld/gWorld/V-JEPA蒸馏；
- 训练轮数。

“WM更强”必须定义为在同一冻结表示和严格held-out转移数据上具有更低的条件NLL、更好的校准和不差于基线的闭环预测，而不能仅以参数量或单步准确率定义。更大的模型若自信地预测错误，反而可能增加条件码长。

### 17.2 比较方式

不能只比较固定预算下的准确率，还要对每个WM训练完整rate-performance曲线，并在固定质量$Q$处读取所需容量：

$$
B^*(Q,W)
=
\min_B
\{B:J(W,M_B)\geq Q\}.
\tag{50}
$$

目标是观察：

$$
W_1<W_2<W_3
\quad\Longrightarrow\quad
B^*(Q,W_1)>B^*(Q,W_2)>B^*(Q,W_3).
\tag{51}
$$

### 17.3 防止伪结论

需要排除：

- 更强WM同时拥有更大未计费hidden state；
- 更强WM直接记住了测试轨迹；
- 编码器随WM变化而丢失信息；
- codec对某个WM单独过拟合；
- WM参数本身包含特定测试实例；
- 不同WM使用了不同原始输入。

---

## 18. 关键技术风险

### 18.1 世界模型并不适用于所有信息

跨会话个人事实中，大量内容是外生披露，无法预测。此时ResidualMem仍然有效，但压缩主要来自条件编码和任务相关丢弃，而不是动作动力学。

应明确报告不同任务类型：

- 动作条件动态；
- 状态持久性；
- 外生更新；
- 随机结果；
- 纯事实披露。

### 18.2 Qwen latent不一定是稳定状态空间

Qwen hidden state可能：

- 对prompt模板敏感；
- 偏向回答语义；
- 缺少时间连续性；
- 对局部视觉变化不稳定。

需要通过状态probe、跨视角一致性、时间稳定性和结构化状态恢复实验验证。

### 18.3 posterior collapse

强解码器可能忽略随机latent，使$q_t$接近$p_t$，造成虚假低残差。应采用：

- KL balancing；
- free bits；
- latent dropout；
- 限制解码器绕过latent；
- 状态probe监督；
- 后验信息量监控。

### 18.4 utility label成本

精确计算每个残差组的反事实价值非常昂贵。需要使用采样mask、代理value model和局部时间窗口近似。

### 18.5 条件码随机访问困难

条件编码提高压缩率，却引入顺序依赖。锚点策略与检索必须共同设计，并真实报告查询延迟。

### 18.6 世界模型更新导致旧残差失效

若WM参数改变，原残差是在旧先验条件下编码的，可能无法由新WM正确解码。因此第一阶段项目建议固定WM。在线巩固作为后续扩展，需要：

- WM版本标记；
- 残差重新编码；
- 或保持旧WM decoder。

### 18.7 任务相关压缩可能牺牲开放式记忆

若utility只由benchmark问题监督，系统可能丢弃未被当前问题覆盖但未来可能重要的信息。需要：

- 多任务效用；
- 状态重建下限；
- 开放式未来query生成；
- 保留少量通用创新预算；
- 测试分布外问题。

### 18.8 编码端信息泄漏到动力学状态

若后验状态或完整观察在未写入时仍进入$h_{t+1}$，系统会通过RSSM隐藏状态绕过外部记忆预算。所有持久预测必须使用解码端可重建状态$\tilde z_t,\tilde h_t$，工作记忆必须固定容量、计费或在会话边界清空。

### 18.9 动作side information未计费

条件预测需要动作序列。若动作日志在查询时并非天然可得，却被当作免费输入，码率会被系统性低估。必须同时报告条件观察码率与包含动作日志的总情景码率。

### 18.10 统一WM的跨域负迁移

文本会话、网页、GUI和具身环境的动力学差异很大。完全共享单一transition head可能导致负迁移。统一应优先体现在接口和目标上，参数层面使用共享骨干加领域adapter或专家路由，并通过跨域留出实验验证。

---

## 19. 当前建议的最终模型配置

| 组件 | 主方案 |
|---|---|
| 观察编码器 | Qwen3.5-9B-Base，冻结 |
| 规模消融 | Qwen3.5-2B/4B/9B-Base |
| 状态tokenizer | Perceiver Resampler |
| 世界模型 | Causally Conditioned Transformer-RSSM + Domain Adapters |
| 随机状态 | Grouped Categorical Latent |
| 理论创新量 | $D_{\mathrm{KL}}(q_t\|p_t)$ |
| 实际记忆编码 | Decoder-Synchronizable Conditional Entropy Coding + Group Masks |
| 写入策略 | End-to-End Rate-Distortion Mask；可选Utility-per-Bit蒸馏 |
| 外部记忆 | Causally Closed Residual Segments + Independently Decodable Anchors |
| 状态恢复 | WM Rollout + Residual Correction Decoder |
| 下游读取 | Qwen3.5 Instruct Reader或任务专用Head |
| 主benchmark | EMemBench、WorldMemArena、LongMemEval-V2 |
| Agent执行补充 | STATE-Bench或MemGUI-Bench（二选一） |

---

## 20. 项目边界

当前ResidualMem主项目不把以下内容作为必要组成：

- 在线更新世界模型；
- 参数化记忆巩固；
- 技能学习；
- 多智能体共享记忆；
- 长程规划；
- 完整像素视频生成；
- 独立的显式知识图谱；
- 用Cosmos或Wan统一覆盖全部任务。

重复残差巩固进WM是自然的后续方向：

```text
重复残差出现
-> 聚类为稳定模式
-> 更新WM adapter
-> 重新编码旧残差
-> 删除已经可预测的信息
```

但它会引入WM版本兼容、灾难性遗忘和全量重编码问题，不应成为第一篇论文主结论成立的前提。

---

## 21. 预期贡献

ResidualMem拟形成以下四项贡献：

### 21.1 世界模型互补记忆定义

将长期记忆定义为相对于世界模型先验的外部修正信息，而非独立的历史存档。

### 21.2 条件可变码率记忆

将记忆写入从二元事件选择推广为条件率失真编码：完全可预测的信息不存，部分可预测的信息少存，不可预测且有用的信息详细存。

### 21.3 任务效用与实际bit联合优化

统一优化状态恢复、未来任务质量和真实存储成本，并计入索引、锚点与metadata。

### 21.4 WM—Memory能力关系

通过受控实验验证：世界模型能力增强是否会系统性降低达到固定任务质量所需的外部记忆容量。

---

## 22. 核心表述

### 22.1 一句话定义

> ResidualMem将长期记忆学习为世界模型的任务相关修正码：模型能够可靠恢复的信息不重复存储，只编码现实相对模型先验的必要创新量。

### 22.2 核心公式

世界模型先验：

$$
p_t=p_\theta(z_t\mid\tilde h_t).
\tag{52}
$$

观察后验：

$$
q_t=q_\phi(z_t\mid\tilde h_t,x_t).
\tag{53}
$$

编码端目标状态：

$$
z_t^+
=
\operatorname{Quantize}
\!\left[
q_\phi(z_t\mid\tilde h_t,x_t)
\right].
\tag{54}
$$

条件残差与解码状态：

$$
c_t=C_\psi(p_t,z_t^+),
\qquad
\tilde z_t=R_\omega(p_t,c_t).
\tag{55}
$$

率失真目标：

$$
\min
\mathcal L_{\mathrm{task}}
+
\alpha\mathcal D_{\mathrm{state}}
+
\beta\operatorname{Bits}(M).
\tag{56}
$$

### 22.3 论文标题候选

首选：

> **ResidualMem: Learning World-Model-Complementary Memory via Conditional Residual Coding**

强调率失真版本：

> **ResidualMem: Rate-Distortion Learning of World-Model-Complementary Memory**

强调核心命题版本：

> **Remember What the World Model Cannot: Residual Memory for Long-Horizon Agents**

---

## 23. 尚待确定的问题

以下问题需要通过原型实验后再固定：

1. Qwen3.5应提取哪一层hidden state；
2. 是否需要额外DINOv3/V-JEPA视觉流；
3. 状态token数量与维度；
4. categorical latent的组数与码本大小；
5. 残差按latent组、实体还是空间区域编码；
6. 锚点采用固定间隔还是学习触发；
7. utility采用未来QA、状态恢复还是Agent reward作为主信号；
8. WorldMemArena中缺乏显式动作的部分如何构造事件条件；
9. 统一WM是否应进行领域adapter分支；
10. 是否能获得足够规模的训练轨迹训练统一WM；
11. 真实算术编码是否进入主实验，还是先报告熵估计；
12. 下游reader是否共享Qwen3.5权重。
13. 动作日志在各benchmark中能否被视为免费side information；
14. utility gate采用组级、条目级还是segment级预算优化；
15. 统一WM采用共享骨干加adapter还是mixture-of-experts；
16. 原生latent reader与官方harness adapter如何保持公平。

这些属于实现选择，不改变ResidualMem的核心科学问题：

> 在给定一个世界模型后，长期智能体能否只保存世界模型无法可靠重建、且对未来任务有价值的信息？

---

## 参考资料

### 近邻方法

- B'MOJO: Hybrid State Space Realizations of Foundation Models with Eidetic and Fading Memory.  
  <https://arxiv.org/abs/2407.06324>
- What Deserves Memory: Adaptive Memory Distillation for LLM Agents.  
  <https://arxiv.org/abs/2508.03341>
- Worth Remembering: Surprise-Gated Robot Episodic Memory.  
  <https://arxiv.org/abs/2606.03787>
- D-MEM: Dopamine-Gated Agentic Memory via Reward Prediction Error Routing.  
  <https://arxiv.org/abs/2603.14597>
- Curious Replay for Model-Based Adaptation.  
  <https://proceedings.mlr.press/v202/kauvar23a.html>

### 模型与世界模型

- Qwen3.5-9B-Base.  
  <https://huggingface.co/Qwen/Qwen3.5-9B-Base>
- V-JEPA 2 / V-JEPA 2-AC.  
  <https://github.com/facebookresearch/vjepa2>
- WebWorld.  
  <https://github.com/QwenLM/WebWorld>
- gWorld-8B.  
  <https://huggingface.co/trillionlabs/gWorld-8B>
- DreamerV3.  
  <https://github.com/danijar/dreamerv3>

### Benchmark

- WorldMemArena.  
  <https://github.com/UCSB-AI/WorldMemArena>
- LongMemEval-V2.  
  <https://github.com/xiaowu0162/LongMemEval-V2>
- EMemBench.  
  <https://github.com/InternLM/EMemBench>
- STATE-Bench Agent Learning Track.  
  <https://github.com/microsoft/STATE-Bench>
- MemGUI-Bench.  
  <https://github.com/lgy0404/MemGUI-Bench>
