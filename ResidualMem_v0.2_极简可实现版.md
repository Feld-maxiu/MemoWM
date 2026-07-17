# ResidualMem：世界模型条件残差记忆

> 历史设计文档：当前实现已升级到 v0.3 exact progressive memory；v0.2 stream/checkpoint 不兼容。当前契约见 `ResidualMem_v0.3_exact_progressive.md`。

> 技术报告（极简可实现版）  
> 版本：v0.2  
> 日期：2026-07-13

---

## 摘要

长时程智能体会不断接收文本、图像、GUI、工具反馈和环境状态。传统长期记忆通常保存完整观察、摘要、检索片段或被判定为“重要”的事件，但很少显式利用智能体已经掌握的环境规律。结果是，外部记忆会重复保存大量可由世界模型恢复的常规状态。

ResidualMem 的核心思想很简单：

> **世界模型给出下一状态的默认解释，外部记忆只保存现实相对该默认解释的必要修正。**

本版本不再把问题建立在随机后验、统一多模态 latent、Transformer-RSSM、学习式效用门控和端到端联合训练之上，而采用四个可独立实现和测试的模块：

1. 确定性的领域状态适配器；
2. 在独立分段内运行的状态预测器；
3. 基于实际字节成本的类型化残差编码器；
4. 带完整锚点、可独立回放的不可变 Segment 存储。

系统首先把每一步观察转换为固定、可序列化的规范状态。世界模型只读取解码端已经能够恢复的历史状态和已知动作，预测当前状态。若预测已经足够准确，则不写入；若某些字段偏离预测，则只保存这些字段的修正。任务相关性不由昂贵的逐字段反事实 rollout 决定，而由固定的字段保真策略和失真权重表达。

本报告只主张压缩**规范状态**，不默认等同于无损压缩原始图像或原始文本。多模态感知质量由状态适配器单独评测。第一阶段论文集中验证三个问题：ResidualMem 是否优于完整状态存储和普通增量日志；更好的世界模型是否会减少达到固定质量所需的残差字节；预测误差与任务相关失真是否需要区分。

---

## 1. 设计结论

### 1.1 一句话定义

> ResidualMem 是一种预测式长期记忆：解码器先用世界模型生成默认状态，再用外部残差将默认状态修正为需要保留的现实状态。

### 1.2 最小系统

```text
Observation
    |
    v
Deterministic State Adapter
    |
    v
Canonical State x_t
    |
    +-------------------------------+
    |                               |
Previous Reconstructed State        |
+ Known Action                      |
    |                               |
    v                               |
Segment-local World Model           |
    |                               |
    v                               |
Predicted Default State x_bar_t     |
    |                               |
    +---------- compare ------------+
                 |
                 v
        Typed Residual Record
                 |
                 v
        Immutable Segment Store
```

### 1.3 五条不可破坏的原则

1. **目标状态固定。** 世界模型训练前，状态定义必须冻结。
2. **解码端闭环。** 下一步预测只能使用已经重建的状态，不能使用未写入的真实状态。
3. **分段独立。** 每个 Segment 从完整锚点开始，编码端与解码端在边界处执行相同重置。
4. **所有持久信息计费。** 残差、锚点、动作、索引、时间和元数据都计入实际字节。
5. **代码正确性先于模型复杂度。** 编解码一致性、随机访问和版本检查未通过前，不增加更复杂的 latent 或 utility 模型。

---

## 2. 对 v0.1 方案的关键修正

### 2.1 不再把 Qwen hidden state 当作唯一“真实状态”

Qwen 或其他多模态模型可以作为观察适配器，但其隐藏状态不是天然稳定、可解释、可精确序列化的世界状态。新版要求观察先被转换为规范字段；精确文本、数值和类别拥有显式类型，视觉细节可选地转换为离散视觉 token。

### 2.2 删除随机后验和 Transformer-RSSM

首版不需要
$$
\
q_\phi(z_t\mid h_t,x_t),
\qquad
D_{\mathrm{KL}}(q_t\|p_t),
\
$$
也不需要解决 posterior collapse、KL balancing、free bits 和随机采样同步问题。世界模型直接预测确定性规范状态的条件分布或默认值。

### 2.3 锚点初始化并非天然不可行，但必须对称

学习式初始化器并不是理论上不可用。真正错误的情况是：编码端继续使用锚点之前的隐藏状态，而解码端只能近似恢复它。

新版采用最简单规则：
$$
\
h_r=0
\
$$
并把锚点状态作为 Segment 的第一个显式输入。编码端和解码端都从零状态开始，因此不存在隐藏历史不一致。

### 2.4 不把生成式 Reader 说成“无法训练”

生成式 Reader 可以通过 teacher forcing、策略梯度或离散近似参与训练，但计算昂贵且容易把多个问题混在一起。新版不将其纳入核心训练图；Reader 只负责最终评测。

### 2.5 动作日志不是天然泄漏

若动作序列在真实部署中可获得，并且所有方法使用相同固定轨迹，那么动作可以作为合法 side information。只有在行为策略与记忆编码器联合优化、且动作又不计费时，动作才可能成为免费通信信道。

因此新版要求：

- 机制实验使用固定离线轨迹；
- 条件观察码率与包含动作的总情景码率分别报告；
- 在线策略实验中，动作必须固定、计费或与记忆目标解耦。

### 2.6 条件算术编码不是第一版的必要条件

上一版建议把整数 CDF 和 range coder 当作主实现。它们对追求最优压缩有价值，但会把浮点确定性、跨硬件同步和熵模型工程引入核心闭环。

新版主实现使用**类型化二进制残差码**：布尔翻转、类别替代码、整数差分、字符串字典和固定宽度视觉 token。这样已经可以真实统计字节并验证核心命题。模型条件熵编码作为后续优化，不是论文成立的前提。

### 2.7 “世界模型越强，记忆必然越小”不是无条件定理

在完整无损条件编码中，更低的 held-out cross-entropy 通常对应更短的期望码长。但在带失真、固定掩码和任务权重的系统中，不存在对任意模型都成立的严格单调定理。

因此新版将其写成两个层次：

- **机制事实：** 更准确的默认状态通常产生更少、更小的修正；
- **经验假设：** 在相同表示、分段、编码器和任务质量下，更好的世界模型需要更少实际字节。

---

## 3. 问题设定与项目边界

考虑轨迹
$$
\
\tau=(o_0,u_0,o_1,u_1,\ldots,o_T),
\
$$
其中：

- \(o_t\) 是时刻 \(t\) 的观察；
- \(u_t\) 是产生 \(o_{t+1}\) 前已经知道的动作、时间间隔或领域条件；
- \(x_t\) 是由观察适配器得到的规范状态；
- \(q\) 是未来查询或任务；
- \(y\) 是正确答案或任务结果。

ResidualMem 学习或构造外部记忆 \(M(\tau)\)，目标是在给定记忆预算时保持尽可能高的任务质量：
$$
\
\min_M
\mathbb E_{\tau,q}
\left[
\mathcal L_{\mathrm{task}}(M,\tau,q)
\right]
\quad
\text{s.t.}
\quad
\operatorname{Bits}(M)\le B.
\
$$

### 3.1 本项目明确做什么

- 压缩规范状态序列；
- 利用动作条件和状态持久性预测默认状态；
- 仅保存必要字段修正；
- 支持独立 Segment 回放；
- 报告真实序列化字节；
- 研究世界模型质量与残差存储的关系。

### 3.2 本项目第一阶段不做什么

- 不无损重建原始像素；
- 不训练统一覆盖所有领域的单一大世界模型；
- 不在线更新世界模型；
- 不学习 Shapley utility 或逐组长期反事实价值；
- 不允许任意删除 Segment 中间残差；
- 不把检索模型、Reader 和 codec 端到端联合训练；
- 不宣称任务相关压缩能保留所有未知未来问题所需信息。

---

## 4. 规范状态

### 4.1 状态接口

每个领域实现一个确定性适配器：

```python
class StateAdapter:
    schema_id: str

    def encode(self, observation) -> "CanonicalState":
        ...

    def render(self, state: "CanonicalState") -> list[dict]:
        ...
```

`encode` 只在写入端运行；解码端不需要重新读取原始观察。`render` 把重建状态转换为 Reader 可消费的证据。

规范状态写成固定或有界字段集合：
$$
\
x_t=(x_t^1,\ldots,x_t^M).
\
$$
每个字段具有：

```python
@dataclass(frozen=True)
class FieldSpec:
    name: str
    field_type: Literal["bool", "categorical", "integer", "literal", "vq"]
    policy: Literal["must", "weighted"]
    weight: float
    num_values: int | None = None
    scale: float = 1.0
```

字段策略含义：

- `must`：目标值与默认值不同时必须保存；
- `weighted`：根据率失真规则决定是否保存。

不需要进入长期记忆的即时信号不应放入 `CanonicalState`；它们由当前时刻的 Agent policy 直接使用。

### 4.2 推荐字段类型

| 类型 | 示例 | 默认恢复 | 残差编码 |
|---|---|---|---|
| Boolean | 对话框是否打开 | 模型预测 | mask 即可表示翻转 |
| Categorical | 任务状态、按钮状态 | argmax | 替代类别编号 |
| Integer | 坐标、库存、计数 | 预测整数/中位数 | ZigZag VarInt 差分 |
| Literal | 用户名、文件名、错误码 | 通常沿用前值 | 字典 ID 或 UTF-8 字节 |
| VQ Token | 低码率视觉细节 | argmax token | 替代 token 编号 |

### 4.3 精确文本的处理

精确文本不应只依赖语义 latent。若用户名、文件名或错误码必须逐字符恢复，则它们使用 `literal` 字段：

1. 若值未变化，不写入；
2. 若值已在 Segment 字典中，写入字典 ID；
3. 若是新值，写入长度和 UTF-8 字节，并加入字典。

这不是说神经 latent 理论上无法表示字符串，而是明确保证精确恢复时需要显式、可验证的无损通道。

### 4.4 多模态观察如何接入

多模态感知与记忆编码分开评测。

#### 机制轨道

使用环境结构化状态或可验证标签生成 \(x_t\)，用于隔离 ResidualMem 本身的效果。这是 oracle-state 诊断，不等同于端到端智能体成绩。

#### 端到端轨道

由冻结适配器从真实观察生成规范状态：

- 原生文本和 OCR 结果直接进入 literal/categorical 字段；
- GUI 结构进入元素、属性和关系字段；
- 视觉细节进入可选 VQ token；
- Qwen3.5 等多模态模型可以执行结构化 JSON 提取或辅助视觉理解，但不是记忆架构的必要组件。若使用生成式提取，必须固定模型与 prompt，采用确定性解码，执行 JSON Schema 校验，并缓存最终规范状态。

必须单独报告：
$$
\
\text{Store-All-Canonical-State Quality}
\
$$

若保存全部规范状态仍无法达到 Store-All-Raw 的任务质量，则瓶颈位于感知适配器，而非 ResidualMem。

---

## 5. Segment 内世界模型

### 5.1 结构

首版世界模型使用小型 GRU；因果 Transformer 仅作为消融。

在 Segment 起点 \(r\)：
$$
\
h_r=0,
\qquad
\tilde x_r=x_r,
\
$$

其中 \(x_r\) 是完整、独立编码的锚点。

对 \(t>r\)：
$$
\
h_t
=
F_\theta
\left(
 h_{t-1},
 \tilde x_{t-1},
 u_{t-1}
\right),
\
$$

$$
\
p_t^j
=
p_\theta(x_t^j\mid h_t).
\
$$

这里的 \(\tilde x_{t-1}\) 必须是解码端能够恢复的状态。

### 5.2 默认状态

每个字段由确定性规则产生默认值：
$$
\
\bar x_t^j=g_j(p_t^j,\tilde x_{t-1}^j).
\
$$

推荐规则：

- Boolean/Categorical/VQ：argmax；
- Integer：离散中位数或预测整数；
- Literal：沿用上一值，无法沿用时输出 `UNKNOWN`。

### 5.3 模型输出

模型至少输出：

- 默认值；
- 每个可枚举字段的概率分布；
- literal 字段的“保持/改变”概率；
- 可选的置信度。

概率主要用于训练、校准、surprise 分析和可选熵编码。主残差格式不依赖浮点概率表。

### 5.4 训练目标
$$
\
\mathcal L_{\mathrm{WM}}
=
\sum_t\sum_j
\ell_j(p_t^j,x_t^j),
\
$$

其中：

- categorical/bool/VQ 使用交叉熵；
- integer 使用离散交叉熵或 Huber；
- literal 使用保持/改变分类损失。

训练分两步：

1. teacher forcing：使用真实上一状态；
2. closed-loop fine-tuning：使用不同码率下生成的 \(\tilde x_{t-1}\)。

训练完成后冻结模型和状态 schema，再生成任何持久码流。

---

## 6. 类型化残差编码

### 6.1 字段失真

若不发送字段 \(j\)，当前损失为：
$$
\
\Delta D_t^j
=
w_j d_j(x_t^j,\bar x_t^j).
\
$$

推荐失真：

- Boolean/Categorical/VQ：$$\(\mathbf 1[x\ne \bar x]\)$$；
- Integer：\(|x-\bar x|/\text{scale}_j\)；
- Literal：相等为 0，不等时由策略决定；
- `must` 字段：不相等时强制发送，不通过有限权重近似。

### 6.2 字段残差字节

用 \(R_t^j\) 表示字段修正的**实际可预估 bit 数**。

#### Boolean

若预测错误，字段 mask 已经说明真实值是相反值，因此可以不写 payload。

#### Categorical/VQ

将默认类别从候选集合中移除，编码真实类别在剩余 \(C-1\) 个类别中的编号：
$$
\
R_t^j=\left\lceil\log_2(C-1)\right\rceil.
\
$$

#### Integer

编码差值
$$
\
\delta_t^j=x_t^j-\bar x_t^j
\
$$

并使用 ZigZag VarInt。其 bit 数等于 VarInt 字节数乘以 8，可在写入前精确计算。

#### Literal

编码已有字典 ID，或编码新字符串的长度与 UTF-8 字节。Segment 字典只由锚点中的 literal 和**已经实际发送**的新字符串构成；编码端不得把被省略的真实字符串加入字典，否则两端会失去同步。

### 6.3 为什么按“残差步骤”存储

若每一步都写一个完整字段 mask，即使没有任何残差，也会产生固定线性开销。新版只记录真正包含残差的步骤。

一个 Segment 存：

```text
num_residual_steps
(delta_step, field_mask, payloads) * K
```

没有记录的步骤表示所有字段均使用世界模型默认值。

### 6.4 字段选择规则

每个残差步骤有共享开销 \(H_t\)：

- 与上一残差步骤的时间差 VarInt；
- 固定 \(M\) bit 字段 mask；
- 记录长度或边界开销。

对非强制字段计算净收益：
$$
\
g_t^j
=
\Delta D_t^j-\lambda R_t^j.
\
$$

令
$$
\
S_t^+
=
\{j:g_t^j>0\},
\
$$

并将所有预测错误的 `must` 字段加入集合。发送规则为：

1. 若存在强制字段，激活该步骤；
2. 否则仅当
$$
\
\sum_{j\in S_t^+}g_t^j
>
\lambda H_t
\
$$

时激活该步骤；
3. 激活后发送所有强制字段和所有 \(g_t^j>0\) 的字段。

上式假设字段失真可加。它不是全局最优率失真求解，但规则透明、确定、无需训练，而且可以逐步替换为更复杂的选择器。

### 6.5 精确模式与有损模式

#### Exact-State 模式

所有需要精确恢复的字段设为 `must`。只要默认值与真实值不同，就发送修正。解码后规范状态逐字段完全一致。

#### Rate-Distortion 模式

部分字段设为 `weighted`。系统允许省略低价值误差，通过 \(\lambda\) 控制质量与存储。

### 6.6 在线码率控制

对未知长度轨迹，不需要事后逐条淘汰。可按 Segment 调整 \(\lambda\)：
$$
\
\lambda_{k+1}
=
\max
\left(
0,
\lambda_k+\eta
\left(
\frac{B_k}{L_k}-\bar R
\right)
\right),
\
$$

其中：

- \(B_k\) 是第 \(k\) 个 Segment 的实际 bit；
- \(L_k\) 是步数；
- \(\bar R\) 是目标 bit/step。

若必须保留的精确信息本身超过预算，则“严格保真”和“严格硬上限”不能同时满足。系统必须明确选择：允许超预算、删除完整 Segment，或放弃部分 exact 约束。

---

## 7. Segment 存储格式

### 7.1 全局文件头

```text
magic               8 bytes
format_version      uint16
schema_hash         32 bytes
world_model_hash    32 bytes
adapter_hash        32 bytes
segment_length      uint16
flags               uint32
segment_count       uint32
index_offset        uint64
```

模型参数不重复写入每个 Segment。实际部署必须能按 hash 获得对应冻结模型。论文另行报告模型参数大小及摊销成本。

### 7.2 Segment

```text
SegmentHeader
    segment_id
    start_step
    num_steps
    anchor_bytes
    action_bytes
    residual_bytes
    actions_are_external
    final_state_hash
    crc32

AnchorPayload
ActionPayload
ResidualPayload
```

### 7.3 锚点

锚点是当前 Segment 第一步的完整规范状态。使用确定性类型编码：

- Boolean：bitset；
- Categorical/VQ：定宽整数；
- Integer：ZigZag VarInt；
- Literal：长度 + UTF-8；
- Optional：presence bit。

所有方法必须使用相同锚点协议。可以对完整 Segment 再统一使用相同版本的通用压缩器，但不能只给 ResidualMem 使用额外压缩。

### 7.4 动作流

动作若可由环境日志免费重放，则在 Conditional Observation Rate 中不计费；若不能，则写入 ActionPayload 并计入 Total Episodic Rate。

动作采用固定 ID、参数 VarInt 和时间差 VarInt。不得把自然语言动作描述当作免费 metadata。

### 7.5 不可变性

Segment 一旦写入即不可局部修改。

原因是后续默认状态依赖此前重建状态。删除中间修正可能改变整个后缀。因此：

- 可以删除整个 Segment；
- 可以从修改点开始重新编码整个后缀；
- 不可以直接删除一条中间 residual 而保留原后缀码流。

---

## 8. 编码与解码算法

### 8.1 数据结构

```python
@dataclass
class Prediction:
    default_state: CanonicalState
    distributions: dict[str, object]

@dataclass
class ResidualRecord:
    step_delta: int
    field_mask: bytes
    payloads: list[bytes]

@dataclass
class Segment:
    header: SegmentHeader
    anchor: bytes
    actions: bytes
    residuals: bytes
```

### 8.2 编码伪代码

```python
def encode_segment(
    observations, actions, adapter, wm, schema, lam, actions_are_external=False
):
    targets = [adapter.encode(obs) for obs in observations]

    anchor = targets[0]
    anchor_bytes = encode_full_state(anchor, schema)

    wm.reset()                       # h = 0
    reconstructed = anchor
    last_record_step = 0
    records = []

    for local_step in range(1, len(targets)):
        action = actions[local_step - 1]
        # predict_next 内部执行 h <- F(h, reconstructed, action)
        prediction = wm.predict_next(reconstructed, action)
        default = prediction.default_state
        target = targets[local_step]

        mandatory = []
        profitable = []

        for j, spec in enumerate(schema.fields):
            if target[j] == default[j]:
                continue

            payload_bits = residual_bit_cost(
                spec=spec,
                actual=target[j],
                default=default[j],
            )

            if spec.policy == "must":
                mandatory.append(j)
                continue

            distortion_gain = spec.weight * field_distortion(
                spec=spec,
                actual=target[j],
                default=default[j],
            )
            net_gain = distortion_gain - lam * payload_bits

            if net_gain > 0:
                profitable.append((j, net_gain))

        selected = set(mandatory)
        selected.update(j for j, _ in profitable)

        if selected:
            shared_bits = residual_step_overhead_bits(
                step_delta=local_step - last_record_step,
                num_fields=len(schema.fields),
            )
            optional_gain = sum(g for _, g in profitable)

            if mandatory or optional_gain > lam * shared_bits:
                record = encode_residual_record(
                    step_delta=local_step - last_record_step,
                    selected=selected,
                    target=target,
                    default=default,
                    schema=schema,
                )
                records.append(record)
                reconstructed = apply_record(default, record, schema)
                last_record_step = local_step
            else:
                reconstructed = default
        else:
            reconstructed = default

    action_bytes = b"" if actions_are_external else encode_actions(actions)

    segment = pack_segment(
        anchor=anchor_bytes,
        actions=action_bytes,
        actions_are_external=actions_are_external,
        records=records,
        final_state_hash=hash_state(reconstructed),
    )
    return segment
```

### 8.3 解码伪代码

```python
def decode_segment(segment, wm, schema, external_actions=None):
    verify_crc(segment)

    anchor = decode_full_state(segment.anchor, schema)
    if segment.header.actions_are_external:
        if external_actions is None:
            raise DecodeError("external actions are required")
        actions = external_actions
    else:
        actions = decode_actions(segment.actions)
    records = iter(decode_records(segment.residuals, schema))

    wm.reset()
    reconstructed = anchor
    states = [anchor]

    next_record = next(records, None)
    record_step = next_record.step_delta if next_record else None

    for local_step in range(1, segment.header.num_steps):
        action = actions[local_step - 1]
        # 与编码端完全相同地推进隐藏状态。
        prediction = wm.predict_next(reconstructed, action)
        current = prediction.default_state

        if record_step == local_step:
            current = apply_record(current, next_record, schema)
            next_record = next(records, None)
            if next_record is not None:
                record_step += next_record.step_delta

        reconstructed = current
        states.append(reconstructed)

    if hash_state(reconstructed) != segment.header.final_state_hash:
        raise DecodeError("model/runtime/state mismatch")

    return states
```

### 8.4 编解码端的唯一差异

编码端额外看到真实 \(x_t\)，仅用于决定和构造当前 residual。除此以外，两端必须执行完全相同的：

- Segment 重置；
- 世界模型前向；
- 默认值规则；
- residual 应用；
- 隐状态更新。

---

## 9. 任务相关性：只保留最简单的版本

### 9.1 静态字段权重

主方法不训练独立 utility head。任务相关性通过字段策略与权重表示：

\[
w_j=1
\]

对应任务无关状态保真；也可以根据训练集问题的证据字段频率设置：

\[
w_j
=
\epsilon+
\Pr_{q\sim\mathcal D_{\mathrm{train}}}
\left(j\text{ 被答案依赖}\right).
\]

测试时权重冻结，不能读取测试问题再重写历史记忆。

### 9.2 可选的小型任务探针

若静态权重不足，可训练一个冻结的小型 probe，在训练阶段估计字段替换对任务标签的影响。该 probe 只作为失真函数，不参与 bitstream 解码。

不建议首版使用：

- 生成式 QA Reader 反传；
- 每个字段的完整未来 rollout；
- Shapley 价值；
- 在线 value head 淘汰。

### 9.3 开放世界设置

当未来问题完全未知时，任务权重没有可靠依据。此时应采用：

- 所有关键结构字段权重相同；
- exact facts 设为 `must`；
- 单独保留少量通用视觉 token；
- 明确承认无法保证回答任意未来问题。

---

## 10. 检索与查询

### 10.1 压缩和检索分开验证

第一阶段使用两条评测轨道。

#### Codec Track

给定正确 Segment 或时间范围，直接解码并评测状态和任务质量。该轨道隔离残差编码本身，不声称端到端检索能力。

#### End-to-End Track

使用真实索引定位 Segment，索引字节与查询延迟全部计入。

### 10.2 检索单位

检索单位只能是独立 Segment，不能是单个 residual。

最小索引条目：

```python
@dataclass
class SegmentIndexEntry:
    segment_id: int
    start_step: int
    end_step: int
    entity_ids: list[int]
    field_type_mask: int
    embedding: bytes | None       # 可选 int8 向量
```

### 10.3 查询流程

1. 根据时间、实体或量化 embedding 检索 top-k Segment；
2. 从每个 Segment 锚点开始完整回放；
3. 把解码状态渲染为结构化证据；
4. 交给固定 Reader；
5. 报告检索耗时、回放步数和实际上下文字节。

若索引中保存 embedding、摘要或实体名，它们都属于持久记忆，必须计费。

---

## 11. 训练流程

### 阶段 A：确定状态 schema

- 定义字段、类型、精确性和失真；
- 实现确定性序列化；
- 验证 Store-All-Canonical-State 上限；
- 冻结 schema 和 adapter 版本。

### 阶段 B：训练世界模型

- 使用训练轨迹；
- teacher forcing 预训练；
- 用多个码率下的重建状态做 closed-loop fine-tuning；
- 在 held-out 轨迹上报告 NLL、默认值准确率和多步漂移；
- 冻结模型。

### 阶段 C：实现无学习残差 codec

- 实现完整锚点；
- 实现 residual step 记录；
- 实现类型化 payload；
- 实现 CRC 和 final-state hash；
- 扫描多个 \(\lambda\) 得到率失真曲线。

### 阶段 D：加入任务权重

- 先使用静态权重；
- 再做训练问题证据频率权重消融；
- 不改世界模型与 bitstream 格式。

### 阶段 E：加入 Segment 检索

- 先时间/实体索引；
- 再量化 embedding；
- 所有索引字节计费；
- 固定 Reader，不端到端联合训练。

---

## 12. 实验设计

### 12.1 核心假设

**H1：残差优于完整状态。** 在固定规范状态和任务质量下，ResidualMem 比逐步保存完整状态使用更少字节。

**H2：残差优于普通 Delta Log。** 在具有可预测动作后果或状态变化的环境中，世界模型默认状态比“上一状态保持不变”产生更少修正。

**H3：世界模型质量与字节相关。** 在表示、分段、payload codec 和任务权重固定时，更低 held-out NLL 和更高默认值准确率通常对应更低实际残差率。

**H4：任务权重过滤无关误差。** 相同字节预算下，任务加权失真比统一字段权重获得更高任务质量。

### 12.2 Benchmark 优先级

#### 第一优先：EMemBench

用于机制实验。其 Jericho 和 Crafter 轨迹可以生成程序化、可验证的记忆问题，适合固定行为轨迹、控制世界模型强度并分析状态字段。

#### 第二优先：LongMemEval-V2

用于端到端 Web/企业轨迹记忆。ResidualMem 后端返回解码后的紧凑证据，并与固定 Reader 对接。重点报告准确率、查询延迟和持久字节。

#### 第三优先：WorldMemArena

作为多模态系统级补充。仅在 Codec Track 和 LongMemEval-V2 接口稳定后接入。

首篇论文不同时扩展到全部 GUI、视频、具身和多会话 benchmark。

### 12.3 固定轨迹协议

为避免动作 side channel 和行为差异：

- 所有记忆方法使用同一批离线轨迹；
- 轨迹由与记忆方法无关的固定行为策略生成；
- 训练、验证和测试环境种子隔离；
- 在线 Agent 成绩作为后续实验。

### 12.4 世界模型强度实验

只改变：

- 训练数据比例；
- GRU hidden size；
- 是否使用动作；
- teacher-forcing 与 closed-loop 训练；
- GRU 与因果 Transformer。

固定：

- 状态 adapter；
- schema；
- Segment 长度；
- payload codec；
- 字段权重；
- Reader；
- 轨迹和索引协议。

“更强”定义为 held-out：

- 更低字段 NLL；
- 更高默认值准确率；
- 更小整数预测误差；
- 不更差的闭环状态质量。

对每个世界模型扫描完整 \(\lambda\) 曲线，读取固定质量 \(Q\) 下所需实际字节：

\[
B^*(Q,W)
=
\min_B\{B:J(W,M_B)\ge Q\}.
\]

该关系是待验证的经验命题，不预先假定严格单调。

---

## 13. Baseline

### 13.1 同状态空间主 Baseline

1. **Store-All-State**：每一步保存完整规范状态；
2. **Independent-State Codec**：每一步独立类型编码，不用时间信息；
3. **Plain Delta Log**：只保存相对上一真实状态改变的字段；
4. **Persistence Model**：默认所有字段保持上一重建值；
5. **Temporal Model**：预测下一状态，但不使用动作；
6. **Surprise-Gated Full State**：高 surprise 时保存完整状态；
7. **ResidualMem-Exact**：发送所有默认预测错误的关键字段；
8. **ResidualMem-RD**：按字段率失真选择修正；
9. **WM-Only**：不保存任何修正。

### 13.2 系统级 Baseline

- 原始轨迹 RAG；
- 摘要记忆；
- 长上下文；
- benchmark 官方 memory backend。

系统级方法与 latent/结构化状态方法不应只在“原始字节比”上混成一个结论。应分别报告：

- 同规范状态空间的严格压缩比较；
- 端到端任务质量、存储和延迟比较。

### 13.3 关键消融

- 无动作；
- 无世界模型，仅 persistence；
- Segment 长度 16/32/64/128；
- exact 与 lossy；
- 统一权重与任务权重；
- 固定字段 mask 与稀疏字段 ID；
- GRU 与因果 Transformer；
- 静态 payload 与可选条件熵编码；
- oracle state 与 observation-derived state。

---

## 14. 指标与 bit 计费

### 14.1 状态和任务指标

- exact 字段准确率；
- weighted 字段失真；
- 多步闭环状态准确率；
- QA Accuracy；
- Agent Task Success；
- 不可回答问题的拒答准确率；
- 查询回放步数；
- 查询延迟。

### 14.2 实际存储

\[
\operatorname{Bits}(M)
=
B_{\mathrm{global}}
+
\sum_k
\left(
B_{\mathrm{header}}^k
+B_{\mathrm{anchor}}^k
+B_{\mathrm{action}}^k
+B_{\mathrm{residual}}^k
+B_{\mathrm{checksum}}^k
\right)
+B_{\mathrm{index}}.
\]

报告：

- bytes/step；
- bytes/episode；
- bytes/answered-query；
- Memory Rate relative to fixed raw serialization；
- Memory Saving@Fixed Quality；
- anchor、action、index、payload 的分项占比。

### 14.3 两种动作口径

**Conditional Observation Rate**：动作由环境免费提供，只计观察记忆。

**Total Episodic Rate**：动作、时间、锚点、索引和残差全部计费。

### 14.4 模型参数成本

外部记忆节省不等于系统总描述长度下降。比较不同世界模型时另报：

\[
\operatorname{TotalCost}(N)
=
\operatorname{Bits}(W)
+
\sum_{n=1}^{N}\operatorname{Bits}(M_n).
\]

并展示模型参数在不同轨迹数或用户数 \(N\) 下的摊销结果。

---

## 15. 正确性测试

在跑 benchmark 前，必须通过以下测试。

### 15.1 Exact-State round trip

在所有字段设为 `must` 时：

```text
encoded target states == decoded states
```

逐字段完全一致。

### 15.2 编码端与解码端状态一致

即使在有损模式：

```text
encoder reconstructed state == decoder reconstructed state
```

真实目标状态可以不同，但两端重建状态必须相同。

### 15.3 因果性

在计算 \(p_t\) 后再随机改变 \(o_t\)，当前 prior 和默认状态不得改变。

### 15.4 无隐藏状态泄漏

把某个未发送字段替换为另一真实值，只要 residual 不变，后续解码轨迹不得访问该真实值。

### 15.5 Segment 独立性

- 单独解码任意 Segment 应成功；
- 删除前一个 Segment 不应影响后一个 Segment；
- 从文件中随机定位 Segment 后结果应与顺序读取一致。

### 15.6 版本检查

修改 schema hash、model hash 或 adapter hash 时，解码器必须拒绝。

### 15.7 损坏检测

翻转任一 payload 字节，CRC 或 final-state hash 必须检测异常。

### 15.8 bit 账单一致

```text
reported bits == actual file size * 8
```

分项统计之和必须等于文件总大小，不允许用理论 NLL 替代真实字节。

---

## 16. 推荐的首版实现配置

| 项目 | 默认值 |
|---|---|
| 领域 | 先 Crafter，再 Jericho/Web |
| 状态 schema | 每领域固定或有界字段，\(M\le128\) |
| Segment 长度 | 64 步，消融 16/32/128 |
| 世界模型 | 2 层 GRU，hidden 512 |
| 世界模型输入 | 上一重建状态 + 动作 + 时间差 |
| 锚点 | 完整规范状态，确定性类型编码 |
| residual step | delta step + 固定字段 mask + payload |
| Boolean payload | 0 bit，mask 表示翻转 |
| Integer payload | ZigZag VarInt delta |
| Literal payload | Segment 字典 + UTF-8 |
| 视觉细节 | 可选 VQ token，不作为首版必要项 |
| 写入策略 | 解析式率失真规则，无 learned gate |
| 预算控制 | 验证集选 \(\lambda\)，可选 Segment 级 dual update |
| 检索 | 先时间/实体索引，后 int8 embedding |
| Reader | 固定，不与 codec 联合训练 |

这些数值是原型起点，不是论文结论。最终配置必须由验证集确定。

---

## 17. 项目实施顺序

### Phase 1：最小正确闭环

- 单一结构化环境；
- 固定 schema；
- persistence predictor；
- exact residual；
- Segment 文件格式；
- 完整单元测试。

### Phase 2：世界模型收益

- 训练 GRU；
- 与 Delta Log、Persistence 比较；
- 扫描世界模型数据量和规模；
- 绘制 WM Quality vs. Actual Bytes。

### Phase 3：有损率失真

- 加入 weighted 字段；
- 扫描 \(\lambda\)；
- 输出 Task Quality vs. Actual Bytes；
- 加入静态任务权重消融。

### Phase 4：端到端多模态适配

- 冻结观察适配器；
- 评测 Store-All-Canonical-State 上限；
- 接入 LongMemEval-V2 或 WorldMemArena；
- 加入计费 Segment 索引。

### Phase 5：可选优化

只有在前四阶段成立后再考虑：

- 条件 range/rANS 编码；
- 因果 Transformer；
- 稀疏字段 ID；
- 动态 Segment 边界；
- 小型任务探针；
- 在线模型巩固和旧码流重编码。

---

## 18. 主要风险与诚实边界

### 18.1 规范状态可能丢失原始信息

ResidualMem 只能保留 adapter 输出的信息。必须用 Store-All-Canonical-State 测量上限，并与 Store-All-Raw 区分。

### 18.2 结构化状态可能限制通用性

固定 schema 更易实现和验证，但对开放网页和任意图像覆盖有限。首篇论文应优先证明机制，而不是同时解决通用感知。

### 18.3 世界模型可能比 Delta Log 更差

在以外生事实为主的场景中，动作条件模型不一定有优势。应按任务类型分组报告：

- 状态持久性；
- 动作条件变化；
- 外生更新；
- 随机结果；
- 纯事实披露。

### 18.4 局部率失真不保证全局最优

字段之间和时间之间可能存在互补关系。当前规则是透明基线，不应宣称全局最优。若结果显示明显交互，再增加一到三步 lookahead，而不是直接引入完整长期 utility 系统。

### 18.5 固定 Segment 有锚点开销

Segment 越短，随机访问越快但锚点越多；越长，存储更省但回放更慢。该权衡必须真实报告。

### 18.6 浮点模型确定性

即使不使用算术编码，编码端和解码端仍需得到相同默认值。官方实验应固定模型、软件和确定性运行配置，并用 final-state hash 检测漂移。跨平台部署可进一步使用量化 CPU/ONNX 模型。

### 18.7 硬预算与精确保真存在冲突

外生新字符串或大量异常事件可能使 exact residual 超出预算。系统不能同时承诺“所有关键事实无损”和“永不超预算”，必须在协议中明确优先级。

---

## 19. 预期贡献

### 19.1 可执行的世界模型互补记忆定义

把长期记忆定义为相对确定性默认状态的显式字段修正，而不是抽象的 posterior-prior 差异。

### 19.2 完整可回放的 Segment codec

给出锚点、动作、残差、版本和校验均可实际序列化的格式，保证编解码端闭环一致。

### 19.3 无需学习 gate 的率失真写入

使用字段失真、实际 payload 成本和共享记录开销直接决定写入，避免复杂效用标签和联合训练。

### 19.4 世界模型质量与外部记忆的受控实验

在固定状态空间和存储协议下，系统测量世界模型质量提高是否真的减少固定任务质量所需的外部字节。

---

## 20. 核心表述

### 20.1 论文主张

> 智能体不应重复保存世界模型已经能可靠恢复的状态；它只需要保存使默认预测变成任务所需现实状态的修正。

### 20.2 核心公式

状态预测：

\[
\bar x_t
=
G_\theta(\tilde x_{<t},u_{<t}).
\]

字段失真收益：

\[
\Delta D_t^j
=
w_jd_j(x_t^j,\bar x_t^j).
\]

字段净收益：

\[
g_t^j
=
\Delta D_t^j-\lambda R_t^j.
\]

重建状态：

\[
\tilde x_t^j
=
\begin{cases}
 x_t^j,& j\text{ 被发送},\\
 \bar x_t^j,& j\text{ 被省略}.
\end{cases}
\]

下一步仅使用重建状态：

\[
h_{t+1}
=
F_\theta(h_t,\tilde x_t,u_t).
\]

### 20.3 标题候选

首选：

> **ResidualMem: Predictive Residual Memory for Long-Horizon Agents**

更强调世界模型：

> **Remember the Exceptions: World-Model-Complementary Memory via Typed Residuals**

更强调存储：

> **ResidualMem: A Replayable Segment Codec for World-Model-Complementary Memory**

---

## 21. 参考资料

### 预测式与选择性记忆

- B'MOJO: Hybrid State Space Realizations of Foundation Models with Eidetic and Fading Memory.  
  https://arxiv.org/abs/2407.06324
- Worth Remembering: Surprise-Gated Robot Episodic Memory.  
  https://arxiv.org/abs/2606.03787
- D-MEM: Dopamine-Gated Agentic Memory via Reward Prediction Error Routing.  
  https://arxiv.org/abs/2603.14597
- Curious Replay for Model-Based Adaptation.  
  https://proceedings.mlr.press/v202/kauvar23a.html

### 模型与实现

- DreamerV3.  
  https://github.com/danijar/dreamerv3
- Qwen3.5-9B-Base（可选观察适配器）。  
  https://huggingface.co/Qwen/Qwen3.5-9B-Base
- CompressAI（可选条件熵编码参考）。  
  https://github.com/InterDigitalInc/CompressAI

### Benchmark

- EMemBench.  
  https://github.com/InternLM/EMemBench
- LongMemEval-V2.  
  https://github.com/xiaowu0162/LongMemEval-V2
- WorldMemArena.  
  https://github.com/UCSB-AI/WorldMemArena

---

## 附录 A：代码仓库建议结构

```text
residualmem/
  schemas/
    base.py
    crafter.py
    jericho.py
    web.py

  adapters/
    base.py
    oracle.py
    multimodal.py

  world_model/
    gru.py
    transformer.py
    train.py
    evaluate.py

  codec/
    varint.py
    typed_fields.py
    residual_record.py
    segment.py
    file_format.py

  retrieval/
    temporal.py
    entity.py
    embedding.py

  evaluation/
    bit_accounting.py
    state_metrics.py
    task_metrics.py

  tests/
    test_roundtrip_exact.py
    test_encoder_decoder_sync.py
    test_segment_random_access.py
    test_no_hidden_leak.py
    test_version_mismatch.py
    test_corruption_detection.py
    test_bit_accounting.py
```

## 附录 B：首个应实现的最小实验

为了最快判断研究方向是否成立，首个实验只需要：

1. 一个可访问结构化状态的 Crafter 轨迹集；
2. 20 至 80 个固定字段；
3. Segment 长度 64；
4. Persistence、Delta Log 和 GRU 三种默认预测器；
5. 所有关键字段设为 exact；
6. 类型化残差编码；
7. 实际文件字节统计。

首张关键图应为：

```text
Held-out Default-State Error  vs.  Actual Residual Bytes / Step
```

第二张图应为：

```text
Task / State Quality  vs.  Actual Total Memory Bytes
```

若 GRU 不能稳定优于 Delta Log，则不应继续增加 Qwen latent、utility head 或复杂检索；应先检查状态定义、动作条件和数据是否真正包含可预测动力学。
