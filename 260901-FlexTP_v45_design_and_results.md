# FlexTP V4/V5 调度器设计与实现说明

本文自包含地说明 FlexTP V4 和 V5 的生产实现，包括运行拓扑、继承自 V3
的公共机制、候选请求如何预测和排序、V4 的请求类别隔离、V5 的 MPS 优先
规则、lease 生命周期、配置参数、实现边界以及模拟验证结果。阅读本文不需要先
阅读 V3 文档。

对应实现：

- V4：`lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v4.py`
- V5：`lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v5.py`
- 公共基础：`lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v3.py`
- Prefill bundle：`lightllm/server/httpserver_for_pd_master/pd_bundle.py`

## 1. 目标与结论

目标拓扑在 4 张 Prefill GPU 上常驻三个进程：

| Prefill 实例 | TP | 物理 GPU | 主要请求类别 |
|---|---:|---|---|
| `p01` | 2 | 0,1 | 短请求 |
| `p23` | 2 | 2,3 | 短请求 |
| `p0123` | 4 | 0,1,2,3 | 长请求 |

三个进程通过 CUDA MPS 共存。两个 TP2 彼此不共享 GPU，但都与 TP4 共享
GPU。Decode 使用另外 4 张 GPU，本轮设计和模拟把 Decode 视为无瓶颈。

V4 解决 V3 的首要缺陷：长请求不再进入 TP2，因此不会出现两个 TP2 都被长
请求占用、后续短请求无处执行的情况。

V5 在 V4 的请求类别隔离之上改变重叠策略：默认接受有收益或不会破坏 deadline
的 MPS 重叠，只在候选请求自身无法按时完成，或重叠会破坏受保护工作时让该候选
继续等待。V5 不是一个带全局布尔变量的显式状态机；“重叠”和“临时非重叠”
是每次重规划时对每个候选重新计算出来的结果。

## 2. 运行范围与拓扑发现

V4/V5 只走普通 PD 路径：

- 只接收 `mode == "prefill"` 的 Prefill 节点；
- 只接收 `mode == "decode"` 的 Decode 节点；
- 不把 NIXL 的 `prompt-id-ready` 当成 Prefill 完成事件；
- 不向 NIXL 握手路径发送普通 PD 的 `REQ_BUNDLE`。

调度器通过 `tp_smt_group_id` 或 shared-weight master port 识别 FlexTP 组，通过
`tp_smt_gpu_ids` 识别物理 GPU 重叠关系。若没有显式 GPU 列表，同一主机的实例
会保守地视为互相重叠，避免错误地假设它们可以独立运行。

当前 V4 的最小/最大 TP 是在全部可用 Prefill 实例上计算的，而不是逐组计算。
目标部署只有一个 FlexTP 组，因此这一点没有歧义。若以后在一个 master 下放置
多个异构 FlexTP 组，需要先把类别判断改成 group-local，否则一个组的 TP 上界
可能影响另一个组。

单一 TP 规格暂时可用时，V4/V5 不会等待另一规格注册：所有请求都可以进入该
唯一规格。多个规格可用时才启用严格的最小 TP/最大 TP 隔离。

## 3. 公共调度基础

V4 继承 V3，V5 继承 V4。除本文件后续明确说明的覆盖点外，二者共享下面的
调度、batching 和生命周期逻辑。

### 3.1 Pending、deadline 与 lease

每个请求到达 master 后先生成 `V3Pending`：

```text
seq_len  = max(1, input_token_num)
deadline = arrival_time + slo_ttft
priority = (deadline, enqueue_order)
```

默认启动脚本使用 `slo_ttft = 3s`。`enqueue_order` 用于 deadline 相同时保持
先来先服务。

请求被选中后转成 `V3Lease`。lease 表示 master 已把该实例的容量分配给这个
请求，不表示 GPU 此刻一定正在执行它。只要一个实例持有至少一个 lease，它就在
调度器中被视为 `active/busy`，并参与 MPS slowdown 计算。

实例的默认准入上限是：

- 最多 64 个并发 lease；
- 最多 16384 个 admitted input tokens；
- 空实例允许接收一个本身超过 token credit 的请求，之后在该 lease 释放前不再
  接收导致 credit 超限的新请求。

### 3.2 8192-token batching 近似

调度器按 lease 插入顺序，把同一实例上的请求贪心装入不超过 8192 token 的预测
batch。一个请求不会在 master 预测器中拆成多个 batch；超过 8192 token 的单个
请求会独占一个超限预测 batch。因此 8192 是多请求合批上限，不是单请求拒绝阈值。
真实 worker 仍由本地 Router 完成最终 KV-safe
batch packing 和 8192-token chunked Prefill。

预测模型对一个 batch 使用：

```text
T(batch, tp) = max(a * sum(L) / tp + b * sum(L^2) / tp, c) + d
```

其中时间单位为秒，当前 TP2/TP4 常数为：

| TP | a | b | c | d |
|---:|---:|---:|---:|---:|
| 2 | `2.018352e-4` | `6.048427e-9` | `2.341091e-2` | `6.794939e-2` |
| 4 | `2.655797e-4` | `2.330274e-9` | `5.651746e-2` | `3.767872e-2` |

这些常数是调度模型参数，不是运行时在线测量结果。

### 3.3 MPS slowdown

对目标实例 `i`，调度器先根据物理 GPU 交集构造当前活动模式，再计算：

```text
effective_time_i = predicted_batch_time_i * slowdown(i, active_mode)
```

测试或模拟器可以按 `(target_tp, sorted_active_tp_signature)` 提供 slowdown 覆盖值，
例如 1.6 或 2.0。没有覆盖值时，生产默认值是该实例任意一张 GPU 上的最大活跃
进程数。目标拓扑中 TP2 与 TP4 同时 active 时默认得到 2.0。

注意，这里依据的是 lease 推导出的 active mode，不是 CUDA MPS 的实时 kernel
occupancy。worker 上报的 queued/running requests 和 token load 当前用于状态记录
与观测，尚未直接进入候选打分公式。

### 3.4 候选完成时间

调度器对每个 `pending × 允许的 instance` 构造候选。先计算不加入新请求时所有
现有 lease 的 baseline 完成时间，再把新请求追加到目标实例并重新计算整个活动
模式下的完成时间。

新请求的预测完成时刻为：

```text
predicted_finish = now
                 + candidate_relative_finish
                 + batch_window_s
                 + prediction_margin_s
```

默认 `batch_window_s = 20ms`，`prediction_margin_s = 80ms`。

候选必须同时检查两个条件：

1. `candidate_meets_deadline`：新请求的 `predicted_finish <= deadline`；
2. `protects_existing`：加入新请求后，不破坏所有现有 lease 的允许完成时间。

对现有 lease，允许完成时间定义为：

```text
allowed = max(lease.deadline - prediction_margin, baseline_finish)
```

因此，如果现有 lease 原本能按时完成，新候选不能把它推迟到保守 deadline 之后；
如果它在 baseline 中已经迟到，新候选至少不能让它比 baseline 更迟。

候选还记录：

- `exclusive_work`：该请求单独运行时的预测时间；
- `incremental_delay`：它给全部现有 lease 增加的预测延迟总和；
- `predicted_finish`：包含 bundle window 和安全余量的绝对完成时刻。

### 3.5 Future reservation

一个请求可能当前不能安全准入，但在冲突实例 drain 后能够独立按时完成。公共调度
逻辑会为这种请求计算一个软 reservation：选择预计最早可用的放置位置，并阻止
优先级更低、且与该位置共享 GPU 的候选继续 refill。其原始目的，是避免较老的
waiter 被不断到达的新工作饿死。

这里有一个继承边界：`_can_meet_alone` 和 future-reservation placement 使用全部可用
实例，没有调用 V4 的 long/short TP 过滤。因此 reservation 可能选择一个 V4 实际
不会准入该请求的 TP。V4/V5 又让短候选绕过整个 reservation 列表，具体影响见第
5 节和第 7 节。

### 3.6 可行候选、排序与重规划

公共调度器通常只保留：

```text
candidate_meets_deadline == True
and protects_existing == True
and reservation_allows == True
```

的候选，然后按以下字典序选最小项：

```text
(
  request.deadline,
  request.enqueue_order,
  instance.tp_size,
  exclusive_work * instance.tp_size,
  predicted_finish,
  incremental_delay,
  instance.lease_count,
  instance.node_key,
)
```

前两项意味着跨请求优先采用 EDF，并在 deadline 相同时保持 FIFO。后续项只在同一
请求或同优先级请求的多个放置方案之间决定 TP 和实例。

没有可行候选时，请求留在 pending 队列。以下事件会触发重试：

- 默认每 50ms 的 replan；
- lease 完成或失败；
- worker 注册、移除或拓扑变化。

默认 `overload_policy=best_effort`。请求 deadline 已过且仍无可行方案时，调度器会
选择影响最小的过期候选执行，避免永久挂起，但此时可以绕过正常 deadline 保护。
配置为 `reject` 时则拒绝物理上不能按时完成或已经过期的请求。

### 3.7 Master bundle 与完成生命周期

准入和 websocket 发送是两个阶段：

```text
Pending
  -> admitted lease
  -> per-node bundle queue
  -> REQ_BUNDLE sent
  -> worker ACK / queued_worker
  -> worker Prefill finished or failed
  -> lease released
  -> replan
```

每个 Prefill 节点有独立的 bundle 队列。默认最多等待 20ms；累计到 4096 token、
64 个请求或零等待配置时立即 flush。多个请求合并时，一个发送 bundle 以 8192
token、64 个请求为上限；单个超过 8192 token 的请求仍可独占一个超限 bundle，
不会被 dispatcher 拒绝。bundle 只合并已经被调度器准入到同一具体实例的请求，
本地 Router 仍拥有最终 batch 决定权。

正常完成、显式失败和 worker 断连都会释放对应 lease。lease 同时校验 request id、
node 地址和 instance generation，避免旧 websocket 的完成消息释放新实例上的工作。
一个长输出请求被拆成多个 PD block 时，每个 continuation 都会用“原 prompt + 已生成
history”的新 token 数重新申请独立 lease，不会在 Decode 阶段长期占住 Prefill。

## 4. V3 的问题

V3 会为请求枚举所有 TP 放置方案，再依据 deadline、预测 GPU 成本和完成时间选择
候选。它没有“长请求专用 TP4、短请求专用 TP2”的硬约束。

当 TP2 对某个长请求看起来成本更低或仍能满足 deadline 时，V3 可以把长请求放入
TP2。连续多个长请求可能分别进入两个 TP2：

```text
TP2(0,1): long A
TP2(2,3): long B
TP4(0,1,2,3): idle or waiting
new short requests: blocked behind A/B
```

即使 8192 chunked Prefill 支持 batching，短请求也只能与两个长请求共享有限 batch
和 token credit，导致明显的 head-of-line blocking。V4 首先修复这个放置问题；V5
再处理 MPS 重叠策略。

## 5. V4：严格的请求类别隔离

### 5.1 分类和放置规则

V4 参数 `long_request_threshold` 默认是 4000，判断使用严格大于：

```text
long  = seq_len > 4000
short = seq_len <= 4000
```

当至少两种 TP 规格可用时：

```text
long  -> 最大可用 TP
short -> 最小可用 TP
```

目标拓扑中即：

```text
1..4000 tokens -> 两个 TP2 之一
4001+ tokens   -> TP4
```

V4 在调用公共 `_candidate` 前执行这一过滤，因此错误类别的实例根本不会产生候选，
也不会在后续 best-effort 排序中被选中。只要 TP2 和 TP4 都可用，超过 4000 token
的请求进入 TP2 数量就是 0。

若存在三种以上 TP 规格，当前实现只使用最小和最大 TP，中间 TP 不会被 V4/V5
选择。这是当前二分类算法的明确边界。

### 5.2 短请求绕过全部 future reservation

V4 对短请求的 `_reservation_allows` 直接返回 `True`。含义是：一个仍在 pending、
等待未来执行窗口的更老请求，无论是 long 还是 short，都不会仅凭软 reservation
阻止当前短请求进入 TP2。设计动机主要是防止 long reservation 冻结短请求池，但
当前实现没有按 reservation 所属请求类别做筛选。

这不等于无条件准入。短请求仍必须：

- 满足实例 lease 数和 token credit；
- 满足自己的 deadline；
- 通过公共 `protects_existing` 检查，不能破坏已经准入的 lease。

区别在于 V4 优先保护“已经准入的工作”和“当前短请求”，不让尚未执行请求的
reservation 冻结两个 TP2。

### 5.3 V4 调度步骤

对每次重规划：

1. 按 `(deadline, enqueue_order)` 遍历 pending 请求；
2. 根据 4000-token 阈值只保留最小 TP 或最大 TP；
3. 运行公共 batching、slowdown 和 deadline 预测；
4. 过滤不能满足自身 deadline 或破坏现有 lease 的候选；
5. 对 short 忽略 future reservation，对 long 保留公共 reservation 检查；
6. 按公共字典序选择候选并创建 lease；
7. 重复执行，直到当前没有更多可安全准入的请求。

V4 并不禁止 MPS 重叠。只要 TP2/TP4 同时持有 lease，它们仍会并发运行，并在
预测中承受相应 slowdown。V4 的核心保证是“实例类别隔离”，不是“禁止重叠”。

## 6. V5：MPS-first 的候选级控制

### 6.1 V5 保留什么、改变什么

V5 完整保留 V4 的 4000-token 分类、TP 放置和短请求 reservation 绕过。它只在
V4 生成候选后，重新解释部分 `protects_existing` 结果。

V5 没有 `mps_mode = on/off` 之类的全局状态。每个候选每次评估时都检查是否存在
“另一个 busy 且物理 GPU 与目标实例相交”的实例：

- 没有重叠实例：原样返回 V4/V3 候选；
- 存在重叠实例：进入 V5 的短/长请求决策；
- 不安全的候选留在 pending，后续 replan 再计算。

因此，“临时进入非重叠状态”的准确含义是某个冲突候选暂不准入，并不是暂停整个
系统，也不是向 CUDA MPS 下发模式切换命令。

### 6.2 第一层条件：候选自身必须按时

若 `candidate_meets_deadline == False`，V5 不会覆盖它。正常调度过滤会使其保持
pending，等待当前重叠 lease 完成后重新预测。

即使短请求积压很多或已经紧急，V5 也不会把一个预测上已无法按时完成的候选强制
标成“满足自身 deadline”。best-effort 的过期兜底是公共调度器的另一条路径。

### 6.3 短请求规则

对 `seq_len <= 4000` 的候选，V5 计算：

```text
mode_factor = 候选加入后，与候选实例相关的最大 slowdown
short_backlog = pending 队列中的短请求数量，包含当前候选
remaining_slack = pending.deadline - candidate.predicted_finish
urgent = remaining_slack <= slo_ttft * 0.05
```

默认 `short_overlap_backlog_trigger = 3`，所以 backlog 达到 3 表示当前候选加上
至少两个其他 pending short。

只要以下任意条件成立，V5 就把 `protects_existing` 强制设为 `True`，允许短请求
在 TP4 工作存在时进入 TP2：

| 条件 | 含义 |
|---|---|
| 原候选已经 `protects_existing` | 重叠不会破坏任何现有 lease，直接并发 |
| `mode_factor < 2.0` | 例如 slowdown=1.6，认为重叠具有正的聚合吞吐收益 |
| `short_backlog >= 3` | 短请求已经形成队列，优先恢复短请求吞吐 |
| `urgent == True` | 候选剩余余量不超过整个 SLO 的 5%，避免继续等待 |

这里“强制设为 True”是有意放宽公共保护：如果原候选因为会延迟已有 lease 而得到
`protects_existing=False`，上述后三个条件仍可允许它重叠。实现上这个布尔值汇总了
对所有现有 lease 的检查，因此放宽并不只针对 TP4 lease；在复杂队列中也可能放宽
对目标 TP2 上旧 lease 的保护。

如果 mode factor 大于等于 2.0、短 backlog 小于 3、请求不紧急，并且原候选会
破坏现有 lease，则不覆盖 `protects_existing=False`，该短请求继续 pending。这是
V5 对低并发、等比例资源分享场景设置的临时串行窗口。

### 6.4 长请求规则

对 `seq_len > 4000` 的 TP4 候选，V5 不放宽现有 lease 保护：

- `candidate_meets_deadline=True` 且 `protects_existing=True`：允许 TP4 与 TP2 重叠；
- 候选自身不能按时完成：继续等待；
- 候选会破坏已准入 TP2/TP4 lease：继续等待，形成候选级非重叠窗口。

也就是说，V5 的非对称策略是：新 long 必须保护已准入 short；新 short 在 slowdown
有收益、形成 backlog 或接近 deadline 时，可以牺牲旧工作的部分余量来维持吞吐。

### 6.5 V5 决策流程

```text
请求到达
  |
  v
V4 类别过滤：short->TP2，long->TP4
  |
  v
公共模型生成 candidate
  |
  +-- 没有其他物理重叠的 busy 实例 --> 保留公共结果
  |
  +-- candidate 无法满足自身 deadline --> pending，等待 replan
  |
  +-- long
  |     +-- protects_existing --> 允许 MPS 重叠
  |     +-- 否则 -------------> pending，等待冲突 lease 释放
  |
  +-- short
        +-- 原本保护 existing ----------------------+
        +-- slowdown < 2 ----------------------------+--> 允许 MPS 重叠
        +-- short backlog >= 3 ----------------------+
        +-- remaining slack <= 5% SLO ---------------+
        +-- 以上均不成立 -----------------------------> pending
```

### 6.6 典型场景

场景 A，TP4 正在运行 long，新 short 到达，slowdown=1.6：只要 short 自己能按时
完成，即使它会压缩旧 long 的余量，也允许进入 TP2。系统保持 MPS 重叠。

场景 B，TP4 正在运行 long，新 short 到达，slowdown=2.0，且只有一个非紧急 short：
如果重叠会破坏现有 lease，则 short 等待；如果本来就不破坏，则仍然立即重叠。

场景 C，在场景 B 中 short pending 数达到 3，或某个 short 剩余余量小于等于
`0.05 * slo_ttft`：V5 放宽 existing 保护，恢复 TP2/TP4 并发。

场景 D，两个 TP2 正在处理 short，新 long 准备进入 TP4：只有当预测表明 TP4
加入后不会破坏现有 lease 时才重叠，否则 long 等待 TP2 工作释放。

场景 E，TP4 空闲且没有其他重叠 busy 实例：long 直接按公共 V4 结果进入 TP4，
不经过 V5 的 overlap 分支。

### 6.7 V5 telemetry

`scheduler_snapshot()` 暴露：

| 字段 | 含义 |
|---|---|
| `overlap_candidate_evaluations` | 检测到物理重叠的候选评估次数 |
| `overlap_admissible_evaluations` | V5 判定可接受重叠的评估次数 |
| `serialization_candidate_evaluations` | V5 保留为不安全、应等待的评估次数 |
| `short_overlap_backlog_trigger` | 短请求积压触发值，默认 3 |
| `urgent_slack_ratio` | 紧急余量比例，默认 0.05 |

这些是候选“评估次数”，不是最终 admission 数。一个 pending 请求会被周期性 replan
多次，因此计数可以远大于请求数量。

## 7. V4/V5 的保证与非保证

### 7.1 当前实现能够保证

- 多种 TP 可用时，超过 4000 token 的 long 不会进入最小 TP；
- 目标 TP2/TP4 拓扑下，两个 TP2 不会被 long lease 占用；
- 在正常 deadline 可行路径中，新 long 不会由 V5 主动绕过 `protects_existing`；
- 候选是否重叠会在完成事件和周期性 replan 后重新判断；
- 普通 PD 的 bundle、ACK、完成、失败、断连和 generation 生命周期保持一致。

### 7.2 当前实现不保证

- 不保证所有负载下都严格优于 naive MPS；低负载 SLO 已饱和时通常只能打平；
- 不保证 V5 的等待时间有严格上界，也没有形式化的无饥饿证明；
- V4/V5 的 short 会忽略全部 future reservation，新 short 仍可能继续 refill TP2，
  使等待 TP2 drain 的 older long 或 short 比预期等待更久；
- “临时非重叠”不是全局 drain，其他无冲突或被 V5 放宽的候选仍可准入；
- predictor 没有读取实时 CUDA kernel occupancy，也没有在线校准 MPS slowdown；
- 多 FlexTP group 和三种以上 TP 规格不是当前二分类策略的完整支持目标；
- `_can_meet_alone` 与 future reservation 尚未套用 V4 的类别过滤，`reject` 模式的
  “可独立完成”判断可能把一个实际上被 V4 禁止的 TP 当成可行位置；
- `best_effort` 在请求已过 deadline 后可能绕过常规保护以避免永久挂起。

这些边界不影响“long 不占 TP2”这一 V4 硬约束，但会影响对 V5 公平性和最坏等待
时间的解释。

## 8. 与 V3 和 naive MPS 的算法差异

| 维度 | naive MPS | V3 | V4 | V5 |
|---|---|---|---|---|
| TP 选择 | 4000 阈值，小/大 TP | 枚举 TP，按 deadline/成本选择 | 4000 阈值硬隔离 | 同 V4 |
| 同 TP 多实例 | 最少 inflight tokens | 预测模型和 lease 排序 | 同 V3 | 同 V3 |
| 全局 batching | 无 bounded master bundle | 有 | 有 | 有 |
| 自身 deadline | 不预测 | 必须检查 | 必须检查 | 必须检查 |
| 保护已准入工作 | 无 | 严格保护 | 严格保护 | short 在指定条件下可放宽 |
| future reservation | 无 | 有 | short 绕过 | short 绕过 |
| MPS 重叠 | 无条件自然发生 | 仅预测安全时 | 仅预测安全时 | MPS-first，必要时候选等待 |
| long 占用 TP2 | 阈值正确时不会 | 可能 | 不会 | 不会 |
| 显式全局模式状态 | 无 | 无 | 无 | 无 |

naive MPS 的优点是简单且容易维持并发，缺点是只按长度和 inflight tokens 立即分配，
不检查 deadline、batching 后完成时间或重叠对已有请求的影响。V4/V5 并不是 naive
的接口差异，而是加入了 lease、预测、保护和重规划后的算法差异。

## 9. 配置参数

启动脚本当前使用：

| 参数 | 值 | 作用 |
|---|---:|---|
| `--flex_tp_slo_ttft` | 3s | TTFT deadline 相对到达时间的窗口 |
| `--flex_tp_long_threshold` | 4000 | V4/V5 long/short 分界，严格 `>` |
| `--flex_tp_bundle_window_ms` | 20ms | master 最大合批等待 |
| `--flex_tp_bundle_token_cap` | 8192 | 多请求预测 batch 和发送 bundle 的合批上限 |
| `--flex_tp_bundle_token_trigger` | 4096 | 提前 flush 的累计 token 阈值 |
| `--flex_tp_max_inflight` | 64 | 每实例最大 lease 数和 bundle request cap |
| `--flex_tp_instance_token_credit` | 16384 | 每实例 admitted token credit |
| `--flex_tp_prediction_margin` | 80ms | deadline 预测安全余量 |
| `--flex_tp_overload_policy` | `best_effort` | 过期后执行而不是立即拒绝 |
| `--chunked_prefill_size` | 8192 | worker 实际 chunked Prefill 大小 |

V5 的 `short_overlap_backlog_trigger=3` 和 `urgent_slack_ratio=0.05` 当前是构造函数
默认值，尚未暴露为 CLI 参数。生产 factory 也没有传入 mode slowdown profile，所以
未配置覆盖时使用“共享 GPU 上最大活跃进程数”这一默认估计。

## 10. 模拟验证

`test/benchmark/service/flex_tp_step_sim.py` 直接加载生产 V4/V5 selector，并对
全局调度器、master bundle 和实例侧 8192-token chunked Prefill 做离散事件模拟。
Decode 默认在 Prefill 完成后立即结束，不构成瓶颈。

回归比较对象是生产 `FlexTPNaiveSelector`，阈值同为 4000，所有实例允许自然 MPS
重叠。负载包括：

- 暴露 V3 TP2 阻塞的 phase-shift；
- 5% long 的合成负载，10 req/s；
- ServeGen `mm-image`，8、9、10 req/s；
- MPS overlap slowdown 1.6 和 2.0。

主指标是固定 offered arrival window 内的 on-time input-token goodput。使用固定窗口
是为了避免某个策略仅因更晚排空队列、makespan 不同而获得虚假吞吐优势。回归还
要求：

- V4/V5 的 TTFT SLO attainment 不低于 naive；
- V4/V5 的 TP2 long admission 数为 0；
- V5 的 overlap wall time 至少是 naive 的 75%。

最终 70 个断言全部通过：

| 负载 | slowdown | V4 token goodput | V4 SLO | V5 token goodput | V5 SLO |
|---|---:|---:|---:|---:|---:|
| phase-shift | 1.6 | +23.6% | +0.16 pp | +18.0% | +1.45 pp |
| synthetic 5% long | 1.6 | +17.0% | +1.00 pp | +15.3% | +0.90 pp |
| mm-image, 8 req/s | 1.6 | +1.0% | +0.07 pp | +0.9% | +0.00 pp |
| mm-image, 9 req/s | 1.6 | +2.0% | +3.03 pp | +2.2% | +3.15 pp |
| mm-image, 10 req/s | 1.6 | +3.2% | +4.68 pp | +3.3% | +4.40 pp |
| phase-shift | 2.0 | +22.4% | +0.00 pp | +12.4% | +0.97 pp |
| synthetic 5% long | 2.0 | +22.6% | +1.20 pp | +14.5% | +0.80 pp |
| mm-image, 8 req/s | 2.0 | +12.3% | +12.14 pp | +10.4% | +13.05 pp |
| mm-image, 9 req/s | 2.0 | +6.6% | +12.05 pp | +14.6% | +20.40 pp |
| mm-image, 10 req/s | 2.0 | +35.2% | +31.24 pp | +31.2% | +27.90 pp |

这些是确定性模型结果，不是 H200 实测。额外的低负载 rate-7、slowdown=2.0 探针
出现过约 0.05% token-goodput 回退；V4/V5 也不总能超过一个看完整条 trace 后再
选择最佳固定阈值的 post-hoc oracle。因此，论文或生产结论仍需用 H200 上实际
batch 和 MPS profile 校准预测参数。

## 11. 启动与复现

普通 PD 启动：

```bash
./start-cluster4_70b_p22.4d4_mps_flex_v4.sh
./start-cluster4_70b_p22.4d4_mps_flex_v5.sh
```

两个 wrapper 都调用同一个 cluster4 基础脚本，只分别设置
`FLEX_TP_VERSION=v4/v5`。基础脚本启动两个 TP2 Prefill、一个 TP4 Prefill 和一个
独立 TP4 Decode，并显式传入 `tp_smt_group_id` 与 `tp_smt_gpu_ids`。

运行完整模拟回归：

```bash
python test/benchmark/service/flex_tp_v45_regression.py \
  --slowdowns 1.6,2.0 \
  --output-dir _/flex_tp_paper_analysis/v45-regression
```

生产 selector 契约测试位于：

```text
test/test_flex_tp_selector_v45.py
```
