# 260907-FlexTP：异构张量并行 Prefill 调度器逻辑说明

本文独立描述一个面向共享 GPU 集群的异构张量并行（Tensor Parallelism，TP）
Prefill 调度器。调度器同时管理多个小规模 TP 组和一个大规模 TP 组，在满足
TTFT（Time To First Token）目标的同时，利用请求长度、预测服务时间、GPU 占用
足迹、队列状态和共享 GPU 并发开销选择执行位置。

本文不依赖任何具体代码库、类名或版本背景，可直接作为论文中的系统设计参考。

## 1. 系统模型

### 1.1 计算资源

假设集群中存在两类 Prefill 实例：

- **小 TP 实例**：例如 TP2，每个实例使用较少 GPU；可以部署多个相互独立的实例。
- **大 TP 实例**：例如 TP4，使用更多 GPU；通常只有一个实例。

不同实例可能共享物理 GPU。共享 GPU 的 Prefill 任务可以通过 MPS 或类似机制
并发执行，但并发会产生 slowdown。调度器需要同时考虑：

1. 实例自身已有的请求和 token 负载；
2. 候选实例与其他活跃实例是否共享 GPU；
3. 不同 TP 大小的单请求服务曲线；
4. 大 TP 实例的额外 GPU footprint。

Decode 资源与 Prefill 资源分离。调度器为请求选择一个 Prefill 实例和一个可用
Decode 节点；本文重点描述 Prefill 选择，Decode 节点仅可采用轮询或其他独立策略。

### 1.2 请求状态

每个请求包含以下字段：

```text
request_id       唯一请求标识
input_tokens     输入 token 数
arrival_time     到达时间
slo_ttft         TTFT 目标
deadline         arrival_time + slo_ttft
enqueue_order    到达顺序，用于 FIFO tie-break
```

请求经历以下状态：

```text
pending → admitted → accepted/queued → running → finished
                                      ↘ failed
```

- `pending`：等待调度的请求。
- `admitted`：调度器已经为请求预留 Prefill 实例，并返回节点选择结果。
- `accepted/queued`：Prefill worker 已确认请求所在 bundle。
- `running`：worker 正在执行请求。
- `finished` 或 `failed`：请求完成或失败，释放实例上的预留资源。

调度器采用 lease 思路：请求一旦被 admitted，就会占用实例的 admission credit，
直到收到完成、失败、取消或节点失效通知。这样可以避免只依据 worker 当前运行
状态做决策而低估已经发出的隐藏工作量。

## 2. Admission 约束

调度器不会把所有可用实例都视为可接纳。候选实例必须同时满足以下条件：

1. 实例处于 available 状态；
2. 实例上的 accepted bundle 数未超过上限；
3. 实例上的 admitted request 数未超过 inflight 上限；
4. 实例已有请求时，加入当前请求后 token credit 不超过上限；
5. 输入长度规则允许该 TP 类型接纳请求；
6. 如果候选是大 TP 实例，还必须通过 TP4/TP2 服务比率保护或压力放宽规则。

其中 token credit 是调度器对一个实例短期隐藏工作量的上界：

```text
admitted_tokens(instance)
    = sum(input_tokens of all non-released leases on instance)
```

当实例已有 lease 时，只有满足

```text
admitted_tokens(instance) + request.input_tokens
    <= instance_token_credit
```

才能继续向该实例 admission。空闲实例可直接接纳一个请求，即使该请求本身超过
credit；这避免单个超长请求因为 credit 上限而永远不可服务。

## 3. 延迟模型

调度器使用离线测量或配置得到的 TP-specific latency profile。对 TP 大小 `p`，
定义四个参数 `a_p, b_p, c_p, d_p`。给定一个 bundle，其请求输入长度为
`l_1, l_2, ..., l_n`，预测 Prefill 服务时间为：

```text
service(bundle, p) = max(
    a_p * sum(l_i) / p
      + b_p * sum(l_i^2) / p,
    c_p
  ) + d_p
```

含义如下：

- 一次项表示输入 token 总量带来的主要计算成本；
- 二次项表示长序列或 attention 相关的非线性成本；
- `c_p` 表示最小 kernel/launch/同步开销；
- `d_p` 表示固定附加开销；
- `p` 表示 TP 并行度对服务曲线的影响。

单请求 exclusive service 是只有该请求的 bundle 预测：

```text
exclusive_service(input_tokens, p)
    = service([input_tokens], p)
```

调度器按照 bundle token cap 将实例上的 lease 和候选请求打包。若当前请求加入
后需要多个 bundle，则逐个累加 bundle service：

```text
base_elapsed(instance, request)
    = sum(service(bundle_j, instance.tp_size) for every predicted bundle_j)
```

## 4. MPS 重叠模型

如果候选实例与其他活跃实例不共享 GPU，则 slowdown factor 为 1。若两者共享
GPU，则使用配置的 overlap slowdown，记为 `s >= 1`。在有多个重叠实例时，可按
实例 TP 组合使用 profile-specific slowdown；没有专门 profile 时使用默认值。

定义候选实例与所有活跃重叠实例的 TP signature：

```text
signature = sorted(
    candidate.tp_size,
    active_overlapping_instance_1.tp_size,
    ...
)
```

则：

```text
overlap_factor(candidate, active_set)
    = configured_slowdown(candidate.tp_size, signature)
      or default_mps_slowdown
```

预测完成时间还包括 bundle 等待窗口和安全余量：

```text
predicted_finish
    = now
      + base_elapsed * overlap_factor
      + bundle_window
      + prediction_margin
```

如果需要针对新硬件进行整体校准，可引入正数 `latency_scale`：

```text
calibrated_predicted_finish
    = now + (predicted_finish - now) * latency_scale
```

该参数应被视为预测模型校准系数，而不是新的排队规则。实验中必须明确记录它
是否也用于 route price，避免将延迟校准和资源价格校准混为一谈。

## 5. TP 类型选择规则

### 5.1 长请求隔离

设置输入长度阈值 `long_request_threshold`。当：

```text
input_tokens > long_request_threshold
```

请求不再进入最小 TP 实例，只保留大 TP 实例候选。该规则的目的不是声称大 TP
对所有长请求都更快，而是避免超长请求长期占用小 TP 队列，形成长尾阻塞。

等于阈值的请求不属于长请求；比较必须使用严格大于关系，以保证边界行为确定。

### 5.2 服务比率保护

对大 TP 候选，先比较其单请求 exclusive service 和小 TP 的单请求 exclusive
service：

```text
tp_ratio = exclusive_service(input_tokens, large_tp)
            / exclusive_service(input_tokens, small_tp)
```

大 TP 正常可选的条件是：

```text
tp_ratio <= tp4_service_ratio_limit
```

等价地：

```text
exclusive_service(input_tokens, large_tp)
    <= tp4_service_ratio_limit
       * exclusive_service(input_tokens, small_tp)
```

该保护避免因为大 TP 的轻微单请求速度优势，就把大量短请求发送到 footprint
更大的实例，导致小 TP 总容量被浪费或共享 GPU 并发增加。

服务比率是逐请求计算的连续规则，不是固定 token bucket。它可以随着请求长度和
测量到的服务曲线自然变化。

### 5.3 压力触发的 TP4 spill

服务比率不满足时，大 TP 默认不可选；但在小 TP 队列形成压力时，可允许请求
spill 到大 TP。

对每个小 TP 实例计算 credit utilization：

```text
credit_utilization(instance)
    = min(
        1,
        admitted_tokens(instance) / max(1, instance_token_credit)
      )
```

整个小 TP 池的压力定义为最忙实例的 utilization：

```text
tp2_pressure
    = max(credit_utilization(instance) for every small_tp instance)
```

当：

```text
tp2_pressure >= tp4_pressure_threshold
```

即使服务比率不满足，也允许大 TP 候选进入后续比较。使用最大值表示调度器重点
保护最忙小 TP lane 的 tail，而不是计算整个小 TP 池的平均利用率或总利用率。

### 5.4 逾期请求救援

对于服务比率不满足且压力未达到阈值的请求，如果：

```text
current_time >= deadline
且不存在任何可接纳的小 TP 实例
```

则允许大 TP 作为 best-effort rescue。这个条件是“已逾期救援”，不是提前的
near-deadline reservation；实现其他调度器时若需要 slack-aware 行为，应显式
使用 `deadline - current_time` 的剩余 slack，而不能复用此条件。

### 5.5 规则优先级

对一个大 TP 候选，规则顺序为：

```text
if only one TP type exists:
    allow
elif candidate is not large TP:
    allow
elif request is long:
    allow
elif service ratio is acceptable:
    allow
elif small-TP pressure reaches threshold:
    allow and mark pressure spill
elif request is overdue and no small TP accepts it:
    allow and mark overdue rescue
else:
    reject candidate
```

长请求规则优先于服务比率规则，因此长请求不会因为大 TP 的 measured ratio 较差
而重新回到小 TP。

## 6. GPU footprint price

候选路由不仅比较预测完成时间，还加入资源足迹价格。设当前可用实例中最小 TP
为 `p_min`，候选 TP 为 `p`，则 footprint：

```text
footprint(p) = max(1, p / p_min)
```

基础 route price 为：

```text
route_price(instance, request)
    = routing_cost_weight
      * exclusive_service(input_tokens, instance.tp_size)
      * footprint(instance.tp_size)
```

最终路由分数为：

```text
effective_cost(instance, request)
    = calibrated_predicted_finish(instance, request)
      + route_price(instance, request)
```

压力超过阈值时，可以降低大 TP 的价格以提高其 work-conserving 程度：

```text
pressure_discount
    = max(0.25, 1 - 0.75 * tp2_pressure)

large_tp_price_under_pressure
    = base_large_tp_price * pressure_discount
```

即使压力为 1，大 TP price 也不低于原价的 25%。这保留了资源 footprint 的
惩罚，避免所有请求在压力检测瞬间无条件涌向大 TP。

## 7. 全局调度流程

调度器每次收到新请求、实例状态变化或周期性 replan 时运行以下流程：

```text
schedule(now):
    if no decode node or no available prefill instance:
        fail or retain pending according to overload policy
        return

    while pending requests exist:
        choose admission mode from workload preview
        candidates = []

        for request in pending sorted by (deadline, enqueue_order):
            for instance in all available instances:
                if not instance_accepts(instance, request):
                    continue
                if not route_allowed(instance, request):
                    continue

                finish = predict_finish(instance, request, mode)
                price = route_price(instance, request)
                effective = finish + price
                add candidate(request, instance, finish, effective)

        if candidates is empty:
            if overload_policy == reject and some requests are expired:
                reject the earliest expired request
                continue
            return

        choose lexicographically smallest candidate:
            (deadline,
             enqueue_order,
             effective_cost,
             exclusive_service,
             predicted_finish,
             admitted_tokens,
             inflight_count,
             tp_size,
             stable_instance_id)

        create lease
        resolve request's Prefill/Decode selection future
```

候选 tuple 的前两项是 EDF/FIFO 层级。因此调度器的精确定义是：

> 先按 deadline 和到达顺序确定服务层级，再在该层级内选择 predicted finish
> 加 resource price 最小的可行实例。

它不是把 deadline 直接转换成一个连续 penalty 后对所有请求做单一加权最小化。

## 8. Bundle 与 worker 交互

被 admission 的请求不会立刻从资源模型中删除。调度器创建 lease，并在预测中将
其 token 计入实例 credit。多个 lease 可以在 bundle window 内合并，bundle 受：

- token cap；
- request 数/inflight cap；
- instance token credit；
- accepted bundle 数上限

共同约束。

worker 接受 bundle 后，调度器将对应 lease 标记为 accepted。请求完成或失败后：

1. 删除全局 lease 索引；
2. 删除实例 lease；
3. 释放 token credit 和 inflight 名额；
4. 触发下一轮 pending 调度。

节点失效时，应释放该节点上的所有 lease，并让 pending 请求重新参与调度或按
overload policy 失败。请求取消时，也必须同时移除 pending 或 lease，避免 phantom
credit 长期阻塞实例。

## 9. 参数语义

| 参数 | 语义 |
|---|---|
| `slo_ttft` | 从到达时间开始计算的 TTFT 目标，用于形成 deadline |
| `long_request_threshold` | 严格大于该输入长度时强制排除小 TP |
| `instance_token_credit` | 每个实例允许被 lease 预留的 token horizon |
| `max_inflight_per_instance` | 每实例未释放请求数上限 |
| `max_accepted_bundles_per_instance` | 每实例已确认 bundle 数上限 |
| `batch_token_cap` | 单 bundle 的 token 容量上限 |
| `batch_window` | 等待更多请求合并 bundle 的时间窗口 |
| `prediction_margin` | 预测中的安全余量 |
| `mps_overlap_slowdown` | 共享 GPU 并发的默认乘法开销 |
| `routing_cost_weight` | GPU footprint price 的权重 |
| `tp4_service_ratio_limit` | 大 TP 正常进入候选的服务比率上限 |
| `tp4_pressure_threshold` | 最忙小 TP credit utilization 的 spill 阈值 |
| `latency_scale` | 整体完成时间预测的校准系数，必须为正 |
| `overload_policy` | 无可行候选时采用 best-effort 等待或 reject |

参数边界建议：

- `latency_scale > 0`；
- `0 <= tp4_service_ratio_limit <= 1`；
- `0 <= tp4_pressure_threshold <= 1`；
- 所有 token 和 request 上限至少为 1；
- slowdown 至少为 1；小于 1 的值会把“重叠”解释成加速而非开销。

实验或生产配置不应只记录原始命令行参数，还应记录经过校验、截断或默认填充后
真正生效的值。

## 10. 调度器应输出的统计量

为了分析策略是否真正达到目标，建议周期性输出 snapshot，包括：

- 当前实例列表、TP size、GPU placement、available/busy 状态；
- 每个实例的 admitted token、lease 数、accepted bundle 数；
- pending 数、最早 deadline、逾期 pending 数；
- TP2/TP4 admission 请求数和 token 数；
- service-ratio 放行数；
- pressure spill 次数；
- overdue rescue 次数；
- 当前预测模型和 slowdown 参数；
- SLO 达标请求数、超时请求数和失败请求数。

“spill count”应明确是候选放行次数还是最终成功路由的请求数。若一个请求会被
多个相同 TP 候选评估，计数器必须在最终 admission 后增加，才能代表请求级指标。

## 11. 设计目标与适用范围

该调度器适合以下场景：

1. 小 TP 适合大多数短请求，并提供较高 aggregate throughput；
2. 大 TP 对部分长度区间或压力状态有更好的 tail latency；
3. 小 TP 与大 TP 可能共享 GPU，不能把实例视为独立无干扰服务器；
4. 请求有明确的 TTFT deadline；
5. Prefill admission 可以通过 lease 和 token horizon 提前建模。

该调度器不等价于：

- 抢占式调度器：已发送到 worker 的 Prefill 通常不可抢占；
- 完整 queueing-theoretic optimizer：延迟模型是离线 profile 的近似；
- 全局 aggregate utilization optimizer：压力默认看最忙小 TP，而非全池总量；
- Decode 调度器：Decode 容量和 decode-side SLO 需要独立建模；
- 自动学习型调度器：参数和服务曲线需要外部测量、校准和 sweep。

论文中应将该方法描述为一种 **deadline-ordered、admission-constrained、
predicted-finish-plus-footprint-price 的异构 TP 路由器**，并明确压力 spill 是
对默认大 TP 保护规则的条件性放宽。

