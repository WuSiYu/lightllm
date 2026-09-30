# 调度算法实现指南

替换现有的 drain 状态机。新设计：所有 TP 实例常驻，大 TP 请求到达时立即派发，不做 drain，通过 contention budget 对小 TP 请求做流控。

## Latency Model

每种 TP 配置独立拟合 4 常数 (a, b, c, d)：

```
latency({s_i}, tp) = max(a·Σs_i/tp + b·Σs_i²/tp, c) + d
```

用 scipy curve_fit 对 (seq_len, measured_latency) profiling 数据拟合。每种 TP 分别拟合，不做跨 TP 统一模型。

## 请求到达时的完整流程

### Phase 1: TP 选择

从小到大遍历候选 TP 组：
1. exclusive = PredictLatency(seq_len, tp)
2. n = 该 tp 组 GPU 上当前活跃进程数
3. exec_time = exclusive × max(n, 1)   // MPS 平分资源
4. predicted_ttft = exec_time + queue_delay
5. 如果 predicted_ttft ≤ T_slo → 选这个 tp，break

queue_delay = 目标实例 queue 中已有请求的 predicted_latency 之和，选 queue_delay 最小的实例。

**TP 选择是 one-shot decision**：后续即使被 block 也不重新评估 TP。禁止重评估，否则会级联退化（block → 升级大 TP → 大 TP 负载增加 → 更多 block → 吞吐坍塌）。

### Phase 2: 大 TP 请求直接派发

如果选定的 TP 不是最小 TP（即需要大 TP）：
1. budget = T_slo - exclusive_latency
2. 注册为 active large-TP instance，记录 gpu_set 和 budget
3. 直接派发，return

### Phase 3: 小 TP 请求准入控制

如果选定的是小 TP 且有大 TP 实例活跃：

1. 计算 delta = min(exclusive_latency, min(remain(I) for 所有活跃大 TP I))
   - remain(I) = (I.start_time + T_slo - I.budget_remaining) - now  // I 的预计剩余墙钟时间
2. 遍历候选实例，每个实例查其 GPU 组上所有更大 TP 实例的最低剩余 budget，选 min_budget 最高的实例（负载均衡）
3. 如果 delta ≤ min_budget → 准入，在所有受影响大 TP 实例上扣 delta，派发
4. 如果 delta > min_budget → block，await 大 TP 完成事件后重试

大 TP 完成时：从 active 列表移除，fire 完成事件，唤醒等待队列中的请求重新跑准入检查。

## Slowdown 模型说明

MPS 下 GPU 吞吐守恒（prefill compute-bound，共置不增加总吞吐）。大请求 L 的实际完成时间 = L 的独占工作量 + 所有并发小请求的独占工作量之和。所以每个小请求对大请求的 slowdown = 该小请求的独占执行时间，与并发数量和到达顺序无关。这不是保守估计，是吞吐守恒下的数学结论。

## 关键注意

- 没有活跃大 TP 实例时，Phase 3 无约束，直接派发（退化为无流控）
- "大/小 TP" 是相对概念：TP2 对 TP4 是小 TP。准入检查看 gpu_set 包含目标 GPU 且 TP 更大的所有实例
- slowdown 的 1:1 系数如果实验偏差大，只需改 delta 计算里的系数
- block 用 python async 实现：大 TP 完成时 fire asyncio.Event，等待的请求 await 后重试
