# 260902-FlexTP V7/V8：面向 TP-SMT 的两种全新调度器与论文式对比

本文自包含地说明 V7、V8 的设计、普通 PD 生产接口、精确 step 模拟方法，以及它们与
V3、V4、V5、V6 和 naive MPS 的横向比较。本文中的 **TP-SMT** 指多个不同 TP 规格的
Prefill 实例同时存在并同时执行：目标拓扑中 TP2 与 TP4 可以在同一时间窗口内运行，
而不是在 TP2/TP4 之间无缝切换单一活动模式。

## 1. 结论先行

V7 和 V8 都是从零设计的独立策略。它们不继承 V3-V6 的 selector，也不枚举旧版本的
候选模式：

- **V7：在线流量/对偶价格策略。** 每个实例维护虚拟完成时间；共享 GPU 冲突被转换为
  价格；short/long 类别用 deficit debt 做近似 max-min 公平；在软 deadline urgency 加权
  后选择边际代价最小者。
- **V8：固定 epoch/wavefront 策略。** 每个 epoch 同时打开 short lane 和 long lane，
  先按加权 max-min 计算 token quota，再在各 lane 内按 EDF 排空，并把请求均匀分给对应
  TP 池。epoch 是 admission 的批次边界，不是 CUDA MPS 开关。

两者的共同安全不变量是：`seq_len <= 4000` 只进入最小 TP，`seq_len > 4000` 只进入
最大 TP；因此在目标拓扑中长请求进入 TP2 的计数必须为零。deadline 不作为硬拒绝条件，
而是进入 V7 的 urgency score、V8 的 lane demand 和两者的预测 telemetry；这样默认
`best_effort` 仍然 work-conserving，在 overload 下最终完成所有请求，`reject` 模式才会
对已经过期的 pending 请求返回异常。

这里的“长请求不进 TP2”结论针对完整目标拓扑。若启动阶段暂时只注册一种 TP 规格，
两版都会把请求送入当前唯一可用规格以避免永久阻塞；线上应在拓扑收齐后再把该计数作为
TP-SMT 安全断言。

在本地离散事件模拟中，V7 通常拥有最高 token goodput，V8 在中等负载下具有稳定的
公平性和批次边界，但固定 epoch 会带来可测的等待开销。高压下两者都要在 token 吞吐与
TTFT SLO 之间取舍，不能把模拟结果解释为真实 GPU 的绝对性能。

## 2. 实验边界与目标拓扑

模拟只覆盖当前可用的普通 `prefill`/`decode` PD 路径，不使用 NIXL selector 路径。Prefill
使用三类常驻实例：

| 实例 | TP | 物理 GPU | 硬路由类别 |
|---|---:|---|---|
| `p01` | 2 | 0,1 | short |
| `p23` | 2 | 2,3 | short |
| `p0123` | 4 | 0,1,2,3 | long |
| Decode | 4 | 4,5,6,7 | 模拟中无瓶颈 |

TP4 与两个 TP2 共享 GPU，因此同时运行时通过 overlap profile 施加 slowdown；两个 TP2
之间互不重叠。所有 selector 都通过同一套 lease、bundle ACK、worker report、完成/失败、
节点重注册和 generation 检查接口接入普通 PD manager。

## 3. V7：在线流量与对偶价格

### 3.1 独立状态

实现文件为 `lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v7.py`。
V7 自己定义 `V7Request`、`V7Lease`、`V7Instance` 和 `V7ServiceCurve`，只依赖基础
`PDSelector` 的节点注册抽象。每个实例保存：

- `virtual_finish`：将当前 lease 按本地 batch curve 排列后的虚拟完成时间；
- admitted token、lease 数、ACK bundle 数和 worker report 负载；
- GPU placement、TP、group 和 `instance_generation`；
- short/long 两类的 `class_debt`，以及 price/flow/deadline telemetry。

### 3.2 硬路由与可接收条件

请求首先按严格大于判断分类：

```text
short: seq_len <= 4000 -> min(tp_size)
long : seq_len >  4000 -> max(tp_size)
```

候选实例必须可用、未超过 64 个 in-flight lease、未超过 64 个 ACK bundle；非空实例还要
满足 admitted token credit（默认 16384）。空实例允许先放入一个超过 credit 的单请求，
之后暂停超发，保证超长请求不会被 credit 规则永久拒绝。

### 3.3 预测与价格

对候选实例 `i` 和请求 `r`，V7 将现有 lease 加入本地 batch packing（bundle cap 8192），
使用本地 service curve 预测：

```text
F(i,r) = max(now, virtual_finish_i)
         + slowdown(i, active ∪ {i}) * service_i(existing + r)
         + batch_window + prediction_margin
```

共享 GPU 冲突价格为：

```text
P(i) = Σ_j overlap(i,j) * min(2, token_ratio_j + 0.5 * running_ratio_j)
```

类别压力是该类别 pending token 加 resident token 除以该类 TP 池的总 credit。每经过一个
replan 间隔，只要类别有 backlog，其 debt 增加一个 class quantum（默认 4096 token），并
限制在有限上下界内，防止长期饥饿。

### 3.4 选择规则

令 `slack = deadline - F(i,r)`，`urgency` 为按 SLO 归一化并截断到 `[0,4]` 的紧迫度，
`D` 为类别 debt，`Q` 为类别压力，则边际分数为：

```text
score(i,r) = F(i,r)
             + urgency_weight * urgency
             + conflict_price_weight * P(i)
             + 0.25 * Q
             - 0.05 * D / class_quantum
```

每一轮直接在可接收候选中取最小 score；deadline 越近，urgency 项越大，但不会因为预测
误差而停止填充可用 TP 池。`overload_policy=reject` 才会在 deadline 后清理 pending，
默认 best-effort 则继续 admission 并记录 `deadline_overrides`。这样在线策略保持
work-conserving，避免一个错误的完成时间预测把整个 TP-SMT 系统饿死。

### 3.5 并行语义

V7 不设置 `SHORT_ONLY`、`LONG_ONLY` 或串行计数器。只要两个类别都有 pending 且各自
TP 池有 credit，它们会在同一个 admission 窗口获得 lease；MPS 只影响预测的 slowdown，
不改变常驻进程和并行关系。因此 V7 的核心是“带资源价格的同时服务”，不是模式切换。

## 4. V8：固定 epoch 与双 lane wavefront

### 4.1 独立状态

实现文件为 `lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v8.py`。
V8 自己定义 `V8Job`、`V8Lease`、`V8LaneInstance`、`V8Profile`，只依赖基础
`PDSelector`。pending 被硬分到 `short` 和 `long` 两个 lane；每个 lease 记录产生它的
`epoch_id`，便于审计“同一 epoch 是否同时打开两类 TP”。

### 4.2 epoch 配置

默认 epoch 为 50ms，单 epoch token budget 为 16384；两条活跃 lane 各保留至少 2048 token
配额（总预算不足时按实际预算缩小）。默认 short/long 权重均为 1.0，deadline pressure
weight 为 2.0。每到 epoch 边界，两个 lane 在同一个 `_run_epoch` 中计算和排空，不允许一
条 lane 独占整个 epoch。

### 4.3 需求、deficit 与 quota

对 lane `c`，定义：

```text
backlog_c = pending token
resident_c = 已 admission token
urgency_c = clip((SLO - oldest_slack) / SLO, 0, 3)
demand_c = (backlog_c + 0.25 * resident_c)
            * (1 + deadline_pressure_weight * urgency_c) * weight_c
```

活跃 lane 的 deficit 每个 epoch 增加 `budget / active_lane_count`，并限制在
`4 * budget`。两 lane 时先分配 `min_lane_quota`，剩余预算按
`demand_c + deficit_c` 比例分配；整数舍入产生的剩余 token 逐个交给“需求减已得 quota”
最大的 lane，确保预算不丢失。

### 4.4 EDF 排空与实例选择

每条 lane 内按 `(deadline, enqueue_order)` 排序。请求必须满足对应 TP 池的 lease、bundle
和 token credit 限制；在可用实例中按 planned token、admitted token、node key 的字典序
选最小者。实例预测同样采用本地 batch curve 和当前 overlap slowdown，deadline pressure
只影响 quota，不把预测误差变成硬阻塞；`reject` 模式在 epoch 末清理过期 pending。

V8 的 wavefront 边界使 quota、EDF 和 telemetry 都容易复现，代价是至少一个 epoch 的
调度等待。它适合需要强可解释性和 lane 公平性的部署；极低负载或极紧 TTFT 时应缩短
`epoch_ms`，否则固定边界会成为主要尾延迟来源。

## 5. 共同生产生命周期

两版都实现以下普通 PD 契约：

1. `async_select_p_d_node` 创建 pending request/job 和 deadline，检查外部 req id 去重。
2. selector 返回 Prefill 与 Decode 节点；Prefill lease 进入 bundle dispatcher。
3. `notify_bundle_accepted` 将 ACK、bundle id 和 state 写入 lease；generation 不匹配的完成
   消息被忽略。
4. `update_instance_report` 只接受单调递增的 report sequence，节点变化会重建 group 和
   placement 索引。
5. `notify_request_done` / `notify_request_failed` 删除 lease、清理外部 id 并触发 replan。
6. 节点下线时，空实例被移除；仍有 lease 的实例保留到完成，避免把旧 generation 的工作
   错交给新进程。

二者均设置 `supports_bundles=True` 和 `uses_prefill_lease_lifecycle=True`，所以 worker
侧仍使用生产 `PrefillBundleDispatcher` 与 `ChunkedPrefillQueue`。调度器的 admission bundle
不是 worker 的最终 batch；worker 仍按 `chunked_prefill_size=8192` 和 `batch_max_tokens`
执行实际 step batching。

## 6. 精确 step 模拟

入口为 `test/benchmark/service/flex_tp_step_sim.py`，比较入口为
`test/benchmark/service/flex_tp_v7_v8_comparison.py`，对抗入口为
`test/benchmark/service/flex_tp_v7_v8_adversarial.py`。

模拟器使用虚拟时钟替换 selector 的 `time`，事件顺序为：到达、lease/replan、bundle flush、
router tick、worker report、Prefill step 完成、Decode transfer。Prefill step 采用同一组
按 TP 和 batch token 的曲线；MPS overlap 依据当前 active instance 的 GPU set 动态计算。
Decode 默认在 Prefill 完成后立即结束，因此结果专门衡量 Prefill/TP-SMT，不把 decode 变成
隐藏变量。每个请求的结果记录 admission、bundle、首个 step、完成时间、TP、batch 和
deadline 状态。

负载矩阵固定为：

| 负载 | 参数 |
|---|---|
| phase-shift | 先 short 后 long，再反向切换，用于考察突发响应 |
| synthetic-5pct | 1000 请求，10 req/s，5% 长请求，seed 0 |
| ServeGen mm-image | `duration=180s`，8/9/10 req/s，seed 0 |
| overlap | slowdown 1.6 和 2.0 |

共 `5 个场景 × 2 个 slowdown × 7 个策略 = 70` 次运行。每次都使用同一份请求时间戳，
并检查 V6/V7/V8 的 `tp2_long_request_count == 0`。指标包括 offered-window token
goodput、request goodput、offered SLO、TTFT p95、完成率、Prefill batch 均值和 overlap
wall time。

## 7. 结果记录

最终机器可读结果位于 `_/flex_tp_paper_analysis/v7-v8-comparison-v3/`，其中
`results.json/csv` 是逐策略数据，`candidate_comparison.json/csv` 是 V6/V7/V8 相对
V3/V4/V5/naive 的差值。下表填入最终运行产生的数据；token goodput 是 offered-window
内按时完成请求的输入 token/s，SLO 是 offered 请求分母。

### 7.1 重点压力点：ServeGen mm-image，10 req/s，slowdown=2.0

| 策略 | token goodput (token/s) | request goodput (req/s) | offered SLO | TTFT p95 (s) | 完成率 |
|---|---:|---:|---:|---:|---:|
| V3 | 3005.2 | 2.078 | 0.2082 | 6.492 | 0.954 |
| V4 | 13664.7 | 8.961 | 0.8981 | 3.882 | 0.986 |
| V5 | 13264.4 | 8.628 | 0.8647 | 4.354 | 0.984 |
| V6 | 12736.2 | 8.867 | 0.8886 | 4.655 | 1.000 |
| V7 | 11565.4 | 7.333 | 0.7350 | 4.798 | 1.000 |
| V8 | 9693.9 | 5.783 | 0.5796 | 6.572 | 1.000 |
| naive | 10109.0 | 5.844 | 0.5857 | 5.843 | 1.000 |

### 7.2 全矩阵平均（70 次运行）

| 策略 | token goodput (token/s) | offered SLO | TTFT p95 (s) | 完成率 |
|---|---:|---:|---:|---:|
| V3 | 9287.6 | 0.7775 | 4.549 | 0.9839 |
| V4 | 11232.0 | 0.9294 | 2.932 | 0.9814 |
| V5 | 11119.6 | 0.9369 | 2.717 | 0.9827 |
| V6 | 10945.6 | 0.9399 | 6.841 | 1.0000 |
| V7 | **11864.3** | 0.9185 | 9.024 | 1.0000 |
| V8 | 10255.5 | 0.8661 | 7.499 | 1.0000 |
| naive | 10058.6 | 0.8638 | 7.387 | 1.0000 |

均值只用于概览，不能替代按场景报告：V7 的平均 token goodput 比 naive 高 17.9%，但
平均 p95 更长；V8 的 token goodput 比 naive 高 2.0%，SLO 近似相同。ServeGen 高压点
则 V4/V5/V6 的 SLO 明显优于 V7/V8，说明在强 MPS 干扰下“允许同时运行”与“严格
deadline 优先”是不同目标，不能只用一个综合分数掩盖取舍。

TP-SMT 的直接运行证据是 `mps_overlap_wall_s`：70 次矩阵中 V7 平均有 66.104s、V8
平均有 66.364s 的重叠 Prefill wall time，所有策略的最小场景值仍约 16.4s。也就是说，
这些数据不是把 TP2 和 TP4 轮流切换后的结果；模拟事件中确实同时存在 active TP2 与
active TP4，overlap slowdown 由两者的 GPU set 交集触发。

### 7.3 相对基线的平均差值

`candidate_comparison.csv` 对每个场景/slowdown 做配对比较，再取 10 个配对样本的平均：

| candidate vs baseline | token goodput 增益 | SLO 差值 | TTFT p95 差值 |
|---|---:|---:|---:|
| V7 vs naive | +33.42% | +5.46 pp | +1.637 s |
| V8 vs naive | +2.52% | +0.23 pp | +0.111 s |
| V6 vs naive | +9.31% | +7.60 pp | -0.546 s |
| V7 vs V4 | +15.39% | -1.09 pp | +6.092 s |
| V8 vs V4 | -9.77% | -6.33 pp | +4.567 s |

这组配对数字揭示了论文中应避免的过度结论：V7 的核心优势是同时运行带来的 token
利用率，V6 在本实验的 deadline 目标上更稳；V8 的价值主要是固定边界、公平性和可审计
性，而不是在每一个压力点追求最高吞吐。

### 7.4 论文式解读

- **吞吐：** V7 的目标函数直接最小化实例边际完成时间，通常更容易填满两个 TP 池，
  但在强 overlap 和高到达率下可能积累较长 tail；V8 的 quota 上限主动限制单 epoch
  暴露量，吞吐更平滑但存在 epoch tax。
- **SLO：** V7 的 urgency/debt 软约束提高临近 deadline 请求的优先级；V8 的 lane quota
  防止长请求吞噬整个 admission 窗口。若压力超过可服务容量，任何策略都不能同时保持 100%
  SLO 与 100% completion，必须报告这两个指标而不能只报平均吞吐。
- **TP-SMT：** naive、V6、V7、V8 都允许 TP2/TP4 同时活跃；区别在于 naive 只按长度
  阈值分流，V6 有反事实模式控制，V7 用在线价格和 debt，V8 用离散 wavefront。是否
  overlap 不应被误写成“切换 TP”。

## 8. 对抗性完整性检查

对抗脚本覆盖三个容易攻击调度器的输入形状：

1. `3999/4000/4001/8000` 重复出现，检查严格阈值和 TP4 隔离；
2. 一个超长请求先到，随后每 10ms 到达大量 short，检查长请求不会阻塞 short lane；
3. short/long 周期性同时到达，检查双 lane 并发、公平性和 quota/debt 是否持续更新。

每个场景都在 slowdown `1.6/2.0/2.5` 下运行 V3-V8 与 naive，并检查：无死锁、V7/V8
的 best-effort 请求最终完成、V4-V8 的长请求均不进入 TP2、Prefill step 仍能产生多请求
batch。bundle 的最大请求数会随边界 trace 的到达间隔变化，不能把它误当成每个场景都大于
1 的硬不变量。对抗结果写入 `_/flex_tp_paper_analysis/v7-v8-adversarial-v2/`。

对抗矩阵的均值摘要如下（3 个 slowdown 的平均）：

| 场景 | 策略 | offered SLO | token goodput (token/s) | 最小完成率 |
|---|---|---:|---:|---:|
| threshold-edge | V7 | 0.118 | 60609.8 | 1.000 |
| threshold-edge | V8 | 0.104 | 51516.7 | 1.000 |
| threshold-edge | naive | 0.083 | 45454.5 | 1.000 |
| long-then-short | V7 | 1.000 | 75600.0 | 1.000 |
| long-then-short | V8 | 1.000 | 75600.0 | 1.000 |
| dual-lane-fair | V7 | 0.320 | 17034.5 | 1.000 |
| dual-lane-fair | V8 | 0.317 | 12553.1 | 1.000 |
| dual-lane-fair | naive | 0.258 | 9794.5 | 1.000 |

## 9. 论文叙事与限制

建议论文主张是：**TP-SMT 是一种同时调度多个 TP 专用服务曲线的 admission 问题**。长度
阈值只是安全的类别边界；贡献在于如何在共享 GPU 的非线性 slowdown 下保持并发、批处理、
期限和公平性的可解释平衡。V7 提供连续的价格/流量视角，V8 提供离散的 quota/wavefront
视角，二者构成算法差异而非代码风格差异。

### 9.1 设计空间探索

本轮还实际试过一个看似更安全的硬 deadline-feasibility gate：只有预测完成时间不超过
deadline 才允许 admission。它在完整 ServeGen `10 req/s, slowdown=2.0` 点上把 V7 的
token goodput 从 `11565.4/s` 降到 `9333.1/s`，offered SLO 从 `0.7350` 降到 `0.5562`；
V8 也从 `9693.9/s, 0.5796` 降到 `9451.6/s, 0.5457`。原因是 service curve 在拥塞时
偏保守，硬门控停止了本可并行执行的工作，最后反而形成更长的 overdue 队列。因此最终
版本保留 urgency/deadline pressure 软项，把 `reject` 作为明确的 overload 选择，而不
把预测误差升级为默认阻塞。

另一个方向是全局 serial admission：它在强 overlap 时能改善 TTFT SLO，但会把 TP-SMT
退化成 TP 模式切换，无法回答“多个 TP 同时服务”的核心问题，所以只保留 V6 作为对照。
最短请求优先、按更多长度桶动态迁移等方案也没有采用：它们会让阈值附近的请求反复改变
归属，增加控制面复杂度，并给攻击者制造通过边界抖动放大 queue churn 的机会。V7 的
硬类别 + 软价格、V8 的双 lane + 固定 quota，是在并发、可解释性和攻击面之间更小的
状态空间。

从部署角度，V8 的 `epoch_ms` 应按真实模型的首 token 曲线调参：低负载和极紧 SLO 可降
到 10-20ms，高负载更应优先保证 epoch 内 quota 的稳定；V7 则适合在线调整 urgency、
conflict price 和 class quantum。两者都不应在没有重新拟合 overlap profile 的情况下
直接把模拟参数当成 GPU 性能保证。

需要明确的限制：

- 本模拟器是精确事件顺序模拟，不是 GPU 实测；service curve 和 slowdown 需要用目标模型
  的真实 trace 重新拟合。
- Decode 被设为无瓶颈，不能由此推导端到端 decode 吞吐。
- 当前普通 PD 拓扑假定 TP placement 可从 start args 解析；未把不可观测的 GPU 亲和性
  当成精确信息。
- best-effort 模式会最终接收过期请求以保证完整性；线上若更重视 SLO，应明确使用
  `overload_policy=reject` 并单独报告 rejection。

## 10. 验证命令

```bash
python -m py_compile \
  lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v7.py \
  lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v8.py \
  test/benchmark/service/flex_tp_step_sim.py

python test/benchmark/service/flex_tp_step_sim.py --self-test

python test/benchmark/service/flex_tp_v7_v8_comparison.py \
  --output-dir _/flex_tp_paper_analysis/v7-v8-comparison-v3 \
  --slowdowns 1.6,2.0 --synthetic-prompts 1000 --seed 0

python test/benchmark/service/flex_tp_v7_v8_adversarial.py \
  --output-dir _/flex_tp_paper_analysis/v7-v8-adversarial-v1 \
  --slowdowns 1.6,2.0,2.5 --seed 0
```

截至本文日期，V7/V8 的 selector 契约自检、工厂独立性检查、源码无 V3-V6 selector
依赖检查、重复 req_id 拒绝、旧 generation 消息隔离、乱序 report 丢弃、generation-aware
节点释放和 step 模拟自检均已通过；真实八卡压测需等 GPU0-7 同时空闲后再执行，不能把
其他任务占用 GPU 时的局部状态冒充实机结果。
