# 260902-FlexTP V6 调度器设计、实现与 V3-V5 对比

本文自包含地说明 FlexTP V6 的设计、生产接口、普通 PD 生命周期、离散事件模拟方法，
以及它与 naive MPS、V3、V4、V5 的对比结果。V6 没有继承 V3-V5 的候选选择算法；
它以 naive MPS 的固定 TP 分流为基础，增加一个默认并发、仅在能够挽救更多按时工作时
才串行的全局模式控制器。

从本文开始，新建 Markdown 文档的文件名和一级标题都使用 `YYMMDD-` 日期前缀。

对应文件：

- 生产调度器：`lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v6.py`
- selector factory：`lightllm/server/httpserver_for_pd_master/pd_selector/__init__.py`
- 普通 PD manager：`lightllm/server/httpserver_for_pd_master/manager.py`
- step 精确模拟器：`test/benchmark/service/flex_tp_step_sim.py`
- V3-V6 对比入口：`test/benchmark/service/flex_tp_v6_comparison.py`
- selector 契约测试：`test/test_flex_tp_selector_v6.py`
- V6 启动脚本：`start-cluster4_70b_p22.4d4_mps_flex_v6.sh`

## 1. 目标拓扑与问题

目标部署在 8 张 H200 上使用 4 张卡做 Prefill、另外 4 张卡做 Decode：

| 实例 | TP | 物理 GPU | V6 请求类别 |
|---|---:|---|---|
| `p01` | 2 | 0,1 | short |
| `p23` | 2 | 2,3 | short |
| `p0123` | 4 | 0,1,2,3 | long |
| `d4567` | 4 | 4,5,6,7 | Decode，无瓶颈假设 |

三个 Prefill 进程常驻并通过 CUDA MPS 共享 GPU。两个 TP2 互不重叠，但每个 TP2 都
与 TP4 重叠。V6 只使用当前可工作的普通 `prefill`/`decode` PD 路径，不接入
`nixl_prefill`、`nixl_decode` 或 NIXL 握手消息。

naive MPS 的重要优点是 work-conserving：只要某个实例可收请求，就立即分流；即使
master 每次只发一个请求，worker 的 `ChunkedPrefillQueue` 仍然会 batching。因此 V6
不能把“支持 batching”当作相对 naive 的优势。它需要保留 naive 的并发性，同时只在
MPS 干扰真的会导致更多 deadline miss 时暂时停止一类新工作。

## 2. V6 的设计原则

V6 使用以下四条原则：

1. **TP 放置保持简单且确定。** `input_tokens <= 4000` 固定走最小常驻 TP，
   `input_tokens > 4000` 固定走最大常驻 TP。在目标拓扑中分别是 TP2 和 TP4。
2. **ALL 是默认模式。** TP2 和 TP4 默认同时收请求并自然形成 MPS 重叠。
3. **串行必须证明有严格收益。** 只有预测的 short-first 或 long-first 能获得更高的
   按时效用时，才进入临时独占 admission 窗口；相同得分始终优先 ALL。
4. **决策只看一个浅执行窗口。** master 不向 worker 无限灌入请求，使下一次模式决策
   不会被大量不可见的 worker backlog 架空。

这里的模式是 **master admission 模式**，不是 CUDA MPS 开关，也不会暂停或杀死常驻
进程。已经交给 worker 的 Prefill 不可抢占；V6 只能停止发放新 lease，等待冲突工作结束。

## 3. 请求、队列与硬路由

每个到达请求先进入全局 pending 队列：

```text
seq_len  = max(1, input_token_num)
deadline = arrival_time + slo_ttft
priority = (deadline, enqueue_order)
```

默认 `slo_ttft=3s`，类别判断使用严格大于：

```text
short: seq_len <= 4000 -> 最小 TP -> 两个 TP2 之一
long:  seq_len >  4000 -> 最大 TP -> TP4
```

同类别有多个实例时，优先选择 admitted token 较少、lease 较少的实例。V6 不枚举错误
类别的 TP 候选，所以在 TP2/TP4 都可用时，long 进入 TP2 的数量是硬性为零。若暂时只有
一种 TP 规格，全部请求使用该唯一规格，避免因集群尚未注册完整而永久等待。

## 4. 浅执行 credit 与 batching

每个 Prefill 实例默认最多持有：

- 64 个在途 lease；
- 16384 个 admitted input tokens；
- 64 个已经 ACK 的 transport bundle，作为异常情况下的第二层保护。

空实例允许先接收一个本身超过 16384 token 的请求，但在它完成前不能继续超发。token
credit 是主要控制量，transport bundle 数不是 worker batch 数；多个小 bundle 可以在
worker 侧进入同一个实际 Prefill batch。

master 的 bundle dispatcher 默认最多等待 20ms，累计到 4096 token 时提前 flush，单个
transport bundle 的多请求 token cap 为 8192、request cap 为 64。worker 配置保持：

```text
--batch_max_tokens 16384
--chunked_prefill_size 8192
```

因此 V6 的作用不是发明新的 batching，而是限制 master 暴露给 worker 的近期工作量，给
后续模式判断留下可执行的控制边界。最终 batch packing 和长请求 8192-token 分块仍由实例
侧 Router 决定。

## 5. 三种模式

V6 每次重规划比较三个反事实：

| 模式 | 含义 |
|---|---|
| `ALL` | short/TP2 与 long/TP4 同时执行，按 MPS profile 施加 slowdown |
| `SHORT_ONLY` | 先让 short 独占重叠 GPU，short 窗口结束后再执行 long |
| `LONG_ONLY` | 先让 long 独占重叠 GPU，long 窗口结束后再执行 short |

当只有一个类别有工作时，直接使用该类别；只有 short 和 long 同时存在时才需要比较三个
反事实。模式会在请求到达、lease 完成/失败、节点变化和默认每 50ms 的 replan 时重新计算，
不是永久状态。

### 5.1 预览窗口

对 short 和 long 分别构造一个有限预览：

1. 纳入目标实例上已有的 lease；
2. pending 按 `(deadline, enqueue_order)` 排序；
3. 最多看 128 个 pending 请求；
4. 每个实例最多增加一个近期 pending 执行窗口；
5. 按已规划 token 最少的实例分配同类别 pending；
6. 用离线 Prefill 模型得到每个请求的相对完成时间。

预测批耗时为：

```text
T(batch, tp) = max(
    a * sum(L) / tp + b * sum(L^2) / tp,
    c
) + d
```

当前 TP2/TP4 常数为：

| TP | a | b | c | d |
|---:|---:|---:|---:|---:|
| 2 | `2.018352e-4` | `6.048427e-9` | `2.341091e-2` | `6.794939e-2` |
| 4 | `2.655797e-4` | `2.330274e-9` | `5.651746e-2` | `3.767872e-2` |

这些是调度模型参数，不是当前机器的在线实测结果。每个预测完成时间还加入 20ms bundle
window 和 80ms 安全余量。

### 5.2 ALL 反事实

若 TP2 和 TP4 重叠运行，两个类别的相对完成时间分别乘以其 mode slowdown：

```text
finish_all(class) = now
                  + relative_finish(class) * overlap_factor(class)
                  + fixed_delay
```

生产默认 `overlap_factor=2.0`。也可以用
`(target_tp, sorted_active_tp_signature)` 提供实测覆盖，例如模拟中的 1.6 或 2.0。

### 5.3 两个串行反事实

short-first 中，short 按 solo 速度立即执行，long 的起点延后一个 short solo window；
long-first 对称：

```text
SHORT_ONLY:
  finish(short) = now + short_relative_finish
  finish(long)  = now + short_solo_duration + long_relative_finish

LONG_ONLY:
  finish(long)  = now + long_relative_finish
  finish(short) = now + long_solo_duration + short_relative_finish
```

实现对每个预测请求检查 `finish <= deadline`，得到按时请求数和按时 input token 数。

### 5.4 效用与选模

三个反事实使用同一个效用：

```text
U = on_time_input_tokens + 2000 * on_time_request_count
```

token 项避免只照顾大量极短请求，请求项避免一个超长请求完全压过大量短请求。当前 2000
是可配置的构造参数，但尚未暴露为生产 CLI。选择规则是：

```text
best = ALL
if U(short-first) > U(best): best = SHORT_ONLY
if U(long-first)  > U(best): best = LONG_ONLY
```

浮点容差内的相同得分由 ALL 获胜。只有两个串行方案都严格优于 ALL 且彼此相同时，若 long
已经在独占执行而 short 尚未开始，V6 保持 long-first，避免在不可抢占工作尚未结束时反复
改变目标。

2000 的选择来自最困难的 `mm-image rate=10, slowdown=2.0` 单点扫描：0、1000、2000、
4000、8000、16000 中 2000 的模拟 SLO/token goodput 最好。这意味着当前比较存在调参点
重用，不能把结果当成完全独立的 held-out 评估；实机部署前应使用训练 trace 定参，再在
不同日期的 trace 上验证。

## 6. 非抢占执行与排空

当选择 `SHORT_ONLY` 时，V6 只准入 short；选择 `LONG_ONLY` 时只准入 long。若相反类别
已经有 lease，V6 不会假装它已停止，而是暂停新的 admission，等现有 lease 的 worker
完成/失败通知将其释放后再规划。这形成非抢占的 drain：

```text
选择 SHORT_ONLY + long lease 存在
  -> 不再发新 lease
  -> 等待 long PREFILL_FINISHED/FAILED
  -> replan
  -> short 可进入 TP2
```

若反事实不显示严格收益，系统保持 ALL，两个 TP2 和 TP4 可以继续同时服务。这一点是 V6
和 V3/V4 的保守保护、V5 的候选级规则之间最主要的算法差异。

## 7. 普通 PD 生命周期与接口兼容

V6 直接继承基础 `PDSelector`，不继承或导入 V3/V4/V5。它暴露能力标记：

```text
supports_bundles = True
uses_prefill_lease_lifecycle = True
```

PD manager 已由硬编码的 V3 类型判断改为能力检测，因此 V3-V6 共同使用：

```text
Pending
  -> admitted lease
  -> per-node bundle queue
  -> REQ_BUNDLE sent
  -> worker BUNDLE_ACK
  -> worker PREFILL_FINISHED / PREFILL_FAILED
  -> lease released
  -> replan
```

请求 ID、节点地址和 `instance_generation` 会共同校验完成事件。节点断连、发送失败和 generation
替换会释放对应 lease；旧 websocket 的 ACK/完成消息不能释放新实例的工作。多段输出的每个
continuation 会按“原 prompt + 已生成 history”的新 Prefill 长度重新申请 lease。

V6 接收 worker 的 queued/running request/token 和 load report，当前主要用于 telemetry；
反事实仍基于 master lease 和静态 profile，没有直接用实时 report 修正服务时间。

## 8. 与 naive、V3、V4、V5 的算法差异

| 维度 | naive MPS | V3 | V4 | V5 | V6 |
|---|---|---|---|---|---|
| TP 路由 | 4K 固定阈值 | 枚举 TP，按 deadline/成本选 | 4K 硬隔离 | 同 V4 | 4K 硬隔离 |
| long 进入 TP2 | 否 | 可能 | 否 | 否 | 否 |
| master bounded bundle | 否 | 是 | 是 | 是 | 是 |
| worker batching | 是 | 是 | 是 | 是 | 是 |
| deadline 模型 | 无 | 逐候选 | 逐候选 | 逐候选 | 类别窗口反事实 |
| 保护旧 lease | 无 | 强保护 | 强保护 | short 可条件放宽 | 非抢占排空 |
| future reservation | 无 | 有 | short 绕过 | short 绕过 | 无 |
| MPS 策略 | 始终自然重叠 | 逐候选安全才重叠 | 同 V3 | 候选级 MPS-first | ALL 默认、严格收益才串行 |
| 优化单位 | 单请求 | 单请求候选 | 单请求候选 | 单请求候选 | short/long 两类近期窗口 |
| 继承关系 | 独立 | 基础实现 | 继承 V3 | 继承 V4 | 独立 |

V3-V5 的主要复杂度来自“pending × instance”的候选、保护已有 lease 和 future reservation。
V6 删除这套逐请求放置搜索，因为 TP 已由阈值确定；剩余决策只是在 ALL、short-first、
long-first 三个模式间比较近期按时价值。

## 9. 模拟方法

模拟器使用虚拟时钟直接加载生产 `FlexTPSelectorV3/V4/V5/V6`，不是另写一个简化版全局
调度器。实例侧直接使用 LightLLM 的 `ChunkedPrefillQueue`，逐个 Router step 模拟
8192 chunked Prefill、16384 batch token 上限、bundle、ACK、worker report、完成和重规划。
Decode 在 Prefill 完成后立即结束，不构成瓶颈。

统一配置：

| 项目 | 值 |
|---|---:|
| TTFT SLO | 3s |
| 长请求阈值 | 严格 `>4000` |
| bundle window / trigger / cap | 20ms / 4096 / 8192 tokens |
| instance token credit | 16384 |
| worker chunk / batch cap | 8192 / 16384 tokens |
| MPS per-instance slowdown | 1.6、2.0 |
| Decode | 无瓶颈、立即完成 |
| seed | 0 |

工作负载：

- `phase-shift`：20s × 20 rps 短请求、20s × 3 rps 的 8000-token 请求、20s × 8 rps
  的混合请求，共 620 个；
- `synthetic-5pct-rate10`：1000 个请求，95% 为 100-1000 tokens，5% 为
  1000-20000 tokens，固定 10 rps；
- ServeGen `mm-image`：180s，分别为 8、9、10 rps。

主指标为固定 offered arrival window 内的按时 input-token goodput，避免一个策略因排空更慢
而虚增或虚减吞吐。`SLO` 使用全部 offered 请求作分母；`p95` 只统计最终完成请求，因此必须
结合 completion fraction 解读。

## 10. V3-V6 完整对比结果

下表每格依次为 `按时 input-token goodput (token/s) / offered SLO / p95 TTFT (s)`。
naive 使用相同 4000-token 阈值并允许全部实例同时服务。

| workload | slowdown | V3 | V4 | V5 | V6 | naive |
|---|---:|---:|---:|---:|---:|---:|
| phase-shift | 1.6 | 3864 / 66.29% / 5.940 | 5575 / 84.19% / 4.117 | 5323 / 85.48% / 4.083 | 4512 / 84.03% / 22.662 | 4512 / 84.03% / 22.534 |
| phase-shift | 2.0 | 3864 / 66.29% / 6.940 | 5358 / 83.87% / 4.137 | 4919 / 84.84% / 4.167 | 4512 / 84.03% / 25.424 | 4378 / 83.87% / 25.216 |
| synthetic-5pct, 10 rps | 1.6 | 9604 / 99.60% / 1.574 | 9659 / 99.60% / 0.650 | 9518 / 99.50% / 0.468 | 9483 / 99.50% / 1.183 | 8257 / 98.60% / 0.489 |
| synthetic-5pct, 10 rps | 2.0 | 9604 / 99.60% / 1.736 | 9273 / 99.30% / 1.574 | 8658 / 98.90% / 0.731 | 9145 / 99.20% / 1.555 | 7565 / 98.10% / 0.774 |
| mm-image, 8 rps | 1.6 | 13289 / 98.26% / 2.635 | 13340 / 99.65% / 2.050 | 13322 / 99.58% / 1.991 | 13217 / 99.51% / 2.023 | 13206 / 99.58% / 2.088 |
| mm-image, 8 rps | 2.0 | 10431 / 81.58% / 4.684 | 12319 / 94.84% / 2.990 | 12117 / 95.74% / 2.854 | 11932 / 97.00% / 2.617 | 10973 / 82.69% / 4.458 |
| mm-image, 9 rps | 1.6 | 13240 / 83.75% / 4.823 | 15182 / 99.26% / 2.160 | 15205 / 99.38% / 2.160 | 15097 / 98.52% / 2.161 | 14880 / 96.23% / 2.613 |
| mm-image, 9 rps | 2.0 | 12487 / 77.26% / 5.989 | 12140 / 81.40% / 5.292 | 13048 / 89.74% / 3.886 | 13280 / 93.57% / 3.225 | 11387 / 69.34% / 6.589 |
| mm-image, 10 rps | 1.6 | 13489 / 84.02% / 4.679 | 15810 / 97.49% / 2.465 | 15821 / 97.22% / 2.475 | 15542 / 95.66% / 2.905 | 15319 / 92.82% / 3.270 |
| mm-image, 10 rps | 2.0 | 3005 / 20.82% / 6.492 | 13665 / 89.81% / 3.882 | 13264 / 86.47% / 4.354 | 12736 / 88.86% / 4.655 | 10109 / 58.57% / 5.843 |

全部 10 组 V6 的 completion fraction 都是 100%，且 `tp2_long_request_count=0`。

## 11. 结果解读

### 11.1 相对 naive

V6 的 token goodput 为 9 胜 1 平，中位增益为 5.89%；SLO 为 9 组不低于 naive，p95
为 6 组不高于 naive。主要结果如下：

| workload | slowdown | V6 token 增益 | V6 SLO 变化 | V6 p95 变化 |
|---|---:|---:|---:|---:|
| phase-shift | 1.6 | 0.00% | 0.00 pp | +0.128s |
| phase-shift | 2.0 | +3.05% | +0.16 pp | +0.208s |
| synthetic-5pct, 10 rps | 1.6 | +14.85% | +0.90 pp | +0.694s |
| synthetic-5pct, 10 rps | 2.0 | +20.89% | +1.10 pp | +0.781s |
| mm-image, 8 rps | 1.6 | +0.08% | -0.07 pp | -0.066s |
| mm-image, 8 rps | 2.0 | +8.73% | +14.31 pp | -1.841s |
| mm-image, 9 rps | 1.6 | +1.46% | +2.29 pp | -0.452s |
| mm-image, 9 rps | 2.0 | +16.63% | +24.23 pp | -3.364s |
| mm-image, 10 rps | 1.6 | +1.46% | +2.84 pp | -0.365s |
| mm-image, 10 rps | 2.0 | +25.99% | +30.29 pp | -1.188s |

`mm-image 8 rps, slowdown=1.6` 的 SLO 低 0.07 pp，虽 token goodput高 0.08%，应视为
统计和模型精度内的近似持平，不应宣称严格胜出。

### 11.2 相对 V3

V6 的 token goodput 为 7/10 胜，中位增益 14.21%；SLO 为 8/10 不低于 V3，p95 为
8/10 不高于 V3。高压 `mm-image rate=10, slowdown=2.0` 中，V6 相对 V3 的 token
goodput 提高 323.8%，SLO 提高 68.04 pp。核心原因是 V6 的 hard routing 不允许 long
占住两个 TP2。

V3 在两个 synthetic 组和低压 `mm-image rate=8, slowdown=1.6` 的 token goodput 更高，
说明 V6 的模式排空和固定 TP 隔离也会损失部分资源弹性。

### 11.3 相对 V4/V5

V6 不是 V4/V5 的全面替代：token goodput 相对 V4 仅 1/10 胜，相对 V5 仅 2/10 胜；
中位差分别为 -1.76% 和 -1.16%。slowdown=1.6 时 MPS 并发本身有聚合吞吐收益，V4/V5
的候选级控制通常比 V6 的类别排空更有效。

V6 最明显的优势点是 `mm-image rate=9, slowdown=2.0`：

- 相对 V4：token goodput +9.39%，SLO +12.18 pp，p95 -2.067s；
- 相对 V5：token goodput +1.78%，SLO +3.83 pp，p95 -0.662s。

`mm-image rate=8, slowdown=2.0` 中 V6 的 SLO 最高，为 97.00%，但 token goodput比 V4
低 3.15%、比 V5 低 1.52%。`rate=10, slowdown=2.0` 中 V6 的 token goodput比 V4 低
6.80%、比 V5 低 3.98%；其 SLO 比 V4 低 0.95 pp，但比 V5 高 2.39 pp。这反映了当前
效用权重在“按时请求数”和“按时 token 数”之间的真实取舍。

### 11.4 phase-shift 的 p95 陷阱

phase-shift 中 V6 的 p95 超过 22s，而 V4/V5 约 4.1s，不能只看这一列得出 V6 完全失败。
V6 完成了 100% offered 请求；V4/V5 的 completion fraction 只有：

| slowdown | V4 | V5 | V6 |
|---:|---:|---:|---:|
| 1.6 | 94.19% | 94.03% | 100% |
| 2.0 | 93.87% | 93.39% | 100% |

V6 的 `best_effort` 会在 deadline 已失守后继续排空长队列，所以完成请求的尾延迟很高；
V4/V5 留下未完成请求后，完成样本的 p95 更好看。固定 offered-window goodput 和 offered SLO
才是这里更公平的主指标。若生产目标要求限制过期请求资源，应改用 `overload_policy=reject`
或增加显式降级队列。

## 12. 当前结论

V6 达到了设计上的三个目标：

1. 独立于 V3-V5，以小于等于/大于 4000 的硬路由消除 long 占用 TP2；
2. 保留 naive MPS 的 ALL 默认并发，只让有严格预测收益的串行方案触发排空；
3. 在当前模拟矩阵中相对 naive 获得稳定的 on-time token goodput，尤其在 slowdown=2.0
   和 ServeGen 高压区间明显改善。

但 V6 目前不能宣称优于 V4/V5。V4/V5 在大多数测试点的 token goodput 更高，尤其是
slowdown=1.6 和 phase-shift。V6 的价值是算法更独立、决策面更小、默认行为更接近 naive，
并在一个关键中高压点同时超过 V4/V5；它不是已经完成的统一最优策略。

## 13. 局限与后续实机验证

- 模拟器执行真实调度代码和实例队列，但不执行 H200 kernel、NCCL、网络、CPU tokenizer、
  KV 传输和真实 CUDA MPS 调度，因此表中数字不是实机性能结果；
- 当前生产 slowdown 默认 2.0，必须用目标机器上 `TP2+TP4` 和 `TP2+TP2+TP4` 的实际
  batch bucket profile 替换；
- 反事实只看每实例一个 pending 窗口、全局最多 128 个请求，可能看不到更远的 phase shift；
- 2000 request utility 在一个高压点调过参，需要 held-out trace 和多 seed 验证；
- worker report 尚未用于估计 remaining chunks，长请求的部分进度仍不够精确；
- admission 不可抢占，没有严格的 starvation 上界或 aging 证明；
- 最小/最大 TP 当前按 master 下全部可用实例计算，不是逐 FlexTP group 计算；多异构组需要
  改成 group-local 模式决策；
- 三种以上 TP 规格时只使用最小和最大 TP；
- 当前检查时 GPU 0-3 空闲，但 GPU 4-7 已被其他任务以接近 100% 利用率占用。为避免干扰
  现有任务，本轮没有启动需要全部 8 张卡的 cluster4 实机 A/B。

实机验证应固定同一 commit、同一到达 trace 和同一 8 卡普通 PD 拓扑，依次启动 V3、V4、
V5、V6 和 naive，并至少报告 offered SLO、on-time request/token goodput、short/long p95/p99、
completion fraction、MPS mode 占比和 physical GPU-seconds。

## 14. 复现命令

运行 V6 模拟器自检：

```bash
python test/benchmark/service/flex_tp_step_sim.py --self-test
```

运行本文的 50 次对比：

```bash
python test/benchmark/service/flex_tp_v6_comparison.py \
  --slowdowns 1.6,2.0 \
  --synthetic-prompts 1000 \
  --seed 0 \
  --output-dir _/flex_tp_paper_analysis/v6-comparison-v2
```

原始结果：

```text
_/flex_tp_paper_analysis/v6-comparison-v2/results.json
_/flex_tp_paper_analysis/v6-comparison-v2/results.csv
_/flex_tp_paper_analysis/v6-comparison-v2/v6_comparison.json
_/flex_tp_paper_analysis/v6-comparison-v2/v6_comparison.csv
```

普通 PD 实机启动入口：

```bash
FLEX_TP_MPS_SLOWDOWN=2.0 ./start-cluster4_70b_p22.4d4_mps_flex_v6.sh
```

启动脚本仍使用两个 TP2 Prefill、一个 TP4 Prefill、一个独立 TP4 Decode，并显式传入
`tp_smt_group_id` 和 `tp_smt_gpu_ids`。它不会走当前存在兼容问题的 NIXL 路径。
