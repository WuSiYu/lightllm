# 260902-FlexTP V9：借鉴操作系统调度的独立 TP-SMT 调度器

本文自包含描述 V9 的动机、算法、普通 PD 生产接入、step 精确模拟和与 V3-V8/naive
的对比结果。V9 是独立实现，不是 V3、V4、V5、V6、V7 或 V8 的参数变体；它只依赖基础
`PDSelector` 的节点注册和 selector 生命周期接口。本文中的 TP-SMT 是 TP2 与 TP4
Prefill 常驻实例在共享 GPU 上同时运行，而不是在两种 TP 模式之间切换。

## 1. 结论

V9 从传统操作系统的多级反馈队列（MLFQ）、aging 和 CFS virtual runtime 获取结构，形成
三个互相配合的调度层：

1. 请求进入三个反馈级别。交互短请求初始在 level 0，中等请求在 level 1，长请求在
   level 2；等待时间达到 aging 周期后逐级提升，避免某一类长期饥饿。
2. short/long 是资源类别而不是串行 lane。每轮 admission 在两类都有 backlog 时，先确保
   两类各获得一次机会，再按 `(反馈级别、类别 vruntime、预测完成时间、deadline、入队序号)`
   选择后续请求。
3. 每个类别维护 CFS 风格的 `virtual runtime`。一次 admission 增加该类的独占服务时间
   除以类别权重，类别权重可配置，因而可以在公平性和吞吐之间调节。
4. 长请求严格进入当前拓扑的最大 TP，短请求严格进入最小 TP。TP2 和 TP4 可以在同一轮
   同时持有 lease；MPS overlap 只影响完成时间预测，不改变实例的常驻和并发关系。

在本地精确 step 模拟中，主矩阵 80 次和对抗矩阵 72 次均通过。V9 在全部运行中完成率
为 1.0、`tp2_long_request_count` 为 0。它相对 naive 在 synthetic-5pct 和 ServeGen
中高负载通常提高 token goodput，但在强 SLO 目标下不保证优于 V6；这是 CFS/aging 公平性
和简单的 deadline/反事实控制之间的明确算法取舍。

## 2. 目标拓扑和硬不变量

实验使用普通 PD 路径，不启用 NIXL selector：

| 实例 | TP | GPU | 请求类别 |
|---|---:|---|---|
| `p01` | 2 | 0,1 | `seq_len <= 4000` |
| `p23` | 2 | 2,3 | `seq_len <= 4000` |
| `p0123` | 4 | 0,1,2,3 | `seq_len > 4000` |
| Decode | 4 | 独立 GPU | 模拟中无瓶颈 |

TP4 与两个 TP2 共享 GPU，step 模型根据 GPU placement 的交集施加 slowdown；两个 TP2
之间没有交集。完整拓扑下 V9 的安全不变量为：

```text
short: seq_len <= long_request_threshold -> min(available_tp)
long : seq_len >  long_request_threshold -> max(available_tp)
```

启动初期若只注册一种 TP，V9 会使用唯一可用 TP 以避免永久阻塞；因此线上应该等拓扑收齐
后再把“长请求不能进 TP2”作为监控断言。

## 3. V9 的调度算法

实现文件是
[`flex_tp_selector_v9.py`](/mtc/wusiyu/work/LightLLM-flex-tp/lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v9.py)。

### 3.1 状态

V9 自己定义 `V9Job`、`V9Lease`、`V9Instance` 和 `V9Profile`。每个 pending job 保存：

- `queued_at`、`base_level`、`level`：反馈队列和 aging 所需状态；
- `deadline`、`seq_len`、`enqueue_order`：EDF tie-break 和 SLO telemetry；
- `future`：与普通 PD 的异步 selector 接口一致。

每个 Prefill instance 保存 TP、GPU placement、generation、lease、ACK bundle 数、worker
report 和 `virtual_load`。selector 级别保存：

- `class_vruntime[short|long]`；
- `admitted_by_class`、`admitted_by_level`、`aging_promotions`；
- `credit_blocked`、`deadline_overrides` 和 `flow_decisions`。

### 3.2 MLFQ 初始级别和 aging

默认配置是 `interactive_token_limit=1024`、`long_request_threshold=4000`、
`aging_interval_s=0.150`：

```text
seq_len <= 1024 -> level 0
1024 < seq_len <= 4000 -> level 1
seq_len > 4000 -> level 2
```

每次 `_schedule_locked(now)` 先执行 aging。对 pending job：

```text
target_level = max(0, base_level - floor((now - queued_at) / aging_interval_s))
level = min(level, target_level)
```

因此 aging 只提升等待优先级，不改变 short/long 的 TP 路由。统计项
`aging_promotions` 记录累计提升级数。level 0 的短请求不再提升，长请求最多从 level 2
提升到 level 0。

### 3.3 类别公平和双 lane admission

每个类别内部按 `(level, deadline, enqueue_order)` 生成候选。实例选择先过滤到类别对应的
TP 池，再按以下键取最小者：

```text
(max(now, instance.virtual_load), admitted_tokens, worker_load, node_key)
```

预测完成时间使用本地 batch curve 和当前 overlap：

```text
F(i, r) = max(now, virtual_load_i)
          + slowdown(i, active_instances + i)
          * service(batch(existing leases + r), tp_i)
          + batch_window + prediction_margin
```

候选总排序键是：

```text
(job.level, class_vruntime[class], F(i,r), deadline, enqueue_order)
```

如果 short 和 long 都存在，当前 `_schedule_locked` 会记录 `opened_classes`，在同一轮先
让两类各 admission 一次，然后继续按上述键选择。这样双 lane 的并发是算法不变量，而不是
通过切换 `SHORT_ONLY/LONG_ONLY` 模式实现。

### 3.4 CFS virtual runtime

请求被 admission 到 TP 池后，使用单请求独占 service curve 更新类别 vruntime：

```text
class_vruntime[class] += exclusive_service(seq_len, tp) / class_weight[class]
```

默认 short/long 权重都是 1.0。降低某类权重会让它的 vruntime 增长更快，从而减少后续
选择频率；提高权重则给予该类更多服务。vruntime 是 admission fairness 信号，不是对
真实 GPU 时间片的抢占，已开始的 Prefill bundle 不会被 V9 中断。

### 3.5 credit、bundle 和 overload

默认每实例最多 64 个 in-flight lease、64 个已 ACK bundle，非空实例的 admitted token
credit 为 16384，bundle token cap 为 8192。空实例允许先接收单个超大请求；之后 credit
才会阻止继续堆积。默认 `overload_policy=best_effort`，预测超过 deadline 只增加
`deadline_overrides` 并继续工作守恒；`reject` 才会清理已过期 pending 并给 future 返回
异常。这样预测误差不会把整个 TP-SMT 拓扑锁死。

## 4. 普通 PD 生产接入

V9 已加入 selector 工厂、API 参数、启动参数类型和 PD master manager：

- selector 名称：`flex_tp_v9`；
- 启动包装器：
  [`start-cluster4_70b_p22.4d4_mps_flex_v9.sh`](/mtc/wusiyu/work/LightLLM-flex-tp/start-cluster4_70b_p22.4d4_mps_flex_v9.sh)；
- 参数：`--flex_tp_v9_aging_interval_ms`、`--flex_tp_v9_interactive_tokens`、
  `--flex_tp_v9_short_weight`、`--flex_tp_v9_long_weight`；
- 共享参数沿用各版本的 `--flex_tp_slo_ttft`、长请求阈值、bundle cap/window、MPS
  slowdown 和 overload policy。

生命周期与普通 PD 保持一致：selector 创建 job 并去重 req id；返回 Prefill/Decode 节点；
bundle ACK 写入 lease；worker report 只接受单调递增序号；完成/失败释放 lease 并 replan；
节点重注册时用 `instance_generation` 隔离旧控制消息。V9 设置
`supports_bundles=True` 和 `uses_prefill_lease_lifecycle=True`，因此继续使用生产的
`PrefillBundleDispatcher` 与 `ChunkedPrefillQueue`，实际 worker step 仍由 8192 chunked
pre-fill 配置决定。

## 5. 模拟器和测试方法

模拟入口是
[`flex_tp_step_sim.py`](/mtc/wusiyu/work/LightLLM-flex-tp/test/benchmark/service/flex_tp_step_sim.py)。
它用虚拟时钟替换 V3-V9 selector 的时间源，按到达、admission/replan、bundle flush、router
tick、worker report、Prefill step 完成和 Decode transfer 的顺序推进离散事件。Decode 默认
在 Prefill 完成后立即结束，因而只比较 Prefill/TP-SMT。

主矩阵入口是
[`flex_tp_v7_v8_comparison.py`](/mtc/wusiyu/work/LightLLM-flex-tp/test/benchmark/service/flex_tp_v7_v8_comparison.py)，
虽然文件名保留历史 V7/V8，当前策略集合已经是 `v3,v4,v5,v6,v7,v8,v9,naive`。负载为
phase-shift、synthetic-5pct（1000 请求、10 req/s）和 ServeGen `mm-image` 的 8/9/10
req/s，MPS slowdown 为 1.6/2.0，共 80 次运行。对抗入口
[`flex_tp_v7_v8_adversarial.py`](/mtc/wusiyu/work/LightLLM-flex-tp/test/benchmark/service/flex_tp_v7_v8_adversarial.py)
覆盖阈值边界、long-then-short 和 dual-lane-fair，slowdown 1.6/2.0/2.5，共 72 次运行。

## 6. 主矩阵结果

原始结果位于 `_/flex_tp_paper_analysis/v7-v9-comparison-v1/`，包含 JSON 和 CSV。下表是
ServeGen `mm-image`、10 req/s、slowdown=2.0 的重点压力点：

| 策略 | token goodput (token/s) | request goodput (req/s) | offered SLO | TTFT p95 (s) |
|---|---:|---:|---:|---:|
| V3 | 3005.2 | 2.078 | 0.2082 | 6.492 |
| V4 | 13664.7 | 8.961 | 0.8981 | 3.882 |
| V5 | 13264.4 | 8.628 | 0.8647 | 4.354 |
| V6 | 12736.2 | 8.867 | 0.8886 | 4.655 |
| V7 | 11565.4 | 7.333 | 0.7350 | 4.798 |
| V8 | 9693.9 | 5.783 | 0.5796 | 6.572 |
| **V9** | **10587.8** | **6.589** | **0.6604** | **5.235** |
| naive | 10109.0 | 5.844 | 0.5857 | 5.843 |

重点点上，V9 相对 naive 的 token goodput 为 `+4.7%`，SLO 提升 `+7.47 pp`，TTFT p95
降低 `0.608s`；V6 仍然更适合这个强 slowdown、强 deadline 的压力点。

全矩阵 80 次的简单平均如下。平均值用于概览，不能替代逐场景结果：

| 策略 | token goodput (token/s) | offered SLO | TTFT p95 (s) | 最小完成率 |
|---|---:|---:|---:|---:|
| V3 | 9287.6 | 0.7775 | 4.549 | 0.9538 |
| V4 | 11232.0 | 0.9294 | 2.932 | 0.9387 |
| V5 | 11119.6 | 0.9369 | 2.717 | 0.9339 |
| V6 | 10945.6 | 0.9399 | 6.841 | 1.0000 |
| V7 | **11864.3** | 0.9185 | 9.024 | 1.0000 |
| V8 | 10255.5 | 0.8661 | 7.499 | 1.0000 |
| **V9** | **10409.9** | **0.8840** | **7.170** | **1.0000** |
| naive | 10058.6 | 0.8638 | 7.387 | 1.0000 |

V9 平均比 naive 高 `3.5%` token goodput，SLO 高 `2.02 pp`，TTFT p95 低 `0.218s`。
V9 在 synthetic-5pct 两个 slowdown 下分别达到 9183.1/8556.3 token/s，高于 naive 的
8257.1/7564.6；ServeGen rate9、slowdown=2 下达到 11783.2 token/s 和 0.7633 SLO，
也优于 naive 的 11387.0 token/s 和 0.6934。phase-shift 中 V9 与 naive 接近，说明
aging/CFS 在低混合度的突发轨迹上没有制造额外收益。

所有 V9 主矩阵运行的 `tp2_long_request_count` 均为 0，且 `mps_overlap_wall_s` 非零，
证明 step 事件中 TP2/TP4 确实重叠活跃，而不是轮流切换。

## 7. 对抗矩阵结果

原始结果位于 `_/flex_tp_paper_analysis/v7-v9-adversarial-v1/`。以下是三个 trace 对三个
slowdown 取平均后的 V9、V6 和 naive 对照：

| 场景 | 策略 | offered SLO | token goodput | TTFT p95 | 最小完成率 |
|---|---|---:|---:|---:|---:|
| threshold-edge | V6 | 0.1181 | 60609.8 | 15.316 | 1.000 |
| threshold-edge | **V9** | **0.1181** | **60609.8** | **15.058** | 1.000 |
| threshold-edge | naive | 0.0833 | 45454.5 | 15.133 | 1.000 |
| long-then-short | V6 | 1.0000 | 75600.0 | 1.893 | 1.000 |
| long-then-short | **V9** | **1.0000** | **75600.0** | **1.227** | 1.000 |
| long-then-short | naive | 1.0000 | 75600.0 | 1.277 | 1.000 |
| dual-lane-fair | V6 | 0.6150 | 20238.5 | 20.343 | 1.000 |
| **dual-lane-fair** | **V9** | **0.3050** | **12252.6** | **20.433** | **1.000** |
| dual-lane-fair | naive | 0.2583 | 9794.5 | 19.995 | 1.000 |

dual-lane-fair 是 V9 的已知弱点：MLFQ 的“每类都要得到服务”和相同类别权重会主动保留
long lane 的 admission，强 slowdown 下牺牲了部分 SLO/吞吐。若部署更重视 short SLO，
可以提高 short weight、降低 long weight，或使用 V6 的反事实模式控制；这属于策略配置和
目标函数选择，不是 V9 的硬路由错误。

## 8. 验证和限制

已执行的验证：

```bash
python -m py_compile \
  lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v9.py \
  lightllm/server/httpserver_for_pd_master/pd_selector/__init__.py \
  lightllm/server/api_cli.py \
  lightllm/server/core/objs/start_args_type.py \
  lightllm/server/httpserver_for_pd_master/manager.py \
  test/benchmark/service/flex_tp_step_sim.py

python test/benchmark/service/flex_tp_step_sim.py --self-test
python test/benchmark/service/flex_tp_step_sim.py \
  --scheduler v9 --dataset synthetic-5pct --num-prompts 100 --rates 10 \
  --output-dir _/flex_tp_paper_analysis/v9-smoke

python test/benchmark/service/flex_tp_v7_v8_comparison.py \
  --output-dir _/flex_tp_paper_analysis/v7-v9-comparison-v1 \
  --slowdowns 1.6,2.0 --synthetic-prompts 1000 --seed 0

python test/benchmark/service/flex_tp_v7_v8_adversarial.py \
  --output-dir _/flex_tp_paper_analysis/v7-v9-adversarial-v1 \
  --slowdowns 1.6,2.0,2.5
```

V9 selector 契约测试在
[`test_flex_tp_selector_v9.py`](/mtc/wusiyu/work/LightLLM-flex-tp/test/test_flex_tp_selector_v9.py)，
覆盖双 lane 硬路由、反馈级别和 aging、重复 req id、乱序 report、generation-aware 完成、
工厂注册以及源码独立性。当前环境没有安装 pytest，因此该文件通过等价的 fake-marker
直接执行；模拟器自测和两套矩阵均正常结束。

这些数字是离散事件模型结果，不是 GPU 实测：Prefill service curve、MPS slowdown 和
worker report 负载仍需在目标模型上重新拟合；Decode 被刻意设为无瓶颈；真实八卡压测需要
GPU0-7 同时空闲。本次查询中 GPU0-3 空闲而 GPU4-7 被其他任务以 100% 利用率占用，所以
没有启动会干扰现有任务的实机压测。
