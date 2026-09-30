# FlexTP V3 与 naive MPS：问题诊断、算法方向和论文实验设计

## 1. 结论

当前 V3 不能宣称在所有负载上优于 4K 阈值 naive，也不能以整体 p95 TTFT
作为主要胜出指标。现有证据说明：

1. 在稳定、长度分布固定且 4K 阈值恰好合适的流量中，naive 已经接近该负载的静态最优；
   V3 多出的 bundle window、credit 和预测控制只会增加少量开销。
2. naive 虽然在 master 一次发送一个请求，worker 的 `ChunkedPrefillQueue` 仍会把等待请求组成
   batch。因此“ours 支持 batching、naive 不支持”不是公平或真实的区别。
3. 当前模拟器把两个重叠 MPS 进程建模成 slowdown `s`。两个进程的聚合服务率约为
   `2/s`：`s < 2` 时盲目并发本身就有吞吐收益，`s = 2` 时近似 work-conserving，只有
   `s > 2` 或 deadline/优先级不对称时，智能 gating 才有明确优势。
4. V3 当前最大的算法缺口是“全局 EDF + 可行时优先最小 TP”的逐请求贪心。它没有优化一个
   horizon 内能按时完成的请求/token 总量，会让旧长请求占用 TP2 并阻塞后续短请求。
5. 因此论文的正确主张应是：在相同 SLO 下，动态 TP、batch 和 MPS mode-aware admission
   提高 non-stationary/heterogeneous workload 的 on-time goodput 或降低 GPU 成本；在稳定
   单峰流量中接近事后最优静态阈值。不能声称所有流量上 p95 都更低。

当前版本还没有达到上述论文主张。模拟器可用于暴露机制和筛选改动，但最终数字必须来自
同一代码、同一 trace、同一 8xH200 拓扑的普通 PD 实机实验。

## 2. 为什么 naive 的缺点不明显

### 2.1 测试分布恰好奖励 4K 阈值

ServeGen mm-image 的输入 p95 约为 4.2K--4.5K，4K 阈值正好把大约尾部 5%--10%
送到 TP4，其余请求分摊到两个 TP2。若负载稳定且两个池都未分别过载，这就是一个很强的
静态策略，而不是刻意设置的弱 baseline。

synthetic-5pct 中长请求比例恰好是 5%。整体 p95 正好落在短/长请求的分界处，可能同时掩盖
短请求的大面积改善和长请求的严重退化。必须分别报告长度 bucket 的 p95/p99、SLO 和
on-time token goodput。

### 2.2 naive 也获得了 worker batching

naive 的 master 每次只选择一个 instance，但请求到达 worker 后进入同一个 waiting queue。
`ChunkedPrefillQueue.generate_new_batch()` 会扫描多个请求，累计原始 `input_len` 超过 6000
token 后才停止；`batch_max_tokens` 和 8192 chunked prefill 还会继续约束真实执行。

因此 ours 的 bundle 主要提供全局可见的 lease、边界和 admission，不天然带来 worker
batching 的独占收益。固定 10 rps 的到达间隔为 100ms，也远大于 20ms bundle window，
master bundle 很少能合并相邻请求。

### 2.3 当前模拟器过度善待无界 inflight

naive 可以持续把请求压入 worker。本模拟器中更大的 local batch 会摊薄固定开销，却没有
完整建模大 waiting queue 的内存压力、CPU 调度、通信排队、p99 抖动和失败。因此无界 inflight
在模型中可能同时得到“更大 batch”和“没有代价”两项不现实的奖励。

已有真实 naive 日志在 ServeGen 10 rps 时为 p95 TTFT 9.84s、完成吞吐约 7.9 req/s；同一
配置的模拟结果约为 p95 5.84s 且完成全部 offered request。这个差距说明模拟的 batch/MPS
surface 不能作为论文结果。

### 2.4 MPS 模型决定了 naive 是否存在结构性缺点

在 synthetic-5pct、10 rps、1000 请求的敏感性实验中：

| overlap slowdown | ours/naive p95 TTFT | ours/naive on-time input tok/s | ours 相对 token goodput | ours/naive GPU-s / on-time req |
|---:|---:|---:|---:|---:|
| 1.3 | 1.685 / 0.302 | 9906 / 9415 | +5.2% | 0.331 / 0.326 |
| 2.0 | 1.736 / 0.774 | 9590 / 7555 | +26.9% | 0.338 / 0.374 |
| 2.5 | 1.976 / 1.468 | 9571 / 5754 | +66.3% | 0.342 / 0.404 |

这组结果不能证明真实 H200 收益，但说明指标和假设的作用：ours 的整体 p95 始终更差；随着
重叠干扰增加，ours 的 SLO goodput 和单位有效请求成本才明显占优。如果实测 H200 的
TP2+TP4 slowdown 接近 1.3，论文不能依靠“减少 MPS 干扰”作为主要收益来源。

### 2.5 V3 的排序目标与“降低 p95”不一致

V3 的候选排序首先按 request deadline/FIFO，然后在可行候选中先比较 `tp_size`，最后才比较
`predicted_finish`。这会有意用 SLO slack 换 TP2 资源效率。只要仍在 3s deadline 内，这个
策略就没有优化 p50/p95 的动机。

如果论文主要画整体 p95，naive 会因为长请求直接 TP4、短请求直接 TP2 而显得更好。若目标是
资源效率，则必须同时画 SLO-goodput/GPU-cost Pareto，不能只画 p95。

### 2.6 predictor 与 worker 的 chunk/batch 语义不一致

V3 `_packed_batches()` 以完整 `seq_len` 在 8192 token envelope 中打包。worker 则先按原始
`input_len > 6000` 停止扫描，再在实际执行中处理 8192 chunk 和剩余 chunk。对 8K、20K
请求，master 预测的批边界、执行轮数和完成顺序都可能错误。

模拟自检中 V3 的预测低估比例达到 95.2%，p95 预测误差约 1.19s。这个误差已经与 3s SLO
同量级，deadline feasibility 不能据此可靠决定 TP2/TP4。

## 3. 穷举阈值 oracle 揭示的当前算法缺口

机制实验对每个 trace 的所有长度断点穷举 naive threshold，然后事后选择 token goodput 或
request goodput 最优者。这比只比较 4K threshold 更强，也更适合论文审稿。

| workload | slowdown | ours 相对 oracle token goodput | ours 相对 oracle request goodput | oracle |
|---|---:|---:|---:|---|
| threshold-cliff | 1.6 / 2.0 / 2.5 | -0.2% | -0.2% | all TP2 |
| phase-shift | 1.6 | -28.1% | -11.2% | threshold 4100 |
| phase-shift | 2.0 | -16.5% | -3.6% | threshold 4100 |
| phase-shift | 2.5 | +13.8% | +5.3% | token: all TP2; request: threshold 1024 |
| overlap-stress | 1.6 / 2.0 / 2.5 | approximately 0% | approximately 0% | workload-dependent |

phase-shift 依次包含 20s 短请求、20s 8K 请求和 20s 混合请求。slowdown=1.6 时逐阶段结果为：

| policy/source | offered | completed | 3s SLO | p95 TTFT | completed TP placement |
|---|---:|---:|---:|---:|---|
| ours / short phase | 400 | 400 | 100% | 0.232s | TP2: 400 |
| ours / long phase | 60 | 52 | 18.3% | 6.613s | TP2: 34, TP4: 18 |
| ours / mixed phase | 160 | 155 | 0% | 6.185s | TP2: 136, TP4: 19 |
| naive-4100 / short phase | 400 | 400 | 100% | 0.208s | TP2: 400 |
| naive-4100 / long phase | 60 | 60 | 15.0% | 22.681s | TP4: 60 |
| naive-4100 / mixed phase | 160 | 160 | 86.9% | 20.679s | TP2: 139, TP4: 21 |

naive-4100 的长请求尾延迟很差，但 TP4 长队列与 TP2 短队列相互隔离，后来的短请求仍能按时
完成。V3 则把 34 个 8K 请求送到 TP2，并严格优先旧 deadline；这些请求形成 head-of-line
blocking，使 mixed phase 的按时完成数为零。

这说明逐请求判断“该请求单独在 TP2 上可满足 SLO”是不够的。调度器需要优化一个 horizon
内的总 on-time value，并显式计算把长请求放入 TP2 对未来短请求的机会成本。

## 4. 如何修改算法才有机会稳定胜过 naive

### 4.1 先把优化问题改对

建议使用词典序目标：

1. 最大化 offered on-time request goodput 和 input-token goodput；
2. 在 goodput 近似相同的方案中最小化 physical GPU-seconds；
3. 在前两项近似相同时，再最小化 p95/p99 TTFT 和控制开销。

不能先固定 FIFO request，再为这一个请求选最小 TP。应在 0.5--3s rolling horizon 中，对
`request x placement x MPS mode x batch` 联合估值，近似求解 deadline-aware knapsack。

### 4.2 消除长请求造成的全局队首阻塞

最低可行实现不必立即上复杂求解器，可以采用：

1. 按剩余 prefill work/slack 分 class queue，class 内 EDF；
2. 为 TP2 short pool 和 TP4 long/urgent pool维护滚动 demand；
3. 若长请求放 TP2 会导致 horizon 内更多短请求 miss，即使它自身在 TP2 上“可行”，也升级
   到 TP4 或等待 TP4 reservation；
4. 已经不可能按时的请求不能继续以最老 deadline 阻塞可按时请求，应明确 reject/best-effort
   降级，并在 offered SLO 分母中保留；
5. 使用 aging 或最小服务份额避免为了 request goodput 永久牺牲长请求。

一个可实现的候选 value 是：

```text
value(candidate) =
    newly_on_time_requests * W_req
  + newly_on_time_input_tokens * W_tok
  - displaced_on_time_value
  - lambda_gpu * physical_gpu_seconds
  - lambda_tail * predicted_lateness
```

`tp_size` 只能作为 value 接近时的 tie-breaker，不能排在 `predicted_finish` 和跨请求机会成本之前。

### 4.3 用 worker report 校准 chunk-aware predictor

worker ACK 至少要返回：waiting/running 请求、每个请求 remaining prefill tokens、下一 chunk、
真实 batch token/request 数、queue wait、execution time、instance generation 和 active MPS mode。

预测器按真实执行单位展开长请求：

```text
20K request -> 8192 + 8192 + 3616 residual chunks
```

batch profile 应按 `(TP, batch request bucket, first-chunk token bucket, residual/context bucket)`
查 p50/p95/p99，不再用完整长度一次性塞入 8192 envelope。

### 4.4 MPS 必须按实测 mode gating

对 TP2、TP4 及 `TP2+TP2`、`TP2+TP4`、`TP2+TP2+TP4`，测量同 batch bucket 下：

```text
mode_gain = on_time_goodput(mode) / on_time_goodput(best serialized schedule)
```

只有 `lower_confidence_bound(mode_gain) > 1 + epsilon` 且不会破坏已有 lease 的 p99 deadline
时才允许 overlap。未知 bucket 默认不重叠。若实测 slowdown 约等于 2，MPS 只是时间分享，
算法优势应来自隔离和 deadline ordering；若明显小于 2，应该保留并发而不是为了降低 overlap
时间强行串行化。

### 4.5 bundle window 应随负载和 slack 自适应

已有模拟中 0/5/20/50ms window 的 p95 分别约 1.755/1.731/1.736/2.085s，5ms 最好但差异
不大。建议：

- 空载或下一到达无法预测时立即发送；
- backlog 已足够填充 batch 时立即发送；
- 只有预计短时间能显著提高 fill，且最紧 deadline 有余量时等待；
- `wait <= min(profiled_fill_wait, deadline_slack - p99_service - margin)`。

这能把稳定低载下相对 naive 的约 20ms 固定税降到接近零。

## 5. 现有实机日志能说明什么

仓库中已有 2026-06-12 的 Flex 运行和 2026-04-09 的 naive-4K ServeGen 运行。它们不是同一
次启动、同一 commit 的严格 A/B，因此只能用于诊断，不能直接放入论文主表。

| rate | Flex p95 / 3s SLO / on-time input tok/s | naive-4K p95 / 3s SLO / on-time input tok/s |
|---:|---:|---:|
| 3 | 0.869s / 100% / 5223 | 0.787s / 100% / 5223 |
| 4 | 1.185s / 100% / 6786 | 1.138s / 100% / 6786 |
| 5 | 1.332--1.527s / 99.78--99.89% / 8438--8447 | 1.197s / 100% / 8454 |
| 7 | 1.906--2.109s / 99.60--99.68% / 11851--11870 | 1.475s / 99.92% / 11780 |

低于饱和点时两者 goodput 基本相同，naive p95 更低，这与上述分析一致。naive 在 8/9/10 rps
开始明显失效，但现有 Flex 目录没有同配置的 8--10 rps 数据。论文实验必须覆盖饱和点两侧，
否则只会测出控制开销，测不到动态 admission 的价值。

## 6. 论文应该如何证明更优

### 6.1 主 claim

建议主 claim 为：

> 在共享 GPU 上同时驻留不同 TP 的 disaggregated Prefill 实例时，FlexTP 通过 chunk-aware
> two-level admission、deadline-aware dynamic TP placement 和 measured MPS-mode gating，
> 在 workload shift 与长度异构下提高 SLO goodput，并在同 goodput 下减少 GPU-seconds；
> 对稳定流量接近事后最优静态长度阈值。

不要使用“始终降低平均/p95 TTFT”“MPS overlap 越多越好”或“naive 不 batching”作为 claim。

### 6.2 公平 baseline

所有 baseline 使用同一 8xH200、普通 PD 路径、模型、权重共享、worker queue、chunk size、
batch limit、arrival trace 和 decode 配置：

1. TP2-only；
2. TP4-only；
3. naive threshold sweep：all-TP4、所有 trace 长度断点、4K、all-TP2；
4. 每个 trace 事后选择的 oracle static threshold；
5. Flex no-MPS；
6. Flex measured-MPS；
7. Flex 各组件 ablation。

只比较 4K naive 不够，因为审稿人可以质疑阈值未调优；oracle static threshold 是必须跨过的
强 baseline。若无法在平稳 trace 上胜过 oracle，应证明在 3% 以内且 phase/adversarial trace
明显胜出。

### 6.3 workload

至少覆盖：ServeGen mm-image 原始 trace；5% 长请求合成 trace；长度比例 0/1/5/10/20%；
阈值附近 3900/4100；短 burst；短到长到混合 phase shift；20K 长请求后跟短 burst；同平均
rate 但不同 burstiness；以及 arrival order shuffle。

每类做从低载到过载的 rate sweep，并细化最大可持续 rate 附近。最大可持续 rate 定义为
offered SLO attainment 不低于目标且失败/reject 计入分母时的最大 offered load。

### 6.4 主指标和图

主指标：offered SLO attainment、on-time request goodput、on-time input-token goodput、p99
TTFT、physical GPU-s/on-time request、physical GPU-s/on-time input token。辅助指标：长度 bucket
p50/p95/p99、mean/max batch、batch tokens、queue wait、TP2/TP4 share、各 MPS mode 驻留时间、
预测误差、reject/failure。

建议主图：

1. offered rate--SLO goodput 曲线和每个策略的最大可持续 rate；
2. on-time token goodput--GPU cost Pareto；
3. phase workload 的到达组成、queue、TP placement、mode 和 SLO 时间序列；
4. length bucket 的 TTFT/SLO；
5. MPS mode x batch bucket 的实测 gain/slowdown heatmap；
6. oracle threshold regret：动态策略相对每个 trace 事后最优静态策略的差值。

每个点至少 5 个 seed，报告置信区间；A/B 使用完全相同的生成 trace，而不是只使用相同分布。

### 6.5 关键 ablation

逐项移除：master bundle；worker feedback；chunk-aware predictor；horizon opportunity-cost；
TP4 future reservation；adaptive bundle window；MPS gating；admission/reject。这样才能回答收益
来自 batching、动态 TP、干扰控制还是过载丢弃，而不是把所有变化归为一个黑盒 scheduler。

### 6.6 验收门槛

在正式实机 sweep 前建议设定：

1. 稳定单峰 trace 的 on-time goodput 不低于 oracle static threshold 的 97%；
2. 未过载时 offered SLO attainment 不低于 99%，且 p99 不因控制开销回退超过 5%；
3. phase/adversarial trace 的 on-time token goodput至少高于 oracle static threshold 10%；
4. 同 SLO/goodput 下 GPU-s/on-time token 至少降低 10%；
5. Flex MPS 只有同时优于 Flex no-MPS 的 goodput 且 p99 不退化，才记为 MPS 收益；
6. 模型 p95 预测误差必须显著小于 SLO margin，建议低于 10%。

当前模拟结果只在 super-linear interference (`slowdown=2.5`) 的 phase-shift 上满足第 3 项；
这意味着应先修 horizon placement/HOL blocking，再启动论文主实验，而不是继续调整 4K 阈值。

## 7. 与相关工作的区分

相关系统通常使用 SLO goodput 而不是单独的平均延迟评价：[DistServe](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
讨论 disaggregated serving 下的 SLO-attainable rate；[Llumnix](https://www.usenix.org/conference/osdi24/presentation/sun-biao)
强调动态异构负载和尾延迟；[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)
证明 chunked prefill/batching 的作用；[MuxServe](https://arxiv.org/abs/2404.02015) 和
[LoongServe](https://arxiv.org/abs/2404.09526) 涉及共享/弹性并行。
[Libra](https://www.usenix.org/conference/nsdi26/presentation/ruan-libra) 的两级动态调度与本
工作尤其接近，是必须正面比较和阐明差异的工作。

可 defend 的差异应限定为：LightLLM 普通 PD 路径中，共享权重、同时驻留且物理 GPU 重叠的
TP2/TP4 Prefill instances；全局 lease/deadline 与本地 chunked queue 协同；根据实测 active
MPS mode 和 batch bucket 做 admission。若没有 mode-aware 实测和 worker feedback，单纯“按
长度动态选 TP”很难构成足够强的算法贡献。

## 8. 已生成的可复现实验产物

- `test/benchmark/service/flex_tp_step_sim.py`：直接加载生产 V3/naive selector、bundle
  dispatcher 和 instance queue 的 step-level mock simulator；decode 默认无瓶颈。
- `test/benchmark/service/flex_tp_paper_experiments.py`：生成 threshold-cliff、phase-shift、
  overlap-stress，并对每个 workload 的全部阈值断点构造 post-hoc oracle。
- `_/flex_tp_paper_analysis/mechanisms-v3/`：当前机制实验的逐策略、分阶段结果和 oracle 对比。该目录的
  数值是模型结果，不是 GPU 测量。
