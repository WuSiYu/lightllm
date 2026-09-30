# 260903-FlexTP V3-V9：学术用途简要总结

> 260904 口径说明：本文早期表格中的 `fixed` 是已废弃的 static-partition 基线。当前固定基线名称和拓扑已改为 `fixed_tp2` 与 `fixed_tp4`，请以最新 rate/TTFT 报告为准。

本文把 V3-V9 看成同一个 TP-SMT 调度问题上的设计空间探索，而不是七个都应单独发表的
算法。目标拓扑为两个 TP2 Prefill 和一个共享 GPU 的 TP4 Prefill；短请求使用 TP2，长请求
使用 TP4，实例可以通过 MPS 同时运行。核心困难是：TP placement 造成非线性 slowdown，
而 Prefill 又希望 batching、吞吐、TTFT SLO 和公平性同时成立。

## 1. 版本演进

| 版本 | 核心机制 | 主要收益 | 主要问题 |
|---|---|---|---|
| V3 | 枚举 `request x instance`，预测完成时间、deadline 和 MPS 冲突 | 通用、可表达不同 TP 候选 | 长请求可能占用 TP2；候选搜索和 reservation 复杂，压力下容易过度保守 |
| V4 | 4000-token 硬类别路由：short 到最小 TP，long 到最大 TP | 消除 long 堵塞 TP2 的安全问题 | 仍沿用逐请求候选预测，资源弹性有限 |
| V5 | V4 路由 + MPS-first；只有会破坏 deadline/已有 lease 时才避免重叠 | 兼顾重叠吞吐和已有工作保护 | 规则较多，参数与边界行为不够统一 |
| V6 | 比较 `ALL`、short-first、long-first 三个反事实；严格有收益才临时串行 | 强 deadline/高 slowdown 下 SLO 稳定，决策面较小 | 类别排空会牺牲并发吞吐，依赖 service curve 和 utility 权重 |
| V7 | 虚拟完成时间、GPU 冲突价格、class debt、urgency | 连续的在线优化，平均 token goodput 最高 | tail TTFT 较长；价格、debt、预测误差需要校准 |
| V8 | 固定 epoch、short/long 双 lane、deficit quota、lane 内 EDF | 批次边界和公平性容易解释、复现和审计 | epoch 带来等待，吞吐通常低于 V7 |
| V9 | MLFQ aging + CFS virtual runtime + 双 lane admission | OS-inspired、公平性和抗饥饿语义清晰 | 相同权重会保护 long lane；强 slowdown 下可能牺牲 short SLO |

共同安全不变量是：完整拓扑注册后，`seq_len <= 4000` 只能进入 TP2，`seq_len > 4000`
只能进入 TP4；已发出的 Prefill lease 不可抢占；bundle/credit 只限制近期 admission，
worker 仍负责真正的 chunked-Prefill batching。

## 2. 模拟结果

结果来自 step 精确事件模拟器，包含普通 PD 生命周期、bundle、8192-token chunk、MPS
placement slowdown 和 fake decode 时间模型。以下是 ServeGen `mm-image`、10 req/s、
slowdown=2.0 的压力点；token goodput 是按时完成的 input token/s。

| 策略 | token goodput | offered SLO | TTFT p95 |
|---|---:|---:|---:|
| V3 | 3005.2 | 0.2082 | 6.492 s |
| V4 | 13664.7 | 0.8981 | 3.882 s |
| V5 | 13264.4 | 0.8647 | 4.354 s |
| V6 | 12736.2 | 0.8886 | 4.655 s |
| V7 | 11565.4 | 0.7350 | 4.798 s |
| V8 | 9693.9 | 0.5796 | 6.572 s |
| V9 | 10587.8 | 0.6604 | 5.235 s |
| naive | 10109.0 | 0.5857 | 5.843 s |

主矩阵 80 次运行的简单平均：

| 策略 | token goodput | offered SLO | TTFT p95 |
|---|---:|---:|---:|
| V3 | 9287.6 | 0.7775 | 4.549 s |
| V4 | 11232.0 | 0.9294 | 2.932 s |
| V5 | 11119.6 | 0.9369 | 2.717 s |
| V6 | 10945.6 | **0.9399** | 6.841 s |
| V7 | **11864.3** | 0.9185 | 9.024 s |
| V8 | 10255.5 | 0.8661 | 7.499 s |
| V9 | 10409.9 | 0.8840 | 7.170 s |
| naive | 10058.6 | 0.8638 | 7.387 s |

解读应保持克制：V7 以吞吐为主目标，V6 以 deadline/SLO 稳定性见长，V4/V5 在某些中高压
点优于 V6；V9 相对 naive 平均 token goodput 约 `+3.5%`，但不是全指标最优。对抗 trace
还显示 V9 的 dual-lane 公平会在强 slowdown 下保护 long，V6 更适合 short SLO 优先的
配置。完整结果见
[`260902-FlexTP_v9_os_smt_design_and_comparison.md`](/mtc/wusiyu/work/LightLLM-flex-tp/260902-FlexTP_v9_os_smt_design_and_comparison.md)。

## 3. H200 实机交叉检查

> **状态：作废（`pd_fake_decode`）。** 本节 H200 实机交叉检查使用 production fake-decode 服务，结果、表格和引用的 `fake-decode-live-v5-v9-260903` 数据全部作废。前面的离线模拟器结果不属于该服务实测。

生产 fake-decode 驱动在 4 张 H200 上使用相同 trace、精确 tokenizer、每策略 24 个预热请求，
正式负载为 ServeGen 8 req/s 和 synthetic-5pct 8 req/s。所有策略正式请求完成率为 100%，
但该负载尚未进入明显拥塞区：

| 策略 | ServeGen token/s | ServeGen TTFT p95 | Synthetic token/s | Synthetic TTFT p95 |
|---|---:|---:|---:|---:|
| fixed | 12569.9 | 0.949 s | 6096.7 | 0.244 s |
| naive | 12612.5 | 0.957 s | 6090.8 | 0.238 s |
| V5 | 12593.5 | 1.062 s | 6096.4 | 0.276 s |
| V6 | 12543.4 | 0.996 s | 6094.7 | 0.413 s |
| V7 | 12598.9 | 0.975 s | 6098.6 | 0.264 s |
| V8 | 12591.5 | 1.079 s | 6095.8 | 0.283 s |
| V9 | 12584.1 | 0.987 s | 6097.4 | 0.286 s |

实机结果的正确结论是“生产协议、TP2/TP4 overlap 和 selector 切换可运行”，而不是某个
版本已经显著优于其他版本。完整 JSON 在
[`fake-decode-live-v5-v9-260903/results.json`](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/fake-decode-live-v5-v9-260903/results.json)。

## 4. ASPLOS/LLM Infra 投稿建议

### 4.1 论文主张

建议把论文主线收敛为：

> **TP-SMT-aware admission scheduling：在共享 GPU 上同时服务多个 TP 专用 Prefill
> 实例，并在非线性 placement slowdown 下联合优化 batching、on-time token goodput、
> TTFT SLO 和公平性。**

不要把“V3、V4、V5、V6、V7、V8、V9”作为七个贡献点。它们更适合作为设计空间、消融和
失败案例：V3 暴露错误 TP 候选的安全问题，V4/V5 展示硬路由和 MPS 保护，V6/V7/V8/V9
分别代表反事实控制、连续价格、离散 quota 和 OS 公平性。

### 4.2 主算法选择

- 若论文首要指标是吞吐，建议以 **V7** 为主算法：它的在线虚拟完成时间和冲突价格最容易
  形成明确的 optimization story；V6 用作 SLO-oriented 强基线，V9 用作简单公平性变体。
- 若论文首要指标是 tail SLO，建议以 **V6** 为主算法：`ALL/SHORT_ONLY/LONG_ONLY`
  反事实和非抢占 drain 更容易给出清晰的 deadline 语义；V7 作为高吞吐对照。
- 不建议当前直接以 V9 作为唯一主贡献。MLFQ/CFS 结构很易解释，但当前相对 naive 的收益
  较小，且 dual-lane-fair 暴露了明确的目标函数弱点。除非补充正式公平性指标和权重自适应，
  V9 更适合作为 OS-inspired ablation。

### 4.3 必须补强的实验

1. **进入饱和区。** 扫描到 SLO 明显下降的到达率，而不是只使用实机 8 req/s；报告吞吐、
   offered/on-time SLO、TTFT/queue p50/p95/p99、完成率和最大等待时间。
2. **重复与统计。** 至少 5 个 workload seed，固定 warmup、固定 commit，给出置信区间或
   bootstrap 区间；service curve 和 slowdown 参数必须在训练 trace 拟合、在 held-out trace
   测试。
3. **消融。** 分别关闭 hard routing、bundle batching、MPS conflict price、deadline urgency、
   aging、virtual runtime、epoch quota；这样才能证明收益来自算法而不是参数组合。
4. **强基线。** 除 naive 和 fixed 外，加入 EDF/SJF、MPS-off 串行 oracle、长度阈值 oracle、
   只做 batching 的 baseline，以及 V6/V7/V9 的同参数版本。
5. **真实系统因素。** 实测 KV bytes transfer/bandwidth、Decode 非零负载、不同模型规模和
   TP placement；fake decode 只能作为 Prefill admission 的隔离实验。
6. **系统开销与鲁棒性。** 报告 selector CPU 时间、每次 replan 延迟、bundle 数、GPU 利用率、
   节点重注册/断连、请求取消和预测误差；证明 bounded admission 不会造成死锁或旧 generation
   消息污染新实例。

### 4.4 推荐论文结构

```text
问题与 TP-SMT placement slowdown
  -> 安全硬路由与非抢占 lease 模型
  -> 一个主算法（V7 或 V6）
  -> V3/V4/V5/V8/V9 作为设计空间与消融
  -> 离散事件模拟 + 饱和区真实 H200 实验
  -> 吞吐/SLO/公平性三维结果和失败案例
```

当前最稳妥的结论是：V7 提供最高潜在吞吐，V6 提供最强 deadline 稳定性，V9 提供最清晰的
OS 公平性解释；论文应展示这种可重复的 trade-off，而不是宣称存在一个在所有 workload
上都最优的调度器。
