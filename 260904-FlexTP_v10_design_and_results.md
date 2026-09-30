260904-FlexTP V10：EEVDF/EDF 与非抢占 slack reservation

> 260904 口径说明：本文早期实验中的 `fixed` 是旧的 static-partition 基线。当前模拟器已改用 `fixed_tp2`（仅两个 TP2）和 `fixed_tp4`（仅一个 TP4），本文历史结果不代表新固定基线。

## 1. 动机

前一轮 v3-v9 扫描暴露了三个相互冲突的目标：

- V4/V5 的已完成请求 p50/p99 较低，但过载时会牺牲完成率；
- V6 的 all-mode/counterfactual 控制能提高 offered SLO，但 p99 仍可能被长请求拖高；
- V9 的 MLFQ+CFS 易解释且保证公平，但固定的 short/long 双 lane 机会会在强 MPS slowdown
  下保护 long lane，不能稳定提高 SLO。

V10 采用传统操作系统中的 EEVDF（Earliest Eligible Virtual Deadline First）和 EDF/LSTF
 思路，目标是用一个小的、可审计的运行队列同时处理公平性、deadline 和共享 GPU 冲突：

1. 完整拓扑下仍使用硬路由：`seq_len <= 4000` 进入 TP2，`seq_len > 4000` 进入 TP4。
2. 所有 pending 请求按 real deadline eligibility 排序；virtual service finish 只解决相近
   deadline 的公平性，不再用 V9 的 MLFQ level 强制优先短请求。
3. 新 lease 若与另一个 TP 实例共享 GPU，只在双方保留足够 laxity 时 overlap；已预测过期的
   lease 不再被 guard 阻塞，避免为无法挽救的请求制造 head-of-line blocking。
4. Prefill lease 非抢占，bundle、ACK、worker report、generation 和 completion 生命周期沿用
   已验证的普通 PD 实现。

## 2. 实现

实现文件：

- selector：[flex_tp_selector_v10.py](/mtc/wusiyu/work/LightLLM-flex-tp/lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v10.py)
- step 模拟器：[flex_tp_step_sim.py](/mtc/wusiyu/work/LightLLM-flex-tp/test/benchmark/service/flex_tp_step_sim.py)
- 扫描入口：[flex_tp_rate_ttft_sweep.py](/mtc/wusiyu/work/LightLLM-flex-tp/test/benchmark/service/flex_tp_rate_ttft_sweep.py)
- 4-GPU fake-decode 包装器：[start-cluster3_70b_p22.4d4_mps_prefill_fake_decode_v10.sh](/mtc/wusiyu/work/LightLLM-flex-tp/start-cluster3_70b_p22.4d4_mps_prefill_fake_decode_v10.sh)

V10 只复用 V9 的 selector 生命周期和拓扑 bookkeeping，调度核心由以下状态组成：

```text
class_vtime[short|long]       每类累计的 virtual service finish
short_weight/long_weight      EEVDF 类别服务权重
slack_weight                  laxity tie-break 权重
overlap_slack_ratio           共享 GPU reservation 占 target TTFT 的比例
```

对每个候选请求，先选择其硬路由 TP 池和最小预测负载 instance，然后计算：

```text
service       = profile.predict_batch([seq_len], tp)
vfinish       = max(class_vtime[class], min(class_vtime)) + service / weight
predicted     = non-preemptive batch finish with current MPS placement
lateness      = predicted - real_deadline
normalized_laxity = (real_deadline - predicted) / max(target_ttft, service)
```

候选排序为：

```text
(predicted_miss, real_deadline, urgency_key, vfinish, predicted, enqueue_order)
```

其中 `urgency_key` 在 feasible 请求中为 normalized laxity，在 miss 请求中为正的 lateness；
因此 real deadline 是第一优先级，virtual finish 只在同 deadline/近 deadline 请求之间提供
可解释的公平 tie-break。admission 后只更新被选类别的 `class_vtime`。

共享 GPU reservation 的规则是：若现有重叠 lease 的剩余 slack 已小于
`target_ttft * overlap_slack_ratio`，只有新请求更紧急时才允许 overlap；若现有 lease 已经
预测过期，则恢复 work-conserving overlap，因为串行等待不能挽救它。

## 3. 模拟配置

主扫描使用普通 PD 离散事件模拟，不使用 GPU：

- dataset：ServeGen `mm-image`（60 s constant-rate）和 synthetic-5pct（1000 请求、5% 长请求）；
- request rate：0.5、1、2、4、6、8、10、12、16、20、24 req/s；
- target TTFT：0.5、1、2、4 s；
- scheduler：fixed、naive、V3、V4、V5、V6、V7、V8、V9、V10，共 880 runs；
- chunked prefill=8192，batch max tokens=16384，bundle cap=8192、trigger=4096、window=20 ms；
- MPS overlap slowdown=2.0，fake decode 固定 KV transfer=20 ms、per-token transfer=0；
- `max_inflight=64`、instance token credit=16384、prediction margin=80 ms、seed=0。

`ttft_p50_s/p99_s` 只在完成请求上统计；所有拒绝和未完成请求都进入
`completion_fraction` 和 `offered_ttft_slo_attainment` 分母。高负载结果必须联合这三个指标
解释，不能用较低的 survivorship-biased p99 直接宣称更好。

## 4. 调参过程

初版只用 virtual deadline tie-break，结果与 V9 在常规到达率下几乎相同，因为 TP2/TP4 有
独立 token credit，二者通常可以直接并行 admission。随后加入两项变化：

- 同一类别允许在 pending 集合中比较候选，但最终 real deadline 仍优先，防止 SJF 式长请求
  饥饿和 ServeGen p99 爆炸；
- 加入 bounded overlap reservation。第一版把已过期 lease 也挡在 guard 后，导致 target=1 s
  burst trace 过度串行化；修正为“已过期 lease 直接恢复 overlap”后，低/中负载保持
  work-conserving，高负载才有 reservation 效果。

当前默认 `overlap_slack_ratio=0.10`。在 400--500 请求 synthetic trace 的 ratio=0/0.05/0.1/0.2/0.3
扫描中，0.1 在 SLO、p99 和 guard 次数之间最均衡；0.3 已出现明显串行化和 p50 退化。

在最终 fingerprint 上又做了针对性的策略消融：least-slack、纯 virtual-finish、
service-aware EDF、V6 式串行 mode reservation，以及“candidate 在独占模式可按时则禁止
overlap”。mode reservation 在 synthetic-5pct、target=4 s、rate=20 req/s 可将 offered
SLO 从约 0.37 提升到约 0.97，但在 ServeGen 的 target=1--2 s 区间会把 SLO 和 P99 显著
拉坏；candidate rescue 也有相同的 workload 敏感性。故默认 V10 保持单一 work-conserving
run queue，只使用 bounded reservation，不把针对某一 trace 的串行 gate 混入生产策略。

## 5. 全量结果

最终结果由完整的 V3--V9 基线（792 行）和当前代码的 V10 重放（88 行）合并得到；两批
实验逐点检查了 trace fingerprint，88/88 个 workload point 一致。结果目录为：

`_/flex_tp_paper_analysis/rate-ttft-sweep-v10-final-260904/`

目录包含 `rate_ttft_sweep.csv/json`、每个数据集的 TTFT P50/P99 线性曲线和 log-y 曲线。
下面的“最大 rate”定义为采样点中 `offered_ttft_slo_attainment >= 0.90` 的最大 offered
rate；它不是插值容量。

| workload | target | V6 | V9 | V10 | naive |
|---|---:|---:|---:|---:|---:|
| ServeGen mm-image | 1 s | 2 | 2 | 2 | 2 |
| ServeGen mm-image | 2 s | 6 | 6 | 6 | 6 |
| ServeGen mm-image | 4 s | 12 | 8 | 8 | 8 |
| synthetic-5pct | 1 s | 10 | 12 | 12 | 12 |
| synthetic-5pct | 2 s | 20 | 12 | 12 | 12 |
| synthetic-5pct | 4 s | 24 | 16 | 16 | 16 |

V10 的主要收益是尾延迟形状而非采样容量普遍提升。例如 synthetic-5pct 在 target=4 s
时，rate=8/12/16 req/s 的 `(P50, P99, offered SLO)` 分别为
`(0.203, 2.230, .998)`、`(0.608, 3.177, .994)`、`(1.532, 13.159, .979)`；V9 对应为
`(0.242, 3.346, .993)`、`(0.594, 4.142, .987)`、`(1.480, 14.037, .963)`。
在 ServeGen 的 rate=12、target=4 s，V10 将 P99 从 40.865 s 降到 34.570 s，SLO 从
0.335 提升到 0.351；但 rate=16 时 P50 会从 9.444 s 上升到 10.714 s，说明 V10
仍不是所有 workload 的支配策略。

因此论文中应把 V10 定位为“可解释的 deadline-aware OS 调度基线”：报告完整曲线、
completion fraction 和 offered SLO，不能只挑完成请求的低 P99；在当前 service curve
下，V6 在部分高容量点仍胜过 V10，V10 的价值主要在统一的 EEVDF/EDF 解释和受限 overlap
下的尾延迟控制。

## 6. 4-GPU fake-decode 实测

> **状态：作废（`pd_fake_decode`）。** 本节及 `fake-decode-live-v5-v10-260904` 的 GPU/请求结果全部作废，不能用于真实 Decode 或容量结论；原始记录仅供审计。前面的离线 timing-model 结果不等价于 GPU wall-clock。

模拟扫描通过后，在 GPU 0--3 启动普通 PD Prefill：两个 TP2（0,1 和 2,3）加一个共享 GPU
TP4（0,1,2,3），Decode 使用 pd_master fake endpoint；GPU 4--7 不纳入测试。worker-only
启动命令为：

```bash
MASTER_PORT=16011 \
PREFILL_PORT_01=18000 PREFILL_PORT_23=18001 PREFILL_PORT_0123=18002 \
START_MASTER=0 \
./start-cluster3_70b_p22.4d4_mps_prefill_fake_decode_v10.sh
```

然后用 live harness 对 fixed、naive、V5、V6、V7、V8、V9、V10 逐策略执行相同 trace，策略
间重新 warmup，记录 ServeGen 8 req/s 和 synthetic-5pct 8 req/s 的成功率、TTFT p50/p95/p99、
按时 SLO、短/长请求 p95、prompt token 校验、fake transfer tail 和 route class match。实测
结果只用于验证生产协议、overlap 和 selector 切换；fake decode 不能替代真实 Decode 负载
下的容量结论。

已在 4 张 H200（GPU 0--3）完成该流程，GPU 4--7 保持外部任务不变。使用
`max_req_total_len=8192`、ServeGen 20 s @ 8 req/s、synthetic 160 请求 @ 8 req/s、
fake KV transfer=20 ms，并在每个策略/场景前执行 2 轮混合 warmup。所有 16 个测量均
`completion_fraction=1`、`offered_slo=1`、route match=1、prompt token exactness=1。

| policy | ServeGen P50/P95/P99 (s) | synthetic P50/P95/P99 (s) |
|---|---:|---:|
| fixed | 0.391/1.065/1.208 | 0.109/0.312/0.592 |
| naive | 0.394/1.018/1.181 | 0.107/0.341/0.606 |
| V5 | 0.406/1.050/1.069 | 0.132/0.385/0.589 |
| V6 | 0.424/1.082/1.175 | 0.131/0.346/0.585 |
| V7 | 0.409/0.950/1.118 | 0.131/0.324/0.584 |
| V8 | 0.438/1.139/1.253 | 0.144/0.383/0.635 |
| V9 | 0.416/1.032/1.079 | 0.132/0.336/0.576 |
| V10 | 0.414/1.013/1.078 | 0.126/0.333/0.606 |

原始结果和逐请求记录位于
`_/flex_tp_paper_analysis/fake-decode-live-v5-v10-260904/`。V10 在该低/中负载 fake-decode
测试中与 V9 接近：ServeGen P99 几乎相同，synthetic P50/P95 略有改善，但 naive/fixed
仍可能拥有更低的总体 P50。因此实测支持“协议兼容、路由正确、没有请求级失败”，不支持
在真实 Decode 容量上宣称 V10 普遍领先。

worker 日志在每次 master 切换/停止时出现 websocket close/reconnect 栈，这是 worker
长驻而 master 按策略重启造成的预期现象。fake-decode 的 KV 清理路径还记录了
`radix_cache` 为 `None` 的非致命异常；请求仍全部返回且 prompt 校验通过。若转向真实
KV transfer 或生产化 fake-decode，应单独修复该清理路径并重新测量。

## 7. 局限

- 模拟 service curve、MPS slowdown 和 KV transfer 是显式模型，不是 H200 kernel 测量；
- 当前 sweep 使用单 seed，论文实验应增加多个 seed 和置信区间；
- 需要增加 burst、不同长请求比例、Decode 非零负载以及预测误差实验；
- 需要报告 selector CPU/replan 开销、GPU 利用率、取消/重注册和 stale generation 行为。
