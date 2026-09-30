# 260904-FlexTP v12：连续预测完成时间调度与 H200 实测

## 1. 范围和结论

v12 是在 v6 lease/admission 生命周期上的一次连续化路由改进：取消 `seq_len <= 4000` 的实际路由切分，对每个待调度请求枚举所有可用 TP2/TP4 实例，以预测完成时间加 GPU footprint 价格选择实例。v12 的目标是保留 v6 的吞吐底线，同时避免把中等长度请求机械地归入某一个 TP lane。

在 synthetic 44 个配对点中，v12 的 completion throughput 和 prefill-token throughput 没有低于 v6；ServeGen 的 88 个配对点中有 1 个点低约 0.025%，其余不低于 v6。部分高负载点的 SLO goodput、p99 TTFT 仍可能劣于 v6，说明“吞吐不低于”不等价于“所有服务质量指标支配”。因此 v12 应作为可调的 elastic policy，并报告参数敏感性，而不应宣称普遍最优。

## 2. 调度逻辑

### 2.1 复用的 V6 机制

- Prefill worker 通过 PD master 注册，校验 TP size、GPU placement、`max_req_total_len` 和 generation。
- pending request 经过 bundle、lease、ACK、worker report、completion/cancel 的完整生命周期。
- 每个实例有 `max_inflight`、token credit、bundle token cap/trigger 和 admission 状态。
- deadline 使用 EDF 顺序；已有 lease 的预测 slack 参与 overlap reservation，避免新任务无界侵占已有请求。
- 服务时间来自 `V9Profile.predict_batch()`，同时考虑 TP size、batch packing、当前重叠 GPU 集合和 MPS slowdown。

### 2.2 无阈值候选集合

对每个 pending 请求 `j`，候选集合是所有可用实例 `i`：

```text
pending request × every available TP instance
```

`long_request_threshold` 仍作为旧启动参数被接受，但 v12 的 `_targets()`、admission 和 TP 路由不读取它。`_request_class()` 仅用独立测得的 TP2/TP4 service curve 产生 overlap 统计提示，不是硬分桶。

### 2.3 目标函数和 TP4 保护

先用已占用 lease 加上候选请求进行 batch packing，得到：

```text
T_pred(j,i) = now + packed_service(j,i) * MPS_overlap_factor
             + bundle_window + prediction_margin
T_eff(j,i)  = T_pred(j,i) + w * service_exclusive(j,i) * footprint(i)
```

其中 `footprint(i) = tp_i / min_available_tp`，在当前 TP2/TP4 拓扑中 TP4 的 footprint 是 TP2 的两倍；默认 `w=2.0`。候选按照 deadline、`T_eff`、独占 service、`T_pred`、token load、TP size 和 node key 稳定排序，然后调用 V6 `_admit()`。独占 service 只影响有效完成时间完全相同的候选，避免浮点平局时无理由偏向 TP2。

为保持 aggregate TP2 capacity，TP4 只有在逐请求服务曲线满足以下条件时才进入候选：

```text
service_tp4(j) <= ratio_limit * service_tp2(j)
```

默认 `ratio_limit=0.60`。这是每个请求的相对性能/资源保护条件，不是 token 长度阈值；可通过 `--flex_tp_v12_tp4_service_ratio_limit` 调整。v12 默认参数为：bundle cap 8192、trigger 4096、token credit 16384、prediction margin 80ms、MPS overlap slowdown 2.0。

## 3. 模拟验证

使用相同 trace fingerprint，扫描 synthetic 5% 长请求和 ServeGen mm-image，target TTFT 为 0.5/1/2/4s，rate 为 0.5/1/2/4/6/8/10/12/16/20/24 req/s；进程池使用 16--64 workers，父进程统一写结果，避免并发写 JSON 导致卡死。

- synthetic 全量结果（最终 tie-break 版本，32 进程）：[rate-ttft-v6-v12-synthetic-tiebreak-mp32-260904](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-v6-v12-synthetic-tiebreak-mp32-260904/rate_ttft_sweep.csv>)。
- ServeGen 完整 v6/v12 结果（最终 tie-break 版本，88 个 dataset-rate-SLO shard，16 进程）：[rate-ttft-v6-v12-servegen-tiebreak-mp16-260904](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-v6-v12-servegen-tiebreak-mp16-260904/rate_ttft_sweep.csv>)。
- ServeGen 小规模交叉检查：[rate-ttft-v6-v12-servegen-small-260904](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-v6-v12-servegen-small-260904/rate_ttft_sweep.csv>)。

在 synthetic 44 个 v6/v12 对比点中，v12 的 completion throughput 和 prefill-token throughput 均达到或超过 v6；on-time goodput 有 19 个点低于 v6，主要集中在高负载或较紧 SLO。ServeGen 完整 sweep 中吞吐只有一个点（rate=16、SLO=4s）低于 v6，差值约 0.025%，其余 87 点不低于 v6。该 trade-off 已保留在原始 CSV/JSON 中，不能只看平均吞吐。

对 `tp4_service_ratio_limit` 的 sensitivity 也已完成（[汇总 CSV](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/v12_ratio_sensitivity_260904.csv>)）：0.55 在 synthetic/ServeGen 分别有 1/8 个吞吐反例，0.59 有 2/4 个，0.60 有 0/1 个，0.61 有 5/23 个；虽然 0.55 的平均吞吐更高，但尾部 SLO 代价更明显，所以默认仍取 0.60。该参数是可调的工作负载策略，不是对任意 trace 的理论吞吐保证。

模拟器是离散 step/timing model：decode 视为无瓶颈，只保留 fake KV transfer 时间；它用于比较算法趋势，不代表真实 kernel wall time。

## 4. H200 四卡 fake-decode 实测

> **状态：作废（`pd_fake_decode`）。** 本节及其引用的 H200 实测数据均来自 fake-decode 服务，不能作为真实 Decode、真实请求 latency 或吞吐证据；原始数据仅供审计。本文前面的离线模拟器结果不因本节作废，但也不等价于 GPU wall-clock。

### 4.1 拓扑定义

本轮只使用 GPU 0--3，MPS 开启，模型为 `Llama-3.3-70B-Instruct`，`max_req_total_len=65536`，chunked prefill=8192，fake decode KV transfer 固定 20ms；每个策略使用同一组 8-request synthetic-5pct trace，rates 为 0.5/1/2/4 req/s。

- `fixed_tp2`：仅两个独立 TP2 实例（GPU 0,1 和 2,3）。
- `fixed_tp4`：仅一个 TP4 实例（GPU 0--3）。
- `naive`、`v6`、`v12`：两个 TP2 与一个 TP4 同时注册，由各自 selector 决定是否重叠使用。

fixed_tp4 的 TP4 worker 使用 shared-weight master；flex TP4 使用 shared-weight slave，并等待两个 TP2 注册后启动。worker 加载和 master 管理均已在脚本中固定，避免端口/权重模式混淆。

### 4.2 已完成结果

fixed_tp4 四档 SLO 扫描结果：

- [SLO 0.5s](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/gpu-sweep/fixed_tp4/slo-0.5-260904/results.json>)
- [SLO 1s](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/gpu-sweep/fixed_tp4/slo-1-260904/results.json>)
- [SLO 2s](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/gpu-sweep/fixed_tp4/slo-2-260904/results.json>)
- [SLO 4s](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/gpu-sweep/fixed_tp4/slo-4-260904/results.json>)

fixed_tp2 四档 SLO 扫描结果：

- [SLO 0.5s](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/gpu-sweep/fixed_tp2/slo-0.5-260904/results.json>)
- [SLO 1s](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/gpu-sweep/fixed_tp2/slo-1-260904/results.json>)
- [SLO 2s](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/gpu-sweep/fixed_tp2/slo-2-260904/results.json>)
- [SLO 4s](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/gpu-sweep/fixed_tp2/slo-4-260904/results.json>)

fixed_tp2 的四个 rate 点均 8/8 成功；0.5s SLO 达成率为 0.875，1/2/4s 为 1.0。实测 input-token throughput 约为 776/1465/2640/4448 token/s（rate 0.5/1/2/4）。fixed_tp4 的同组实测约为 793/1533/2873/5076 token/s，0.5s SLO 达成率同样为 0.875，宽松 SLO 为 1.0。两组 fixed 结果说明当前短 trace 主要受请求规模和 fake transfer 控制，不能据此外推高并发结论。

flex 的 naive/v6/v12 三策略四档 SLO 扫描已完成，结果位于 `gpu-sweep/flex-naive-v6-v12/slo-{0.5,1,2,4}-260904`。每个策略的 16 个点均 8/8 成功；0.5s SLO 达成率为 0.875，1/2/4s 为 1.0。rate=4 的结果如下（`p50/p99` 单位为秒，吞吐为 input token/s）：

| SLO | naive | v6 | v12 |
|---|---:|---:|---:|
| 0.5s | 0.112 / 0.493 / 5076 | 0.152 / 0.507 / 5048 | 0.137 / 0.508 / 5042 |
| 1s | 0.119 / 0.486 / 5091 | 0.142 / 0.501 / 5060 | 0.137 / 0.505 / 5051 |
| 2s | 0.120 / 0.486 / 5093 | 0.136 / 0.478 / 5114 | 0.142 / 0.483 / 5103 |
| 4s | 0.115 / 0.482 / 5101 | 0.142 / 0.487 / 5095 | 0.138 / 0.486 / 5097 |

完整的 req-rate/p50/p99 序列保留在上述 4 个 `results.json`；与 fixed 结果对应的曲线应按拓扑分图，不能把 fixed_tp2、fixed_tp4 和 flex 的吞吐直接合并成一条曲线。当前 8-request 短 trace 下 v12 与 v6 吞吐差异在测量噪声范围内：16 个主测点均成功，均值约为 2563 与 2568 input token/s，不能据此宣称实机严格 floor；需要更长 trace/重复实验作显著性检验。

所有 GPU 结果的统一 CSV 为：[gpu_sweep_summary_260904.csv](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/gpu-sweep/gpu_sweep_summary_260904.csv>)。其中 `main-8req` 是 fixed/naive/v6/v12 的主测，`quick-4req` 是 v3/v4/v5/v7/v8/v9/v10/v11 的快速交叉测；两类样本不要直接做显著性比较。

其余 selector 的 flex 快速交叉测结果位于 `gpu-sweep/flex-other/slo-{0.5,1,2,4}-260904`。v3、v4、v5、v7、v8、v9、v10、v11 在所有 128 个策略-rate-SLO 点（8 策略 × 4 SLO × 4 rate）均 4/4 成功；该结果主要确认生产 selector、worker 注册、bundle/ACK 和 fake-decode 路径兼容，复杂 workload 的算法排名仍以 simulator 全量 sweep 为准。

快速交叉测的 16 点均值如下（4-request trace，吞吐单位 input token/s）：

| selector | p50 TTFT (s) | p99 TTFT (s) | 吞吐 |
|---|---:|---:|---:|
| v3 | 0.139 | 0.158 | 1680 |
| v4 | 0.140 | 0.166 | 1676 |
| v5 | 0.139 | 0.175 | 1678 |
| v7 | 0.140 | 0.176 | 1674 |
| v8 | 0.143 | 0.168 | 1680 |
| v9 | 0.138 | 0.169 | 1673 |
| v10 | 0.139 | 0.170 | 1682 |
| v11 | 0.122 | 0.138 | 1680 |

## 5. 复现和测试状态

```bash
python -m py_compile lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v12.py
python -u test/benchmark/service/flex_tp_step_sim.py --self-test
```

当前 simulator self-test 已覆盖 v3--v12、naive、fixed_tp2、fixed_tp4 和 fake-decode harness；环境未安装 pytest，因此 pytest 回归文件只能完成静态检查。GPU 实测脚本为 `test/benchmark/service/flex_tp_fake_decode_live_benchmark.py`，默认由 harness 管理 master，worker 启动脚本只负责拓扑和 MPS。

论文报告建议同时给出：TTFT p50/p99--rate 曲线、SLO attainment、input-token goodput、TP2/TP4 selection share、MPS overlap wall time，以及 `routing_cost_weight` 和 `tp4_service_ratio_limit` 的 sensitivity ablation。
