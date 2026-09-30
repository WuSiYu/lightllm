# 260904-FlexTP v11：无长度阈值的弹性 TP 调度

## 1. 目标与结论

v11 基于 v10 的普通 PD lease 生命周期、EEVDF 风格 deadline 排序和 MPS overlap reservation，但移除了 `seq_len <= 4000` 的硬路由。每个待处理请求都会同时评估所有可用 TP2/TP4 实例，路由结果由预测完成时间、队列负载、实例 TP footprint、overlap slowdown 和请求 deadline 共同决定。

这不是把阈值换成另一个 token 阈值，而是把 TP 选择变成一个连续的成本比较。模拟结果显示，v11 在高负载、`target_ttft=4s` 时对 ServeGen 尾部更有韧性；在低 target 或低负载时，v10/v6 仍可能更好，因此 v11 应作为可调的 elastic policy，而不是声称对所有工作负载支配旧版本。

## 2. v11 的调度状态

v11 复用 `FlexTPSelectorV10` 的以下实现：

- Prefill 实例发现、TP-SMT group/GPU placement 和 generation 校验。
- pending request、lease、bundle ACK、worker report、completion 和取消的生命周期。
- `V9Profile.predict_batch()` 服务曲线、token credit、最大 inflight 和 bundle 限制。
- EEVDF 虚拟服务时间、deadline guard 和非抢占式 overlap slack reservation。

v11 覆盖了 v9/v10 中会引入长度边界的两个方法：

- `_class(seq_len)` 对任何长度都返回同一个队列类 `short`。
- `_initial_level(seq_len)` 对任何长度都返回 0。

构造函数仍接受 `long_request_threshold`，只是为了让旧启动脚本和 `start_args` 能继续加载；v11 的 admission、routing、queue level 和 snapshot 都不读取这个值，snapshot 中的 `routing_threshold` 固定为 `null`。

## 3. 动态 TP 选择

### 3.1 候选集合

每次调度从 deadline 最早、enqueue order 最早的 pending 请求开始，枚举：

```text
pending request × every available Prefill instance
```

对每个候选先检查 `_accepts()`（实例可用、inflight、bundle 和 token credit），再检查 v10 的 `_overlap_allowed()`。因此 TP2/TP4 都可以被同一个请求选择；TP4 不再是“长请求专属 lane”。

### 3.2 预测完成时间与连续资源价格

对请求 `j` 和实例 `i`，沿用 v10 的实际预测：

```text
T_pred(j,i) = max(now, virtual_load_i)
              + batch_service(j and existing leases, tp_i)
                * slowdown(active placement)
              + bundle_window + prediction_margin
```

仅最小化 `T_pred` 会在 500～2000 token 的请求上过度偏向 TP4。TP4 在本集群占用整组 GPU，和 TP2 同时运行还会触发 MPS slowdown，所以 v11 加入连续 routing price：

```text
footprint = max(0, tp_i / min_available_tp - 1)
overlap   = max(0, slowdown(active placement) - 1)
P(j,i)    = routing_cost_weight * service(j,i) * (footprint + overlap)
T_eff     = T_pred + P(j,i)
```

默认 `routing_cost_weight=0.50`。它不是长度阈值：长度只通过测量得到的 `service(j,i)` 进入公式；实例当前负载和 overlap 状态也会实时改变价格。

在当前 TP2/TP4 profile 下，短请求通常因 TP footprint 和固定开销选择 TP2；当 TP2 排队、TP2 overlap 代价变大，或请求服务曲线使 TP4 的完成时间优势足够大时，候选会转向 TP4。中间长度没有特殊分支。

### 3.3 全局排序

所有请求/实例候选用同一 global queue 比较，排序键为：

```text
(actual_deadline_miss,
 deadline,
 effective_lateness,
 T_eff,
 T_pred,
 exclusive_service,
 admitted_tokens,
 tp_size,
 enqueue_order,
 node_key)
```

其中 `actual_deadline_miss` 仍使用 `T_pred > deadline`，保证资源价格不会把一个实际可按时完成的请求错误标成 miss；可行候选内部用 `T_eff` 反映资源代价。v10 的 overlap reservation 继续阻止新 admission 消耗已有 lease 的保留 slack；如果已有 lease 已经预测超时，则保持 work-conserving，避免无效的 head-of-line blocking。

## 4. 生产接口与工具

新增文件：

- `lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v11.py`
- `test/test_flex_tp_selector_v11.py`
- `start-cluster4_70b_p22.4d4_mps_flex_v11.sh`
- `start-cluster3_70b_p22.4d4_mps_prefill_fake_decode_v11.sh`

已接入 selector factory、`api_cli.py`、`start_args_type.py`、PD master manager、step simulator、rate sweep 和 fake-decode live harness。生产选择器名为 `flex_tp_v11`。

可调参数：

```text
--flex_tp_v11_routing_cost_weight       default 0.50
--flex_tp_v11_slack_weight              default 1.0
--flex_tp_v11_overlap_slack_ratio       default 0.10
```

`v11_short_weight`/`v11_long_weight` 保留为 v10 CLI 兼容项；v11 是单队列，默认不会形成两类长度队列。

## 5. 全量模拟

使用 `servegen-mm-image` 和 5% 长请求 synthetic trace，4 个 target TTFT（0.5/1/2/4s），request rate（0.5/1/2/4/6/8/10/12/16/20/24 req/s）。每个 `(dataset, rate)` trace 只生成一次并复用到所有 policy/target；v11 的 88 行与 v10 artifact 的对应 88 行 fingerprint 全部一致。

结果 artifact：

- [v11-only JSON](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-only-final-260904/rate_ttft_sweep.json>)
- [v3-v11 merged CSV](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904/rate_ttft_sweep.csv>)
- [v3-v11 merged JSON](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904/rate_ttft_sweep.json>)
- 曲线文件位于同一目录，分别提供 `ttft_p50`/`ttft_p99` 的线性和 log y-axis 版本。

下表选择高负载点，数值为 `p50 / p99 / offered SLO`：

| Dataset, target | Policy | rate 12 | rate 16 | rate 20 | rate 24 |
|---|---|---:|---:|---:|---:|
| ServeGen, 4s | v9 | 5.38 / 40.86 / 0.335 | 9.44 / 77.34 / 0.227 | 27.14 / 96.93 / 0.073 | 34.74 / 120.25 / 0.099 |
| ServeGen, 4s | v10 | 5.26 / 34.57 / 0.351 | 10.71 / 72.03 / 0.255 | 25.08 / 100.70 / 0.124 | 32.97 / 117.11 / 0.105 |
| ServeGen, 4s | v11 | 3.29 / 38.36 / 0.681 | 3.88 / 67.73 / 0.529 | 5.43 / 98.75 / 0.420 | 6.99 / 110.54 / 0.399 |
| Synthetic, 4s | v9 | 0.59 / 4.14 / 0.987 | 1.48 / 14.04 / 0.963 | 4.52 / 23.39 / 0.376 | 6.25 / 29.59 / 0.216 |
| Synthetic, 4s | v10 | 0.61 / 3.18 / 0.994 | 1.53 / 13.16 / 0.979 | 4.47 / 27.27 / 0.369 | 6.03 / 34.37 / 0.238 |
| Synthetic, 4s | v11 | 0.17 / 3.37 / 0.998 | 1.77 / 9.52 / 0.963 | 2.61 / 30.98 / 0.927 | 3.14 / 35.80 / 0.820 |

解读：v11 在 ServeGen 4s 的 rate 12～24 明显减少 p50 backlog，并保持更高 SLO；synthetic rate 20/24 也优于 v9/v10 的 SLO，但 p50 和低 target 结果存在代价。该差异来自 trace 的长度/到达模式以及 TP4 overlap，而不是测试 trace 不配对。

## 6. 验证

通过：

```text
python -m py_compile <all touched Python files>
python test/benchmark/service/flex_tp_step_sim.py --self-test
```

step simulator 自测覆盖 v3～v11、ChunkedPrefillQueue、bundle ACK、MPS overlap、fake Decode 和 v11 的 TP2/TP4 动态路由。`test/test_flex_tp_selector_v11.py` 已加入 pytest 回归用例；当前环境没有安装 pytest，因此这里只能完成静态编译和 simulator 自测，未执行 pytest runner。

## 7. 使用建议

v11 适合作为“无长度阈值 + deadline/resource-price”研究方案和高负载 ServeGen 场景的候选实现。论文中应报告完整 p50/p99-vs-rate 曲线、SLO attainment、TP selection share、MPS overlap wall time，并做 `routing_cost_weight` 的 sensitivity ablation（至少 0、0.25、0.5、1.0）。生产部署前应使用真实 worker report 重新校准 `V9Profile` 和 overlap slowdown；模拟器结果不能替代 H200 实机验证。
