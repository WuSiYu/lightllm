# 260907 MPS fake Decode 数据审计

## 结论

`pd_fake_decode=True` 只在 MPS slowdown 的 worker 侧执行时间中保留为例外。它不能用于客户端请求 latency、TTFT、吞吐、完整 PD 服务行为或调度器比较。

现有数据中，`_/mps_real_260906/highqps_trace/` 有符合“高 QPS、solo 与 overlap 分开、worker 侧计时、尾部过滤、检查满载”的可用子集：

- 2048 短请求和 8192 背景请求的 overlap 段合格；同形状 TP4 worker 的 `model_forward_ms` p50 slowdown 为 **3.51 倍**。
- 1024 短请求和 8192 背景请求也有合格段，作为额外参考为 **3.39 倍**。
- 32、64、128 没有足够长的同时满载段，不能从现有 trace 得出可信 slowdown。

`_/mps_batch_260906/` 的三组 fixed/flex 混合流有原始 `PERF - prefill`，但 FlexTP all-mix 没有让两个 TP2 worker 和 TP4 worker 同时满载，所以不能作为最终三组结果。原始 `PERF` 数值只能保留为 MPS worker 批处理候选观察值。

需要区分计时来源：上面合格的 1024/2048 数值来自 `highqps_trace` 的 CUDA-event `model_forward_ms`；该 high-QPS 运行目录没有保存与每个选中饱和段一一对应的原始 `PERF - prefill` 文本日志。因此，如果最终口径强制要求“必须是 PERF 行”，现有数据没有完整合格的 1024/2048 PERF 结果，不能把 CUDA-event 数值冒充 PERF。

## 资格规则

1. 计时只取 worker 的 CUDA-event `model_forward_ms`；不使用客户端请求的排队 latency。
2. 每个 TP rank 的重复记录按 worker 端口、batch shape、请求长度和 30 ms 起始窗口去重。
3. overlap 必须分别检查 `8000`、`8001`、`8002` 三个 worker，而不是把两个 TP2 端口做 union 后当成一个 worker。
4. 以每个端口的执行区间为基础寻找初始饱和段：相邻间隙不超过 100 ms 的事件先用于分段；随后重新用未补间隙的区间计算 busy fraction，因此间隙不会被计入执行时间。
5. 合格条件为每个所需端口 busy fraction 至少 0.90，overlap 三端口交集至少 0.90，且饱和段至少 10 秒并包含至少 10 个可比 TP4 batch。尾部 drain 不计入该段。

## 高 QPS trace

`highqps_runs.jsonl` 的短请求发送速率为 500 req/s，8192 背景速率为 30 req/s，运行时 `max_inflight=null`；每个长度都有独立 solo 和 overlap 运行。客户端 JSON 仅作发送元数据，不能用作 latency 结论。

下面是 overlap 全窗口的端口级占用率。它们用于说明为什么不能直接使用旧的 aggregate 数字；最终合格段的结果见下一表。

| 短请求长度 | TP2(0,1) 8000 | TP2(2,3) 8001 | TP4 8002 | 三端口交集 | 结论 |
|---:|---:|---:|---:|---:|---|
| 32 | 0.068 | 0.078 | 0.767 | 0.048 | 不合格 |
| 64 | 0.147 | 0.170 | 0.830 | 0.104 | 不合格 |
| 128 | 0.260 | 0.294 | 0.842 | 0.219 | 不合格 |
| 1024 | 0.842 | 0.887 | 0.855 | 0.797 | 全窗口不合格，初始饱和段合格 |
| 2048 | 0.854 | 0.902 | 0.578 | 0.565 | 全窗口不合格，初始饱和段合格 |

对 1024 和 2048，按上面的分段规则过滤掉开始建立队列前的空闲期和结束后的 tailing，得到：

| overlap 条件 | 饱和段（相对该 run start） | 8000 / 8001 / 8002 busy | 三端口交集 | TP4 可比 shape 样本 | p50 `model_forward_ms` | p90 | 相对 solo p50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 + 8192 | 15.46--129.55 s | 0.977 / 0.978 / 0.995 | 0.951 | 23 | 3766.41 ms | 4574.99 ms | 3.39x |
| 2048 + 8192 | 23.33--139.89 s | 0.987 / 0.989 / 0.995 | 0.971 | 24 | 3905.89 ms | 4259.86 ms | 3.51x |

两行的 solo 基线是同一份独立的 `solo 8192` run，形状固定为 `batch_tokens=16384, batch_size=2, request_input_lens=(8192,8192)`，23 个样本的 `model_forward_ms` p50/p90 为 **1111.70/1114.60 ms**。因此 2048 行的 p50 比值为 `3905.89 / 1111.70 = 3.51x`，不是客户端排队 latency 比值。

32、64、128 的最长三端口重叠段只有约 0.95、1.72、2.02 秒，且每个端口只有约一个可比 batch；即使这些片段的局部 occupancy 接近 1，也不足以形成稳定的 p50 样本，故标为不合格而不补算 slowdown。

## 原始 PERF chunk 数据

`_/mps_batch_260906/perf_chunk_8192_summary.json` 由原始 `PERF - prefill` 行汇总，服务启动参数记录了 `pd_fake_decode=True`、`enable_mps=True`、`chunked_prefill_size=8192`、`batch_max_tokens=16384`。这些数据适合做 MPS worker 批处理的形状对照，但不能越过上面的满载资格规则：

| 条件 | shape | `model` p50 | 完整 `PERF latency` p50 | 样本数 | 资格 |
|---|---|---:|---:|---:|---|
| fixed TP4 | token 8192, bs 1 | 549.16 ms | 1308.90 ms | 8 | 基线可用 |
| FlexTP all-mix | token 8192, bs 1 | 1253.20 ms | 2908.68 ms | 8 | FlexTP 未通过三端口满载 |
| FlexTP all-mix（p90） | token 8192, bs 1 | 1936.78 ms | 4596.63 ms | 8 | 只能作候选观察 |

对应的 FlexTP all-mix trace 全窗口只有 8000/8001/8002 = **0.179/0.169/0.990** 的 busy fraction；因此旧报告中的约 3.5 倍不能作为严格三组 slowdown。客户端 `fixed_tp2_client/short.json`、`fixed_tp4_client/long.json`、`flextp_client/all.json` 的请求 latency/吞吐仍然作废。

## chunked-prefill 边界

现有 no-chunk 对照中，TP4 同形状 8192-token batch 的纯 forward 中位数约 564 ms；启用 16k chunk 的对照约 571 ms，均接近 fixed TP4 约 547 ms。这支持“chunk slicing 本身没有造成固定 3 倍基线膨胀”，但不能排除 chunked-prefill 通过改变 batch packing 和调度节奏间接影响 MPS 竞争。它不是 32/64/128/2048 三组请求 slowdown 的替代结果。

## 可复查文件

- worker trace：`_/mps_real_260906/highqps_trace/`
- run 边界：`_/mps_real_260906/highqps_runs.jsonl`
- 逐 worker 满载审计结果：`_/mps_real_260906/highqps_trace_analysis_260907.json`
- 原始批处理 trace：`_/mps_batch_260906/fixed_tp2_trace/`、`fixed_tp4_trace/`、`flextp_trace/`
- 原始 PERF 汇总：`_/mps_batch_260906/perf_chunk_8192_summary.json`
- 解析器：`test/benchmark/service/analyze_mps_trace_260906.py`

本次只做了离线读取和重新核对，没有启动、停止或重跑任何服务。
