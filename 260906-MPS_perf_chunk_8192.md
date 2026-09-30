# 260906 MPS 8192-token Chunk Prefill 对照

> **状态：仅限 MPS worker slowdown 审计。** `pd_fake_decode=True` 使客户端 serving latency/吞吐和完整 PD 结论作废；原始 `PERF - prefill` worker 批处理时间可作为 MPS 竞争的候选证据，但必须通过每个 worker 的满载检查。当前资格结论以 `260907-MPS_pd_fake_decode_data_audit.md` 为准。

> **260907 更正：** 原文的 3.5 倍是 `PERF` 形状分组的候选比值，不等于严格满载三组实验的 slowdown；对应 FlexTP all-mix trace 未通过两个 TP2 worker 与 TP4 同时满载检查。

## 结论先行

这次按 worker 日志中的 `PERF - prefill` 直接比较每次最大 8192-token chunk，而不是比较客户端请求 latency。对同一形状 `batch size=1, token=8192`：

| 条件 | `model` 时间 | 完整 `latency` | 说明 |
|---|---:|---:|---|
| fixed TP4 | p50 549 ms，p90 560 ms | p50 1,309 ms，p90 1,415 ms | 单独 TP4 |
| FlexTP，低竞争时段 | 约 572 ms | 约 1,224 ms | 18:21:40 UTC，未观察到明显 TP2 大 batch 竞争 |
| FlexTP，TP2 并发大 batch 时段 | 约 1,936 ms | 约 4,596 ms | 18:21:06 UTC，TP2 同时执行大 batch |

因此，按 `PERF` 中名为 `model` 的区间计算，FlexTP 在明确并发时段的 8192-token chunk 观察到约 **3.5 倍候选比值**（1936/549），完整 `PERF latency` 也约为 **3.5 倍候选比值**（4596/1309）。低竞争时段只有约 1.04 倍，说明高值不是固定的 TP4 基线差异，而是并发资源竞争/批处理状态造成的；是否能作为 slowdown，必须再通过逐 worker 满载审计。

严格说，现有 `PERF` 实现中的 `model` 计时点是 `t[0]` 到 `t[1]`：`t[0]` 在 `prepare_prefill_inputs` 之前，`t[1]` 在 model forward 和 sampling 之后。因此它是稳定的 worker 第一阶段区间，适合做同代码路径的相对比较，但不是只包住 `model.forward` 的纯 CUDA 时间。纯 forward 应以 `LIGHTLLM_MPS_TRACE_DIR` 生成的 CUDA-event trace 为准；本报告使用 `PERF` 日志是因为它能直接按每个 8192-token chunk 对齐到现有服务日志。

## 日志中的直接证据

fixed TP4 的 `p0123.log` 中，8192-token、batch size 1 的 8 个 rank 记录去重后得到 `model` p50 549.16 ms、`latency` p50 1308.90 ms。

FlexTP 的同形状事件分成两组：

- 18:21:06: `model=1933--1938 ms`，`latency=4593--4599 ms`；
- 18:21:40: `model=572--573 ms`，`latency=1223--1224 ms`。

第一组发生时，TP2 日志同一时间段正在执行以下 batch：

| worker | batch | `model` 时间 |
|---|---:|---:|
| TP2(2,3) | batch size 18，8864 token | 约 2086 ms |
| TP2(2,3) | batch size 15，10208 token | 约 2483 ms |
| TP2(0,1) | batch size 2，2080 token | 约 523--543 ms |
| TP2(0,1) | batch size 32，16160 token | 约 3773--3816 ms |

这说明 1936 ms 的 TP4 chunk 不是 HTTP 排队时间：`model` 字段来自 worker 第一阶段的 CUDA-event 计时；但它确实处在 TP2 大 batch 和 TP4 forward 同时占用 GPU 的窗口中。由于该字段的起止点比纯 `model.forward` 更宽，不能把它当成严格的单算子时间。完整 `latency` 还包含 post、stream 等待或其他非模型部分，也不能用来替代 `model` 时间。

## 与 chunked-prefill 配置的关系

本次 worker 参数为：

- `chunked_prefill_size=8192`；
- `batch_max_tokens=16384`；
- `disable_chunked_prefill=False`；
- 日志打印 `enable_mps=True`。

因此这里的 `token=8192 (may chunked)` 是调度器允许的最大单次 chunk；不能把它解释成一个没有 chunk 的完整请求。当前对照确认的是“8192-token chunk 在单独执行和 TP2/TP4 并发执行时的差异”。它还没有把 chunk slicing 本身与两个完整 model forward 同时竞争的影响分离出来。

补充复查了一个有效的 no-chunk 对照：保持 `batch_max_tokens=16384`，仅增加 `--disable_chunked_prefill`。该对照中 TP4 的 `batch_tokens=8192,batch_size=1` 纯 forward trace 中位数约 564 ms；`chunked_prefill_size=16384` 时同形状约 571 ms；生产配置 fixed TP4（8192 chunk）约 547 ms。三者相近，说明 chunk slicing 本身没有造成 3 倍级别的固定基线膨胀。3 倍级别只在 TP2/TP4 同时运行大 batch 的 overlap 窗口中出现。

这个 no-chunk 对照的 overlap 长度覆盖不完整，因此只能排除“切分本身单独导致 3 倍 slowdown”，不能排除 chunked-prefill 通过改变 batch packing、调度节奏和同时满载概率而间接放大 MPS 竞争。严格的 slowdown 数值仍以 worker CUDA-event trace 的 `model_forward_ms` 为准。

## MPS 是否正确打开

配置和运行证据均支持“按预期启用了 MPS”：

1. 三组服务的 `all start args:Namespace` 都记录 `enable_mps=True`、`chunked_prefill_size=8192`、`batch_max_tokens=16384`，且 `disable_chunked_prefill=False`。
2. 启动脚本为 TP2(0,1)、TP2(2,3)、TP4(0,1,2,3) 设置同一个专用 `CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_prefill`，并执行 `nvidia-cuda-mps-control -d`。
3. worker 日志中的 TP2/TP4 forward 区间确实发生时间交叠；这与 MPS 允许多个 CUDA client 在同一组 GPU 上并发运行的现象一致。
4. 事后用独立目录启动/查询 MPS daemon 的 sanity check 成功，说明当前驱动和 MPS 工具链可用。

保留一个证据边界：历史测试没有保存 MPS daemon 的完整生命周期遥测，不能仅凭今天的进程状态数学地证明 daemon 在每条旧日志期间都持续存活。因此最终表述应是“启动配置正确，日志行为与 MPS overlap 一致，且存在直接并发 slowdown 证据”，而不是声称有完整 daemon 级别审计记录。

## 文件和复现入口

- 解析脚本：`test/benchmark/service/analyze_perf_chunk_260906.py`；
- 机器可读汇总：`_/mps_batch_260906/perf_chunk_8192_summary.json`；
- fixed TP4 日志：`_/server_log_70b_p22.4d4_mps_prefill_fake_decode_fixed_tp4_round_robin/p0123.log`；
- FlexTP 日志：`_/server_log_70b_p22.4d4_mps_prefill_fake_decode_flex_flex_tp_naive/p01.log`、`p23.log`、`p0123.log`。
