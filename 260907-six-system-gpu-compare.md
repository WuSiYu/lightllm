# 260907 六套系统真实 GPU ServeGen 对照

> **状态：作废（含 `pd_fake_decode`）。** 本报告的统一六系统表格包含 `fixed_tp2`、`fixed_tp4`、`naive`、`naive_switch` 的 `pd_fake_decode=True` 结果，因此整张聚合表、图和由此得出的比较结论均不得使用。v12/v13 的 `pd_fake_decode=False` 原始配对数据另见 `260906-v12-v13-gpu-compare.md`，不因本横幅单独作废。

## 实验范围

2026 年 9 月 6--7 日在同一台 8 张 H200 主机上，分别启动六套系统完成 ServeGen fake-decode 矩阵：`fixed_tp2`、`fixed_tp4`、`naive`、`naive_switch`、`v12`、`v13`。每套系统均覆盖三个子数据集（`mm-image`、`m-large`、`deepseek-r1`）和四档发送速率（2、4、6、8 req/s），每档名义运行 120 秒，共 72 个 JSON，`failed_requests=0`。

benchmark 使用 `--servegen-output-divisor 10`，fake decode 只返回一个 token。因此本报告只比较首 token 延迟（TTFT），不把它解释为真实 decode 吞吐；吞吐列是成功请求数除以名义 120 秒窗口。

## TTFT p99（秒）

表中每个单元格顺序为 `fixed_tp2 / fixed_tp4 / naive / naive_switch / v12 / v13`。

| 数据集 | 2 req/s | 4 req/s | 6 req/s | 8 req/s |
|---|---:|---:|---:|---:|
| mm-image | 1.194 / 1.009 / 1.400 / 0.942 / 2.105 / 1.624 | 1.655 / 1.607 / 2.022 / 2.057 / 4.185 / 1.967 | 1.782 / 1.724 / 1.896 / 1.886 / 24.184 / 1.855 | 103.405 / 4.058 / 67.584 / 68.003 / 147.424 / 3.161 |
| m-large | 1.220 / 1.136 / 1.224 / 1.217 / 1.911 / 1.329 | 1.283 / 1.370 / 1.815 / 1.278 / 3.539 / 1.930 | 1.526 / 1.327 / 1.536 / 1.537 / 3.320 / 1.552 | 1.606 / 1.539 / 1.666 / 1.673 / 2.177 / 1.587 |
| deepseek-r1 | 2.486 / 1.748 / 1.796 / 2.155 / 3.212 / 1.741 | 4.144 / 4.796 / 5.290 / 9.625 / 10.164 / 5.129 | 4.956 / 3.450 / 3.724 / 9.959 / 13.928 / 3.708 | 5.485 / 5.099 / 5.377 / 15.160 / 16.255 / 3.885 |

## 直接观察

- `v13` 在本矩阵的 12 个点都低于 `v12`。其中 `mm-image` rate=6/8 的 v12 已进入严重排队区间，不能把改善简单归因于单一 kernel 或硬件加速。
- `naive` 与 `naive_switch` 在 `m-large` 上接近；在 `deepseek-r1` 的 rate=4/6/8，`naive_switch` 明显更差，这是串行化 TP2/TP4 admission 后等待另一类请求的直接代价。
- `mm-image` rate=8 的 fixed TP2、v12、naive、`naive_switch` 都出现几十到百秒级 p99，说明该点已经严重过载；应与正常负载点分开解读。
- 各 JSON 均为 0 failed，但 master 日志在 worker 尚未全部注册的 warmup 早期出现过短暂的 `no available nodes (prefill=0, decode=1)`。这些请求由 warmup 重试消化，没有计入正式 JSON 的失败数。

## 结果有效性边界

1. 六套系统是不同时间的独立运行，不是同一 MPS daemon 生命周期内的随机交叉 A/B；表格适合方向和长尾比较，不足以支持论文级的严格显著性结论。
2. `naive` 和 `naive_switch` 正式 12 档矩阵在 `radix_cache=None` 守卫修复之前运行。请求首 token 已返回后，PD freeze 辅助路径会记录被捕获的 `AttributeError: 'NoneType' object has no attribute 'insert'`；benchmark 仍报告 0 failed，但该异常可能影响请求清理和长期队列行为。因此这两套的 p99 只能作为探索性结果，修复后的 smoke 已确认该异常消失，正式矩阵需要在修复后重跑才能作为最终证据。
3. fake decode 输出长度被压到 1 token，不能外推真实 decode latency、decode 吞吐或端到端生成体验。
4. 名义吞吐没有等待全部尾部请求 drain；要研究完成吞吐，应另外记录最后一个请求完成时间并报告有效时间窗。

## MPS 与 chunked-prefill 核验

所有六套 GPU worker 的启动参数包含 `enable_mps=True`、`chunked_prefill_size=8192`、`batch_max_tokens=16384`、`disable_chunked_prefill=False`。worker 使用专用 `CUDA_MPS_PIPE_DIRECTORY`，并由启动脚本启动 MPS control daemon。独立修复后 smoke 等待三个 prefill worker 均出现 `server start up ok`，三组 CUDA-event trace 均有数据，且 `naive_switch` 该档 TP2/TP4 forward 区间交集为 0。

MPS slowdown 不从本表的 HTTP TTFT 推导，而使用 worker 侧 CUDA-event `model_forward_ms`。8192-token chunk 的 no-chunk 对照约为 564 ms，16k chunk 对照约为 571 ms，生产 fixed TP4 约为 547 ms；3 倍级别只出现在 TP2/TP4 同时运行大 batch 的 overlap 窗口。详见 `260906-mps-slowdown-investigation.md` 和 `260906-MPS_perf_chunk_8192.md`。

修复后的 `naive_switch` 全拓扑 smoke trace 位于 `_/gpu_live_260907/naive_switch_trace_fullpatch_260907a/trace/`：四个 TP2 rank 文件和四个 TP4 rank 文件均有记录，按 worker/shape 去重后 110 个 batch。该档 `mm-image rate=2` 不是满载实验，TP2 forward busy fraction 为 17.0%，TP4 为 7.3%，但 TP2/TP4 forward 区间交集为 0；它只用于确认拓扑、MPS trace 和 switch 的串行 admission，不用于吞吐结论。

## 产物

- 机器可读汇总：`_/gpu_live_260907/260907-six-system-compare/gpu_rate_ttft_compare.csv`（72 行，全部 0 failed）；
- p99 图：同目录下 `servegen-mm-image.ttft_p99.compare.png`、`servegen-m-large.ttft_p99.compare.png`、`servegen-deepseek-r1.ttft_p99.compare.png`；
- 原始矩阵：`_/gpu_live_260906/v12_sweep/`、`test/benchmark/service/_/gpu_live_260906/v13_sweep_260906b/`、`_/gpu_live_260906/fixed_tp2_sweep_260906/`、`_/gpu_live_260906/fixed_tp4_sweep_260906/`、`_/gpu_live_260907/naive_sweep/`、`_/gpu_live_260907/naive_switch_sweep/`；
- 对照脚本：`test/benchmark/service/compare_gpu_sweeps_260906.py`。
