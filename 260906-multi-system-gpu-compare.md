# 260906 多系统 GPU ServeGen 对照

> **状态：作废（含 `pd_fake_decode`）。** 本文件的 fixed TP2/TP4 聚合表来自 `pd_fake_decode=True` 服务；混合聚合结果、图和比较结论全部作废。v12/v13 的真实 Decode 配对数据请单独查看 `260906-v12-v13-gpu-compare.md`。

本文件保留 9 月 6 日完成的四套系统历史快照；包含 naive 和 `naive_switch` 的六套系统最新合并结果请看 `260907-six-system-gpu-compare.md`。

## 结果概览

在 2026 年 9 月 6 日完成了四套系统的独立 GPU ServeGen 对照：fixed TP2、fixed TP4、v12 和 v13；数据集为 mm-image、m-large、deepseek-r1，发送速率为 2/4/6/8 req/s，共 48 个 benchmark JSON，全部 0 failed。每次运行持续 120 秒，fake decode 返回 1 个 token，因此本表只解读 TTFT p99；decode per-token latency 不适用。

下面给出 TTFT p99（秒），每个单元格顺序为 `fixed_tp2 / fixed_tp4 / v12 / v13`。

| 数据集 | 2 req/s | 4 req/s | 6 req/s | 8 req/s |
|---|---:|---:|---:|---:|
| mm-image | 1.194 / 1.009 / 2.105 / 1.624 | 1.655 / 1.607 / 4.185 / 1.967 | 1.782 / 1.724 / 24.184 / 1.855 | 103.405 / 4.058 / 147.424 / 3.161 |
| m-large | 1.220 / 1.136 / 1.911 / 1.329 | 1.283 / 1.370 / 3.539 / 1.930 | 1.526 / 1.327 / 3.320 / 1.552 | 1.606 / 1.539 / 2.177 / 1.587 |
| deepseek-r1 | 2.486 / 1.748 / 3.212 / 1.741 | 4.144 / 4.796 / 10.164 / 5.129 | 4.956 / 3.450 / 13.928 / 3.708 | 5.485 / 5.099 / 16.255 / 3.885 |

## 解读

- v12 在 mm-image 的 6/8 req/s 和 deepseek-r1 的 4/6/8 req/s 出现明显长尾；v13 在这些点显著降低 TTFT p99。
- v13 并非所有点都优于两个固定基线：例如 mm-image 2/4 req/s 仍高于 fixed TP4；这应视为调度开销/分流差异，而不是吞吐结论。
- mm-image 的 fixed TP2 在 8 req/s 达到 103.405 秒，说明该点已经进入严重长尾；不能用它与正常负载点做简单平均。
- 四套系统的 nominal throughput 都约等于发送速率（成功请求数/120 秒）；这轮 fake-decode 只测 prefill/TTFT，不代表真实 decode 吞吐。

## MPS 与 chunked-prefill 口径

本矩阵使用生产 worker 参数 `enable_mps=True`、`chunked_prefill_size=8192`、`batch_max_tokens=16384`。这些端到端 TTFT 数字不用于推导 MPS slowdown；slowdown 使用 worker CUDA-event trace 的 `model_forward_ms`，并单独检查 TP2/TP4 forward 区间是否重叠。关于 chunked-prefill 的 no-chunk 对照，见 `260906-mps-slowdown-investigation.md` 和 `260906-MPS_perf_chunk_8192.md`。

## 数据和图

- 机器可读 CSV：`_/gpu_live_260906/260906-multi-system-compare/gpu_rate_ttft_compare.csv`；
- TTFT p99 图：`_/gpu_live_260906/260906-multi-system-compare/ttft_p99_servegen-*.png`；
- 汇总脚本：`test/benchmark/service/compare_gpu_sweeps_260906.py`。

v12 与 v13 来自不同时间的独立运行，固定基线也是独立运行；因此表格适合做方向和长尾比较，不应视为严格同秒配对实验。若要做论文级结论，还需要同一轮、同一 MPS daemon 生命周期下的随机交叉重复。
