# 260906 MPS 批处理延迟测量

> **状态：仅限 MPS worker slowdown 审计。** `pd_fake_decode=True` 使客户端 serving latency/吞吐和完整 PD 结论作废；worker 侧执行区间只有在每个 worker 满载检查通过时才可作为 MPS slowdown 证据。当前三组资格结论以 `260907-MPS_pd_fake_decode_data_audit.md` 为准。

> **260907 更正：** 原表的 TP2 聚合是两个 TP2 端口的 union，不能证明 TP2(0,1) 与 TP2(2,3) 各自持续满载。FlexTP all-mix 的 TP2 两端口分别约 17.9% 和 16.9%，因此不能据此宣称三组 overlap slowdown。

本实验记录 worker 侧 CUDA `model.forward` 的批处理时间，不是客户端请求延迟。不同 TP rank 会记录同一批次的副本，分析时按 worker、结束时间、`batch_tokens` 和 `batch_size` 去重。

| 条件 | 请求长度 | worker 拓扑 | 去重后批次数 | batch token 中位数（最大） | batch size 中位数（最大） | forward 时间 p50 / p90 |
|---|---|---|---:|---:|---:|---:|
| fixed TP2 | 32/64/128/256/1024/2048 混合 | TP2(0,1)、TP2(2,3) | 12 | 6,976（16,384） | 17（31） | 814.6 / 1,905.8 ms |
| fixed TP4 | 4096/8192/16384 混合 | TP4(0,1,2,3) | 12 | 12,288（24,576） | 2（2） | 843.1 / 1,259.6 ms |
| FlexTP naive | 全长度混合，阈值 4000 | TP2 + TP4 | 48 | 按形状统计，TP4 为主 | 按形状统计，多数为 2 | 1,104.3 / 2,320.5 ms |

FlexTP 的 TP2 与 TP4 在约 18:21:01--18:21:07 UTC 有时间重叠，因此这确实是 overlap 条件。但 TP2 先排空、TP4 后续仍在处理，所以表中的 FlexTP 聚合行不能作为“整个窗口持续满载”的严格证据；不同 batch shape 的聚合 p50 也不能直接当作 slowdown。严格结论应使用高 QPS worker-forward 分析以及 `260906-MPS_perf_chunk_8192.md`。

原始 trace、请求元数据和去重结果位于 `_/mps_batch_260906/`。
