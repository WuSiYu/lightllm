# 260906 MPS slowdown 调查

> **状态：仅限 MPS worker slowdown 审计。** `pd_fake_decode=True` 使客户端 serving latency/吞吐和完整 PD 结论作废；worker CUDA-event trace 可用于 MPS 竞争分析，但必须按每个 TP2 worker、TP4 worker 分别检查满载。当前资格结论以 `260907-MPS_pd_fake_decode_data_audit.md` 为准。

> **260907 更正：** 原表的 `tp2_busy_fraction` 是两个 TP2 端口的 union，并不能证明两个 TP2 worker 同时满载。因此原文约 3.2--3.3 倍只保留为“观察到的候选值”，不作为严格满载 slowdown 结论。

## 结论

之前看到的 3 倍以上结果不能直接解释为“单个 GEMM 的 MPS slowdown”，因为当时 HTTP 延迟包含排队，而且 worker 的 batch shape 发生了变化。不过，在 worker 侧直接用 CUDA event 包住 `model.forward` 后，确实观察到约 3.2--3.3 倍的同形状候选 slowdown。按 260907 的逐 worker 审计，原窗口不能证明两个 TP2 worker 都持续满载，因此该数值不作为严格满载结论；它与孤立 GEMM 也不是同一种工作负载。

对于 8192-token 的 TP4 请求，可比形状是 `batch_tokens=16384, batch_size=2`（两个 8192-token 请求）。TP4 单独运行时 forward 约 1.11 秒；在高 QPS 的 1024-token 短请求与相同 8192-token 背景请求同时运行时，该形状的 forward p50 约 3.52 秒，约 **3.2 倍候选值**。短请求长度为 2048 时 p50 约 3.70 秒，约 **3.3 倍候选值**。两者都必须先满足逐 worker 满载条件，当前记录未满足。

## 测量方法

- worker 在 overlap stream 上、紧邻 `model.forward` 前后记录 CUDA event。这个时间不包含 HTTP 排队、请求 admission、输入准备、sampling 和 post-processing，但包含另一个 TP 进程造成的真实 GPU stream 竞争。现有 `PERF - prefill` 的 `model` 字段计时范围更宽（输入准备前到 sampling 后），因此只作为日志对照，不冒充纯 forward 时间。
- 每个 TP rank 都会写入同一个 batch 的 trace 副本。分析按 worker port、batch shape、请求长度和 30 ms 起始时间窗口去重。
- 通过合并每个 worker 的 forward 区间计算 `tp2_busy_fraction`、`tp4_busy_fraction` 以及二者交集 `both_busy_fraction`。这些是 worker 执行区间指标，不是客户端 `inflight` 计数。
- 原始高 QPS 测试的 worker 注册参数确认是 `batch_max_tokens=16384`、`chunked_prefill_size=8192`。

## Trace 证据

| 测试 | TP2 busy | TP4 busy | 同时 busy | TP4 forward p50 | TP4 batch token p50 |
|---|---:|---:|---:|---:|---:|
| solo 8192 | 0% | 81.6% | 0% | 1108 ms | 16384 |
| overlap 32 + 8192 | 9.6% | 76.7% | 9.2% | 1122 ms | 16384 |
| overlap 128 + 8192 | 33.3% | 84.2% | 33.0% | 1124 ms | 16384 |
| overlap 1024 + 8192 | 90.5% | 85.5% | 85.4% | 3519 ms | 16384 |
| overlap 2048 + 8192 | 91.2% | 57.8% | 57.8% | 3696 ms | 16384 |
| overlap 8192 + 256 | 51.9% | 87.7% | 51.6% | 1112 ms | 16384 |

前几行说明：短请求的发送 QPS 很高，并不自动意味着每个 TP2 worker 满载。表中是旧的 TP2 union 统计；逐 worker、过滤 tailing 后，只有 1024/2048 能截出合格的长饱和段，32/64/128 的重叠片段过短。

## 为什么会超过 GEMM slowdown

GEMM 测试固定一个算子和一个矩阵 shape；serving 测试不是这样。chunked-prefill scheduler 会把请求聚合成大 batch，高速短请求流会在 TP2 上形成约 16k token 的 batch，同时 TP4 执行 16k token batch。此时竞争的是两个完整 model forward，包括 attention/KV、通信、cache/allocator 行为以及多层 kernel 调度。因此不能把结果解释为“一个 GEMM 在另一个进程存在时变慢了多少”。

Trace 中有直接对照：TP4 的 16384-token batch 单独运行约 1.1 秒；当 TP2 同时执行 16--17k-token batch 时，TP4 同形状 batch 会达到约 3.5--4.8 秒。这与 chunked-prefill 的 batch packing 放大效应以及完整模型资源竞争相符，并不说明孤立 GEMM 的 2 倍结果错误。

## 对 chunked-prefill 干扰的复查

已经补查了两组 worker CUDA-event trace：生产配置 `chunked_prefill_size=8192`，以及保持 `batch_max_tokens=16384`、仅设置 `--disable_chunked_prefill` 的 no-chunk 配置。后者不是把 batch 上限放大到 65536 的旧尝试，因此可以作为方向性的单变量对照。

在 no-chunk trace 中，TP4 的 `batch_tokens=8192, batch_size=1` 事件为 32 个，forward 中位数约 **564 ms**；在 `chunked_prefill_size=16384` 的对照中，同形状事件为 40 个，中位数约 **571 ms**。两者都接近生产配置 fixed TP4 的 **547 ms**，没有出现 3 倍级别的固定基线膨胀。相反，3 倍级别只出现在 TP2/TP4 同时满载时的 overlap trace：同形状 TP4 forward 从约 1.1 秒升到约 3.5--3.7 秒。

因此目前最可信的解释是：**chunked-prefill 负责决定 chunk/batch 的形状和切分节奏，但 3 倍 slowdown 的主因是 MPS 下两个大模型 forward 的同时资源竞争，而不是 chunk slicing 本身。** 仍需注意，no-chunk 这次覆盖的 overlap 长度不如生产配置完整，不能据此声称“chunked-prefill 完全没有影响”；它只能排除“8192 切分单独导致固定 3 倍变慢”这一解释。

另外，`PERF - prefill` 的 `model` 字段并非纯 `model.forward`：计时从 `prepare_prefill_inputs` 前开始，到 forward 后的 sampling 区间结束。最终 slowdown 数值应以 `LIGHTLLM_MPS_TRACE_DIR` 的 CUDA-event `model_forward_ms` 为准；PERF 日志只用于按 chunk 做形状对齐和检查调度阶段。

原始 trace 分析位于 `_/mps_real_260906/highqps_trace_analysis.json`，复现脚本为 `test/benchmark/service/analyze_mps_trace_260906.py`。
