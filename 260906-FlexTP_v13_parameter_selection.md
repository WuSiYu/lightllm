# 260906-FlexTP v13 参数选择记录

## Default

当前 v13 默认参数为：`long_threshold=12000`、`latency_scale=1.0`、`routing_cost_weight=2.0`、`tp4_service_ratio_limit=0.60`、`tp4_pressure_threshold=1.0`。v12 参数和性能模型没有被改写。

## DeepSeek threshold ablation

两组实验使用同一 DeepSeek trace、输出长度除以 10、request rate 12/16/24 req/s、目标 TTFT 2 s。

| v13 threshold | rate | v13 p99 TTFT | v13 offered SLO | v12 p99 TTFT | v12 offered SLO |
|---:|---:|---:|---:|---:|---:|
| 4000 | 12 | 21.973 | 0.933 | 9.406 | 0.647 |
| 12000 | 12 | 12.108 | 0.844 | 9.406 | 0.647 |
| 4000 | 16 | 32.261 | 0.928 | 11.924 | 0.323 |
| 12000 | 16 | 12.255 | 0.397 | 11.924 | 0.323 |
| 4000 | 24 | 67.630 | 0.910 | 54.705 | 0.163 |
| 12000 | 24 | 56.614 | 0.213 | 54.705 | 0.163 |

证据来源：

- `_/flex_tp_paper_analysis/260905-v13-deepseek-threshold-4000/rate_ttft_sweep.csv`
- `_/flex_tp_paper_analysis/260905-v13-deepseek-threshold-12000/rate_ttft_sweep.csv`

结论是：12000 阈值相对 4000 阈值改善 v13 自身的 p99，尤其在 12/16 req/s；但在该 reasoning trace 上 v12 仍有更低 p99，v13 的价值主要是保留更高的 offered-SLO 区间和可调的 TP4 spill，而不是宣称全面优于 v12。真实 H200 重跑仍是最终参数确认门槛。

补充的 mm-image 阈值对照位于 `_/flex_tp_paper_analysis/260906-mm-image-threshold-16000/rate_ttft_sweep.csv`；在该 trace 上 16000 与 12000 的 v13 p99 基本相同，因此没有足够证据把默认值改到 16000。

Mooncake 高 rate 的 16000 对照位于 `_/flex_tp_paper_analysis/260906-mooncake-threshold-16000/summary.md`；它将 v13 p99 降低约 4--10 秒，但 offered-SLO 同时下降。因此 16000 作为可选 profile 保留，默认仍为 12000，等待真实 GPU 结果确认。

其余 ratio/cost/latency-scale grid 主要是小规模 synthetic smoke，结果差异不足以支持进一步改默认值；它们只作为敏感性记录，不作为 GPU 性能结论。

## 260906 Mooncake 小网格复核

在同一 `mooncake_trace.jsonl`、输出长度除以 10、MPS slowdown=2、目标 TTFT=2 s
下，用 16/24 req/s 各 100 个请求复核 v13 参数。结果为 timing-model smoke，
不是 GPU wall-clock：

| threshold | latency scale | pressure threshold | rate=16 p99/SLO | rate=24 p99/SLO |
|---:|---:|---:|---:|---:|
| 12000 | 0.9 | 0.5 | 54.999 / 0.040 | 56.196 / 0.020 |
| 12000 | 1.0 | 1.0 | 55.344 / 0.070 | 57.211 / 0.050 |
| 16000 | 0.9 | 0.5 | 53.534 / 0.030 | 54.777 / 0.030 |
| 16000 | 1.0 | 1.0 | 53.324 / 0.030 | 54.882 / 0.040 |

16000 阈值对 p99 有小幅改善，但 offered-SLO 没有一致提升；latency scale=0.9
也没有稳定支配 scale=1.0。因此默认仍保持 `threshold=12000`、`latency_scale=1.0`、
`pressure_threshold=1.0`，等待真实 8 卡配对实验再校准。
