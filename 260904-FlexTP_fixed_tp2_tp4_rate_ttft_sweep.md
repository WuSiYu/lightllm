260904-FlexTP fixed_tp2/fixed_tp4 请求率-目标 TTFT 全量扫描报告

## 结论

本次重新定义并测试了两个固定容量基线，删除了旧的 `fixed`（static-partition）基线：

- `fixed_tp2`：仅注册两个独立 TP2 Prefill 实例，分别使用 GPU `(0,1)` 和 `(2,3)`；所有请求都使用 TP2。
- `fixed_tp4`：仅注册一个 TP4 Prefill 实例，使用 GPU `(0,1,2,3)`；所有请求都使用 TP4。
- `naive`：保留共享拓扑 `TP2(0,1) + TP2(2,3) + TP4(0,1,2,3)`，按 4000 token 阈值路由并允许 MPS 重叠。

两种固定基线都没有 TP2/TP4 混合，因此模拟中的 `mps_overlap_wall_s` 恒为 0。固定基线不进行长度路由、动态 TP 切换或全局准入；只在已注册的同质 TP 实例中按 in-flight token 选择最轻实例。

## 实验规模

使用生产 selector 和生产 `ChunkedPrefillQueue` 的 CPU 离散事件模拟器，完成：

- workload：`servegen-mm-image`、`synthetic-5pct`；
- request rate：0.5、1、2、4、6、8、10、12、16、20、24 req/s；
- target TTFT：0.5、1、2、4 s；
- scheduler：`fixed_tp2`、`fixed_tp4`、`naive`、V3、V4、V5、V6、V7、V8、V9、V10、V11；
- 总计 2 x 11 x 4 x 12 = 1056 个组合；每个 scheduler 88 行。

每个 `(dataset, rate)` 只生成一条 workload trace，并在 target/scheduler 间复用。新版本使用 32 个进程，按 `(dataset, rate, target)` 分片，每个分片串行执行 12 个 scheduler；运行耗时 365.0 s。之前 8-worker 版本耗时 723.6 s，二者 CSV SHA-256 均为 `76f8890e83cea49e0b721c0e43699659469f86a7e29e741a7ea234e96825ef93`。

参数与此前扫描保持一致：chunked prefill 8192、Prefill batch token 上限 16384、bundle cap 8192、trigger 4096、窗口 20 ms、MPS overlap slowdown 2.0、fake Decode（固定 KV transfer 20 ms）、`max_inflight=64`、instance token credit 16384、schedule interval 5 ms、seed=0。模拟不调用 GPU kernel，结果是 timing model 下的调度对比，不是 H200 实测。

## Artifact 和曲线

完整结果位于 [rate_ttft_sweep.json](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-fixed-baselines-mp32-260904/rate_ttft_sweep.json)，表格位于 [rate_ttft_sweep.csv](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-fixed-baselines-mp32-260904/rate_ttft_sweep.csv)。

- ServeGen： [p50 线性](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-fixed-baselines-mp32-260904/servegen-mm-image.ttft_p50.png)、[p99 线性](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-fixed-baselines-mp32-260904/servegen-mm-image.ttft_p99.png)、[p50 log-y](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-fixed-baselines-mp32-260904/servegen-mm-image.ttft_p50.log.png)、[p99 log-y](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-fixed-baselines-mp32-260904/servegen-mm-image.ttft_p99.log.png)。
- Synthetic 5% long： [p50 线性](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-fixed-baselines-mp32-260904/synthetic-5pct.ttft_p50.png)、[p99 线性](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-fixed-baselines-mp32-260904/synthetic-5pct.ttft_p99.png)、[p50 log-y](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-fixed-baselines-mp32-260904/synthetic-5pct.ttft_p50.log.png)、[p99 log-y](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-fixed-baselines-mp32-260904/synthetic-5pct.ttft_p99.log.png)。

## 最大 SLO 负载点

下表是扫描网格中 `offered_ttft_slo_attainment >= 0.90` 的最大 request rate。拒绝或截止前未完成请求计入 offered 分母；`--` 表示没有满足点。

| dataset | target | fixed_tp2 | fixed_tp4 | naive | v3 | v4 | v5 | v6 | v7 | v8 | v9 | v10 | v11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ServeGen | 0.5 s | -- | -- | -- | 0.5 | -- | -- | -- | -- | -- | -- | -- | -- |
| ServeGen | 1 s | 4 | 2 | 2 | 4 | 6 | 6 | 2 | 2 | 2 | 2 | 2 | 2 |
| ServeGen | 2 s | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| ServeGen | 4 s | 8 | 8 | 8 | 8 | 8 | 8 | 12 | 8 | 8 | 8 | 8 | 8 |
| Synthetic 5% | 0.5 s | 6 | 2 | 8 | 4 | 6 | 6 | 4 | 8 | 8 | 8 | 8 | 6 |
| Synthetic 5% | 1 s | 8 | 6 | 12 | 8 | 10 | 12 | 10 | 12 | 12 | 12 | 12 | 8 |
| Synthetic 5% | 2 s | 12 | 10 | 12 | 10 | 16 | 16 | 20 | 12 | 12 | 12 | 12 | 12 |
| Synthetic 5% | 4 s | 16 | 12 | 16 | 12 | 20 | 20 | 24 | 16 | 16 | 16 | 16 | 20 |

固定基线的含义是“单一 TP 配置的容量上限”，不是长度分类策略。Synthetic 5% long 中，fixed_tp2 在高 rate 更有利于短请求并行，fixed_tp4 的单实例容量更容易形成队列；ServeGen 的输入更重，fixed_tp2/4 均在较低 rate 处进入过载区。V6 在 synthetic target=4 s 达到扫描内最高 24 req/s，V11 达到 20 req/s；这些结果仍应结合完成率、已完成请求 p99 和 offered SLO 一起解释。

## 高负载样例：Synthetic 5% long，rate=24 req/s，target=4 s

| scheduler | TTFT p50 (s) | TTFT p99 (s) | offered SLO | completed/offered |
|---|---:|---:|---:|---:|
| fixed_tp2 | 11.265 | 19.962 | 0.099 | 1000/1000 |
| fixed_tp4 | 18.107 | 31.826 | 0.034 | 1000/1000 |
| naive | 9.502 | 25.595 | 0.138 | 1000/1000 |
| v3 | 6.842 | 8.488 | 0.057 | 872/1000 |
| v4 | 3.737 | 8.092 | 0.497 | 906/1000 |
| v5 | 3.039 | 7.135 | 0.820 | 929/1000 |
| v6 | 2.404 | 31.368 | 0.959 | 1000/1000 |
| v7 | 4.271 | 43.006 | 0.383 | 1000/1000 |
| v8 | 9.170 | 27.458 | 0.182 | 1000/1000 |
| v9 | 6.245 | 29.587 | 0.216 | 1000/1000 |
| v10 | 6.035 | 34.373 | 0.238 | 1000/1000 |
| v11 | 3.136 | 35.803 | 0.820 | 1000/1000 |

fixed_tp2/fixed_tp4 和 naive 都完成全部请求，因此其 p99 反映完整队列；V4/V5 的较低 p99 部分来自未完成请求不进入已完成分位数。V6 在该点同时完成全部请求并获得最高 offered SLO，但 p99 仍受少数长请求拖尾。

## 复现

在仓库根目录执行：

```bash
MPLCONFIGDIR=/tmp/flex_tp_mpl_cache python -u test/benchmark/service/flex_tp_rate_ttft_sweep.py \
  --jobs 32 \
  --output-dir _/flex_tp_paper_analysis/rate-ttft-sweep-fixed-baselines-mp32-260904
```

`--jobs` 默认最多 32；任务按 `(dataset, rate, target_ttft)` 分片。主进程负责结果排序、CSV/JSON 和曲线写出，子进程只执行模拟，因此不会改变 trace pairing。

## 验证和局限

- `fixed_tp2` 和 `fixed_tp4` 各 88 行，`mps_overlap_wall_s` 唯一值均为 0.0；`fixed_tp4` 的 `tp2_long_request_count` 恒为 0，`fixed_tp2` 将长请求留在 TP2 是其定义的一部分。
- 新 mp32 CSV 与 8-worker 完整 CSV 字节级一致；没有残留的 `fixed` scheduler key。
- Decode 只保留 fake KV transfer，未模拟 Decode compute；MPS slowdown、Prefill latency curve 和 KV transfer 是显式 timing model。
- 当前每个点仍使用单个 seed；论文实验应增加多个 seed、burst workload、长请求比例和真实 H200 实测。
