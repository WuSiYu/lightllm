260903-FlexTP 请求率-目标 TTFT 全量扫描报告（历史版）

> 260904 更新：本文中的 `fixed` 是已废弃的 static-partition 旧基线，不能代表当前定义。当前固定基线改为 `fixed_tp2`（仅两个 TP2）和 `fixed_tp4`（仅一个 TP4），请以 [260904 fixed_tp2/fixed_tp4 全量扫描报告](</mtc/wusiyu/work/LightLLM-flex-tp/260904-FlexTP_fixed_tp2_tp4_rate_ttft_sweep.md>) 及其 artifact 为准。

## 结论摘要

本次使用离散事件模拟器加载生产调度器和生产 `ChunkedPrefillQueue`，完成了 968 个组合：

- 2 个 workload：`servegen-mm-image`、`synthetic-5pct`。
- 11 个请求率：0.5、1、2、4、6、8、10、12、16、20、24 req/s。
- 4 个目标 TTFT：0.5、1、2、4 s。
- 11 个调度器：`fixed`、`naive`、V3、V4、V5、V6、V7、V8、V9、V10、V11。

每个数据集和请求率只生成一条 trace，并在四个目标和十一个调度器之间复用，因此比较是 paired 的。v3-v9 的历史 792 行、v10 的 88 行和 v11 的 88 行已合并为 968 行；未使用 GPU。完整原始结果、trace 指纹和配置保存在 [v3-v11 rate_ttft_sweep.json](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904/rate_ttft_sweep.json)，表格版本见 [v3-v11 rate_ttft_sweep.csv](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904/rate_ttft_sweep.csv)；原始 v3-v9 结果仍保留在 [260903 rate_ttft_sweep.json](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-260903/rate_ttft_sweep.json)。

从扫描结果看，V4/V5/V6 在合成 5% 长请求 workload 上改善最明显；其中 V6 在目标 TTFT 为 2 s 和 4 s 时仍达到最高的扫描内 SLO 负载点。V10 保留 v9 的 EEVDF/deadline 保护，并在部分点降低尾延迟；V11 取消长度硬阈值，按预测完成时间、TP footprint 和 MPS overlap price 动态选择 TP，在 ServeGen target=4 s 的高负载区间显著提高 offered SLO（rate=24 时为 0.399）。V4/V5 的 p50/p99 通常更低，但高负载时可能通过拒绝请求换取较低的完成请求延迟，不能只看 p99。ServeGen 多模态 workload 的长请求和到达形态更不利，所有策略的可行 SLO 负载点明显下降。

## 实验口径

### 模拟器和调度器

入口脚本是 [flex_tp_rate_ttft_sweep.py](/mtc/wusiyu/work/LightLLM-flex-tp/test/benchmark/service/flex_tp_rate_ttft_sweep.py)，底层 step 模拟器是 [flex_tp_step_sim.py](/mtc/wusiyu/work/LightLLM-flex-tp/test/benchmark/service/flex_tp_step_sim.py)。每个请求经历到达、全局准入、PD bundle、instance prefill step、KV transfer 和完成等虚拟时间事件；调度决策调用仓库中的生产 selector。

本次参数固定为：

- chunked prefill size 8192，prefill batch token 上限 16384；bundle token cap 8192、trigger 4096、窗口 20 ms。
- 普通 PD 路径的 MPS 重叠 slowdown=2.0。
- `fake_decode=True`：不模拟 decode step，只保留固定 20 ms 的 KV transfer；因此 decode 被视为非瓶颈，符合本轮测试约定。
- naive 基线按输入长度阈值 4000 选择 TP=2/TP=4；`fixed` 与 `naive` 在本配置下作为两个显式 baseline 名称保留。V10 也保留该阈值作为硬路由基线；V11 不使用长度阈值，`long_threshold` 仅为兼容旧 CLI 参数。
- `max_inflight=64`、instance token credit=16384、schedule interval=5 ms，随机种子为 0。

`servegen-mm-image` 使用仓库中的 ServeGen `mm-image` 数据和 60 s constant-rate 生成；`synthetic-5pct` 生成 1000 个请求，其中 5% 为 1000--20000 token 的长请求，其余为 100--1000 token，输出长度为 50--500 token。请求率扫描只改变到达时间，不改变同一 `(dataset, rate)` 下的请求内容和顺序。

### 指标解释

- `ttft_p50_s`、`ttft_p99_s`：只在已完成 prefill 的请求上统计；被拒绝或在 drain 截止前未完成的请求不进入分位数。
- `completion_fraction`：完成请求数 / offered 请求数。
- `offered_ttft_slo_attainment`：满足 `TTFT <= target_ttft` 的请求数 / offered 请求数，拒绝请求计入分母。这是判断“服务全部到达请求”时的主要指标。
- 下文的“最大 SLO 负载点”是扫描网格中 `offered_ttft_slo_attainment >= 0.90` 的最大请求率，不是连续容量估计；没有满足点记为 `--`。

因此，在高负载下出现“p99 较低但 completion_fraction 较低”是有意保留的结果，不能当作优于完整服务的证据。

## 曲线和原始数据

每张图均为 2x2 facet，四个子图分别对应 target TTFT=0.5/1/2/4 s；横轴为 request rate，纵轴为已完成请求的 TTFT。

- ServeGen： [p50 线性](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904/servegen-mm-image.ttft_p50.png)、[p99 线性](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904/servegen-mm-image.ttft_p99.png)、[p50 log-y](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904/servegen-mm-image.ttft_p50.log.png)、[p99 log-y](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904/servegen-mm-image.ttft_p99.log.png)。
- Synthetic 5% long： [p50 线性](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904/synthetic-5pct.ttft_p50.png)、[p99 线性](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904/synthetic-5pct.ttft_p99.png)、[p50 log-y](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904/synthetic-5pct.ttft_p50.log.png)、[p99 log-y](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904/synthetic-5pct.ttft_p99.log.png)。

线性图适合观察过载后的绝对长尾；log-y 图适合比较低负载和目标线附近的策略差异。

## 最大 SLO 负载点

下表给出每个数据集和目标 TTFT 下，扫描网格内 `offered_ttft_slo_attainment >= 0.90` 的最大请求率（req/s）：

| dataset | target | fixed | naive | v3 | v4 | v5 | v6 | v7 | v8 | v9 | v10 | v11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ServeGen | 0.5 s | -- | -- | 0.5 | -- | -- | -- | -- | -- | -- | -- | -- |
| ServeGen | 1 s | 2 | 2 | 4 | 6 | 6 | 2 | 2 | 2 | 2 | 2 | 2 |
| ServeGen | 2 s | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 |
| ServeGen | 4 s | 8 | 8 | 8 | 8 | 8 | 12 | 8 | 8 | 8 | 8 | 8 |
| Synthetic 5% | 0.5 s | 8 | 8 | 4 | 6 | 6 | 4 | 8 | 8 | 8 | 8 | 6 |
| Synthetic 5% | 1 s | 12 | 12 | 8 | 10 | 12 | 10 | 12 | 12 | 12 | 12 | 8 |
| Synthetic 5% | 2 s | 12 | 12 | 10 | 16 | 16 | 20 | 12 | 12 | 12 | 12 | 12 |
| Synthetic 5% | 4 s | 16 | 16 | 12 | 20 | 20 | 24 | 16 | 16 | 16 | 16 | 20 |

几个直接观察：

1. 在 synthetic-5pct、target=2 s/4 s 时，V6 的最大扫描点分别为 20/24 req/s，优于其他策略；V4/V5 次之。
2. ServeGen 的 target=0.5 s 极其严格，除 V3 在 0.5 req/s 外没有策略达到 90% offered SLO；这说明该 workload 的多模态输入长度和到达过程使 0.5 s 目标不现实。
3. V7/V8/V9/V10 在本组参数下没有超过 V6 的 SLO 负载点；V11 在 synthetic-5pct、target=4 s 上达到 20 req/s，超过 V9/V10 的 16 req/s，但仍低于 V6 的 24 req/s。各版本的价值仍需在更丰富的短长混合、突发和公平性 workload 上单独评估。

## 高负载样例

### Synthetic 5% long，request rate=24 req/s，target TTFT=4 s

| scheduler | TTFT p50 (s) | TTFT p99 (s) | offered SLO | completed/offered |
|---|---:|---:|---:|---:|
| fixed | 9.502 | 25.595 | 0.138 | 1000/1000 |
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

这个点清楚展示了指标取舍：V5 的完成请求 p50/p99 最低，但有 7.1% 请求未完成；V6 完成全部请求且 offered SLO=0.959，但 p99 受少数长请求拖高。若论文目标是“all offered requests 的 SLO”，V6 更有说服力；若目标是 tail latency under admission control，则必须同时报告拒绝率和 admission policy。

### ServeGen，request rate=24 req/s，target TTFT=1 s

baseline `fixed/naive` 完成 1435/1435，但 p50/p99=64.858/87.432 s，offered SLO=0.002。V3/V4/V5 的已完成请求 p99 约 5.8 s，但只完成 740/1435、625/1435、625/1435；V6/V7/V9 完成全部请求，但 p99 约 120--132 s；V8 的 p99 为 92.222 s；V10 为 35.430/119.626 s、offered SLO=0.007，V11 为 17.006/110.244 s、offered SLO=0.013，均完成 1435/1435。该点不应以单一 p99 排名，而应按完成率、offered SLO 和 p99 联合解释。

## 复现

在仓库根目录执行以下命令可重跑同一矩阵（CPU 模拟，预计约 36 分钟，时间随机器而变）：

```bash
python -u test/benchmark/service/flex_tp_rate_ttft_sweep.py \
  --output-dir _/flex_tp_paper_analysis/rate-ttft-sweep-v11-final-260904
```

脚本会重新生成 v3-v11 的 CSV、JSON、线性图和 log-y 图。历史 v3-v9 artifact 不会被覆盖；若需保留运行日志，可在命令末尾追加 `2>&1 | tee sweep.log`。

## 局限和后续测试

- 这是生产 selector 的 CPU 离散事件模拟，不是 H200 实测；MPS slowdown 和 KV transfer 使用显式模型参数，不能替代真实 kernel/通信测量。
- decode 被简化为非瓶颈，当前结论主要针对 prefill 调度、TP 放置、bundle 和 admission。
- 每个点只有一个固定随机种子，尚未给出置信区间；论文实验应至少增加 3--5 个 seed，并报告均值、p50/p99 和误差范围。
- 请求率网格在 8--24 req/s 之间较稀疏，若要精确估计拐点，应围绕 SLO 从 1 req/s 间隔细扫，并增加 burst、短请求连续到达和不同长请求比例。
- 后续应把 `offered SLO`、`completion_fraction`、拒绝原因、prefill token goodput 和公平性（短请求 p95、长请求 p95）作为联合指标，避免 admission control 造成的 survivorship bias。
