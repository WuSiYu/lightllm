# 260906 v12/v13 真实 GPU 对照结果

本文只记录 v12/v13 配对快照；fixed TP2、fixed TP4、naive、`naive_switch` 的统一 GPU 对照已补充到 `260907-six-system-gpu-compare.md`。

## 实验范围

本轮在 8 张 H200 上完成 v13 的 ServeGen 实测矩阵：

- 数据集：`mm-image`、`m-large`、`deepseek-r1`；
- 请求速率：2、4、6、8 req/s；
- 每档生成时长：120 秒；
- 输出长度：按 benchmark 配置除以 10；
- 每档均先完成三类 prefill 和 decode 注册，再执行混合长短 warmup；
- v13 共 12/12 档完成，全部请求成功。

v12 对照使用此前同样的 3 个数据集、4 个速率和 120 秒配置，已有结果也是 12/12 档成功。TTFT 取 benchmark 结果中每个请求的首 token 延迟 `token_latencys[0]`，p99 使用线性插值百分位数计算。

## TTFT p99（秒）

| 数据集 | 速率 | v12 | v13 | v13/v12 |
|---|---:|---:|---:|---:|
| mm-image | 2 | 2.105 | 1.624 | 0.771 |
| mm-image | 4 | 4.185 | 1.967 | 0.470 |
| mm-image | 6 | 24.184 | 1.855 | 0.077 |
| mm-image | 8 | 147.424 | 3.161 | 0.021 |
| m-large | 2 | 1.911 | 1.329 | 0.696 |
| m-large | 4 | 3.539 | 1.930 | 0.545 |
| m-large | 6 | 3.320 | 1.552 | 0.468 |
| m-large | 8 | 2.177 | 1.587 | 0.729 |
| deepseek-r1 | 2 | 3.212 | 1.741 | 0.542 |
| deepseek-r1 | 4 | 10.164 | 5.129 | 0.505 |
| deepseek-r1 | 6 | 13.928 | 3.708 | 0.266 |
| deepseek-r1 | 8 | 16.255 | 3.885 | 0.239 |

v13 在全部 12 个配对点的 TTFT p99 都低于 v12：改善幅度约为 23%--98%。在 `m-large` 上改善约 27%--53%；在 `deepseek-r1` 上改善约 46%--76%；在 `mm-image` 的 rate=6/8 点，v12 已进入严重排队区间，v13 将 p99 从 24.2/147.4 秒降到 1.86/3.16 秒。

## 成功率和吞吐

两套结果的每个 JSON 都是 0 个失败请求。benchmark 生成的请求数随速率分别约为 235/477/717/955（deepseek）、232/471/704/947（m-large）和 235/477/720/958（mm-image）； nominal throughput 是成功数除以 120 秒，主要用于确认档位负载，不应当当作严格 drain-complete 吞吐。

## 结果解读和限制

这组结果支持 v13 在本配置、这三个 ServeGen 子集和 2--8 req/s 范围内改善 TTFT p99，但不能据此宣称普遍优于所有固定 TP 基线。原因包括：

- v12 和 v13 是不同时间启动的独立 GPU 运行，虽然使用相同数据集、随机种子、速率和时长，但不是同一进程内交替 A/B；
- v12 的 `mm-image` rate=6/8 明显过载，改善幅度应先解释为 v13 避免了该排队状态，而不是线性硬件加速；
- benchmark 的 nominal throughput 使用 120 秒名义窗口，尾部 drain 可能延长实际完成时间；
- 输出长度除以 10 是为了降低 decode 瓶颈，因此不能直接外推到原始输出长度；
- 当前 v12/v13 矩阵没有与 fixed TP2、fixed TP4、naive 或 `naive_switch` 做同批次交叉运行；六套系统的独立 GPU 结果虽已补齐，但仍不能和本表混写成同一轮严格 A/B 实测。

## 产物

- 对照 CSV：`_/gpu_live_260906/260906-v12-v13-compare/gpu_rate_ttft_compare.csv`；
- 三张 p99 对照图：同目录下 `servegen-*.ttft_p99.compare.png`；
- v13 原始 JSON：`test/benchmark/service/_/gpu_live_260906/v13_sweep_260906b/benchmarks/servegen/`；
- v12 原始 JSON：`_/gpu_live_260906/v12_sweep/servegen/`；
- 汇总脚本：`test/benchmark/service/compare_gpu_sweeps_260906.py`。
