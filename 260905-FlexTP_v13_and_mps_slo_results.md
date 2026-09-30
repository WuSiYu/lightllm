# 260905-FlexTP v13 与 MPS SLO slowdown 实验

本文是早期 v13/MPS 记录；最新六套系统 GPU 对照及其有效性边界见 `260907-six-system-gpu-compare.md`，不要将本文的历史矩阵描述当作最新完成状态。

## v13 改动

v13 复制 v12 的 lease、bundle、ACK、deadline 和连续 TP 选择生命周期，只在 v13 增加两类策略变化；相关阈值和代价参数均可校准：

- TP2 token credit 达到压力阈值，或请求接近 TTFT deadline 时，允许 TP4 接收原本未通过固定 service-ratio guard 的请求；
- `latency_scale` 只作用于 v13 的预测路径，供新硬件 profile 校准，默认值为 1.0；v13 独立使用 12000-token 长请求 TP4 fallback 阈值，v12 仍保持 4000。

默认启动入口是 `start-cluster4_70b_p22.4d4_mps_flex_v13.sh`，v12 文件没有被改写。生产选择器名为 `flex_tp_v13`，参数包括 `--flex_tp_v13_long_threshold`、`--flex_tp_v13_latency_scale`、`--flex_tp_v13_routing_cost_weight`、`--flex_tp_v13_tp4_service_ratio_limit` 和 `--flex_tp_v13_tp4_pressure_threshold`。

## 离线验证

- `python -u test/benchmark/service/flex_tp_step_sim.py --self-test`：V3--V13、fixed TP2/TP4、naive 和 fake-decode self-test 全部通过。
- ServeGen `m-large`、`deepseek-r1`、`mm-image` 均可从测试目录加载；数据路径现在按仓库中的 `_/ServeGen` 解析，不依赖当前工作目录。
- benchmark 客户端的 `ClientPool` 与 `generate_workload` 现在共享同一个 ServeGen cwd；从仓库根目录回归生成 `m-large`/`deepseek-r1`/`mm-image` 分别得到 3/6/35 个请求。
- 客户端默认每请求 HTTP timeout 为 180 s；`loop4-sg3-v13.sh` 额外清理所有 proxy 环境变量并用 `timeout` 包住每个 benchmark，避免客户端永久等待。
- 旧的 `loop4-sg3.sh` 也增加了同样的请求/进程超时保护，原始版本保存在 `loop4-sg3.sh.before_timeout`；遇到单档失败会记录并继续后续 request rate。
- `loop3-sub5-2.sh` 已同步加入代理清理、请求/进程超时和失败后继续机制，原始版本保存在 `loop3-sub5-2.sh.before_timeout`。
- v12 启动脚本的 client 窗口已切换到长短请求混合 warmup，旧版本保存在 `start-cluster4_70b_p22.4d4_mps_flex_v12.sh.before_mixed_warmup`；v12 selector 和性能模型未改动。
- 混合 warmup 增加默认 600 秒总 deadline（可用 `WARMUP_TIMEOUT_S` 调整），启动窗口再用外层 timeout 保护；服务注册失败不会无限等待。
- `wait_warmup_mixed.sh` 的长请求长度现在可由 `WARMUP_LONG_INPUT_TOKENS` 控制，v12 默认 4096，v13 启动窗口设为 16000，确保 v13 的 12000-token fallback 在服务启动前确实被触发过。
- LightLLM SSE 客户端改为按换行缓冲完整 `data:` 帧，兼容网络分片/多帧合并，并对空流快速报错；模拟分片流回归通过。
- benchmark 客户端在所有请求失败、没有成功样本时返回非零退出码；空服务回归（100/100 失败）已验证，loop 不会把该档误记为成功。
- Mooncake/超时相关客户端改动保留原始版本于 `test/benchmark/service/benchmark_serving_chat_req_rate.py.before_mooncake_260906`；loop 脚本也保留了各自的 `before_timeout` 备份。
- v13 启动脚本在创建 tmux/MPS 窗口前用 10 秒 `nvidia-smi -L` 检查驱动，硬件不可用时立即退出，避免伪装成服务注册超时。
- naive 4k 启动脚本也已切换到短/长 mixed warmup、完整 proxy 清理、有限 warmup timeout 和 GPU 预检；原脚本保留为 `start-cluster3_70b_p22.4d4_mps_flex_naive_4k.sh.before_mixed_warmup_260906`。
- 离线 rate/TTFT sweep 的 worker context 已改为 `spawn`；重型 CUDA/NIXL 导入后不再 `fork`，避免首个 shard 前的 multiprocessing 死锁。Matplotlib 仅在父进程写图时延迟加载，ABI 不匹配时自动保留 CSV/SVG fallback。
- live fake-decode harness 的 mixed warmup/TTFT 分桶现在按 policy 使用阈值：v12 仍为 4000，v13 为 12000，并确保 v13 warmup 至少发送一个超过 12000 token 的请求；默认最大输入调整为 16384 以覆盖该 fallback。
- 新增不依赖 pytest 的 `test/test_flex_tp_selector_v13_smoke.py`，直接执行可验证 v13 factory、节点注册、选择、snapshot 和 completion 生命周期。
- 新增 `test/benchmark/service/run-v13-live-sweep.sh`：预检 8 张可见 GPU、清理 proxy、启动 v13、等待 mixed warmup 完成后调用 ServeGen 多 rate loop，并在退出时清理自身 tmux session；同名 session 已存在时会拒绝接管。
- 新增配对的 `test/benchmark/service/run-v12-live-sweep.sh`；`loop4-sg3.sh` 支持通过 `RATES` 和 `LOG_DIR` 固定档位/输出目录，恢复 GPU 后可直接复用同一 runner 对比 v12 与 v13。两者在 `nvidia-smi -L` 失败或可见 GPU 少于 8 张时均快速非零退出。
- `loop4-sg3.sh`、v13 ServeGen/Mooncake loop 和 `loop3-sub5-2.sh` 增加可配置的 `BENCHMARK_KILL_AFTER_S`；默认保持 20 秒，故障注入用 1 秒宽限即可在单档 timeout 后继续后续 rate。原版本分别保存在 `*.before_kill_after_260906`。
- 新增 `run-fixed-fake-decode-live-sweep.sh`，用 `PROFILE=fixed_tp2` 或 `fixed_tp4` 启动对应 Prefill 拓扑，完成相同 mixed warmup 后复用 `benchmark_serving_chat_req_rate.py` 跑 ServeGen 多 rate；fixed runner 和 worker 启动脚本均有 GPU 预检，原 worker 脚本保留为 `*.before_live_runner_260906`。

## 已有 H200 基线

> **作废：** [260904-FlexTP v12：连续预测完成时间与 H200 实测](</mtc/wusiyu/work/LightLLM-flex-tp/260904-FlexTP_v12_design_and_gpu_results.md>) 中的 H200 四卡 fake-decode 基线使用了 `pd_fake_decode=True`，不能作为真实 Decode 或请求 latency 证据。

另在 260906 完成了 v12/v13 的真实 GPU ServeGen 配对矩阵：三个子数据集（mm-image、m-large、deepseek-r1）和 2/4/6/8 req/s 共 24 个 JSON，全部 0 失败；详细结果见 `260906-v12-v13-gpu-compare.md`。该矩阵使用 `pd_fake_decode=False`，不因上述 fake-decode 基线作废；它也不等价于已经完成 fixed TP2、fixed TP4、naive、`naive_switch` 的完整同批次 GPU 矩阵。

## MPS slowdown SLO 实验

脚本：`test/benchmark/service/flex_tp_mps_slo_down.py`。

默认维度为输入长度 256/1024/4096/8192/16384、MPS slowdown 1/1.25/1.5/2/3、request rate 4/8 req/s，输出 CSV 和可选 PNG。当前无 GPU 的环境已完成 200 个离线离散 step 组合，最新数值结果见 [mps_slo_down.csv](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-mps-slo-down/mps_slo_down.csv>)，代表性数字见 [summary.md](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-mps-slo-down/summary.md>)；无 Matplotlib 时脚本同时生成 [TTFT p99 SVG](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-mps-slo-down/mps_slo_down_ttft_p99_s.svg>) 和 [SLO SVG](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-mps-slo-down/mps_slo_down_offered_slo.svg>)。旧的 260905 目录保留作为前一轮结果。该实验使用 fake Decode 和生产 ChunkedPrefillQueue，`mps_overlap_wall_s`、TTFT p50/p99、SLO attainment、完成吞吐和 TP4 request share 都保留在 CSV 中；它是 timing model 结果，不是 GPU wall-clock 测量。
代表性结果（rate=4 req/s）：输入 8192 时 v12/v13 的 TTFT p99 都从 slowdown=1 的 4.61 s 增至 slowdown=3 的 50.58 s；输入 4096 时 v13 从 1.76 s 增至 3.82 s，而 fixed TP2/v12 在该点约为 1.76 s；输入 256 时所有策略均约 0.09–0.11 s，说明 slowdown 主要影响长请求的共享 GPU 路径。

GPU 驱动恢复后，已进行独立的 H200 MPS contention 实测，结果见 [260906-MPS_slowdown_gpu_results.md](</mtc/wusiyu/work/LightLLM-flex-tp/260906-MPS_slowdown_gpu_results.md>)。GPU0 上两个 width=8192 的 CUDA prefill proxy worker 并发时，256/1024/4096/8192/16384 token 的 wall-clock slowdown 分别为 2.143/2.087/1.970/1.996/2.010。该测量用于校准共享计算资源的 slowdown，不等价于 70B 服务端到端 TTFT。

## ServeGen 子数据集

扩展后的 sweep 支持 `servegen-lang-large` 和 `servegen-deepseek-r1`，并对所有 ServeGen 输出提供 `--servegen-output-divisor`；v13 实测循环位于 `test/benchmark/service/loop4-sg3-v13.sh`，默认除以 10 以避免 Decode 成为瓶颈。离线结果写入 [260905-v13-servegen-subdatasets](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260905-v13-servegen-subdatasets>)。

补充的 `servegen-mm-image` paired sweep（50 点，rate=4/8/12/16/24，目标 TTFT=2/4 s）没有显示 v13 的稳定收益；目标 4 s、offered-SLO 0.90 时，v6 可达 12 req/s，而 fixed TP2、fixed TP4、v12 和 v13 均为 8 req/s；目标 2 s 时所有策略最高为 4 req/s。详细表格见 [mm-image summary](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-servegen-mm-image-v13-sweep/summary.md>) 和 [rate_ttft_sweep.csv](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-servegen-mm-image-v13-sweep/rate_ttft_sweep.csv>)。该负结果支持把后续真实数据集重点放在 Mooncake/WildChat，而不是宣称 v13 普遍优于 v6。

独立 MPS slowdown 实测、v12/v13 GPU ServeGen 矩阵和离线 timing-model 结果已分别保存，三者不混用。v13 GPU runner 已验证会先完成混合长短 warmup，再运行 ServeGen 多 rate 循环；初始的 tmux quoting 和相对日志路径问题已修复。

截至 260905，`servegen-lang-large` 和 `servegen-deepseek-r1` 的低/中负载配对 sweep 已完成（90 行，见 [rate_ttft_sweep.csv](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260905-v13-servegen-subdatasets/rate_ttft_sweep.csv>)）。在 2/4/8 req/s、1/2/4 s 目标范围内，五个策略的最大达标率网格点均为 8 req/s；reason workload 在 8 req/s、2 s 目标下 fixed TP4 的 p99 为 1.395 s，v12/v13 为 1.636 s，language workload 在同一点 v12/v13 为 0.765 s。也就是说，这两个短 sweep 尚未证明 FlexTP 优于 fixed 基线，不能据此宣称收益。

高请求率补充 sweep（12/16/24 req/s）结果位于 [260905-v13-servegen-highrate](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260905-v13-servegen-highrate>)，应优先按 SLO 达标 rate 和 offered p99 解释，不把单一已完成请求 p99 当作唯一排名。

高 rate sweep 已完成 90 行。`deepseek-r1` 在 24 req/s、2 s 目标下，V6 的 offered SLO 为 0.927，v12 为 0.186；`servegen-lang-large` 在 24 req/s、2 s 目标下，v12 为 0.577，fixed TP2 为 0.328。原始 sweep 使用的是压力 spill 调参前的 v13，v13 在 deepseek 该点为 0.551；随后加入了超长请求 TP4 fallback。最终默认阈值已调为 12000（`input_tokens > 12000` 时 TP2 不再作为候选），以改善 v13 p99，需在 GPU 恢复后或单独离线 sweep 重跑确认最终参数。该 fallback 的目标是吸收 V6 在 reasoning workload 上已经显示出的长请求收益，同时保留短请求的连续路由。

最终 fallback 参数已通过 synthetic 12/16 req/s、v12/v13 配对 smoke（4 行，见 [rate_ttft_sweep.csv](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260905-v13-default-threshold-12000/rate_ttft_sweep.csv>)）。本轮新增的 2/4/6/8 req/s 真实 GPU 结果见 `260906-v12-v13-gpu-compare.md`；更高 12/16/24 req/s 的 ServeGen high-rate 结果仍属于另一组压力实验，不能和本矩阵混写。
在同一 synthetic trace 上，v13 从 4000 调到 12000 后，rate12 的 p99 从 7.02 s 降至 6.54 s，rate16 从 6.75 s 降至 6.67 s；对应 offered SLO 分别为 0.90 和 0.65，说明这是 p99/SLO 的折中而非无条件收益。

参数选择的逐点对照和限制见 [260906-FlexTP_v13_parameter_selection.md](</mtc/wusiyu/work/LightLLM-flex-tp/260906-FlexTP_v13_parameter_selection.md>)。

由于 ServeGen 子集的离线结果尚未显示稳定收益，补充了 Mooncake 路径：`benchmark_serving_chat_req_rate.py` 支持 `--mooncake-output-divisor`，可用 [loop4-mooncake-v13.sh](</mtc/wusiyu/work/LightLLM-flex-tp/test/benchmark/service/loop4-mooncake-v13.sh>) 直接对真实 JSONL trace 做 v13 请求率 sweep；默认输出长度除 10，旧默认仍为 1。

仓库内的 `mooncake_trace.jsonl` 已完成 4/8/12 req/s、1/2/4 s 目标的 36 个离线 paired 点，结果见 [rate_ttft_sweep.csv](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260905-mooncake-v13-sweep/rate_ttft_sweep.csv>)。该 trace 的长输入压力很高，所有策略在这些 request rate 下均未达到 0.9 SLO；v13 在 4 s 目标下 offered SLO 约 0.50，高于 v12 约 0.25 和 fixed TP2 约 0.19，但 p99 仍较高，不能据此宣称普遍收益。

补充的 Mooncake 高 rate sweep（16/24/32 req/s，30 点）显示这是明确的 overload boundary：目标 4 s、rate 16 时 v6 offered SLO 为 0.299，v13 为 0.034，v12 为 0.025；三档 rate 下没有策略达到 0.9 SLO。v13 的 pressure/long-request spill 相对 v12 有小幅 SLO 改善，但 p99 仍高于 fixed TP2，不能解释为普遍 p99 收益。详细结果见 [high-rate summary](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-mooncake-highrate-v13/summary.md>) 和 [rate_ttft_sweep.csv](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-mooncake-highrate-v13/rate_ttft_sweep.csv>)。

同一 trace 的 `long_threshold=16000` ablation 将 v13 p99 再降低约 4--10 秒，但 offered-SLO 下降，详见 [threshold summary](</mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-mooncake-threshold-16000/summary.md>)；该值暂不替换 v13 默认 12000。
