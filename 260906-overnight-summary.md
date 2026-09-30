# 260906 FlexTP 夜间实验总结

> **状态更新：** 使用服务参数 `pd_fake_decode=True` 的真实请求 latency/吞吐和完整 PD 聚合结果均作废。仅 MPS slowdown 的 worker 侧 PERF/CUDA-event 子集保留为窄例外，并须满足每个 worker 的满载与重叠条件；具体资格见 `260907-pd-fake-decode-invalid.md` 和 `260907-MPS_pd_fake_decode_data_audit.md`。

## 总体状态

已完成并核验（带 `pd_fake_decode=True` 的真实请求条目作废；MPS worker slowdown 按窄例外单独审计）：

- worker 侧 MPS forward trace 和高 QPS 分析；
- 三组批处理延迟实验（FlexTP 批处理实验的持续满载结论仍有明确限制）；
- Mooncake/Conversation 的带时间戳长度分布分析；
- 生产 `naive_switch` 选择器及模拟器实现；
- 五个数据集、五个系统的模拟器 rate/TTFT 对照扫参；
- 三个 ServeGen 子数据集、每个四档速率的真实 GPU v12 扫参；
- 同一组数据集和速率的真实 GPU v13 配对矩阵。
- fixed TP2、fixed TP4 的三数据集 × 四速率真实 GPU 对照矩阵；
- naive、`naive_switch` 的三数据集 × 四速率真实 GPU 矩阵，以及六套系统统一 CSV 和 p99 图。

六套系统的统一 GPU 矩阵现已整体作废：其中 fixed/naive/naive_switch 使用了 `pd_fake_decode=True`，不能代表真实 Decode；v12/v13 的真实 Decode 原始配对仍单独保留，见 `260906-v12-v13-gpu-compare.md`。作废范围和证据见 `260907-pd-fake-decode-invalid.md`。

## MPS 批处理实验

按要求运行的三种条件曾使用 `pd_fake_decode=True`。客户端请求 latency/吞吐作废；worker 侧 PERF/CUDA-event 仅作为 MPS slowdown 候选，并且 FlexTP all-mix 未通过每个 worker 的同时满载检查，不能作为最终三组结论。

需要保留的偏差是：FlexTP 中 TP2 与 TP4 有时间重叠，但 TP2 先排空，因此批处理聚合表不能证明整个窗口持续满载。后续严格判断应以 worker forward trace 和 `PERF - prefill` 日志为准。

## MPS slowdown 调查

此前 `PERF - prefill` 的 8192-token 对照来自 `pd_fake_decode=True` 服务。它不能用于请求层结果；其中的 3.5 倍只是未通过严格满载检查的候选观察值，不能作为最终三组 slowdown。

此前高 QPS worker trace 同样来自 `pd_fake_decode=True` serving。它可用于 MPS worker 竞争审计，但 32/64/128/2048 overlap 均未通过每个 worker 的同时满载条件，不能据此声称最终 slowdown。

`model` 字段是 worker 第一阶段区间，`latency` 字段还包含 post、等待 CUDA stream 等部分；严格的 MPS slowdown 数值使用 CUDA-event `model_forward_ms`，客户端请求延迟不用于该数值。补充的 no-chunk 对照保持 `batch_max_tokens=16384`、只关闭 chunked-prefill：TP4 8192-token/bs1 forward 中位数约 564 ms，而 16k chunk 对照约 571 ms，均接近生产 fixed TP4 的 547 ms。因此没有证据表明 chunk splitting 单独造成固定 3x；但它仍可能通过改变 batch packing 和调度节奏间接放大并发竞争。

## MPS 是否正确打开

作废的三组服务虽然记录了 MPS 参数，但 `pd_fake_decode=True` 已使对应 serving 结果失效；MPS 参数正确不等于该实验可以作为真实 Decode 证据。

限制是：历史运行没有保存 MPS daemon 的完整生命周期遥测，因此不能仅凭事后状态证明 daemon 在每一个旧事件上都持续存活；配置日志、独立 daemon 启动检查以及 worker 区间交叠共同构成当前可获得的证据。

## 调度器与 rate/TTFT 实验

### `naive_switch`

- 新增生产选择器 `lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_naive_switch.py`；
- 完成 API/工厂注册；
- 模拟器使用显式 pending FIFO，活动类持有 lease 时不接纳相反类别；
- self-test 和 100 请求混合 smoke test 通过；
- 模拟器中的 `mps_overlap_wall_s=0`，确认 switch 语义不是同时执行 TP2/TP4。

### 模拟器矩阵

`_/flex_tp_paper_analysis/260906-naive-switch-sim-sweep/` 有 75 个完成 shard：5 个数据集、3 个速率（2/4/8）和 5 个系统（fixed_tp2、fixed_tp4、naive、naive_switch、v12），每个数据集均生成 p50/p99 TTFT 图。

### 真实 GPU v12 矩阵

`_/gpu_live_260906/v12_sweep/` 的 12/12 运行成功，覆盖 `servegen-mm-image`、`servegen-m-large`、`servegen-deepseek-r1`，每个数据集速率为 2/4/6/8。解析器使用 benchmark 的首 token 延迟 `token_latencys[0]` 计算 TTFT p50/p99；吞吐字段按名义 120 秒窗口计算，严格完成吞吐应回看原始日志。

### 真实 GPU v13 矩阵

`test/benchmark/service/_/gpu_live_260906/v13_sweep_260906b/` 的 12/12 运行成功，使用相同三个数据集、四档速率和 120 秒时长。每一档均先完成三类 prefill 与 decode 注册，再完成混合长短 warmup；所有 JSON 均报告 0 个失败请求。与 v12 的配对表和 p99 图见 `260906-v12-v13-gpu-compare.md` 以及 `_/gpu_live_260906/260906-v12-v13-compare/`。

初始 v13 smoke 曾因两个 runner 问题失败：tmux 长命令中的 `;` 被转义，以及 warmup helper 使用相对日志路径。两处均已修复，随后单档 smoke 和完整 12 档矩阵均成功；初始失败运行不计入结果。

### fixed TP2/TP4 真实 GPU 矩阵

fixed TP4 使用 `_/gpu_live_260906/fixed_tp4_sweep_260906/`，fixed TP2 使用 `_/gpu_live_260906/fixed_tp2_sweep_260906/`；两者均完成 12/12，所有请求成功。统一 p99 表和图见 `260906-multi-system-gpu-compare.md` 以及 `_/gpu_live_260906/260906-multi-system-compare/`。这四套系统的矩阵仍是不同时间的独立运行，适合做方向性比较；fake decode 只返回一个 token，因此不解读 decode 吞吐。

### naive/naive_switch 真实 GPU 矩阵

`_/gpu_live_260907/naive_sweep/` 和 `_/gpu_live_260907/naive_switch_sweep/` 均完成 12/12 档，覆盖同样的三个数据集和四档速率；每个 JSON 都是 0 failed。六套系统合并结果和图位于 `_/gpu_live_260907/260907-six-system-compare/`，汇总报告为 `260907-six-system-gpu-compare.md`。

正式矩阵执行时，`prefill_node_impl.py` 尚未加入 `radix_cache is None` 守卫。由于启动参数显式使用 `--disable_dynamic_prompt_cache`，请求首 token 返回后 PD freeze 辅助路径会捕获并记录 `AttributeError: 'NoneType' object has no attribute 'insert'`。修复后的全拓扑 `naive_switch` smoke 已确认三个 worker 均启动、trace 均有数据且该异常消失，但 smoke 不是完整 rate 矩阵；正式矩阵必须在修复后重跑。

## 可能误导或尚未完成的部分

- 旧版客户端请求延迟混入排队和尾部 drain，不能作为 MPS slowdown；
- 不同 batch shape 的聚合 p50 不是纯 slowdown；
- trace 去重保留一个代表 rank，rank 间抖动可能造成轻微偏差；
- 当前 MPS 证据确认的是 chunked-prefill/high-load 交互；no-chunk 对照排除了固定 3x 的切分基线膨胀，但尚未完全量化它对 batch packing/满载概率的间接贡献；
- 模拟器使用校准 timing model、fake decode 和 KV transfer，不是 GPU wall-clock；
- v13 的完整 GPU 结果已完成，但 v12/v13 仍是不同时间启动的独立运行；`mm-image` v12 的 rate=6/8 进入严重排队区间，因此对应的巨大改善不能简单解释成硬件加速。
- fixed TP2/TP4 早期补跑曾受 GPU3 Xid 31、worker 加载退出和 runner 配置问题影响；这些失败运行未计入结果。使用实际 host、独立 MPS pipe、修复日志目录传递后，fixed TP2/TP4 各自的 12 档正式矩阵均已成功完成。
- naive/`naive_switch` 正式矩阵虽然请求层面全部成功，但含有上述被捕获的 `NoneType.insert` 日志异常；在修复后重跑前不得把对应 p99 当作最终论文数据。
