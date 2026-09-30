# 260907 `pd_fake_decode` 结果作废清单

> **状态：分层清单。** `pd_fake_decode=True` 的真实请求 latency/吞吐和完整 PD 结果作废；仅限 MPS slowdown 的 worker 侧批处理计时可按下方例外使用。

## 作废原因

本轮审计在服务启动日志中确认了 `pd_fake_decode=True`。该模式会跳过真实 Decode 模型计算，只保留协议/虚拟 KV transfer 生命周期，因此不能代表真实请求 latency、吞吐或完整 PD 服务行为。

用户后续明确了一个窄例外：MPS slowdown 可以使用 fake Decode，但只允许使用 worker 侧 `PERF - prefill` 或 CUDA-event 的执行区间来衡量 GPU 竞争。它不能用于客户端请求 latency/TTFT、吞吐、完整 PD 服务或调度器优劣结论；并且必须通过每个 worker 的满载与重叠检查。审计结果见 `260907-MPS_pd_fake_decode_data_audit.md`。

## 作废的原始数据目录

- `_/gpu_live_260906/fixed_tp2_sweep_260906/`
- `_/gpu_live_260906/fixed_tp2_sweep_260906c/`
- `_/gpu_live_260906/fixed_tp4_sweep_260906/`
- `_/gpu_live_260906/fixed_tp4_smoke_260906/`
- `_/gpu_live_260906/fixed_tp4_smoke_260906b/`
- `_/gpu_live_260906/fixed_tp4_smoke_260906c/`
- `_/gpu_live_260906/fixed_tp4_smoke_260906d/`
- `_/gpu_live_260907/naive_sweep/`
- `_/gpu_live_260907/naive_sweep_postpatch/`（中断的部分矩阵也作废）
- `_/gpu_live_260907/naive_switch_sweep/`
- `_/gpu_live_260907/naive_switch_trace_fullpatch_260907a/`
- `_/gpu_live_260907/naive_switch_trace_smoke/`
- `_/server_log_70b_p22.4d4_mps_prefill_fake_decode_fixed_tp4_round_robin/`
- `_/server_log_70b_p22.4d4_mps_prefill_fake_decode_flex_flex_tp_naive/`
- `_/mps_batch_260906/` 和 `_/mps_real_260906/` 中由上述 fake-decode 服务日志产生的客户端 serving 数据；其中 worker MPS slowdown 子集按例外规则单独审计。

每个目录均放置 `_INVALID_pd_fake_decode.txt` 作为机器可检索标记；MPS 两个目录的标记进一步区分了作废的客户端数据和可审计的 worker slowdown 子集。原始文件保留，不代表结果自动有效。

## 作废的报告和聚合

- `260903-FlexTP_fake_decode_prefill_only_comparison.md`：全文实机部分作废。
- `260904-FlexTP_v12_design_and_gpu_results.md`：其中“H200 四卡 fake-decode 实测”及其引用数据作废；离线模拟器章节不属于该实机结果。
- `260905-FlexTP_v13_and_mps_slo_results.md`：旧 H200 fake-decode 基线段落作废；离线 timing-model 模拟结果单独保留并明确不是 GPU wall-clock。
- `260906-FlexTP_v13_live_experiment_runbook.md`：fixed fake-decode 基线命令和结果作废；v12/v13 真实 Decode 命令不在此作废范围。
- `260906-MPS_batch_latency_results.md`、`260906-MPS_perf_chunk_8192.md`、`260906-mps-slowdown-investigation.md`：客户端 serving 数值作废；worker MPS slowdown 数值仅按 `260907-MPS_pd_fake_decode_data_audit.md` 的资格结论使用。
- `260906-multi-system-gpu-compare.md`、`260907-six-system-gpu-compare.md`：包含 fake-decode 系统的统一表格和图全部作废，不能保留其中的混合聚合结论。
- `260906-overnight-summary.md`：其中引用上述 fake-decode GPU/MPS 结果的状态和结论作废，文件已补充本清单引用。

## 不在本清单中的数据

- `_/gpu_live_260906/v12/`、`_/gpu_live_260906/v12_sweep/`、`_/gpu_live_260906/v13_sweep_260906b/` 等生产 v12/v13 日志已核对为 `pd_fake_decode=False`，不因本清单作废。
- `260906-MPS_slowdown_gpu_results.md` 是独立 CUDA GEMM/矩阵代理，不含 `pd_fake_decode`；它不是本次真实请求实验，不能替代请求层结果，但不属于本清单的 fake-decode 作废项。
- 离线模拟器中的 `fake decode` 是模型化仿真开关，不是服务启动参数 `pd_fake_decode`；其结果仍需按“非 GPU wall-clock”口径解读。

## 复核命令

```bash
rg -l --glob '*.log' 'pd_fake_decode=True' _/gpu_live_260906 _/gpu_live_260907 _/server_log_*
```

除 `260907-MPS_pd_fake_decode_data_audit.md` 明确标为合格的 worker MPS slowdown 子集外，禁止从上述目录或报告复制任何数值。
