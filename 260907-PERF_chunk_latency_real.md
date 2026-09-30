# 260907 真实 Decode 的 PERF chunk 延迟记录

## 数据来源

`lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py` 会在每个 Prefill chunk 输出 `PERF - prefill` 行。该行直接包含：

- `bs`：本次 batch size；
- `token`：本次 chunk 的 token 数；
- `latency`：该 chunk 的完整 worker 阶段延迟；
- `model`：其中的模型计算区间。

本次只对已经核对为 `pd_fake_decode=False` 的真实 Decode 日志做离线解析：

- `_/gpu_live_260906/v12/`；
- `_/gpu_live_260906/v13_sweep_260906b/`。

机器可读输出为 `_/gpu_live_260906/perf_chunk_real_v12_v13.json`，解析入口为 `test/benchmark/service/analyze_perf_chunk_260906.py`。

## 8192-token chunk

筛选条件为 `batch_size=1`、`batch_tokens=8192`，每个版本各 4 条事件：

| 服务 | 样本数 | model p50 / p90（ms） | 完整 chunk latency p50 / p90（ms） |
|---|---:|---:|---:|
| v12，真实 Decode | 4 | 626.96 / 637.42 | 1265.08 / 1275.97 |
| v13，真实 Decode | 4 | 610.66 / 619.74 | 1234.09 / 1243.36 |

## 有效性边界

这证明原始 `PERF - prefill` trace 中确实有 chunk latency 数据，但上述两组是 v12/v13 的生产真实 Decode运行，不是用户要求的三组独立 MPS slowdown 条件（fixed TP2 短请求、fixed TP4 长请求、FlexTP 混合请求）。因此不能用它们单独计算目标 slowdown。

此前 fixed/naive/FlexTP 的对应 serving 日志含 `pd_fake_decode=True`，已在 `260907-pd-fake-decode-invalid.md` 中作废；其 `PERF` 数值不再引用。
