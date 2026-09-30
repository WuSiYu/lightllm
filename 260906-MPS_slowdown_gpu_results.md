# 260906-MPS GPU slowdown 实测

## 测量定义

本实验独立于 FlexTP 主实验和离散 simulator。使用 GPU0（NVIDIA H200）运行
两个常驻 CUDA worker；每个 worker 对 `[input_tokens, 8192]` 的 fp16 activation
执行两次矩阵乘，并在 GPU 上同步计时。每个输入长度先测单 worker baseline，
再启动私有 `CUDA_MPS_PIPE_DIRECTORY` 下的两个并发 worker。报告的 slowdown 定义为：

`median(two-worker MPS wall-clock per iteration) / median(single-worker wall-clock per iteration)`

正式测量参数为 warmup=5、repeats=10、iterations=50。GPU5 上已有的外部进程
没有使用；MPS daemon 在每个长度后停止。该 proxy 测量的是 MPS 共享计算资源的
contention，不是 70B 模型端到端 TTFT，因此不能直接替代模型服务的 latency profile。

运行脚本：

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
python -u test/benchmark/service/mps_slowdown_gpu_benchmark_260906.py \
  --gpu 0 \
  --lengths 256,1024,4096,8192,16384 \
  --width 8192 \
  --warmup 5 --repeats 10 --iterations 50 \
  --output-dir _/flex_tp_paper_analysis/260906-mps-slowdown-gpu
```

## 结果

硬件与软件：NVIDIA H200，driver 580.126.09，PyTorch 2.8.0+cu128，CUDA runtime 12.8。

| 输入 token | 单 worker (s) | 两 worker + MPS (s) | slowdown |
|---:|---:|---:|---:|
| 256 | 0.000105 | 0.000225 | 2.143 |
| 1024 | 0.000412 | 0.000861 | 2.087 |
| 4096 | 0.001707 | 0.003362 | 1.970 |
| 8192 | 0.003456 | 0.006896 | 1.996 |
| 16384 | 0.006978 | 0.014026 | 2.010 |

4096 token 以上的长 prefill proxy 稳定在约 1.97--2.01 倍；短 token 的结果受 kernel
launch/同步固定开销影响，不能单独用来校准长请求 slowdown。原始数据、元数据和图表：

- [mps_slowdown_gpu.csv](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-mps-slowdown-gpu-width8192/mps_slowdown_gpu.csv)
- [mps_slowdown_gpu_metadata.json](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-mps-slowdown-gpu-width8192/mps_slowdown_gpu_metadata.json)
- [mps_slowdown_gpu.svg](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-mps-slowdown-gpu-width8192/mps_slowdown_gpu.svg)

作为 shape 敏感性参考，`width=4096` 的同一实验在 256/1024/4096/8192/16384
token 上得到 1.704/2.033/2.018/2.001/1.974；原始文件保留在
`_/flex_tp_paper_analysis/260906-mps-slowdown-gpu/`。

之前的 [260906-mps-slo-down](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/260906-mps-slo-down)
仍是 simulator sensitivity sweep；其中的 slowdown=1/1.25/1.5/2/3 是模型输入，
不应引用为 GPU 实测值。
