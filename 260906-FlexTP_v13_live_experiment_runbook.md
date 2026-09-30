# 260906-FlexTP v13 实测运行手册

## 前置检查

```bash
cd /mtc/wusiyu/work/LightLLM-flex-tp
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
nvidia-smi -L
```

需要看到至少 8 张 GPU。`gpu5` 上已有的小进程不需要清理；启动脚本只使用既定的 TP2(0,1)、TP2(2,3)、TP4(0,1,2,3) 和 decode(4--7) 拓扑。

## v13 ServeGen

端到端 runner 会负责 GPU 预检、proxy 清理、启动、16000-token mixed warmup、多个 request rate 和退出清理：

```bash
RATES="10 8 6 4 2 1" \
SERVEGEN_MODES="mm-image m-large deepseek-r1" \
WARMUP_TIMEOUT_S=600 \
BENCHMARK_TIMEOUT_S=900 \
REQUEST_TIMEOUT_S=180 \
./test/benchmark/service/run-v13-live-sweep.sh
```

默认 ServeGen 输出长度除以 10。每个 request-rate 超时或失败会记录日志并继续后续档位；服务异常时 runner 的退出 trap 会停止自身 v13 session。
服务日志位于 `LOGDIR`，benchmark 日志位于 `LOGDIR/benchmarks`。

同一套流程的 v12 对照 runner：

```bash
RATES="10 8 6 4 2 1" \
LOGDIR=_/260906-v12-live \
./test/benchmark/service/run-v12-live-sweep.sh
```

两个 live runner 都要求至少 8 张可见 GPU，并在创建 tmux/MPS 前执行
`nvidia-smi -L` 预检；驱动不可用时应立即以非零状态退出。直接调用
`loop4-sg3.sh` 或 `loop4-sg3-v13.sh` 时，可用 `RATES` 和 `LOG_DIR` 固定
配对的 request-rate 与输出目录，便于 v12/v13 结果逐档比较；可用
`BENCHMARK_KILL_AFTER_S` 调整 timeout 的强杀宽限（默认 20 秒）。

loop4/loop3 的本次 timeout 参数变更前版本保存在对应的
`*.before_kill_after_260906` 备份文件中。

## Fixed baseline

> **状态：作废。** 以下 fixed 基线命令启动 `pd_fake_decode=True` 服务；由此得到的请求结果、延迟和吞吐均不得使用。这里只保留命令历史，不能作为有效实验入口。

fixed TP2x2 和 TP4x1 使用同一 fake-decode Prefill worker 拓扑，并通过同一个
`benchmark_serving_chat_req_rate.py` 客户端进行 ServeGen 对照：

```bash
PROFILE=fixed_tp2 \
RATES="10 8 6 4 2 1" \
SERVEGEN_MODES="mm-image m-large deepseek-r1" \
./test/benchmark/service/run-fixed-fake-decode-live-sweep.sh

PROFILE=fixed_tp4 LOGDIR=_/260906-fixed-tp4-fake-decode-live \
  ./test/benchmark/service/run-fixed-fake-decode-live-sweep.sh
```

runner 会先发送短/长 mixed warmup（包含 16000-token 请求），清理六个 proxy
变量，并在每个 rate 使用有限 benchmark timeout。worker 启动脚本的原始版本
保存在 `start-cluster3_70b_p22.4d4_mps_prefill_fake_decode.sh.before_live_runner_260906`。

也可以手动启动并观察日志：

```bash
NO_ATTACH=1 ./start-cluster4_70b_p22.4d4_mps_flex_v13.sh
WARMUP_LONG_INPUT_TOKENS=16000 \
  ./test/benchmark/service/wait_warmup_mixed.sh --url http://127.0.0.1:60011/generate
./test/benchmark/service/loop4-sg3-v13.sh
./stop-cluster-v13.sh
```

`stop-cluster-v13.sh` 只清理 v13 的 tmux session、60011/8000--8003
端口对应的 LightLLM 进程和 MPS daemon，不调用仓库中会广泛匹配 Python
进程的旧版 `stop-cluster.sh`。

## Mooncake

```bash
MOONCAKE_DATASET=/mtc/wusiyu/work/LightLLM-flex-tp/mooncake_trace.jsonl \
RATES="10 8 6 4 2 1" \
MOONCAKE_OUTPUT_DIVISOR=10 \
./test/benchmark/service/loop4-mooncake-v13.sh
```

## 离线 MPS 实验

无 GPU 时仍可生成论文用 timing-model 结果：

```bash
python -u test/benchmark/service/flex_tp_mps_slo_down.py \
  --output-dir _/flex_tp_paper_analysis/260906-mps-slo-down
```

产物包括 `mps_slo_down.csv`、TTFT p99 图和 offered-SLO 图。该结果不是 GPU wall-clock 测量；真实 GPU 结果必须单独记录。

## 结果要求

- 每个策略必须先完成短/长 warmup，再开始 measured requests。
- benchmark 命令必须清除六个 proxy 变量。
- 报告至少保留 request rate、TTFT p50/p99、SLO attainment、完成吞吐、input-token goodput 和 TP4 request share。
- 客户端或服务超过 timeout 后重启 session，不要无限等待。
