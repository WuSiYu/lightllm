# 260903-FlexTP：Fake Decode 仅 Prefill 实验与 V5-V9 对比

> **状态：作废（`pd_fake_decode` 实机结果）。** 本文的实机服务使用 fake decode，GPU/请求结果和由此形成的比较结论全部作废；原始记录仅保留作审计。离线模拟器描述不等价于真实 GPU 结果。

> 260904 口径说明：本文中的 `fixed` 是旧的 static-partition 名称，已从当前模拟器移除。新的固定基线为 `fixed_tp2` 和 `fixed_tp4`；本文数据保留作历史记录，不应与新定义混用。

本文自包含说明 fake decode 的语义、GPU0-3 上的 Prefill 拓扑、fixed/naive/V5-V9
调度器、step 精确模拟器，以及 2026-09-03 在 4 张 H200 上完成的实机对比。实机部分使用
生产 `PDMasterManager`、生产 Prefill router、`ChunkedPrefillQueue` 和真实 selector；Decode
只保留协议和可配置的 KV transfer 等待，不执行 Decode 模型计算。

## 1. Fake Decode 语义

系统支持三种模式：

| 模式 | Prefill 完成后 | Decode step | 适用场景 |
|---|---|---:|---|
| 默认 no-decode | 立即结束 | 0 | 只测 Prefill 计算上界 |
| `--fake-decode` | 等待虚拟 KV transfer 后结束 | 0 | 仅 Prefill，保留 PD 通信成本 |
| `--simulate-decode`（step simulator） | 等待 transfer 后进入 Decode 队列 | >0 | 研究 Decode 计算瓶颈 |

fake decode 的生命周期为：

```text
Prefill step 完成
  -> 完成 Prefill lease/通知 master
  -> 等待 fixed_ms + us_per_token * input_tokens / 1e6
  -> 将一个输出 token 标记为 FINISHED_LENGTH，关闭 HTTP 流
```

生产参数定义在
[`start_args_type.py`](/mtc/wusiyu/work/LightLLM-flex-tp/lightllm/server/core/objs/start_args_type.py)
和 [`api_cli.py`](/mtc/wusiyu/work/LightLLM-flex-tp/lightllm/server/api_cli.py)：

```text
--pd_fake_decode
--pd_fake_decode_kv_transfer_fixed_ms 20
--pd_fake_decode_kv_transfer_us_per_token 0
```

在 fake 模式中，master 创建 `node_id=-1` 的虚拟 Decode 节点，让 selector 继续看到完整
的 PD 节点集合，但不启动真实 Decode KV manager，也不调用 `QueueForPDDecode`。Prefill
router 跳过 Decode KV manager 初始化；关闭 `--pd_fake_decode` 后，普通 PD/NIXL 路径不变。
`max_new_tokens` 在 fake 模式被限制为 1，因此返回一个 token 只表示 Prefill 完成，不能
用来推断 Decode 吞吐。

step simulator 的入口是
[`flex_tp_step_sim.py`](/mtc/wusiyu/work/LightLLM-flex-tp/test/benchmark/service/flex_tp_step_sim.py)：

```bash
python test/benchmark/service/flex_tp_step_sim.py \
  --scheduler v9 --dataset synthetic-5pct --num-prompts 100 --rates 10 \
  --fake-decode --kv-transfer-fixed-ms 20 --kv-transfer-us-per-token 0.5 \
  --output-dir _/flex_tp_paper_analysis/fake-decode-smoke
```

simulator 的 fake-decode self-test 为 100/100 完成、0 个 Decode step；固定 20 ms 加每 token
项后，KV transfer p95 约 24.836 ms。

## 2. Prefill 拓扑和策略

实机只使用 GPU0-3，Decode GPU 不加入本次实验：

| instance | TP | GPU | 典型路由 |
|---|---:|---|---|
| `p01` | 2 | 0,1 | `<4000` token |
| `p23` | 2 | 2,3 | `<4000` token |
| `p0123` | 4 | 0,1,2,3 | `>=4000` token |

TP4 与两个 TP2 共享 GPU placement；三个实例可以同时 active，因此 MPS overlap 是真实
共享资源约束，而不是将实例串行切换。实机固定参数为 `long_threshold=4000`、
`mps_slowdown=2.0`、`max_req_total_len=8192`、`max_total_token_num=9000`、
`batch_max_tokens=4096`、`chunked_prefill_size=4096`、`graph_max_len_in_batch=8192`。

策略含义如下：

- `fixed`：`FlexTPStatic2NodeSelector`，4000 token 硬阈值；短请求在两个 TP2 池内按在途
  token 最少选择，长请求固定到 TP4。
- `naive`：`FlexTPNaiveSelector`，同一硬阈值和普通 PD 单请求路径，作为 naive MPS 基线。
- `v5`：bundle/lease 计数和 token credit 的生产实现。
- `v6`：在 v5 上加入 deadline/SLO 导向的调度权重。
- `v7`：bundle batching 与 work-conserving 选择，step 模拟中平均 token goodput 最优，故
  实机测试顺序第一。
- `v8`：更积极的 bundle/负载预测策略。
- `v9`：MLFQ aging + CFS virtual runtime，使用队列 aging 防止长请求饿死，并用虚拟运行时
  在同类候选之间保持公平。

所有策略共享同一请求到达时间、同一长度 trace、同一 Prefill batch 曲线和同一 MPS
overlap 参数。fixed/naive 在当前单主机硬分流下接近是预期的：物理 TP 池相同，差异只
出现在负载计数和队列路径。

## 3. Step 精确模拟矩阵

主脚本
[`flex_tp_v7_v8_comparison.py`](/mtc/wusiyu/work/LightLLM-flex-tp/test/benchmark/service/flex_tp_v7_v8_comparison.py)
（文件名保留历史名称）现在包含 fixed、naive 和 V3-V9。它覆盖 phase-shift、
synthetic-5pct、ServeGen `mm-image`、MPS slowdown 1.6/2.0，共 90 次运行；
[`flex_tp_v7_v8_adversarial.py`](/mtc/wusiyu/work/LightLLM-flex-tp/test/benchmark/service/flex_tp_v7_v8_adversarial.py)
覆盖 threshold-edge（3999/4000/4001/8000）、long-then-short、dual-lane-fair，
slowdown 1.6/2.0/2.5，共 81 次运行。

历史 step 矩阵的 ServeGen、10 req/s、slowdown=2.0 单点如下；token goodput 是按时完成的
输入 token/s：

| 策略 | token goodput | offered SLO | TTFT p95 |
|---|---:|---:|---:|
| fixed | 10109.0 | 0.5857 | 5.843 |
| naive | 10109.0 | 0.5857 | 5.843 |
| V3 | 3005.2 | 0.2082 | 6.492 |
| V4 | 13664.7 | 0.8981 | 3.882 |
| V5 | 13264.4 | 0.8647 | 4.354 |
| V6 | 12736.2 | 0.8886 | 4.655 |
| V7 | 11565.4 | 0.7350 | 4.798 |
| V8 | 9693.9 | 0.5796 | 6.572 |
| V9 | 10587.8 | 0.6604 | 5.235 |

结果文件为
[`fake-decode-v3-v9-v1/results.json`](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/fake-decode-v3-v9-v1/results.json)
和
[`fake-decode-v3-v9-adversarial-v1/results.json`](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/fake-decode-v3-v9-adversarial-v1/results.json)。
这些压力矩阵用于观察算法差异；实机矩阵见下一节，二者不能混用吞吐数字。

## 4. H200 实机对比

### 4.1 可复现实验驱动

实机驱动为
[`flex_tp_fake_decode_live_benchmark.py`](/mtc/wusiyu/work/LightLLM-flex-tp/test/benchmark/service/flex_tp_fake_decode_live_benchmark.py)。
它启动一个临时 master、等待三个 Prefill worker 注册、按策略顺序运行，然后杀掉自己
启动的 master。worker 只启动一次，master 在策略之间重启，避免重复加载 70B 权重。

每个 policy/scenario 开始正式计时前，强制执行两轮分阶段预热：

```text
每轮：8 个短请求，4 路并发；再 4 个长请求，1 路串行
共 2 轮，即每个 scenario 24 个预热请求
```

短预热覆盖 128/512/1024/2048 token，长预热覆盖 4096/6000/7600 token。短、长分段是为了
避免 TP2 和 TP4 同时首次编译时争抢显存；预热失败会直接终止该策略，不会静默写入结果。
正式 trace 的每个 prompt 使用同一 Llama tokenizer 精确补齐到目标 token 数，并用唯一 SHA
前缀避免跨请求/跨策略 radix-cache 复用。因此服务端报告的 `prompt_tokens` 必须等于 trace，
4000 边界不会因字符串前缀漂移。

本次命令（worker-only 模式由套件管理 master）为：

```bash
NO_ATTACH=1 START_MASTER=0 SELECTOR=flex_tp_v7 MASTER_PORT=16011 \
  PREFILL_PORT_01=18100 PREFILL_PORT_23=18101 PREFILL_PORT_0123=18102 \
  MPS_PIPE=/tmp/codex_mps_prefill_live_260903 \
  MAX_REQ_TOTAL_LEN=8192 MAX_TOTAL_TOKEN_NUM=9000 \
  BATCH_MAX_TOKENS=4096 CHUNKED_PREFILL_SIZE=4096 GRAPH_MAX_LEN_IN_BATCH=8192 \
  ./start-cluster3_70b_p22.4d4_mps_prefill_fake_decode.sh

python -u test/benchmark/service/flex_tp_fake_decode_live_benchmark.py \
  --host 10.120.178.80 --port 16011 \
  --policies v7,v6,v5,v9,v8,fixed,naive --scenarios all \
  --servegen-duration 20 --servegen-rate 8 \
  --synthetic-prompts 160 --synthetic-rate 8 \
  --warmup-rounds 2 --warmup-short-requests 8 --warmup-long-requests 4 \
  --warmup-short-concurrency 4 --warmup-long-concurrency 1 \
  --output-dir _/flex_tp_paper_analysis/fake-decode-live-v5-v9-260903
```

该专用启动器给三个 Prefill worker 设置 `DISABLE_GPU_TENSOR_CACHE=1` 和
`--disable_dynamic_prompt_cache`。70B 共享权重在 H200 上若允许 GPU tensor cache 按多种
shape 永久增长，会在 TP2/TP4 overlap 时耗尽显存；关闭这两个请求缓存不改变模型权重、KV
通信或调度策略。实测空载约 77--80 GiB/卡，正式负载期间无 OOM。

### 4.2 负载和结果

ServeGen `mm-image` 使用 20 s、8 req/s，实际 trace 为 159 个请求（其中 12 个长请求）；
synthetic-5pct 使用 160 个请求、8 req/s（7 个长请求）。fake KV transfer 为固定 20 ms、
每 token 0；所有请求 `max_new_tokens=1`。结果原文件：
[`results.json`](/mtc/wusiyu/work/LightLLM-flex-tp/_/flex_tp_paper_analysis/fake-decode-live-v5-v9-260903/results.json)。

| 策略 | ServeGen token/s | ServeGen TTFT p95 | Synthetic token/s | Synthetic TTFT p95 |
|---|---:|---:|---:|---:|
| fixed | 12569.9 | 0.949 s | 6096.7 | 0.244 s |
| naive | 12612.5 | 0.957 s | 6090.8 | 0.238 s |
| V5 | 12593.5 | 1.062 s | 6096.4 | 0.276 s |
| V6 | 12543.4 | 0.996 s | 6094.7 | 0.413 s |
| V7 | 12598.9 | 0.975 s | 6098.6 | 0.264 s |
| V8 | 12591.5 | 1.079 s | 6095.8 | 0.283 s |
| V9 | 12584.1 | 0.987 s | 6097.4 | 0.286 s |

实机审计结果：

- 7 个策略、2 个 scenario 均完成；ServeGen `159/159`，synthetic `160/160`。
- 每个 policy/scenario 的预热均为 `24/24`；正式请求 offered SLO 达成率均为 `1.0`。
- `prompt_token_delta_mean=0`、`route_class_match_fraction=1.0`，两个 scenario 的 trace
  fingerprint 在 7 个策略间完全相同。
- 每个 master 日志均有 3 次 Prefill 注册和 `pd_fake_decode=True`；master 日志无 OOM、
  Traceback 或线程异常。GPU4-7 未被本实验触碰。

在 8 req/s 的低/中负载下，7 个策略的结果集中在约 0.6% 的 token/s 范围内；naive 的平均
token/s 仅比 fixed 高约 0.2%，不应解读为统计显著优势。step simulator 的高压力矩阵才是
比较队列公平性、bundle batching 和 deadline 取舍的主要依据；本次 H200 实测首先证明了
生产 fake-decode 协议、TP2/TP4 并发和调度器切换可运行。

## 5. 验证与限制

```bash
python test/benchmark/service/flex_tp_step_sim.py --self-test

python test/benchmark/service/flex_tp_v7_v8_comparison.py \
  --fake-decode --slowdowns 1.6,2.0 --synthetic-prompts 1000 \
  --output-dir _/flex_tp_paper_analysis/fake-decode-v3-v9-v1

python test/benchmark/service/flex_tp_v7_v8_adversarial.py \
  --fake-decode --slowdowns 1.6,2.0,2.5 \
  --output-dir _/flex_tp_paper_analysis/fake-decode-v3-v9-adversarial-v1
```

已通过 Python 编译、shell 语法、step simulator self-test、精确 prompt factory self-test，
以及本文件第 4 节的 7 策略实机套件。fake decode 是固定项/每 token 项的通信时间模型，
不是实际 KV bytes 带宽测量；应使用真实 PD transport trace 重新拟合两个参数。由于 Decode
计算被跳过，本实验不能推导真实 Decode 吞吐、Decode 显存容量或完整端到端服务容量。
