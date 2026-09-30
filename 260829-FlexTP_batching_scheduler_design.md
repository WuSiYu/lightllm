# FlexTP Prefill batching 与 MPS 调度器重设计

本文基于当前工作区代码和现有 TTFT sweep 数据，回答三个问题：当前调度器是否破坏了
Prefill batching；batching 应该放在哪一层；怎样在 SLO 约束下同时利用低 TP 的资源效率
和重叠 TP instance 的 MPS 并发。最后给出可复现的离散事件模拟，而不是只做定性讨论。

## 0. 结论先行

### 0.1 当前版本确实有一个高优先级缺陷

worker 端能够接收并 batch 多个请求，但新的 `FlexTPSelectorV2` 把每个 Prefill
instance 限制为一个在途请求：

- 文件说明明确写着“每个 instance 只允许一个在途 Prefill 请求”
  ([flex_tp_selector_v2.py:26](lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v2.py#L26))；
- `InstanceState` 只有一个标量 `running`
  ([flex_tp_selector_v2.py:327](lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v2.py#L327))；
- 候选只从 `not instance.busy` 的 instance 中产生
  ([flex_tp_selector_v2.py:1094](lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v2.py#L1094))；
- 主调度路径全部使用单请求 `predict_exclusive`，`predict_batch` 没有进入 admission
  主路径。

因此，一个请求从 PD master 发到 Prefill instance 后，要等它返回首 token，master
才会给该 instance 发下一个请求。worker 的 local Router 很难同时看到多个独立请求，
短请求 batch size 基本退化为 1。例外只有 `n/best_of` 在一个 API 请求内产生的多个
sub-request，不能解决多个独立短请求的吞吐问题。

模拟中的 64 req/s 短请求 burst，在相同 8-GPU 预算下：

- 每 instance 单请求版本只能完成约 40.43 req/s，SLO 命中率 6.2%，p95 TTFT
  11.21s；
- 两级 batching 版本完成约 63.48 req/s，SLO 命中率 100%，平均 batch size
  12.31，p95 TTFT 0.514s。

这些数值依赖本文声明的 batch 模型，不能当作真机 benchmark；但“单请求 admission
让 worker 无请求可 batch”是直接由代码确定的，不依赖模拟假设。

### 0.2 batching 不应该完全搬到任意单层

推荐“PD master 决定 batch intent，instance local Router 决定物理 batch”的两级协同：

- PD master 负责全局 pending、deadline、TP/placement、合批等待窗口、每个 instance
  的 admission credit，以及允许哪些重叠 instance 进入 MPS mode；
- worker local Router 负责最终内存可行性、KV/token 容量、chunk 切分、实际 tensor
  batch 和 kernel 提交；
- master 发送带 `bundle_id` 的原子 bundle/flush 意图，worker 返回 accepted/started/
  finished 和周期状态报告。

只把 master 的单请求锁改成无限 inflight，虽然能立刻恢复 worker batching，但 master
看不到真实 batch 边界和队列，无法可靠保证 SLO。反过来把 chunk/tensor batch 全搬到
master，会侵入共享内存、KV 和模型执行边界，复杂度过高。

### 0.3 MPS 是受测量约束的附加能力，不是无条件收益

同一 4-GPU Flex group 的合法运行 mode 至少包括：

```text
{TP2-A}  {TP2-B}  {TP2-A, TP2-B}  {TP4}
{TP4, TP2-A}  {TP4, TP2-B}  {TP4, TP2-A, TP2-B}
```

TP2-A 和 TP2-B 物理 GPU 不重叠；TP4 与两者都重叠。batch 中有 20 个请求仍只是
一个 instance 进程，不应把请求数误算成 MPS 进程数。

必须测量不同 mode、长度桶和 batch-token 桶下的服务率/slowdown matrix。模拟显示，
当重叠 slowdown=1.3 时，对抗性的长短并发流量可由静态最佳的 93.6% SLO 提升到
100%；slowdown=1.6 时，同样的盲目 MPS 策略反而降到 89.4%。因此生产规则应是：
只有实测矩阵证明有吞吐收益，并且所有已接纳 deadline 留有 p99 margin 时才重叠。

### 0.4 还存在两个必须先修的生命周期/模型问题

1. NIXL 分支在 Prefill 节点仅完成 tokenize 并上传 `prompt_ids` 时，就通知 selector
   “prefill done” ([manager.py:345](lightllm/server/httpserver_for_pd_master/manager.py#L345))。
   worker 实际上随后还要等待 Decode 信息，之后才申请资源和执行 Prefill
   ([httpserver/manager.py:313](lightllm/server/httpserver/manager.py#L313))。这会让调度器
   过早释放 GPU reservation。
2. 原 TP4 默认延迟常数来自只含约 20K-35K token 的旧数据，给短请求造成约 408ms
   的错误下限；较新的 v8 sweep 覆盖约 42-30K token，短请求平台约 94ms。本文工作
   已将默认 TP4 常数更新为 v8 拟合值。部署到不同模型/GPU 前仍必须重测。

---

## 1. 当前请求执行链路

为避免“worker 不支持 batching”的误判，下面区分 PD master 和 instance local Router。

### 1.1 PD master：一次选定一个 P/D 节点

1. `HttpServerManager.generate` 在 master 先 tokenize，得到精确 `input_token_num`
   ([manager.py:100](lightllm/server/httpserver_for_pd_master/manager.py#L100))。
2. master 为原始请求生成 `origin_group_request_id`，只调用一次 selector，将输入长度、
   到达时间和请求 ID 传进去
   ([manager.py:141](lightllm/server/httpserver_for_pd_master/manager.py#L141))。
3. selector 返回一个具体 Prefill node 和 Decode node。
4. normal PD 路径通过 WebSocket 发送一个 `ObjType.REQ`
   ([manager.py:249](lightllm/server/httpserver_for_pd_master/manager.py#L249))。
5. Prefill 返回首 token 后，master 调 `notify_flex_tp_request_done`
   ([manager.py:270](lightllm/server/httpserver_for_pd_master/manager.py#L270))，selector 才把
   instance 标为空闲。

normal PD 的“首 token 返回”可以作为 Prefill 计算完成事件。异常路径则在真正 abort
完成之前就释放 reservation
([manager.py:184](lightllm/server/httpserver_for_pd_master/manager.py#L184))，仍有旧计算与新计算
短暂重叠的风险；推荐协议需要 `ABORT_ACK` 或 lease generation，而不是把异常等同于完成。

### 1.2 Prefill HTTP server：接收端本身支持多个并发请求

Prefill 节点的 `pd_loop` 每收到一个 `ObjType.REQ`，都会创建独立 asyncio task
([pd_loop.py:107](lightllm/server/httpserver/pd_loop.py#L107))。WebSocket 接收循环不会等待前一个
请求执行完成，因此 master 如果连续发送多个请求，worker 能并发接收。

每个 task 进入 worker 的 `HttpServerManager.generate`：

1. worker 再 tokenize 一次；
2. 为 `sampling_params.n` 个 sub-request 申请共享内存 request slot；
3. 初始化 request 时写入 `chunked_prefill_size`
   ([httpserver/manager.py:349](lightllm/server/httpserver/manager.py#L349))；
4. 通过 ZMQ 把 `GroupReqIndexes` 送到 local Router
   ([httpserver/manager.py:557](lightllm/server/httpserver/manager.py#L557))。

所以接收和共享内存层不是“只能一个请求”。限制来自 master selector。

### 1.3 instance local Router：这里已经有真正的 batching

local Router 默认每 30ms 执行一次调度循环
([api_cli.py:643](lightllm/server/api_cli.py#L643))。每次先收 ZMQ 新请求，再生成新 batch：

- 常态一次最多收 64 个请求；若 ZMQ 仍有积压，下一轮逐步提高到 256
  ([router/manager.py:524](lightllm/server/router/manager.py#L524))；
- `generate_new_batch` 从 `waiting_req_list` 顺序扫描多个请求
  ([chunked_prefill/impl.py:63](lightllm/server/router/req_queue/chunked_prefill/impl.py#L63))；
- 通过 request 数、first-router token 数、KV 峰值估计共同判断能否加入；
- 新 batch 会 merge 到 `running_batch`
  ([router/manager.py:348](lightllm/server/router/manager.py#L348))；
- 模型后端收到 `List[InferReq]`，一次 `prepare_prefill_inputs` 后做一次 forward
  ([chunked_prefill backend:103](lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py#L103))。

后端 PERF 日志已经记录 `batch_size`、`total_token_num` 和 latency
([chunked_prefill backend:148](lightllm/server/router/model_infer/mode_backend/chunked_prefill/impl.py#L148))，后续
建立真实 batch profile 可直接利用。

### 1.4 “8192 chunked prefill”不等于“batch 聚合到 8192”

dp=1 且未显式配置时：

```text
batch_max_tokens     = 16384
chunked_prefill_size = 8192
```

来源见 [api_start.py:203](lightllm/server/api_start.py#L203)。`get_first_router_need_tokens`
对单请求取 `min(input+output, 8192)`
([req.py:360](lightllm/server/core/objs/req.py#L360))；模型执行时每次也最多推进一个 8192-token
chunk ([infer_batch.py:468](lightllm/server/router/model_infer/infer_batch.py#L468))。

但普通 `ChunkedPrefillQueue` 还有另一条独立规则：累计原始 `req.input_len` 超过 6000
后立即停止继续扫 waiting queue
([chunked_prefill/impl.py:84](lightllm/server/router/req_queue/chunked_prefill/impl.py#L84))。判断发生在把
当前请求加入之后，所以它不是严格 6000 上限：一个 8K 请求仍可单独进入；很多短请求则在
第一个让累计值越过 6000 的请求之后停止。

此外该 Queue 又把 `batch_max_tokens` 乘 2
([chunked_prefill/impl.py:11](lightllm/server/router/req_queue/chunked_prefill/impl.py#L11))。当前系统实际同时
存在 8192 chunk、放大后的 first-router token 上限和约 6000 原始输入扫描阈值，含义不统一。
推荐将它们拆成显式参数：

```text
prefill_chunk_tokens       # 单请求每次推进的 chunk，例如 8192
batch_first_chunk_cap      # 一个执行 batch 的 first-chunk 总 token 硬上限
batch_fill_target          # 达到后提前 seal 的软目标
max_batch_requests         # request 数上限
```

### 1.5 master 当前看不到足够的 worker 状态

Prefill 节点仅在上报 token pack 时顺带发送平均 `dynamic_max_load`
([pd_loop.py:228](lightllm/server/httpserver/pd_loop.py#L228))。master 只保存
`total_token_usage_rate` ([PD manager:672](lightllm/server/httpserver_for_pd_master/manager.py#L672))。

master 看不到：

- worker 已接收但尚未 tokenize 的请求；
- local Router waiting request 数和 first-chunk token 数；
- open/sealed/running batch 边界；
- 每个 running request 还剩几个 chunk；
- 下一次 local Router 调度时刻；
- accepted/started/finished request ID；
- 实际活跃 MPS mode 和 instance generation。

因此旧版虽然允许一个 node 有多个 inflight request
([old selector:171](lightllm/server/httpserver_for_pd_master/pd_selector/flex_tp_selector_v2.py.old#L171))，
并用 `predict_queue` 估计队列，但仍不能严格保证 deadline。新版用单请求锁消除了不可见队列，
代价是消灭了 batching。两者都不是最终解。

### 1.6 NIXL 分支的完成语义是错误的

NIXL Prefill worker 的顺序是：

```text
encode prompt
-> 上传 prompt_ids
-> 等待 master 下发 Decode node NIXL 信息
-> 申请 request slot
-> 送 local Router
-> 真正执行 Prefill
```

证据在 [worker manager:313](lightllm/server/httpserver/manager.py#L313)。master 却在收到
`prompt_ids` 后立即调用 selector done callback
([PD master:342](lightllm/server/httpserver_for_pd_master/manager.py#L342))，然后才把请求发到 Decode、等待
NIXL 信息并回发 Prefill node。这不是统计误差，而是 reservation 生命周期提前结束。

新的协议必须由 Prefill model backend 或 worker output path 发权威 `PREFILL_FINISHED`；
`PROMPT_IDS_READY` 只能表示 tokenize 阶段完成。

---

## 2. 目标、指标与不可实现的承诺

### 2.1 调度目标的正确优先级

建议使用词典序目标，不把不同目标简单揉成一个难解释的权重和：

1. 已接纳请求和新候选的 p99 TTFT 都不得超过 deadline；
2. 在约束 1 下最大化 admitted goodput，即按时完成的 request/s 和 input token/s；
3. 在 goodput 近似相同的候选中，选择可满足 SLO 的最小 TP，降低通信和 GPU-time；
4. 提高 batch fill，减少固定开销；
5. 只有实测有正增益时，使用重叠 TP 的 MPS mode 增加总吞吐；
6. deadline 相同按 FIFO，防止短请求或长请求饿死。

“最小 TP”不能理解为只看单请求独占延迟。它必须把 master 等待、目标 instance 已接纳
工作、batch 机会、MPS slowdown 和未来大 TP reservation 都算进去。TP2 在空闲时可满足，
并不代表排在 TP2 长队后仍可满足。

### 2.2 应报告的生产指标

- 全量与 admitted 两种口径的 SLO attainment；
- p50/p95/p99 TTFT，按长度桶和 TP 分开；
- on-time request goodput、on-time input-token goodput；
- rejected、expired-before-dispatch、worker-rejected 数；
- batch size、first-chunk tokens、fill ratio、master/local wait；
- TP2/TP4 request 和 token 占比；
- exclusive GPU-seconds/request 与真实 process-GPU-seconds；
- 每个 MPS mode 的驻留时间、吞吐增益和 SLO 回退；
- prediction error 的 p50/p95/p99，不能只看平均误差。

### 2.3 不可能在所有流量上严格胜过所有静态最优

纯短、稳定、高负载下，静态全 TP2 已经是匹配该分布的专用最优，Flex 最多做到相同
batching 和相同 TP2 placement，再扣除很小的控制开销。纯长且 SLO 很紧时，静态全 TP4
同理。声称 Flex 在每一个这样的点上都严格更快是不成立的。

可验证且合理的目标是：

- 在纯分布上匹配对应静态最优的 SLO/goodput；
- 在混合或随时间变化的分布上，动态借用全部 GPU，优于固定 TP2/TP4 分区；
- 在相同 SLO/goodput 下，比全 TP4 使用更多 TP2、消耗更少 GPU-seconds；
- 在长短同时出现且 MPS 有实测增益时，优于不能重叠执行的 Flex/静态配置。

---

## 3. 四个候选改造方案

| 方案 | master 控制 | worker 控制 | 优点 | 主要问题 | 建议 |
|---|---|---|---|---|---|
| A. 有界 inflight credit | 每 instance 请求/token credit、TP 选择 | 沿用现有 30ms batching | 改动最小，立即恢复 batching | master 不知道真实 batch 和排队阶段，SLO 模型弱 | 短期止血 |
| B. master microbatch | master 聚合请求后逐个/批量发送 | local Router 再 pack | batch 机会和 deadline 可见 | 没有 bundle ACK 时不原子；会叠加 master 等待和 local 30ms 等待 | 可做原型，不宜停在这里 |
| C. 两级协同 bundle/lease | deadline、TP、batch intent、credit、MPS mode | 内存校验、chunk、物理 batch、执行、状态 ACK | 全局 SLO 与本地安全兼顾；边界清晰 | 需要扩展协议和 worker report | 推荐 |
| D. master 全集中执行 epoch | 精确决定每个 chunk 和 batch | worker 变成薄 executor | 理论控制最强 | 深度侵入 Router/KV/shm，多节点故障复杂 | 暂不推荐 |

### 3.1 方案 A：有界 inflight credit

最小修改是把 `InstanceState.running` 改成 request map，允许在一个 instance 内同时有
多个 admitted request：

```text
max_unacked_requests
max_admitted_requests
max_admitted_first_chunk_tokens
max_admitted_total_tokens
```

master 依据本地计数扣 credit，完成或失败时归还。不能只用 request 数，因为 32 个 128-token
请求与 32 个 8K 请求完全不同；至少要有 first-chunk token credit。

该方案会让现有 local Router 重新得到 batch，但 master 仍无法区分网络、worker tokenize、
waiting queue 和 running batch。适合先解除单请求瓶颈并做真机 profile，不适合作为最终 SLO
保证机制。credit 必须有上限，不能恢复旧版的任意深队列。

### 3.2 方案 B：master 先组成 microbatch

master 可将到达请求留在全局 pending 0-20ms，达到 token target、request cap 或最早 deadline
的 latest-dispatch 时 seal，然后向同一 instance 发送。

如果仍然逐条发旧 `ObjType.REQ`，这个 batch 只是“希望一起执行”：worker tokenize 完成顺序、
ZMQ 到达顺序和 local Router 30ms tick 都可能改变边界。master 等 20ms、worker 再等最多 30ms，
还会出现 double wait。

要让方案 B 有确定语义，至少需要 `REQ_BUNDLE(bundle_id, requests, flush=True)` 和
`BUNDLE_ACCEPTED`。否则只能把它当作 batch hint。

### 3.3 方案 C：两级协同，推荐设计

master 管“何时、给谁、允许多少”；worker 管“实际怎样执行”。建议新增消息：

```text
REQ_BUNDLE
  instance_generation, bundle_id, requests[]
  request: req_id, prompt, input_tokens, deadline, sampling_params
  seal_reason, master_send_time

BUNDLE_ACCEPTED
  instance_generation, bundle_id, accepted_ids[], rejected_ids[]
  local_queue_first_chunk_tokens, local_queue_requests

BATCH_STARTED
  instance_generation, batch_id, bundle_ids[], req_ids[]
  tp, first_chunk_tokens, batch_size, active_mode, start_time

PREFILL_FINISHED
  req_id, batch_id, finish_time, actual_prefill_ms, chunks

INSTANCE_REPORT
  instance_generation, monotonic_seq
  unacked/queued/running req IDs and token counts
  running batch remaining chunks/work estimate
  KV/token usage, next schedule time, active MPS mode

ABORT_ACK / BUNDLE_REJECTED
```

worker 在状态变化时立即 report，并在忙时每 10-20ms 心跳；master 用 monotonic sequence 和
instance generation 去重。WebSocket 断开或 generation 改变后，旧 ACK 不能释放新 lease。

master credit 只覆盖已发送但未得到权威 finished/rejected 的工作。worker 永远保留最终
拒绝权，因为共享内存 slot、KV 峰值和 multimodal 预处理只能由本地准确判断。

### 3.4 方案 D：把 chunk/epoch 全搬到 master

该方案要求 master 理解每个 request 的 `cur_kv_len`、暂停/恢复、prefix cache、KV 内存、
multimodal 和模型 backend 的 step 语义，并精确驱动 local shared-memory IO。它会复制 local
Router 的大量职责，网络抖动也进入每个执行 epoch 的关键路径。

除非未来要做跨 instance KV 迁移或统一 GPU kernel orchestrator，否则收益不足以抵消实现和
故障恢复复杂度。

---

## 4. 推荐架构的状态与不变量

### 4.1 request 状态机

```text
PENDING_MASTER
  -> BUNDLED_UNACKED
  -> QUEUED_WORKER
  -> RUNNING_PREFILL
  -> PREFILL_FINISHED
  -> DECODE

任意未完成状态 -> ABORTING -> ABORT_ACK / LEASE_EXPIRED
任意 admission 状态 -> REJECTED
```

只有 `PREFILL_FINISHED`、`BUNDLE_REJECTED` 或 generation 已确认失效，才能归还执行 credit。
`prompt_ids ready`、预测剩余工作归零、客户端断开和 WebSocket send 成功都不是完成。

### 4.2 instance 状态

每个 instance 不再是 `running: Optional[Request]`，而是：

```text
InstanceState
  topology: tp, physical_gpu_set, group_id, generation
  open_bundle: requests, tokens, open_since, seal_deadline
  unacked_bundles: bundle_id -> work/deadline
  worker_queued: req_id -> report state
  running_batches: batch_id -> work/deadline/chunks
  credits: requests, first_chunk_tokens, total_tokens
  report_seq, last_report_time, health
```

同一 instance 内 batch 多个请求不会增加 MPS process count；`running_batches` 正常情况下
只有一个模型执行 batch，但请求可跨 chunk 留在 running set，后续新 batch 可 merge。

### 4.3 必须维持的不变量

1. master 中每个 request 至多绑定一个 Prefill instance/generation；
2. 未 ACK bundle 也占 credit，防止网络重试造成过量 admission；
3. worker report 的 queued+running 必须能与 master admitted ID 对账；
4. 任一候选不能让已接纳请求的预测 p99 completion 越过 deadline；如果基线已超时，
   候选不能进一步推迟它；
5. 空系统可满足、只是当前 placement 不可用的请求应等待并预留未来 GPU 窗口，不能立刻
   best-effort 降级到过慢 TP；
6. MPS mode 只能来自显式物理 GPU 拓扑和已测 profile；拓扑缺失时 fail closed，不猜；
7. master 与 local Router 只能有一处主要 batching wait，避免 20ms+30ms 叠加。

当前 selector 支持 `tp_smt_gpu_ids`，但启动参数/脚本没有实际提供该字段；缺失时同组全部
保守视为重叠。正式实验必须把物理 placement 作为注册协议的一部分，不能靠端口或
`CUDA_VISIBLE_DEVICES` 字符串猜测。

---

## 5. latency、batch 和 MPS 模型

### 5.1 仓库已有的单请求测量

现有 CSV：

- `ttft_sweep_v7_tp2.csv`：55 个 TP2 点，约 40-9989 token；
- `ttft_sweep_v8_tp4.csv`：63 个 TP4 点，约 42-29990 token。

使用当前模型形式：

```text
T_exclusive(tp, L) = max(a * L / tp + b * L^2 / tp, c) + d
```

拟合得到的单请求预测如下。GPU-s 是 `T * TP`，只是资源效率代理，不等价于直接测得的
通信时间。

| input tokens | TP2 秒 | TP4 秒 | TP2 GPU-s | TP4 GPU-s |
|---:|---:|---:|---:|---:|
| 100 | 0.0914 | 0.0942 | 0.1827 | 0.3768 |
| 500 | 0.1192 | 0.0942 | 0.2383 | 0.3768 |
| 1000 | 0.1719 | 0.1047 | 0.3438 | 0.4186 |
| 2000 | 0.2819 | 0.1728 | 0.5638 | 0.6912 |
| 4000 | 0.5200 | 0.3126 | 1.0400 | 1.2503 |
| 8000 | 1.0688 | 0.6061 | 2.1377 | 2.4245 |
| 10000 | 1.3795 | 0.7599 | 2.7591 | 3.0395 |

TP4 的 wall time 从中等长度开始明显更低，但 TP2 在这些点的 GPU-time 更低。这正是
“SLO 允许时用较小 TP”的依据：不是说 TP2 latency 更快，而是它用更少 GPU 资源完成工作，
释放出的 GPU 可服务更多请求。

TP2 拟合 R2 约 0.9986、MAE 8.44ms；TP4 R2 约 0.9998、MAE约 6ms。它们只是 exclusive
单请求均值模型，不足以提供 p99 SLO 保证。

### 5.2 不能直接把单请求曲线当真实 batch 曲线

本文模拟为了研究结构，使用：

```text
T_batch(tp, B) = max(A_tp * sum(L_i) + B_tp * sum(L_i^2), c_tp) + d_tp
```

它让一个 batch 只支付一次固定项 `d`。单请求时等于已有拟合；多请求时的固定项摊销
尚未由仓库数据验证。现有 CSV 没有 batch-size/token matrix，因而模拟结果必须标为
“模型推演”，不能标成测量结果。

生产模型应直接拟合特征：

```text
Q99_prefill(tp, active_mode,
            batch_size, first_chunk_tokens,
            sum_squared_chunk_tokens, max_chunk_tokens,
            cached_tokens, chunk_index_bucket)
```

对长度超过 8192 的请求，应按真实 chunk step 模拟：请求只有在最后一个 Prefill chunk
和首 token 完成后才算结束。不能用完整长度的一次 batch 公式掩盖 chunk 之间插入新请求的
排队影响。

### 5.3 TTFT 预算必须拆段

对请求 `i` 和候选 bundle `B`：

```text
predicted_TTFT_i =
    master_batch_wait
  + websocket_and_worker_receive
  + worker_tokenize_and_shm
  + local_router_wait
  + queued/running_batch_work
  + Q99_prefill(B, tp, active_mode)
  + first_token_return
```

候选最晚 seal/send 时刻：

```text
latest_send_i = deadline_i
              - predicted_remaining_path_i
              - clock/network/model_error_margin
```

seal 条件取最先发生者：达到 `batch_fill_target`、达到 request cap、最老请求到
`latest_send`、或 master batch window 到期。

### 5.4 MPS profile 必须按 mode 和 batch 桶测

当前 selector 未配置 profile 时，用每张 GPU 上的最大进程数近似 slowdown。这只能做
fail-safe 初值，不能用于性能结论。需要至少测：

```text
TP2-A alone
TP2-B alone
TP4 alone
TP2-A + TP2-B                    # 物理不重叠，应接近各自 alone
TP4 + TP2-A
TP4 + TP2-B
TP4 + TP2-A + TP2-B
```

每个 mode 再按 `batch_size`、first-chunk tokens、短/长/混合长度分桶。记录每个 instance
的 slowdown 和 mode 总 goodput。若 slowdown 接近 2，两个重叠进程通常只是时间分享，
未必增加总吞吐；若通信、launch gap 或小 batch 低占用可被填充，slowdown 才可能显著小于 2。

代码中的 `--enable_prefill_microbatch_overlap` 是单 instance 内部将 batch 分成 microbatch
并调用模型 overlap backend；它与多个 TP instance 经 CUDA MPS 并发是两件事，不能混为同一
优化。仓库有设置 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 的 helper，但当前启动脚本没有使用，
也应作为 profile 维度而不是直接设固定比例。

---

## 6. 推荐调度算法

### 6.1 每次事件的候选构造

事件来源包括新请求、bundle ACK、batch started/finished、worker report、节点 generation
变化和 batching timer。

```text
1. 合并 worker report，校验 generation/report_seq，更新 credit 和剩余工作。
2. 清理取消请求；过期或空系统也不可行的请求进入 reject/best-effort 队列。
3. 按 deadline、arrival、request_id 排序 master pending。
4. 对每个空闲/有 credit 的 instance：
   a. 从 EDF 队列选第一个请求；
   b. 在不越过 first-chunk cap、request cap 和 deadline inversion 的条件下继续 pack；
   c. 计算 seal deadline；未到 fill/时间条件则保持 open bundle。
5. 枚举每个 bundle 的具体 TP placement 和合法 active MPS mode。
6. 从当前 running/sealed 状态开始做事件推进，预测所有已接纳和候选的 p99 finish。
7. 检查当前请求 deadline、已接纳请求保护、未来 TP4 reservation 和 credit。
8. 词典序选择：feasible goodput -> 最小 TP/GPU-time -> batch fill -> MPS 增益 -> FIFO。
9. 原子扣 credit、记录 bundle lease、发送 REQ_BUNDLE。
10. 若还有安全候选继续构造；否则设下一个 seal/latest-send timer。
```

### 6.2 bundle packing

纯 FIFO 会让一个 8K 请求挡住后面大量短请求；纯 shortest-job-first 又可能让长请求饿死。
建议 EDF 为主，在相近 deadline 窗口内做 best-fit：

```text
deadline_bucket = [oldest_deadline, oldest_deadline + reorder_slack]
在 bucket 内选择能提高 first-chunk fill、且不增加最老请求预测完成时间的请求
```

对 `L > chunk_size` 的长请求，master 可按 first chunk 记 credit，但 worker report 必须继续
报告剩余 chunks。一个长请求并非首 chunk 完成就释放 instance 工作量。

### 6.3 最小 TP 选择和升级

请求留在 `PENDING_MASTER/open_bundle` 时可重新选择 TP；收到 `BUNDLE_ACCEPTED` 后不迁移，
避免双执行和 KV 状态迁移。

```text
eligible placements = 所有预测 p99 finish <= deadline - margin 的 placement
首选 = eligible 中资源成本最低的 TP/instance
```

随着等待时间增加，TP2 可能不再可行，此时未提交请求升级到 TP4。不要使用一个固定长度
threshold：相同 4K 请求在空载可能用 TP2，在 TP2 队列忙或 deadline 较紧时应使用 TP4。

### 6.4 TP4 防饥饿和未来窗口预留

一个等待 TP4 的长请求可能暂时无法与当前 TP2 mode 安全并发。如果 scheduler 持续给 TP2
补短请求，TP4 永远拿不到完整 4-GPU 窗口。

对空系统可行但当前不可行的最早 deadline 请求，计算 `earliest_safe_start`，创建覆盖其
`gpu_set` 的 reservation。优先级更低的 refill 只有在模拟证明不会推迟该 start 时才允许。
不重叠 GPU 的 instance 继续工作。

若当前 candidate 已经会超时，但在另一 placement 完成后可以满足，则等待；不能因为某个
较小 TP 当前 idle 就立刻 best-effort 派发。本文模拟在加入这条规则前出现了级联超时，说明
它是正确性条件，不是微优化。

### 6.5 MPS admission

对于候选 active set `M`：

```text
allow_mps(M) =
  topology_explicit
  and profile_confidence(M, batch_buckets) >= threshold
  and predicted_p99(existing + candidate, M) <= all deadlines - margin
  and mode_goodput(M) > best_nonoverlap_goodput * (1 + min_gain)
  and future_reserved_windows remain feasible
```

如果没有对应 mode/bucket 数据，默认不重叠，或者使用 equal-share 上界只做 SLO 检查但
不声称吞吐收益。MPS 是 secondary objective；不能为追求 overlap time 而牺牲低 TP batching。

### 6.6 过载与 admission

本文模拟为了展示排队，会把所有请求最终执行；生产系统若 offered load 超过 deadline
capacity，必须区分：

- `admitted`: p99 可满足，纳入强 SLO；
- `best_effort`: 客户允许降级，单独统计；
- `rejected`: 空系统或当前 admission window 已无可行解。

“全部接收再报告 SLO 下降”不叫保证 SLO。可使用滚动 horizon 的 demand-bound 检查：任意
deadline window 内，已接纳 exclusive work 经 placement/mode capacity 折算后不能超过可用服务量。

---

## 7. 离散事件模拟

可复现脚本：

```bash
python test/benchmark/service/flex_tp_batching_sim.py
python test/benchmark/service/flex_tp_batching_sim.py --json
```

### 7.1 公平拓扑和比较对象

所有配置使用 8 张物理 GPU，分成两个 4-GPU group：

```text
每组 Flex: TP2-A(0,1) + TP2-B(2,3) + TP4(0,1,2,3)
```

| policy | 8-GPU 部署 |
|---|---|
| `static_tp2` | 4 个互不重叠 TP2 |
| `static_tp4` | 2 个 TP4 |
| `static_split` | 一组固定为 2 个 TP2，另一组固定为 1 个 TP4；2K 静态路由阈值，不能借容量 |
| `flex_single` | 两个 Flex group，可 MPS，但每 batch 只有一个请求，模拟当前 selector 限制 |
| `flex_no_mps` | 两个 Flex group，动态 TP 和 batching，不允许物理重叠运行 |
| `flex_mps` | 两个 Flex group，动态 TP、batching 和 deadline-checked overlap |

主结果的 MPS slowdown 假设为 1.6，并另做 1.3/1.6/2.0 敏感性。batch limit 8192，
fill trigger 4096，master window 20ms。它们是设计参数，不是当前代码默认值。

### 7.2 流量

| scenario | 请求数/时长 | 长度分布 | SLO |
|---|---:|---|---|
| `sparse_short` | 222 / 60s，Poisson 4/s | 65% 128，35% 512 | 1s |
| `bursty_short` | 1280 / 20s，每 250ms 16 个 | 128/256/512 | 1s |
| `mixed_poisson` | 808 / 60s，Poisson 14/s | 62% 256，20% 1K，12% 4K，6% 8K | 1/2/3/4s |
| `phase_shift` | 1313 / 60s | 20s 短高载 -> 20s 长请求 -> 20s 混合 | 按长度 1-4s，长阶段收紧 20% |
| `long_tight` | 121 / 45s，Poisson 2.6/s | 100% 8K | 1.25s |
| `long_short_overlap` | 700 / 40s | 每 0.8s 两个 8K，随后 12 个短请求 | 长 1.05s，短 0.9s |

### 7.3 模拟边界

- exclusive TP2/TP4 曲线来自仓库 CSV；
- 多请求 batch 固定开销摊销是假设，仓库尚无 batch profile；
- MPS slowdown 是参数扫描，不是测量；
- 主要长度不超过 8K，未精细模拟多 chunk 交错；
- 不含网络、重复 tokenize、KV transfer 的实测分布；各 policy 都省略这些公共项；
- 调度为滚动 EDF 贪心，不是全知最优；
- 所有请求最终执行，不做 admission reject，所以过载场景会显示 SLO miss。

因此结果用于验证架构行为、发现反例和确定 profiling 需求，不能替代真机结论。

### 7.4 SLO 命中率

| scenario | 全 TP2 | 全 TP4 | 静态分区 | 当前单请求 | Flex 无 MPS | Flex MPS(1.6) |
|---|---:|---:|---:|---:|---:|---:|
| sparse short | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| bursty short | 100.0% | 100.0% | 100.0% | **6.2%** | 100.0% | 100.0% |
| mixed Poisson | 99.9% | 99.5% | 89.4% | 100.0% | 99.9% | 100.0% |
| phase shift | 99.4% | 100.0% | 94.4% | 29.5% | 99.4% | 100.0% |
| long tight | 78.5% | 80.2% | 3.3% | 75.2% | 78.5% | 76.9% |
| long+short overlap | 85.7% | 93.6% | 77.3% | 68.9% | 93.0% | 89.4% |

`long_tight` 对所有 8-GPU 配置都是 deadline overload，不能用调度技巧变成 100%；应触发
admission/reject。`long+short overlap` 则证明 slowdown=1.6 时不应积极 MPS，Flex 无 MPS
反而更稳。这个负结果是 mode gating 的依据。

### 7.5 关键吞吐、延迟和资源结果

| scenario / policy | SLO | p95 TTFT | req/s | exclusive GPU-s/req | TP2 请求占比 |
|---|---:|---:|---:|---:|---:|
| burst / 当前单请求 | 6.2% | 11.210s | 40.43 | 0.201 | 97.9% |
| burst / Flex batching | 100% | 0.514s | 63.48 | 0.062 | 100% |
| burst / 全 TP4 | 100% | 0.330s | 63.93 | 0.081 | 0% |
| mixed / 全 TP2 | 99.9% | 1.069s | 13.34 | 0.438 | 100% |
| mixed / 全 TP4 | 99.5% | 1.169s | 13.24 | 0.514 | 0% |
| mixed / 静态分区 | 89.4% | 8.591s | 10.81 | 0.480 | 79.2% |
| mixed / Flex MPS | 100% | 1.131s | 13.35 | 0.484 | 75.1% |
| phase / 全 TP4 | 100% | 0.655s | 21.80 | 0.295 | 0% |
| phase / 静态分区 | 94.4% | 3.415s | 18.97 | 0.234 | 91.8% |
| phase / Flex MPS | 100% | 0.896s | 21.67 | 0.274 | 82.0% |

解释：

- burst 中 batching 将模拟 GPU 成本从 0.201 降到 0.062 GPU-s/req，主要来自固定开销
  摊销；这个比例需真机验证。
- burst 的全 TP4 wall time 更低，但 GPU 成本 0.081，比 TP2 batch 的 0.062 高约 31%。
  在同样 100% SLO 下，TP2 是更符合目标的选择。
- phase shift 中 Flex 与全 TP4 都是 100% SLO；Flex 让 82% 请求使用 TP2，exclusive
  GPU 成本比全 TP4 低约 7%。静态分区无法把短/长阶段的闲置容量互借，出现排队。
- mixed 中全 TP2 已接近饱和但仍几乎满足，因此 Flex 的优势主要是尾部保险；如果 margin
  更宽，推荐策略应更少启用 TP4/MPS，靠 TP2 batch 获得更低成本。

### 7.6 batching window/容量敏感性

| scenario | 配置 | SLO | p95 TTFT | req/s | 平均 batch |
|---|---|---:|---:|---:|---:|
| sparse | 单请求 | 100% | 0.120s | 3.72 | 1.00 |
| sparse | 约 6000 / 30ms | 100% | 0.176s | 3.72 | 1.13 |
| sparse | 8192 / 20ms 协同 | 100% | 0.153s | 3.72 | 1.10 |
| sparse | 8192 / 50ms double wait | 100% | 0.197s | 3.72 | 1.16 |
| burst | 单请求 | 6.2% | 11.210s | 40.43 | 1.00 |
| burst | 约 6000 / 30ms | 100% | 0.607s | 63.45 | 16.00 |
| burst | 8192 / 20ms 协同 | 100% | 0.514s | 63.48 | 12.31 |
| mixed | 约 6000 / 30ms | 99.8% | 1.080s | 13.34 | 1.58 |
| mixed | 8192 / 20ms 协同 | 99.9% | 1.069s | 13.34 | 1.45 |
| mixed | 8192 / 50ms double wait | 99.5% | 1.090s | 13.34 | 1.69 |

低负载时没有吞吐需要挽救，batch wait 只增加 TTFT；调度器应花 SLO slack 换 batch，而不是
固定等满。高 burst 时必须允许多个请求进入同一 instance。50ms double wait 只换到很小的
batch fill 增益，却损伤尾延迟，所以 master bundle 应能立即唤醒/flush local Router，或把
local 30ms polling 缩短，而不是两层都等。

### 7.7 MPS slowdown 敏感性

| workload | slowdown | SLO | p95 TTFT | req/s | 重叠 GPU-s |
|---|---:|---:|---:|---:|---:|
| mixed | 1.3 | 100% | 1.072s | 13.36 | 104.96 |
| mixed | 1.6 | 100% | 1.131s | 13.35 | 155.99 |
| mixed | 2.0 | 97.3% | 1.666s | 13.30 | 207.06 |
| phase shift | 1.3 | 100% | 0.691s | 21.72 | 77.21 |
| phase shift | 1.6 | 100% | 0.896s | 21.67 | 114.69 |
| phase shift | 2.0 | 99.2% | 1.069s | 21.65 | 165.72 |
| long+short | 1.3 | 100% | 0.709s | 17.53 | 49.16 |
| long+short | 1.6 | 89.4% | 1.069s | 17.38 | 19.74 |
| long+short | 2.0 | 86.3% | 1.069s | 17.38 | 3.37 |

重叠 GPU-s 越高不是越好。slowdown=2 时 mixed workload 的重叠时间更多，但 SLO 更差。
应该优化 on-time goodput，不应该把 MPS utilization/overlap 当目标函数。

---

## 8. 六类到达模式的逐步推演

### 8.1 稀疏短请求

1. 请求到达时 TP2 单请求约 90-120ms，1s SLO 有大量 slack。
2. open bundle 可等待最多约 20ms；Poisson 4/s 下多数仍是单请求。
3. 到 window 或 latest-send 后发到 TP2；不应为更低 wall time 使用 TP4。
4. 不开启 MPS，因为没有第二个并发工作，且无吞吐收益。

表现：所有配置满足 SLO；batching 将 p95 从单请求 0.120s 提到 0.153s。这个 33ms 差值说明
低载 batch window 必须纳入 SLO，而不是无条件追求 batch size。

### 8.2 短请求 burst

1. 16 个请求在约 3ms 内到达，远早于 20ms window。
2. fill trigger 或 request cap 触发 seal，同组两个 TP2 可在不重叠 GPU 上同时执行。
3. 多个请求共享固定 launch/同步项；TP2 的低 GPU-time 比 TP4 更适合。
4. 当前单请求 selector 每个 instance 只能卖给一个请求，其余留在 master，worker 无法 batch。

表现：当前版本队列不稳定，SLO 6.2%；协同 batch 为 100%。这是优先级最高的修复场景。

### 8.3 稳态混合 Poisson

1. 大多数 256/1K 请求进入 TP2 bundle。
2. 4K/8K 请求如果 TP2 连同排队仍有 margin，也先用 TP2；deadline 收紧时升级 TP4。
3. TP4 运行期间，只在 profile/margin 允许时让另一 TP2 重叠；否则在另一 4-GPU group 工作。
4. 静态分区的一侧会因随机长请求 burst 排队，同时另一侧无法借出容量。

表现：静态分区 SLO 89.4%、吞吐 10.81 req/s；Flex MPS 100%、13.35 req/s。全 TP2 在这个
特定负载已达 99.9%，说明默认策略还可通过更保守的 TP4 upgrade 降低 GPU 成本。

### 8.4 分阶段流量

1. 前 20s 短请求阶段，两个 group 都应运行 TP2-A+TP2-B，不保留固定 TP4 分区。
2. 中间长请求阶段，未提交 bundle 升级 TP4；为紧 deadline 预留完整 group。
3. 后 20s 混合阶段恢复低 TP 优先，并选择性 MPS。
4. 静态 split 始终有半数 GPU 绑定某个请求类型，阶段变化时容量闲置。

表现：Flex MPS 与全 TP4 都为 100% SLO，但 Flex 82% 请求用 TP2、GPU 成本更低；静态分区
只有 94.4% SLO。这是动态 TP 相对单一 TP/静态分区的主要目标场景。

### 8.5 全 8K、紧 SLO、接近过载

1. 单请求 TP2 约 1.069s，只剩约 181ms 排队 margin；Poisson burst 很容易超时。
2. TP4 约 0.606s，但两个 TP4 的短期 deadline capacity 仍不足以吸收所有 burst。
3. batching 对单个 8K 请求帮助很小，MPS 与 TP2 重叠还会减少 TP4 margin。
4. scheduler 应 admission reject，而不是硬塞入内部队列。

表现：所有动态/单 TP 配置都只有约 77-80% SLO。这个场景否定“只要调度足够聪明就能保证
任意 offered load”的说法。

### 8.6 两个紧 TP4 长请求后跟短 burst

1. 两个 8K 请求几乎同时到达，分别占用两个 group 的 TP4。
2. 35ms 后短 burst 到达；TP2 单独运行可 batch，但与 TP4 物理重叠。
3. slowdown=1.3 时，TP4+TP2 仍能守住长 1.05s 和短 0.9s deadline，允许 overlap。
4. slowdown=1.6/2.0 时，新短 batch 虽可能对“当前一次完成”看似可行，却会减少下一周期
   TP4 capacity；需要 p99 margin 和 horizon reservation，必要时等 TP4 完成。

表现：1.3 时 Flex MPS 100%，优于全 TP4 93.6%；1.6 时盲目 MPS 只有 89.4%，无 MPS Flex
为 93.0%。这是 MPS mode profile 和未来容量保护的压力测试。

---

## 9. 实施顺序

### P0：先保证状态正确并恢复可测性

1. 修 NIXL：新增真实 `PREFILL_FINISHED`，删除 `prompt_ids ready -> done`。
2. 异常/取消改为 `ABORTING`，收到 worker ACK 或 generation 失效后再归还 credit。
3. 注册协议强制携带物理 GPU IDs、group ID、instance generation。
4. 增加 request/batch 生命周期日志和 worker `INSTANCE_REPORT`。
5. 统一 6000/8192/`batch_max_tokens*2` 的参数语义。

没有 P0，后续 SLO 模型输入就是错误的。

### P1：有界 credit，解除单请求瓶颈

1. `InstanceState.running` 改为 admitted request/bundle map。
2. 先设置保守 request 和 first-chunk token credit；允许一个 instance 接收多个短请求。
3. batching 仍由 local Router 完成，采集真实 batch PERF 数据。
4. 保留当前 selector 作为 `max_admitted_requests=1` 的回退开关，便于 A/B。

这是最小可用阶段，但不宣称严格 SLO admission。

### P2：bundle/ACK 两级协同

1. 实现 `REQ_BUNDLE/BUNDLE_ACCEPTED/BATCH_STARTED/INSTANCE_REPORT`。
2. master 实现 dynamic batch window、EDF packing、deadline credit。
3. bundle 到达时立即唤醒 local Router，避免额外 30ms；旧单请求继续走兼容路径。
4. 以 batch profile 的 p95/p99 代替本文解析假设。

### P3：动态 TP、未来 reservation 和 admission control

1. 枚举 TP placement，选择满足 margin 的最小 TP。
2. 实现 pending 请求升级、TP4 future reservation、防 refill 饥饿。
3. 加 rolling deadline-window capacity check 和 reject/best-effort 分流。
4. 用 prediction error 在线校准安全 margin，但限制更新速率，避免抖动。

### P4：实测 MPS mode gating

1. 完成 mode x batch bucket profile matrix；
2. 加 MPS candidate simulation 和最小吞吐增益阈值；
3. 对 1.3/1.6/2.0 类 slowdown 反例做回归；
4. 仅当 Flex 无 MPS 已达到 SLO 基线后启用。

---

## 10. 真机 profiling 与验收

### 10.1 profiling 矩阵

对 TP2/TP4、每个 MPS mode，至少覆盖：

```text
input length:       64, 128, 256, 512, 1K, 2K, 4K, 8K, 16K, 32K
batch size:         1, 2, 4, 8, 16, 32, 64
first-chunk tokens: 0.5K, 1K, 2K, 4K, 6K, 8K, 12K, 16K
distribution:       equal short, equal long, bimodal, production trace
MPS mode:           alone, TP2+TP2, TP4+TP2, TP4+TP2+TP2
thread percentage:  default and selected caps if used
```

每点预热后至少取足够样本报告 p50/p95/p99；同时记录 model forward、postprocess、local queue、
worker tokenize 和端到端 TTFT，不能只取平均 forward latency。

### 10.2 必跑 workload

- 本文六类 synthetic trace；
- 相同均值但不同 burstiness 的 arrival；
- 90/10、50/50、10/90 短长比例；
- trace 中途改变比例和 rate；
- 大量相同 deadline、deadline 逆序、长请求持续到达；
- worker ACK 丢失、WebSocket 重连、instance generation 改变、abort race；
- NIXL 和 normal PD 分开验收；
- chunk >8192 的 16K/32K 请求与短请求交错。

### 10.3 通过标准

建议用同一 GPU 数量和同一输入 trace，对比全 TP2、全 TP4、固定静态分区、Flex 无 MPS、
Flex MPS：

1. 未过载 trace 的 admitted p99 TTFT 达标，SLO attainment 不低于目标；
2. bursty short 的平均 batch size 显著大于 1，on-time goodput 不低于静态最优；
3. 混合/phase trace 的 on-time goodput 高于每个静态分区；
4. 与全 TP4 同 SLO 时，Flex 的 TP2 占比和 GPU-s/request 显著改善；
5. Flex MPS 只有在 on-time goodput 高于 Flex 无 MPS 且 p99 不回退时才算收益；
6. prediction p99 error 被 safety margin 覆盖；
7. master admitted ID 与 worker queued/running ID 长时间不一致时自动 fail closed；
8. NIXL `prompt_ids ready` 后到真实 Prefill finished 期间，reservation 始终存在。

---

## 11. 最终建议

不要继续维护“master 精确知道每个 instance 是否忙，因此每 instance 只发一个请求”这个
抽象。它与 LightLLM local Router 的 continuous/chunked batching 能力冲突，对短请求吞吐是
结构性损失。

落地顺序应是：先修真实完成信号和拓扑/状态报告，再用有界 credit 恢复 batching，随后加入
bundle/ACK 的两级协同和 deadline admission，最后才开启经实测矩阵门控的 MPS。主要收益应由
“满足 SLO 的最小 TP + 高 fill batch”提供；MPS 用来填充可证明安全的重叠窗口，而不是成为
调度器必须追求的表面指标。
