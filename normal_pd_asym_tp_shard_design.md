# Normal PD 非对称 TP 分片传输设计（tp2 -> tp4）

## 1. 背景与目标

在 Normal PD（非 NIXL）模式下，`prefill_tp_in_node != decode_tp_in_node` 时，传统对称 rank-to-rank 传输路径不成立，需要做 KV 头维度重分片（reshard）。

本方案目标：

- 支持 Normal PD 非对称 TP（重点覆盖 `tp2 -> tp4`）
- 保持结果正确（长压测不漂移、不乱码）
- 在正确性前提下恢复并行吞吐（接近无锁版本）
- 保持对称路径的性能与行为不回退

## 2. 核心数据结构

关键结构在 `lightllm/server/pd_io_struct.py`：

- `KVMoveTask`
  - `prefill_tp_in_node` / `decode_tp_in_node`：拓扑元信息
  - `shard_total` / `shard_id`：单请求多分片传输元信息
  - `route_key`：稳定路由键
  - `task_cache_key()`：分片缓存键（单分片为 req_id，多分片为 `(req_id, shard_id)`）

## 3. 关键代码路径

### 3.1 Prefill 侧任务生成与路由

1) 任务生成

- 文件：`lightllm/server/router/model_infer/mode_backend/continues_batch/pd_mode/prefill_node_impl/prefill_impl.py`
- 函数：`_prefill_req_frozen_tokens_and_put_to_kvmove_taskqueue`
- 行为：
  - 先将 prompt KV 固化到 radix cache
  - 构建 `base_task`
  - 非对称拓扑时拆分为 `shard_total > 1` 的多个 `shard_task`
  - 所有分片入 `info_queue`

2) 连接选择与稳定路由

- 文件：`.../prefill_kv_move_manager.py`
- 函数：`__get_trans_obj`
- 行为：
  - 基于 `(decode_node_id, decode_tp_in_node)` + `route_key/group_request_id` 做 slot 路由
  - slot 绑定 `connect_id`，减少抖动和连接风暴

3) 请求批次分桶（避免分片混批）

- 文件：`.../prefill_trans_obj.py`
- 函数：`_split_by_transfer_signature`
- 签名：`(prefill_tp_in_node, decode_tp_in_node, shard_total, shard_id)`
- 行为：
  - `request_data_transfer` 返回成功任务后，按签名分桶再进入 `ready_kv_trans_task_queue`
  - 避免一个传输批次内混入多个 shard 配置造成错误重分片

### 3.2 Decode 侧分配、传输与落库

1) token 分配与分片缓存

- 文件：`.../decode_infer_rpyc.py`
- 关键状态：
  - `req_shared_alloc_cache`
  - `req_shard_cache_keys`
  - `req_put_progress`
  - `req_failed_released`
- 行为：
  - 多分片请求共享一次 decode token 分配
  - 分片按 `task_cache_key` 独立缓存
  - put/fail 时做“请求级”一次性提交或释放

2) 调度与分桶

- 文件：`.../decode_kv_move_manager.py`
- 函数：`exposed_request_data_transfer`
- 行为：
  - 按 `task_cache_key` 做 DP 分配
  - 成功任务按 transfer signature 分桶，再入 `ready_to_move_queue`

3) 传输后上报屏障（跨连接共享）

- 文件：`.../decode_kv_move_manager.py`
- 函数：`should_up_status`
- 行为：
  - 分片完成计数提升为 manager 进程级共享状态
  - 不再使用连接级局部计数，避免“每连接只见一片”导致永不 up

4) 连接对象处理

- 文件：`.../decode_trans_obj.py`
- 函数：`kv_move_loop`、`put_to_radix_loop`
- 行为：
  - 传输完成后写 radix
  - `should_up_status` 为真时才上报 `UpKVStatus`

### 3.3 KV 重分片与收发内核

文件：`lightllm/common/kv_cache_mem_manager/mem_manager.py`

1) 发送：`send_to_decode_node`

- 对称路径：保留原 rank-to-rank 快路径
- 非对称路径：
  - 按头维做 reshard
  - 按 `shard_id` 计算本分片负责的 `active_dst_ranks`
  - 仅发送本分片负责的 dst rank 数据

2) 接收：`receive_from_prefill_node`

- 非对称路径按相同分片规则仅接收/写入本分片负责的 dst rank
- 收发循环顺序严格一致（layer-major + dst-rank-major）

3) 并发安全改造（关键）

- 不再让远端卡使用共享 `kv_move_buffer` 做临时 staging
- 远端读取改为从 `kv_buffer` 切片直接拷贝到本次传输临时 buffer
- 远端接收写入改为每次传输独立临时 buffer
- 目的：避免长压测下共享 staging 被并发覆盖导致累计性乱码

## 4. 端到端数据通路（简化）

1. prefill 完成请求，构建 `KVMoveTask`（可能拆多 shard）
2. prefill manager 选择连接并发起 `request_data_transfer`
3. decode manager 分配 decode token（多 shard 共享一次）
4. prefill/decode 双侧按 transfer signature 分桶
5. 传输进程执行 reshard send/recv
6. decode 写 radix cache
7. manager 级分片屏障判定完成后上报 `UpKVStatus`
8. pd master 放行请求进入 decode 推理

## 5. 正确性机制总结

- 分片键：`task_cache_key()` 防止缓存覆盖
- 分桶：同一传输批次中只允许单一 `(tp_topology + shard)`
- 上报屏障：manager 级共享计数，防止跨连接漏计
- 收发顺序一致：避免 K/V 或 rank 写入错位
- 并发 staging 隔离：消除长压下数据污染

## 6. 当前性能观测（你提供的数据）

- prefill 发送窗口（asym）：
  - `window_s=5.49`
  - `send_asym_calls=4`
  - `send_tokens=42442`
  - `send_total_ms=43.895`

- decode 接收窗口（asym）：
  - `window_s=5.62`
  - `recv_asym_calls=6`
  - `recv_tokens=48214`
  - `recv_total_ms=28.682`

说明：

- 吞吐已恢复到接近无锁并行形态
- 结果正确，且上报链路闭环（`put kv` + `up kv status` 连续出现）

## 7. 进一步优化空间（按优先级）

### P0：低风险、可快速落地

1) 减少 Python 层循环开销

- 将 `active_dst_ranks`、overlap 映射预计算并缓存（按拓扑 + shard）
- 降低每 layer 每 rank 的动态计算成本

2) 降低临时 buffer 分配频率

- 将当前每次传输临时 buffer 升级为连接级复用池（按最大 token watermark 扩容）
- 减少 allocator 压力与碎片

3) 指标细化

- 补充 `send_copy_ms`、`recv_copy_ms`、`pack_ms`，把 `prepare_ms` 拆细
- 便于确认下一瓶颈在 copy 还是 nccl

### P1：中风险、高收益

1) 重分片打包 kernel 化

- 当前 pack 仍有较多 Python + Tensor 切片 copy
- 可考虑 Triton/CUDA kernel 直接完成 K/V 重排与打包

2) 跨连接调度自适应

- 结合历史 `pd_trans_perf` 动态调整 shard 到 connect slot 的映射
- 避免某连接长尾拖慢整体

3) 双向流水化

- 在一个连接内尝试 layer chunk pipeline（pack/send overlap）

### P2：架构级优化

1) 拓扑感知 direct path

- 在保证稳定性的前提下，逐步引入“单请求跨卡直发”更细粒度并行
- 需要更强的失败回退与一致性屏障

2) 统一传输抽象

- 将对称/非对称、p2p/non-p2p 路径抽象成统一 planner
- 减少分支复杂度和后续维护成本

## 8. 回归验证建议

每次优化至少做以下回归：

- 正确性：
  - `tp2->tp4` 长压（高并发 + 长时）
  - `tp4->tp4`、`tp4->tp2` 对照

- 性能：
  - `pd_trans_perf`：`kv_tokens_per_s`、`ms_per_kv_token`
  - `pd_kv_perf`：`prepare_ms/reshard_ms/nccl_ms/total_ms`

- 稳定性：
  - 观察是否出现 `kv move timeout`、异常重连风暴、分片漏上报
