### 2026/03/06

basic cross-tp gemm weight sharing

```
CUDA_VISIBLE_DEVICES=0 LOADWORKER=12 python -m lightllm.server.api_server --port 8000 --model_dir /mtc/models/qwen3-8b --tp 1 --disable_dynamic_prompt_cache --disable_cudagraph --mem_fraction 0.3 --shared_weight=master --shared_weight_master_port_start=10000 --nccl_port 20000
CUDA_VISIBLE_DEVICES=1 LOADWORKER=12 python -m lightllm.server.api_server --port 8001 --model_dir /mtc/models/qwen3-8b --tp 1 --disable_dynamic_prompt_cache --disable_cudagraph --mem_fraction 0.3 --shared_weight=master --shared_weight_master_port_start=10001 --nccl_port 20001


CUDA_VISIBLE_DEVICES=0,1 LOADWORKER=12 python -m lightllm.server.api_server --port 8002 --model_dir /mtc/models/qwen3-8b --tp 2 --disable_dynamic_prompt_cache --disable_cudagraph --mem_fraction 0.5 --shared_weight=slave --shared_weight_master_port_start=10000 --nccl_port 20002

```

### 2026/03/26

2×tp2 + 1×tp4 shared weight (physical GPU ID port routing)

端口计算改用物理 GPU ID (`get_physical_device_id()`)，所有实例共用同一个 `port_start`。
slave 通过 `CUDA_VISIBLE_DEVICES` 重排实现 rank 布局 `0,2,1,3`，与两组 tp2 的 rank `0,1,0,1` 对齐。

通用公式：N 组 tpK master + 1 组 tp(N*K) slave，slave rank r 放在物理 GPU `(r % N) * K + (r // N)` 上。

```
# Master A (tp2, GPU 0,1)
CUDA_VISIBLE_DEVICES=0,1 LOADWORKER=12 python -m lightllm.server.api_server --port 8000 --model_dir /mtc/wusiyu/models/llama3-70b --tp 2 --disable_dynamic_prompt_cache --disable_cudagraph --max_total_token_num 17000 --shared_weight=master --shared_weight_master_port_start=10000 --nccl_port 20000

# Master B (tp2, GPU 2,3)
CUDA_VISIBLE_DEVICES=2,3 LOADWORKER=12 python -m lightllm.server.api_server --port 8001 --model_dir /mtc/wusiyu/models/llama3-70b --tp 2 --disable_dynamic_prompt_cache --disable_cudagraph --max_total_token_num 17000 --shared_weight=master --shared_weight_master_port_start=10002 --nccl_port 20001

# Slave (tp4, GPU 重排: 0,2,1,3 使 tp4 rank 0,2,1,3 分别对应物理 GPU 0,1,2,3)
CUDA_VISIBLE_DEVICES=0,2,1,3 LOADWORKER=12 python -m lightllm.server.api_server --port 8002 --model_dir /mtc/wusiyu/models/llama3-70b --tp 4 --disable_dynamic_prompt_cache --disable_cudagraph --max_total_token_num 17000 --shared_weight=slave --shared_weight_master_port_start=10000 --nccl_port 20002
```
