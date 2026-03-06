### 2026/03/06

basic cross-tp gemm weight sharing

```
CUDA_VISIBLE_DEVICES=0 LOADWORKER=12 python -m lightllm.server.api_server --port 8000 --model_dir /mtc/models/qwen3-8b --tp 1 --disable_dynamic_prompt_cache --disable_cudagraph --mem_fraction 0.3 --shared_weight=master --shared_weight_master_port_start=10000 --nccl_port 20000
CUDA_VISIBLE_DEVICES=1 LOADWORKER=12 python -m lightllm.server.api_server --port 8001 --model_dir /mtc/models/qwen3-8b --tp 1 --disable_dynamic_prompt_cache --disable_cudagraph --mem_fraction 0.3 --shared_weight=master --shared_weight_master_port_start=10001 --nccl_port 20001


CUDA_VISIBLE_DEVICES=0,1 LOADWORKER=12 python -m lightllm.server.api_server --port 8002 --model_dir /mtc/models/qwen3-8b --tp 2 --disable_dynamic_prompt_cache --disable_cudagraph --mem_fraction 0.5 --shared_weight=slave --shared_weight_master_port_start=10000 --nccl_port 20002

```
