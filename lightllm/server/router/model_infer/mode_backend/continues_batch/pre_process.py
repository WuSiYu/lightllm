import torch
import numpy as np
from lightllm.server.router.model_infer.infer_batch import requests_mapping, InferReq, InferBatch
from lightllm.server.io_struct import ReqRunStatus
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.common.mem_manager import MemoryManager
from lightllm.common.basemodel.infer_lock import g_infer_state_lock

# @calculate_time(show=True, min_cost_ms=1)
def prepare_prefill_inputs(batch: InferBatch, radix_cache: RadixCache, is_multimodal=False):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    batch_multimodal_params = []
    b_ready_cache_len = []
    for request_id in batch.request_ids:
        req: InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING

        run_reqs.append(req)
        batch_multimodal_params.append(req.multimodal_params)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)

        seq_len = len(req.input_token_ids)
        input_token_len = seq_len - req.cur_kv_len

        input_id = req.input_token_ids[req.cur_kv_len :]

        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, input_token_len)
        b_ready_cache_len.append(req.cur_kv_len)
        start_loc += input_token_len

    input_ids = np.concatenate(input_ids, dtype=np.int64)

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")
    b_ready_cache_len = torch.tensor(b_ready_cache_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if radix_cache is not None:
        radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])
    mem_indexes = batch.req_manager.mem_manager.alloc(input_ids.shape[0])
    g_infer_state_lock.release()

    kwargs = {
        "batch_size": len(batch),
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "mem_indexes": mem_indexes,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "b_ready_cache_len": b_ready_cache_len,
        "is_prefill": True,
    }
    if is_multimodal:
        kwargs["multimodal_params"] = batch_multimodal_params

    return kwargs, run_reqs


# @calculate_time(show=True, min_cost_ms=1)
def prepare_decode_inputs(batch: InferBatch, radix_cache: RadixCache):
    run_reqs = []
    nopad_total_token_num = 0
    nopad_max_len_in_batch = 0
    start_loc = 0
    input_ids = []
    nopad_b_req_idx = []
    nopad_b_start_loc = []
    nopad_b_seq_len = []
    for request_id in batch.request_ids:
        req: InferReq = requests_mapping[request_id]
        assert req.req_status == ReqRunStatus.RUNNING
        run_reqs.append(req)
        nopad_b_req_idx.append(req.req_idx)
        nopad_b_start_loc.append(start_loc)
        input_id = req.input_token_ids[-1]
        seq_len = len(req.input_token_ids)
        assert req.cur_kv_len == seq_len - 1
        nopad_b_seq_len.append(seq_len)
        input_ids.append(input_id)
        nopad_total_token_num += seq_len
        nopad_max_len_in_batch = max(nopad_max_len_in_batch, seq_len)
        start_loc += seq_len

    input_ids = torch.tensor(input_ids, dtype=torch.int64, device="cuda")
    nopad_b_req_idx = torch.tensor(nopad_b_req_idx, dtype=torch.int32, device="cuda")
    nopad_b_start_loc = torch.tensor(nopad_b_start_loc, dtype=torch.int32, device="cuda")
    nopad_b_seq_len = torch.tensor(nopad_b_seq_len, dtype=torch.int32, device="cuda")

    # dynamic prompt cache 准备 token
    g_infer_state_lock.acquire()
    if radix_cache is not None:
        radix_cache.free_radix_cache_to_get_enough_token(input_ids.shape[0])
    mem_indexes = batch.req_manager.mem_manager.alloc(input_ids.shape[0])
    g_infer_state_lock.release()

    kwargs = {
        "batch_size": len(batch),
        "total_token_num": nopad_total_token_num,
        "max_len_in_batch": nopad_max_len_in_batch,
        "input_ids": input_ids,
        "mem_indexes": mem_indexes,
        "b_req_idx": nopad_b_req_idx,
        "b_start_loc": nopad_b_start_loc,
        "b_seq_len": nopad_b_seq_len,
        "is_prefill": False,
    }
    return kwargs, run_reqs
