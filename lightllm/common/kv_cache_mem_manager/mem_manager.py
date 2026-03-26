import re
import os
import time
import threading
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from typing import List, Union, Tuple, Any
from lightllm.common.kv_trans_kernel.kv_trans_v2 import kv_trans_for_dp
from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.dynamic_prompt.shared_arr import SharedInt
from lightllm.utils.profile_max_tokens import get_available_gpu_memory, get_total_gpu_memory
from lightllm.common.kv_trans_kernel.kv_trans import kv_trans
from lightllm.utils.dist_utils import get_current_rank_in_node, get_node_world_size
from lightllm.utils.envs_utils import get_unique_server_name, get_env_start_args
from lightllm.distributed.pynccl import PyNcclCommunicator
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.config_utils import get_num_key_value_heads
from lightllm.common.kv_trans_kernel.nixl_kv_trans import page_io
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.shm_utils import create_or_link_shm
from multiprocessing.reduction import ForkingPickler
from filelock import FileLock


logger = init_logger(__name__)


class _PDKVPerfMetric:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_log_ts = time.time()
        self._log_interval_s = 5.0
        self._stats = {
            "send_sym_calls": 0,
            "send_asym_calls": 0,
            "recv_sym_calls": 0,
            "recv_asym_calls": 0,
            "send_tokens": 0,
            "recv_tokens": 0,
            "send_bytes": 0,
            "recv_bytes": 0,
            "send_broadcast_ops": 0,
            "recv_broadcast_ops": 0,
            "send_prepare_ms": 0.0,
            "send_reshard_ms": 0.0,
            "send_nccl_ms": 0.0,
            "send_total_ms": 0.0,
            "recv_nccl_ms": 0.0,
            "recv_write_ms": 0.0,
            "recv_total_ms": 0.0,
        }

    def add(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if key in self._stats:
                    self._stats[key] += value
            self._maybe_log_locked()

    def _maybe_log_locked(self):
        now = time.time()
        if now - self._last_log_ts < self._log_interval_s:
            return
        s = self._stats
        interval = now - self._last_log_ts
        self._last_log_ts = now
        logger.info(
            "pd_kv_perf "
            f"window_s={interval:.2f} "
            f"send_sym_calls={s['send_sym_calls']} send_asym_calls={s['send_asym_calls']} "
            f"recv_sym_calls={s['recv_sym_calls']} recv_asym_calls={s['recv_asym_calls']} "
            f"send_tokens={s['send_tokens']} recv_tokens={s['recv_tokens']} "
            f"send_bytes={s['send_bytes']} recv_bytes={s['recv_bytes']} "
            f"send_broadcast_ops={s['send_broadcast_ops']} recv_broadcast_ops={s['recv_broadcast_ops']} "
            f"send_prepare_ms={s['send_prepare_ms']:.3f} send_reshard_ms={s['send_reshard_ms']:.3f} "
            f"send_nccl_ms={s['send_nccl_ms']:.3f} send_total_ms={s['send_total_ms']:.3f} "
            f"recv_nccl_ms={s['recv_nccl_ms']:.3f} recv_write_ms={s['recv_write_ms']:.3f} "
            f"recv_total_ms={s['recv_total_ms']:.3f}"
        )
        for key in self._stats:
            self._stats[key] = 0 if not isinstance(self._stats[key], float) else 0.0


_pd_kv_perf_metric = _PDKVPerfMetric()


class MemoryManager:
    def __init__(self, size, dtype, head_num, head_dim, layer_num, always_copy=False, mem_fraction=0.9):
        self.size = size
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.always_copy = always_copy
        self.dtype = dtype
        # profile the max total token num if the size is None
        self.profile_size(mem_fraction)

        self.mem_state = torch.arange(
            0, self.size, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self._mem_state_return = torch.arange(
            0, self.size * 3, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self._return_start = 0
        self.mark_start = 0
        self.mark_end = self.size

        self.can_use_mem_size = self.size

        # 用共享内存进行共享，router 模块读取进行精确的调度估计, nccl port 作为一个单机中单实列的标记。防止冲突。
        from lightllm.utils.envs_utils import get_unique_server_name

        rank_in_node = get_current_rank_in_node()
        self.shared_can_use_token_num = SharedInt(
            f"{get_unique_server_name()}_mem_manger_can_use_token_num_{rank_in_node}"
        )

        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self._init_buffers(
            self.size,
            dtype,
            head_num,
            head_dim,
            layer_num,
        )
        self.HOLD_TOKEN_MEMINDEX = self.size

    def copy_kv_to_mem_manager(self, layer_index: int, mem_index: torch.Tensor, kv: torch.Tensor):
        """
        将每一层生成的kv拷贝到mem manager对应mem_index 位置中
        """
        from lightllm.common.basemodel.triton_kernel.destindex_copy_kv import destindex_copy_kv

        destindex_copy_kv(kv, mem_index, self.kv_buffer[layer_index])
        return

    def get_att_input_params(self, layer_index: int) -> Tuple[Any, Any]:
        k = self.kv_buffer[layer_index][:, : self.head_num, :]
        v = self.kv_buffer[layer_index][:, self.head_num :, :]
        return k, v

    def get_cell_size(self):
        return 2 * self.head_num * self.head_dim * self.layer_num * torch._utils._element_size(self.dtype)

    def profile_size(self, mem_fraction):
        if self.size is not None:
            return

        world_size = dist.get_world_size()
        total_memory = get_total_gpu_memory()
        available_memory = get_available_gpu_memory(world_size) - total_memory * (1 - mem_fraction)
        cell_size = self.get_cell_size()
        self.size = int(available_memory * 1024 ** 3 / cell_size)
        if world_size > 1:
            tensor = torch.tensor(self.size, dtype=torch.int64, device=f"cuda:{get_current_device_id()}")
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            self.size = tensor.item()
        logger.info(
            f"{str(available_memory)} GB space is available after load the model weight\n"
            f"{str(cell_size / 1024 ** 2)} MB is the size of one token kv cache\n"
            f"{self.size} is the profiled max_total_token_num with the mem_fraction {mem_fraction}\n"
        )
        return

    def _init_buffers(self, size, dtype, head_num, head_dim, layer_num):
        # 在初始化 kv_buffer 的时候，每层多初始化了一个 token，这个 token 永远不会被真的被对外
        # 分配，内部实际也没有管理，这个token是预留来对一些特殊的运行模式，如多dp下，overlap microbatch
        # 等模式下 padding 一些请求，使推理过程可以正常运行采用的，其索引值为size，存储在HOLD_TOKEN_MEMINDEX
        # 成员变量中，其与 req_manager 中的HOLD_REQUEST_ID具有类似的作用和意义。
        self.kv_buffer = torch.empty((layer_num, size + 1, 2 * head_num, head_dim), dtype=dtype, device="cuda")

    def alloc_kv_move_buffer(self, max_req_total_len):
        """
        pd 分离模式使用的特殊接口
        """
        if isinstance(self, MemoryManager) and type(self) is not MemoryManager:
            raise NotImplementedError("subclass need reimpl this method")
        self.kv_move_buffer = torch.empty(
            (1, max_req_total_len + 8, 2 * self.head_num, self.head_dim), dtype=self.dtype, device="cuda"
        )
        self.kv_move_buf_indexes = torch.arange(0, max_req_total_len + 8, dtype=torch.int64, device="cuda")
        self.token_dim_size = self.kv_move_buffer.shape[-2] * self.kv_move_buffer.shape[-1]
        return

    def alloc_paged_kv_move_buffer(self, page_num, page_size) -> torch.Tensor:
        if isinstance(self, MemoryManager) and type(self) is not MemoryManager:
            raise NotImplementedError("subclass need reimpl this method")

        num_kv_head = get_num_key_value_heads(get_env_start_args().model_dir)
        self.kv_move_buffer = torch.empty(
            (page_num, page_size, self.layer_num, 2 * num_kv_head, self.head_dim), dtype=self.dtype, device="cuda"
        )
        self._buffer_mem_indexes_tensors = [
            torch.empty((page_size,), dtype=torch.int64, device="cpu", pin_memory=True) for _ in range(page_num)
        ]
        return self.kv_move_buffer

    def write_mem_to_page_kv_move_buffer(
        self,
        mem_indexes: List[int],
        page_index: int,
        dp_index: int,
        mem_managers: List["MemoryManager"],
        dp_world_size: int,
    ):
        cur_page = self.kv_move_buffer[page_index]
        pin_mem_indexes = self._buffer_mem_indexes_tensors[page_index][0 : len(mem_indexes)]
        pin_mem_indexes.numpy()[:] = mem_indexes
        mem_indexes_gpu = pin_mem_indexes.cuda(non_blocking=True)
        repeat_count = dp_world_size * self.kv_buffer.shape[2] // self.kv_move_buffer.shape[3]
        dp_mems = mem_managers[(dp_index * dp_world_size) : ((dp_index + 1) * dp_world_size)]
        for tp_index in range(dp_world_size):
            if tp_index % repeat_count == 0:
                page_io(
                    mem_indexes=mem_indexes_gpu,
                    page_tensor=cur_page,
                    kv_buffer=dp_mems[tp_index].kv_buffer,
                    tp_index=tp_index,
                    tp_world_size=dp_world_size,
                    mode="write",
                )
        # keep for debug
        # logger.info(f"src token tensor {self.kv_buffer[:, mem_indexes[0], 0, 0]}")
        # logger.info(f"src page token tensor {cur_page[0, :, 0, 0]}")
        return

    def read_page_kv_move_buffer_to_mem(
        self,
        mem_indexes: List[int],
        page_index: int,
        dp_index: int,
        mem_managers: List["MemoryManager"],
        dp_world_size: int,
    ):
        cur_page = self.kv_move_buffer[page_index]
        pin_mem_indexes = self._buffer_mem_indexes_tensors[page_index][0 : len(mem_indexes)]
        pin_mem_indexes.numpy()[:] = mem_indexes
        mem_indexes_gpu = pin_mem_indexes.cuda(non_blocking=True)
        dp_mems = mem_managers[(dp_index * dp_world_size) : ((dp_index + 1) * dp_world_size)]
        mem_indexes_gpu = torch.tensor(mem_indexes, dtype=torch.int64, device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )
        for tp_index in range(dp_world_size):
            page_io(
                mem_indexes=mem_indexes_gpu,
                page_tensor=cur_page,
                kv_buffer=dp_mems[tp_index].kv_buffer,
                tp_index=tp_index,
                tp_world_size=dp_world_size,
                mode="read",
            )
        # keep for debug
        # logger.info(f"dst token tensor {self.kv_buffer[:, mem_indexes[0], 0, 0]}")
        # logger.info(f"dst page token tensor {cur_page[0, :, 0, 0]}")

    def send_to_decode_node(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["MemoryManager"],
        dp_size_in_node: int,
        nccl_comm: PyNcclCommunicator,
    ):
        assert dp_size_in_node == 1
        total_start = time.perf_counter()

        # 先将数据发送到指定的一张卡上的buffer，再发送。

        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.prefill_token_indexes[-task.move_kv_len :])

        if len(move_token_indexes) == 0:
            return

        src_tp_in_node = len(mem_managers)
        dst_tp_in_node = move_tasks[0].decode_tp_in_node if move_tasks[0].decode_tp_in_node is not None else src_tp_in_node
        token_num = len(move_token_indexes)
        move_size = self.token_dim_size * token_num
        token_indexes_by_device = {}
        for mem in mem_managers:
            dev_idx = mem.kv_buffer.device.index
            if dev_idx not in token_indexes_by_device:
                token_indexes_by_device[dev_idx] = torch.tensor(
                    move_token_indexes, dtype=torch.int64, device=mem.kv_buffer.device
                )

        # fast path: keep old rank-to-rank behavior for symmetric topology.
        if src_tp_in_node == dst_tp_in_node:
            cur_device_index = self.kv_buffer.get_device()
            cur_mem = mem_managers[cur_device_index]
            remote_move_buffer_views = {
                i: cur_mem.kv_move_buffer.view(-1)[0:move_size].view(1, token_num, 2 * mem.head_num, self.head_dim)
                for i, mem in enumerate(mem_managers)
                if i != cur_device_index
            }
            prepare_ms = 0.0
            nccl_ms = 0.0
            bcast_ops = 0
            for i, mem in enumerate(mem_managers):
                for layer_index in range(mem.layer_num):
                    t0 = time.perf_counter()
                    token_indexes = token_indexes_by_device[mem.kv_buffer.device.index]
                    if i == cur_device_index:
                        move_buffer = mem._get_kv_move_data(token_indexes, layer_index)
                        prepare_ms += (time.perf_counter() - t0) * 1000.0
                        t1 = time.perf_counter()
                        nccl_comm.send(move_buffer, dst=1)
                        nccl_ms += (time.perf_counter() - t1) * 1000.0
                    else:
                        src_slice = mem.kv_buffer[layer_index : layer_index + 1, token_indexes, :, :]
                        t1 = time.perf_counter()
                        new_move_buffer = remote_move_buffer_views[i]
                        new_move_buffer.copy_(src_slice, non_blocking=False)
                        bcast_ops += 1
                        prepare_ms += (time.perf_counter() - t0) * 1000.0
                        t2 = time.perf_counter()
                        nccl_comm.send(new_move_buffer, dst=1)
                        nccl_ms += (time.perf_counter() - t2) * 1000.0
            total_ms = (time.perf_counter() - total_start) * 1000.0
            send_bytes = (
                len(move_token_indexes)
                * mem_managers[0].layer_num
                * (2 * mem_managers[0].head_num * src_tp_in_node)
                * self.head_dim
                * torch._utils._element_size(self.dtype)
            )
            _pd_kv_perf_metric.add(
                send_sym_calls=1,
                send_tokens=len(move_token_indexes),
                send_bytes=send_bytes,
                send_broadcast_ops=bcast_ops,
                send_prepare_ms=prepare_ms,
                send_nccl_ms=nccl_ms,
                send_total_ms=total_ms,
            )
            return

        # asymmetric path: re-shard kv by global head dimension and send by decode tp layout.
        cur_device_index = self.kv_buffer.get_device()
        cur_mem = mem_managers[cur_device_index]

        src_head_num = mem_managers[0].head_num
        total_head_num = src_head_num * src_tp_in_node
        if total_head_num % dst_tp_in_node != 0:
            raise ValueError(
                f"Unsupported asymmetric tp layout: total_head_num={total_head_num}, "
                f"dst_tp_in_node={dst_tp_in_node}"
            )
        dst_head_num = total_head_num // dst_tp_in_node
        shard_total = int(getattr(move_tasks[0], "shard_total", 1))
        shard_id = int(getattr(move_tasks[0], "shard_id", 0))
        if shard_total > 1:
            shard_total = min(shard_total, dst_tp_in_node)
            base = dst_tp_in_node // shard_total
            rem = dst_tp_in_node % shard_total
            start_rank = shard_id * base + min(shard_id, rem)
            rank_num = base + (1 if shard_id < rem else 0)
            active_dst_ranks = list(range(start_rank, start_rank + rank_num))
        else:
            active_dst_ranks = list(range(dst_tp_in_node))
        if len(active_dst_ranks) == 0:
            return

        # Pre-compute reshard plan: determine which src_ranks contribute to each
        # dst_rank, and cache the head-overlap slicing parameters. This avoids
        # recomputing overlaps in the O(layers * dst_ranks * src_ranks) inner loop
        # and, critically, identifies which source ranks are actually needed so we
        # skip expensive cross-GPU gathers for unused sources.
        active_src_ranks = set()
        # reshard_plan: list of (dst_rank, ops, expected_packed)
        #   ops: list of (src_rank, local_start, local_end, copy_len, pack_offset)
        reshard_plan = []
        for dst_rank in active_dst_ranks:
            global_start = dst_rank * dst_head_num
            global_end = global_start + dst_head_num
            ops = []
            packed = 0
            for src_rank in range(src_tp_in_node):
                src_start = src_rank * src_head_num
                src_end = src_start + src_head_num
                overlap_start = max(global_start, src_start)
                overlap_end = min(global_end, src_end)
                if overlap_start >= overlap_end:
                    continue
                local_start = overlap_start - src_start
                local_end = overlap_end - src_start
                copy_len = local_end - local_start
                ops.append((src_rank, local_start, local_end, copy_len, packed))
                packed += copy_len
                active_src_ranks.add(src_rank)
            if packed != dst_head_num:
                raise ValueError(
                    f"Invalid overlap while resharding: dst_rank={dst_rank}, "
                    f"packed_head_num={packed}, dst_head_num={dst_head_num}, "
                    f"src_head_num={src_head_num}, src_tp_in_node={src_tp_in_node}"
                )
            reshard_plan.append((dst_rank, ops))

        # Only allocate staging buffers for remote active source ranks (skip unused
        # sources entirely — e.g. for tp2→tp4 shard_total=2, each shard only needs
        # one of the two source ranks).
        remote_staging = {
            src_rank: torch.empty(
                (1, token_num, 2 * src_head_num, self.head_dim),
                dtype=self.dtype,
                device=cur_mem.kv_buffer.device,
            )
            for src_rank in active_src_ranks
            if src_rank != cur_device_index
        }

        send_pack_buffer = torch.empty(
            (1, token_num, 2 * dst_head_num, self.head_dim), dtype=self.dtype, device=cur_mem.kv_buffer.device
        )

        prepare_ms = 0.0
        reshard_ms = 0.0
        nccl_ms = 0.0
        bcast_ops = 0

        for layer_index in range(mem_managers[0].layer_num):
            # Gather phase: only stage from active source ranks.
            t_prep = time.perf_counter()
            local_src_buffer = None
            for src_rank in active_src_ranks:
                if src_rank == cur_device_index:
                    mem = mem_managers[src_rank]
                    local_src_buffer = mem._get_kv_move_data(
                        token_indexes_by_device[mem.kv_buffer.device.index], layer_index
                    )
                else:
                    mem = mem_managers[src_rank]
                    token_indexes = token_indexes_by_device[mem.kv_buffer.device.index]
                    remote_staging[src_rank].copy_(
                        mem.kv_buffer[layer_index : layer_index + 1, token_indexes, :, :],
                        non_blocking=False,
                    )
                    bcast_ops += 1
            prepare_ms += (time.perf_counter() - t_prep) * 1000.0

            # Reshard + send: pack heads for each dst_rank using pre-computed plan.
            for dst_rank, ops in reshard_plan:
                t_reshard = time.perf_counter()
                for src_rank, ls, le, cl, ps in ops:
                    src_buf = local_src_buffer if src_rank == cur_device_index else remote_staging[src_rank]
                    send_pack_buffer[:, :, ps : ps + cl, :].copy_(
                        src_buf[:, :, ls:le, :], non_blocking=False
                    )
                    send_pack_buffer[:, :, dst_head_num + ps : dst_head_num + ps + cl, :].copy_(
                        src_buf[:, :, src_head_num + ls : src_head_num + le, :], non_blocking=False
                    )
                reshard_ms += (time.perf_counter() - t_reshard) * 1000.0

                t_nccl = time.perf_counter()
                nccl_comm.send(send_pack_buffer, dst=1)
                nccl_ms += (time.perf_counter() - t_nccl) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0
        send_bytes = (
            len(move_token_indexes)
            * mem_managers[0].layer_num
            * (2 * dst_head_num * len(active_dst_ranks))
            * self.head_dim
            * torch._utils._element_size(self.dtype)
        )
        _pd_kv_perf_metric.add(
            send_asym_calls=1,
            send_tokens=len(move_token_indexes),
            send_bytes=send_bytes,
            send_broadcast_ops=bcast_ops,
            send_prepare_ms=prepare_ms,
            send_reshard_ms=reshard_ms,
            send_nccl_ms=nccl_ms,
            send_total_ms=total_ms,
        )
        return

    def _get_kv_move_data(self, token_indexes: Union[List[int], torch.Tensor], layer_index: int):
        move_size = self.token_dim_size * len(token_indexes)
        move_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(
            1, len(token_indexes), 2 * self.head_num, self.head_dim
        )
        move_buffer[:, :, :, :] = self.kv_buffer[layer_index, token_indexes, :, :]
        return move_buffer

    def receive_from_prefill_node(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["MemoryManager"],
        dp_size_in_node: int,
        nccl_comm: PyNcclCommunicator,
    ):
        assert dp_size_in_node == 1
        total_start = time.perf_counter()

        # 先将数据接受到指定的一张卡上的buffer，再复制到其他的卡上。

        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.decode_token_indexes[-task.move_kv_len :])

        if len(move_token_indexes) == 0:
            return

        token_indexes_by_device = {}
        for mem in mem_managers:
            dev_idx = mem.kv_buffer.device.index
            if dev_idx not in token_indexes_by_device:
                token_indexes_by_device[dev_idx] = torch.tensor(
                    move_token_indexes, dtype=torch.int64, device=mem.kv_buffer.device
                )

        decode_tp_in_node = move_tasks[0].decode_tp_in_node if move_tasks[0].decode_tp_in_node is not None else len(mem_managers)
        prefill_tp_in_node = move_tasks[0].prefill_tp_in_node if move_tasks[0].prefill_tp_in_node is not None else decode_tp_in_node
        if decode_tp_in_node != len(mem_managers):
            raise ValueError(
                f"Decode tp topology mismatch: task decode_tp_in_node={decode_tp_in_node}, "
                f"local mem_managers={len(mem_managers)}"
            )

        if move_tasks[0].move_kv_len != 0:
            logger.info(
                f"pd receive kv topology prefill_tp_in_node={prefill_tp_in_node} "
                f"decode_tp_in_node={decode_tp_in_node} token_num={len(move_token_indexes)}"
            )

        cur_device_index = self.kv_buffer.get_device()
        token_num = len(move_token_indexes)
        move_size = self.token_dim_size * token_num
        recive_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(1, token_num, 2 * self.head_num, self.head_dim)
        expected_shape = (1, token_num, 2 * self.head_num, self.head_dim)
        remote_recv_buffer_views = {
            i: torch.empty(recive_buffer.shape, dtype=self.dtype, device=mem.kv_buffer.device)
            for i, mem in enumerate(mem_managers)
            if i != cur_device_index
        }

        if prefill_tp_in_node == decode_tp_in_node:
            recv_nccl_ms = 0.0
            recv_write_ms = 0.0
            bcast_ops = 0
            for i, mem in enumerate(mem_managers):
                for layer_index in range(mem.layer_num):
                    t0 = time.perf_counter()
                    nccl_comm.recv(recive_buffer, src=0)
                    recv_nccl_ms += (time.perf_counter() - t0) * 1000.0
                    if recive_buffer.shape != expected_shape:
                        raise ValueError(f"Unexpected recv buffer shape {recive_buffer.shape}, expect {expected_shape}")
                    if i == cur_device_index:
                        t1 = time.perf_counter()
                        token_indexes = token_indexes_by_device[mem.kv_buffer.device.index]
                        mem._write_kv_move_data(token_indexes, recive_buffer, layer_index)
                        recv_write_ms += (time.perf_counter() - t1) * 1000.0
                    else:
                        new_recive_buffer = remote_recv_buffer_views[i]
                        t1 = time.perf_counter()
                        new_recive_buffer.copy_(recive_buffer, non_blocking=False)
                        bcast_ops += 1
                        token_indexes = token_indexes_by_device[mem.kv_buffer.device.index]
                        mem._write_kv_move_data(token_indexes, new_recive_buffer, layer_index)
                        recv_write_ms += (time.perf_counter() - t1) * 1000.0
            total_ms = (time.perf_counter() - total_start) * 1000.0
            recv_bytes = (
                len(move_token_indexes)
                * mem_managers[0].layer_num
                * (2 * self.head_num * decode_tp_in_node)
                * self.head_dim
                * torch._utils._element_size(self.dtype)
            )
            _pd_kv_perf_metric.add(
                recv_sym_calls=1,
                recv_tokens=len(move_token_indexes),
                recv_bytes=recv_bytes,
                recv_broadcast_ops=bcast_ops,
                recv_nccl_ms=recv_nccl_ms,
                recv_write_ms=recv_write_ms,
                recv_total_ms=total_ms,
            )
            return

        # For asymmetric topology, recv order must match sender: layer-major then dst-rank-major.
        shard_total = int(getattr(move_tasks[0], "shard_total", 1))
        shard_id = int(getattr(move_tasks[0], "shard_id", 0))
        if shard_total > 1:
            shard_total = min(shard_total, decode_tp_in_node)
            base = decode_tp_in_node // shard_total
            rem = decode_tp_in_node % shard_total
            start_rank = shard_id * base + min(shard_id, rem)
            rank_num = base + (1 if shard_id < rem else 0)
            active_dst_ranks = list(range(start_rank, start_rank + rank_num))
        else:
            active_dst_ranks = list(range(decode_tp_in_node))
        if len(active_dst_ranks) == 0:
            return

        # Pre-compute per-rank write targets to avoid dict lookups and conditionals
        # in the O(layers * active_dst_ranks) inner loop.
        recv_rank_info = []  # [(mem, token_indexes, is_local, remote_buf)]
        for i in active_dst_ranks:
            mem = mem_managers[i]
            tidx = token_indexes_by_device[mem.kv_buffer.device.index]
            is_local = i == cur_device_index
            remote_buf = None if is_local else remote_recv_buffer_views[i]
            recv_rank_info.append((mem, tidx, is_local, remote_buf))

        recv_nccl_ms = 0.0
        recv_write_ms = 0.0
        bcast_ops = 0
        for layer_index in range(mem_managers[0].layer_num):
            for mem, tidx, is_local, remote_buf in recv_rank_info:
                t0 = time.perf_counter()
                nccl_comm.recv(recive_buffer, src=0)
                recv_nccl_ms += (time.perf_counter() - t0) * 1000.0
                t1 = time.perf_counter()
                if is_local:
                    mem._write_kv_move_data(tidx, recive_buffer, layer_index)
                else:
                    remote_buf.copy_(recive_buffer, non_blocking=False)
                    bcast_ops += 1
                    mem._write_kv_move_data(tidx, remote_buf, layer_index)
                recv_write_ms += (time.perf_counter() - t1) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        recv_bytes = (
            len(move_token_indexes)
            * mem_managers[0].layer_num
            * (2 * self.head_num * len(active_dst_ranks))
            * self.head_dim
            * torch._utils._element_size(self.dtype)
        )
        _pd_kv_perf_metric.add(
            recv_asym_calls=1,
            recv_tokens=len(move_token_indexes),
            recv_bytes=recv_bytes,
            recv_broadcast_ops=bcast_ops,
            recv_nccl_ms=recv_nccl_ms,
            recv_write_ms=recv_write_ms,
            recv_total_ms=total_ms,
        )
        return

    def _write_kv_move_data(self, token_indexes: torch.Tensor, buffer_tensor: torch.Tensor, layer_index):
        self.kv_buffer[layer_index : layer_index + 1, token_indexes, :, :] = buffer_tensor
        return

    def send_to_decode_node_p2p(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["MemoryManager"],
        dp_size_in_node: int,
        nccl_comm: PyNcclCommunicator,
    ):
        """
        使用 p2p triton kernel 进行数据复制和传输的实现方式。
        """
        assert dp_size_in_node == 1

        # 先将数据发送到指定的一张卡上的buffer，再发送。

        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.prefill_token_indexes[-task.move_kv_len :])

        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")
        for i, mem in enumerate(mem_managers):
            for layer_index in range(mem.layer_num):
                move_buffer = mem._get_kv_move_data_p2p(move_token_indexes, layer_index, self.kv_move_buffer)
                nccl_comm.send(move_buffer, dst=1)
        return

    def _get_kv_move_data_p2p(self, token_indexes: torch.Tensor, layer_index: int, kv_move_buffer: torch.Tensor):
        move_token_num = len(token_indexes)
        move_size = self.token_dim_size * move_token_num
        move_buffer = kv_move_buffer.view(-1)[0:move_size].view(move_token_num, 2 * self.head_num, self.head_dim)
        kv_trans(
            self.kv_buffer[layer_index, :, :, :], token_indexes, move_buffer, self.kv_move_buf_indexes[0:move_token_num]
        )
        return move_buffer

    def receive_from_prefill_node_p2p(
        self,
        move_tasks: List[KVMoveTask],
        mem_managers: List["MemoryManager"],
        dp_size_in_node: int,
        nccl_comm: PyNcclCommunicator,
    ):
        assert dp_size_in_node == 1

        # 先将数据接受到指定的一张卡上的buffer，再复制到其他的卡上。

        move_token_indexes = []
        for task in move_tasks:
            if task.move_kv_len != 0:
                move_token_indexes.extend(task.decode_token_indexes[-task.move_kv_len :])

        move_token_indexes = torch.tensor(move_token_indexes, dtype=torch.int64, device="cuda")

        token_num = len(move_token_indexes)
        move_size = self.token_dim_size * token_num
        recive_buffer = self.kv_move_buffer.view(-1)[0:move_size].view(token_num, 2 * self.head_num, self.head_dim)
        for i, mem in enumerate(mem_managers):
            for layer_index in range(mem.layer_num):
                nccl_comm.recv(recive_buffer, src=0)
                mem._write_kv_move_data_p2p(move_token_indexes, recive_buffer, layer_index)
        return

    def _write_kv_move_data_p2p(self, token_indexes: torch.Tensor, buffer_tensor: torch.Tensor, layer_index):
        move_token_num = len(token_indexes)
        kv_trans(buffer_tensor, self.kv_move_buf_indexes[0:move_token_num], self.kv_buffer[layer_index], token_indexes)
        return

    def _free_buffers(self):
        self.kv_buffer = None

    def alloc(self, need_size) -> torch.Tensor:
        if need_size > self.mark_end - self.mark_start:
            logger.error(f"warn no enough cache need_size {need_size} left_size {self.can_use_mem_size}")
            assert False, "error alloc state"

        start = self.mark_start
        end = self.mark_start + need_size
        self.mark_start += need_size

        self.can_use_mem_size -= need_size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

        # 利用缓冲区返回，避免异步情况下的内存竞争
        if self._return_start + need_size > self._mem_state_return.shape[0]:
            self._return_start = 0
        ans = self._mem_state_return[self._return_start : self._return_start + need_size]
        ans.copy_(self.mem_state[start:end])
        self._return_start += need_size
        return ans

    def free(self, free_index: Union[torch.Tensor, List[int]]):
        """_summary_

        Args:
            free_index (torch.Tensor): _description_
        """

        end = self.mark_start
        start = self.mark_start - len(free_index)
        assert start >= 0, f"error free state start: {self.mark_start} free len {len(free_index)}"

        if isinstance(free_index, list):
            self.mem_state.numpy()[start:end] = free_index
        else:
            # 从 gpu 到 cpu 的拷贝操作是流内阻塞操作
            self.mem_state[start:end] = free_index

        self.mark_start -= len(free_index)

        self.can_use_mem_size += len(free_index)
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)

        if self.can_use_mem_size == len(self.mem_state):
            logger.debug(f"freed all gpu mem size {self.can_use_mem_size}")
        return

    def free_all(self):
        self.can_use_mem_size = len(self.mem_state)
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self.mem_state.numpy()[:] = list(range(0, len(self.mem_state)))
        self.mark_start = 0
        self.mark_end = len(self.mem_state)

    def resize_mem(self, new_size):
        """
        just for test code
        """
        size = new_size
        dtype = self.dtype
        head_num = self.head_num
        head_dim = self.head_dim
        layer_num = self.layer_num

        self.size = new_size
        self.mem_state = torch.arange(
            0, self.size, dtype=torch.int32, device="cpu", requires_grad=False, pin_memory=True
        )
        self.mark_start = 0
        self.mark_end = self.size
        self.can_use_mem_size = self.size
        self.shared_can_use_token_num.set_value(self.can_use_mem_size)
        self._free_buffers()
        self._init_buffers(size, dtype, head_num, head_dim, layer_num)
        return

    def get_index_kv_buffer(self, index):
        return {"kv_buffer": self.kv_buffer[:, index]}

    def load_index_kv_buffer(self, index, load_tensor_dict):
        self.kv_buffer[:, index].copy_(load_tensor_dict["kv_buffer"])

    def copy_kv_from_other_dp_ranks(
        self,
        mem_managers: List["MemoryManager"],
        move_token_indexes: torch.Tensor,
        token_dp_indexes: torch.Tensor,
        mem_indexes: torch.Tensor,
        dp_size_in_node: int,
        rank_in_dp: int,
    ):
        if not hasattr(self, "mem_ptrs_tensor"):
            # 构建一个2D tensor，shape为(layer_num, mem_num)
            mems_ptr_list = []
            for i in range(0, len(mem_managers)):
                mems_ptr_list.append(mem_managers[i].kv_buffer.data_ptr())
            self.mem_ptrs_tensor = torch.tensor(mems_ptr_list, dtype=torch.uint64, device="cpu", pin_memory=True)

        # 一次性传输所有层
        kv_trans_for_dp(
            input_mems=self.mem_ptrs_tensor.cuda(non_blocking=True),
            input_idx=move_token_indexes,
            input_dp_idx=token_dp_indexes,
            output=self.kv_buffer,
            output_idx=mem_indexes,
            dp_size_in_node=dp_size_in_node,
            rank_in_dp=rank_in_dp,
        )

    def write_to_shm(self, req_manager):
        """
        将 mem manager 写入到 shm中，方便pd分离等特性直接从中读取，不依赖进程间队列。
        """
        if kv_trans_use_p2p():
            from lightllm.server.router.model_infer.mode_backend.continues_batch.pd_mode.p2p_fix import reduce_tensor

            mp.reductions.reduce_tensor.__code__ = reduce_tensor.__code__

        from lightllm.common.req_manager import ReqManager

        req_manager: ReqManager = req_manager

        # 这个地方是一个不太优雅的设计，但是暂时这么做，可以让dp shared kv swap模块直接访问 req_manager 中的 req_to_token_indexs
        # 避免过多无用的数据复制和传输开销。
        self.req_to_token_indexs: torch.Tensor = req_manager.req_to_token_indexs

        lock = FileLock(f"/tmp/{get_unique_server_name()}_mem_manager_lock")
        with lock:
            node_world_size = get_node_world_size()
            shm_name = f"{get_unique_server_name()}_mem_manager_{get_current_rank_in_node()}"
            obj_bytes_array = [ForkingPickler.dumps(self).tobytes() for _ in range(node_world_size * 2)]
            obj_size = len(obj_bytes_array[0])
            shm = create_or_link_shm(
                name=shm_name, expected_size=obj_size * (node_world_size * 2) + 4 + 4, force_mode="create"
            )
            logger.info(f"create shm {shm.name} size {shm.size} for mem manger shared buffer")
            shm.buf[0:4] = (node_world_size * 2).to_bytes(4, "little")
            shm.buf[4:8] = obj_size.to_bytes(4, "little")
            start_index = 8
            for obj_bytes in obj_bytes_array:
                shm.buf[start_index : start_index + obj_size] = obj_bytes
                start_index += obj_size

    @staticmethod
    def loads_from_shm(rank_in_node: int) -> "MemoryManager":
        shm_name = f"{get_unique_server_name()}_mem_manager_{rank_in_node}"
        lock = FileLock(f"/tmp/{get_unique_server_name()}_mem_manager_lock")
        logger.info(f"get memmanager from shm {shm_name}")
        with lock:
            shm = create_or_link_shm(name=shm_name, expected_size=-1, force_mode="link")
            left_num = int.from_bytes(shm.buf[0:4], "little")
            obj_size = int.from_bytes(shm.buf[4:8], "little")
            assert left_num > 0
            end_index = 8 + left_num * obj_size
            start_index = 8 + (left_num - 1) * obj_size
            obj_bytes = shm.buf[start_index:end_index].tobytes()
            shm.buf[0:4] = (left_num - 1).to_bytes(4, byteorder="little")
            shm.close()
            return ForkingPickler.loads(obj_bytes)


class ReadOnlyStaticsMemoryManager:
    """
    读取一些统计信息
    """

    def __init__(self) -> None:
        args = get_env_start_args()
        self.global_world_size = args.tp
        self.node_world_size = args.tp // args.nnodes
        self.dp_world_size = self.global_world_size // args.dp
        # 兼容多机 dp size=1 纯 tp 模式的情况
        self.is_multinode_tp = args.dp == 1 and args.nnodes > 1
        self.shared_tp_infos = [
            SharedInt(f"{get_unique_server_name()}_mem_manger_can_use_token_num_{rank_in_node}")
            for rank_in_node in range(0, self.node_world_size, self.dp_world_size)
        ]

    def get_unrefed_token_num(self, dp_rank_in_node: int):
        if self.is_multinode_tp:
            return self.shared_tp_infos[0].get_value()
        return self.shared_tp_infos[dp_rank_in_node].get_value()
