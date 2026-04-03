import torch
import time
import sys
import inspect
import threading
import setproctitle
import torch.multiprocessing as mp
from torch.distributed import TCPStore
from datetime import timedelta
from typing import List, Dict, Union
from lightllm.utils.log_utils import init_logger
from lightllm.common.kv_cache_mem_manager import MemoryManager
from lightllm.server.pd_io_struct import KVMoveTask, PDTransJoinInfo, PDTransLeaveInfo, KVMoveTaskGroup
from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.distributed.pynccl import StatelessP2PProcessGroup, PyNcclCommunicator
from lightllm.utils.envs_utils import get_unique_server_name


logger = init_logger(__name__)


def _can_use_p2p_for_tasks(move_tasks: List[KVMoveTask], mem_managers: List[MemoryManager]) -> bool:
    if not kv_trans_use_p2p() or len(move_tasks) == 0:
        return False
    src_tp = move_tasks[0].prefill_tp_in_node if move_tasks[0].prefill_tp_in_node is not None else len(mem_managers)
    dst_tp = move_tasks[0].decode_tp_in_node if move_tasks[0].decode_tp_in_node is not None else len(mem_managers)
    return src_tp == dst_tp == len(mem_managers)


def _handle_kvmove_task(
    move_tasks: List[KVMoveTask],
    task_out_queue: mp.Queue,
    mem_managers: List[MemoryManager],
    connect_id_to_comm: Dict[str, PyNcclCommunicator],
    connect_id: str,
    dp_size_in_node: int,
):
    total_move_kv_len = sum([task.move_kv_len for task in move_tasks])
    try:
        device_index = connect_id_to_comm[connect_id].device.index
        start = time.time()
        if total_move_kv_len != 0:
            logger.info(f"trans start: {move_tasks[0].to_prefill_log_info()}")
            cur_mem = mem_managers[device_index]
            if _can_use_p2p_for_tasks(move_tasks, mem_managers):
                cur_mem.send_to_decode_node_p2p(
                    move_tasks, mem_managers, dp_size_in_node, connect_id_to_comm[connect_id]
                )
            else:
                cur_mem.send_to_decode_node(move_tasks, mem_managers, dp_size_in_node, connect_id_to_comm[connect_id])
            logger.info(f"trans finished: {move_tasks[0].to_prefill_log_info()} move len: {total_move_kv_len}")
        torch.cuda.synchronize()
        path_tag = "p2p" if _can_use_p2p_for_tasks(move_tasks, mem_managers) else (
            "sym" if move_tasks[0].prefill_tp_in_node == move_tasks[0].decode_tp_in_node else "asym"
        )
        cost_s = time.time() - start
        tps = (total_move_kv_len / cost_s) if cost_s > 0 else 0.0
        logger.info(
            f"trans cost time: {(time.time() - start)},"
            f"move_total_kv_len: {total_move_kv_len}, {move_tasks[0].to_prefill_log_info()}"
        )
        logger.info(
            f"pd_trans_perf role=prefill path={path_tag} "
            f"prefill_tp_in_node={move_tasks[0].prefill_tp_in_node} decode_tp_in_node={move_tasks[0].decode_tp_in_node} "
            f"move_total_kv_len={total_move_kv_len} cost_s={cost_s:.6f} kv_tokens_per_s={tps:.2f} "
            f"ms_per_kv_token={(cost_s * 1000.0 / max(total_move_kv_len, 1)):.6f}"
        )
        task_out_queue.put("ok")
    except BaseException as e:
        logger.exception(str(e))
        task_out_queue.put("fail")


def _handle_decode_join(
    node_info: PDTransJoinInfo,
    task_out_queue: mp.Queue,
    connect_id_to_comm: Dict[str, PyNcclCommunicator],
    store: TCPStore,
):
    try:
        logger.info(f"connect start {node_info}")
        src_id = node_info.prefill_id
        dest_id = node_info.connect_id
        logger.info(f"connect src_id {src_id} dest_id {dest_id}")
        result_list = []

        def async_connect():
            torch.cuda.set_device(node_info.prefill_device_id)
            group = StatelessP2PProcessGroup.create(src_id=src_id, dest_id=dest_id, is_server=True, store=store)
            comm = PyNcclCommunicator(group, node_info.prefill_device_id)
            result_list.append(comm)
            return

        connect_task = threading.Thread(target=async_connect, daemon=True)
        connect_task.start()
        connect_task.join(timeout=36)
        if connect_task.is_alive():
            raise Exception(f"{node_info} connect time out")

        connect_id_to_comm[node_info.connect_id] = result_list[0]
        logger.info(f"{node_info} kv trans connected!")
        task_out_queue.put("nccl_ok")
    except Exception as e:
        task_out_queue.put("nccl_fail")
        logger.warning(f"error while connect to decode node: {e} node_info {node_info}")


def _init_env(
    args,
    store_ip,
    port_min,
    port_max,
    device_id,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
):
    import os
    from lightllm.utils.net_utils import find_available_port

    # os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_MAX_NCHANNELS"] = "2"
    os.environ["NCCL_NSOCKS_PER_CHANNEL"] = "1"
    os.environ["NCCL_SOCKET_NTHREADS"] = "1"
    torch.backends.cudnn.enabled = False

    dp_size_in_node = max(1, args.dp // args.nnodes)
    setproctitle.setproctitle(
        f"lightllm::{get_unique_server_name()}::prefill_trans:Device{device_id}_DpSizeInNode{dp_size_in_node}"
    )

    try:
        torch.cuda.set_device(device_id)
        graceful_registry(inspect.currentframe().f_code.co_name)
        # Retry to handle TOCTOU race: find_available_port may return a port that gets
        # taken by another process (e.g. another shared_weight instance) before TCPStore binds.
        master_store = None
        store_port = None
        _last_tried_port = port_min - 1
        for _attempt in range(10):
            store_port = find_available_port(max(port_min, _last_tried_port + 1), port_max)
            if store_port is None:
                # Wrap around and retry from the beginning
                store_port = find_available_port(port_min, port_max)
            if store_port is None:
                raise RuntimeError(f"No available port found in range [{port_min}, {port_max}]")
            _last_tried_port = store_port
            try:
                master_store = TCPStore(
                    host_name=store_ip, port=store_port, is_master=True, use_libuv=True, timeout=timedelta(seconds=30)
                )
                break
            except RuntimeError:
                logger.warning(f"Port {store_port} bind failed (race), retrying...")
                continue
        if master_store is None:
            raise RuntimeError(f"Failed to bind TCPStore after retries in port range [{port_min}, {port_max}]")
        task_out_queue.put(("proc_start", store_port))

        # 从共享内存读取所有rank的mem_manager
        node_world_size = args.tp // args.nnodes
        mem_managers: List[MemoryManager] = [
            MemoryManager.loads_from_shm(rank_in_node=rank) for rank in range(node_world_size)
        ]
        task_out_queue.put("get_mem_managers_ok")
        connect_id_to_comm: Dict[str, PyNcclCommunicator] = {}

        while True:
            task: Union[KVMoveTaskGroup, PDTransJoinInfo, PDTransLeaveInfo] = task_in_queue.get()
            if isinstance(task, KVMoveTaskGroup):
                _handle_kvmove_task(
                    task.tasks, task_out_queue, mem_managers, connect_id_to_comm, task.connect_id, dp_size_in_node
                )
            elif isinstance(task, PDTransJoinInfo):
                _handle_decode_join(task, task_out_queue, connect_id_to_comm, master_store)
            elif isinstance(task, PDTransLeaveInfo):
                if task.connect_id in connect_id_to_comm:
                    connect_id_to_comm[task.connect_id].destroy()
                    connect_id_to_comm.pop(task.connect_id, None)
                    logger.info(f"destory {task} nccl communicator.")
                else:
                    logger.error(f"connect id {task.connect_id} dont exist in connect_id_to_comm")
            else:
                logger.warning(f"unexpected task type: {task}")

    except Exception as e:
        logger.error(f"Fatal error happened in kv trans process: {e}")
        pass


def start_prefill_trans_process(
    args,
    store_ip,
    port_min,
    port_max,
    device_id,
    task_in_queue: mp.Queue,
    task_out_queue: mp.Queue,
):
    proc = mp.Process(
        target=_init_env, args=(args, store_ip, port_min, port_max, device_id, task_in_queue, task_out_queue)
    )
    proc.start()
    assert proc.is_alive()
    logger.info(f"prefill trans kv process for device: {device_id} started!")
    return proc
