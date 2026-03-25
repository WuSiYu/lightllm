import asyncio
import time
import rpyc
import sys
import os
import gc
import signal
import copy
import psutil
import threading
import inspect
import collections
import setproctitle
from typing import List, Dict, Union
import zlib
from lightllm.utils.log_utils import init_logger
from .prefill_infer_rpyc import PDPrefillInferRpcServer
import torch.multiprocessing as mp
from lightllm.server.pd_io_struct import KVMoveTask
from lightllm.utils.retry_utils import retry
from rpyc import AsyncResult
from lightllm.utils.net_utils import get_hostname_ip
from ..task_queue import TaskQueue
from lightllm.utils.graceful_utils import graceful_registry
from lightllm.utils.envs_utils import get_unique_server_name

KV_MOVE_MAX_NUM = 16

logger = init_logger(__name__)


class PrefillKVMoveManager:
    def __init__(self, args, info_queue: mp.Queue):
        self.args = args
        # args.dp // args.nnodes 在跨机tp的场景下，可能为0
        self.dp_size_in_node = max(1, args.dp // args.nnodes)
        self.node_world_size = args.tp // args.nnodes
        self.dp_world_size = args.tp // args.dp
        # 不支持跨机tp的pd 分离策略
        assert self.dp_world_size <= self.node_world_size

        self.info_queue = info_queue
        self.infer_rpyc_objs: List[PDPrefillInferRpcServer] = []

        from .prefill_trans_obj import KVTransConnectObj

        self.connect_id_to_trans_obj: Dict[str, KVTransConnectObj] = {}
        # route_key -> {slot_id: connect_id}
        self.route_slot_to_connect_id: Dict[tuple, Dict[int, str]] = {}
        self.decode_node_rr_cursor: Dict[tuple, int] = {}

        for port in self.args.pd_node_infer_rpyc_ports:
            socket_path = f"/tmp/{get_unique_server_name()}_prefill_node_infer_rpyc_{port}"
            from rpyc.utils.factory import unix_connect

            con = retry(max_attempts=20, wait_time=2)(unix_connect)(socket_path, config={"allow_pickle": True})
            self.infer_rpyc_objs.append(con.root)
            logger.info(f"rpyc connect to infer rpyc port: {port} ok")
        self.host_ip = get_hostname_ip()
        if self.host_ip is None:
            self.host_ip = args.host

        self.infer_rpyc_lock = threading.Lock()

        self.kv_trans_lock = threading.Lock()
        # 释放token的task队列
        self.release_task_queue = TaskQueue(lambda datas: datas[0:KV_MOVE_MAX_NUM], fail_func=None)
        self.release_tasks_thread = threading.Thread(target=self.handle_release_task_loop, daemon=True)
        self.release_tasks_thread.start()

        from .prefill_trans_obj import KVTransProcess

        self.kv_trans_processes: List[KVTransProcess] = [None] * self.node_world_size
        for device_id in range(self.node_world_size):
            self.kv_trans_processes[device_id] = KVTransProcess()
            assert self.kv_trans_processes[device_id].init_all(device_id, self)

        return

    # ==================================================================================
    # 主任务循环，接收需要进行kv传输的请求进行处理
    # ==================================================================================

    def task_dispatcher_loop(self):
        try:
            # 获取任务，并分发给相关卡的处理队列
            while True:
                move_task: KVMoveTask = self.info_queue.get()
                try:
                    trans_obj = self.__get_trans_obj(move_task)
                    trans_obj.request_kv_trans_task_queue.put(move_task)
                except BaseException as e:
                    logger.exception(str(e))
                    self.put_to_release_task_queue(move_task)
                finally:
                    trans_obj = None

        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))
            raise e

    # ==================================================================================
    # 请求出错或者完成kv传输后的处理队列和线程loop
    # ==================================================================================

    def put_to_release_task_queue(self, task: Union[KVMoveTask, List[KVMoveTask]]):
        if isinstance(task, KVMoveTask):
            self.release_task_queue.put(task)
        elif isinstance(task, list):
            self.release_task_queue.put_list(task)
        else:
            logger.error("error input in put_to_release_task_queue func")
        return

    def handle_release_task_loop(self):
        while True:
            handle_list: List[KVMoveTask] = self.release_task_queue.get_tasks(log_tag="release_task_queue")
            if len(handle_list) == 0:
                time.sleep(0.01)
            else:
                self._remove_req_refs_from_prompt_cache(handle_list)
        return

    # ==================================================================================
    # 定时检测传输进程的健康状态，出现问题拉崩整个系统触发重启
    # ==================================================================================

    def check_trans_process_loop(self):
        try:
            while True:
                for device_id in range(self.node_world_size):
                    if not self.kv_trans_processes[device_id].is_trans_process_health():
                        raise Exception(f"device_id {device_id} kv process is unhealth")

                time.sleep(10.0)
        except (BaseException, RuntimeError) as e:
            logger.exception(str(e))

            for device_id in range(self.node_world_size):
                self.kv_trans_processes[device_id].killself()

            # 杀掉当前进程的父进程（router), 触发全局崩溃
            os.kill(os.getppid(), signal.SIGKILL)
            os.kill(os.getpid(), signal.SIGKILL)
            raise e

    # ==================================================================================
    # 与推理进程交互接口,  _remove_req_refs_from_prompt_cache
    # ==================================================================================

    def _remove_req_refs_from_prompt_cache(self, tasks: List[KVMoveTask]):
        with self.infer_rpyc_lock:
            dp_to_tasks = collections.defaultdict(list)
            for task in tasks:
                dp_to_tasks[task.prefill_dp_index].append(task)
            futures: List[AsyncResult] = []
            for prefill_dp_index, _tasks in dp_to_tasks.items():
                conn_start = prefill_dp_index * self.dp_world_size
                conn_end = (prefill_dp_index + 1) * self.dp_world_size
                conns = self.infer_rpyc_objs[conn_start:conn_end]
                for conn in conns:
                    futures.append(
                        rpyc.async_(conn.remove_req_refs_from_prompt_cache)([task.group_request_id for task in _tasks])
                    )
            asyncio.run(self.wait_all_future_finish(futures))
        return

    async def wait_all_future_finish(self, futures: List[AsyncResult]):
        await asyncio.gather(*[asyncio.to_thread(future.wait) for future in futures])
        return

    # ==================================================================================
    # 辅助功能接口
    # ==================================================================================

    def get_next_device_index(self):
        counts = [0 for _ in range(self.node_world_size)]
        for obj in self.connect_id_to_trans_obj.values():
            counts[obj.device_index] += 1
        min_count = min(counts)
        candidate_indexes = [i for i, c in enumerate(counts) if c == min_count]
        # Break ties by rotating from a moving cursor to avoid long-term bias to low device ids.
        start = self.decode_node_rr_cursor.get((-1, -1), 0) % self.node_world_size
        device_index = candidate_indexes[0]
        for offset in range(self.node_world_size):
            idx = (start + offset) % self.node_world_size
            if idx in candidate_indexes:
                device_index = idx
                break
        self.decode_node_rr_cursor[(-1, -1)] = (device_index + 1) % self.node_world_size
        return device_index

    def remove_trans_obj(self, connect_id):
        if connect_id in self.connect_id_to_trans_obj:
            trans_obj = self.connect_id_to_trans_obj.pop(connect_id, None)
            if trans_obj is not None:
                trans_obj.set_has_error()
                logger.error(f"remove tran obj decode_node_id {trans_obj.decode_node_id}")
        return

    def __get_trans_obj(self, task: KVMoveTask):
        self.__remove_dead_trans_obj()
        route_key = (task.decode_node.node_id, int(task.decode_tp_in_node))
        route_slot_num = max(1, min(self.node_world_size, int(task.decode_tp_in_node)))
        req_key = task.route_key if getattr(task, "route_key", 0) != 0 else task.group_request_id
        if isinstance(req_key, int):
            slot_id = req_key % route_slot_num
        else:
            slot_id = zlib.adler32(str(req_key).encode("utf-8")) % route_slot_num

        if route_key not in self.route_slot_to_connect_id:
            self.route_slot_to_connect_id[route_key] = {}

        slot_connect_ids = self.route_slot_to_connect_id[route_key]
        connect_id = slot_connect_ids.get(slot_id, None)
        if connect_id is not None and connect_id in self.connect_id_to_trans_obj:
            return self.connect_id_to_trans_obj[connect_id]

        gc.collect()
        from .prefill_trans_obj import KVTransConnectObj

        trans_obj = KVTransConnectObj()
        trans_obj.create(
            task.decode_node.node_id,
            task.decode_node.ip,
            task.decode_node.rpyc_port,
            task.decode_tp_in_node,
            self,
            preferred_device_index=(slot_id % self.node_world_size),
        )
        self.connect_id_to_trans_obj[trans_obj.connect_id] = trans_obj
        slot_connect_ids[slot_id] = trans_obj.connect_id
        return trans_obj

    def __remove_dead_trans_obj(self):
        del_connect_ids = []
        for connect_id, t_obj in self.connect_id_to_trans_obj.items():
            if t_obj.has_error_status():
                del_connect_ids.append(connect_id)

        for connect_id in del_connect_ids:
            self.connect_id_to_trans_obj.pop(connect_id, None)

        if del_connect_ids:
            del_connect_ids_set = set(del_connect_ids)
            for route_key in list(self.route_slot_to_connect_id.keys()):
                slot_map = self.route_slot_to_connect_id[route_key]
                for slot_id in list(slot_map.keys()):
                    if slot_map[slot_id] in del_connect_ids_set:
                        slot_map.pop(slot_id, None)
                if len(slot_map) == 0:
                    self.route_slot_to_connect_id.pop(route_key, None)

        if del_connect_ids:
            gc.collect()
        return


def _init_env(args, info_queue: mp.Queue, event):
    import lightllm.utils.rpyc_fix_utils as _

    # 注册graceful 退出的处理
    graceful_registry(inspect.currentframe().f_code.co_name)
    setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::prefill_kv_move_manager")

    manager = PrefillKVMoveManager(args, info_queue)
    kv_trans_process_check = threading.Thread(target=manager.check_trans_process_loop, daemon=True)
    kv_trans_process_check.start()
    event.set()
    # 进入主循环
    manager.task_dispatcher_loop()
    return


def start_prefill_kv_move_manager_process(args, info_queue: mp.Queue):
    event = mp.Event()
    proc = mp.Process(target=_init_env, args=(args, info_queue, event))
    proc.start()
    event.wait()
    assert proc.is_alive()
    logger.info("prefill kv move manager process started")
    return
