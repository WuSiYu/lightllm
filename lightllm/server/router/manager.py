import copy
import itertools
import json
import time
import uuid
import torch
import uvloop
import asyncio
import rpyc

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from typing import Coroutine, Dict, List, Tuple, Union, Literal, Optional
from ..sampling_params import SamplingParams
from ..io_struct import Req, NormalReq, SplitFuseReq, Batch
from ..multimodal_params import MultimodalParams
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import build_req_queue
from rpyc.utils.classic import obtain
from lightllm.utils.infer_utils import calculate_time
from .dynamic_prompt.shared_arr import SharedInt
from .dynamic_prompt.radix_cache import RadixCacheReadOnlyClient
from ..io_struct import BatchTokenIdOut, AbortReq, ReqRunStatus, FinishStatus, ReqDetokenizationState
from .stats import Stats
from .pause_strategy import Fcfs, select_paused_reqs
from ..tokenizer import get_tokenizer
from lightllm.utils.log_utils import init_logger
from lightllm.server.router.token_load import TokenLoad
from lightllm.server.req_id_generator import convert_sub_id_to_group_id
from lightllm.server.metrics.manager import MetricClient

logger = init_logger(__name__)

def DeDupPrinter(log_func=print):
    last_print_ = f"DeDupPrinter({log_func=})"
    last_print_n_ = 1
    def print_(*args, sep=' ', **kargs):
        nonlocal last_print_, last_print_n_
        string = sep.join(map(str, args))
        if last_print_ != string:
            if last_print_n_ > 1:
                if last_print_n_ > 2:
                    print(f"({last_print_n_-2} same line omitted)")
                print(last_print_)
            print(string, **kargs)
            last_print_n_ = 1
            last_print_ = string
        else:
            last_print_n_ += 1
    return print_

print_ = DeDupPrinter()

cnt_ = itertools.count()


class ModelInstanceHandle():
    def __init__(self, inital_mode: Literal["prefill", "decode"], gpu_list: Tuple[int], kv_token_capacity: int):
        self.state = inital_mode
        self.gpu_list = gpu_list
        self.world_size = len(gpu_list)
        self.kv_token_capacity = kv_token_capacity
        self.dist = f"tp{self.world_size}"    # tp only for now
        self.model_rpcs: List[ModelRpcClient] = []
        self.batch: Optional[Batch] = None
        self._batch_lock: Union[asyncio.Lock, None] = None      # lazy init, lock must created in same event_loop when python <= 3.9
        self._batch_send_pending: List[Batch] = []  # for prefill inst only
        self._batch_sending: Optional[Batch] = None     # for prefill inst only
        self._batch_receving: Optional[Batch] = None    # for docode inst only
        self.current_task = {"compute": None, "io": None}

    def get_batch_lock(self):
        if self._batch_lock is None:
            self._batch_lock = asyncio.Lock()
        return self._batch_lock

    def get_used_kv_token(self):
        used_kv_token = 0
        if self.batch:
            used_kv_token += sum(req.get_used_tokens() for req in self.batch.reqs)
        if self._batch_send_pending:
            used_kv_token += sum(req.get_used_tokens() for b in self._batch_send_pending for req in b.reqs)
        if self._batch_sending:
            used_kv_token += sum(req.get_used_tokens() for req in self._batch_sending.reqs)
        if self._batch_receving:
            used_kv_token += sum(req.get_used_tokens() for req in self._batch_receving.reqs)
        return used_kv_token

    def __repr__(self) -> str:
        return f"MODEL_INST <{self.state}> @ GPU{self.gpu_list}, batch={self.batch.batch_id if self.batch is not None else 'N/A'}, current_task={[k for k, v in self.current_task.items() if v is not None]})"
        # return f"{self.__class__.__name__} @ {id(self)} ({self.state = }, {self.gpu_list = }, {self.dist}, batch={id(self.batch) if self.batch is not None else None}, current_task={[k for k, v in self.current_task.items() if v is not None]})"
        # return f"{self.__class__.__name__} @ {id(self)} ({self.state = }, {self.gpu_list = }, {self.dist}, {self.model_rpcs}, {self.batch}, {self.current_task = })"


# dist_config:  {"prefill": [(0, 1), (2, 3), (8, 9, 10, 11)], "decode": [(4, 5, 6, 7), (12, 13, 14, 15)]}
# globle rank:  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
# node rank:    0   0   0   0   0   0   0   0   1   1   1   1   1   1   1   1           (0 is master node)
# local rank:   0   1   0   1   0   1   2   3   0   1   2   3   0   1   2   3
# gpu_id:       0   1   2   3   4   5   6   7   0   1   2   3   4   5   6   7

class RouterManager:
    def __init__(self, args, router_port, detokenization_port, model_rpc_ports, metric_port):
        self.args = args
        self.node_rank: int = args.dist_node_rank
        self.node_count: int = args.dist_node_count
        self.master_addr: str = args.dist_master_addr
        self.gpu_per_node: int = args.dist_gpu_per_node
        self.model_instences: List[ModelInstanceHandle] = []

        _default = {"prefill": [(0, 1), (2, 3)], "decode": [(4, 5, 6, 7)]}

        self.local_dist_config = json.loads(args.dist_node_init_override) if args.dist_node_init_override else _default
        logger.info(f"init with {self.local_dist_config = }")
        _gpus = [g for type, gpus_list in self.local_dist_config.items() for gpus in gpus_list for g in gpus]
        assert all(gpu_id < self.gpu_per_node * self.node_count for gpu_id in _gpus)

        self.model_weightdir = args.model_dir
        self.load_way = args.load_way
        self.mode = args.mode
        self.max_total_token_num = args.max_total_token_num
        # 用共享内存进行共享，router 模块读取进行精确的调度估计
        self.shared_can_use_token_num = SharedInt(f"{args.nccl_port}_mem_manger_can_use_token_num")
        # 初始化 radix_cache_client 用于读取 prompt cache 的管理信息
        self.radix_cache_client = None
        if self.args.use_dynamic_prompt_cache:
            self.radix_cache_client = RadixCacheReadOnlyClient(str(args.nccl_port), self.max_total_token_num, tp_id=0)

        # 共享变量，用于存储router端调度分析得到的机器负载信息
        self.shared_token_load = TokenLoad(f"{str(args.nccl_port)}_shared_token_load")
        self.shared_token_load.set_current_load(0.0)
        self.shared_token_load.set_logical_max_load(0.0)
        self.shared_token_load.set_dynamic_max_load(0.0)

        self.pause_strategy = Fcfs()
        self.running_batch: Batch = None
        self.eos_id = args.eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = args.router_max_wait_tokens

        context = zmq.asyncio.Context(2)
        if self.node_rank == 0:
            self.recv_from_httpserver = context.socket(zmq.PULL)
            self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")

        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://{self.master_addr}:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.is_splitfuse_mode = args.splitfuse_mode
        self.splitfuse_block_size = args.splitfuse_block_size

        self.stats_tool = Stats(not args.disable_log_stats, args.log_stats_interval)
        self.metric_client = MetricClient(metric_port)
        return

    async def wait_to_model_ready(self):

        # 初始化模型
        # self.model_rpcs: List[ModelRpcClient] = []
        init_model_ret = []
        model_rpc_ports_iter = iter(self.model_rpc_ports)
        for instance_mode, instances_gpus_list in self.local_dist_config.items():
            print(f"--{instance_mode = }, {instances_gpus_list = }")

            for gpu_list in instances_gpus_list:    # global gpu index
                print(f"----{gpu_list = }")
                # per model instance

                model_instence = ModelInstanceHandle(
                    inital_mode=instance_mode, gpu_list=gpu_list, kv_token_capacity=self.max_total_token_num
                )

                for local_rank, global_rank in enumerate(gpu_list):
                    print(f"------{local_rank = }, {global_rank = }")
                    print()
                    model_rpc_client = await start_model_process(port=next(model_rpc_ports_iter))
                    model_instence.model_rpcs.append(model_rpc_client)

                    kvargs = {
                        "inital_mode": instance_mode,
                        "local_rank": local_rank,   # local tp rank in the model instance
                        "local_gpu_id": global_rank % self.gpu_per_node,
                        "local_world_size": len(gpu_list),
                        "gpu_per_node": self.gpu_per_node,
                        "node_rank": self.node_rank,
                        "node_count": self.node_count,
                        "global_rank": global_rank,
                        "global_world_size": self.gpu_per_node * self.node_count,
                        "master_addr": self.master_addr,
                        "weight_dir": self.model_weightdir,
                        "load_way": self.load_way,
                        "max_total_token_num": self.max_total_token_num,
                        "mode": self.mode,
                        "max_req_num": self.args.running_max_req_size + 8,
                        "max_seq_length": self.args.max_req_total_len + 8,  # 留一点余量
                        "nccl_port": self.args.nccl_port,
                        "is_splitfuse_mode": self.is_splitfuse_mode,
                        "splitfuse_block_size": self.splitfuse_block_size,
                        "is_token_healing": self.args.token_healing_mode,
                        "return_all_prompt_logprobs": self.args.return_all_prompt_logprobs,
                        "use_reward_model": self.args.use_reward_model,
                        "use_dynamic_prompt_cache": self.args.use_dynamic_prompt_cache,
                        "data_type": self.args.data_type,
                        "eos_id": self.eos_id,
                        "beam_mode": self.args.beam_mode,
                        "diverse_mode": self.args.diverse_mode,
                    }
                    print(f"init model {kvargs = }")
                    init_model_ret.append(model_rpc_client.init_model(kvargs))

                self.model_instences.append(model_instence)

        print("pre")
        print(init_model_ret)
        print(self.model_instences)
        await asyncio.gather(*init_model_ret)
        print("post")

        self.req_queue = build_req_queue(self.args, self)
        logger.info(f"use req queue {self.req_queue.__class__.__name__}")
        return

    def add_req(
        self,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        group_req_id: int,
        start_time: float,
    ):
        req_group = []
        for i in range(sampling_params.best_of):
            if self.is_splitfuse_mode:
                req = SplitFuseReq(
                    group_req_id + i,
                    copy.deepcopy(prompt_ids),
                    sampling_params,
                    multimodal_params,
                    self.splitfuse_block_size,
                )
            else:
                req = NormalReq(group_req_id + i, copy.deepcopy(prompt_ids), sampling_params, multimodal_params)
            req.start_time = start_time
            req_group.append(req)

        self.req_queue.extend(req_group)
        self.send_to_detokenization.send_pyobj(
            ReqDetokenizationState(
                group_req_id,
                prompt_ids,
                sampling_params.max_new_tokens,
                sampling_params.ignore_eos,
                sampling_params.skip_special_tokens,
                sampling_params.add_spaces_between_special_tokens,
                sampling_params.print_eos_token,
                sampling_params.best_of,
            )
        )
        return

    async def abort(self, group_req_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if convert_sub_id_to_group_id(req.request_id) == group_req_id:
                    req.finish_status = FinishStatus.FINISHED_ABORT
        for req in self.req_queue.waiting_req_list:
            if convert_sub_id_to_group_id(req.request_id) == group_req_id:
                req.finish_status = FinishStatus.FINISHED_ABORT
        return

    async def loop_for_fwd(
        self,
    ):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    token_ratio1 = self.get_used_tokens() / self.max_total_token_num
                    token_ratio2 = (
                        self.max_total_token_num - self.shared_can_use_token_num.get_value()
                    ) / self.max_total_token_num
                    logger.debug(
                        f"current batch size: {len(self.running_batch.reqs)} \n"
                        f"paused req num: {len(self.req_queue.pause_req_dict)} \n"
                        f"token used ratio: {token_ratio1} not contain prompt cache tree unrefed tokens\n"
                        f"token used ratio: {token_ratio2} contain prompt cache tree unrefed tokens"
                    )
                    self.shared_token_load.set_current_load(token_ratio1)
                    self.req_queue.update_token_load(self.running_batch)
                    pass
                self.stats_tool.print_stats()
                self.metric_client.gauge_set("lightllm_batch_current_size", len(self.running_batch.reqs))
                self.metric_client.gauge_set("lightllm_batch_pause_size", len(self.req_queue.pause_req_dict))
                self.metric_client.gauge_set("lightllm_queue_size", len(self.req_queue.waiting_req_list))
                self.metric_client.gauge_set(
                    "lightllm_batch_current_max_tokens",
                    int(self.shared_token_load.get_dynamic_max_load() * self.max_total_token_num),
                )
            else:
                self.shared_token_load.set_dynamic_max_load(0.0)
                self.shared_token_load.set_current_load(0.0)
                if counter_count % 300 == 0:
                    self.metric_client.gauge_set("lightllm_batch_current_size", 0.0)
                    self.metric_client.gauge_set("lightllm_batch_pause_size", 0.0)
                    self.metric_client.gauge_set("lightllm_queue_size", 0.0)
                    self.metric_client.gauge_set("lightllm_batch_current_max_tokens", 0.0)

            if self.running_batch is None:
                await asyncio.sleep(0.001)  # 1ms

    def _background_task(self, model_instance: ModelInstanceHandle, slot: Literal['compute', 'io'], task: Coroutine, task_name: str = ""):
        model_instance.current_task[slot] = task
        async def _task_wrapper():
            await task
            print_(f" ==> task {task_name} DONE @ {time.time()}, {slot = }, {model_instance} {task}")
            model_instance.current_task[slot] = None
        print_(f" ==> task {task_name} START @ {time.time()}, {slot = }, {model_instance} {task}")
        # asyncio.create_task(_task_wrapper(), name=task_name)
        asyncio.create_task(_task_wrapper(), name=task_name)

    async def _step(self):
        """
        事件处理循环
        """
        print_("----_step")
        step_cnt = next(cnt_)
        for model_instance in self.model_instences:
            # print("+---inst", model_instance)
            if model_instance.state == 'prefill':
                if model_instance.current_task["compute"] is None and self.req_queue.waiting_req_list:
                    print_(f"    +---prefill inst compute free", model_instance)
                    print_(f"    +---and has waiting {len(self.req_queue.waiting_req_list)} reqs")
                    async def _task(inst: ModelInstanceHandle):
                        task_name = asyncio.current_task().get_name()

                        # tmp req_queue logic
                        # MAX_REQ = 10
                        # for _ in range(MAX_REQ):
                        #     if not self.req_queue.waiting_req_list:
                        #         break

                        # reqs = self.req_queue.waiting_req_list[:MAX_REQ]
                        # self.req_queue.waiting_req_list = self.req_queue.waiting_req_list[len(reqs):]

                        reqs = await self.inst_fetch_prefill_reqs_naive(inst, self.req_queue.waiting_req_list)
                        if not reqs:
                            print_("    ==> _task({task_name}) WARNING: FAILED, prefill inst is too full for prefill")
                            return

                        new_batch = Batch(uuid.uuid4().hex, reqs)
                        self.metric_client.histogram_observe("lightllm_batch_next_size", len(new_batch.reqs))
                        for req in new_batch.reqs:
                            self.metric_client.histogram_observe(
                                "lightllm_request_queue_duration_bucket", time.time() - req.start_time
                            )
                        # self.stats_tool.count_prompt_tokens(new_batch)
                        assert inst.batch is None
                        inst.batch = new_batch
                        print_(f"    ==> _task({task_name}) prefill reqs req_id={[req.request_id for req in inst.batch.reqs]}")

                        await self._prefill_batch(inst, inst.batch)
                        if not inst.batch.is_clear():
                            inst._batch_send_pending.append(inst.batch)
                        inst.batch = None

                    task = _task(model_instance)
                    print_(f"    +---_background_task new prefill compute task: {model_instance} {task}")
                    self._background_task(model_instance=model_instance, slot='compute', task=task, task_name=f"prefill_{step_cnt}")

                if model_instance.current_task["io"] is None and model_instance._batch_send_pending:
                    print_(f"    +---prefill inst io free", model_instance)
                    print_(f"    +---{len(model_instance._batch_send_pending) = }")
                    async def _task(inst: ModelInstanceHandle):
                        task_name = asyncio.current_task().get_name()

                        assert inst._batch_sending is None
                        inst._batch_sending = inst._batch_send_pending.pop(0)

                        recv_inst = await self.get_recv_decode_inst(inst._batch_sending)
                        if recv_inst is None:
                            print_("    ==> _task({task_name}) WARNING: FAILED, no available decode inst to send, task exit")
                            inst._batch_send_pending.insert(0, inst._batch_sending)
                            inst._batch_sending = None
                            return

                        assert recv_inst._batch_receving is None
                        recv_inst._batch_receving = inst._batch_sending

                        print_(f"    ==> _task({task_name}) send kv reqs req_id={[req.request_id for req in inst._batch_sending.reqs]}")
                        await self._transfer_kv_batch(inst, inst._batch_sending, recv_inst)
                        await self._remove_batch(inst, inst._batch_sending)
                        if recv_inst.batch is None:
                            recv_inst.batch = inst._batch_sending
                        else:
                            print_(f"    ==> _task({task_name}) send kv merge reqs: {[req.request_id for req in recv_inst.batch.reqs]} with {[req.request_id for req in inst._batch_sending.reqs]}")
                            async with recv_inst.get_batch_lock():
                                recv_inst.batch.merge(inst._batch_sending)        # local batch merge (this line) must before remote call (line below)
                                await self._merge_batch(recv_inst, recv_inst.batch, inst._batch_sending)
                        inst._batch_sending = None
                        recv_inst._batch_receving = None

                    task = _task(model_instance)
                    print_(f"    +---_background_task new prefill io task: {model_instance} {task}")
                    self._background_task(model_instance=model_instance, slot='io', task=task, task_name=f"send_kv_{step_cnt}")


            elif model_instance.state == 'decode':
                if model_instance.current_task["compute"] is None and model_instance.batch:
                    print_(f"    +---decode inst compute free", model_instance)
                    async def _task(inst: ModelInstanceHandle):
                        task_name = asyncio.current_task().get_name()

                        # self.stats_tool.count_output_tokens(model_instance.batch)
                        # print_(f"    ==> _task({task_name}) decode bs={len(model_instance.batch.reqs)}")
                        print_(f"    ==> _task({task_name}) decode bs={len(inst.batch.reqs)} req_id={[req.request_id for req in inst.batch.reqs]}")
                        async with inst.get_batch_lock():
                            await self._decode_batch(inst, inst.batch)
                        if inst.batch.is_clear():
                            inst.batch = None

                    task = _task(model_instance)
                    print_(f"    +---_background_task new decode {model_instance} compute {task}")
                    self._background_task(model_instance=model_instance, slot='compute', task=task, task_name=f"decode_{step_cnt}")


    async def inst_fetch_prefill_reqs_naive(self, model_instance: ModelInstanceHandle, req_queue: List[Req]):
        MAX_BATCH_TOKEN = 8192
        reqs_to_prefill: List[Req] = []
        available_tokens = model_instance.kv_token_capacity - model_instance.get_used_kv_token()
        available_tokens = min(available_tokens, MAX_BATCH_TOKEN)
        batched_tokens = 0
        while req_queue:
            pending_req_tokens = req_queue[0].input_len
            if available_tokens >= batched_tokens + pending_req_tokens:
                reqs_to_prefill.append(req_queue.pop(0))
                batched_tokens += pending_req_tokens
            else:
                break
        return reqs_to_prefill



    async def get_recv_decode_inst(self, batch: Batch):
        #tmp
        MINIMAL_FREE_RATIO = 0.25
        batch_tokens = sum(req.get_used_tokens() for req in batch.reqs)
        best_inst, best_ratio = None, MINIMAL_FREE_RATIO
        for inst in self.model_instences:
            if inst.state == 'decode' and inst._batch_receving is None:
                minimal_needed_token = inst.get_used_kv_token() + batch_tokens
                free_ratio = 1 - minimal_needed_token / inst.kv_token_capacity
                if free_ratio > best_ratio:
                    best_inst, best_ratio = inst, free_ratio

        return best_inst

    async def _step_old(self):
        """
        事件处理循环
        """
        # 删除所有已经 finished 的 req
        # 当前无运行请求时
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is not None:
                self.metric_client.histogram_observe("lightllm_batch_next_size", len(new_batch.reqs))
                for req in new_batch.reqs:
                    self.metric_client.histogram_observe(
                        "lightllm_request_queue_duration_bucket", time.time() - req.start_time
                    )
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                await self._prefill_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        # 有运行请求，但是已经到了可以调度新的请求合并推理的时机
        if self.has_wait_tokens >= self.max_wait_tokens:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch)
            self.has_wait_tokens = 0
            if new_mini_batch is not None:
                self.stats_tool.count_prompt_tokens(new_mini_batch)
                await self._prefill_batch(new_mini_batch)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                return

        # 正常 decode 阶段， 如果可以直接decode就直接decode，否则通过暂停策略暂停一些请求
        # 释放一些管理的 token
        if self._can_decode(self.running_batch):
            self.stats_tool.count_output_tokens(self.running_batch)
            await self._decode_batch(self.running_batch)
            self._filter_runing_batch()
            self.has_wait_tokens += 1
            return
        else:
            # pause strategy
            paused_reqs = select_paused_reqs(
                self.running_batch, self.pause_strategy, self.req_queue, self.max_total_token_num
            )
            await self._pause_reqs(self.running_batch, paused_reqs)
            logger.debug(f"pasued req num: {len(self.req_queue.pause_req_dict)}")
            self.has_wait_tokens = 0
            return
        return

    async def _init_batch(self, instance: ModelInstanceHandle, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [instance.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(instance.world_size)]
        ans = await asyncio.gather(*rets)
        req_to_req_status = obtain(ans[0])
        # if self.world_size != 1:
        #     req_to_req_status = obtain(ans[0])
        # else:
        #     req_to_req_status = ans[0]

        self._update_init_status_to_batch(batch, req_to_req_status)
        for req in batch.reqs:
            prompt_cache_len = req.cur_kv_len
            prompt_cache_ratio = req.cur_kv_len / req.input_len
            self.metric_client.histogram_observe("lightllm_cache_length", prompt_cache_len)
            self.metric_client.histogram_observe("lightllm_cache_ratio", prompt_cache_ratio)
            logger.info(
                f"lightllm_req_id:{req.request_id} "
                f"prompt_cache_len:{prompt_cache_len} "
                f"prompt_cache_ratio:{prompt_cache_ratio} "
            )
        logger.debug(f"Init Batch: {batch.simple_log()} \n")
        return

    async def _prefill_batch(self, instance: ModelInstanceHandle, batch: Batch):
        start_time = time.time()
        self.metric_client.counter_inc("lightllm_batch_inference_count", "prefill")
        await self._init_batch(instance, batch)
        if not self.is_splitfuse_mode:
            # 在 非 splitfuse 模式下，才需要真的执行 prefill 的操作。
            prefill_len = batch.input_tokens()
            # print(f"{batch.reqs = }")
            # print(f"{batch.reqs[0].input_len = }")
            # print(f"{batch.reqs[0].prompt_ids = }")
            # torch.cuda.nvtx.range_push("PROFILE")
            torch.cuda.nvtx.range_push(f"prefill {prefill_len}")
            es, ee = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            es.record()
            pts = time.time()
            rets = [instance.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(instance.world_size)]
            ans = await asyncio.gather(*rets)
            ee.record()
            pte = time.time()
            torch.cuda.nvtx.range_pop()
            # torch.cuda.nvtx.range_pop()
            req_to_out_status = obtain(ans[0])
            # if self.world_size != 1:
            #     req_to_out_status = obtain(ans[0])
            # else:
            #     req_to_out_status = ans[0]

            self._update_out_status_to_batch(batch, req_to_out_status)
            unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status()
            self._send_to_detokenization_proc(batch, req_to_out_status)
            batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
            await self._handle_finish_req(instance, batch, unfinished_req_ids, finished_req_ids)
            torch.cuda.synchronize()
            print(f"prefill prof:\t{prefill_len}\t{es.elapsed_time(ee)}")
            print(f"(python time {(pte - pts) * 1000})")
        self.metric_client.histogram_observe(
            "lightllm_batch_inference_duration_bucket", time.time() - start_time, "prefill"
        )
        return

    async def _transfer_kv_batch(self, sender: ModelInstanceHandle, batch: Batch, receiver: ModelInstanceHandle):
        for req in batch.reqs:
            req.req_status = ReqRunStatus.DIST_KV_SENDING
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        if sender.world_size == receiver.world_size:
            await self._init_batch(receiver, batch)
            rets1 = [sender.model_rpcs[tp_rank].send_kv_batch(batch.batch_id, reqs, receiver.gpu_list[tp_rank]) for tp_rank in range(sender.world_size)]
            rets2 = [receiver.model_rpcs[tp_rank].recv_kv_batch(batch.batch_id, reqs, sender.gpu_list[tp_rank]) for tp_rank in range(receiver.world_size)]
            ans = await asyncio.gather(*rets1, *rets2)
        else:
            raise NotImplementedError()
        return

    async def _decode_batch(self, instance: ModelInstanceHandle, batch: Batch):
        start_time = time.time()
        self.metric_client.counter_inc("lightllm_batch_inference_count", "decode")
        # async with instance.get_batch_lock():
        rets = [instance.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(instance.world_size)]
        ans = await asyncio.gather(*rets)
        req_to_out_status = obtain(ans[0])
        # if self.world_size != 1:
        #     req_to_out_status = obtain(ans[0])
        # else:
        #     req_to_out_status = ans[0]

        self._update_out_status_to_batch(batch, req_to_out_status)
        unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status()
        self._send_to_detokenization_proc(batch, req_to_out_status)
        batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)

        await self._handle_finish_req(instance, batch, unfinished_req_ids, finished_req_ids)
        self.metric_client.histogram_observe(
            "lightllm_batch_inference_duration_bucket", time.time() - start_time, "decode"
        )
        return

    async def _filter_batch(self, instance: ModelInstanceHandle, batch: Batch, unfinished_req_ids, finished_req_ids: List):
        print_(f"{asyncio.current_task().get_name()} {finished_req_ids=}")
        rets = [
            instance.model_rpcs[tp_rank].filter_batch(batch.batch_id, unfinished_req_ids, finished_req_ids)
            for tp_rank in range(instance.world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, instance: ModelInstanceHandle, batch1, batch2):
        rets = [
            instance.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(instance.world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, instance: ModelInstanceHandle, batch):
        rets = [instance.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(instance.world_size)]
        await asyncio.gather(*rets)
        return

    async def _pause_reqs(self, batch: Batch, pasue_reqs):
        pasue_reqs_info = [(r.request_id, r.req_status) for r in pasue_reqs]
        rets = [
            self.model_rpcs[tp_rank].pause_reqs(batch.batch_id, pasue_reqs_info) for tp_rank in range(self.world_size)
        ]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, instance: ModelInstanceHandle, batch: Batch, unfinished_req_ids, finished_req_ids):
        if len(finished_req_ids) != 0:
            if batch.is_clear():
                await self._remove_batch(instance, batch)
            else:
                await self._filter_batch(instance, batch, unfinished_req_ids, finished_req_ids)
        return

    def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None
            return

    def _update_init_status_to_batch(self, batch: Batch, req_to_req_status):
        self._update_out_status_to_batch(batch, req_to_req_status)
        return

    def _update_out_status_to_batch(self, batch: Batch, req_to_out_status):
        new_batch_decode_need_tokens = 0  # 只有在 splitfuse 模式下有意义
        for req_id, (
            req_status,
            cur_kv_len,
            cur_output_len,
            token_info_list,
            finish_status_value,
            extral_info,
        ) in req_to_out_status.items():
            req: Req = batch.id_to_reqs[req_id]
            # try:
            #     req: Req = batch.id_to_reqs[req_id]
            # except KeyError:
            #     print_(f"RMUBKE: task {asyncio.current_task().get_name()} req {req_id=} not found, this should not happend, {req_to_out_status=}")
            req.req_status = req_status
            req.cur_kv_len = cur_kv_len
            req.cur_output_len = cur_output_len
            # 暂时不维护 output_ids 和 output_metadata_list
            for (new_token_id, new_gen_metadata) in token_info_list:
                req.output_ids.append(new_token_id)
            #     req.output_metadata_list.append(new_gen_metadata)
            # 当没有被 aborted 的时候，才更新请求状态。
            if not req.finish_status.is_aborted():
                req.finish_status = FinishStatus(finish_status_value)
            new_batch_decode_need_tokens += req.get_decode_need_tokens()

        batch.batch_decode_need_tokens = new_batch_decode_need_tokens
        return

    def _can_decode(self, batch: Batch):
        return batch.batch_decode_need_tokens + self.get_used_tokens() <= self.max_total_token_num

    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        for req_id, (_, _, _, token_info_list, _, _) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            for idx, (new_token_id, new_gen_metadata) in enumerate(token_info_list):
                # req.finish_status 传输 value值 不传送对象，可以减少序列化对象的大小。
                if idx == len(token_info_list) - 1:
                    batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.finish_status.value))
                else:
                    batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, FinishStatus.NO_FINISH))

        self.send_to_detokenization.send_pyobj(batch_out)
        return

    def get_used_tokens(self):
        if self.args.use_dynamic_prompt_cache:
            return (
                self.max_total_token_num
                - self.shared_can_use_token_num.get_value()
                - self.radix_cache_client.get_unrefed_tokens_num()
            )
        else:
            return self.max_total_token_num - self.shared_can_use_token_num.get_value()

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 5:
                prompt_ids, sampling_params, multimodal_params, group_req_id, start_time = recv_req
                self.add_req(prompt_ids, sampling_params, multimodal_params, group_req_id, start_time)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                group_req_id = abort_req.group_req_id
                await self.abort(group_req_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return


def start_router_process(args, router_port, detokenization_port, model_rpc_ports, metric_port, pipe_writer):
    # 注册graceful 退出的处理
    from lightllm.utils.graceful_utils import graceful_registry
    import inspect

    graceful_registry(inspect.currentframe().f_code.co_name)

    try:
        router = RouterManager(
            args,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports,
            metric_port=metric_port,
        )

        asyncio.run(router.wait_to_model_ready())
    except:
        import traceback
        import sys

        etype, evalue, tb = sys.exc_info()
        err_str = "\n".join(traceback.format_exception(etype, evalue, tb))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send("init ok")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
