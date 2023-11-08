import time
import uvloop
import asyncio

from lightllm.server.router.ending_predict import EndingPredictState
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from typing import Dict, List, Optional
from ..sampling_params import SamplingParams
from ..io_struct import Req, Batch, RunStatus
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import AdaptiveBatchsizeRouter, ReqQueue
from .selection import SelectionManager
from .strategy import SJF_ABSR
from rpyc.utils.classic import obtain
from lightllm.utils.infer_utils import calculate_time
from ..io_struct import BatchTokenIdOut, AbortReq, InferState
from .stats import Stats

class RouterManager:

    def __init__(self, weightdir, load_way, world_size, max_total_token_num, batch_max_tokens, running_max_req_size, eos_id, token_ratio, max_new_token_len,
                 reserve_token_num, offload, strategy, router_port, detokenization_port, model_rpc_ports, mode=[], log_stats=True, log_stats_interval=10, adaptive_batchsize_router: bool = False):
        self.model_weightdir = weightdir
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.max_total_token_num = max_total_token_num
        self.router_max_total_token_num = self.max_total_token_num - reserve_token_num
        self.token_traio = token_ratio
        if adaptive_batchsize_router:
            offload = False             # adaptive_batchsize_router use it's own recompute (aka evict, offload) implementation
            self._last_staged_bs = 0
            self._last_running_bs = 0
            self.state = InferState.IDLE
            self.batch_to_prefill: Batch = None
            self.adaptive_batchsize_router = AdaptiveBatchsizeRouter(max_total_token_num)
            self.ending_predict_state = EndingPredictState()
            self.wait_strategy = SJF_ABSR(self.max_total_token_num, absr=self.adaptive_batchsize_router)
            self._last_evicted_n = 0
            self._last_evicted_size = 0
        else:
            self.adaptive_batchsize_router = None
            self.ending_predict_state = None
        self.req_queue = ReqQueue(self.router_max_total_token_num, batch_max_tokens, running_max_req_size, max_new_token_len, token_ratio, self.adaptive_batchsize_router, self.ending_predict_state)
        self.selector = SelectionManager.getSelection(strategy, self.req_queue, self.router_max_total_token_num, offload=offload)
        self.running_batch: Batch = None
        self.eos_id = eos_id
        self.offload = offload
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10
        # if self.adaptive_batchsize_router and self.adaptive_batchsize_router.RANDOM_SLOW_ADD_NEW_REQ:
        #     self.max_wait_tokens = 1

        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")

        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.stats_tool = Stats(log_stats, log_stats_interval)

    async def wait_to_model_ready(self):
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
            init_model_ret.append(
                self.model_rpcs[rank_id].init_model(
                    rank_id,
                    self.world_size,
                    self.model_weightdir,
                    self.max_total_token_num,
                    self.load_way,
                    self.eos_id,
                    self.mode))

        await asyncio.gather(*init_model_ret)
        return

    def add_req(
        self,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str
    ):
        req = Req(request_id, prompt_ids, sampling_params, self.ending_predict_state)
        self.req_queue.append(req)
        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
        return

    async def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.has_generate_finished = True
                req.aborted = True
        return

    async def loop_for_fwd(self,):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    print("current batch size:", len(self.running_batch.reqs), "token used ratio:", (self.req_queue.calcu_stopd_tokens() / self.router_max_total_token_num,
                          (self.running_batch.calcu_used_tokens() + self.req_queue.calcu_stopd_tokens()) / self.router_max_total_token_num))
                    pass
                self.stats_tool.print_stats()

            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    def _update_adaptive_batchsize_router(self, **kargs):
        print(time.time())
        staged_bs = (len(self.running_batch.reqs) if self.running_batch else 0) + len(self.wait_strategy.stopped_req_list)
        staged_bs_delta = staged_bs - self._last_staged_bs
        self._last_staged_bs = staged_bs
        self.adaptive_batchsize_router.update(
            state=self.state,
            staged_bs=staged_bs,
            running_bs=self._last_running_bs,
            current_batch=self.running_batch,
            stopped_reqs=self.wait_strategy.stopped_req_list,
            reqs_pending=len(self.req_queue.waiting_req_list),
            staged_bs_delta=staged_bs_delta,
            evicted_n=self._last_evicted_n,
            evicted_size=self._last_evicted_size,
            **kargs,
        )
        # reset step-wise counters
        self._last_evicted_n = 0
        self._last_evicted_size = 0
        print(time.time())

    def _repack_recompute_req(self, req: Req):
        print("_repack_recompute_req before", req.request_id, req.input_len, len(req.output_ids), req.max_output_len)
        assert req.input_len == len(req.prompt_ids)
        r = Req(req.request_id, req.prompt_ids + req.output_ids, req.sample_params)
        r.ending_predict = req.ending_predict
        r.output_metadata_list = req.output_metadata_list
        r.max_output_len = req.max_output_len - len(req.output_ids)
        r._evicted_at = len(req.output_ids)
        print("_repack_recompute_req after", r.request_id, r.input_len, len(r.output_ids), r.max_output_len)
        return r

    def _restore_recompute_req(self, req: Req):
        # to keep the same state before and after the recompute
        req.max_output_len += req._evicted_at
        req.input_len -= req._evicted_at
        req.output_ids = req.prompt_ids[-req._evicted_at:] + req.output_ids
        req.prompt_ids = req.prompt_ids[:-req._evicted_at]
        del req._evicted_at


    async def _evict_reqs(self, space_needs: int):
        used_tokens = self.adaptive_batchsize_router._total_staged_tokens(self.running_batch, self.wait_strategy.stopped_req_list)
        # print(f"_evict_reqs self.running_batch len {len(self.running_batch.reqs)}, self.wait_strategy.stopped_req_list len {len(self.wait_strategy.stopped_req_list)}")
        # print(f"batch.calcu_used_tokens() + self.calcu_stopd_tokens() {self.running_batch.calcu_used_tokens()} + {self.wait_strategy.calcu_stopd_tokens()}")
        # print(f"batch.calcu_used_tokens() + self.calcu_stopd_tokens() {sum(r.calcu_used_tokens() for r in self.running_batch.reqs)} + {sum(r.calcu_used_tokens() for r in self.wait_strategy.stopped_req_list)}")
        # print(f"_evict_reqs {space_needs} - ({self.max_total_token_num} - {used_tokens})")
        to_evict = max(0, space_needs - (self.max_total_token_num - used_tokens))
        if to_evict > 0:
            # no enough VRAM
            evict_reqs: List[Req] = self.wait_strategy.select_evict_reqs(self.running_batch, to_evict)
            new_reqs = [self._repack_recompute_req(r) for r in evict_reqs]
            self.req_queue.waiting_req_list = new_reqs + self.req_queue.waiting_req_list    # put to front

            all_staged_reqs = self.running_batch.reqs + self.wait_strategy.stopped_req_list
            await self._filter_batch(self.running_batch, [r.request_id for r in evict_reqs])
            self.adaptive_batchsize_router.staged_bs = len(all_staged_reqs)
            self._last_evicted_n = len(evict_reqs)
            self._last_evicted_size = sum(r.calcu_used_tokens() for r in evict_reqs)
            return evict_reqs


    async def _step(self):
        """
        事件处理循环
        """
        if self.adaptive_batchsize_router:

            # 使用状态机简化逻辑
            next_state = None
            if self.state != InferState.IDLE:
                print(self.state)

            if self.state == InferState.IDLE:
                _, self.batch_to_prefill = self.req_queue.generate_new_batch(None)
                if self.batch_to_prefill:
                    self.stats_tool.count_prompt_tokens(self.batch_to_prefill)

            if self.state == InferState.PREFILL:
                self.has_wait_tokens = 0
                for r in self.batch_to_prefill.reqs:
                    self.adaptive_batchsize_router.new_req_prompt_len_ema(r.input_len)
                await self._prefill_batch(self.batch_to_prefill)
                self.batch_to_prefill = self.batch_to_prefill if not self.batch_to_prefill.is_clear() else None

                if self.batch_to_prefill:
                    for r in self.batch_to_prefill.reqs:
                        print('new req', r.request_id, r.input_len, len(r.output_ids), r.max_output_len)
                        if hasattr(r, '_evicted_at'):
                            self._restore_recompute_req(r)
                            print('    after _restore_recompute_req', r.request_id, r.input_len, len(r.output_ids), r.max_output_len)

                if self.running_batch and self.batch_to_prefill:
                    await self._merge_batch(self.running_batch, self.batch_to_prefill)
                    self.running_batch.merge(self.batch_to_prefill)
                else:
                    self.running_batch = self.running_batch or self.batch_to_prefill
                self.batch_to_prefill = None
                self._update_adaptive_batchsize_router()

            if self.state == InferState.DECODE:
                if self.wait_strategy.can_decode(self.running_batch):
                    self.stats_tool.count_output_tokens(self.running_batch)
                    self._last_running_bs = len(self.running_batch.reqs)
                    await self._decode_batch(self.running_batch)
                    self._filter_runing_batch()
                    self.has_wait_tokens += 1
                    print(f'self.has_wait_tokens {self.has_wait_tokens}, is_waiting_list_empty {self.req_queue.is_waiting_list_empty()}')
                    self._update_adaptive_batchsize_router()
                    if self.has_wait_tokens >= self.max_wait_tokens and not self.req_queue.is_waiting_list_empty():
                        _, self.batch_to_prefill = self.req_queue.generate_new_batch(self.running_batch)
                else:
                    self.wait_strategy.select_reqs(self.running_batch)
                    await self._evict_reqs(space_needs=len(self.running_batch.reqs))    # freed evicted reqs before the split (._stop_reqs)
                    if not self.wait_strategy.is_stopped_list_empty():
                        await self._pause_reqs(self.running_batch, [r.request_id for r in self.wait_strategy.stopped_req_list])
                        next_state = InferState.PARTIAL_DECODE

            if self.state == InferState.PARTIAL_DECODE:
                self.stats_tool.count_output_tokens(self.running_batch)
                self._last_running_bs = len(self.running_batch.reqs)
                await self._decode_batch(self.running_batch)
                self.has_wait_tokens += 1
                print('self.has_wait_tokens', self.has_wait_tokens)
                self._update_adaptive_batchsize_router()
                restore_batch = self.wait_strategy.restore_batch(self.running_batch)
                await self._restore_batch(self.running_batch, restore_batch)
                self.running_batch.merge(restore_batch)
                if self.has_wait_tokens >= self.max_wait_tokens and not self.req_queue.is_waiting_list_empty():
                    _, self.batch_to_prefill = self.req_queue.generate_new_batch(self.running_batch)

            if next_state is None:
                if self.batch_to_prefill:
                    next_state = InferState.PREFILL
                elif self.running_batch:
                    next_state = InferState.DECODE
                else:
                    next_state = InferState.IDLE
            self.state = next_state
            return

        # 删除所有已经 finished 的 req
        if self.running_batch is None:
            _, new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is not None:
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                await self._prefill_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens = 0
            return
        token_ratio = (self.running_batch.calcu_used_tokens() + self.req_queue.calcu_stopd_tokens()) / self.router_max_total_token_num
        if self.has_wait_tokens < self.max_wait_tokens:
            if not self._can_decode(self.running_batch):
                await self._selection_batch()
                if self.running_batch.is_clear():
                    self.has_wait_tokens = self.max_wait_tokens
                    return
            self.stats_tool.count_output_tokens(self.running_batch)
            await self._decode_batch(self.running_batch)
            self._filter_runing_batch()
            self.has_wait_tokens += 1
            return
        else:
            if not self.req_queue.is_waiting_list_empty():
                restore, new_mini_batch = self.req_queue.generate_new_batch(self.running_batch, token_ratio)
                if new_mini_batch is not None:
                    if restore:
                        await self._restore_batch(self.running_batch, new_mini_batch)
                        self.running_batch.merge(new_mini_batch)
                    else:
                        self.stats_tool.count_prompt_tokens(new_mini_batch)
                        await self._prefill_batch(new_mini_batch)
                        if not new_mini_batch.is_clear():
                            await self._merge_batch(self.running_batch, new_mini_batch)
                            self.running_batch.merge(new_mini_batch)
                        self._filter_runing_batch()
                    self.has_wait_tokens = 0
                    return
            if not self._can_decode(self.running_batch):
                await self._selection_batch()
                if self.running_batch.is_clear():
                    return
            self.stats_tool.count_output_tokens(self.running_batch)
            await self._decode_batch(self.running_batch)
            self._filter_runing_batch()
            self.has_wait_tokens += 1
        return

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _prefill_batch(self, batch):
        await self._init_batch(batch)
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        batch.update_run_status(RunStatus.NORMAL)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req)
        return

    async def _decode_batch(self, batch:Batch):
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req)
        return

    async def _selection_batch(self):
        while True:
            pause_reqs = self.selector.select_reqs(self.running_batch)
            if pause_reqs:
                await self._pause_reqs(self.running_batch, pause_reqs)
            token_ratio = (self.running_batch.calcu_used_tokens() + self.req_queue.calcu_stopd_tokens()) / self.router_max_total_token_num
            if self.running_batch.is_clear() or token_ratio <= self.token_traio:
                break
            self.stats_tool.count_output_tokens(self.running_batch)
            await self._decode_batch(self.running_batch)
        return

    async def _pause_reqs(self, batch: Batch, req_id_list):
        req_objs = [(req_id, self.offload) for req_id in req_id_list]
        rets = [self.model_rpcs[tp_rank].pause_reqs(batch.batch_id, req_objs) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _restore_batch(self, batch1: Batch, batch2: Batch):
        req_ids = [r.request_id for r in batch2.reqs]
        rets = [self.model_rpcs[tp_rank].restore_reqs(batch1.batch_id, req_ids) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        batch2.update_run_status(RunStatus.NORMAL)
        return

    async def _filter_batch(self, batch: Batch, finished_req_id: List):
        req_id_list = [r.request_id for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, req_id_list, finished_req_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, has_new_finished_req):
        if has_new_finished_req:
            finished_req = batch.filter_finished()
            finished_req_id = [req.request_id for req in finished_req]
            if self.adaptive_batchsize_router:
                for r in finished_req:
                    self.adaptive_batchsize_router.finished_req_prompt_len_ema(r.input_len)
                    self.adaptive_batchsize_router.finished_req_decode_len_ema(len(r.output_ids) - 1)
            if batch.is_clear() and not self.req_queue.has_pending_reqs():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch, finished_req_id)
        return

    def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None
            return

    def _add_token_id_to_req(self, batch: Batch, req_ans):
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            if new_gen_metadata.get('offload'):
                continue
            req.output_ids.append(new_token_id)
            req.output_metadata_list.append(new_gen_metadata)
            if req.ending_predict:
                req.ending_predict.step(new_gen_metadata['_eosprob'], new_token_id == self.eos_id)
        return

    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            if new_gen_metadata.pop('offload'):
                continue
            req = batch.id_to_reqs[req_id]
            batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted))

        self.send_to_detokenization.send_pyobj(batch_out)
        return

    def _can_decode(self, batch: Batch):
        remaining_tokens = self.router_max_total_token_num - batch.calcu_used_tokens() - self.req_queue.calcu_stopd_tokens()
        return len(batch.reqs) <= remaining_tokens

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 3:
                prompt_ids, sampling_params, request_id = recv_req
                self.add_req(prompt_ids, sampling_params, request_id)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return

def start_router_process(args, router_port, detokenization_port, model_rpc_ports, mode, pipe_writer):
    try:
        router = RouterManager(
            args.model_dir,
            load_way="HF",
            world_size=args.tp,
            max_total_token_num=args.max_total_token_num,
            batch_max_tokens=args.batch_max_tokens,
            running_max_req_size=args.running_max_req_size,
            eos_id=args.eos_id,
            token_ratio=args.token_ratio,
            max_new_token_len=args.max_new_token_len,
            reserve_token_num=args.reserve_token_num,
            offload=not args.not_offload,
            strategy=args.strategy,
            adaptive_batchsize_router=args.adaptive_batchsize_router,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports,
            mode=mode,
            log_stats = not args.disable_log_stats,
            log_stats_interval = args.log_stats_interval)

        asyncio.run(router.wait_to_model_ready())
    except Exception as e:
        import traceback
        err_str = '\n'.join(traceback.format_exception(e))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send('init ok')

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
