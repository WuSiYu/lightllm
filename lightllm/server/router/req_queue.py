import bisect
from collections import deque
import json
import uuid
import asyncio
import numpy as np
from typing import List, Union
from ..io_struct import Batch, InferState, Req
from lightllm.utils.infer_utils import calculate_time
from lightllm.server.io_struct import Req
from lightllm.server.io_struct import ReqRunStatus

def EMA(initial: float, length: int, sliding_window=False):
    if not sliding_window:
        _val = initial
        def f(update: Union[float, None] = None):
            nonlocal _val
            if update is not None:
                _val = _val * (1 - 1/length) + update * 1/length
            return _val
    else:
        _q = deque([initial] * length, maxlen=length)
        _sum = initial * length
        def f(update: Union[float, None] = None):
            nonlocal _q, _sum
            if update is not None:
                _sum = _sum - _q[0] + update
                _q.append(update)
            return _sum / length
    return f

class AdaptiveBatchsizeRouter:
    INITAL_TARGET_BS = 50
    HISTORY_LEN = 2048              # steps
    HISTORY_REQ_LEN_EMA_LEN = 100   # reqs

    AUTO_RATE = True   # if True, following value will been overwriten
    LOWER_MEM_RATE = 0.95
    UPPER_MEM_RATE = 0.98

    BALANCE_CHECK_MIN_BS = 30
    BALANCE_MIN_COV = 0
    # BALANCE_MIN_COV = 0.5
    # BALANCE_MIN_COV = 0.8
    DECODE_FULL_ONGOING_RATIO = 0.7
    RE_BALANCE_PARTIAL_RATIO = 0.5
    RE_BALANCE_MIN_INTERVAL = 1

    HIGH_BS_TOKEN_RATIO = 1
    MINIMAL_BS_RATIO = 0.8

    BS_INCREASE_PROB_RATIO = 0.5

    # RANDOM_SLOW_ADD_NEW_REQ = True
    RANDOM_SLOW_ADD_NEW_REQ = False

    # INITAL_TARGET_BS = 500
    # HISTORY_LEN = 200               # steps
    # HISTORY_REQ_LEN_EMA_LEN = 50    # reqs

    # LOWER_MEM_RATE = 1
    # UPPER_MEM_RATE = 1

    # BALANCE_CHECK_MIN_BS = 30
    # BALANCE_MIN_COV = 0
    # RE_BALANCE_PARTIAL_RATIO = 0.5

    # HIGH_BS_TOKEN_RATIO = 100
    # MINIMAL_BS_RATIO = 1

    # BS_INCREASE_PROB_RATIO = 100
    # RANDOM_SLOW_ADD_NEW_REQ = False

    def __init__(self, max_total_token_num: int, inital_target_bs: int = INITAL_TARGET_BS):
        self._last_used_tokens = 0
        # self._free_alloc_ratio_ema = 0

        self._history_req_prompt_len = deque([1024] * self.HISTORY_REQ_LEN_EMA_LEN, self.HISTORY_REQ_LEN_EMA_LEN)
        self._history_req_decode_len = deque([1024] * self.HISTORY_REQ_LEN_EMA_LEN, self.HISTORY_REQ_LEN_EMA_LEN)

        self._step = 0
        # self._history_freed = deque()
        # self._history_allocated = deque()
        # self._history_finished = deque([self.INITAL_TARGET_BS])   # boostrap
        # self.recent_finished = self.INITAL_TARGET_BS
        # self._history_started = deque()
        self.finished_req_prompt_len_ema = EMA(1024, self.HISTORY_REQ_LEN_EMA_LEN, sliding_window=False)
        self.finished_req_decode_len_ema = EMA(2048, self.HISTORY_REQ_LEN_EMA_LEN, sliding_window=False)
        # self.new_req_prompt_len_ema = EMA(1024, self.HISTORY_REQ_LEN_EMA_LEN, sliding_window=True)
        # self.running_bs_ema = EMA(self.INITAL_TARGET_BS, self.HISTORY_REQ_LEN_EMA_LEN, sliding_window=True)
        self.max_total_token_num = max_total_token_num
        # self.minimal_bs = 30    # FIXME
        # self.is_balance = True
        # self._last_rebalance = 0
        # self.stopped_cause = None
        # self.can_prefill = True

        self.target_staged_bs = inital_target_bs    # controlled, decide by policy
        self.target_running_bs = inital_target_bs   # controlled, to make staged_bs ---> target_staged_bs, which also makes staged_bs == target_staged_bs
        self.staged_bs = 0                          # runtime status
        self.running_bs = 0                         # runtime status

    def record_finished_req_len(self, prompt_len: int, decode_len: int):
        self._history_req_prompt_len.append(prompt_len)
        self._history_req_decode_len.append(decode_len)

    def _get_ideal_mem_req_len(self):
        "return prompt_avg_len, in_batch_decode_avg_len (predicted from history)"
        # prompt_avg_len = np.average(self._history_req_prompt_len)

        decode_full_len = np.array(self._history_req_decode_len)
        decode_full_len_weighted = decode_full_len * (decode_full_len / np.average(decode_full_len))
        in_batch_decode_avg_full_len = np.average(decode_full_len_weighted)
        in_batch_decode_avg_len = in_batch_decode_avg_full_len / 2

        prompt_len = np.array(self._history_req_prompt_len)
        prompt_len_weighted = prompt_len * (decode_full_len / np.average(decode_full_len))
        prompt_avg_len = np.average(prompt_len_weighted)

        print(f"    _get_ideal_mem_req_len is {prompt_avg_len} + {in_batch_decode_avg_len}")
        return prompt_avg_len, in_batch_decode_avg_len

    def _total_staged_tokens(self, current_batch: Union[Batch, None], stopped_reqs: List[Req]):
        sum_ = 0
        if current_batch:
            sum_ += sum(req.get_used_tokens() for req in current_batch.reqs)
        sum_ += sum(req.get_used_tokens() for req in stopped_reqs)
        return sum_

    def _is_balance(self, reqs: List[Req]):
        # print = lambda *x: None

        decoded_lens = np.array([len(x.output_ids) - 1 for x in reqs])
        # print("prompt_lens[]:", json.dumps([x.input_len for x in reqs]))
        # print("decoded_lens[]:", json.dumps([len(x.output_ids) - 1 for x in reqs]))
        avg = np.average(decoded_lens)
        std = np.std(decoded_lens)
        cov = std / np.average(decoded_lens)
        print(f"        prompt len average {np.average([x.input_len for x in reqs])}")
        print(f"        decoded len average {avg}")
        print(f"        decoded len std {std}")
        print(f"        decoded len cov {cov}")
        # print(f"        history_prompt_len_ema {self.finished_req_prompt_len_ema()}")
        # print(f"        history_decode_len_ema {self.finished_req_decode_len_ema()}")
        ongoing_avg_decode_len_ratio = np.average(decoded_lens) / self.finished_req_decode_len_ema()
        print(f"        ongoing_avg_decode_len_ratio {ongoing_avg_decode_len_ratio}")
        if len(reqs) < self.BALANCE_CHECK_MIN_BS:
            return True
        return cov > self.BALANCE_MIN_COV
        # return cov > self.BALANCE_MIN_COV and ongoing_avg_decode_len_ratio < self.DECODE_FULL_ONGOING_RATIO
        # return ongoing_avg_decode_len_ratio < self.DECODE_FULL_ONGOING_RATIO

    # def _is_balance(self, reqs: List[Req]):
    #     if len(reqs) < self.BALANCE_CHECK_MIN_BS:
    #         return True
    #     decoded_lens = np.array([len(x.output_ids) for x in reqs])
    #     decoded_lens.sort()
    #     target_lens = np.linspace(0, decoded_lens[-1], len(decoded_lens))
    #     print(decoded_lens, target_lens)
    #     cov = np.sqrt(np.sum(np.square(decoded_lens - target_lens)) / len(decoded_lens)) / decoded_lens[-1]

    #     print(f"        decoded len average {np.average(decoded_lens)}")
    #     print(f"        decoded len std {np.std(decoded_lens)}")
    #     print(f"        decoded len cov {cov}")
    #     return cov < 0.5

    def update(self, state: InferState, staged_bs: int, running_bs: int,
               running_batch: Union[Batch, None],
               reqs_pending: int, staged_bs_delta: int,
               evicted_n: int = 0):
        # print = lambda *x: None

        self._step += 1

        self.staged_bs = staged_bs
        self.running_bs = running_bs
        # self.running_bs_ema(self.running_bs)
        used_tokens = self._total_staged_tokens(running_batch, [])
        rate = used_tokens / self.max_total_token_num

        if evicted_n:
            print(f"  will evict {evicted_n} reqs")


        reqs = running_batch.reqs if running_batch else []
        _is_balance = self._is_balance(reqs)
        # if not _is_balance and self._step - self._last_rebalance > self.RE_BALANCE_MIN_INTERVAL:
        #     self.is_balance = False
        #     self._last_rebalance = self._step
        # else:
        #     self.is_balance = True



        ### update mem state ###

        # if state == InferState.DECODE or state == InferState.PARTIAL_DECODE:
        #     last_freed = self._last_used_tokens - used_tokens + running_bs - evicted_size   # not include evicted
        #     last_allocated = running_bs
        # elif state == InferState.PREFILL:
        #     last_freed = 0
        #     last_allocated = used_tokens - self._last_used_tokens

        # self._history_freed.append(last_freed)
        # self._history_allocated.append(last_allocated)

        # if self._step > self.HISTORY_LEN:
        #     self._history_freed.popleft()
        #     self._history_allocated.popleft()

        # recent_frees = sum(self._history_freed)
        # recent_allocs = sum(self._history_allocated)
        # self._recent_freed_allocated_ratio = recent_frees / recent_allocs if recent_allocs else 1
        # print(f"    last_freed: {last_freed}, last_allocated: {last_allocated}")
        # print(f"    self._recent_freed_allocated_ratio: {self._recent_freed_allocated_ratio}")



        ### update bs history state ###

        n_finished = -staged_bs_delta - evicted_n if state == InferState.DECODE else 0
        n_started = staged_bs_delta if state == InferState.PREFILL else 0

        print(f"    n_finished {n_finished} - n_started {n_started}")

        # self._history_finished.append(n_finished)
        # self._history_started.append(n_started)

        # recent_finished = sum(self._history_finished)
        # self.recent_finished = recent_finished
        # recent_started = sum(self._history_started)
        # self._recent_finished_start_ratio = recent_finished / recent_started if recent_started else 1
        # print(f"    recent_finished: {recent_finished}, recent_started: {recent_started}")
        # print(f"    self._recent_finished_start_ratio: {self._recent_finished_start_ratio}")

        # if self._step > self.HISTORY_LEN:
        #     self._history_finished.popleft()
        #     self._history_started.popleft()



        # ### update history len ema ###

        # if state == InferState.DECODE or state == InferState.PARTIAL_DECODE:
        #     if n_finished:
        #         last_freed_req_len = last_freed / n_finished
        #         self._req_len_ema = self._req_len_ema * (1 - 1 / self.HISTORY_REQ_LEN_EMA_LEN) + last_freed_req_len * (1 / self.HISTORY_REQ_LEN_EMA_LEN)
        #         print(f"    last_freed_req_len: {last_freed_req_len}")
        #         print(f"    self._req_len_ema: {self._req_len_ema}")

        # if state == InferState.PREFILL:
        #     if n_started:
        #         last_allocated_req_len = last_allocated / n_started
        #         self._req_prompt_len_ema = self._req_prompt_len_ema * (1 - 1 / self.HISTORY_REQ_LEN_EMA_LEN) + last_allocated_req_len * (1 / self.HISTORY_REQ_LEN_EMA_LEN)
        #         print(f"    last_allocated_req_len: {last_allocated_req_len}")
        #         print(f"    self._req_prompt_len_ema: {self._req_prompt_len_ema}")


        if self.AUTO_RATE:
            req_len = self.finished_req_prompt_len_ema() + self.finished_req_decode_len_ema()
            self.UPPER_MEM_RATE = (self.max_total_token_num - req_len*2) / self.max_total_token_num
            self.LOWER_MEM_RATE = (self.max_total_token_num - req_len*4) / self.max_total_token_num
            print(f"    auto mem rate: self.LOWER_MEM_RATE {self.LOWER_MEM_RATE:.4f}, self.UPPER_MEM_RATE {self.UPPER_MEM_RATE:.4f}")


        ### bs policy ###
        # target_staged_delta = 0

        # _history_len = len(self._history_finished)
        # if rate < self.LOWER_MEM_RATE:
        #     finish_prob = recent_finished / _history_len
        #     raise_prob = finish_prob * self.BS_INCREASE_PROB_RATIO
        #     print(f"    raise_prob = {raise_prob} ({recent_finished} / {_history_len} * {self.BS_INCREASE_PROB_RATIO})")
        #     target_staged_delta = int(np.random.rand() < raise_prob)
        # elif rate < self.UPPER_MEM_RATE:
        #     pass
        # else:
        #     # target_staged_delta = -1
        #     # target_staged_delta = int(np.random.rand() < 0.2)
        #     start_prob = recent_started / _history_len
        #     fall_prob = start_prob * 0.5
        #     print(f"    fall_prob = {fall_prob} ({recent_started} / {_history_len} * {0.5})")
        #     target_staged_delta = -int(np.random.rand() < fall_prob)


        # if reqs_pending == 0 or reqs_pending < self.target_staged_bs - self.staged_bs:
        #     # undersaturation, don't increase bs now
        #     print(" + limit: delta <= 0, cause: WAIT_NEW_REQ")
        #     target_staged_delta = min(0, target_staged_delta)

        # if self._recent_freed_allocated_ratio < 0.75:
        #     # too many requests on-going and not finished soon
        #     print(" + limit: delta <= 0, cause: TOO_LESS_FREED")
        #     target_staged_delta = min(0, target_staged_delta)

        # if self._recent_finished_start_ratio < 0.67:
        #     # too many requests started
        #     print(" + limit: delta <= 0, cause: TOO_MANY_STARTS")
        #     target_staged_delta = min(0, target_staged_delta)

        # ideal_mem_req_len = self.finished_req_prompt_len_ema() + self.finished_req_decode_len_ema() / 2

        # self.UPPER_MEM_RATE = 0.75  # test
        if reqs:
            prompt_len = [x.input_len for x in reqs]
            decoded_len = [len(x.output_ids) - 1 for x in reqs]
            # prompt_avg_len, in_batch_decode_avg_len = self._get_ideal_mem_req_len()
            # ideal_mem_req_len = prompt_avg_len + in_batch_decode_avg_len
            # high_bs = int(self.max_total_token_num * self.HIGH_BS_TOKEN_RATIO / ideal_mem_req_len)
            # print(f" -> high_bs {high_bs} (prompt_avg_len + in_batch_decode_avg_len)")

            # ideal_mem_req_len_mixed = prompt_avg_len + max(in_batch_decode_avg_len, np.average(decoded_len))
            # high_bs_mixed = int(self.max_total_token_num * self.UPPER_MEM_RATE / ideal_mem_req_len_mixed)
            # print(f" -> high_bs_mixed {high_bs_mixed} (prompt_avg_len + max(in_batch_decode_avg_len, np.average(decoded_len)))")

            # ideal_mem_req_len_mixed = np.average(prompt_len) + max(in_batch_decode_avg_len, np.average(decoded_len))
            # high_bs_mixed_l = int(self.max_total_token_num * self.UPPER_MEM_RATE / ideal_mem_req_len_mixed)
            # print(f" -> high_bs_mixed_l {high_bs_mixed_l}")

            # ideal_mem_req_len_mixed2 = np.average(prompt_len) + max(in_batch_decode_avg_len, np.average(decoded_len))
            # high_bs_mixed2 = int(self.max_total_token_num * self.HIGH_BS_TOKEN_RATIO / ideal_mem_req_len_mixed2)
            # print(f" -> high_bs_mixed2 {high_bs_mixed2} (np.average(prompt_len) + max(in_batch_decode_avg_len, np.average(decoded_len)))")

            # high_bs_mixed3 = self.staged_bs + int((self.max_total_token_num - ideal_mem_req_len_mixed2 * self.staged_bs) / ideal_mem_req_len)
            # print(f" -> high_bs_mixed3 {high_bs_mixed3} (staged_bs + (total - mixed2 * staged_bs) / ideal)")


            # cur_len_bs = int(self.max_total_token_num * self.UPPER_MEM_RATE / (np.average(prompt_len) + np.average(decoded_len)))
            # print(f"cur {np.average(prompt_len)} + {np.average(decoded_len)}")
            # print(f" -> cur_len_bs {cur_len_bs}")

            his_Lo = sorted(self._history_req_decode_len)
            his_Lo_postfix_sum = np.flip(np.flip(his_Lo, 0).cumsum(), 0)
            Lo_exps = []
            for dl in decoded_len:
                pos = bisect.bisect(his_Lo, dl)
                if pos == len(his_Lo):
                    exp = dl
                else:
                    exp = his_Lo_postfix_sum[pos] / (len(his_Lo) - pos)
                Lo_exps.append(exp)
            Ld_exp = np.average(Lo_exps) * 0.5
            print(f"Ld_exp {Ld_exp}")
            ideal_mem_req_len_exp = np.average(prompt_len) + Ld_exp
            # _upper_mem_rate = 1
            _upper_mem_rate = self.UPPER_MEM_RATE
            # _upper_mem_rate = min(0.95, self.UPPER_MEM_RATE)
            high_bs_exp = int(self.max_total_token_num * _upper_mem_rate / ideal_mem_req_len_exp)
            print(f" -> high_bs_exp {high_bs_exp}")


        else:
            high_bs = high_bs_mixed = high_bs_mixed_l = high_bs_mixed2 = high_bs_mixed3 = cur_len_bs = high_bs_exp = int(self.max_total_token_num * self.HIGH_BS_TOKEN_RATIO / sum(self._get_ideal_mem_req_len()))

        # # if self.target_staged_bs > high_bs_mixed2:
        # if self.target_staged_bs > high_bs_mixed:
        #     # bs too high
        #     print(" + limit: delta <= 0, cause: BS_HIGH")
        #     target_staged_delta = min(0, target_staged_delta)

        # low_bs = self.max_total_token_num * self.LOWER_MEM_RATE \
        #     / (self.finished_req_prompt_len_ema() + self.finished_req_decode_len_ema()) * self.MINIMAL_BS_RATIO
        # if self.target_staged_bs < low_bs and rate < self.LOWER_MEM_RATE:
        # # if self.target_staged_bs < low_bs and target_staged_delta == 1:
        #     # bs too low
        #     print(" + limit: delta 1 ~ 10, cause: BS_LOW")
        #     target_staged_delta = min(int(low_bs) - self.target_staged_bs, 10)

        # if evicted_n:
        #     # has evicted
        #     target_staged_delta = min(-1, target_staged_delta)
        #     print(" + limit: delta <= -1, cause: HAS_EVICTED")

        # if self.target_staged_bs < self.staged_bs:
        #     # staged not lowered yet, not further decrease the bs
        #     print(" + limit: delta >= 0, cause: WAIT_RADUCE")
        #     target_staged_delta = max(0, target_staged_delta)

        # print(f" = final target_staged_delta: {target_staged_delta}")
        # self.target_staged_bs += target_staged_delta
        # self.target_staged_bs = max(1, self.target_staged_bs)

        # if self.staged_bs <= self.target_staged_bs:
        #     self.target_running_bs = self.target_staged_bs
        # else:
        #     to_reduce = self.staged_bs - self.target_staged_bs
        #     running_bs_reduce_factor = 1
        #     self.target_running_bs = max(1, self.staged_bs - to_reduce * running_bs_reduce_factor)

        #     if self.target_running_bs < self.minimal_bs:
        #         print(" + limit: target_running_bs >= minimal_bs, cause: MINIMAL_BS")
        #         self.target_running_bs = self.minimal_bs

        # self.target_staged_bs = self.target_running_bs = high_bs
        # self.target_staged_bs = self.target_running_bs = high_bs_mixed_l
        # self.target_staged_bs = self.target_running_bs = 200
        # if rate < 0.9:
        #     self.target_staged_bs = self.target_running_bs = 200
        # else:
        #     self.target_staged_bs = self.target_running_bs = self.staged_bs
        self.target_staged_bs = self.target_running_bs = high_bs_exp
        # self.target_staged_bs = self.target_running_bs = high_bs_mixed3
        # self.target_staged_bs = self.target_running_bs = cur_len_bs



        # baseline_bs = int(self.max_total_token_num * self.UPPER_MEM_RATE / (np.average(self._history_req_prompt_len) + np.average(self._history_req_decode_len)))
        # print(f"baseline_bs {baseline_bs}")
        # self.target_staged_bs = self.target_running_bs = baseline_bs    # baseline

        self._last_used_tokens = used_tokens

        if self.running_bs != 0:
            print(f"AdaptiveBatchsizeRouter usage {rate*100:.2f}% ({used_tokens} / {self.max_total_token_num})")
            print(self)

    def __repr__(self):
        return f"AdaptiveBatchsizeRouter(target_staged_bs={self.target_staged_bs}, target_running_bs={self.target_running_bs}, staged_bs={self.staged_bs}, running_bs={self.running_bs})"


class ReqQueue:

    def __init__(self, args, prompt_cache_used_tokens, prompt_cache_req_num, adaptive_batchsize_router: Union[AdaptiveBatchsizeRouter, None] = None) -> None:
        self.max_total_tokens = args.max_total_token_num
        assert args.batch_max_tokens is not None
        self.batch_max_tokens = args.batch_max_tokens
        self.running_max_req_size = args.running_max_req_size
        self.waiting_req_list: List[Req] = []
        self.router_token_ratio = args.router_token_ratio
        self.router_max_new_token_len = args.router_max_new_token_len
        self.pause_req_dict = {} # 用于保存队列中被暂停的请求，暂停原因为 ReqRunStatus.PAUSED_AND_KVKEEP  ReqRunStatus.PAUSED_AND_OFFLOAD
        self.pause_req_used_tokens = 0

        self.is_splitfuse_mode = args.splitfuse_mode
        self.splitfuse_block_size = args.splitfuse_block_size

        # 当使用 prompt cache 特性时的维护变量
        self.prompt_cache_used_tokens = prompt_cache_used_tokens
        self.prompt_cache_req_num = prompt_cache_req_num

        self.adaptive_batchsize_router = adaptive_batchsize_router

    def append(self, req):
        self.waiting_req_list.append(req)
        return

    def back_to_wait_list(self, req_list:List[Req]):
        for req in req_list:
            if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
                self.pause_req_dict[req.request_id] = req
        self.waiting_req_list = req_list + self.waiting_req_list
        self.recalcu_pause_req_used_tokens()
        return

    def _init_cache_list(self, current_batch:Batch, is_busy):
        self.cache_pause_reqs_used_tokens = self.pause_req_used_tokens
        self.cache_pause_reqs_num = len(self.pause_req_dict)
        if current_batch is not None:
            self.cache_len_list = [req.get_tuple_tokens(is_busy, self.router_max_new_token_len) for req in current_batch.reqs]
        else:
            self.cache_len_list = []

    # @calculate_time(show=True, min_cost_ms=0.1)
    def _can_add_new_req(self, req:Req, is_busy):
        self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len)) # hard to analysis

        if self.adaptive_batchsize_router:
            max_total_tokens = self.adaptive_batchsize_router.max_total_token_num
            if len(self.cache_len_list) <= self.adaptive_batchsize_router.target_running_bs:
                # if sum(e[0] for e in self.cache_len_list) <= max_total_tokens - len(self.cache_len_list):
                if sum(e[0] for e in self.cache_len_list) <= max_total_tokens - 2048 - len(self.cache_len_list):
                    # vram is enough for prefill and next decode
                    return True
                else:
                    print(f' + prefill will limited by vram at bs {len(self.cache_len_list)}')
                    print(f' + {sum(e[0] for e in self.cache_len_list)} > {max_total_tokens} - 2048 - {len(self.cache_len_list)}')
                    # print(f' + {sum(e[0] for e in self.cache_len_list)} > {max_total_tokens} * {self.adaptive_batchsize_router.UPPER_MEM_RATE} - {len(self.cache_len_list)}')
                    # may reduce target_staged_bs if memory not enough for new refill
                    # self.adaptive_batchsize_router.target_staged_bs = min(len(self.cache_len_list), self.adaptive_batchsize_router.target_staged_bs)
                    pass
            return False

        self.cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
            self.cache_pause_reqs_used_tokens -= req.get_used_tokens()
            self.cache_pause_reqs_num -= 1

        ok_token_num = need_max_token_num < self.max_total_tokens - self.cache_pause_reqs_used_tokens - self.prompt_cache_used_tokens
        ok_req_num = len(self.cache_len_list) + self.cache_pause_reqs_num + self.prompt_cache_req_num <= self.running_max_req_size

        if ok_token_num and ok_req_num:
            return True
        else:
            return False

    #@calculate_time(show=True, min_cost_ms=10)
    def generate_new_batch(self, current_batch:Batch):

        # 如果当前已经被调度的请求数量超过了上限，直接不调度新的请求了。
        exist_req_num = self.prompt_cache_req_num
        exist_req_num += 0 if current_batch is None else len(current_batch.reqs)
        exist_req_num += len(self.pause_req_dict)
        req_is_full = exist_req_num >= self.running_max_req_size
        if req_is_full:
            return None

        # 计算当前所有的token使用量，包括当前使用和暂停的
        cur_all_used_tokens = 0 if current_batch is None else current_batch.batch_used_tokens
        cur_all_used_tokens += self.recalcu_pause_req_used_tokens() + self.prompt_cache_used_tokens

        # 判断当前服务是否处于token使用率过高的状态，过高的情况下，调度要偏向保守
        cur_token_ratio = cur_all_used_tokens / self.max_total_tokens
        is_busy = cur_token_ratio >= self.router_token_ratio

        # 得到当前batch 往前 decode 一次，需要的token量，在 splitfuse 模式下才有用，因为splitfuse
        # 模式下 类似prefill 和 deocde 是在一起进行的，所以需要合并考虑。
        # 普通模式是 先prefill 后 decode，所以只考虑prefill的时候 token使用量不要超过限制。
        if not self.is_splitfuse_mode:
            cur_batch_decode_need_tokens = 0
        else:
            cur_batch_decode_need_tokens = 0 if current_batch is None else current_batch.batch_decode_need_tokens

        self._init_cache_list(current_batch, is_busy)
        can_run_list = []
        new_batch_first_router_need_tokens = 0 # 主要是对 prefill 或者 splitfuse 大块计算时候的限制
        aborted_count = 0
        for req in self.waiting_req_list:
            if req.aborted and req.req_status == ReqRunStatus.WAIT_IN_QUEUE:
                # 由于管理的复杂性，只有没有被调度运行过的请求可以因为abort直接在队列中忽略掉.
                # 暂停的请求需要恢复后，由 router manager 部分来过滤。暂时保持这种处理方法, 否则会导致管理token的泄漏
                aborted_count += 1
                continue
            req_first_router_need_tokens = req.get_first_router_need_tokens()
            if self._can_add_new_req(req, is_busy) and cur_batch_decode_need_tokens + new_batch_first_router_need_tokens + req_first_router_need_tokens <= self.batch_max_tokens:
                can_run_list.append(req)
                new_batch_first_router_need_tokens += req_first_router_need_tokens
                if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
                    self.pause_req_dict.pop(req.request_id)
            else:
                break

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            # 生成新 batch 以后，更新一下状态
            self.recalcu_pause_req_used_tokens()
            return new_batch
        else:
            return None

    def recalcu_pause_req_used_tokens(self):
        used_tokens = 0
        for req_id, req_obj in self.pause_req_dict.items():
            used_tokens += req_obj.get_used_tokens()
        self.pause_req_used_tokens = used_tokens
        return self.pause_req_used_tokens
