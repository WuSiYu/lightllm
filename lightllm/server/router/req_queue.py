from ast import NotIn
import bisect
from collections import deque
import json
import math
import random
import uuid
import asyncio
import numpy as np
from typing import List, Tuple, Union
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
    INITAL_TARGET_BS = 100
    HISTORY_LEN = 2048              # steps
    HISTORY_REQ_LEN_EMA_LEN = 100   # reqs

    AUTO_RATE = True   # if True, following value will been overwriten
    LOWER_MEM_RATE = 0.95
    UPPER_MEM_RATE = 0.98

    BALANCE_CHECK_MIN_BS = 30
    BALANCE_MIN_COV = 0
    # BALANCE_MIN_COV = 0.5
    # BALANCE_MIN_COV = 0.8
    # DECODE_FULL_ONGOING_RATIO = 0.7
    # RE_BALANCE_PARTIAL_RATIO = 0.5
    # RE_BALANCE_MIN_INTERVAL = 1

    HIGH_BS_TOKEN_RATIO = 1
    MINIMAL_BS_RATIO = 0.8

    # BS_INCREASE_PROB_RATIO = 0.5

    # RANDOM_SLOW_ADD_NEW_REQ = True
    # RANDOM_SLOW_ADD_NEW_REQ = False

    _HISTORY_REQ_PROMPT_LEN_INITAL = 1024   # FIXME
    _HISTORY_REQ_DECODE_LEN_INITAL = 1024

    # START_RATE_LIMIT = True
    START_RATE_LIMIT = False
    START_RATE_LIMIT_WINDOW = 100     # steps
    START_RATE_LIMIT_RATIO = 1.05

    PREFILL_RESERVED_TOKEN_RATIO = 0.01

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

        self._history_req_prompt_len = deque([self._HISTORY_REQ_PROMPT_LEN_INITAL] * (self.HISTORY_REQ_LEN_EMA_LEN // 3), maxlen=self.HISTORY_REQ_LEN_EMA_LEN)
        self._history_req_decode_len = deque([self._HISTORY_REQ_DECODE_LEN_INITAL] * (self.HISTORY_REQ_LEN_EMA_LEN // 3), maxlen=self.HISTORY_REQ_LEN_EMA_LEN)

        self._recent_starts_steps = deque([-1])


        self._step = 0
        # self._history_freed = deque()
        # self._history_allocated = deque()
        # self._history_finished = deque([self.INITAL_TARGET_BS])   # boostrap
        # self.recent_finished = self.INITAL_TARGET_BS
        # self._history_started = deque()
        # self.predicted_output_len_dict = {}
        # self.predicted_output_len = 1024
        self.finished_req_prompt_len_ema = EMA(1024, self.HISTORY_REQ_LEN_EMA_LEN, sliding_window=False)    # TODO: old, update AUTO_RATE
        self.finished_req_decode_len_ema = EMA(2048, self.HISTORY_REQ_LEN_EMA_LEN, sliding_window=False)
        # self.new_req_prompt_len_ema = EMA(1024, self.HISTORY_REQ_LEN_EMA_LEN, sliding_window=True)
        # self.running_bs_ema = EMA(self.INITAL_TARGET_BS, self.HISTORY_REQ_LEN_EMA_LEN, sliding_window=True)
        self.max_total_token_num = max_total_token_num
        # self.minimal_bs = 30    # FIXME
        # self.is_balance = True
        # self._last_rebalance = 0
        # self.stopped_cause = None
        # self.can_prefill = True
        if self.START_RATE_LIMIT:
            self._last_target_bs = inital_target_bs
            self.ideal_start_rate = 1
            self.bs_change_floating_room = 0

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
        decode_full_len_avg = np.average(decode_full_len)
        decode_full_len_weighted = decode_full_len * (decode_full_len / decode_full_len_avg)
        in_batch_decode_avg_full_len = np.average(decode_full_len_weighted)
        in_batch_decode_avg_len = in_batch_decode_avg_full_len / 2

        prompt_len = np.array(self._history_req_prompt_len)
        prompt_len_weighted = prompt_len * (decode_full_len / decode_full_len_avg)
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

    def recent_starts_num(self):
        assert self.START_RATE_LIMIT
        return len(self._recent_starts_steps)

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
        if reqs:
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

        if self.START_RATE_LIMIT:
            for _ in range(n_started):
                self._recent_starts_steps.append(self._step)

            while self._recent_starts_steps and self._recent_starts_steps[0] < self._step - self.START_RATE_LIMIT_WINDOW:
                self._recent_starts_steps.popleft()


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
            prompt_avg_len, in_batch_decode_avg_len = self._get_ideal_mem_req_len()
            print(f"  inbatch balance factor {np.average(decoded_len) / in_batch_decode_avg_len}")
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

            predicted_req_mem_len = np.average(prompt_len) + in_batch_decode_avg_len
            used_tokens_mixed = max(used_tokens, self.staged_bs * predicted_req_mem_len)
            delta = math.floor((self.max_total_token_num * self.UPPER_MEM_RATE - used_tokens_mixed) / predicted_req_mem_len)
            high_bs_mixed3 = self.staged_bs + delta
            print(f" -> high_bs_mixed3 {high_bs_mixed3:4} (staged_bs + (capacity * UPPER_RATE - used_tokens_mixed) / ideal)")
            print(f" ->                     ({self.staged_bs} + ({self.max_total_token_num * self.UPPER_MEM_RATE - used_tokens_mixed} / {predicted_req_mem_len})")


            # cur_len_bs = int(self.max_total_token_num * self.UPPER_MEM_RATE / (np.average(prompt_len) + np.average(decoded_len)))
            # print(f"cur {np.average(prompt_len)} + {np.average(decoded_len)}")
            # print(f" -> cur_len_bs {cur_len_bs}")

            # 条件概率
            # his_Lo = sorted(self._history_req_decode_len)
            # his_Lo_postfix_sum = np.flip(np.flip(his_Lo, 0).cumsum(), 0)
            # Lo_exps_dict = {}
            # for req in reqs:
            #     dl = len(req.output_ids)
            #     pos = bisect.bisect(his_Lo, dl)
            #     if pos == len(his_Lo):
            #         exp = dl
            #     else:
            #         if pos > 0:     # TODO: 换成插值
            #             pos -= 1
            #         exp = his_Lo_postfix_sum[pos] / (len(his_Lo) - pos)
            #     Lo_exps_dict[req.request_id] = exp
            # Lo_exp = np.average(list(Lo_exps_dict.values()))
            # # self.predicted_output_len = Lo_exp
            # # self.predicted_output_len_dict = Lo_exps_dict

            # Ld_exp = Lo_exp * 0.5
            # print(f"Ld_exp {Ld_exp}")
            # ideal_mem_req_len_exp = np.average(prompt_len) + Ld_exp
            # # _upper_mem_rate = 1
            # _upper_mem_rate = self.UPPER_MEM_RATE
            # # _upper_mem_rate = min(0.95, self.UPPER_MEM_RATE)
            # high_bs_exp = int(self.max_total_token_num * _upper_mem_rate / ideal_mem_req_len_exp)
            # print(f" -> high_bs_exp {high_bs_exp}")


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
        # self.target_staged_bs = self.target_running_bs = high_bs_exp
        self.target_staged_bs = self.target_running_bs = high_bs_mixed3
        # self.target_staged_bs = self.target_running_bs = cur_len_bs

        if self.START_RATE_LIMIT:
            ideal_req_output_len = np.average(self._history_req_decode_len)
            self.ideal_start_rate = self.target_running_bs / ideal_req_output_len

            bs_delta = self.target_staged_bs - self._last_target_bs
            self.bs_change_floating_room = max(0, self.bs_change_floating_room + bs_delta)

        # # FIXME: test only: maxbs
        # self.target_staged_bs = self.target_running_bs = 300


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

    def _sample_cache_list(self, reqs: List[Req]) -> List[Tuple[int, int]]:
        cache_len_list = []
        his_Lo = sorted(self.adaptive_batchsize_router._history_req_decode_len)
        for req in reqs:
            dl = len(req.output_ids)
            pos = bisect.bisect(his_Lo, dl)
            points = 1 + len(his_Lo) - pos
            if points == 1:
                sampled = dl
            else:
                rand_p = random.random() * (points - 1)

                # (伪) 线性差值
                if rand_p < 1:
                    sampled = dl + (his_Lo[pos] - dl) * rand_p
                else:
                    l = his_Lo[pos + int(rand_p) - 1]
                    u = his_Lo[pos + int(rand_p)]
                    sampled = l + (u - l) * (rand_p - int(rand_p))

            cache_len_list.append(req.get_tuple_tokens(False, sampled))
        return cache_len_list

    def _calc_max_token_num_needed(self, cache_len_list: List[Tuple[int, int]]) -> int:
        cache_len_list.sort(key=lambda x: -x[1])

        left_out_len_array = np.array([e[1] for e in cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(cache_len_list) + 1, 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        return need_max_token_num

    def _init_cache_list(self, current_batch:Batch, is_busy):
        self.cache_pause_reqs_used_tokens = self.pause_req_used_tokens
        self.cache_pause_reqs_num = len(self.pause_req_dict)
        if current_batch is not None:
            if self.adaptive_batchsize_router:
                MINIMUM_SAMPLES = 200
                MAXIMUM_LISTS = 5
                n_lists = int(MINIMUM_SAMPLES / len(current_batch.reqs)) + 1
                n_lists = min(MAXIMUM_LISTS, n_lists)

                self._cache_len_lists = [self._sample_cache_list(current_batch.reqs) for _ in range(n_lists)]
                # p_dict = self.adaptive_batchsize_router.predicted_output_len_dict
                # print(" * p_dict", p_dict)
                # self.cache_len_list = [req.get_tuple_tokens(is_busy, p_dict[req.request_id]) for req in current_batch.reqs]

                self.cache_len_list = self._cache_len_lists[0]   # keep compatibility
                if random.random() < 0.1:  # debug
                    for i, li in enumerate(self._cache_len_lists):
                        print(" * cache_len_list", i, li)
                    # print(" * sum cache_len_list[1]", sum(x[1] for x in self.cache_len_list))
                    # print(" * len cache_len_list", len(self.cache_len_list))
                    # print(" * avg", sum(x[1] for x in self.cache_len_list) / len(self.cache_len_list))
            else:
                self.cache_len_list = [req.get_tuple_tokens(is_busy, self.router_max_new_token_len) for req in current_batch.reqs]
        else:
            if self.adaptive_batchsize_router:
                self._cache_len_lists = [[]]
                self.cache_len_list = self._cache_len_lists[0]   # keep compatibility
            else:
                self.cache_len_list = []

    @calculate_time(show=True, min_cost_ms=0)
    def _can_add_new_req(self, req:Req, is_busy):
        if self.adaptive_batchsize_router:
            need_max_token_nums = []
            for li in self._cache_len_lists:     # TODO: parallel?
                newreq_output_len_sample = random.choice(self.adaptive_batchsize_router._history_req_decode_len)
                print('newreq_output_len_sample', newreq_output_len_sample)
                li.append(req.get_tuple_tokens(is_busy, newreq_output_len_sample))
                need_max_token_num_sample = self._calc_max_token_num_needed(li)
                print(" * need_max_token_num_sample", need_max_token_num_sample)
                need_max_token_nums.append(need_max_token_num_sample)
            print('max(need_max_token_nums)', max(need_max_token_nums))
            print('average(need_max_token_nums)', np.average(need_max_token_nums))
            # need_max_token_num = max(need_max_token_nums)
            need_max_token_num = np.average(need_max_token_nums)

        else:
            self.cache_len_list.append(req.get_tuple_tokens(is_busy, self.router_max_new_token_len)) # hard to analysis
            need_max_token_num = self._calc_max_token_num_needed(self.cache_len_list)

        print(" * need_max_token_num", need_max_token_num)
        if req.req_status in [ReqRunStatus.PAUSED_AND_KVKEEP, ReqRunStatus.PAUSED_AND_OFFLOAD]:
            self.cache_pause_reqs_used_tokens -= req.get_used_tokens()
            self.cache_pause_reqs_num -= 1

        REVERSED = 0.05
        ok_token_num = need_max_token_num < self.max_total_tokens * (1 - REVERSED) - self.cache_pause_reqs_used_tokens - self.prompt_cache_used_tokens
        ok_req_num = len(self.cache_len_list) + self.cache_pause_reqs_num + self.prompt_cache_req_num <= self.running_max_req_size

        if ok_token_num and ok_req_num:
            return True
        else:
            return False

    @calculate_time(show=True, min_cost_ms=0.03)
    def generate_new_batch(self, current_batch:Batch):

        if not self.waiting_req_list:
            return None

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
            if cur_batch_decode_need_tokens + new_batch_first_router_need_tokens + req_first_router_need_tokens <= self.batch_max_tokens \
                and self._can_add_new_req(req, is_busy):
                # and cur_token_ratio < 0.99 and (self._can_add_new_req(req, is_busy) or True) and sum(x[0] for x in self.cache_len_list) < self.max_total_tokens:

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

    def get_max_prefill_token(self, current_batch:Batch):
        used_tokens = self.pause_req_used_tokens
        if current_batch is not None:
            used_tokens += sum(req.get_tuple_tokens(False, self.router_max_new_token_len)[0] for req in current_batch.reqs)
        if self.adaptive_batchsize_router:
            absr = self.adaptive_batchsize_router
            max_prefill_token = absr.max_total_token_num * (1 - absr.PREFILL_RESERVED_TOKEN_RATIO) - used_tokens
            max_prefill_token -= absr.target_running_bs     # lower bond
            max_prefill_token = max(0, max_prefill_token)
            return max_prefill_token
        else:
            raise NotImplementedError
