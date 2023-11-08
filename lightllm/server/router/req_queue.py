from collections import deque
import random
from signal import raise_signal
from typing import List, Union
import uuid
import asyncio
import numpy as np
from typing import List

from lightllm.server.router.ending_predict import EndingPredictState
from ..io_struct import Batch, Req, RunStatus, InferState
from lightllm.utils.infer_utils import  calculate_time

def EMA(inital: float, length: int):
    _val = inital
    def f(update: Union[float, None] = None):
        nonlocal _val
        if update is not None:
            _val = _val * (1 - 1/length) + update * 1/length
        return _val
    return f

class AdaptiveBatchsizeRouter:
    INITAL_TARGET_BS = 100
    HISTORY_LEN = 2048              # steps
    HISTORY_REQ_LEN_EMA_LEN = 50    # reqs

    LOWER_MEM_RATE = 0.95
    UPPER_MEM_RATE = 0.98
    CRITICAL_MEM_RATE = 1

    BALANCE_CHECK_MIN_BS = 30
    BALANCE_MIN_COV = 0.5
    RE_BALANCE_PARTIAL_RATIO = 0.5

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
    # CRITICAL_MEM_RATE = 1

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

        self._step = 0
        self._history_freed = deque()
        self._history_allocated = deque()
        self._history_finished = deque([self.INITAL_TARGET_BS])   # boostrap
        self.recent_finished = self.INITAL_TARGET_BS
        self._history_started = deque()
        self.finished_req_prompt_len_ema = EMA(1024, self.HISTORY_REQ_LEN_EMA_LEN)
        self.finished_req_decode_len_ema = EMA(2048, self.HISTORY_REQ_LEN_EMA_LEN)
        self.new_req_prompt_len_ema = EMA(1024, self.HISTORY_REQ_LEN_EMA_LEN)
        self.running_bs_ema = EMA(self.INITAL_TARGET_BS, self.HISTORY_REQ_LEN_EMA_LEN)
        self.max_total_token_num = max_total_token_num
        self.minimal_bs = 30    # FIXME
        self.is_balance = True
        self.stopped_cause = None
        self.can_prefill = True

        self.target_staged_bs = inital_target_bs    # controlled, decide by policy
        self.target_running_bs = inital_target_bs   # controlled, to make staged_bs ---> target_staged_bs, which also makes staged_bs == target_staged_bs
        self.staged_bs = 0                          # runtime status
        self.running_bs = 0                         # runtime status

    def _total_staged_tokens(self, current_batch: Union[Batch, None], stopped_reqs: List[Req]):
        sum_ = 0
        if current_batch:
            sum_ += sum(req.calcu_used_tokens() for req in current_batch.reqs)
        sum_ += sum(req.calcu_used_tokens() for req in stopped_reqs)
        return sum_

    def _is_balance(self, reqs: List[Req]):
        # print = lambda *x: None

        if len(reqs) < self.BALANCE_CHECK_MIN_BS:
            return True
        decoded_lens = np.array([len(x.output_ids) for x in reqs])
        cov = np.std(decoded_lens) / np.average(decoded_lens)
        print(f"        decoded len average {np.average(decoded_lens)}")
        print(f"        decoded len std {np.std(decoded_lens)}")
        print(f"        decoded len cov {cov}")
        print(f"        history_decode_len_ema {self.finished_req_decode_len_ema()}")
        return cov > self.BALANCE_MIN_COV
        # return np.average(decoded_lens) < history_decode_len_ema * 0.67

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
               current_batch: Union[Batch, None], stopped_reqs: List[Req],
               reqs_pending: int, staged_bs_delta: int,
               evicted_n: int = 0, evicted_size: int = 0):
        # print = lambda *x: None

        self._step += 1

        self.staged_bs = staged_bs
        self.running_bs = running_bs
        self.running_bs_ema(self.running_bs)
        used_tokens = self._total_staged_tokens(current_batch, stopped_reqs)
        rate = used_tokens / self.max_total_token_num

        reqs = stopped_reqs + (current_batch.reqs if current_batch else [])
        self.is_balance = self._is_balance(reqs)



        ### update mem state ###

        if state == InferState.DECODE or state == InferState.PARTIAL_DECODE:
            last_freed = self._last_used_tokens - used_tokens + running_bs - evicted_size   # not include evicted
            last_allocated = running_bs
        elif state == InferState.PREFILL:
            last_freed = 0
            last_allocated = used_tokens - self._last_used_tokens

        self._history_freed.append(last_freed)
        self._history_allocated.append(last_allocated)

        if self._step > self.HISTORY_LEN:
            self._history_freed.popleft()
            self._history_allocated.popleft()

        recent_frees = sum(self._history_freed)
        recent_allocs = sum(self._history_allocated)
        self._recent_freed_allocated_ratio = recent_frees / recent_allocs if recent_allocs else 1
        print(f"    last_freed: {last_freed}, last_allocated: {last_allocated}")
        print(f"    self._recent_freed_allocated_ratio: {self._recent_freed_allocated_ratio}")



        ### update bs history state ###

        n_finished = -staged_bs_delta - evicted_n if state == InferState.DECODE or state == InferState.PARTIAL_DECODE else 0
        n_started = staged_bs_delta if state == InferState.PREFILL else 0

        print(f"    n_finished {n_finished} - n_started {n_started}")

        self._history_finished.append(n_finished)
        self._history_started.append(n_started)

        recent_finished = sum(self._history_finished)
        self.recent_finished = recent_finished
        recent_started = sum(self._history_started)
        self._recent_finished_start_ratio = recent_finished / recent_started if recent_started else 1
        print(f"    recent_finished: {recent_finished}, recent_started: {recent_started}")
        print(f"    self._recent_finished_start_ratio: {self._recent_finished_start_ratio}")

        if self._step > self.HISTORY_LEN:
            self._history_finished.popleft()
            self._history_started.popleft()



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



        ### bs policy ###
        target_staged_delta = 0

        _history_len = len(self._history_finished)
        if rate < self.LOWER_MEM_RATE:
            finish_prob = recent_finished / _history_len
            raise_prob = finish_prob * self.BS_INCREASE_PROB_RATIO
            print(f"    raise_prob = {raise_prob} ({recent_finished} / {_history_len} * {self.BS_INCREASE_PROB_RATIO})")
            target_staged_delta = int(np.random.rand() < raise_prob)
        elif rate < self.UPPER_MEM_RATE:
            pass
        elif rate < self.CRITICAL_MEM_RATE:
            # target_staged_delta = -1
            # target_staged_delta = int(np.random.rand() < 0.2)
            start_prob = recent_started / _history_len
            fall_prob = start_prob * 0.5
            print(f"    fall_prob = {fall_prob} ({recent_started} / {_history_len} * {0.5})")
            if evicted_n:
                print(f"has evict, force fall")
                fall_prob = 0.5
            target_staged_delta = -int(np.random.rand() < fall_prob)
        else:
            # target_staged_delta = -5
            target_staged_delta = -1


        if reqs_pending == 0 or reqs_pending < self.target_staged_bs - self.staged_bs:
            # undersaturation, don't increase bs now
            print(" + limit: delta <= 0, cause: WAIT_NEW_REQ")
            target_staged_delta = min(0, target_staged_delta)

        # if self._recent_freed_allocated_ratio < 0.75:
        #     # too many requests on-going and not finished soon
        #     print(" + limit: delta <= 0, cause: TOO_LESS_FREED")
        #     target_staged_delta = min(0, target_staged_delta)

        # if self._recent_finished_start_ratio < 0.67:
        #     # too many requests started
        #     print(" + limit: delta <= 0, cause: TOO_MANY_STARTS")
        #     target_staged_delta = min(0, target_staged_delta)

        ideal_mem_req_len = self.finished_req_prompt_len_ema() + self.finished_req_decode_len_ema() / 2
        high_bs = self.max_total_token_num * self.HIGH_BS_TOKEN_RATIO / ideal_mem_req_len
        if self.target_staged_bs > high_bs:
            # bs too high
            print(" + limit: delta <= 0, cause: BS_HIGH")
            target_staged_delta = min(0, target_staged_delta)

        low_bs = self.max_total_token_num * self.LOWER_MEM_RATE \
            / (self.finished_req_prompt_len_ema() + self.finished_req_decode_len_ema()) * 0.9
        if self.target_staged_bs < low_bs:
        # if self.target_staged_bs < low_bs and target_staged_delta == 1:
            # bs too low
            print(" + limit: delta 1 ~ 10, cause: BS_LOW")
            target_staged_delta = max(int(low_bs) - self.target_staged_bs, 10)

        if self.target_staged_bs < self.staged_bs:
        # if self.target_staged_bs < self.staged_bs * 0.8:
            # staged not lowered yet, not further decrease the bs
            print(" + limit: delta >= 0, cause: WAIT_RADUCE")
            target_staged_delta = max(0, target_staged_delta)

        print(f" = final target_staged_delta: {target_staged_delta}")
        self.target_staged_bs += target_staged_delta
        self.target_staged_bs = max(1.0, self.target_staged_bs)

        if self.staged_bs <= self.target_staged_bs:
            self.target_running_bs = self.target_staged_bs
        else:
            to_reduce = self.staged_bs - self.target_staged_bs
            running_bs_reduce_factor = 1 if rate < self.CRITICAL_MEM_RATE else 2
            # running_bs_reduce_factor = 1 if rate < self.CRITICAL_MEM_RATE else 10
            self.target_running_bs = max(1, self.staged_bs - to_reduce * running_bs_reduce_factor)

            if self.target_running_bs < self.minimal_bs:
                print(" + limit: target_running_bs >= minimal_bs, cause: MINIMAL_BS")
                self.target_running_bs = self.minimal_bs


        self._last_used_tokens = used_tokens

        if self.running_bs != 0:
            print(f"AdaptiveBatchsizeRouter usage {rate*100:.2f}% ({used_tokens} / {self.max_total_token_num})")
            print(self)

    def __repr__(self):
        return f"AdaptiveBatchsizeRouter(target_staged_bs={self.target_staged_bs}, target_running_bs={self.target_running_bs}, staged_bs={self.staged_bs}, running_bs={self.running_bs})"


class ReqQueue:

    def __init__(self, max_total_tokens, batch_max_tokens, running_max_req_size, max_new_token_len, token_ratio, \
                 adaptive_batchsize_router: Union[AdaptiveBatchsizeRouter, None] = None, ending_predict_state: Union[EndingPredictState, None] = None) -> None:
        self.max_total_tokens = max_total_tokens
        assert batch_max_tokens is not None
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.waiting_req_list: List[Req] = []
        self.token_ratio = token_ratio
        self.max_new_token_len = max_new_token_len
        self.pending_count = 0

        self.adaptive_batchsize_router = adaptive_batchsize_router
        if adaptive_batchsize_router:
            self.ending_predict_state = ending_predict_state
            self._last_try_step = -100
            self._rate_accumulate = 0

    def append(self, req: Req):
        self.waiting_req_list.append(req)
        return

    def _init_cache_list(self, current_batch:Batch, token_ratio=0.):
        self.cache_empty = None
        if current_batch is not None:
            self.cache_len_list = [(req.input_len + len(req.output_ids) - 1, max(1, self.calcu_max_new_token(req, token_ratio) - len(req.output_ids))) for req in current_batch.reqs]
            self.cache_empty = True if len(self.cache_len_list) == 0 else False
        else:
            self.cache_len_list = []

    def _can_add_new_req(self, req, token_ratio=0.):
        self.cache_len_list.append((req.calcu_need_tokens(), max(1, self.calcu_max_new_token(req, token_ratio) - len(req.output_ids) - 1))) # hard to analysis
        self.cache_len_list.sort(key=lambda x: -x[1])
        if self.adaptive_batchsize_router:
            if len(self.cache_len_list) <= self.adaptive_batchsize_router.target_running_bs:
                if sum(e[0] for e in self.cache_len_list) < self.max_total_tokens * self.adaptive_batchsize_router.CRITICAL_MEM_RATE:
                    return True
                else:
                    print(f' + prefill will limited by vram at bs {len(self.cache_len_list)}')
                    # may reduce target_staged_bs if memory not enough for new refill
                    # self.adaptive_batchsize_router.target_staged_bs = min(len(self.cache_len_list) - 1, self.adaptive_batchsize_router.target_staged_bs)
                    pass
            return False

        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        # assert left_out_len_array.min() >= 0
        has_run_len_array = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(self.cache_len_list) + 1, 1)
        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()
        if need_max_token_num <= (self.max_total_tokens - self.stoped_token_num) and len(self.cache_len_list) <= self.running_max_req_size or self.cache_empty:
            return True
        else:
            return False

    def generate_new_batch(self, current_batch:Batch, token_ratio=0.):
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None

        self._init_cache_list(current_batch, token_ratio)
        can_run_list = []
        new_batch_total_tokens = 0
        aborted_count = 0
        restore = False
        self.stoped_token_num = self.calcu_stopd_tokens()
        for req in self.waiting_req_list:
            if req.aborted:
                aborted_count += 1
                continue
            if restore and self.pending_count == 0:
                break
            if self._can_add_new_req(req, token_ratio) and new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                if self.pending_count > 0:
                    restore = True if req.runstatus == RunStatus.PAUSED else False
                    self.pending_count -= 1
                can_run_list.append(req)
                new_batch_total_tokens += req.input_len
            else:
                break

        if self.adaptive_batchsize_router and self.adaptive_batchsize_router.RANDOM_SLOW_ADD_NEW_REQ and can_run_list:
            absr = self.adaptive_batchsize_router
            old_len = len(can_run_list)

            # # free_ratio = old_len / absr.target_staged_bs
            # # prefill_len = max(1, round((1 - free_ratio**2) * old_len))
            # step_passed = absr._step - self._last_try_step
            # print(f"step_passed {step_passed}")
            # print(f"absr.target_staged_bs / absr.running_bs_ema(): {absr.target_staged_bs} / {absr.running_bs_ema()}")
            # self._last_try_step = absr._step
            # rand_thr = max(old_len, absr.recent_finished)/absr.HISTORY_LEN * step_passed * (absr.target_staged_bs / absr.running_bs_ema())
            # print(rand_thr)
            # new_len = int(random.random() < rand_thr) * max(1, round(rand_thr))
            # # old_len = len(can_run_list)
            # # new_len = sum(random.random() < absr.recent_finished/absr.HISTORY_LEN for _ in range(old_len))
            # # can_run_list = can_run_list[:new_len]
            # free_ratio = old_len / absr.target_staged_bs
            # prefill_len = max(1, round((1 - free_ratio**2) * old_len))

            predict_finished_per_step = absr.target_staged_bs / absr.finished_req_decode_len_ema()
            step_passed = absr._step - self._last_try_step
            self._last_try_step = absr._step
            self._rate_accumulate += predict_finished_per_step * step_passed
            new_len = int(self._rate_accumulate)    # if > 1
            self._rate_accumulate -= int(self._rate_accumulate)
            can_run_list = can_run_list[:new_len]

            print(f"predict_finished_per_step {predict_finished_per_step} = {absr.target_staged_bs} / {absr.finished_req_decode_len_ema()}")
            print(f"step_passed {step_passed}")
            print(f"self._rate_accumulate {self._rate_accumulate} = old {self._rate_accumulate} + acc {predict_finished_per_step * step_passed}")
            print(new_len)
            print(f"RANDOM_SLOW_ADD_NEW_REQ: can_run_list len {old_len} -> {new_len}")

        if len(can_run_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            return restore, new_batch
        else:
            return restore, None

    def calcu_stopd_tokens(self):
        stopd_tokens = 0
        for i in range(self.pending_count):
            stopd_tokens += self.waiting_req_list[i].calcu_used_tokens()
        return stopd_tokens

    def insert(self, req):
        self.waiting_req_list.insert(self.pending_count, req)
        self.pending_count += 1

    def is_waiting_list_empty(self):
        return len(self.waiting_req_list) == 0

    def has_pending_reqs(self):
        if self.pending_count > 0:
            return True
        return False

    def calcu_max_new_token(self, req, token_ratio):
        if req.max_output_len < self.max_new_token_len or token_ratio > self.token_ratio:
            return req.max_output_len
        return self.max_new_token_len
