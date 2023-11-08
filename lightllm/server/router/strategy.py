import random
import uuid
import numpy as np
from typing import Dict, List, Tuple
from lightllm.server.io_struct import Batch, Req
from lightllm.server.router.req_queue import AdaptiveBatchsizeRouter


class WaitStrategy:

    def __init__(self, max_total_token_num, *args, **kwargs) -> None:
        self.max_total_token_num = max_total_token_num
        self.moving_max_new_tokens = False
        self.stopped_req_list: List[Req] = []

    def append_to_stopped(self, req):
        self.stopped_req_list.append(req)
        return

    def _order(self, req: Req):
        raise NotImplementedError

    def _ordering_index(self, batch: Batch):
        req_index = range(len(batch.reqs))
        return sorted(req_index, key=lambda i: self._order(batch.reqs[i]), reverse=True)

    def _selection(self, batch: Batch):
        raise NotImplementedError

    def restore_batch(self, batch: Batch):
        raise NotImplementedError

    def select_reqs(self, batch: Batch):
        request_ids = self._selection(batch)
        # print("select_reqs", request_ids)
        req_id_list = []
        for request_id in request_ids:
            req = batch.pop(request_id)
            assert req is not None
            self.append_to_stopped(req)
            req_id_list.append(request_id)
        if len(batch.reqs)  == 0:
            raise RuntimeError(f"No enough memory to run, current batch size {len(req_id_list)}")
        return req_id_list

    def _calcu_max_tokens(self, req_len_list: List[Tuple[int, int]]):
        if not req_len_list:
            return -1
        left_len = np.array([e[1] for e in req_len_list])
        run_len = np.array([e[0] for e in req_len_list])
        cum_run_len = np.cumsum(run_len)
        size_array = np.arange(1, len(req_len_list) + 1, 1)
        return (left_len * size_array + cum_run_len).max()

    def can_decode(self, batch: Batch):
        raise NotImplementedError

    def is_stopped_list_empty(self):
        return len(self.stopped_req_list) == 0

    def calcu_stopd_prompt_tokens(self):
        stopd_prompt_tokens = 0
        for req in self.stopped_req_list:
            stopd_prompt_tokens += req.input_len
        return stopd_prompt_tokens

    def calcu_stopd_output_tokens(self):
        stopd_output_tokens = 0
        for req in self.stopped_req_list:
            stopd_output_tokens += len(req.output_ids) - 1
        return stopd_output_tokens

    def calcu_stopd_tokens(self):
        raise NotImplementedError


class SJF_ABSR(WaitStrategy):
    """ Shortest Job First for AdaptiveBatchsizeRouter
    """

    def __init__(self, max_total_token_num, *args, **kwargs) -> None:
        super().__init__(max_total_token_num, *args, **kwargs)
        self.absr: AdaptiveBatchsizeRouter = kwargs['absr']  # adaptive_batchsize_router

    def _ordering_index(self, reqs: List[Req]):
        req_index = range(len(reqs))
        return sorted(req_index, key=lambda i: self._order(reqs[i]), reverse=True)

    def _order(self, req: Req):
        return req.max_output_len - len(req.output_ids)

    def calcu_stopd_tokens(self):
        stopd_tokens = 0
        for req in self.stopped_req_list:
            stopd_tokens += req.calcu_used_tokens()
        return stopd_tokens

    def _select_evict(self, batch: Batch, evict_size: int):
        req_index = range(len(batch.reqs))
        # sorted_index = sorted(req_index, key=lambda i: batch.reqs[i].calcu_used_tokens())
        sorted_index = list(req_index)
        random.shuffle(sorted_index)    # random
        n_evict = 0
        while evict_size > 0:
            evict_size -= batch.reqs[sorted_index[n_evict]].calcu_used_tokens()
            evict_size -= 1   # also 1 req less to decode
            n_evict += 1
        return [batch.reqs[i] for i in sorted_index[:n_evict]]

    def can_decode(self, batch: Batch):
        # limit_by_target_running_bs = self.absr.staged_bs > self.absr.target_running_bs + 10
        limit_by_target_running_bs = False  # never partial decode when low-ram

        remain_tokens = self.max_total_token_num - (batch.calcu_used_tokens() + self.calcu_stopd_tokens())
        # print(f"batch.calcu_used_tokens() + self.calcu_stopd_tokens() {batch.calcu_used_tokens()} + {self.calcu_stopd_tokens()}")
        bs_to_run = len(batch.reqs) if not limit_by_target_running_bs else min(len(batch.reqs), self.absr.target_running_bs)
        if remain_tokens < bs_to_run:
            # not enough for current decode

            evict_size = bs_to_run - remain_tokens
            self._reqs_to_evict = self._select_evict(batch, evict_size)

            print(f"no enough vram for decode once, minimal evict n_evict = {len(self._reqs_to_evict)}")

            maximum_running_bs = bs_to_run - len(self._reqs_to_evict)
            self.absr.target_running_bs = maximum_running_bs if not limit_by_target_running_bs else min(self.absr.target_running_bs, maximum_running_bs)
            self.absr.stopped_cause = 'low_mem+evict'
            return False
        if limit_by_target_running_bs:
            self.absr.stopped_cause = 'low_mem'
            return False
        if not self.absr.is_balance:
            self.absr.stopped_cause = 'imbalance'
            return False
        self.absr.stopped_cause = None
        return True

    def restore_batch(self, batch: Batch):
        can_restore_list = []
        for req in self.stopped_req_list:
            can_restore_list.append(req)
        self.stopped_req_list = []
        if len(can_restore_list) != 0:
            new_batch = Batch(uuid.uuid4().hex, can_restore_list)
            return new_batch

    def _selection(self, batch: Batch):
        'select reqs to evict or stop'

        all_reqs = batch.reqs
        evict_reqs = []
        if self.absr.stopped_cause == 'low_mem+evict':
            assert self._reqs_to_evict
            evict_set = set(self._reqs_to_evict)
            all_reqs = [r for r in all_reqs if r not in evict_set]
            evict_reqs = [r.request_id for r in self._reqs_to_evict]

        if self.absr.stopped_cause == 'low_mem' or self.absr.stopped_cause == 'low_mem+evict':
            # remain_tokens = self.max_total_token_num - (batch.calcu_used_tokens() + self.calcu_stopd_tokens())
            # sorted_index = self._ordering_index(batch)
            # req_len_list = [(batch.reqs[index].input_len + len(batch.reqs[index].output_ids) - 1,
            #             self.ema.get_max_output_len(batch.reqs[index]) - len(batch.reqs[index].output_ids)) for index in sorted_index]
            # for i in range(len(req_len_list)):
            #     top_req = req_len_list.pop(0)
            #     update_token_num = remain_tokens - top_req[0]
            #     if self._calcu_max_tokens(req_len_list) <= update_token_num:
            #         break
            # select_index = sorted_index[:(i+1)]
            # max_necessary_stop = [batch.reqs[sorted_index[j]].request_id for j in select_index]

            remain_tokens = self.max_total_token_num
            sorted_index = self._ordering_index(all_reqs)
            guaranteed_bs = 0
            for idx in reversed(sorted_index):
                remain_tokens -= all_reqs[idx].input_len + all_reqs[idx].max_output_len
                if remain_tokens < 0:
                    break
                guaranteed_bs += 1
            guaranteed_bs = round(guaranteed_bs * self.absr.MINIMAL_BS_RATIO)
            assert guaranteed_bs > 0
            select_index = sorted_index[:-guaranteed_bs]
            min_necessary_stop = [all_reqs[sorted_index[j]].request_id for j in select_index]

            num_to_stop = len(all_reqs) - self.absr.target_running_bs
            assert num_to_stop >= 0
            print(f"    len(all_reqs) {len(all_reqs)}, self.absr.target_running_bs {self.absr.target_running_bs}")
            print(f"    len(min_necessary_stop) {len(min_necessary_stop)}, num_to_stop {num_to_stop}")
            if len(min_necessary_stop) < num_to_stop:
                self.absr.target_staged_bs = max(guaranteed_bs, self.absr.target_staged_bs)
                return evict_reqs + min_necessary_stop
            else:
                selected_index = sorted_index[:num_to_stop]
                return evict_reqs + [all_reqs[i].request_id for i in selected_index]

        if self.absr.stopped_cause == 'imbalance':
            # TODO: 加个比例参数，避免饥饿
            n_reqs = len(all_reqs)
            to_select = int(n_reqs * (1 - self.absr.RE_BALANCE_PARTIAL_RATIO))
            sorted_index = self._ordering_index(all_reqs)
            triangular_distribution_p = np.arange(n_reqs-1, -1, -1) / (n_reqs*(n_reqs-1)/2)
            selected_sorted = np.random.choice(n_reqs, to_select, replace=False, p=triangular_distribution_p)
            return evict_reqs + [all_reqs[sorted_index[i]].request_id for i in selected_sorted]

    def select_evict_reqs(self, batch: Batch, evict_size: int):
        print(f"  select_evict_reqs, evict_size: {evict_size}")

        # sorted_reqs = sorted(self.stopped_req_list, key=req_vram)
        # sorted_reqs = sorted(self.stopped_req_list, key=self._order, reverse=True)
        # to_evict: List[Req] = []
        # for req in sorted_reqs:
        #     if evict_size <= 0:
        #         break
        #     to_evict.append(req)
        #     evict_size -= req.calcu_used_tokens()
        # assert evict_size <= 0, f"evict failed, stopped reqs{sorted_reqs}"
        to_evict = self._reqs_to_evict
        self._reqs_to_evict = None

        print(f"  will evict {len(to_evict)} reqs, size {sum(r.calcu_used_tokens() for r in to_evict)}")
        for req in to_evict:
            if req.request_id in batch.id_to_reqs:
                batch.pop(req.request_id)
        to_evict_set = set(to_evict)
        self.stopped_req_list = [r for r in self.stopped_req_list if r not in to_evict_set]
        return to_evict
