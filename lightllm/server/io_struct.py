import enum
from lightllm.server.router.ending_predict import EndingPredictState, EndingPredictItem
from .sampling_params import SamplingParams
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import enum

class RunStatus(enum.Enum):
    NORMAL = 0
    PAUSED = 1
    OFFLOAD = 2

class Req:
    def __init__(self, request_id, prompt_ids, sample_params: SamplingParams, ending_predict_state: Union[EndingPredictState, None] = None):
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.input_len = len(prompt_ids)
        self.max_output_len = sample_params.max_new_tokens
        self.sample_params = sample_params
        self.output_ids = []
        self.output_metadata_list = []
        self.has_generate_finished = False
        self.aborted = False
        self.runstatus = RunStatus.NORMAL
        self.ending_predict = EndingPredictItem(global_state=ending_predict_state) if ending_predict_state else None

    def to_rpc_obj(self):
        return {"request_id": self.request_id,
                "input_id": self.prompt_ids[:],
                "output_len": self.max_output_len,
                "offload": True if self.runstatus == RunStatus.OFFLOAD else False,
                "sampling_param": self.sample_params.to_dict() }

    def to_req_detokenization_state(self):
        out = ReqDetokenizationState(self.request_id, self.prompt_ids, self.max_output_len, self.sample_params.ignore_eos)
        if self.output_metadata_list:
            out.gen_metadata.update(self.output_metadata_list[-1])
        return out

    def stop_sequences_matched(self):
        for stop_token_ids in self.sample_params.stop_sequences:
            stop_len = len(stop_token_ids)
            if stop_len > 0:
                if len(self.output_ids) >= stop_len:
                    if all(self.output_ids[-(stop_len - i)] == stop_token_ids[i] for i in range(stop_len)):
                        return True
        return False

    def calcu_used_tokens(self):
        if self.runstatus == RunStatus.OFFLOAD:
            return len(self.output_ids) - 1
        return self.input_len + len(self.output_ids) - 1

    def calcu_need_tokens(self):
        if self.runstatus == RunStatus.PAUSED:
            return 0
        return self.input_len

    def __repr__(self):
        return (f"request_id(n={self.request_id}, "
                f"prompt_ids={self.prompt_ids}, ")


class ReqDetokenizationState:
    def __init__(
        self,
        request_id: str,
        prompt_ids: List[int],
        max_output_len: int,
        ignore_eos: bool,
    ) -> None:
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.output_ids = []
        self.output_tokens = []
        self.output_str = ""
        self.sub_texts = []
        self.current_sub_text = []
        self.max_output_len = max_output_len
        self.ignore_eos = ignore_eos
        self.gen_metadata = {}

class Batch:
    def __init__(self, batch_id, reqs: List[Req]):
        self.batch_id = batch_id
        self.reqs = reqs
        self.id_to_reqs = {req.request_id: req for req in reqs}

    def input_tokens(self):
        batch_input_tokens = 0
        for req in self.reqs:
            batch_input_tokens += req.input_len
        return batch_input_tokens

    def output_tokens(self):
        batch_output_tokens = 0
        for req in self.reqs:
            batch_output_tokens += len(req.output_ids) - 1
        return batch_output_tokens

    def calcu_max_tokens(self):
        tokens = 0
        for req in self.reqs:
            tokens += req.input_len + req.max_output_len - 1
        return tokens

    def calcu_used_tokens(self):
        tokens_num = 0
        for req in self.reqs:
            tokens_num += req.calcu_used_tokens()
        return tokens_num

    def mark_finished_req(self, eos_id):
        has_new_finish = False
        for req in self.reqs:
            if req.stop_sequences_matched():
                req.has_generate_finished = True
                has_new_finish = True
            if req.output_ids[-1] == eos_id and req.sample_params.ignore_eos == False:
                req.has_generate_finished = True
                has_new_finish = True
            if len(req.output_ids) >= req.max_output_len or req.aborted:
                req.has_generate_finished = True
                has_new_finish = True
        return has_new_finish

    def filter_finished(self):
        unfinished_req, finished_req = [], []
        for req in self.reqs:
            if not req.has_generate_finished:
                unfinished_req.append(req)
            else:
                finished_req.append(req)
        self.reqs = unfinished_req
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return finished_req

    def is_clear(self):
        return len(self.reqs) == 0

    def merge(self, mini_batch):
        for _req in mini_batch.reqs:
            self.reqs.append(_req)
        self.id_to_reqs = {req.request_id: req for req in self.reqs}
        return

    def pop(self, request_id):
        req = None
        for index, req in enumerate(self.reqs):
            if req.request_id == request_id:
                req = self.reqs.pop(index)
                self.id_to_reqs.pop(request_id)
                req.runstatus = RunStatus.PAUSED
                break
        return req

    def offload(self, request_id):
        req = None
        for index, req in enumerate(self.reqs):
            if req.request_id == request_id:
                req = self.reqs.pop(index)
                self.id_to_reqs.pop(request_id)
                req.runstatus = RunStatus.OFFLOAD
                break
        return req

    def update_run_status(self, runstatus: RunStatus):
        for req in self.reqs:
            req.runstatus = runstatus
        return

    def __repr__(self):
        return (f"batch_id={self.batch_id}, "
                f"reqs={self.reqs}, ")

class BatchTokenIdOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, int, Dict, bool, bool]] = []  # [req_id, new_token_id, gen_metadata, finished_state, abort_state]

class BatchStrOut:
    def __init__(self):
        self.reqs_infs: List[Tuple[str, str, Dict, bool, bool]] = [] # [req_id, token_str, gen_metadata, finished_state, abort_state]

class AbortReq:
    def __init__(self, req_id):
        self.req_id = req_id

class InferState(enum.Enum):
    IDLE = enum.auto()
    PREFILL = enum.auto()
    DECODE = enum.auto()
    PARTIAL_DECODE = enum.auto()
