#!/usr/bin/env python3
"""Step-level mock simulator for the production LightLLM FlexTP schedulers.

Unlike ``flex_tp_batching_sim.py``, this program does not reimplement the
scheduling decisions.  It imports and executes:

* ``FlexTPSelectorV3/V4/V5/V6/V7/V8/V9/V10`` for global admission and TP placement;
* ``FlexTPNaiveSelector`` for the overlapping 4000-token threshold baseline;
* ``FixedTPSingleSelector`` for the single-TP fixed baselines;
* ``ChunkedPrefillQueue`` for each Prefill instance's local admission;
* ``QueueForPDDecode`` for the Decode instance's local admission.

Only the wall clock, websocket/shared-memory request objects, and GPU model
forward are mocked.  Model forwards are advanced as discrete steps with the
same router interval, chunk size, and batch token limit as the referenced
cluster scripts. Decode is non-bottlenecking by default; ``--fake-decode``
retains the KV transfer event without running Decode steps. Timing is a
replaceable model, so simulated latency must not be presented as a GPU
measurement.

Examples, run from the repository root::

    # Compare ours and naive on loop4-sg3.sh's ServeGen mm-image sweep.
    python test/benchmark/service/flex_tp_step_sim.py \
        --dataset servegen-mm-image --rates 10,9,8,7,6,5,4,3,2,1 \
        --duration 180 --output-dir _/flex_tp_step_sim/servegen

    # Compare ours and naive on loop3-sub5.sh's simple.1-5 workload.
    python test/benchmark/service/flex_tp_step_sim.py \
        --dataset synthetic-5pct --rates 20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1 \
        --num-prompts 2000 --output-dir _/flex_tp_step_sim/simple-1-5

    # Small deterministic integration check, including multi-request batching
    # and a request that needs three 8192-token Prefill chunks.
    python test/benchmark/service/flex_tp_step_sim.py --self-test
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import csv
import dataclasses
import heapq
import json
import logging
import math
import os
import pickle
import random
import statistics
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

# Running a script by path puts test/benchmark/service at sys.path[0].  Pin the
# current checkout ahead of any system-wide /lightllm installation so the
# simulator always exercises the code under test.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import lightllm.utils.device_utils as device_utils_module

# Importing the production request queues also imports Triton autotuners.  They
# only need a cache namespace here, so avoid CUDA discovery in this mock process
# and use the target cluster's device name.  No CUDA kernel is invoked.
device_utils_module.get_current_device_name = lambda: "NVIDIA_H200"

import lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v3 as selector_module
import lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v6 as selector_v6_module
import lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v7 as selector_v7_module
import lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v8 as selector_v8_module
import lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v9 as selector_v9_module
import lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v13 as selector_v13_module
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_naive import (
    FlexTPNaiveSelector,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.pd_selector import PDSelector
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v3 import (
    FlexTPSelectorV3,
    V3Pending,
)
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v4 import FlexTPSelectorV4
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v5 import FlexTPSelectorV5
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v6 import FlexTPSelectorV6
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v7 import FlexTPSelectorV7
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v8 import FlexTPSelectorV8
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v9 import FlexTPSelectorV9
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v10 import FlexTPSelectorV10
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v11 import FlexTPSelectorV11
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v12 import FlexTPSelectorV12
from lightllm.server.httpserver_for_pd_master.pd_selector.flex_tp_selector_v13 import FlexTPSelectorV13
from lightllm.server.httpserver_for_pd_master.pd_bundle import PrefillBundleDispatcher
from lightllm.server.pd_io_struct import ObjType, PD_Client_Obj
from lightllm.server.router.batch import Batch
from lightllm.server.router.req_queue import base_queue as base_queue_module
from lightllm.server.router.req_queue.chunked_prefill.impl import ChunkedPrefillQueue
from lightllm.server.router.req_queue.chunked_prefill.impl_for_pd_decode import QueueForPDDecode
from lightllm.common.basemodel.infer_lock import g_router_lock


EPS = 1e-9
MAX_REQ_TOTAL_TOKENS = 65000

# BaseQueue normally reads the launched model's config through
# LIGHTLLM_START_ARGS only to reserve fixed prompt-cache KV.  There is no model
# process or fixed prompt cache in this mock runtime, so its exact equivalent is
# zero.  All queue admission code remains the production implementation.
base_queue_module.get_fixed_kv_len = lambda: 0
g_router_lock.obj = threading.RLock()


@dataclass
class WorkloadRequest:
    request_id: int
    arrival_s: float
    input_tokens: int
    output_tokens: int
    source: str
    is_long: bool = False


@dataclass
class RequestResult:
    workload: WorkloadRequest
    status: str = "new"
    instance: Optional[str] = None
    tp_size: Optional[int] = None
    global_admit_s: Optional[float] = None
    predicted_prefill_finish_s: Optional[float] = None
    bundle_id: Optional[int] = None
    worker_enqueue_s: Optional[float] = None
    first_prefill_step_s: Optional[float] = None
    prefill_finish_s: Optional[float] = None
    decode_enqueue_s: Optional[float] = None
    finish_s: Optional[float] = None
    rejection_reason: Optional[str] = None
    prefill_steps: int = 0
    decode_steps: int = 0

    @property
    def ttft_s(self) -> Optional[float]:
        if self.prefill_finish_s is None:
            return None
        return self.prefill_finish_s - self.workload.arrival_s

    @property
    def e2e_s(self) -> Optional[float]:
        if self.finish_s is None:
            return None
        return self.finish_s - self.workload.arrival_s


@dataclass
class StepRecord:
    stage: str
    instance: str
    tp_size: int
    step_index: int
    start_s: float
    end_s: float
    exclusive_work_s: float
    effective_slowdown: float
    request_ids: List[int]
    token_counts: List[int]
    batch_tokens: int
    waiting_at_start: int
    running_at_start: int


@dataclass
class BundleRecord:
    bundle_id: int
    instance: str
    dispatch_s: float
    request_ids: List[int]
    input_tokens: int
    transport: str = "bundle"


@dataclass
class SimulationConfig:
    scheduler: str = "ours"
    naive_length_threshold: int = 4000
    mps_overlap_slowdown: float = 2.0
    slo_ttft_s: float = 3.0
    bundle_window_s: float = 0.020
    bundle_token_cap: int = 8192
    bundle_token_trigger: int = 4096
    max_inflight: int = 64
    instance_token_credit: int = 16384
    prediction_margin_s: float = 0.080
    replan_interval_s: float = 0.050
    schedule_interval_s: float = 0.005
    worker_report_interval_s: float = 0.200
    chunked_prefill_size: int = 8192
    prefill_batch_max_tokens: int = 16384
    max_total_tokens: int = 70000
    running_max_req_size: int = 1000
    router_token_ratio: float = 0.80
    router_max_wait_tokens: int = 1
    kv_transfer_fixed_s: float = 0.020
    kv_transfer_us_per_token: float = 0.0
    decode_step_base_ms: float = 6.0
    decode_step_per_request_ms: float = 0.08
    decode_step_per_context_token_us: float = 0.0
    simulate_decode: bool = False
    fake_decode: bool = False
    latency_scale: float = 1.0
    overload_policy: str = "best_effort"
    v5_short_overlap_backlog_trigger: int = 3
    v5_urgent_slack_ratio: float = 0.05
    v6_request_utility_tokens: float = 2000.0
    v7_class_quantum_tokens: int = 4096
    v7_conflict_price_weight: float = 0.35
    v7_urgency_weight: float = 2.0
    v8_epoch_ms: float = 50.0
    v8_epoch_token_budget: int = 16384
    v8_min_lane_quota: int = 2048
    v8_deadline_pressure_weight: float = 2.0
    v9_aging_interval_ms: float = 150.0
    v9_interactive_tokens: int = 1024
    v9_short_weight: float = 1.0
    v9_long_weight: float = 1.0
    v10_short_weight: float = 1.0
    v10_long_weight: float = 1.0
    v10_slack_weight: float = 1.0
    v10_overlap_slack_ratio: float = 0.10
    v11_short_weight: float = 1.0
    v11_long_weight: float = 1.0
    v11_slack_weight: float = 1.0
    v11_overlap_slack_ratio: float = 0.10
    v11_routing_cost_weight: float = 0.50
    v12_routing_cost_weight: float = 2.0
    v12_tp4_service_ratio_limit: float = 0.60
    v13_latency_scale: float = 1.0
    v13_long_threshold: int = 12000
    v13_routing_cost_weight: float = 2.0
    v13_tp4_service_ratio_limit: float = 0.60
    v13_tp4_pressure_threshold: float = 1.0
    max_drain_s: float = 3600.0


def _mode_slowdowns(overlap_slowdown: float) -> Dict[Tuple[int, Tuple[int, ...]], float]:
    factor = max(1.0, float(overlap_slowdown))
    return {
        (2, (2, 4)): factor,
        (4, (2, 4)): factor,
        (4, (2, 2, 4)): factor,
    }


class VirtualClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now


class VirtualTimeBundleDispatcher(PrefillBundleDispatcher):
    """Production bundle packing with its wall-clock timer externally driven."""

    async def _delayed_flush(self, p_node: PD_Client_Obj) -> None:
        # The discrete-event loop calls the production _flush_node exactly at
        # batch_window_s. Returning here prevents a real wall-clock sleep.
        return


class MockBundleWebsocket:
    def __init__(self, node_key: str, captured: Deque[Tuple[str, tuple]]) -> None:
        self.node_key = node_key
        self.captured = captured

    async def send_bytes(self, data: bytes) -> None:
        self.captured.append((self.node_key, pickle.loads(data)))


class MockSharedTokenLoad:
    def __init__(self) -> None:
        self.frozen = collections.defaultdict(int)
        self.estimated = collections.defaultdict(int)
        self.dynamic = collections.defaultdict(float)
        self.current = collections.defaultdict(float)

    def get_frozened_token_count(self, dp_index: int) -> int:
        return self.frozen[dp_index]

    def set_estimated_peak_token_count(self, value: int, dp_index: int) -> None:
        self.estimated[dp_index] = int(value)

    def get_estimated_peak_token_count(self, dp_index: int) -> int:
        return self.estimated[dp_index]

    def set_dynamic_max_load(self, value: float, dp_index: int) -> None:
        self.dynamic[dp_index] = float(value)

    def get_dynamic_max_load(self, dp_index: int) -> float:
        return self.dynamic[dp_index]

    def set_current_load(self, value: float, dp_index: int) -> None:
        self.current[dp_index] = float(value)

    def need_update_dynamic_max_load(self) -> bool:
        return True


class MockShmReqManager:
    def put_back_req_obj(self, req) -> None:
        return None


class MockRouter:
    def __init__(self, max_total_tokens: int) -> None:
        self.max_total_token_num = int(max_total_tokens)
        self.shared_token_load = MockSharedTokenLoad()
        self.router_statics = SimpleNamespace(ema_req_out_len=1)
        self.shm_req_manager = MockShmReqManager()
        self._running_provider = lambda: ()

    def set_running_provider(self, provider) -> None:
        self._running_provider = provider

    def get_used_tokens(self, dp_index: int) -> int:
        return sum(max(0, int(req.shm_cur_kv_len)) for req in self._running_provider())


@dataclass
class MockSamplingParams:
    max_new_tokens: int
    ignore_eos: bool = True
    suggested_dp_index: int = -1


class MockReq:
    """Minimal shared-memory request surface consumed by real Router queues."""

    def __init__(
        self,
        result: RequestResult,
        *,
        stage: str,
        chunked_prefill_size: int,
        router_max_wait_tokens: int,
        max_new_tokens: int,
    ) -> None:
        self.result = result
        self.request_id = result.workload.request_id
        self.index_in_shm_mem = self.request_id
        self.input_len = result.workload.input_tokens
        self.stage = stage
        self.chunked_prefill_size = int(chunked_prefill_size)
        self.router_max_wait_tokens = int(router_max_wait_tokens)
        self.sample_params = MockSamplingParams(max_new_tokens=max(1, int(max_new_tokens)))
        self.is_aborted = False
        self.is_paused = False
        self.shm_cur_kv_len = 0
        self.shm_cur_output_len = 0
        self.remaining_output_tokens = max(0, int(max_new_tokens))

    def get_tuple_tokens(self, is_busy: bool, ema_req_out_len: int) -> Tuple[int, int]:
        has_out_len = self.shm_cur_output_len
        if self.sample_params.ignore_eos or is_busy:
            max_new = self.sample_params.max_new_tokens
        else:
            max_new = min(
                self.sample_params.max_new_tokens,
                max(int(1.1 * has_out_len), int(ema_req_out_len)),
            )
        a_len = max(self.input_len + has_out_len + 1, self.shm_cur_kv_len + 1)
        remaining = max(0, self.input_len + has_out_len - self.shm_cur_kv_len)
        chunk_count = (remaining + self.chunked_prefill_size - 1) // self.chunked_prefill_size
        b_len = (
            chunk_count * (self.router_max_wait_tokens + 1)
            + max_new
            - has_out_len
            - 1
        )
        # Keep the production ChunkedPrefillReq's conservative padding.
        return a_len, max(0, b_len) + 16

    def get_first_router_need_tokens(self) -> int:
        return min(self.input_len + self.shm_cur_output_len, self.chunked_prefill_size)

    def get_decode_need_tokens(self) -> int:
        return min(
            max(0, self.input_len + self.shm_cur_output_len - self.shm_cur_kv_len),
            self.chunked_prefill_size,
        )


@dataclass
class ActiveStep:
    stage: str
    instance: str
    tp_size: int
    step_index: int
    start_s: float
    requests: List[MockReq]
    token_counts: List[int]
    exclusive_work_s: float
    remaining_work_s: float
    waiting_at_start: int
    running_at_start: int
    wall_elapsed_s: float = 0.0


def _queue_args(config: SimulationConfig, *, batch_max_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        max_total_token_num=config.max_total_tokens,
        batch_max_tokens=int(batch_max_tokens),
        running_max_req_size=config.running_max_req_size,
        router_token_ratio=config.router_token_ratio,
        enable_cpu_cache=False,
    )


class PrefillInstanceHarness:
    def __init__(self, node: PD_Client_Obj, selector_state, config: SimulationConfig) -> None:
        self.node = node
        self.selector_state = selector_state
        self.name = node.client_ip_port
        self.tp_size = int(node.start_args["tp"])
        self.config = config
        self.router = MockRouter(config.max_total_tokens)
        self.queue = ChunkedPrefillQueue(
            _queue_args(config, batch_max_tokens=config.prefill_batch_max_tokens),
            self.router,
            0,
            1,
        )
        self.running: List[MockReq] = []
        self.scheduled: List[MockReq] = []
        self.active_step: Optional[ActiveStep] = None
        self.next_router_tick: Optional[float] = None
        self.step_count = 0
        self.router.set_running_provider(lambda: self.running + self.scheduled)

    def _next_tick(self, now: float) -> float:
        interval = self.config.schedule_interval_s
        return math.ceil((now - EPS) / interval) * interval

    def enqueue(self, req: MockReq, now: float) -> None:
        self.queue.extend([req])
        tick = self._next_tick(now)
        if self.next_router_tick is None or tick < self.next_router_tick:
            self.next_router_tick = tick

    def router_tick(self, now: float) -> None:
        current_reqs = self.running + self.scheduled
        current = Batch(-1, current_reqs, dp_size_in_node=1) if current_reqs else None
        new_batch = self.queue.generate_new_batch(current)
        if new_batch is not None:
            self.scheduled.extend(new_batch.reqs)
        if self.queue.waiting_req_list and new_batch is not None:
            self.next_router_tick = now + self.config.schedule_interval_s
        else:
            # With an active model step and no newly admitted batch, every
            # intervening Router poll sees identical queue/KV state. Wake on
            # the next arrival or model-step completion instead.
            self.next_router_tick = None

    def start_step(self, now: float, latency_model) -> None:
        if self.active_step is not None:
            return
        if self.scheduled:
            self.running.extend(self.scheduled)
            self.scheduled = []
        if not self.running:
            return

        selected: List[MockReq] = []
        token_counts: List[int] = []
        batch_tokens = 0
        for req in self.running:
            need = min(
                max(0, req.input_len - req.shm_cur_kv_len),
                self.config.chunked_prefill_size,
            )
            if need <= 0:
                continue
            if batch_tokens + need > self.config.prefill_batch_max_tokens:
                continue
            selected.append(req)
            token_counts.append(need)
            batch_tokens += need
        if not selected:
            raise RuntimeError(f"{self.name}: no request fits the Prefill model step")

        self.step_count += 1
        exclusive = latency_model.predict_batch(token_counts, self.tp_size) * self.config.latency_scale
        for req in selected:
            if req.result.first_prefill_step_s is None:
                req.result.first_prefill_step_s = now
            req.result.prefill_steps += 1
        self.active_step = ActiveStep(
            stage="prefill",
            instance=self.name,
            tp_size=self.tp_size,
            step_index=self.step_count,
            start_s=now,
            requests=selected,
            token_counts=token_counts,
            exclusive_work_s=exclusive,
            remaining_work_s=exclusive,
            waiting_at_start=len(self.queue.waiting_req_list),
            running_at_start=len(self.running),
        )

    def complete_step(self, now: float) -> Tuple[StepRecord, List[MockReq]]:
        step = self.active_step
        if step is None:
            raise RuntimeError("complete_step called without an active Prefill step")
        finished: List[MockReq] = []
        for req, token_count in zip(step.requests, step.token_counts):
            req.shm_cur_kv_len += token_count
            if req.shm_cur_kv_len >= req.input_len:
                finished.append(req)
        finished_ids = {req.request_id for req in finished}
        self.running = [req for req in self.running if req.request_id not in finished_ids]
        self.active_step = None
        if self.queue.waiting_req_list and self.next_router_tick is None:
            self.next_router_tick = self._next_tick(now)
        record = StepRecord(
            stage=step.stage,
            instance=step.instance,
            tp_size=step.tp_size,
            step_index=step.step_index,
            start_s=step.start_s,
            end_s=now,
            exclusive_work_s=step.exclusive_work_s,
            effective_slowdown=(step.wall_elapsed_s / step.exclusive_work_s if step.exclusive_work_s else 1.0),
            request_ids=[req.request_id for req in step.requests],
            token_counts=list(step.token_counts),
            batch_tokens=sum(step.token_counts),
            waiting_at_start=step.waiting_at_start,
            running_at_start=step.running_at_start,
        )
        return record, finished

    def has_work(self) -> bool:
        return bool(
            self.queue.waiting_req_list
            or self.running
            or self.scheduled
            or self.active_step is not None
        )

    def report(self) -> Dict:
        return {
            "queued_requests": len(self.queue.waiting_req_list) + len(self.scheduled),
            "queued_tokens": sum(req.input_len for req in self.queue.waiting_req_list + self.scheduled),
            "running_requests": len(self.running),
            "running_tokens": sum(max(0, req.input_len - req.shm_cur_kv_len) for req in self.running),
            "queued_group_ids": [req.request_id for req in self.queue.waiting_req_list + self.scheduled],
            "running_group_ids": [req.request_id for req in self.running],
            "total_token_usage_rate": self.router.get_used_tokens(0) / self.config.max_total_tokens,
        }


class DecodeInstanceHarness:
    def __init__(self, name: str, config: SimulationConfig) -> None:
        self.name = name
        self.tp_size = 4
        self.config = config
        self.router = MockRouter(config.max_total_tokens)
        self.queue = QueueForPDDecode(
            _queue_args(config, batch_max_tokens=config.prefill_batch_max_tokens),
            self.router,
            0,
            1,
        )
        self.running: List[MockReq] = []
        self.scheduled: List[MockReq] = []
        self.active_step: Optional[ActiveStep] = None
        self.next_router_tick: Optional[float] = None
        self.step_count = 0
        self.router.set_running_provider(lambda: self.running + self.scheduled)

    def _next_tick(self, now: float) -> float:
        interval = self.config.schedule_interval_s
        return math.ceil((now - EPS) / interval) * interval

    def enqueue(self, req: MockReq, now: float) -> None:
        self.queue.extend([req])
        tick = self._next_tick(now)
        if self.next_router_tick is None or tick < self.next_router_tick:
            self.next_router_tick = tick

    def router_tick(self, now: float) -> None:
        current_reqs = self.running + self.scheduled
        current = Batch(-1, current_reqs, dp_size_in_node=1) if current_reqs else None
        new_batch = self.queue.generate_new_batch(current)
        if new_batch is not None:
            self.scheduled.extend(new_batch.reqs)
        if self.queue.waiting_req_list:
            self.next_router_tick = now + self.config.schedule_interval_s
        else:
            self.next_router_tick = None

    def start_step(self, now: float) -> None:
        if self.active_step is not None:
            return
        if self.scheduled:
            self.running.extend(self.scheduled)
            self.scheduled = []
        if not self.running:
            return
        self.step_count += 1
        batch_size = len(self.running)
        context_tokens = sum(req.input_len + req.shm_cur_output_len for req in self.running)
        exclusive = (
            self.config.decode_step_base_ms / 1000.0
            + self.config.decode_step_per_request_ms / 1000.0 * batch_size
            + self.config.decode_step_per_context_token_us / 1_000_000.0 * context_tokens
        ) * self.config.latency_scale
        self.active_step = ActiveStep(
            stage="decode",
            instance=self.name,
            tp_size=self.tp_size,
            step_index=self.step_count,
            start_s=now,
            requests=list(self.running),
            token_counts=[1] * batch_size,
            exclusive_work_s=exclusive,
            remaining_work_s=exclusive,
            waiting_at_start=len(self.queue.waiting_req_list),
            running_at_start=batch_size,
        )

    def complete_step(self, now: float) -> Tuple[StepRecord, List[MockReq]]:
        step = self.active_step
        if step is None:
            raise RuntimeError("complete_step called without an active Decode step")
        finished: List[MockReq] = []
        for req in step.requests:
            req.remaining_output_tokens -= 1
            req.shm_cur_output_len += 1
            req.result.decode_steps += 1
            if req.remaining_output_tokens <= 0:
                finished.append(req)
        finished_ids = {req.request_id for req in finished}
        self.running = [req for req in self.running if req.request_id not in finished_ids]
        self.active_step = None
        record = StepRecord(
            stage=step.stage,
            instance=step.instance,
            tp_size=step.tp_size,
            step_index=step.step_index,
            start_s=step.start_s,
            end_s=now,
            exclusive_work_s=step.exclusive_work_s,
            effective_slowdown=(step.wall_elapsed_s / step.exclusive_work_s if step.exclusive_work_s else 1.0),
            request_ids=[req.request_id for req in step.requests],
            token_counts=list(step.token_counts),
            batch_tokens=len(step.requests),
            waiting_at_start=step.waiting_at_start,
            running_at_start=step.running_at_start,
        )
        return record, finished

    def has_work(self) -> bool:
        return bool(
            self.queue.waiting_req_list
            or self.running
            or self.scheduled
            or self.active_step is not None
        )


class GlobalSchedulerHarness:
    """Virtual-time adapter around a production bounded-bundle selector."""

    supports_bundles = True

    def __init__(self, clock: VirtualClock, config: SimulationConfig, selector_cls=FlexTPSelectorV3) -> None:
        self.clock = clock
        self.config = config
        self.loop = asyncio.new_event_loop()
        selector_kwargs = {}
        if issubclass(selector_cls, FlexTPSelectorV4):
            selector_kwargs["long_request_threshold"] = config.naive_length_threshold
        if issubclass(selector_cls, FlexTPSelectorV5):
            selector_kwargs["short_overlap_backlog_trigger"] = (
                config.v5_short_overlap_backlog_trigger
            )
            selector_kwargs["urgent_slack_ratio"] = config.v5_urgent_slack_ratio
        if selector_cls is FlexTPSelectorV6:
            selector_kwargs["long_request_threshold"] = config.naive_length_threshold
            selector_kwargs["mps_overlap_slowdown"] = config.mps_overlap_slowdown
            selector_kwargs["request_utility_tokens"] = config.v6_request_utility_tokens
        if selector_cls is FlexTPSelectorV7:
            selector_kwargs["long_request_threshold"] = config.naive_length_threshold
            selector_kwargs["mps_overlap_slowdown"] = config.mps_overlap_slowdown
            selector_kwargs["class_quantum_tokens"] = config.v7_class_quantum_tokens
            selector_kwargs["conflict_price_weight"] = config.v7_conflict_price_weight
            selector_kwargs["urgency_weight"] = config.v7_urgency_weight
        if selector_cls is FlexTPSelectorV8:
            selector_kwargs["long_request_threshold"] = config.naive_length_threshold
            selector_kwargs["mps_overlap_slowdown"] = config.mps_overlap_slowdown
            selector_kwargs["epoch_s"] = config.v8_epoch_ms / 1000.0
            selector_kwargs["epoch_token_budget"] = config.v8_epoch_token_budget
            selector_kwargs["min_lane_quota_tokens"] = config.v8_min_lane_quota
            selector_kwargs["deadline_pressure_weight"] = config.v8_deadline_pressure_weight
        if selector_cls is FlexTPSelectorV9:
            selector_kwargs["long_request_threshold"] = config.naive_length_threshold
            selector_kwargs["mps_overlap_slowdown"] = config.mps_overlap_slowdown
            selector_kwargs["aging_interval_s"] = config.v9_aging_interval_ms / 1000.0
            selector_kwargs["interactive_token_limit"] = config.v9_interactive_tokens
            selector_kwargs["short_class_weight"] = config.v9_short_weight
            selector_kwargs["long_class_weight"] = config.v9_long_weight
        if selector_cls is FlexTPSelectorV10:
            selector_kwargs["long_request_threshold"] = config.naive_length_threshold
            selector_kwargs["mps_overlap_slowdown"] = config.mps_overlap_slowdown
            selector_kwargs["short_weight"] = config.v10_short_weight
            selector_kwargs["long_weight"] = config.v10_long_weight
            selector_kwargs["slack_weight"] = config.v10_slack_weight
            selector_kwargs["overlap_slack_ratio"] = config.v10_overlap_slack_ratio
        if selector_cls is FlexTPSelectorV11:
            selector_kwargs["long_request_threshold"] = config.naive_length_threshold
            selector_kwargs["mps_overlap_slowdown"] = config.mps_overlap_slowdown
            selector_kwargs["short_weight"] = config.v11_short_weight
            selector_kwargs["long_weight"] = config.v11_long_weight
            selector_kwargs["slack_weight"] = config.v11_slack_weight
            selector_kwargs["overlap_slack_ratio"] = config.v11_overlap_slack_ratio
            selector_kwargs["routing_cost_weight"] = config.v11_routing_cost_weight
        if selector_cls is FlexTPSelectorV12:
            selector_kwargs["long_request_threshold"] = config.naive_length_threshold
            selector_kwargs["mps_overlap_slowdown"] = config.mps_overlap_slowdown
            selector_kwargs["request_utility_tokens"] = config.v6_request_utility_tokens
            selector_kwargs["routing_cost_weight"] = config.v12_routing_cost_weight
            selector_kwargs["tp4_service_ratio_limit"] = config.v12_tp4_service_ratio_limit
        if selector_cls is FlexTPSelectorV13:
            selector_kwargs["long_request_threshold"] = config.v13_long_threshold
            selector_kwargs["mps_overlap_slowdown"] = config.mps_overlap_slowdown
            selector_kwargs["request_utility_tokens"] = config.v6_request_utility_tokens
            selector_kwargs["latency_scale"] = config.v13_latency_scale
            selector_kwargs["routing_cost_weight"] = config.v13_routing_cost_weight
            selector_kwargs["tp4_service_ratio_limit"] = config.v13_tp4_service_ratio_limit
            selector_kwargs["tp4_pressure_threshold"] = config.v13_tp4_pressure_threshold
        self.selector = selector_cls(
            object(),
            slo_ttft=config.slo_ttft_s,
            batch_token_cap=config.bundle_token_cap,
            batch_token_trigger=config.bundle_token_trigger,
            max_inflight_per_instance=config.max_inflight,
            max_admitted_tokens_per_instance=config.instance_token_credit,
            batch_window_s=config.bundle_window_s,
            prediction_margin_s=config.prediction_margin_s,
            replan_interval_s=config.replan_interval_s,
            overload_policy=config.overload_policy,
            mode_slowdowns=_mode_slowdowns(config.mps_overlap_slowdown),
            **selector_kwargs,
        )
        self.prefill_nodes = self._make_prefill_nodes()
        self.decode_node = PD_Client_Obj(
            node_id=9000,
            client_ip_port="sim:9000",
            mode="decode",
            start_args={"tp": 4, "nnodes": 1, "host": "sim", "pd_node_id": 9000},
            instance_generation="decode-generation",
        )
        self.selector.update_nodes(self.prefill_nodes, [self.decode_node])
        self.pending_results: Dict[int, RequestResult] = {}
        self.pending_futures: Dict[int, asyncio.Future] = {}
        self.report_seq = collections.defaultdict(int)

    @property
    def instances(self):
        return self.selector.instances

    @property
    def latency_model(self):
        return self.selector.latency_model

    @property
    def pending_count(self) -> int:
        return len(self.selector._pending)

    @property
    def lease_count(self) -> int:
        return len(self.selector._leases)

    @property
    def production_selector_class(self) -> str:
        return f"{type(self.selector).__module__}.{type(self.selector).__name__}"

    def node(self, node_key: str) -> PD_Client_Obj:
        return self.instances[node_key].node

    def slowdown(self, selector_state, active_states) -> float:
        return self.selector._slowdown(selector_state, active_states)

    @staticmethod
    def _make_prefill_nodes() -> List[PD_Client_Obj]:
        specs = [
            (8000, 2, "0,1"),
            (8001, 2, "2,3"),
            (8002, 4, "0,1,2,3"),
        ]
        return [
            PD_Client_Obj(
                node_id=port,
                client_ip_port=f"sim:{port}",
                mode="prefill",
                start_args={
                    "tp": tp,
                    "nnodes": 1,
                    "host": "sim",
                    "pd_node_id": port,
                    "shared_weight": True,
                    "shared_weight_master_port_start": 1300,
                    "tp_smt_group_id": "flex0",
                    "tp_smt_gpu_ids": gpu_ids,
                },
                instance_generation=f"prefill-{port}-generation",
            )
            for port, tp, gpu_ids in specs
        ]

    def submit(self, result: RequestResult) -> List[RequestResult]:
        self.selector._request_counter += 1
        self.selector._enqueue_counter += 1
        internal_id = self.selector._request_counter
        future = self.loop.create_future()
        pending_type = getattr(self.selector, "pending_type", V3Pending)
        seq_len = max(1, result.workload.input_tokens)
        pending_kwargs = {
            "internal_id": internal_id,
            "external_req_id": result.workload.request_id,
            "seq_len": seq_len,
            "deadline": result.workload.arrival_s + self.config.slo_ttft_s,
            "future": future,
            "enqueue_order": self.selector._enqueue_counter,
        }
        if getattr(self.selector, "is_flex_tp_v9", False):
            level = self.selector._initial_level(seq_len)
            pending_kwargs.update(
                queued_at=result.workload.arrival_s,
                base_level=level,
                level=level,
            )
        pending = pending_type(**pending_kwargs)
        self.pending_results[internal_id] = result
        self.pending_futures[internal_id] = future
        self.selector._pending[internal_id] = pending
        self.selector._fail_unserviceable_pending()
        self.selector._schedule_locked(self.clock.now)
        return self._drain_resolved()

    def replan(self) -> List[RequestResult]:
        self.selector._schedule_locked(self.clock.now)
        return self._drain_resolved()

    def _drain_resolved(self) -> List[RequestResult]:
        admitted: List[RequestResult] = []
        for internal_id, result in list(self.pending_results.items()):
            pending_future = self.pending_futures[internal_id]
            lease = self.selector._leases.get(internal_id)
            if lease is not None:
                # Consume the production Future so result/exception handling is
                # identical to the public async selector entry point.
                selected_node, _ = pending_future.result()
                if selected_node.client_ip_port != lease.node_key:
                    raise RuntimeError("selector Future and lease disagree on placement")
                result.instance = lease.node_key
                result.tp_size = lease.tp_size
                result.global_admit_s = self.clock.now
                result.predicted_prefill_finish_s = lease.predicted_finish
                result.status = "admitted"
                admitted.append(result)
                self.pending_results.pop(internal_id, None)
                self.pending_futures.pop(internal_id, None)
                continue
            pending = self.selector._pending.get(internal_id)
            if pending is None and pending_future.done():
                result.status = "rejected"
                error = pending_future.exception()
                result.rejection_reason = str(error or "global selector rejected request")
                self.pending_results.pop(internal_id, None)
                self.pending_futures.pop(internal_id, None)
                continue
            if pending_future.done() and pending_future.exception() is not None:
                result.status = "rejected"
                result.rejection_reason = str(pending_future.exception())
                self.pending_results.pop(internal_id, None)
                self.pending_futures.pop(internal_id, None)
        return admitted

    def bundle_accepted(self, node: PD_Client_Obj, bundle_id: int, req_ids: Sequence[int]) -> None:
        self.loop.run_until_complete(self.selector.notify_bundle_accepted(node, bundle_id, req_ids))

    def complete(self, result: RequestResult) -> List[RequestResult]:
        node = self.selector.instances[result.instance].node
        self.loop.run_until_complete(
            self.selector.notify_request_done(
                node,
                input_token_num=result.workload.input_tokens,
                actual_ttft=result.ttft_s,
                req_id=result.workload.request_id,
            )
        )
        return self._drain_resolved()

    def report(self, instance: PrefillInstanceHarness) -> None:
        self.report_seq[instance.name] += 1
        self.loop.run_until_complete(
            self.selector.update_instance_report(
                instance.name,
                instance.report(),
                report_seq=self.report_seq[instance.name],
            )
        )

    def close(self) -> None:
        self.loop.close()


class NaiveGlobalSchedulerHarness:
    """Adapter around the production 4K threshold FlexTPNaiveSelector.

    The selector makes every placement decision.  A V3 model-only instance is
    used solely for the common latency curve, GPU placement metadata, and MPS
    slowdown calculation so both policies are compared under identical worker
    timing assumptions.
    """

    supports_bundles = False

    def __init__(
        self,
        clock: VirtualClock,
        config: SimulationConfig,
        selector_cls=FlexTPNaiveSelector,
        prefill_nodes: Optional[List[PD_Client_Obj]] = None,
    ) -> None:
        self.clock = clock
        self.config = config
        self.loop = asyncio.new_event_loop()
        selector_kwargs = {}
        if selector_cls is FlexTPNaiveSelector:
            selector_kwargs["length_threshold"] = config.naive_length_threshold
        self.selector = selector_cls(object(), **selector_kwargs)
        self.prefill_nodes = (
            prefill_nodes
            if prefill_nodes is not None
            else GlobalSchedulerHarness._make_prefill_nodes()
        )
        self.decode_node = PD_Client_Obj(
            node_id=9000,
            client_ip_port="sim:9000",
            mode="decode",
            start_args={"tp": 4, "nnodes": 1, "host": "sim", "pd_node_id": 9000},
            instance_generation="decode-generation",
        )
        self.selector.update_nodes(self.prefill_nodes, [self.decode_node])

        self.model_selector = FlexTPSelectorV3(
            object(),
            slo_ttft=config.slo_ttft_s,
            mode_slowdowns=_mode_slowdowns(config.mps_overlap_slowdown),
        )
        self.model_selector.update_nodes(self.prefill_nodes, [self.decode_node])

    @property
    def instances(self):
        return self.model_selector.instances

    @property
    def latency_model(self):
        return self.model_selector.latency_model

    @property
    def pending_count(self) -> int:
        return 0

    @property
    def lease_count(self) -> int:
        if hasattr(self.selector, "flex_groups"):
            return sum(
                group.inflight_small_tp + group.inflight_large_tp
                for group in self.selector.flex_groups.values()
            )
        return sum(getattr(self.selector, "node_inflight_requests", {}).values())

    @property
    def production_selector_class(self) -> str:
        return f"{type(self.selector).__module__}.{type(self.selector).__name__}"

    def node(self, node_key: str) -> PD_Client_Obj:
        return self.instances[node_key].node

    def slowdown(self, selector_state, active_states) -> float:
        return self.model_selector._slowdown(selector_state, active_states)

    def submit(self, result: RequestResult) -> List[RequestResult]:
        node, _ = self.loop.run_until_complete(
            self.selector.async_select_p_d_node(
                [],
                None,
                None,
                input_token_num=result.workload.input_tokens,
                arrival_time=result.workload.arrival_s,
                req_id=result.workload.request_id,
            )
        )
        result.instance = node.client_ip_port
        result.tp_size = int(node.start_args.get("tp", 1))
        result.global_admit_s = self.clock.now
        result.status = "admitted"
        return [result]

    def replan(self) -> List[RequestResult]:
        return []

    def bundle_accepted(self, node: PD_Client_Obj, bundle_id: int, req_ids: Sequence[int]) -> None:
        return None

    def complete(self, result: RequestResult) -> List[RequestResult]:
        self.loop.run_until_complete(
            self.selector.notify_request_done(
                self.node(result.instance),
                input_token_num=result.workload.input_tokens,
                actual_ttft=result.ttft_s,
                req_id=result.workload.request_id,
            )
        )
        return []

    def report(self, instance: PrefillInstanceHarness) -> None:
        return None

    def close(self) -> None:
        self.loop.close()


class NaiveSwitchGlobalSchedulerHarness(NaiveGlobalSchedulerHarness):
    """Simulator adapter for production ``naive_switch`` semantics.

    The production selector waits before admitting the opposite length class.
    The discrete simulator cannot block inside ``run_until_complete`` without
    deadlocking, so this adapter keeps the same semantics with an explicit
    pending FIFO: once a class is active, only that class is admitted until its
    last worker request completes.
    """

    def __init__(self, clock: VirtualClock, config: SimulationConfig) -> None:
        super().__init__(clock, config)
        self._active_large: Optional[bool] = None
        self._pending_switch: Deque[RequestResult] = collections.deque()

    @property
    def pending_count(self) -> int:
        return len(self._pending_switch)

    def _is_large(self, result: RequestResult) -> bool:
        return result.workload.input_tokens > self.config.naive_length_threshold

    def _try_admit(self, result: RequestResult) -> List[RequestResult]:
        desired = self._is_large(result)
        if self._active_large is None:
            self._active_large = desired
        if desired != self._active_large:
            self._pending_switch.append(result)
            return []
        return super().submit(result)

    def submit(self, result: RequestResult) -> List[RequestResult]:
        return self._try_admit(result)

    def _drain_switch_pending(self) -> List[RequestResult]:
        if self._active_large is not None:
            return []
        if not self._pending_switch:
            return []
        desired = self._is_large(self._pending_switch[0])
        self._active_large = desired
        admitted: List[RequestResult] = []
        while self._pending_switch and self._is_large(self._pending_switch[0]) == desired:
            admitted.extend(super().submit(self._pending_switch.popleft()))
        return admitted

    def replan(self) -> List[RequestResult]:
        return self._drain_switch_pending()

    def complete(self, result: RequestResult) -> List[RequestResult]:
        admitted = super().complete(result)
        if self.lease_count == 0:
            self._active_large = None
            admitted.extend(self._drain_switch_pending())
        return admitted


class FixedTPSingleSelector(PDSelector):
    """Least-inflight selector for a homogeneous fixed-TP deployment."""

    def __init__(self, pd_manager) -> None:
        super().__init__(pd_manager)
        self.node_inflight_tokens: Dict[str, int] = {}
        self.node_inflight_requests: Dict[str, int] = {}
        self._decode_rr_index = 0

    async def async_select_p_d_node(
        self,
        prompt,
        sampling_params,
        multimodal_params,
        input_token_num: Optional[int] = None,
        arrival_time: Optional[float] = None,
        req_id: Optional[int] = None,
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        if not self.prefill_nodes or not self.decode_nodes:
            raise RuntimeError(
                f"FixedTPSingleSelector: req_id={req_id} no available nodes "
                f"(prefill={len(self.prefill_nodes)}, decode={len(self.decode_nodes)})"
            )
        node = min(
            self.prefill_nodes,
            key=lambda item: self.node_inflight_tokens.get(item.client_ip_port, 0),
        )
        token_num = input_token_num or 0
        key = node.client_ip_port
        self.node_inflight_requests[key] = self.node_inflight_requests.get(key, 0) + 1
        self.node_inflight_tokens[key] = self.node_inflight_tokens.get(key, 0) + token_num
        decode = self.decode_nodes[self._decode_rr_index % len(self.decode_nodes)]
        self._decode_rr_index += 1
        return node, decode

    async def notify_request_done(
        self,
        p_node: PD_Client_Obj,
        input_token_num: int = 0,
        actual_ttft: Optional[float] = None,
        req_id: Optional[int] = None,
    ) -> None:
        if p_node is None:
            return
        key = p_node.client_ip_port
        request_count = max(0, self.node_inflight_requests.get(key, 0) - 1)
        self.node_inflight_requests[key] = request_count
        if request_count == 0:
            self.node_inflight_tokens[key] = 0
        else:
            self.node_inflight_tokens[key] = max(
                0, self.node_inflight_tokens.get(key, 0) - input_token_num
            )

    def select_p_d_node(self, prompt, sampling_params, multimodal_params):
        raise NotImplementedError(
            "FixedTPSingleSelector requires async_select_p_d_node."
        )


class FixedTPGlobalSchedulerHarness(NaiveGlobalSchedulerHarness):
    """Adapter for a fixed, homogeneous Prefill deployment.

    ``fixed_tp2`` has two independent TP2 instances; ``fixed_tp4`` has one
    TP4 instance. The selector performs no length routing: all requests use
    the configured TP size, and fixed instances are physically non-overlap.
    """

    @staticmethod
    def _make_fixed_prefill_nodes(tp_size: int) -> List[PD_Client_Obj]:
        if tp_size == 2:
            specs = [
                (8000, 2, "0,1"),
                (8001, 2, "2,3"),
            ]
        elif tp_size == 4:
            specs = [(8002, 4, "0,1,2,3")]
        else:
            raise ValueError(f"unsupported fixed TP size: {tp_size}")
        return [
            PD_Client_Obj(
                node_id=port,
                client_ip_port=f"sim:{port}",
                mode="prefill",
                start_args={
                    "tp": tp,
                    "nnodes": 1,
                    "host": "sim",
                    "pd_node_id": port,
                    "shared_weight": True,
                    "shared_weight_master_port_start": 1300,
                    "tp_smt_group_id": f"fixed_tp{tp_size}",
                    "tp_smt_gpu_ids": gpu_ids,
                },
                instance_generation=f"prefill-{port}-generation",
            )
            for port, tp, gpu_ids in specs
        ]

    def __init__(self, clock: VirtualClock, config: SimulationConfig, tp_size: int) -> None:
        super().__init__(
            clock,
            config,
            selector_cls=FixedTPSingleSelector,
            prefill_nodes=self._make_fixed_prefill_nodes(tp_size),
        )
        self.fixed_tp_size = tp_size


class FlexTPStepSimulator:
    def __init__(self, workload: Sequence[WorkloadRequest], config: SimulationConfig) -> None:
        self.config = config
        self.clock = VirtualClock()
        self._original_selector_time = selector_module.time
        self._original_selector_v6_time = selector_v6_module.time
        self._original_selector_v7_time = selector_v7_module.time
        self._original_selector_v8_time = selector_v8_module.time
        self._original_selector_v9_time = selector_v9_module.time
        self._original_selector_v13_time = selector_v13_module.time
        selector_module.time = self.clock
        selector_v6_module.time = self.clock
        selector_v7_module.time = self.clock
        selector_v8_module.time = self.clock
        selector_v9_module.time = self.clock
        selector_v13_module.time = self.clock
        selector_classes = {
            "ours": FlexTPSelectorV3,
            "v3": FlexTPSelectorV3,
            "v4": FlexTPSelectorV4,
            "v5": FlexTPSelectorV5,
            "v6": FlexTPSelectorV6,
            "v7": FlexTPSelectorV7,
            "v8": FlexTPSelectorV8,
            "v9": FlexTPSelectorV9,
            "v10": FlexTPSelectorV10,
            "v11": FlexTPSelectorV11,
            "v12": FlexTPSelectorV12,
            "v13": FlexTPSelectorV13,
        }
        if config.scheduler in selector_classes:
            self.global_scheduler = GlobalSchedulerHarness(
                self.clock,
                config,
                selector_cls=selector_classes[config.scheduler],
            )
        elif config.scheduler == "naive":
            self.global_scheduler = NaiveGlobalSchedulerHarness(self.clock, config)
        elif config.scheduler == "naive_switch":
            self.global_scheduler = NaiveSwitchGlobalSchedulerHarness(self.clock, config)
        elif config.scheduler == "fixed_tp2":
            self.global_scheduler = FixedTPGlobalSchedulerHarness(self.clock, config, tp_size=2)
        elif config.scheduler == "fixed_tp4":
            self.global_scheduler = FixedTPGlobalSchedulerHarness(self.clock, config, tp_size=4)
        else:
            raise ValueError(f"unknown scheduler: {config.scheduler}")
        self.results = {
            req.request_id: RequestResult(workload=req)
            for req in sorted(workload, key=lambda item: (item.arrival_s, item.request_id))
        }
        self.arrivals = sorted(workload, key=lambda item: (item.arrival_s, item.request_id))
        self.next_arrival = 0
        self.prefill_instances: Dict[str, PrefillInstanceHarness] = {
            node.client_ip_port: PrefillInstanceHarness(
                node,
                self.global_scheduler.instances[node.client_ip_port],
                config,
            )
            for node in self.global_scheduler.prefill_nodes
        }
        self.decode = DecodeInstanceHarness("sim:9000", config)
        self.captured_bundles: Deque[Tuple[str, tuple]] = collections.deque()
        self.bundle_dispatcher = VirtualTimeBundleDispatcher(
            batch_window_s=config.bundle_window_s,
            token_trigger=config.bundle_token_trigger,
            max_bundle_tokens=config.bundle_token_cap,
            max_bundle_requests=config.max_inflight,
        )
        self.bundle_enqueue_tasks: set[asyncio.Task] = set()
        for node in self.global_scheduler.prefill_nodes:
            node.websocket = MockBundleWebsocket(node.client_ip_port, self.captured_bundles)
        self.bundle_flush_at: Dict[str, float] = {}
        self.decode_transfers: List[Tuple[float, int, RequestResult]] = []
        self.step_records: List[StepRecord] = []
        self.bundle_records: List[BundleRecord] = []
        self.next_replan_s = config.replan_interval_s
        self.next_report_s = config.worker_report_interval_s
        self.max_global_pending = 0
        self.max_global_leases = 0
        self.mps_overlap_wall_s = 0.0
        self.physical_prefill_gpu_s = 0.0
        self.mode_wall_s: Dict[str, float] = collections.defaultdict(float)
        self.event_count = 0

    def close(self) -> None:
        pending = [task for task in self.bundle_enqueue_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            self.global_scheduler.loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        self.global_scheduler.close()
        selector_module.time = self._original_selector_time
        selector_v6_module.time = self._original_selector_v6_time
        selector_v7_module.time = self._original_selector_v7_time
        selector_v8_module.time = self._original_selector_v8_time
        selector_v9_module.time = self._original_selector_v9_time
        selector_v13_module.time = self._original_selector_v13_time

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _dispatcher_queue(self, instance_name: str) -> List:
        node = self.global_scheduler.node(instance_name)
        return self.bundle_dispatcher._queues.get(self.bundle_dispatcher._node_key(node), [])

    def _settle_bundle_loop(self) -> None:
        # Two zero-time turns cover enqueue -> create_task(_flush_node) ->
        # websocket send -> sent_future completion without advancing wall time.
        loop = self.global_scheduler.loop
        loop.run_until_complete(asyncio.sleep(0))
        loop.run_until_complete(asyncio.sleep(0))
        for task in list(self.bundle_enqueue_tasks):
            if task.done():
                task.result()
                self.bundle_enqueue_tasks.discard(task)
        self._drain_captured_bundles()

    def _accept_admissions(self, admitted: Iterable[RequestResult]) -> None:
        for result in admitted:
            if not self.global_scheduler.supports_bundles:
                self._enqueue_individual(result)
                continue
            node = self.global_scheduler.node(result.instance)
            was_empty = not self._dispatcher_queue(result.instance)
            if was_empty:
                self.bundle_flush_at[result.instance] = self.clock.now + self.config.bundle_window_s
            task = self.global_scheduler.loop.create_task(
                self.bundle_dispatcher.enqueue(
                    node,
                    result.workload.request_id,
                    None,
                    None,
                    None,
                    input_token_num=result.workload.input_tokens,
                    lease_request_id=result.workload.request_id,
                )
            )
            self.bundle_enqueue_tasks.add(task)
            self._settle_bundle_loop()
            if not self._dispatcher_queue(result.instance):
                self.bundle_flush_at.pop(result.instance, None)

    def _enqueue_individual(self, result: RequestResult) -> None:
        """Model the legacy normal-PD REQ path used by the naive selector."""
        result.worker_enqueue_s = self.clock.now
        result.status = "worker_queued"
        self.bundle_records.append(
            BundleRecord(
                bundle_id=-result.workload.request_id,
                instance=result.instance,
                dispatch_s=self.clock.now,
                request_ids=[result.workload.request_id],
                input_tokens=result.workload.input_tokens,
                transport="single",
            )
        )
        self.prefill_instances[result.instance].enqueue(
            MockReq(
                result,
                stage="prefill",
                chunked_prefill_size=self.config.chunked_prefill_size,
                router_max_wait_tokens=self.config.router_max_wait_tokens,
                max_new_tokens=1,
            ),
            self.clock.now,
        )

    def _drain_captured_bundles(self) -> None:
        while self.captured_bundles:
            instance_name, envelope = self.captured_bundles.popleft()
            if len(envelope) != 4 or envelope[0] != ObjType.REQ_BUNDLE:
                raise RuntimeError(f"unexpected mock websocket envelope: {envelope[0]}")
            _, generation, bundle_id, payload = envelope
            node = self.global_scheduler.node(instance_name)
            if generation != node.instance_generation:
                raise RuntimeError("bundle carries a stale instance generation")
            request_ids = [int(item[1]) for item in payload]
            self.global_scheduler.bundle_accepted(node, bundle_id, request_ids)
            self.bundle_records.append(
                BundleRecord(
                    bundle_id=bundle_id,
                    instance=instance_name,
                    dispatch_s=self.clock.now,
                    request_ids=request_ids,
                    input_tokens=sum(int(item[2]) for item in payload),
                )
            )
            instance = self.prefill_instances[instance_name]
            for item in payload:
                group_request_id, lease_request_id, input_token_num, _, _, _ = item
                if group_request_id != lease_request_id:
                    raise RuntimeError("simulator expects one lease per workload request")
                result = self.results[int(group_request_id)]
                if int(input_token_num) != result.workload.input_tokens:
                    raise RuntimeError("bundle input token count changed in transit")
                result.bundle_id = bundle_id
                result.worker_enqueue_s = self.clock.now
                result.status = "worker_queued"
                instance.enqueue(
                    MockReq(
                        result,
                        stage="prefill",
                        chunked_prefill_size=self.config.chunked_prefill_size,
                        router_max_wait_tokens=self.config.router_max_wait_tokens,
                        max_new_tokens=1,
                    ),
                    self.clock.now,
                )

    def _process_prefill_completions(self) -> None:
        completed_instances = [
            instance
            for instance in self.prefill_instances.values()
            if instance.active_step is not None and instance.active_step.remaining_work_s <= EPS
        ]
        for instance in completed_instances:
            record, finished = instance.complete_step(self.clock.now)
            self.step_records.append(record)
            for req in finished:
                result = req.result
                result.prefill_finish_s = self.clock.now
                result.status = "prefill_finished"
                admitted = self.global_scheduler.complete(result)
                self._accept_admissions(admitted)
                if self.config.fake_decode or (
                    self.config.simulate_decode and result.workload.output_tokens > 1
                ):
                    transfer_s = (
                        self.config.kv_transfer_fixed_s
                        + self.config.kv_transfer_us_per_token / 1_000_000.0 * result.workload.input_tokens
                    )
                    heapq.heappush(
                        self.decode_transfers,
                        (self.clock.now + transfer_s, result.workload.request_id, result),
                    )
                else:
                    result.finish_s = self.clock.now
                    result.status = "completed"

    def _process_decode_completion(self) -> None:
        step = self.decode.active_step
        if step is None or step.remaining_work_s > EPS:
            return
        record, finished = self.decode.complete_step(self.clock.now)
        self.step_records.append(record)
        for req in finished:
            req.result.finish_s = self.clock.now
            req.result.status = "completed"

    def _process_arrivals(self) -> None:
        while self.next_arrival < len(self.arrivals):
            req = self.arrivals[self.next_arrival]
            if req.arrival_s > self.clock.now + EPS:
                break
            result = self.results[req.request_id]
            result.status = "global_pending"
            admitted = self.global_scheduler.submit(result)
            self._accept_admissions(admitted)
            self.next_arrival += 1

    def _process_decode_transfers(self) -> None:
        while self.decode_transfers and self.decode_transfers[0][0] <= self.clock.now + EPS:
            _, _, result = heapq.heappop(self.decode_transfers)
            result.decode_enqueue_s = self.clock.now
            if self.config.fake_decode:
                result.finish_s = self.clock.now
                result.status = "completed"
                continue
            result.status = "decode_queued"
            self.decode.enqueue(
                MockReq(
                    result,
                    stage="decode",
                    chunked_prefill_size=self.config.chunked_prefill_size,
                    router_max_wait_tokens=self.config.router_max_wait_tokens,
                    max_new_tokens=max(1, result.workload.output_tokens - 1),
                ),
                self.clock.now,
            )

    def _process_due_bundles(self) -> None:
        while True:
            due = [
                name
                for name, flush_at in self.bundle_flush_at.items()
                if flush_at <= self.clock.now + EPS
            ]
            if not due:
                return
            for name in sorted(due):
                node = self.global_scheduler.node(name)
                self.global_scheduler.loop.run_until_complete(
                    self.bundle_dispatcher._flush_node(node)
                )
                self._settle_bundle_loop()
                if self._dispatcher_queue(name):
                    self.bundle_flush_at[name] = self.clock.now + self.config.bundle_window_s
                else:
                    self.bundle_flush_at.pop(name, None)

    def _process_router_ticks(self) -> None:
        for instance in self.prefill_instances.values():
            if instance.next_router_tick is not None and instance.next_router_tick <= self.clock.now + EPS:
                instance.router_tick(self.clock.now)
        if self.decode.next_router_tick is not None and self.decode.next_router_tick <= self.clock.now + EPS:
            self.decode.router_tick(self.clock.now)

    def _start_model_steps(self) -> None:
        latency_model = self.global_scheduler.latency_model
        for instance in self.prefill_instances.values():
            instance.start_step(self.clock.now, latency_model)
        if self.config.simulate_decode:
            self.decode.start_step(self.clock.now)

    def _report_workers(self) -> None:
        if self.clock.now + EPS < self.next_report_s:
            return
        for instance in self.prefill_instances.values():
            self.global_scheduler.report(instance)
        while self.next_report_s <= self.clock.now + EPS:
            self.next_report_s += self.config.worker_report_interval_s

    def _replan(self) -> None:
        if self.clock.now + EPS < self.next_replan_s:
            return
        admitted = self.global_scheduler.replan()
        self._accept_admissions(admitted)
        while self.next_replan_s <= self.clock.now + EPS:
            self.next_replan_s += self.config.replan_interval_s

    def _active_prefill_instances(self) -> List[PrefillInstanceHarness]:
        return [instance for instance in self.prefill_instances.values() if instance.active_step is not None]

    def _advance(self, target_s: float) -> None:
        if target_s < self.clock.now - EPS:
            raise RuntimeError("simulation clock moved backwards")
        delta = max(0.0, target_s - self.clock.now)
        active = self._active_prefill_instances()
        selector_states = [item.selector_state for item in active]
        if delta > 0 and active:
            signature = "+".join(str(item.tp_size) for item in sorted(active, key=lambda item: item.name))
            self.mode_wall_s[signature] += delta
            active_gpus = set().union(*(item.selector_state.gpu_set for item in active))
            self.physical_prefill_gpu_s += len(active_gpus) * delta
            if any(
                left.selector_state.gpu_set & right.selector_state.gpu_set
                for index, left in enumerate(active)
                for right in active[index + 1 :]
            ):
                self.mps_overlap_wall_s += delta
        for instance in active:
            step = instance.active_step
            slowdown = self.global_scheduler.slowdown(instance.selector_state, selector_states)
            step.remaining_work_s = max(0.0, step.remaining_work_s - delta / slowdown)
            step.wall_elapsed_s += delta
        if self.decode.active_step is not None:
            self.decode.active_step.remaining_work_s = max(
                0.0,
                self.decode.active_step.remaining_work_s - delta,
            )
            self.decode.active_step.wall_elapsed_s += delta
        self.clock.now = target_s

    def _next_event_time(self) -> float:
        candidates: List[float] = []
        if self.next_arrival < len(self.arrivals):
            candidates.append(self.arrivals[self.next_arrival].arrival_s)
        candidates.extend(self.bundle_flush_at.values())
        if self.decode_transfers:
            candidates.append(self.decode_transfers[0][0])
        for instance in self.prefill_instances.values():
            if instance.next_router_tick is not None:
                candidates.append(instance.next_router_tick)
        if self.decode.next_router_tick is not None:
            candidates.append(self.decode.next_router_tick)
        if self.global_scheduler.pending_count:
            candidates.append(self.next_replan_s)
        if any(instance.has_work() for instance in self.prefill_instances.values()):
            candidates.append(self.next_report_s)

        active = self._active_prefill_instances()
        selector_states = [item.selector_state for item in active]
        for instance in active:
            slowdown = self.global_scheduler.slowdown(instance.selector_state, selector_states)
            candidates.append(self.clock.now + instance.active_step.remaining_work_s * slowdown)
        if self.decode.active_step is not None:
            candidates.append(self.clock.now + self.decode.active_step.remaining_work_s)
        if not candidates:
            return math.inf
        return max(self.clock.now, min(candidates))

    def _done(self) -> bool:
        terminal = all(result.status in {"completed", "rejected"} for result in self.results.values())
        return bool(
            terminal
            and self.next_arrival >= len(self.arrivals)
            and not self.global_scheduler.pending_count
            and not self.global_scheduler.lease_count
            and not self.bundle_dispatcher._queues
            and not self.bundle_enqueue_tasks
            and not self.captured_bundles
            and not self.bundle_flush_at
            and not self.decode_transfers
            and not any(instance.has_work() for instance in self.prefill_instances.values())
            and (not self.config.simulate_decode or not self.decode.has_work())
        )

    def run(self) -> Dict:
        last_arrival = self.arrivals[-1].arrival_s if self.arrivals else 0.0
        deadline = last_arrival + self.config.max_drain_s
        while not self._done():
            next_time = self._next_event_time()
            if not math.isfinite(next_time):
                statuses = collections.Counter(result.status for result in self.results.values())
                raise RuntimeError(f"simulation deadlocked at {self.clock.now:.6f}s: {dict(statuses)}")
            if next_time > deadline:
                raise RuntimeError(
                    f"simulation drain exceeded {self.config.max_drain_s}s at virtual time {next_time:.3f}s"
                )
            self._advance(next_time)
            self._process_prefill_completions()
            self._process_decode_completion()
            self._process_decode_transfers()
            self._process_arrivals()
            self._replan()
            self._process_due_bundles()
            self._process_router_ticks()
            self._report_workers()
            self._start_model_steps()
            self.max_global_pending = max(
                self.max_global_pending,
                self.global_scheduler.pending_count,
            )
            self.max_global_leases = max(
                self.max_global_leases,
                self.global_scheduler.lease_count,
            )
            self.event_count += 1
        return self.summary()

    @staticmethod
    def _percentile(values: Sequence[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        position = (len(ordered) - 1) * percentile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def summary(self) -> Dict:
        completed = [result for result in self.results.values() if result.status == "completed"]
        rejected = [result for result in self.results.values() if result.status == "rejected"]
        ttfts = [result.ttft_s for result in completed if result.ttft_s is not None]
        e2es = [result.e2e_s for result in completed if result.e2e_s is not None]
        short_ttfts = [result.ttft_s for result in completed if not result.workload.is_long and result.ttft_s is not None]
        long_ttfts = [result.ttft_s for result in completed if result.workload.is_long and result.ttft_s is not None]
        master_waits = [
            result.global_admit_s - result.workload.arrival_s
            for result in completed
            if result.global_admit_s is not None
        ]
        transport_waits = [
            result.worker_enqueue_s - result.global_admit_s
            for result in completed
            if result.worker_enqueue_s is not None and result.global_admit_s is not None
        ]
        kv_transfer_waits = [
            result.decode_enqueue_s - result.prefill_finish_s
            for result in completed
            if result.decode_enqueue_s is not None and result.prefill_finish_s is not None
        ]
        worker_waits = [
            result.first_prefill_step_s - result.worker_enqueue_s
            for result in completed
            if result.first_prefill_step_s is not None and result.worker_enqueue_s is not None
        ]
        execution_times = [
            result.prefill_finish_s - result.first_prefill_step_s
            for result in completed
            if result.prefill_finish_s is not None and result.first_prefill_step_s is not None
        ]
        prediction_errors = [
            result.prefill_finish_s - result.predicted_prefill_finish_s
            for result in completed
            if result.prefill_finish_s is not None and result.predicted_prefill_finish_s is not None
        ]
        on_time = [
            result
            for result in completed
            if result.ttft_s is not None and result.ttft_s <= self.config.slo_ttft_s + EPS
        ]
        prefill_steps = [record for record in self.step_records if record.stage == "prefill"]
        decode_steps = [record for record in self.step_records if record.stage == "decode"]
        arrivals = [item.arrival_s for item in self.arrivals]
        arrival_gaps = [right - left for left, right in zip(arrivals, arrivals[1:])]
        offered_duration = max(arrivals) - min(arrivals) if len(arrivals) > 1 else 0.0
        makespan = self.clock.now - min(arrivals) if arrivals else 0.0
        tp_counts = collections.Counter(result.tp_size for result in completed)
        multi_bundles = [record for record in self.bundle_records if len(record.request_ids) > 1]
        multi_prefill_steps = [record for record in prefill_steps if len(record.request_ids) > 1]
        overdue_admitted = [
            result
            for result in completed
            if result.global_admit_s is not None
            and result.global_admit_s > result.workload.arrival_s + self.config.slo_ttft_s + EPS
        ]
        deadline_rejections = [
            result
            for result in rejected
            if result.rejection_reason and "deadline" in result.rejection_reason.lower()
        ]
        ttft_slo_successes = sum(value <= self.config.slo_ttft_s + EPS for value in ttfts)
        exclusive_prefill_gpu_s = sum(
            record.exclusive_work_s * record.tp_size for record in prefill_steps
        )
        process_prefill_gpu_s = sum(
            (record.end_s - record.start_s) * record.tp_size for record in prefill_steps
        )
        return {
            "scheduler": self.config.scheduler,
            "naive_length_threshold": self.config.naive_length_threshold,
            "production_classes": {
                "global_selector": self.global_scheduler.production_selector_class,
                "bundle_dispatcher": (
                    f"{PrefillBundleDispatcher.__module__}.{PrefillBundleDispatcher.__name__}"
                    if self.global_scheduler.supports_bundles
                    else None
                ),
                "prefill_queue": f"{ChunkedPrefillQueue.__module__}.{ChunkedPrefillQueue.__name__}",
                "decode_queue": (
                    f"{QueueForPDDecode.__module__}.{QueueForPDDecode.__name__}"
                    if self.config.simulate_decode and not self.config.fake_decode
                    else None
                ),
                "decode_transport": (
                    "fake-kv-transfer" if self.config.fake_decode else None
                ),
            },
            "scheduler_stats": (
                self.global_scheduler.selector.scheduler_snapshot()
                if hasattr(self.global_scheduler.selector, "scheduler_snapshot")
                else {}
            ),
            "timing_model": {
                "prefill": (
                    "V3-family latency_model.predict_batch per model step"
                    if self.config.scheduler in {"naive", "naive_switch"}
                    else f"{self.config.scheduler.upper()} local batch curve per model step"
                ),
                "mps": (
                    f"{('V3-family' if self.config.scheduler in {'naive', 'naive_switch'} else self.config.scheduler.upper())} "
                    "_slowdown with dynamic active placement set; "
                    f"overlap factor={self.config.mps_overlap_slowdown:g}"
                ),
                "decode": (
                    "configurable base + per-request + per-context-token step model"
                    if self.config.simulate_decode and not self.config.fake_decode
                    else (
                        "fake decode: no Decode step; fixed/per-token KV transfer is simulated"
                        if self.config.fake_decode
                        else "non-bottlenecking: complete immediately after prefill"
                    )
                ),
                "is_gpu_measurement": False,
            },
            "offered_requests": len(self.results),
            "completed_requests": len(completed),
            "completion_fraction": len(completed) / len(self.results) if self.results else 0.0,
            "rejected_requests": len(rejected),
            "deadline_rejected_requests": len(deadline_rejections),
            "overdue_admitted_requests": len(overdue_admitted),
            "long_request_fraction": (
                sum(item.is_long for item in self.arrivals) / len(self.arrivals) if self.arrivals else 0.0
            ),
            "arrival_gap_p50_s": self._percentile(arrival_gaps, 0.50),
            "arrival_gap_p95_s": self._percentile(arrival_gaps, 0.95),
            "arrival_pairs_within_bundle_window_fraction": (
                sum(gap <= self.config.bundle_window_s + EPS for gap in arrival_gaps) / len(arrival_gaps)
                if arrival_gaps else 0.0
            ),
            "offered_duration_s": offered_duration,
            "makespan_s": makespan,
            "completion_throughput_rps": len(completed) / makespan if makespan else 0.0,
            "prefill_token_throughput_s": (
                sum(result.workload.input_tokens for result in completed) / makespan if makespan else 0.0
            ),
            "output_token_throughput_s": (
                sum(result.workload.output_tokens for result in completed) / makespan if makespan else 0.0
            ),
            "ttft_mean_s": statistics.mean(ttfts) if ttfts else 0.0,
            "ttft_p50_s": self._percentile(ttfts, 0.50),
            "ttft_p95_s": self._percentile(ttfts, 0.95),
            "ttft_p99_s": self._percentile(ttfts, 0.99),
            "ttft_max_s": max(ttfts, default=0.0),
            "ttft_slo_attainment": (
                ttft_slo_successes / len(ttfts) if ttfts else 0.0
            ),
            "offered_ttft_slo_attainment": (
                ttft_slo_successes / len(self.results) if self.results else 0.0
            ),
            "on_time_requests": len(on_time),
            "on_time_input_tokens": sum(result.workload.input_tokens for result in on_time),
            "on_time_goodput_rps": len(on_time) / makespan if makespan else 0.0,
            "on_time_input_token_goodput_s": (
                sum(result.workload.input_tokens for result in on_time) / makespan
                if makespan else 0.0
            ),
            "offered_window_on_time_goodput_rps": (
                len(on_time) / offered_duration if offered_duration else 0.0
            ),
            "offered_window_on_time_input_token_goodput_s": (
                sum(result.workload.input_tokens for result in on_time) / offered_duration
                if offered_duration else 0.0
            ),
            "short_ttft_p95_s": self._percentile(short_ttfts, 0.95),
            "long_ttft_p95_s": self._percentile(long_ttfts, 0.95),
            "master_wait_p95_s": self._percentile(master_waits, 0.95),
            "transport_wait_p95_s": self._percentile(transport_waits, 0.95),
            "kv_transfer_wait_p95_s": self._percentile(kv_transfer_waits, 0.95),
            "fake_decode_transfer_count": (
                len(kv_transfer_waits) if self.config.fake_decode else 0
            ),
            "worker_wait_p95_s": self._percentile(worker_waits, 0.95),
            "prefill_execution_p95_s": self._percentile(execution_times, 0.95),
            "prediction_error_mean_s": (
                statistics.mean(prediction_errors) if prediction_errors else 0.0
            ),
            "prediction_error_p95_s": self._percentile(prediction_errors, 0.95),
            "prediction_underestimate_fraction": (
                sum(error > EPS for error in prediction_errors) / len(prediction_errors)
                if prediction_errors else 0.0
            ),
            "e2e_p50_s": self._percentile(e2es, 0.50),
            "e2e_p95_s": self._percentile(e2es, 0.95),
            "bundle_count": len(self.bundle_records),
            "bundle_mean_requests": (
                statistics.mean(len(item.request_ids) for item in self.bundle_records)
                if self.bundle_records else 0.0
            ),
            "bundle_max_requests": max((len(item.request_ids) for item in self.bundle_records), default=0),
            "multi_request_bundle_fraction": (
                len(multi_bundles) / len(self.bundle_records) if self.bundle_records else 0.0
            ),
            "requests_in_multi_request_bundles_fraction": (
                sum(len(item.request_ids) for item in multi_bundles) / len(completed) if completed else 0.0
            ),
            "bundle_mean_tokens": (
                statistics.mean(item.input_tokens for item in self.bundle_records)
                if self.bundle_records else 0.0
            ),
            "prefill_step_count": len(prefill_steps),
            "prefill_step_mean_batch": (
                statistics.mean(len(item.request_ids) for item in prefill_steps) if prefill_steps else 0.0
            ),
            "prefill_step_max_batch": max((len(item.request_ids) for item in prefill_steps), default=0),
            "multi_request_prefill_step_fraction": (
                len(multi_prefill_steps) / len(prefill_steps) if prefill_steps else 0.0
            ),
            "prefill_step_mean_tokens": (
                statistics.mean(item.batch_tokens for item in prefill_steps) if prefill_steps else 0.0
            ),
            "decode_step_count": len(decode_steps),
            "decode_step_mean_batch": (
                statistics.mean(len(item.request_ids) for item in decode_steps) if decode_steps else 0.0
            ),
            "tp2_request_share": tp_counts[2] / len(completed) if completed else 0.0,
            "tp4_request_share": tp_counts[4] / len(completed) if completed else 0.0,
            "tp2_long_request_count": sum(
                result.tp_size == 2
                and result.workload.input_tokens > self.config.naive_length_threshold
                for result in completed
            ),
            "mps_overlap_wall_s": self.mps_overlap_wall_s,
            "exclusive_prefill_gpu_s": exclusive_prefill_gpu_s,
            "process_prefill_gpu_s": process_prefill_gpu_s,
            "physical_prefill_gpu_s": self.physical_prefill_gpu_s,
            "physical_gpu_s_per_completed_request": (
                self.physical_prefill_gpu_s / len(completed) if completed else 0.0
            ),
            "physical_gpu_s_per_on_time_request": (
                self.physical_prefill_gpu_s / len(on_time) if on_time else 0.0
            ),
            "prefill_mode_wall_s": dict(sorted(self.mode_wall_s.items())),
            "max_global_pending": self.max_global_pending,
            "max_global_leases": self.max_global_leases,
            "event_count": self.event_count,
            "virtual_end_s": self.clock.now,
        }

    def write_artifacts(
        self,
        output_dir: Path,
        run_name: str,
        trace_steps: bool,
        summary: Optional[Dict] = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / f"{run_name}.summary.json"
        summary_path.write_text(json.dumps(summary or self.summary(), indent=2, sort_keys=True) + "\n")

        requests_path = output_dir / f"{run_name}.requests.csv"
        rows = []
        for result in sorted(self.results.values(), key=lambda item: item.workload.request_id):
            row = dataclasses.asdict(result.workload)
            row.update(
                {
                    "status": result.status,
                    "instance": result.instance,
                    "tp_size": result.tp_size,
                    "global_admit_s": result.global_admit_s,
                    "bundle_id": result.bundle_id,
                    "worker_enqueue_s": result.worker_enqueue_s,
                    "first_prefill_step_s": result.first_prefill_step_s,
                    "prefill_finish_s": result.prefill_finish_s,
                    "decode_enqueue_s": result.decode_enqueue_s,
                    "finish_s": result.finish_s,
                    "ttft_s": result.ttft_s,
                    "e2e_s": result.e2e_s,
                    "prefill_steps": result.prefill_steps,
                    "decode_steps": result.decode_steps,
                    "rejection_reason": result.rejection_reason,
                }
            )
            rows.append(row)
        with requests_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["request_id"])
            writer.writeheader()
            writer.writerows(rows)

        bundles_path = output_dir / f"{run_name}.bundles.jsonl"
        with bundles_path.open("w") as file:
            for record in self.bundle_records:
                file.write(json.dumps(dataclasses.asdict(record), sort_keys=True) + "\n")

        if trace_steps:
            steps_path = output_dir / f"{run_name}.steps.jsonl"
            with steps_path.open("w") as file:
                for record in self.step_records:
                    file.write(json.dumps(dataclasses.asdict(record), sort_keys=True) + "\n")


@contextlib.contextmanager
def _working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def generate_servegen_mm_image(
    *, repo_root: Path, request_rate: float, duration_s: int, seed: int,
    output_divisor: int = 1,
) -> List[WorkloadRequest]:
    servegen_root = repo_root / "_" / "ServeGen"
    if not servegen_root.is_dir():
        raise FileNotFoundError(f"ServeGen checkout not found: {servegen_root}")
    sys.path.insert(0, str(servegen_root))
    try:
        from servegen import Category
        from servegen.clientpool import ClientPool
        from servegen.construct import generate_workload
        from servegen.utils import get_constant_rate_fn

        # ServeGen currently resolves its data/ directory relative to cwd.
        with _working_directory(servegen_root):
            pool = ClientPool(Category.MULTIMODAL, "mm-image")
            rate_fn = get_constant_rate_fn(pool.span(0, duration_s), request_rate)
            generated = generate_workload(pool, rate_fn, duration=duration_s, seed=seed)
    finally:
        try:
            sys.path.remove(str(servegen_root))
        except ValueError:
            pass

    requests: List[WorkloadRequest] = []
    for item in generated:
        input_tokens = max(4, int(item.data.get("text_tokens", 16)))
        input_tokens += int(sum(item.data.get("image_tokens", 0)))
        input_tokens += int(sum(item.data.get("audio_tokens", 0)))
        input_tokens += int(sum(item.data.get("video_tokens", 0)))
        output_tokens = max(4, int(item.data.get("output_tokens", 16)) // max(1, int(output_divisor)))
        if input_tokens + output_tokens > MAX_REQ_TOTAL_TOKENS:
            continue
        requests.append(
            WorkloadRequest(
                request_id=len(requests) + 1,
                arrival_s=float(item.timestamp),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                source="servegen-mm-image",
                is_long=input_tokens >= 1000,
            )
        )
    return requests


def generate_servegen_dataset(
    *, repo_root: Path, model: str, category: str, request_rate: float,
    duration_s: int, seed: int, output_divisor: int = 10,
) -> List[WorkloadRequest]:
    """Generate a ServeGen language/reason trace with bounded decode work."""
    servegen_root = repo_root / "_" / "ServeGen"
    sys.path.insert(0, str(servegen_root))
    try:
        from servegen import Category
        from servegen.clientpool import ClientPool
        from servegen.construct import generate_workload
        from servegen.utils import get_constant_rate_fn

        category_enum = {
            "language": Category.LANGUAGE,
            "reason": Category.REASON,
        }[category]
        with _working_directory(servegen_root):
            pool = ClientPool(category_enum, model)
            rate_fn = get_constant_rate_fn(pool.span(0, duration_s), request_rate)
            generated = generate_workload(pool, rate_fn, duration=duration_s, seed=seed)
    finally:
        try:
            sys.path.remove(str(servegen_root))
        except ValueError:
            pass

    divisor = max(1, int(output_divisor))
    requests: List[WorkloadRequest] = []
    for item in generated:
        input_tokens = max(4, int(item.data.get("input_tokens", 16)))
        output_tokens = max(4, int(item.data.get("output_tokens", 16)) // divisor)
        if input_tokens + output_tokens > MAX_REQ_TOTAL_TOKENS:
            continue
        requests.append(WorkloadRequest(
            request_id=len(requests) + 1,
            arrival_s=float(item.timestamp),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            source=f"servegen-{model}",
            is_long=input_tokens >= 4000,
        ))
    return requests


def generate_synthetic_5pct(
    *, request_rate: float, num_prompts: int, seed: int
) -> List[WorkloadRequest]:
    """Match benchmark_serving_chat_req_rate.py's simple.1-5 distribution."""
    rng = random.Random(seed)
    requests: List[WorkloadRequest] = []
    for index in range(num_prompts):
        is_long = rng.random() >= 0.95
        input_tokens = rng.randint(1000, 20000) if is_long else rng.randint(100, 1000)
        output_tokens = rng.randint(50, 500)
        requests.append(
            WorkloadRequest(
                request_id=index + 1,
                arrival_s=index / request_rate,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                source="synthetic-5pct",
                is_long=is_long,
            )
        )
    return requests


def generate_mooncake_workload(
    *, dataset_path: Path, request_rate: float, num_prompts: int, seed: int,
    output_divisor: int = 10,
) -> List[WorkloadRequest]:
    """Replay a Mooncake JSONL length trace at a requested constant rate."""
    entries = []
    with dataset_path.open() as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            input_tokens = int(item["input_length"])
            output_tokens = int(item["output_length"])
            if input_tokens < 4 or output_tokens < 4:
                continue
            output_tokens = max(4, output_tokens // max(1, int(output_divisor)))
            if input_tokens + output_tokens > MAX_REQ_TOTAL_TOKENS:
                continue
            entries.append((input_tokens, output_tokens))
    if not entries:
        raise RuntimeError(f"Mooncake dataset has no usable entries: {dataset_path}")
    rng = random.Random(seed)
    if len(entries) >= num_prompts:
        entries = rng.sample(entries, num_prompts)
    else:
        entries = [entries[index % len(entries)] for index in range(num_prompts)]
    return [
        WorkloadRequest(
            request_id=index + 1,
            arrival_s=index / request_rate,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            source="mooncake-trace",
            is_long=input_tokens >= 4000,
        )
        for index, (input_tokens, output_tokens) in enumerate(entries)
    ]


def _parse_rates(value: str) -> List[float]:
    rates = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not rates or any(not math.isfinite(item) or item <= 0 for item in rates):
        raise argparse.ArgumentTypeError("rates must be comma-separated positive finite numbers")
    return rates


def _repo_root() -> Path:
    return REPO_ROOT


def _quiet_production_loggers() -> None:
    selector_module.logger.setLevel(logging.ERROR)
    selector_v6_module.logger.setLevel(logging.ERROR)
    selector_v7_module.logger.setLevel(logging.ERROR)
    selector_v8_module.logger.setLevel(logging.ERROR)
    selector_v9_module.logger.setLevel(logging.ERROR)
    selector_v13_module.logger.setLevel(logging.ERROR)
    logging.getLogger(FlexTPNaiveSelector.__module__).setLevel(logging.ERROR)
    logging.getLogger(ChunkedPrefillQueue.__module__).setLevel(logging.ERROR)
    logging.getLogger(QueueForPDDecode.__module__).setLevel(logging.ERROR)


def _run_self_test() -> None:
    _quiet_production_loggers()
    workload = [
        WorkloadRequest(index + 1, 0.0, 256, 8, "self-test", False)
        for index in range(19)
    ]
    workload.append(WorkloadRequest(20, 0.0, 4000, 8, "self-test", False))
    workload.append(WorkloadRequest(21, 0.100, 17000, 8, "self-test", True))
    config = SimulationConfig(slo_ttft_s=30.0, simulate_decode=False)
    with FlexTPStepSimulator(workload, config) as simulator:
        summary = simulator.run()
        assert summary["completed_requests"] == len(workload), summary
        assert summary["bundle_max_requests"] > 1, summary
        assert summary["prefill_step_max_batch"] > 1, summary
        assert simulator.results[21].prefill_steps == 3, simulator.results[21]
        assert summary["production_classes"]["global_selector"].endswith("FlexTPSelectorV3")
        assert summary["production_classes"]["prefill_queue"].endswith("ChunkedPrefillQueue")
        print("SELF_TEST_OK")
        print(json.dumps(summary, indent=2, sort_keys=True))

    naive_config = dataclasses.replace(
        config,
        scheduler="naive",
        naive_length_threshold=4000,
    )
    with FlexTPStepSimulator(workload, naive_config) as simulator:
        summary = simulator.run()
        assert summary["completed_requests"] == len(workload), summary
        assert summary["bundle_max_requests"] == 1, summary
        assert summary["prefill_step_max_batch"] > 1, summary
        assert simulator.results[21].tp_size == 4, simulator.results[21]
        assert all(simulator.results[index].tp_size == 2 for index in range(1, 21))
        assert summary["mps_overlap_wall_s"] > 0.0, summary
        assert summary["production_classes"]["global_selector"].endswith("FlexTPNaiveSelector")
        print("NAIVE_SELF_TEST_OK")

    fixed_tp2_config = dataclasses.replace(
        config,
        scheduler="fixed_tp2",
        naive_length_threshold=4000,
    )
    with FlexTPStepSimulator(workload, fixed_tp2_config) as simulator:
        summary = simulator.run()
        assert summary["completed_requests"] == len(workload), summary
        assert summary["bundle_max_requests"] == 1, summary
        assert summary["prefill_step_max_batch"] > 1, summary
        assert all(simulator.results[index].tp_size == 2 for index in range(1, 22))
        assert summary["mps_overlap_wall_s"] == 0.0, summary
        assert summary["production_classes"]["global_selector"].endswith("FixedTPSingleSelector")
        print("FIXED_TP2_SELF_TEST_OK")

    fixed_tp4_config = dataclasses.replace(
        config,
        scheduler="fixed_tp4",
        naive_length_threshold=4000,
    )
    with FlexTPStepSimulator(workload, fixed_tp4_config) as simulator:
        summary = simulator.run()
        assert summary["completed_requests"] == len(workload), summary
        assert summary["bundle_max_requests"] == 1, summary
        assert summary["prefill_step_max_batch"] > 1, summary
        assert all(simulator.results[index].tp_size == 4 for index in range(1, 22))
        assert summary["mps_overlap_wall_s"] == 0.0, summary
        assert summary["production_classes"]["global_selector"].endswith("FixedTPSingleSelector")
        print("FIXED_TP4_SELF_TEST_OK")

    fake_config = dataclasses.replace(
        config,
        scheduler="fixed_tp2",
        fake_decode=True,
        kv_transfer_fixed_s=0.020,
        kv_transfer_us_per_token=0.0,
    )
    with FlexTPStepSimulator(workload, fake_config) as simulator:
        summary = simulator.run()
        assert summary["completed_requests"] == len(workload), summary
        assert summary["decode_step_count"] == 0, summary
        assert summary["fake_decode_transfer_count"] == len(workload), summary
        assert abs(summary["kv_transfer_wait_p95_s"] - 0.020) < 1e-12, summary
        assert summary["timing_model"]["decode"].startswith("fake decode"), summary
        assert summary["production_classes"]["decode_transport"] == "fake-kv-transfer"
        assert all(
            simulator.results[index].finish_s > simulator.results[index].prefill_finish_s
            for index in range(1, len(workload) + 1)
        )
        print("FAKE_DECODE_SELF_TEST_OK")

    v6_config = dataclasses.replace(config, scheduler="v6")
    with FlexTPStepSimulator(workload, v6_config) as simulator:
        summary = simulator.run()
        assert summary["completed_requests"] == len(workload), summary
        assert summary["bundle_max_requests"] > 1, summary
        assert summary["prefill_step_max_batch"] > 1, summary
        assert summary["tp2_long_request_count"] == 0, summary
        assert simulator.results[21].tp_size == 4, simulator.results[21]
        assert all(simulator.results[index].tp_size == 2 for index in range(1, 21))
        assert summary["production_classes"]["global_selector"].endswith("FlexTPSelectorV6")
        print("V6_SELF_TEST_OK")

    for version in ("v7", "v8", "v9", "v10", "v11", "v12", "v13"):
        config = dataclasses.replace(config, scheduler=version)
        with FlexTPStepSimulator(workload, config) as simulator:
            summary = simulator.run()
            assert summary["completed_requests"] == len(workload), summary
            assert summary["bundle_max_requests"] > 1, summary
            assert summary["prefill_step_max_batch"] > 1, summary
            if version not in ("v11", "v12", "v13"):
                assert summary["tp2_long_request_count"] == 0, summary
            assert summary["mps_overlap_wall_s"] > 0.0, summary
            assert any(
                "2" in mode and "4" in mode
                for mode in summary["prefill_mode_wall_s"]
            ), summary
            if version in ("v11", "v12", "v13"):
                assert all(simulator.results[index].tp_size in (2, 4) for index in range(1, 22))
                assert any(simulator.results[index].tp_size == 2 for index in range(1, 22))
                assert any(simulator.results[index].tp_size == 4 for index in range(1, 22))
            else:
                assert simulator.results[21].tp_size == 4, simulator.results[21]
                assert all(simulator.results[index].tp_size == 2 for index in range(1, 21))
            assert summary["production_classes"]["global_selector"].endswith(
                f"FlexTPSelector{version.upper()}"
            )
            print(f"{version.upper()}_SELF_TEST_OK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("servegen-mm-image", "synthetic-5pct", "all"),
        default="all",
    )
    parser.add_argument("--rates", type=_parse_rates, default=[10.0])
    parser.add_argument(
        "--scheduler",
        choices=("fixed_tp2", "fixed_tp4", "ours", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "naive", "naive_switch", "compare"),
        default="compare",
        help="global scheduling policy; compare runs both on the same workload",
    )
    parser.add_argument("--naive-length-threshold", type=int, default=4000)
    parser.add_argument(
        "--mps-overlap-slowdown",
        type=float,
        default=2.0,
        help="per-instance slowdown whenever TP4 overlaps a TP2 process",
    )
    parser.add_argument("--duration", type=int, default=180, help="ServeGen duration in seconds")
    parser.add_argument("--num-prompts", type=int, default=2000, help="synthetic-5pct request count")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("_/flex_tp_step_sim"))
    parser.add_argument("--trace-steps", action="store_true", help="write every model step to JSONL")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verbose-scheduler", action="store_true")

    parser.add_argument("--slo-ttft", type=float, default=3.0)
    parser.add_argument("--bundle-window-ms", type=float, default=20.0)
    parser.add_argument("--bundle-token-cap", type=int, default=8192)
    parser.add_argument("--bundle-token-trigger", type=int, default=4096)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--instance-token-credit", type=int, default=16384)
    parser.add_argument("--prediction-margin-ms", type=float, default=80.0)
    parser.add_argument("--v6-request-utility-tokens", type=float, default=2000.0)
    parser.add_argument("--v7-class-quantum-tokens", type=int, default=4096)
    parser.add_argument("--v7-conflict-price-weight", type=float, default=0.35)
    parser.add_argument("--v7-urgency-weight", type=float, default=2.0)
    parser.add_argument("--v8-epoch-ms", type=float, default=50.0)
    parser.add_argument("--v8-epoch-token-budget", type=int, default=16384)
    parser.add_argument("--v8-min-lane-quota", type=int, default=2048)
    parser.add_argument("--v8-deadline-pressure-weight", type=float, default=2.0)
    parser.add_argument("--v9-aging-interval-ms", type=float, default=150.0)
    parser.add_argument("--v9-interactive-tokens", type=int, default=1024)
    parser.add_argument("--v9-short-weight", type=float, default=1.0)
    parser.add_argument("--v9-long-weight", type=float, default=1.0)
    parser.add_argument("--v10-short-weight", type=float, default=1.0)
    parser.add_argument("--v10-long-weight", type=float, default=1.0)
    parser.add_argument("--v10-slack-weight", type=float, default=1.0)
    parser.add_argument("--v10-overlap-slack-ratio", type=float, default=0.10)
    parser.add_argument("--v11-short-weight", type=float, default=1.0)
    parser.add_argument("--v11-long-weight", type=float, default=1.0)
    parser.add_argument("--v11-slack-weight", type=float, default=1.0)
    parser.add_argument("--v11-overlap-slack-ratio", type=float, default=0.10)
    parser.add_argument("--v11-routing-cost-weight", type=float, default=0.50)
    parser.add_argument("--v12-routing-cost-weight", type=float, default=2.0)
    parser.add_argument("--v12-tp4-service-ratio-limit", type=float, default=0.60)
    parser.add_argument("--v13-latency-scale", type=float, default=1.0)
    parser.add_argument("--v13-long-threshold", type=int, default=12000)
    parser.add_argument("--v13-routing-cost-weight", type=float, default=2.0)
    parser.add_argument("--v13-tp4-service-ratio-limit", type=float, default=0.60)
    parser.add_argument("--v13-tp4-pressure-threshold", type=float, default=1.0)
    parser.add_argument("--schedule-interval-ms", type=float, default=5.0)
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument("--batch-max-tokens", type=int, default=16384)
    parser.add_argument("--max-total-tokens", type=int, default=70000)
    parser.add_argument("--latency-scale", type=float, default=1.0)
    parser.add_argument("--overload-policy", choices=("best_effort", "reject"), default="best_effort")
    decode_group = parser.add_mutually_exclusive_group()
    decode_group.add_argument(
        "--simulate-decode",
        action="store_true",
        help="enable the optional decode model (decode is instantaneous by default)",
    )
    decode_group.add_argument(
        "--fake-decode",
        action="store_true",
        help="skip Decode compute but retain fixed/per-token KV transfer communication delay",
    )
    decode_group.add_argument("--no-decode", action="store_false", dest="simulate_decode", help=argparse.SUPPRESS)
    parser.set_defaults(simulate_decode=False, fake_decode=False)
    parser.add_argument("--kv-transfer-fixed-ms", type=float, default=20.0)
    parser.add_argument("--kv-transfer-us-per-token", type=float, default=0.0)
    parser.add_argument("--decode-step-base-ms", type=float, default=6.0)
    parser.add_argument("--decode-step-per-request-ms", type=float, default=0.08)
    parser.add_argument("--decode-step-per-context-token-us", type=float, default=0.0)
    return parser


def _config_from_args(args) -> SimulationConfig:
    positive = {
        "slo_ttft": args.slo_ttft,
        "schedule_interval_ms": args.schedule_interval_ms,
        "chunked_prefill_size": args.chunked_prefill_size,
        "batch_max_tokens": args.batch_max_tokens,
        "max_total_tokens": args.max_total_tokens,
        "latency_scale": args.latency_scale,
        "naive_length_threshold": args.naive_length_threshold,
        "mps_overlap_slowdown": args.mps_overlap_slowdown,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"arguments must be positive: {', '.join(invalid)}")
    return SimulationConfig(
        scheduler="ours" if args.scheduler == "compare" else args.scheduler,
        naive_length_threshold=args.naive_length_threshold,
        mps_overlap_slowdown=args.mps_overlap_slowdown,
        slo_ttft_s=args.slo_ttft,
        bundle_window_s=args.bundle_window_ms / 1000.0,
        bundle_token_cap=args.bundle_token_cap,
        bundle_token_trigger=args.bundle_token_trigger,
        max_inflight=args.max_inflight,
        instance_token_credit=args.instance_token_credit,
        prediction_margin_s=args.prediction_margin_ms / 1000.0,
        v6_request_utility_tokens=args.v6_request_utility_tokens,
        v7_class_quantum_tokens=args.v7_class_quantum_tokens,
        v7_conflict_price_weight=args.v7_conflict_price_weight,
        v7_urgency_weight=args.v7_urgency_weight,
        v8_epoch_ms=args.v8_epoch_ms,
        v8_epoch_token_budget=args.v8_epoch_token_budget,
        v8_min_lane_quota=args.v8_min_lane_quota,
        v8_deadline_pressure_weight=args.v8_deadline_pressure_weight,
        v9_aging_interval_ms=args.v9_aging_interval_ms,
        v9_interactive_tokens=args.v9_interactive_tokens,
        v9_short_weight=args.v9_short_weight,
        v9_long_weight=args.v9_long_weight,
        v10_short_weight=args.v10_short_weight,
        v10_long_weight=args.v10_long_weight,
        v10_slack_weight=args.v10_slack_weight,
        v10_overlap_slack_ratio=args.v10_overlap_slack_ratio,
        v11_short_weight=args.v11_short_weight,
        v11_long_weight=args.v11_long_weight,
        v11_slack_weight=args.v11_slack_weight,
        v11_overlap_slack_ratio=args.v11_overlap_slack_ratio,
        v11_routing_cost_weight=args.v11_routing_cost_weight,
        v12_routing_cost_weight=args.v12_routing_cost_weight,
        v12_tp4_service_ratio_limit=args.v12_tp4_service_ratio_limit,
        v13_latency_scale=args.v13_latency_scale,
        v13_long_threshold=args.v13_long_threshold,
        v13_routing_cost_weight=args.v13_routing_cost_weight,
        v13_tp4_service_ratio_limit=args.v13_tp4_service_ratio_limit,
        v13_tp4_pressure_threshold=args.v13_tp4_pressure_threshold,
        schedule_interval_s=args.schedule_interval_ms / 1000.0,
        chunked_prefill_size=args.chunked_prefill_size,
        prefill_batch_max_tokens=args.batch_max_tokens,
        max_total_tokens=args.max_total_tokens,
        latency_scale=args.latency_scale,
        overload_policy=args.overload_policy,
        simulate_decode=args.simulate_decode,
        fake_decode=args.fake_decode,
        kv_transfer_fixed_s=args.kv_transfer_fixed_ms / 1000.0,
        kv_transfer_us_per_token=args.kv_transfer_us_per_token,
        decode_step_base_ms=args.decode_step_base_ms,
        decode_step_per_request_ms=args.decode_step_per_request_ms,
        decode_step_per_context_token_us=args.decode_step_per_context_token_us,
    )


COMPARISON_METRICS = (
    "completion_fraction",
    "completion_throughput_rps",
    "prefill_token_throughput_s",
    "ttft_mean_s",
    "ttft_p50_s",
    "ttft_p95_s",
    "ttft_p99_s",
    "ttft_slo_attainment",
    "offered_ttft_slo_attainment",
    "on_time_goodput_rps",
    "on_time_input_token_goodput_s",
    "offered_window_on_time_goodput_rps",
    "offered_window_on_time_input_token_goodput_s",
    "short_ttft_p95_s",
    "long_ttft_p95_s",
    "master_wait_p95_s",
    "transport_wait_p95_s",
    "worker_wait_p95_s",
    "prefill_execution_p95_s",
    "prefill_step_mean_batch",
    "prefill_step_max_batch",
    "mps_overlap_wall_s",
    "exclusive_prefill_gpu_s",
    "process_prefill_gpu_s",
    "physical_prefill_gpu_s",
    "physical_gpu_s_per_completed_request",
    "physical_gpu_s_per_on_time_request",
    "tp2_request_share",
    "tp4_request_share",
    "makespan_s",
)


def _comparison_record(dataset: str, rate: float, seed: int, summaries: Dict[str, Dict]) -> Dict:
    ours = summaries["ours"]
    naive = summaries["naive"]
    record = {
        "dataset": dataset,
        "request_rate": rate,
        "seed": seed,
        "ours": {metric: ours[metric] for metric in COMPARISON_METRICS},
        "naive": {metric: naive[metric] for metric in COMPARISON_METRICS},
        "ours_minus_naive": {
            metric: ours[metric] - naive[metric] for metric in COMPARISON_METRICS
        },
        "ours_over_naive": {
            metric: ours[metric] / naive[metric] if naive[metric] else None
            for metric in COMPARISON_METRICS
        },
    }
    return record


def _comparison_csv_row(record: Dict) -> Dict:
    row = {
        "dataset": record["dataset"],
        "request_rate": record["request_rate"],
        "seed": record["seed"],
    }
    for section in ("ours", "naive", "ours_minus_naive", "ours_over_naive"):
        for metric, value in record[section].items():
            row[f"{section}.{metric}"] = value
    return row


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        _run_self_test()
        return 0
    if not args.verbose_scheduler:
        _quiet_production_loggers()

    base_config = _config_from_args(args)
    schedulers = ("ours", "naive") if args.scheduler == "compare" else (args.scheduler,)
    datasets = (
        ["servegen-mm-image", "synthetic-5pct"]
        if args.dataset == "all"
        else [args.dataset]
    )
    repo_root = _repo_root()
    all_summaries = []
    comparisons = []
    for dataset in datasets:
        for rate in args.rates:
            if dataset == "servegen-mm-image":
                workload = generate_servegen_mm_image(
                    repo_root=repo_root,
                    request_rate=rate,
                    duration_s=args.duration,
                    seed=args.seed,
                )
            else:
                workload = generate_synthetic_5pct(
                    request_rate=rate,
                    num_prompts=args.num_prompts,
                    seed=args.seed,
                )
            paired_summaries = {}
            for scheduler in schedulers:
                config = dataclasses.replace(base_config, scheduler=scheduler)
                run_name = f"{dataset}.{scheduler}.rate-{rate:g}.seed-{args.seed}"
                with FlexTPStepSimulator(workload, config) as simulator:
                    summary = simulator.run()
                    summary.update(
                        {
                            "dataset": dataset,
                            "request_rate": rate,
                            "seed": args.seed,
                            "config": dataclasses.asdict(config),
                        }
                    )
                    simulator.write_artifacts(
                        args.output_dir,
                        run_name,
                        args.trace_steps,
                        summary=summary,
                    )
                paired_summaries[scheduler] = summary
                all_summaries.append(summary)
                print(
                    f"{dataset} scheduler={scheduler} rate={rate:g} "
                    f"offered={summary['offered_requests']} "
                    f"completed={summary['completed_requests']} rejected={summary['rejected_requests']} "
                    f"throughput={summary['completion_throughput_rps']:.3f}r/s "
                    f"ttft_p95={summary['ttft_p95_s']:.3f}s "
                    f"slo={summary['ttft_slo_attainment']:.3f} "
                    f"dispatch_mean={summary['bundle_mean_requests']:.2f} "
                    f"prefill_step_mean={summary['prefill_step_mean_batch']:.2f}"
                )
            if len(paired_summaries) == 2:
                comparison = _comparison_record(dataset, rate, args.seed, paired_summaries)
                comparisons.append(comparison)
                ours = paired_summaries["ours"]
                naive = paired_summaries["naive"]
                print(
                    f"COMPARE {dataset} rate={rate:g} "
                    f"p95 ours/naive={ours['ttft_p95_s']:.3f}/{naive['ttft_p95_s']:.3f}s "
                    f"SLO ours/naive={ours['ttft_slo_attainment']:.3f}/{naive['ttft_slo_attainment']:.3f} "
                    f"completion ours/naive={ours['completion_fraction']:.3f}/{naive['completion_fraction']:.3f}"
                )

    aggregate_path = args.output_dir / "all_runs.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(all_summaries, indent=2, sort_keys=True) + "\n")
    print(f"wrote {aggregate_path}")
    if comparisons:
        comparison_path = args.output_dir / "comparisons.json"
        comparison_path.write_text(json.dumps(comparisons, indent=2, sort_keys=True) + "\n")
        comparison_rows = [_comparison_csv_row(record) for record in comparisons]
        comparison_csv_path = args.output_dir / "comparisons.csv"
        with comparison_csv_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(comparison_rows[0]))
            writer.writeheader()
            writer.writerows(comparison_rows)
        print(f"wrote {comparison_path}")
        print(f"wrote {comparison_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
