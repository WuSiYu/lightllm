"""Independent epoch/wavefront scheduler for simultaneous TP-SMT Prefill.

V8 is a from-scratch deterministic policy.  It has no inheritance relationship
with V3, V4, V5, V6, or V7 and does not perform per-candidate admission search.
Requests enter one of two hard-routed lanes.  At a fixed epoch boundary the
scheduler computes weighted max-min token quotas for the lanes, drains each
lane by EDF, and distributes the selected work over the lane's TP pool.

Both lanes are open in the same epoch.  An epoch is a bounded wave of work, not
a MPS mode switch; TP2 and TP4 can therefore execute simultaneously throughout
the normal-PD lifecycle.
"""

from __future__ import annotations

import asyncio
import collections
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Literal, Optional, Sequence, Tuple

from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.log_utils import init_logger

from .pd_selector import PDSelector


logger = init_logger(__name__)
_ALL_GENERATIONS = object()
Lane = Literal["short", "long"]


@dataclass
class V8Job:
    internal_id: int
    external_req_id: Optional[int]
    seq_len: int
    deadline: float
    future: asyncio.Future
    enqueue_order: int


@dataclass
class V8Lease:
    internal_id: int
    external_req_id: Optional[int]
    seq_len: int
    deadline: float
    lane: Lane
    node_key: str
    tp_size: int
    predicted_finish: float
    dispatch_time: float
    instance_generation: Optional[str]
    epoch_id: int
    accepted: bool = False
    bundle_id: Optional[int] = None
    state: str = "admitted"


@dataclass
class V8LaneInstance:
    node: PD_Client_Obj
    node_key: str
    group_id: str
    tp_size: int
    gpu_set: FrozenSet[str]
    explicit_placement: bool
    instance_generation: Optional[str]
    available: bool = True
    leases: "collections.OrderedDict[int, V8Lease]" = field(default_factory=collections.OrderedDict)
    virtual_load: float = 0.0
    worker_queued_requests: int = 0
    worker_queued_tokens: int = 0
    worker_running_requests: int = 0
    worker_running_tokens: int = 0
    worker_load: float = 0.0
    last_report_seq: int = -1

    @property
    def admitted_tokens(self) -> int:
        return sum(lease.seq_len for lease in self.leases.values())

    @property
    def busy(self) -> bool:
        return bool(self.leases)

    @property
    def accepted_bundle_count(self) -> int:
        return len(
            {
                lease.bundle_id
                for lease in self.leases.values()
                if lease.accepted and lease.bundle_id is not None
            }
        )


class V8Profile:
    """Local service curve used to size each bounded epoch."""

    COEFFICIENTS: Dict[int, Tuple[float, float, float, float]] = {
        1: (2.4e-4, 1e-8, 0.05, 0.05),
        2: (2.018352e-4, 6.048427e-9, 2.341091e-2, 6.794939e-2),
        4: (2.655797e-4, 2.330274e-9, 5.651746e-2, 3.767872e-2),
    }

    def batch(self, lengths: Sequence[int], tp_size: int) -> float:
        if not lengths:
            return 0.0
        a, b, floor, fixed = self.COEFFICIENTS.get(
            tp_size,
            self.COEFFICIENTS[min(self.COEFFICIENTS, key=lambda value: abs(value - tp_size))],
        )
        values = [max(1, int(value)) for value in lengths]
        return max(
            a * sum(values) / tp_size + b * sum(value * value for value in values) / tp_size,
            floor,
        ) + fixed

    def predict_batch(self, lengths: Sequence[int], tp_size: int) -> float:
        return self.batch(lengths, tp_size)


class FlexTPSelectorV8(PDSelector):
    """Deterministic weighted-epoch allocator for two simultaneous TP lanes."""

    supports_bundles = True
    uses_prefill_lease_lifecycle = True
    is_flex_tp_v8 = True
    pending_type = V8Job

    def __init__(
        self,
        pd_manager,
        slo_ttft: Optional[float] = 5.0,
        *,
        long_request_threshold: int = 4000,
        batch_token_cap: int = 8192,
        batch_token_trigger: int = 4096,
        max_inflight_per_instance: int = 64,
        max_admitted_tokens_per_instance: int = 16384,
        batch_window_s: float = 0.020,
        prediction_margin_s: float = 0.080,
        replan_interval_s: float = 0.050,
        mode_slowdowns: Optional[Dict[Tuple[int, Tuple[int, ...]], float]] = None,
        mps_overlap_slowdown: float = 2.0,
        max_accepted_bundles_per_instance: int = 64,
        overload_policy: str = "best_effort",
        epoch_s: float = 0.050,
        epoch_token_budget: int = 16384,
        min_lane_quota_tokens: int = 2048,
        deadline_pressure_weight: float = 2.0,
        short_lane_weight: float = 1.0,
        long_lane_weight: float = 1.0,
    ) -> None:
        super().__init__(pd_manager)
        self.slo_ttft = 5.0 if slo_ttft is None else float(slo_ttft)
        if not math.isfinite(self.slo_ttft) or self.slo_ttft <= 0:
            raise ValueError("slo_ttft must be a positive finite number")
        if overload_policy not in ("best_effort", "reject"):
            raise ValueError("overload_policy must be 'best_effort' or 'reject'")
        self.long_request_threshold = max(1, int(long_request_threshold))
        self.batch_token_cap = max(1, int(batch_token_cap))
        self.batch_token_trigger = max(1, min(int(batch_token_trigger), self.batch_token_cap))
        self.max_inflight_per_instance = max(1, int(max_inflight_per_instance))
        self.token_credit = max(1, int(max_admitted_tokens_per_instance))
        self.batch_window_s = max(0.0, float(batch_window_s))
        self.prediction_margin_s = max(0.0, float(prediction_margin_s))
        self.replan_interval_s = max(0.005, float(replan_interval_s))
        self.max_accepted_bundles_per_instance = max(1, int(max_accepted_bundles_per_instance))
        self.overload_policy = overload_policy
        self.epoch_s = max(0.005, float(epoch_s))
        self.epoch_token_budget = max(1, int(epoch_token_budget))
        self.min_lane_quota_tokens = max(1, int(min_lane_quota_tokens))
        self.deadline_pressure_weight = max(0.0, float(deadline_pressure_weight))
        self.lane_weights: Dict[Lane, float] = {
            "short": max(0.01, float(short_lane_weight)),
            "long": max(0.01, float(long_lane_weight)),
        }
        self.mps_overlap_slowdown = max(1.0, float(mps_overlap_slowdown))
        self.mode_slowdowns = {
            (int(tp), tuple(sorted(signature))): max(1.0, float(value))
            for (tp, signature), value in (mode_slowdowns or {}).items()
        }
        self.profile = V8Profile()
        # The instance harness consumes the same timing contract as the
        # production selector, while the profile implementation is V8-local.
        self.latency_model = self.profile

        self.instances: Dict[str, V8LaneInstance] = {}
        self.groups: Dict[str, Dict[str, V8LaneInstance]] = {}
        self.node_to_group: Dict[str, Dict[str, V8LaneInstance]] = {}
        self.node_tp_size: Dict[str, int] = {}
        self._pending: "collections.OrderedDict[int, V8Job]" = collections.OrderedDict()
        self._leases: Dict[int, V8Lease] = {}
        self._external_to_internal: Dict[int, int] = {}
        self._request_counter = 0
        self._enqueue_counter = 0
        self._decode_rr_index = 0
        self._epoch_id = -1
        self._next_epoch_at = 0.0
        self._lane_deficit: Dict[Lane, float] = {"short": 0.0, "long": 0.0}
        self._lock = asyncio.Lock()
        self._owner_loop: Optional[asyncio.AbstractEventLoop] = None
        self._replan_task: Optional[asyncio.Task] = None

        self.epoch_count = 0
        self.lane_quota_tokens = collections.Counter()
        self.admitted_by_lane = collections.Counter()
        self.admitted_by_epoch = collections.Counter()
        self.credit_blocked = 0
        self.deadline_overrides = 0

    @staticmethod
    def _args(node: PD_Client_Obj) -> Dict:
        return node.start_args if isinstance(node.start_args, dict) else {}

    @staticmethod
    def _host(node: PD_Client_Obj) -> str:
        return str(node.client_ip_port).split(":", 1)[0]

    @staticmethod
    def _parse_gpu_ids(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set, frozenset)):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, int):
            return [str(value)]
        return [item for item in re.split(r"[,;\s]+", str(value).strip()) if item]

    def _group_id(self, node: PD_Client_Obj) -> str:
        args = self._args(node)
        host = self._host(node)
        explicit = args.get("tp_smt_group_id") or args.get("flex_tp_group_id")
        if explicit is not None:
            return f"{host}:{explicit}"
        port = args.get("shared_weight_master_port_start")
        if args.get("shared_weight") and port is not None:
            return f"{host}:{port}"
        return f"{host}:ungrouped"

    def _gpu_set(self, node: PD_Client_Obj) -> Tuple[FrozenSet[str], bool]:
        values = self._parse_gpu_ids(self._args(node).get("tp_smt_gpu_ids"))
        host = self._host(node)
        if values:
            return frozenset(f"{host}/gpu:{value}" for value in values), True
        return frozenset({f"unknown-placement:{host}"}), False

    @staticmethod
    def _overlap(left: V8LaneInstance, right: V8LaneInstance) -> bool:
        return bool(left.gpu_set & right.gpu_set)

    def update_nodes(self, prefill_nodes, decode_nodes) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._owner_loop is not None and loop is not self._owner_loop:
            self._owner_loop.call_soon_threadsafe(self.update_nodes, prefill_nodes, decode_nodes)
            return
        normal_prefill = [node for node in prefill_nodes if node.mode == "prefill"]
        normal_decode = [node for node in decode_nodes if node.mode == "decode"]
        super().update_nodes(normal_prefill, normal_decode)
        for instance in self.instances.values():
            instance.available = False
        for node in normal_prefill:
            key = str(node.client_ip_port)
            group_id = self._group_id(node)
            tp_size = int(self._args(node).get("tp", 1))
            gpu_set, explicit = self._gpu_set(node)
            instance = self.instances.get(key)
            if instance is None:
                instance = V8LaneInstance(
                    node=node,
                    node_key=key,
                    group_id=group_id,
                    tp_size=tp_size,
                    gpu_set=gpu_set,
                    explicit_placement=explicit,
                    instance_generation=getattr(node, "instance_generation", None),
                )
                self.instances[key] = instance
            elif instance.busy and (
                instance.group_id != group_id
                or instance.tp_size != tp_size
                or instance.gpu_set != gpu_set
            ):
                logger.warning("keep busy V8 instance %s on old topology", key)
                continue
            else:
                if instance.instance_generation != getattr(node, "instance_generation", None):
                    instance.last_report_seq = -1
                instance.node = node
                instance.group_id = group_id
                instance.tp_size = tp_size
                instance.gpu_set = gpu_set
                instance.explicit_placement = explicit
                instance.instance_generation = getattr(node, "instance_generation", None)
            instance.available = True
            if not explicit:
                logger.warning("V8 instance %s has no explicit GPU placement; assume overlap", key)
        for key, instance in list(self.instances.items()):
            if not instance.available and not instance.busy:
                del self.instances[key]
        grouped: Dict[str, Dict[str, V8LaneInstance]] = collections.defaultdict(dict)
        for instance in self.instances.values():
            grouped[instance.group_id][instance.node_key] = instance
        self.groups = dict(grouped)
        self.node_to_group = {
            key: self.groups[instance.group_id] for key, instance in self.instances.items()
        }
        self.node_tp_size = {key: instance.tp_size for key, instance in self.instances.items()}
        self._fail_unserviceable_pending()
        self._request_reschedule()

    def _bounds(self) -> Tuple[int, int]:
        values = [instance.tp_size for instance in self.instances.values() if instance.available]
        return (min(values), max(values)) if values else (1, 1)

    def _lane(self, seq_len: int) -> Lane:
        return "long" if int(seq_len) > self.long_request_threshold else "short"

    def _targets(self, lane: Lane) -> List[V8LaneInstance]:
        low, high = self._bounds()
        target_tp = high if lane == "long" and low != high else low
        return sorted(
            [
                instance
                for instance in self.instances.values()
                if instance.available and instance.tp_size == target_tp
            ],
            key=lambda instance: instance.node_key,
        )

    def _active(self, include: Optional[V8LaneInstance] = None) -> List[V8LaneInstance]:
        active = [instance for instance in self.instances.values() if instance.busy]
        if include is not None and include not in active:
            active.append(include)
        return active

    def _signature(self, target: V8LaneInstance, active: Sequence[V8LaneInstance]) -> Tuple[int, ...]:
        return tuple(
            sorted(
                [target.tp_size]
                + [instance.tp_size for instance in active if instance is not target and self._overlap(target, instance)]
            )
        )

    def _slowdown(self, target: V8LaneInstance, active: Sequence[V8LaneInstance]) -> float:
        signature = self._signature(target, active)
        if len(signature) <= 1:
            return 1.0
        return self.mode_slowdowns.get((target.tp_size, signature), self.mps_overlap_slowdown)

    def _pending_lane(self, lane: Lane) -> List[V8Job]:
        return sorted(
            [job for job in self._pending.values() if self._lane(job.seq_len) == lane],
            key=lambda job: (job.deadline, job.enqueue_order),
        )

    def _lane_demand(self, lane: Lane, now: float) -> float:
        jobs = self._pending_lane(lane)
        if not jobs:
            return 0.0
        backlog = sum(job.seq_len for job in jobs)
        oldest_slack = jobs[0].deadline - now
        urgency = max(0.0, min(3.0, (self.slo_ttft - oldest_slack) / max(self.slo_ttft, 1e-6)))
        resident = sum(instance.admitted_tokens for instance in self._targets(lane))
        return (
            backlog + 0.25 * resident
        ) * (1.0 + self.deadline_pressure_weight * urgency) * self.lane_weights[lane]

    def _compute_quotas(self, now: float) -> Dict[Lane, int]:
        active_lanes = [lane for lane in ("short", "long") if self._pending_lane(lane)]
        if not active_lanes:
            return {"short": 0, "long": 0}
        for lane in active_lanes:
            self._lane_deficit[lane] = min(
                4.0 * self.epoch_token_budget,
                self._lane_deficit[lane] + self.epoch_token_budget / len(active_lanes),
            )
        demand = {lane: self._lane_demand(lane, now) + self._lane_deficit[lane] for lane in active_lanes}
        total_budget = self.epoch_token_budget
        if len(active_lanes) == 1:
            quotas = {active_lanes[0]: total_budget}
        else:
            base = min(self.min_lane_quota_tokens, total_budget // len(active_lanes))
            remaining = total_budget - base * len(active_lanes)
            total_demand = sum(demand.values())
            quotas = {
                lane: base + int(remaining * demand[lane] / max(total_demand, 1.0))
                for lane in active_lanes
            }
            # Integer rounding must not silently lose the epoch budget.
            while sum(quotas.values()) < total_budget:
                lane = max(active_lanes, key=lambda item: demand[item] - quotas[item])
                quotas[lane] += 1
        for lane in ("short", "long"):
            quotas.setdefault(lane, 0)
            self.lane_quota_tokens[lane] += quotas[lane]
        return quotas

    def _accepts(self, instance: V8LaneInstance, job: V8Job) -> bool:
        if not instance.available:
            return False
        if len(instance.leases) >= self.max_inflight_per_instance:
            return False
        if instance.accepted_bundle_count >= self.max_accepted_bundles_per_instance:
            return False
        if instance.leases and instance.admitted_tokens + job.seq_len > self.token_credit:
            self.credit_blocked += 1
            return False
        return True

    def _pack(self, entries: Sequence[Tuple[int, int, float]]) -> List[List[Tuple[int, int, float]]]:
        batches: List[List[Tuple[int, int, float]]] = []
        current: List[Tuple[int, int, float]] = []
        tokens = 0
        for entry in entries:
            if current and tokens + entry[1] > self.batch_token_cap:
                batches.append(current)
                current = []
                tokens = 0
            current.append(entry)
            tokens += entry[1]
        if current:
            batches.append(current)
        return batches

    def _finish_prediction(self, now: float, instance: V8LaneInstance, job: V8Job) -> float:
        entries = [(lease.internal_id, lease.seq_len, lease.deadline) for lease in instance.leases.values()]
        entries.append((job.internal_id, job.seq_len, job.deadline))
        service = sum(
            self.profile.batch([entry[1] for entry in batch], instance.tp_size)
            for batch in self._pack(entries)
        )
        return max(now, instance.virtual_load) + service * self._slowdown(instance, self._active(instance)) + self.batch_window_s + self.prediction_margin_s

    def _pick_decode(self) -> PD_Client_Obj:
        if not self.decode_nodes:
            raise RuntimeError("no Decode node is registered")
        node = self.decode_nodes[self._decode_rr_index % len(self.decode_nodes)]
        self._decode_rr_index += 1
        return node

    def _pick_instance(self, lane: Lane, job: V8Job, planned: Dict[str, int]) -> Optional[V8LaneInstance]:
        options = [instance for instance in self._targets(lane) if self._accepts(instance, job)]
        if not options:
            return None
        return min(options, key=lambda instance: (planned[instance.node_key], instance.admitted_tokens, instance.node_key))

    def _admit(self, now: float, job: V8Job, instance: V8LaneInstance, epoch_id: int, planned: Dict[str, int]) -> None:
        self._pending.pop(job.internal_id, None)
        predicted = self._finish_prediction(now, instance, job)
        lane = self._lane(job.seq_len)
        lease = V8Lease(
            internal_id=job.internal_id,
            external_req_id=job.external_req_id,
            seq_len=job.seq_len,
            deadline=job.deadline,
            lane=lane,
            node_key=instance.node_key,
            tp_size=instance.tp_size,
            predicted_finish=predicted,
            dispatch_time=now,
            instance_generation=instance.instance_generation,
            epoch_id=epoch_id,
        )
        instance.leases[lease.internal_id] = lease
        instance.virtual_load = max(now, instance.virtual_load) + self.profile.batch([job.seq_len], instance.tp_size)
        planned[instance.node_key] += job.seq_len
        self._leases[lease.internal_id] = lease
        if lease.external_req_id is not None:
            self._external_to_internal[lease.external_req_id] = lease.internal_id
        self.admitted_by_lane[lane] += 1
        self.admitted_by_epoch[epoch_id] += 1
        self._lane_deficit[lane] -= job.seq_len
        if predicted > job.deadline:
            self.deadline_overrides += 1
        if not job.future.done():
            job.future.set_result((instance.node, self._pick_decode()))

    def _drain_lane(self, now: float, lane: Lane, quota: int, epoch_id: int) -> int:
        if quota <= 0:
            return 0
        planned = {instance.node_key: instance.admitted_tokens for instance in self._targets(lane)}
        used = 0
        for job in self._pending_lane(lane):
            if used and used + job.seq_len > quota:
                break
            instance = self._pick_instance(lane, job, planned)
            if instance is None:
                continue
            self._admit(now, job, instance, epoch_id, planned)
            used += job.seq_len
        return used

    def _run_epoch(self, now: float) -> None:
        self._epoch_id += 1
        self.epoch_count += 1
        quotas = self._compute_quotas(now)
        # The two lane drains are deliberately independent and happen in one
        # epoch, so an active TP2 and TP4 are visible to the worker together.
        for lane in ("short", "long"):
            used = self._drain_lane(now, lane, quotas[lane], self._epoch_id)
            self.lane_quota_tokens[f"used_{lane}"] += used
        self._next_epoch_at = now + self.epoch_s

    def _schedule_locked(self, now: float) -> None:
        if not self.decode_nodes or not any(instance.available for instance in self.instances.values()):
            self._fail_unserviceable_pending()
            return
        if now + 1e-9 < self._next_epoch_at:
            return
        self._run_epoch(now)
        if self._pending and self.overload_policy == "reject":
            for job in list(self._pending.values()):
                if job.deadline <= now:
                    self._pending.pop(job.internal_id, None)
                    if not job.future.done():
                        job.future.set_exception(RuntimeError("FlexTP V8 rejected request after deadline"))

    def _fail_unserviceable_pending(self) -> None:
        if self.decode_nodes and any(instance.available for instance in self.instances.values()):
            return
        reason = "no Decode node is registered" if not self.decode_nodes else "no available Prefill instance is registered"
        for internal_id, job in list(self._pending.items()):
            self._pending.pop(internal_id, None)
            if not job.future.done():
                job.future.set_exception(RuntimeError(f"FlexTP V8: {reason}"))

    def _request_reschedule(self) -> None:
        if self._owner_loop is not None and (self._replan_task is None or self._replan_task.done()):
            self._replan_task = self._owner_loop.create_task(self._replan_loop())

    async def _replan_loop(self) -> None:
        while True:
            await asyncio.sleep(self.replan_interval_s)
            async with self._lock:
                self._schedule_locked(time.time())
                if not self._pending:
                    return

    async def async_select_p_d_node(
        self,
        prompt,
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        input_token_num: Optional[int] = None,
        arrival_time: Optional[float] = None,
        req_id: Optional[int] = None,
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        loop = asyncio.get_running_loop()
        if self._owner_loop is None:
            self._owner_loop = loop
        elif self._owner_loop is not loop:
            raise RuntimeError("FlexTP V8 cannot be used across event loops")
        now = time.time()
        self._request_counter += 1
        self._enqueue_counter += 1
        internal_id = self._request_counter
        job = V8Job(
            internal_id=internal_id,
            external_req_id=req_id,
            seq_len=max(1, int(input_token_num or 1)),
            deadline=(arrival_time if arrival_time is not None else now) + self.slo_ttft,
            future=loop.create_future(),
            enqueue_order=self._enqueue_counter,
        )
        async with self._lock:
            if req_id is not None and (
                req_id in self._external_to_internal
                or any(item.external_req_id == req_id for item in self._pending.values())
            ):
                raise ValueError(f"FlexTP V8 req_id={req_id} is already admitted or pending")
            self._pending[internal_id] = job
            self._fail_unserviceable_pending()
            self._schedule_locked(now)
            if not job.future.done():
                self._request_reschedule()
        try:
            return await job.future
        except asyncio.CancelledError:
            async with self._lock:
                self._pending.pop(internal_id, None)
                lease = self._leases.pop(internal_id, None)
                if lease is not None:
                    instance = self.instances.get(lease.node_key)
                    if instance is not None:
                        instance.leases.pop(internal_id, None)
                    if lease.external_req_id is not None:
                        self._external_to_internal.pop(lease.external_req_id, None)
                self._schedule_locked(time.time())
            raise

    async def select_p_d_node(self, *args, **kwargs):
        return await self.async_select_p_d_node(*args, **kwargs)

    def _find_lease(self, p_node: Optional[PD_Client_Obj], req_id: Optional[int]) -> Optional[V8Lease]:
        if p_node is None:
            return None
        if req_id is None:
            instance = self.instances.get(str(p_node.client_ip_port))
            if instance is None or len(instance.leases) != 1:
                return None
            lease = next(iter(instance.leases.values()))
        else:
            internal_id = self._external_to_internal.get(req_id)
            lease = self._leases.get(internal_id) if internal_id is not None else None
        if lease is None or lease.node_key != str(p_node.client_ip_port):
            return None
        generation = getattr(p_node, "instance_generation", None)
        if generation is not None and generation != lease.instance_generation:
            return None
        return lease

    async def _finish(self, p_node: Optional[PD_Client_Obj], *, req_id: Optional[int], failed: bool, reason: str = "") -> None:
        async with self._lock:
            lease = self._find_lease(p_node, req_id)
            if lease is None:
                logger.warning("FlexTP V8 completion for unknown req=%s node=%s", req_id, getattr(p_node, "client_ip_port", None))
                return
            lease.state = "failed" if failed else "finished"
            self._leases.pop(lease.internal_id, None)
            if lease.external_req_id is not None:
                self._external_to_internal.pop(lease.external_req_id, None)
            instance = self.instances.get(lease.node_key)
            if instance is not None:
                instance.leases.pop(lease.internal_id, None)
                if not instance.leases:
                    instance.virtual_load = 0.0
            if failed:
                logger.warning("FlexTP V8 failed req=%s node=%s reason=%s", lease.external_req_id, lease.node_key, reason)
            self._schedule_locked(time.time())
            self._request_reschedule()

    async def notify_request_done(self, p_node, input_token_num: int = 0, actual_ttft: Optional[float] = None, req_id: Optional[int] = None) -> None:
        await self._finish(p_node, req_id=req_id, failed=False)

    async def notify_request_failed(self, p_node, input_token_num: int = 0, req_id: Optional[int] = None, reason: str = "") -> None:
        await self._finish(p_node, req_id=req_id, failed=True, reason=reason)

    async def notify_bundle_accepted(self, p_node, bundle_id: int, req_ids: Sequence[int]) -> None:
        async with self._lock:
            for req_id in req_ids:
                lease = self._find_lease(p_node, req_id)
                if lease is not None:
                    lease.accepted = True
                    lease.bundle_id = int(bundle_id)
                    lease.state = "queued_worker"

    async def update_instance_report(self, node_key: str, report: Dict, report_seq: Optional[int] = None) -> None:
        async with self._lock:
            instance = self.instances.get(str(node_key))
            if instance is None:
                return
            if report_seq is not None:
                report_seq = int(report_seq)
                if report_seq <= instance.last_report_seq:
                    return
                instance.last_report_seq = report_seq
            instance.worker_queued_requests = int(report.get("queued_requests", 0))
            instance.worker_queued_tokens = int(report.get("queued_tokens", 0))
            instance.worker_running_requests = int(report.get("running_requests", 0))
            instance.worker_running_tokens = int(report.get("running_tokens", 0))
            instance.worker_load = float(report.get("total_token_usage_rate", instance.worker_load))

    def notify_node_removed(self, node_key: str, instance_generation=_ALL_GENERATIONS) -> None:
        if self._owner_loop is not None and not self._owner_loop.is_closed():
            self._owner_loop.create_task(self._release_node(str(node_key), instance_generation))

    async def _release_node(self, node_key: str, instance_generation=_ALL_GENERATIONS) -> None:
        async with self._lock:
            instance = self.instances.get(node_key)
            if instance is None:
                return
            for lease in list(instance.leases.values()):
                if instance_generation is not _ALL_GENERATIONS and lease.instance_generation != instance_generation:
                    continue
                instance.leases.pop(lease.internal_id, None)
                self._leases.pop(lease.internal_id, None)
                if lease.external_req_id is not None:
                    self._external_to_internal.pop(lease.external_req_id, None)
            if not instance.leases:
                instance.virtual_load = 0.0
            self._schedule_locked(time.time())

    def snapshot(self) -> Dict[str, Dict]:
        return {
            key: {
                "group_id": instance.group_id,
                "tp": instance.tp_size,
                "gpus": sorted(instance.gpu_set),
                "leases": len(instance.leases),
                "accepted_bundle_count": instance.accepted_bundle_count,
                "admitted_tokens": instance.admitted_tokens,
                "virtual_load": instance.virtual_load,
                "worker_queued_requests": instance.worker_queued_requests,
                "worker_queued_tokens": instance.worker_queued_tokens,
                "worker_running_requests": instance.worker_running_requests,
                "worker_running_tokens": instance.worker_running_tokens,
                "worker_load": instance.worker_load,
            }
            for key, instance in self.instances.items()
        }

    def scheduler_snapshot(self) -> Dict:
        return {
            "version": 8,
            "policy": "deterministic_weighted_epoch_wavefront",
            "long_request_threshold": self.long_request_threshold,
            "token_credit": self.token_credit,
            "epoch_s": self.epoch_s,
            "epoch_token_budget": self.epoch_token_budget,
            "min_lane_quota_tokens": self.min_lane_quota_tokens,
            "deadline_pressure_weight": self.deadline_pressure_weight,
            "lane_weights": dict(self.lane_weights),
            "epoch_id": self._epoch_id,
            "epoch_count": self.epoch_count,
            "lane_quota_tokens": dict(self.lane_quota_tokens),
            "admitted_by_lane": dict(sorted(self.admitted_by_lane.items())),
            "admitted_by_epoch": dict(self.admitted_by_epoch),
            "credit_blocked": self.credit_blocked,
            "deadline_overrides": self.deadline_overrides,
            "mps_overlap_slowdown": self.mps_overlap_slowdown,
        }
