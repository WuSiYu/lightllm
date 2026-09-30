"""Independent online flow scheduler for TP-SMT Prefill.

V7 is a from-scratch policy.  It does not inherit a FlexTP scheduler and does
not enumerate V3/V4/V5/V6 candidates.  A request is routed by a hard length
partition, then admitted by an online marginal-cost rule.  Every resident
instance has a virtual finish clock; overlapping instances add a GPU-conflict
price, while each request class carries a deficit debt for max-min fairness.

The policy is deliberately work-conserving: TP2 and TP4 remain concurrently
active whenever their queues and credits permit it.  The price is a scheduling
cost, not a command to turn CUDA MPS on or off.
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
RequestClass = Literal["short", "long"]


@dataclass
class V7Request:
    internal_id: int
    external_req_id: Optional[int]
    seq_len: int
    deadline: float
    future: asyncio.Future
    enqueue_order: int


@dataclass
class V7Lease:
    internal_id: int
    external_req_id: Optional[int]
    seq_len: int
    deadline: float
    request_class: RequestClass
    node_key: str
    tp_size: int
    predicted_finish: float
    dispatch_time: float
    instance_generation: Optional[str]
    accepted: bool = False
    bundle_id: Optional[int] = None
    state: str = "admitted"


@dataclass
class V7Instance:
    node: PD_Client_Obj
    node_key: str
    group_id: str
    tp_size: int
    gpu_set: FrozenSet[str]
    explicit_placement: bool
    instance_generation: Optional[str]
    available: bool = True
    leases: "collections.OrderedDict[int, V7Lease]" = field(default_factory=collections.OrderedDict)
    virtual_finish: float = 0.0
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


class V7ServiceCurve:
    """Common measured curve, kept local so V7 has no scheduler dependency."""

    CONSTANTS: Dict[int, Tuple[float, float, float, float]] = {
        1: (2.4e-4, 1e-8, 0.05, 0.05),
        2: (2.018352e-4, 6.048427e-9, 2.341091e-2, 6.794939e-2),
        4: (2.655797e-4, 2.330274e-9, 5.651746e-2, 3.767872e-2),
    }

    def predict(self, lengths: Sequence[int], tp_size: int) -> float:
        if not lengths:
            return 0.0
        coefficients = self.CONSTANTS.get(tp_size)
        if coefficients is None:
            coefficients = self.CONSTANTS[min(self.CONSTANTS, key=lambda value: abs(value - tp_size))]
        a, b, floor, fixed = coefficients
        values = [max(1, int(length)) for length in lengths]
        return max(
            a * sum(values) / tp_size + b * sum(value * value for value in values) / tp_size,
            floor,
        ) + fixed

    def predict_batch(self, lengths: Sequence[int], tp_size: int) -> float:
        return self.predict(lengths, tp_size)


class FlexTPSelectorV7(PDSelector):
    """Online dual-price flow balancing for simultaneous TP-SMT instances."""

    supports_bundles = True
    uses_prefill_lease_lifecycle = True
    is_flex_tp_v7 = True
    pending_type = V7Request

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
        class_quantum_tokens: int = 4096,
        conflict_price_weight: float = 0.35,
        urgency_weight: float = 2.0,
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
        self.class_quantum_tokens = max(1, int(class_quantum_tokens))
        self.conflict_price_weight = max(0.0, float(conflict_price_weight))
        self.urgency_weight = max(0.0, float(urgency_weight))
        self.mps_overlap_slowdown = max(1.0, float(mps_overlap_slowdown))
        self.mode_slowdowns = {
            (int(tp), tuple(sorted(signature))): max(1.0, float(value))
            for (tp, signature), value in (mode_slowdowns or {}).items()
        }
        self.curve = V7ServiceCurve()
        # Keep the production selector/simulator timing contract explicit.
        self.latency_model = self.curve

        self.instances: Dict[str, V7Instance] = {}
        self.groups: Dict[str, Dict[str, V7Instance]] = {}
        self.node_to_group: Dict[str, Dict[str, V7Instance]] = {}
        self.node_tp_size: Dict[str, int] = {}
        self._pending: "collections.OrderedDict[int, V7Request]" = collections.OrderedDict()
        self._leases: Dict[int, V7Lease] = {}
        self._external_to_internal: Dict[int, int] = {}
        self._request_counter = 0
        self._enqueue_counter = 0
        self._decode_rr_index = 0
        self._lock = asyncio.Lock()
        self._owner_loop: Optional[asyncio.AbstractEventLoop] = None
        self._replan_task: Optional[asyncio.Task] = None
        self._last_price_time: Optional[float] = None
        self.class_debt: Dict[RequestClass, float] = {"short": 0.0, "long": 0.0}

        self.admitted_by_class = collections.Counter()
        self.admitted_by_instance = collections.Counter()
        self.flow_decisions = 0
        self.price_evaluations = 0
        self.credit_blocked = 0
        self.deadline_overrides = 0

    @staticmethod
    def _node_args(node: PD_Client_Obj) -> Dict:
        return node.start_args if isinstance(node.start_args, dict) else {}

    @staticmethod
    def _host(node: PD_Client_Obj) -> str:
        return str(node.client_ip_port).split(":", 1)[0]

    @staticmethod
    def _gpu_ids(value) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set, frozenset)):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, int):
            return [str(value)]
        return [item for item in re.split(r"[,;\s]+", str(value).strip()) if item]

    def _group(self, node: PD_Client_Obj) -> str:
        args = self._node_args(node)
        host = self._host(node)
        explicit = args.get("tp_smt_group_id") or args.get("flex_tp_group_id")
        if explicit is not None:
            return f"{host}:{explicit}"
        port = args.get("shared_weight_master_port_start")
        if args.get("shared_weight") and port is not None:
            return f"{host}:{port}"
        return f"{host}:ungrouped"

    def _placement(self, node: PD_Client_Obj) -> Tuple[FrozenSet[str], bool]:
        values = self._gpu_ids(self._node_args(node).get("tp_smt_gpu_ids"))
        host = self._host(node)
        if values:
            return frozenset(f"{host}/gpu:{value}" for value in values), True
        return frozenset({f"unknown-placement:{host}"}), False

    @staticmethod
    def _overlap(left: V7Instance, right: V7Instance) -> bool:
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
            group_id = self._group(node)
            tp_size = int(self._node_args(node).get("tp", 1))
            gpu_set, explicit = self._placement(node)
            instance = self.instances.get(key)
            if instance is None:
                instance = V7Instance(
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
                logger.warning("keep busy V7 instance %s on old topology", key)
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
                logger.warning("V7 instance %s has no explicit GPU placement; assume overlap", key)
        for key, instance in list(self.instances.items()):
            if not instance.available and not instance.busy:
                del self.instances[key]
        grouped: Dict[str, Dict[str, V7Instance]] = collections.defaultdict(dict)
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

    def _class(self, seq_len: int) -> RequestClass:
        return "long" if int(seq_len) > self.long_request_threshold else "short"

    def _targets(self, request_class: RequestClass) -> List[V7Instance]:
        low, high = self._bounds()
        target_tp = high if request_class == "long" and low != high else low
        return sorted(
            [
                instance
                for instance in self.instances.values()
                if instance.available and instance.tp_size == target_tp
            ],
            key=lambda instance: instance.node_key,
        )

    def _active(self, include: Optional[V7Instance] = None) -> List[V7Instance]:
        values = [instance for instance in self.instances.values() if instance.busy]
        if include is not None and include not in values:
            values.append(include)
        return values

    def _signature(self, target: V7Instance, active: Sequence[V7Instance]) -> Tuple[int, ...]:
        return tuple(
            sorted(
                [target.tp_size]
                + [instance.tp_size for instance in active if instance is not target and self._overlap(target, instance)]
            )
        )

    def _slowdown(self, target: V7Instance, active: Sequence[V7Instance]) -> float:
        signature = self._signature(target, active)
        if len(signature) <= 1:
            return 1.0
        return self.mode_slowdowns.get((target.tp_size, signature), self.mps_overlap_slowdown)

    def _conflict_price(self, target: V7Instance) -> float:
        price = 0.0
        for active in self._active():
            if active is target or not self._overlap(target, active):
                continue
            token_ratio = active.admitted_tokens / max(1, self.token_credit)
            run_ratio = min(1.0, max(0.0, active.worker_running_tokens / max(1, self.token_credit)))
            price += min(2.0, token_ratio + 0.5 * run_ratio)
        self.price_evaluations += 1
        return price

    def _backlog_pressure(self, request_class: RequestClass) -> float:
        target_tokens = sum(
            item.seq_len
            for item in self._pending.values()
            if self._class(item.seq_len) == request_class
        )
        resident = sum(instance.admitted_tokens for instance in self._targets(request_class))
        return (target_tokens + resident) / max(1, len(self._targets(request_class)) * self.token_credit)

    def _advance_debt(self, now: float) -> None:
        if self._last_price_time is None:
            self._last_price_time = now
            return
        elapsed = max(0.0, now - self._last_price_time)
        self._last_price_time = now
        for request_class in ("short", "long"):
            if self._backlog_pressure(request_class) > 0:
                self.class_debt[request_class] += self.class_quantum_tokens * min(4.0, elapsed / max(self.replan_interval_s, 1e-6))
            self.class_debt[request_class] = max(-4.0 * self.class_quantum_tokens, min(8.0 * self.class_quantum_tokens, self.class_debt[request_class]))

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

    def _projected_finish(self, now: float, instance: V7Instance, request: V7Request) -> float:
        entries = [(lease.internal_id, lease.seq_len, lease.deadline) for lease in instance.leases.values()]
        entries.append((request.internal_id, request.seq_len, request.deadline))
        service = sum(
            self.curve.predict([entry[1] for entry in batch], instance.tp_size)
            for batch in self._pack(entries)
        )
        overlap = self._slowdown(instance, self._active(instance))
        start = max(now, instance.virtual_finish)
        return start + service * overlap + self.batch_window_s + self.prediction_margin_s

    def _candidate_score(self, now: float, request: V7Request, instance: V7Instance) -> Tuple[float, ...]:
        finish = self._projected_finish(now, instance, request)
        slack = request.deadline - finish
        urgency = max(0.0, min(4.0, (self.slo_ttft - slack) / max(self.slo_ttft, 1e-6)))
        price = self._conflict_price(instance)
        pressure = self._backlog_pressure(self._class(request.seq_len))
        debt_bonus = self.class_debt[self._class(request.seq_len)] / max(1, self.class_quantum_tokens)
        # Smaller is better. Debt bonus is negative for a class that has waited.
        score = (
            finish
            + self.urgency_weight * urgency
            + self.conflict_price_weight * price
            + 0.25 * pressure
            - 0.05 * debt_bonus
        )
        return (score, request.deadline, request.enqueue_order, instance.admitted_tokens, instance.node_key)

    def _request_order(self, now: float) -> List[V7Request]:
        self._advance_debt(now)
        pending = list(self._pending.values())
        if not pending:
            return []
        # EDF is retained inside the flow rule, but class debt can pull a
        # slightly newer request forward when it has waited for a full quantum.
        return sorted(
            pending,
            key=lambda request: (
                request.deadline - 0.05 * self.class_debt[self._class(request.seq_len)] / max(1, self.class_quantum_tokens),
                request.enqueue_order,
            ),
        )

    def _accepts(self, instance: V7Instance, request: V7Request) -> bool:
        if not instance.available:
            return False
        if len(instance.leases) >= self.max_inflight_per_instance:
            return False
        if instance.accepted_bundle_count >= self.max_accepted_bundles_per_instance:
            return False
        if instance.leases and instance.admitted_tokens + request.seq_len > self.token_credit:
            self.credit_blocked += 1
            return False
        return True

    def _pick_decode(self) -> PD_Client_Obj:
        if not self.decode_nodes:
            raise RuntimeError("no Decode node is registered")
        node = self.decode_nodes[self._decode_rr_index % len(self.decode_nodes)]
        self._decode_rr_index += 1
        return node

    def _admit(self, now: float, request: V7Request, instance: V7Instance) -> None:
        self._pending.pop(request.internal_id, None)
        finish = self._projected_finish(now, instance, request)
        lease = V7Lease(
            internal_id=request.internal_id,
            external_req_id=request.external_req_id,
            seq_len=request.seq_len,
            deadline=request.deadline,
            request_class=self._class(request.seq_len),
            node_key=instance.node_key,
            tp_size=instance.tp_size,
            predicted_finish=finish,
            dispatch_time=now,
            instance_generation=instance.instance_generation,
        )
        instance.leases[lease.internal_id] = lease
        instance.virtual_finish = max(now, instance.virtual_finish) + self.curve.predict([request.seq_len], instance.tp_size)
        self._leases[lease.internal_id] = lease
        if lease.external_req_id is not None:
            self._external_to_internal[lease.external_req_id] = lease.internal_id
        request_class = lease.request_class
        self.class_debt[request_class] -= request.seq_len
        self.admitted_by_class[request_class] += 1
        self.admitted_by_instance[instance.node_key] += 1
        if finish > request.deadline:
            self.deadline_overrides += 1
        if not request.future.done():
            request.future.set_result((instance.node, self._pick_decode()))

    def _schedule_locked(self, now: float) -> None:
        if not self.decode_nodes or not any(instance.available for instance in self.instances.values()):
            self._fail_unserviceable_pending()
            return
        self.flow_decisions += 1
        while self._pending:
            selected: Optional[Tuple[Tuple[float, ...], V7Request, V7Instance]] = None
            for request in self._request_order(now):
                request_class = self._class(request.seq_len)
                for instance in self._targets(request_class):
                    if not self._accepts(instance, request):
                        continue
                    candidate = (self._candidate_score(now, request, instance), request, instance)
                    if selected is None or candidate[0] < selected[0]:
                        selected = candidate
            if selected is None:
                expired = [request for request in self._pending.values() if request.deadline <= now]
                if expired and self.overload_policy == "reject":
                    request = min(expired, key=lambda item: (item.deadline, item.enqueue_order))
                    self._pending.pop(request.internal_id, None)
                    if not request.future.done():
                        request.future.set_exception(RuntimeError("FlexTP V7 rejected request after deadline"))
                    continue
                return
            _, request, instance = selected
            self._admit(now, request, instance)

    def _fail_unserviceable_pending(self) -> None:
        if self.decode_nodes and any(instance.available for instance in self.instances.values()):
            return
        reason = "no Decode node is registered" if not self.decode_nodes else "no available Prefill instance is registered"
        for internal_id, request in list(self._pending.items()):
            self._pending.pop(internal_id, None)
            if not request.future.done():
                request.future.set_exception(RuntimeError(f"FlexTP V7: {reason}"))

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
            raise RuntimeError("FlexTP V7 cannot be used across event loops")
        now = time.time()
        self._request_counter += 1
        self._enqueue_counter += 1
        internal_id = self._request_counter
        request = V7Request(
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
                raise ValueError(f"FlexTP V7 req_id={req_id} is already admitted or pending")
            self._pending[internal_id] = request
            self._fail_unserviceable_pending()
            self._schedule_locked(now)
            if not request.future.done():
                self._request_reschedule()
        try:
            return await request.future
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

    def _find_lease(self, p_node: Optional[PD_Client_Obj], req_id: Optional[int]) -> Optional[V7Lease]:
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

    async def _finish(
        self,
        p_node: Optional[PD_Client_Obj],
        *,
        req_id: Optional[int],
        failed: bool,
        reason: str = "",
    ) -> None:
        async with self._lock:
            lease = self._find_lease(p_node, req_id)
            if lease is None:
                logger.warning("FlexTP V7 completion for unknown req=%s node=%s", req_id, getattr(p_node, "client_ip_port", None))
                return
            lease.state = "failed" if failed else "finished"
            self._leases.pop(lease.internal_id, None)
            if lease.external_req_id is not None:
                self._external_to_internal.pop(lease.external_req_id, None)
            instance = self.instances.get(lease.node_key)
            if instance is not None:
                instance.leases.pop(lease.internal_id, None)
                if not instance.leases:
                    instance.virtual_finish = 0.0
            if failed:
                logger.warning("FlexTP V7 failed req=%s node=%s reason=%s", lease.external_req_id, lease.node_key, reason)
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
                instance.virtual_finish = 0.0
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
                "virtual_finish": instance.virtual_finish,
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
            "version": 7,
            "policy": "online_dual_price_flow",
            "long_request_threshold": self.long_request_threshold,
            "token_credit": self.token_credit,
            "class_debt": dict(self.class_debt),
            "class_quantum_tokens": self.class_quantum_tokens,
            "conflict_price_weight": self.conflict_price_weight,
            "urgency_weight": self.urgency_weight,
            "mps_overlap_slowdown": self.mps_overlap_slowdown,
            "flow_decisions": self.flow_decisions,
            "price_evaluations": self.price_evaluations,
            "credit_blocked": self.credit_blocked,
            "deadline_overrides": self.deadline_overrides,
            "admitted_by_class": dict(sorted(self.admitted_by_class.items())),
            "admitted_by_instance": dict(sorted(self.admitted_by_instance.items())),
        }
