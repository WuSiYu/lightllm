"""Deadline-aware, bounded multi-request FlexTP selector.

This selector is the normal-PD implementation of the two-level scheduler.  It
owns admission and placement, while the Prefill instance's local Router still
owns the final KV-safe batch packing.  A lease is deliberately bounded by both
request count and first-chunk tokens; this restores worker batching without
turning the master into an unbounded hidden queue.

NIXL is intentionally not coupled to this module.  Its prompt-id-ready event
has different semantics from normal-PD Prefill completion and must use the old
path until a real Prefill-finished event is available.
"""

from __future__ import annotations

import asyncio
import collections
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Literal, Optional, Sequence, Tuple

from lightllm.server.core.objs import SamplingParams
from lightllm.server.multimodal_params import MultimodalParams
from lightllm.server.pd_io_struct import PD_Client_Obj
from lightllm.utils.log_utils import init_logger

from .pd_selector import PDSelector


logger = init_logger(__name__)
_ALL_GENERATIONS = object()


@dataclass
class V3Lease:
    internal_id: int
    external_req_id: Optional[int]
    seq_len: int
    deadline: float
    node_key: str
    tp_size: int
    predicted_finish: float
    dispatch_time: float
    instance_generation: Optional[str] = None
    accepted: bool = False
    state: str = "admitted"


@dataclass
class V3InstanceState:
    node: PD_Client_Obj
    node_key: str
    group_id: str
    tp_size: int
    gpu_set: FrozenSet[str]
    placement_is_explicit: bool
    instance_generation: Optional[str] = None
    available: bool = True
    leases: "collections.OrderedDict[int, V3Lease]" = field(default_factory=collections.OrderedDict)
    worker_queued_requests: int = 0
    worker_queued_tokens: int = 0
    worker_running_requests: int = 0
    worker_running_tokens: int = 0
    worker_queued_group_ids: Tuple[int, ...] = ()
    worker_running_group_ids: Tuple[int, ...] = ()
    worker_load: float = 0.0
    last_report_time: float = 0.0
    last_report_seq: int = -1

    @property
    def busy(self) -> bool:
        return bool(self.leases)

    @property
    def admitted_tokens(self) -> int:
        return sum(lease.seq_len for lease in self.leases.values())


@dataclass
class V3Candidate:
    pending: "V3Pending"
    instance: V3InstanceState
    predicted_finish: float
    exclusive_work: float
    candidate_meets_deadline: bool
    protects_existing: bool
    incremental_delay: float

    @property
    def priority(self) -> Tuple[float, int]:
        return (self.pending.deadline, self.pending.enqueue_order)


@dataclass
class V3Pending:
    internal_id: int
    external_req_id: Optional[int]
    seq_len: int
    deadline: float
    future: asyncio.Future
    enqueue_order: int


class LatencyModelV3:
    """Measured single-request curves plus conservative batch extrapolation."""

    # Stored as (a, b, c, d) for max(a*L/tp + b*L^2/tp, c) + d.
    DEFAULT_CONSTANTS: Dict[int, Tuple[float, float, float, float]] = {
        1: (2.4e-4, 1e-8, 0.05, 0.05),
        2: (2.018352e-4, 6.048427e-9, 2.341091e-2, 6.794939e-2),
        4: (2.655797e-4, 2.330274e-9, 5.651746e-2, 3.767872e-2),
    }

    def __init__(self, constants: Optional[Dict[int, Tuple[float, float, float, float]]] = None):
        self.constants = dict(self.DEFAULT_CONSTANTS)
        if constants:
            self.constants.update(constants)

    def _get(self, tp_size: int) -> Tuple[float, float, float, float]:
        if tp_size in self.constants:
            return self.constants[tp_size]
        nearest = min(self.constants, key=lambda value: abs(value - tp_size))
        return self.constants[nearest]

    def predict_batch(self, seq_lens: Sequence[int], tp_size: int) -> float:
        if not seq_lens:
            return 0.0
        a, b, c, d = self._get(tp_size)
        total = sum(max(1, int(length)) for length in seq_lens)
        squared = sum(max(1, int(length)) ** 2 for length in seq_lens)
        return max(a * total / tp_size + b * squared / tp_size, c) + d

    def predict_exclusive(self, seq_len: int, tp_size: int) -> float:
        return self.predict_batch([seq_len], tp_size)


class FlexTPSelectorV3(PDSelector):
    """Normal-PD FlexTP admission with bounded multi-request leases.

    The selector does not pretend to know the exact local Router batch.  It
    packs admitted lengths into the same 8192-token envelope used by the
    bundle dispatcher and uses a safety margin around an optimistic p99 model.
    The worker report is used for observability and future calibration; lease
    release still requires an authoritative normal-PD first-token completion
    or failure callback.

    A PD request split into multiple ``max_new_tokens`` blocks owns one bounded
    lease per Prefill block. The master re-admits continuation blocks after
    the previous block's KV transfer, so growing prompt/history work remains
    visible without reserving an instance throughout Decode.
    """

    supports_bundles = True
    is_flex_tp_v3 = True

    def __init__(
        self,
        pd_manager,
        slo_ttft: Optional[float] = 5.0,
        *,
        batch_token_cap: int = 8192,
        batch_token_trigger: int = 4096,
        max_inflight_per_instance: int = 64,
        max_admitted_tokens_per_instance: int = 16384,
        batch_window_s: float = 0.020,
        prediction_margin_s: float = 0.080,
        replan_interval_s: float = 0.050,
        latency_constants: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        mode_slowdowns: Optional[Dict[Tuple[int, Tuple[int, ...]], float]] = None,
        overload_policy: Literal["best_effort", "reject"] = "best_effort",
    ) -> None:
        super().__init__(pd_manager)
        self.slo_ttft = 5.0 if slo_ttft is None else float(slo_ttft)
        if not math.isfinite(self.slo_ttft) or self.slo_ttft <= 0:
            raise ValueError("slo_ttft must be a positive finite number")
        self.batch_token_cap = max(1, int(batch_token_cap))
        self.batch_token_trigger = max(1, min(int(batch_token_trigger), self.batch_token_cap))
        self.max_inflight_per_instance = max(1, int(max_inflight_per_instance))
        self.max_admitted_tokens_per_instance = max(1, int(max_admitted_tokens_per_instance))
        self.batch_window_s = max(0.0, float(batch_window_s))
        self.prediction_margin_s = max(0.0, float(prediction_margin_s))
        self.replan_interval_s = max(0.005, float(replan_interval_s))
        if overload_policy not in ("best_effort", "reject"):
            raise ValueError("overload_policy must be 'best_effort' or 'reject'")
        self.overload_policy = overload_policy
        self.latency_model = LatencyModelV3(latency_constants)
        self.mode_slowdowns = {
            (int(tp), tuple(sorted(signature))): max(1.0, float(value))
            for (tp, signature), value in (mode_slowdowns or {}).items()
        }

        self.instances: Dict[str, V3InstanceState] = {}
        self.flex_groups: Dict[str, Dict[str, V3InstanceState]] = {}
        self.node_to_group: Dict[str, Dict[str, V3InstanceState]] = {}
        self.node_tp_size: Dict[str, int] = {}
        self.ungrouped_prefill_nodes: List[PD_Client_Obj] = []

        self._pending: "collections.OrderedDict[int, V3Pending]" = collections.OrderedDict()
        self._leases: Dict[int, V3Lease] = {}
        self._external_to_internal: Dict[int, int] = {}
        self._request_counter = 0
        self._enqueue_counter = 0
        self._decode_rr_index = 0
        self._state_lock = asyncio.Lock()
        self._owner_loop: Optional[asyncio.AbstractEventLoop] = None
        self._replan_task: Optional[asyncio.Task] = None

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

    def _group_id(self, node: PD_Client_Obj) -> Optional[str]:
        args = self._args(node)
        host = self._host(node)
        explicit = args.get("tp_smt_group_id") or args.get("flex_tp_group_id")
        if explicit is not None:
            return f"{host}:{explicit}"
        port = args.get("shared_weight_master_port_start")
        if args.get("shared_weight") and port is not None:
            return f"{host}:{port}"
        # Ordinary PD deployments may not carry TP-SMT metadata at all.  Keep
        # them schedulable with a host-scoped synthetic group; the matching
        # unknown placement key below intentionally treats same-host nodes as
        # overlapping instead of guessing that their GPUs are disjoint.
        return f"{host}:ungrouped"

    def _gpu_set(self, node: PD_Client_Obj, group_id: str) -> Tuple[FrozenSet[str], bool]:
        values = self._parse_gpu_ids(self._args(node).get("tp_smt_gpu_ids"))
        host = self._host(node)
        if values:
            return frozenset(f"{host}/gpu:{value}" for value in values), True
        # Without explicit placement, every instance on the same host may
        # overlap. Use a host-level sentinel rather than a group-level one so
        # separate groups cannot accidentally be treated as independent GPUs.
        return frozenset({f"unknown-placement:{host}"}), False

    @staticmethod
    def _overlap(left: V3InstanceState, right: V3InstanceState) -> bool:
        return bool(left.gpu_set & right.gpu_set)

    def update_nodes(self, prefill_nodes, decode_nodes) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._owner_loop is not None and loop is not self._owner_loop:
            self._owner_loop.call_soon_threadsafe(self.update_nodes, prefill_nodes, decode_nodes)
            return
        # Scheme C is deliberately scoped to ordinary PD.  Keeping NIXL
        # nodes out of both lists prevents a mixed deployment from routing a
        # normal REQ_BUNDLE into the NIXL prompt-id handshake.
        normal_prefill_nodes = [node for node in prefill_nodes if node.mode == "prefill"]
        normal_decode_nodes = [node for node in decode_nodes if node.mode == "decode"]
        super().update_nodes(normal_prefill_nodes, normal_decode_nodes)
        discovered = []
        group_tp_sizes: Dict[str, set] = collections.defaultdict(set)
        for node in normal_prefill_nodes:
            group_id = self._group_id(node)
            if group_id is None:
                continue
            args = self._args(node)
            tp_size = int(args.get("tp", 1))
            gpu_set, explicit = self._gpu_set(node, group_id)
            discovered.append((node, group_id, tp_size, gpu_set, explicit))
            group_tp_sizes[group_id].add(tp_size)

        # A group with only one observed TP size is still schedulable (for
        # example while the TP4 sibling is starting).  New sizes become
        # eligible automatically when their registration arrives; keeping the
        # single-size group avoids an otherwise indefinite startup wait.
        eligible_groups = set(group_tp_sizes)
        for instance in self.instances.values():
            instance.available = False

        for node, group_id, tp_size, gpu_set, explicit in discovered:
            if group_id not in eligible_groups:
                continue
            node_key = str(node.client_ip_port)
            instance = self.instances.get(node_key)
            if instance is None:
                instance = V3InstanceState(
                    node=node,
                    node_key=node_key,
                    group_id=group_id,
                    tp_size=tp_size,
                    gpu_set=gpu_set,
                    placement_is_explicit=explicit,
                    instance_generation=getattr(node, "instance_generation", None),
                )
                self.instances[node_key] = instance
            elif instance.busy and (
                instance.group_id != group_id
                or instance.tp_size != tp_size
                or instance.gpu_set != gpu_set
            ):
                logger.warning("keep busy V3 instance %s on its old topology", node_key)
                instance.available = False
                continue
            else:
                if instance.instance_generation != getattr(node, "instance_generation", None):
                    instance.last_report_seq = -1
                instance.node = node
                instance.group_id = group_id
                instance.tp_size = tp_size
                instance.gpu_set = gpu_set
                instance.placement_is_explicit = explicit
                instance.instance_generation = getattr(node, "instance_generation", None)
            instance.available = True

        for node_key, instance in list(self.instances.items()):
            if not instance.available and not instance.busy:
                del self.instances[node_key]

        self.flex_groups = collections.defaultdict(dict)
        self.node_to_group = {}
        self.node_tp_size = {}
        for instance in self.instances.values():
            self.flex_groups[instance.group_id][instance.node_key] = instance
            self.node_to_group[instance.node_key] = self.flex_groups[instance.group_id]
            self.node_tp_size[instance.node_key] = instance.tp_size
            if not instance.placement_is_explicit:
                logger.warning("V3 group %s node %s has no explicit GPU placement; assume overlap", instance.group_id, instance.node_key)
        self.flex_groups = dict(self.flex_groups)
        self._fail_unserviceable_pending()
        self._request_reschedule()

    def _fail_unserviceable_pending(self) -> None:
        """Fail pending requests when no execution topology exists at all."""
        if self.decode_nodes and any(instance.available for instance in self.instances.values()):
            return
        reason = "no Decode node is registered" if not self.decode_nodes else "no available Prefill instance is registered"
        for internal_id, pending in list(self._pending.items()):
            self._pending.pop(internal_id, None)
            if not pending.future.done():
                pending.future.set_exception(RuntimeError(f"FlexTP V3: {reason}"))

    def _request_reschedule(self) -> None:
        if self._owner_loop is None:
            return
        if self._replan_task is None or self._replan_task.done():
            self._replan_task = self._owner_loop.create_task(self._replan_loop())

    async def _replan_loop(self) -> None:
        while True:
            await asyncio.sleep(self.replan_interval_s)
            async with self._state_lock:
                self._schedule_locked(time.time())
                if not self._pending:
                    return

    def _pick_decode_node(self) -> PD_Client_Obj:
        if not self.decode_nodes:
            raise RuntimeError("no Decode node is registered")
        index = self._decode_rr_index % len(self.decode_nodes)
        self._decode_rr_index += 1
        return self.decode_nodes[index]

    def _active_instances(self, include: Optional[V3InstanceState] = None) -> List[V3InstanceState]:
        values = [instance for instance in self.instances.values() if instance.busy]
        if include is not None and include not in values:
            values.append(include)
        return values

    def _mode_signature(self, target: V3InstanceState, active: Sequence[V3InstanceState]) -> Tuple[int, ...]:
        signature = [target.tp_size]
        signature.extend(
            instance.tp_size
            for instance in active
            if instance is not target and self._overlap(target, instance)
        )
        return tuple(sorted(signature))

    def _slowdown(self, target: V3InstanceState, active: Sequence[V3InstanceState]) -> float:
        signature = self._mode_signature(target, active)
        override = self.mode_slowdowns.get((target.tp_size, signature))
        if override is not None:
            return override
        max_processes = 1
        for gpu in target.gpu_set:
            max_processes = max(max_processes, sum(gpu in instance.gpu_set for instance in active))
        return float(max_processes)

    def _packed_batches(self, instance: V3InstanceState, extra: Optional[V3Pending] = None):
        entries: List[Tuple[int, int]] = [
            (internal_id, lease.seq_len) for internal_id, lease in instance.leases.items()
        ]
        if extra is not None:
            entries.append((extra.internal_id, extra.seq_len))
        batches: List[List[Tuple[int, int]]] = []
        current: List[Tuple[int, int]] = []
        current_tokens = 0
        for internal_id, seq_len in entries:
            seq_len = max(1, int(seq_len))
            if current and current_tokens + seq_len > self.batch_token_cap:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append((internal_id, seq_len))
            current_tokens += seq_len
        if current:
            batches.append(current)
        return batches

    def _finish_map(
        self,
        active: Sequence[V3InstanceState],
        candidate: V3InstanceState,
        extra: V3Pending,
    ) -> Tuple[Dict[int, float], float, Dict[str, float]]:
        finish_times: Dict[int, float] = {}
        instance_finish: Dict[str, float] = {}
        candidate_active = list(active)
        if candidate not in candidate_active:
            candidate_active.append(candidate)
        for instance in candidate_active:
            extra_for_instance = extra if instance is candidate else None
            batches = self._packed_batches(instance, extra_for_instance)
            factor = self._slowdown(instance, candidate_active)
            elapsed = 0.0
            for batch in batches:
                elapsed += self.latency_model.predict_batch([length for _, length in batch], instance.tp_size) * factor
                for internal_id, _ in batch:
                    finish_times[internal_id] = elapsed
            instance_finish[instance.node_key] = elapsed
        candidate_finish = finish_times[extra.internal_id]
        return finish_times, candidate_finish, instance_finish

    def _candidate(self, now: float, pending: V3Pending, instance: V3InstanceState) -> Optional[V3Candidate]:
        if not instance.available:
            return None
        if len(instance.leases) >= self.max_inflight_per_instance:
            return None
        if instance.admitted_tokens + pending.seq_len > self.max_admitted_tokens_per_instance and instance.leases:
            return None

        active = self._active_instances()
        # Compute the current schedule separately from the candidate schedule;
        # this keeps admission side-effect free while preserving existing
        # leases' finish-time protection.
        baseline_finish = {}
        for active_instance in active:
            batches = self._packed_batches(active_instance)
            factor = self._slowdown(active_instance, active)
            elapsed = 0.0
            for batch in batches:
                elapsed += self.latency_model.predict_batch([length for _, length in batch], active_instance.tp_size) * factor
                for internal_id, _ in batch:
                    baseline_finish[internal_id] = elapsed

        finish_times, candidate_relative, instance_finish = self._finish_map(active, instance, pending)
        candidate_finish = now + candidate_relative + self.batch_window_s + self.prediction_margin_s
        protects_existing = True
        incremental = 0.0
        for lease in self._leases.values():
            if lease.node_key not in self.instances:
                continue
            before = now + baseline_finish.get(lease.internal_id, 0.0)
            after = now + finish_times.get(lease.internal_id, baseline_finish.get(lease.internal_id, 0.0))
            incremental += max(0.0, after - before)
            allowed = max(lease.deadline - self.prediction_margin_s, before)
            if after > allowed:
                protects_existing = False

        candidate_meets = candidate_finish <= pending.deadline
        exclusive = self.latency_model.predict_exclusive(pending.seq_len, instance.tp_size)
        return V3Candidate(
            pending=pending,
            instance=instance,
            predicted_finish=candidate_finish,
            exclusive_work=exclusive,
            candidate_meets_deadline=candidate_meets,
            protects_existing=protects_existing,
            incremental_delay=incremental,
        )

    @staticmethod
    def _pending_priority(pending: V3Pending) -> Tuple[float, int]:
        return (pending.deadline, pending.enqueue_order)

    def _can_meet_alone(self, now: float, pending: V3Pending) -> bool:
        """Whether the request can meet its deadline on an empty instance.

        This is deliberately independent of current leases.  It distinguishes a
        request that is genuinely overloaded from one that merely needs a future
        TP4 window, which lets the scheduler protect the latter from refill.
        """
        for instance in self.instances.values():
            if not instance.available:
                continue
            finish = (
                now
                + self.latency_model.predict_exclusive(pending.seq_len, instance.tp_size)
                + self.batch_window_s
                + self.prediction_margin_s
            )
            if finish <= pending.deadline:
                return True
        return False

    def _baseline_finish(self, now: float) -> Dict[int, float]:
        """Predict current lease completion times without admitting a waiter."""
        active = self._active_instances()
        result: Dict[int, float] = {}
        for instance in active:
            batches = self._packed_batches(instance)
            factor = self._slowdown(instance, active)
            elapsed = 0.0
            for batch in batches:
                elapsed += self.latency_model.predict_batch(
                    [length for _, length in batch], instance.tp_size
                ) * factor
                for internal_id, _ in batch:
                    result[internal_id] = elapsed
        return result

    def _future_reservations(
        self,
        now: float,
        feasible: Sequence[V3Candidate],
        baseline_finish: Dict[int, float],
    ) -> List[Tuple[Tuple[float, int], V3InstanceState]]:
        """Reserve the earliest safe placement for blocked-but-serviceable waiters.

        A pending request that cannot be admitted in the current mode may still be
        serviceable after a busy overlapping instance drains.  Without this soft
        reservation, continuously arriving short TP2 requests can refill the same
        GPUs and starve an older TP4 request indefinitely.
        """
        feasible_ids = {item.pending.internal_id for item in feasible}
        reservations: List[Tuple[Tuple[float, int], V3InstanceState]] = []
        for pending in sorted(self._pending.values(), key=self._pending_priority):
            if pending.internal_id in feasible_ids or not self._can_meet_alone(now, pending):
                continue
            # A placement can be full under the current admission credit and
            # still be the correct future window after its leases drain.  Do
            # not restrict reservations to candidates that are immediately
            # admissible, or a long TP4 waiter can be starved by refill.
            placements = [
                instance
                for instance in self.instances.values()
                if instance.available
                and now
                + self.latency_model.predict_exclusive(pending.seq_len, instance.tp_size)
                + self.batch_window_s
                + self.prediction_margin_s
                <= pending.deadline
            ]
            if not placements:
                continue

            def ready_finish(instance: V3InstanceState) -> float:
                current_finish = 0.0
                for lease in instance.leases.values():
                    current_finish = max(current_finish, baseline_finish.get(lease.internal_id, 0.0))
                return now + current_finish + self.latency_model.predict_exclusive(
                    pending.seq_len, instance.tp_size
                ) + self.batch_window_s + self.prediction_margin_s

            placement = min(
                placements,
                key=lambda instance: (
                    ready_finish(instance),
                    instance.tp_size,
                    instance.node_key,
                ),
            )
            reservations.append((self._pending_priority(pending), placement))
        return reservations

    def _reservation_allows(
        self,
        candidate: V3Candidate,
        reservations: Sequence[Tuple[Tuple[float, int], V3InstanceState]],
    ) -> bool:
        priority = candidate.priority
        return not any(
            reserved_priority < priority and self._overlap(candidate.instance, reserved_instance)
            for reserved_priority, reserved_instance in reservations
        )

    def _schedule_locked(self, now: float) -> None:
        if not self.decode_nodes:
            expired = [pending for pending in self._pending.values() if pending.deadline <= now]
            for pending in expired:
                self._pending.pop(pending.internal_id, None)
                if not pending.future.done():
                    pending.future.set_exception(RuntimeError("FlexTP V3 has no Decode node before deadline"))
            return
        while self._pending:
            candidates: List[V3Candidate] = []
            for pending in sorted(self._pending.values(), key=lambda item: (item.deadline, item.enqueue_order)):
                for instance in self.instances.values():
                    candidate = self._candidate(now, pending, instance)
                    if candidate is not None:
                        candidates.append(candidate)
            if not candidates:
                expired = [pending for pending in self._pending.values() if pending.deadline <= now]
                if expired:
                    pending = min(expired, key=lambda item: (item.deadline, item.enqueue_order))
                    self._pending.pop(pending.internal_id, None)
                    if not pending.future.done():
                        pending.future.set_exception(RuntimeError(
                            "FlexTP V3 rejected request: no admissible Prefill instance before deadline"
                        ))
                    continue
                return
            feasible = [
                candidate
                for candidate in candidates
                if candidate.candidate_meets_deadline and candidate.protects_existing
            ]
            reservations = self._future_reservations(
                now,
                feasible,
                self._baseline_finish(now),
            )
            feasible = [candidate for candidate in feasible if self._reservation_allows(candidate, reservations)]
            if not feasible:
                expired = [candidate for candidate in candidates if candidate.pending.deadline <= now]
                if not expired:
                    if self.overload_policy == "reject":
                        impossible = [
                            pending
                            for pending in self._pending.values()
                            if not self._can_meet_alone(now, pending)
                        ]
                        if impossible:
                            pending = min(impossible, key=self._pending_priority)
                            self._pending.pop(pending.internal_id, None)
                            if not pending.future.done():
                                pending.future.set_exception(RuntimeError(
                                    "FlexTP V3 rejected request: no placement can meet its deadline"
                                ))
                            continue
                    return
                if self.overload_policy == "reject":
                    pending = min(
                        (candidate.pending for candidate in expired),
                        key=self._pending_priority,
                    )
                    self._pending.pop(pending.internal_id, None)
                    if not pending.future.done():
                        pending.future.set_exception(RuntimeError(
                            "FlexTP V3 rejected request: deadline already expired"
                        ))
                    continue
                # A request must not hang forever when the configured SLO is
                # physically unattainable.  Admit the least harmful overdue
                # candidate and expose the miss through the prediction log.
                logger.warning(
                    "FlexTP V3 admitting overdue request req=%s candidates=%d",
                    expired[0].pending.external_req_id,
                    len(expired),
                )
                candidates_to_rank = expired
            else:
                candidates_to_rank = feasible
            selected = min(
                candidates_to_rank,
                key=lambda candidate: (
                    candidate.pending.deadline,
                    candidate.pending.enqueue_order,
                    candidate.instance.tp_size,
                    candidate.exclusive_work * candidate.instance.tp_size,
                    candidate.predicted_finish,
                    candidate.incremental_delay,
                    len(candidate.instance.leases),
                    candidate.instance.node_key,
                ),
            )
            pending = selected.pending
            instance = selected.instance
            self._pending.pop(pending.internal_id, None)
            lease = V3Lease(
                internal_id=pending.internal_id,
                external_req_id=pending.external_req_id,
                seq_len=pending.seq_len,
                deadline=pending.deadline,
                node_key=instance.node_key,
                tp_size=instance.tp_size,
                predicted_finish=selected.predicted_finish,
                dispatch_time=now,
                instance_generation=instance.instance_generation,
            )
            instance.leases[lease.internal_id] = lease
            self._leases[lease.internal_id] = lease
            if lease.external_req_id is not None:
                self._external_to_internal[lease.external_req_id] = lease.internal_id
            if not pending.future.done():
                pending.future.set_result((instance.node, self._pick_decode_node()))
            logger.info(
                "FlexTP V3 admit req=%s len=%d tp=%d node=%s leases=%d predicted=%.1fms deadline=%.1fms",
                lease.external_req_id,
                lease.seq_len,
                lease.tp_size,
                lease.node_key,
                len(instance.leases),
                max(0.0, lease.predicted_finish - now) * 1000,
                max(0.0, lease.deadline - now) * 1000,
            )

    async def async_select_p_d_node(
        self,
        prompt,
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        input_token_num: Optional[int] = None,
        arrival_time: Optional[float] = None,
        req_id: Optional[int] = None,
    ) -> Tuple[PD_Client_Obj, PD_Client_Obj]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError("FlexTP V3 requires an asyncio event loop") from exc
        if self._owner_loop is None:
            self._owner_loop = loop
        elif self._owner_loop is not loop:
            raise RuntimeError("FlexTP V3 cannot be used across event loops")

        now = time.time()
        self._request_counter += 1
        self._enqueue_counter += 1
        internal_id = self._request_counter
        future = loop.create_future()
        pending = V3Pending(
            internal_id=internal_id,
            external_req_id=req_id,
            seq_len=max(1, int(input_token_num or 1)),
            deadline=(arrival_time if arrival_time is not None else now) + self.slo_ttft,
            future=future,
            enqueue_order=self._enqueue_counter,
        )
        async with self._state_lock:
            if req_id is not None and (
                req_id in self._external_to_internal
                or any(item.external_req_id == req_id for item in self._pending.values())
            ):
                raise ValueError(f"FlexTP V3 req_id={req_id} is already admitted or pending")
            self._pending[internal_id] = pending
            self._fail_unserviceable_pending()
            self._schedule_locked(now)
            if not future.done():
                self._request_reschedule()
        try:
            return await future
        except asyncio.CancelledError:
            async with self._state_lock:
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

    def _find_lease(self, p_node: Optional[PD_Client_Obj], req_id: Optional[int]) -> Optional[V3Lease]:
        # Completion and failure messages normally carry the originating lease
        # id.  For compatibility with older selectors, an id-less callback may
        # resolve only when this node has exactly one lease; never guess among
        # multiple requests, since that could release a neighboring request.
        if req_id is None:
            if p_node is None:
                return None
            instance = self.instances.get(p_node.client_ip_port)
            if instance is None or len(instance.leases) != 1:
                return None
            lease = next(iter(instance.leases.values()))
        else:
            internal_id = self._external_to_internal.get(req_id)
            lease = self._leases.get(internal_id) if internal_id is not None else None
        if lease is None or p_node is None:
            return None
        if lease.node_key != p_node.client_ip_port:
            return None
        generation = getattr(p_node, "instance_generation", None)
        if generation is not None and generation != lease.instance_generation:
            return None
        return lease

    async def _finish_request(
        self,
        p_node: Optional[PD_Client_Obj],
        *,
        input_token_num: int = 0,
        actual_ttft: Optional[float] = None,
        req_id: Optional[int] = None,
        failed: bool = False,
        reason: str = "",
    ) -> None:
        async with self._state_lock:
            lease = self._find_lease(p_node, req_id)
            if lease is None:
                logger.warning("FlexTP V3 completion for unknown req=%s node=%s", req_id, getattr(p_node, "client_ip_port", None))
                return
            lease.state = "failed" if failed else "finished"
            self._leases.pop(lease.internal_id, None)
            self._external_to_internal.pop(lease.external_req_id, None)
            instance = self.instances.get(lease.node_key)
            if instance is not None:
                instance.leases.pop(lease.internal_id, None)
            if failed:
                logger.warning("FlexTP V3 failed req=%s node=%s reason=%s", lease.external_req_id, lease.node_key, reason)
            else:
                logger.info(
                    "FlexTP V3 finished req=%s node=%s actual_ttft=%s",
                    lease.external_req_id,
                    lease.node_key,
                    actual_ttft,
                )
            self._schedule_locked(time.time())
            self._request_reschedule()

    async def notify_request_done(self, p_node, input_token_num: int = 0, actual_ttft: Optional[float] = None, req_id: Optional[int] = None):
        await self._finish_request(p_node, input_token_num=input_token_num, actual_ttft=actual_ttft, req_id=req_id)

    async def notify_request_failed(self, p_node, input_token_num: int = 0, req_id: Optional[int] = None, reason: str = ""):
        await self._finish_request(p_node, input_token_num=input_token_num, req_id=req_id, failed=True, reason=reason)

    async def notify_bundle_accepted(self, p_node, bundle_id: int, req_ids: Sequence[int]) -> None:
        async with self._state_lock:
            for req_id in req_ids:
                lease = self._find_lease(p_node, req_id)
                if lease is not None:
                    lease.accepted = True
                    lease.state = "queued_worker"
            logger.debug("FlexTP V3 bundle accepted node=%s bundle=%s reqs=%s", getattr(p_node, "client_ip_port", None), bundle_id, list(req_ids))

    async def update_instance_report(
        self, node_key: str, report: Dict, report_seq: Optional[int] = None
    ) -> None:
        def _report_ids(value) -> Tuple[int, ...]:
            if not isinstance(value, (list, tuple, set, frozenset)):
                return ()
            ids = []
            for item in value:
                try:
                    ids.append(int(item))
                except (TypeError, ValueError):
                    continue
            return tuple(sorted(set(ids)))

        async with self._state_lock:
            instance = self.instances.get(node_key)
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
            instance.worker_queued_group_ids = _report_ids(report.get("queued_group_ids"))
            instance.worker_running_group_ids = _report_ids(report.get("running_group_ids"))
            instance.worker_load = float(report.get("total_token_usage_rate", instance.worker_load))
            instance.last_report_time = time.time()

    def notify_node_removed(self, node_key: str, instance_generation=_ALL_GENERATIONS) -> None:
        """Release leases when a worker websocket disappears.

        The normal completion callback cannot arrive after a broken websocket;
        retaining those leases would permanently hide the instance capacity.
        """
        loop = self._owner_loop
        if loop is None or loop.is_closed():
            return
        loop.create_task(self._release_node_leases(node_key, instance_generation))

    async def _release_node_leases(self, node_key: str, instance_generation=_ALL_GENERATIONS) -> None:
        async with self._state_lock:
            instance = self.instances.get(node_key)
            if instance is None:
                return
            released = [
                lease
                for lease in instance.leases.values()
                if instance_generation is _ALL_GENERATIONS or lease.instance_generation == instance_generation
            ]
            for lease in released:
                instance.leases.pop(lease.internal_id, None)
                self._leases.pop(lease.internal_id, None)
                if lease.external_req_id is not None:
                    self._external_to_internal.pop(lease.external_req_id, None)
            if released:
                logger.warning("FlexTP V3 released %d leases after node removal: %s", len(released), node_key)
            self._schedule_locked(time.time())

    def snapshot(self) -> Dict[str, Dict]:
        return {
            node_key: {
                "group_id": instance.group_id,
                "instance_generation": instance.instance_generation,
                "tp": instance.tp_size,
                "gpus": sorted(instance.gpu_set),
                "leases": len(instance.leases),
                "admitted_tokens": instance.admitted_tokens,
                "worker_queued_requests": instance.worker_queued_requests,
                "worker_queued_tokens": instance.worker_queued_tokens,
                "worker_queued_group_ids": list(instance.worker_queued_group_ids),
                "worker_running_requests": instance.worker_running_requests,
                "worker_running_tokens": instance.worker_running_tokens,
                "worker_running_group_ids": list(instance.worker_running_group_ids),
                "worker_load": instance.worker_load,
            }
            for node_key, instance in self.instances.items()
        }
