"""Independent rescue-only MPS scheduler for ordinary PD Prefill.

V6 starts from the useful behavior of the naive threshold scheduler instead
of inheriting the V3 candidate/reservation machinery.  Requests are split into
short and long queues, routed to the smallest and largest resident TP sizes,
and run in the ALL mode by default.  A class receives a temporary exclusive
bundle window only when the profiled solo mode can rescue work that the ALL
mode predicts will miss its deadline.

The execution credit is deliberately shallow: each instance exposes only a
small input-token horizon to the worker.  This bounds the amount of hidden work
that can delay a later mode decision while preserving the normal-PD
bundle/ACK/completion protocol.
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
RunMode = Literal["all", "short_only", "long_only"]


@dataclass
class V6Pending:
    internal_id: int
    external_req_id: Optional[int]
    seq_len: int
    deadline: float
    future: asyncio.Future
    enqueue_order: int


@dataclass
class V6Lease:
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
    bundle_id: Optional[int] = None
    state: str = "admitted"


@dataclass
class V6InstanceState:
    node: PD_Client_Obj
    node_key: str
    group_id: str
    tp_size: int
    gpu_set: FrozenSet[str]
    placement_is_explicit: bool
    instance_generation: Optional[str] = None
    available: bool = True
    leases: "collections.OrderedDict[int, V6Lease]" = field(default_factory=collections.OrderedDict)
    worker_queued_requests: int = 0
    worker_queued_tokens: int = 0
    worker_running_requests: int = 0
    worker_running_tokens: int = 0
    worker_load: float = 0.0
    last_report_seq: int = -1

    @property
    def busy(self) -> bool:
        return bool(self.leases)

    @property
    def admitted_tokens(self) -> int:
        return sum(lease.seq_len for lease in self.leases.values())

    @property
    def has_accepted_bundle(self) -> bool:
        return any(lease.accepted for lease in self.leases.values())

    @property
    def accepted_bundle_count(self) -> int:
        return len(
            {
                lease.bundle_id
                for lease in self.leases.values()
                if lease.accepted and lease.bundle_id is not None
            }
        )


@dataclass
class V6Preview:
    solo_duration: float = 0.0
    completions: List[Tuple[float, int, float]] = field(default_factory=list)

    def outcome(
        self,
        now: float,
        *,
        factor: float,
        start_delay: float,
        fixed_delay: float,
    ) -> Tuple[int, int]:
        on_time_count = 0
        on_time_tokens = 0
        for relative_finish, seq_len, deadline in self.completions:
            finish = now + start_delay + relative_finish * factor + fixed_delay
            if finish <= deadline:
                on_time_count += 1
                on_time_tokens += seq_len
        return (on_time_count, on_time_tokens)


class V6LatencyModel:
    """Small offline profile used for bundle-level counterfactuals."""

    DEFAULT_CONSTANTS: Dict[int, Tuple[float, float, float, float]] = {
        1: (2.4e-4, 1e-8, 0.05, 0.05),
        2: (2.018352e-4, 6.048427e-9, 2.341091e-2, 6.794939e-2),
        4: (2.655797e-4, 2.330274e-9, 5.651746e-2, 3.767872e-2),
    }

    def __init__(
        self,
        constants: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    ) -> None:
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
        lengths = [max(1, int(length)) for length in seq_lens]
        return max(
            a * sum(lengths) / tp_size + b * sum(length * length for length in lengths) / tp_size,
            c,
        ) + d

    def predict_exclusive(self, seq_len: int, tp_size: int) -> float:
        return self.predict_batch([seq_len], tp_size)


class FlexTPSelectorV6(PDSelector):
    """Threshold routing with ALL-by-default, rescue-only MPS gating."""

    supports_bundles = True
    uses_prefill_lease_lifecycle = True
    is_flex_tp_v6 = True
    pending_type = V6Pending

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
        latency_constants: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
        mode_slowdowns: Optional[Dict[Tuple[int, Tuple[int, ...]], float]] = None,
        mps_overlap_slowdown: float = 2.0,
        max_accepted_bundles_per_instance: int = 64,
        preview_request_limit: int = 128,
        preview_pending_batches_per_instance: int = 1,
        request_utility_tokens: float = 2000.0,
        overload_policy: Literal["best_effort", "reject"] = "best_effort",
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
        # Bound hidden work by the configured 16K worker-step horizon. Small
        # requests may arrive in many transport bundles, so bundle count is a
        # secondary safety cap rather than the primary execution credit.
        self.bundle_credit_tokens = max(1, int(max_admitted_tokens_per_instance))
        self.max_accepted_bundles_per_instance = max(
            1, int(max_accepted_bundles_per_instance)
        )
        self.preview_request_limit = max(1, int(preview_request_limit))
        self.preview_pending_batches_per_instance = max(
            1, int(preview_pending_batches_per_instance)
        )
        self.request_utility_tokens = max(0.0, float(request_utility_tokens))
        self.batch_window_s = max(0.0, float(batch_window_s))
        self.prediction_margin_s = max(0.0, float(prediction_margin_s))
        self.replan_interval_s = max(0.005, float(replan_interval_s))
        self.overload_policy = overload_policy
        self.latency_model = V6LatencyModel(latency_constants)
        self.mps_overlap_slowdown = max(1.0, float(mps_overlap_slowdown))
        self.mode_slowdowns = {
            (int(tp), tuple(sorted(signature))): max(1.0, float(value))
            for (tp, signature), value in (mode_slowdowns or {}).items()
        }

        self.instances: Dict[str, V6InstanceState] = {}
        self.flex_groups: Dict[str, Dict[str, V6InstanceState]] = {}
        self.node_to_group: Dict[str, Dict[str, V6InstanceState]] = {}
        self.node_tp_size: Dict[str, int] = {}
        self._pending: "collections.OrderedDict[int, V6Pending]" = collections.OrderedDict()
        self._leases: Dict[int, V6Lease] = {}
        self._external_to_internal: Dict[int, int] = {}
        self._request_counter = 0
        self._enqueue_counter = 0
        self._decode_rr_index = 0
        self._state_lock = asyncio.Lock()
        self._owner_loop: Optional[asyncio.AbstractEventLoop] = None
        self._replan_task: Optional[asyncio.Task] = None

        self.mode_evaluations = 0
        self.all_mode_decisions = 0
        self.short_only_decisions = 0
        self.long_only_decisions = 0
        self.opposite_drain_waits = 0
        self.admitted_by_mode = collections.Counter()

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
    def _overlap(left: V6InstanceState, right: V6InstanceState) -> bool:
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
            node_key = str(node.client_ip_port)
            group_id = self._group_id(node)
            tp_size = int(self._args(node).get("tp", 1))
            gpu_set, explicit = self._gpu_set(node)
            instance = self.instances.get(node_key)
            if instance is None:
                instance = V6InstanceState(
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
                logger.warning("keep busy V6 instance %s on its old topology", node_key)
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

        groups: Dict[str, Dict[str, V6InstanceState]] = collections.defaultdict(dict)
        for instance in self.instances.values():
            groups[instance.group_id][instance.node_key] = instance
            if not instance.placement_is_explicit:
                logger.warning(
                    "V6 group %s node %s has no explicit GPU placement; assume overlap",
                    instance.group_id,
                    instance.node_key,
                )
        self.flex_groups = dict(groups)
        self.node_to_group = {
            node_key: self.flex_groups[instance.group_id]
            for node_key, instance in self.instances.items()
        }
        self.node_tp_size = {
            node_key: instance.tp_size for node_key, instance in self.instances.items()
        }
        self._fail_unserviceable_pending()
        self._request_reschedule()

    def _tp_bounds(self) -> Tuple[int, int]:
        values = [state.tp_size for state in self.instances.values() if state.available]
        if not values:
            return (1, 1)
        return (min(values), max(values))

    def _request_class(self, seq_len: int) -> RequestClass:
        return "long" if seq_len > self.long_request_threshold else "short"

    def _targets(self, request_class: RequestClass) -> List[V6InstanceState]:
        min_tp, max_tp = self._tp_bounds()
        target_tp = max_tp if request_class == "long" and min_tp != max_tp else min_tp
        return sorted(
            (
                state
                for state in self.instances.values()
                if state.available and state.tp_size == target_tp
            ),
            key=lambda state: state.node_key,
        )

    def _active_instances(
        self,
        include: Optional[V6InstanceState] = None,
    ) -> List[V6InstanceState]:
        values = [state for state in self.instances.values() if state.busy]
        if include is not None and include not in values:
            values.append(include)
        return values

    def _mode_signature(
        self,
        target: V6InstanceState,
        active: Sequence[V6InstanceState],
    ) -> Tuple[int, ...]:
        signature = [target.tp_size]
        signature.extend(
            state.tp_size
            for state in active
            if state is not target and self._overlap(target, state)
        )
        return tuple(sorted(signature))

    def _slowdown(
        self,
        target: V6InstanceState,
        active: Sequence[V6InstanceState],
    ) -> float:
        signature = self._mode_signature(target, active)
        override = self.mode_slowdowns.get((target.tp_size, signature))
        if override is not None:
            return override
        if len(signature) > 1:
            return self.mps_overlap_slowdown
        return 1.0

    def _class_has_leases(self, request_class: RequestClass) -> bool:
        return any(state.busy for state in self._targets(request_class))

    def _class_has_pending(self, request_class: RequestClass) -> bool:
        return any(
            self._request_class(pending.seq_len) == request_class
            for pending in self._pending.values()
        )

    def _class_has_work(self, request_class: RequestClass) -> bool:
        return self._class_has_leases(request_class) or self._class_has_pending(request_class)

    def _pack_entries(
        self,
        entries: Sequence[Tuple[int, int, float]],
    ) -> List[List[Tuple[int, int, float]]]:
        batches: List[List[Tuple[int, int, float]]] = []
        current: List[Tuple[int, int, float]] = []
        current_tokens = 0
        for entry in entries:
            seq_len = entry[1]
            if current and current_tokens + seq_len > self.batch_token_cap:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(entry)
            current_tokens += seq_len
        if current:
            batches.append(current)
        return batches

    def _preview_class(
        self,
        request_class: RequestClass,
    ) -> V6Preview:
        targets = self._targets(request_class)
        if not targets:
            return V6Preview()

        plans: Dict[str, List[List[Tuple[int, int, float]]]] = {}
        pending_batches: Dict[str, List[List[Tuple[int, int, float]]]] = {
            state.node_key: [] for state in targets
        }
        planned_tokens: Dict[str, int] = {}
        for state in targets:
            existing = [
                (lease.internal_id, lease.seq_len, lease.deadline)
                for lease in state.leases.values()
            ]
            plans[state.node_key] = self._pack_entries(existing)
            planned_tokens[state.node_key] = sum(entry[1] for entry in existing)

        pending = sorted(
            (
                item
                for item in self._pending.values()
                if self._request_class(item.seq_len) == request_class
            ),
            key=lambda item: (item.deadline, item.enqueue_order),
        )
        for item in pending[: self.preview_request_limit]:
            eligible = [
                state
                for state in targets
                if len(pending_batches[state.node_key])
                < self.preview_pending_batches_per_instance
                or (
                    pending_batches[state.node_key]
                    and sum(entry[1] for entry in pending_batches[state.node_key][-1])
                    + item.seq_len
                    <= self.bundle_credit_tokens
                )
            ]
            if not eligible:
                break
            state = min(
                eligible,
                key=lambda candidate: (
                    planned_tokens[candidate.node_key],
                    len(pending_batches[candidate.node_key]),
                    candidate.node_key,
                ),
            )
            batches = pending_batches[state.node_key]
            if (
                not batches
                or (
                    batches[-1]
                    and sum(entry[1] for entry in batches[-1]) + item.seq_len
                    > self.bundle_credit_tokens
                )
            ):
                batches.append([])
            batches[-1].append((item.internal_id, item.seq_len, item.deadline))
            planned_tokens[state.node_key] += item.seq_len

        for state in targets:
            plans[state.node_key].extend(pending_batches[state.node_key])

        preview = V6Preview()
        for state in targets:
            elapsed = 0.0
            for batch in plans[state.node_key]:
                service = self.latency_model.predict_batch(
                    [entry[1] for entry in batch],
                    state.tp_size,
                )
                elapsed += service
                for _, seq_len, deadline in batch:
                    preview.completions.append((elapsed, seq_len, deadline))
            preview.solo_duration = max(preview.solo_duration, elapsed)
        return preview

    def _overlap_factor_for_class(self, request_class: RequestClass) -> float:
        targets = self._targets(request_class)
        if not targets:
            return self.mps_overlap_slowdown
        factors = []
        for target in targets:
            if target.tp_size == self._tp_bounds()[0]:
                signature = tuple(sorted((target.tp_size, self._tp_bounds()[1])))
            else:
                short_count = max(1, len(self._targets("short")))
                signature = tuple(sorted([target.tp_size] + [self._tp_bounds()[0]] * short_count))
            factors.append(
                self.mode_slowdowns.get(
                    (target.tp_size, signature),
                    self.mps_overlap_slowdown,
                )
            )
        return max(factors, default=self.mps_overlap_slowdown)

    def _choose_mode(self, now: float) -> RunMode:
        if self._tp_bounds()[0] == self._tp_bounds()[1]:
            return "all"
        short_work = self._class_has_work("short")
        long_work = self._class_has_work("long")
        if not long_work:
            return "short_only"
        if not short_work:
            return "long_only"

        self.mode_evaluations += 1
        short = self._preview_class("short")
        long = self._preview_class("long")
        fixed_delay = self.batch_window_s + self.prediction_margin_s
        short_factor = self._overlap_factor_for_class("short")
        long_factor = self._overlap_factor_for_class("long")

        all_short = short.outcome(
            now, factor=short_factor, start_delay=0.0, fixed_delay=fixed_delay
        )
        all_long = long.outcome(
            now, factor=long_factor, start_delay=0.0, fixed_delay=fixed_delay
        )
        all_outcome = (
            all_short[0] + all_long[0],
            all_short[1] + all_long[1],
        )

        short_first_short = short.outcome(
            now, factor=1.0, start_delay=0.0, fixed_delay=fixed_delay
        )
        short_first_long = long.outcome(
            now,
            factor=1.0,
            start_delay=short.solo_duration,
            fixed_delay=fixed_delay,
        )
        short_first_outcome = (
            short_first_short[0] + short_first_long[0],
            short_first_short[1] + short_first_long[1],
        )

        long_first_long = long.outcome(
            now, factor=1.0, start_delay=0.0, fixed_delay=fixed_delay
        )
        long_first_short = short.outcome(
            now,
            factor=1.0,
            start_delay=long.solo_duration,
            fixed_delay=fixed_delay,
        )
        long_first_outcome = (
            long_first_short[0] + long_first_long[0],
            long_first_short[1] + long_first_long[1],
        )

        def utility(outcome: Tuple[int, int]) -> float:
            return outcome[1] + self.request_utility_tokens * outcome[0]

        all_score = utility(all_outcome)
        short_first_score = utility(short_first_outcome)
        long_first_score = utility(long_first_outcome)

        # ALL wins every tie. Serialization must demonstrate a strict
        # counterfactual gain over the default overlapping mode.
        mode: RunMode = "all"
        best_score = all_score
        if short_first_score > best_score + 1e-9:
            mode = "short_only"
            best_score = short_first_score
        prefer_active_long_on_tie = (
            mode == "short_only"
            and abs(long_first_score - best_score) <= 1e-9
            and self._class_has_leases("long")
            and not self._class_has_leases("short")
        )
        if long_first_score > best_score + 1e-9 or prefer_active_long_on_tie:
            mode = "long_only"

        if mode == "all":
            self.all_mode_decisions += 1
        elif mode == "short_only":
            self.short_only_decisions += 1
        else:
            self.long_only_decisions += 1
        return mode

    def _instance_accepts(self, state: V6InstanceState, pending: V6Pending) -> bool:
        if (
            not state.available
            or state.accepted_bundle_count >= self.max_accepted_bundles_per_instance
        ):
            return False
        if len(state.leases) >= self.max_inflight_per_instance:
            return False
        if state.leases and state.admitted_tokens + pending.seq_len > self.bundle_credit_tokens:
            return False
        return True

    def _predict_finish(
        self,
        now: float,
        state: V6InstanceState,
        pending: V6Pending,
        mode: RunMode,
    ) -> float:
        entries = [(lease.internal_id, lease.seq_len, lease.deadline) for lease in state.leases.values()]
        entries.append((pending.internal_id, pending.seq_len, pending.deadline))
        elapsed = sum(
            self.latency_model.predict_batch([entry[1] for entry in batch], state.tp_size)
            for batch in self._pack_entries(entries)
        )
        factor = 1.0
        if mode == "all" and self._class_has_work("short") and self._class_has_work("long"):
            factor = self._overlap_factor_for_class(self._request_class(pending.seq_len))
        return now + elapsed * factor + self.batch_window_s + self.prediction_margin_s

    def _pick_decode_node(self) -> PD_Client_Obj:
        if not self.decode_nodes:
            raise RuntimeError("no Decode node is registered")
        index = self._decode_rr_index % len(self.decode_nodes)
        self._decode_rr_index += 1
        return self.decode_nodes[index]

    def _admit(self, now: float, pending: V6Pending, state: V6InstanceState, mode: RunMode) -> None:
        self._pending.pop(pending.internal_id, None)
        lease = V6Lease(
            internal_id=pending.internal_id,
            external_req_id=pending.external_req_id,
            seq_len=pending.seq_len,
            deadline=pending.deadline,
            node_key=state.node_key,
            tp_size=state.tp_size,
            predicted_finish=self._predict_finish(now, state, pending, mode),
            dispatch_time=now,
            instance_generation=state.instance_generation,
        )
        state.leases[lease.internal_id] = lease
        self._leases[lease.internal_id] = lease
        if lease.external_req_id is not None:
            self._external_to_internal[lease.external_req_id] = lease.internal_id
        self.admitted_by_mode[mode] += 1
        if not pending.future.done():
            pending.future.set_result((state.node, self._pick_decode_node()))
        logger.info(
            "FlexTP V6 admit req=%s class=%s len=%d mode=%s tp=%d node=%s credit=%d",
            lease.external_req_id,
            self._request_class(lease.seq_len),
            lease.seq_len,
            mode,
            lease.tp_size,
            lease.node_key,
            state.admitted_tokens,
        )

    def _schedule_locked(self, now: float) -> None:
        if not self.decode_nodes or not any(state.available for state in self.instances.values()):
            self._fail_unserviceable_pending()
            return

        while self._pending:
            mode = self._choose_mode(now)
            allowed: Tuple[RequestClass, ...]
            if mode == "short_only":
                allowed = ("short",)
            elif mode == "long_only":
                allowed = ("long",)
            else:
                allowed = ("short", "long")

            # An exclusive decision is non-preemptive. Stop issuing work and
            # let the opposite accepted bundle drain before starting rescue.
            if mode != "all":
                target_class: RequestClass = "short" if mode == "short_only" else "long"
                opposite: RequestClass = "long" if target_class == "short" else "short"
                if self._class_has_leases(opposite):
                    self.opposite_drain_waits += 1
                    return

            candidates = []
            for pending in sorted(
                self._pending.values(),
                key=lambda item: (item.deadline, item.enqueue_order),
            ):
                request_class = self._request_class(pending.seq_len)
                if request_class not in allowed:
                    continue
                for state in self._targets(request_class):
                    if self._instance_accepts(state, pending):
                        predicted_finish = self._predict_finish(now, state, pending, mode)
                        candidates.append(
                            (
                                pending.deadline,
                                pending.enqueue_order,
                                predicted_finish,
                                state.admitted_tokens,
                                len(state.leases),
                                state.node_key,
                                pending,
                                state,
                            )
                        )
            if not candidates:
                expired = [item for item in self._pending.values() if item.deadline <= now]
                if expired and self.overload_policy == "reject":
                    pending = min(expired, key=lambda item: (item.deadline, item.enqueue_order))
                    self._pending.pop(pending.internal_id, None)
                    if not pending.future.done():
                        pending.future.set_exception(
                            RuntimeError("FlexTP V6 rejected request after deadline")
                        )
                    continue
                return

            *_, pending, state = min(candidates)
            self._admit(now, pending, state, mode)

    def _fail_unserviceable_pending(self) -> None:
        if self.decode_nodes and any(state.available for state in self.instances.values()):
            return
        reason = (
            "no Decode node is registered"
            if not self.decode_nodes
            else "no available Prefill instance is registered"
        )
        for internal_id, pending in list(self._pending.items()):
            self._pending.pop(internal_id, None)
            if not pending.future.done():
                pending.future.set_exception(RuntimeError(f"FlexTP V6: {reason}"))

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
            raise RuntimeError("FlexTP V6 requires an asyncio event loop") from exc
        if self._owner_loop is None:
            self._owner_loop = loop
        elif self._owner_loop is not loop:
            raise RuntimeError("FlexTP V6 cannot be used across event loops")

        now = time.time()
        self._request_counter += 1
        self._enqueue_counter += 1
        internal_id = self._request_counter
        future = loop.create_future()
        pending = V6Pending(
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
                raise ValueError(f"FlexTP V6 req_id={req_id} is already admitted or pending")
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
                    state = self.instances.get(lease.node_key)
                    if state is not None:
                        state.leases.pop(internal_id, None)
                    if lease.external_req_id is not None:
                        self._external_to_internal.pop(lease.external_req_id, None)
                self._schedule_locked(time.time())
            raise

    async def select_p_d_node(self, *args, **kwargs):
        return await self.async_select_p_d_node(*args, **kwargs)

    def _find_lease(
        self,
        p_node: Optional[PD_Client_Obj],
        req_id: Optional[int],
    ) -> Optional[V6Lease]:
        if p_node is None:
            return None
        if req_id is None:
            state = self.instances.get(p_node.client_ip_port)
            if state is None or len(state.leases) != 1:
                return None
            lease = next(iter(state.leases.values()))
        else:
            internal_id = self._external_to_internal.get(req_id)
            lease = self._leases.get(internal_id) if internal_id is not None else None
        if lease is None or lease.node_key != p_node.client_ip_port:
            return None
        generation = getattr(p_node, "instance_generation", None)
        if generation is not None and generation != lease.instance_generation:
            return None
        return lease

    async def _finish_request(
        self,
        p_node: Optional[PD_Client_Obj],
        *,
        req_id: Optional[int] = None,
        failed: bool = False,
        reason: str = "",
    ) -> None:
        async with self._state_lock:
            lease = self._find_lease(p_node, req_id)
            if lease is None:
                logger.warning(
                    "FlexTP V6 completion for unknown req=%s node=%s",
                    req_id,
                    getattr(p_node, "client_ip_port", None),
                )
                return
            lease.state = "failed" if failed else "finished"
            self._leases.pop(lease.internal_id, None)
            if lease.external_req_id is not None:
                self._external_to_internal.pop(lease.external_req_id, None)
            state = self.instances.get(lease.node_key)
            if state is not None:
                state.leases.pop(lease.internal_id, None)
            if failed:
                logger.warning(
                    "FlexTP V6 failed req=%s node=%s reason=%s",
                    lease.external_req_id,
                    lease.node_key,
                    reason,
                )
            self._schedule_locked(time.time())
            self._request_reschedule()

    async def notify_request_done(
        self,
        p_node,
        input_token_num: int = 0,
        actual_ttft: Optional[float] = None,
        req_id: Optional[int] = None,
    ) -> None:
        await self._finish_request(p_node, req_id=req_id)

    async def notify_request_failed(
        self,
        p_node,
        input_token_num: int = 0,
        req_id: Optional[int] = None,
        reason: str = "",
    ) -> None:
        await self._finish_request(p_node, req_id=req_id, failed=True, reason=reason)

    async def notify_bundle_accepted(
        self,
        p_node,
        bundle_id: int,
        req_ids: Sequence[int],
    ) -> None:
        async with self._state_lock:
            for req_id in req_ids:
                lease = self._find_lease(p_node, req_id)
                if lease is not None:
                    lease.accepted = True
                    lease.bundle_id = int(bundle_id)
                    lease.state = "queued_worker"

    async def update_instance_report(
        self,
        node_key: str,
        report: Dict,
        report_seq: Optional[int] = None,
    ) -> None:
        async with self._state_lock:
            state = self.instances.get(node_key)
            if state is None:
                return
            if report_seq is not None:
                report_seq = int(report_seq)
                if report_seq <= state.last_report_seq:
                    return
                state.last_report_seq = report_seq
            state.worker_queued_requests = int(report.get("queued_requests", 0))
            state.worker_queued_tokens = int(report.get("queued_tokens", 0))
            state.worker_running_requests = int(report.get("running_requests", 0))
            state.worker_running_tokens = int(report.get("running_tokens", 0))
            state.worker_load = float(report.get("total_token_usage_rate", state.worker_load))

    def notify_node_removed(self, node_key: str, instance_generation=_ALL_GENERATIONS) -> None:
        loop = self._owner_loop
        if loop is None or loop.is_closed():
            return
        loop.create_task(self._release_node_leases(node_key, instance_generation))

    async def _release_node_leases(self, node_key: str, instance_generation=_ALL_GENERATIONS) -> None:
        async with self._state_lock:
            state = self.instances.get(node_key)
            if state is None:
                return
            released = [
                lease
                for lease in state.leases.values()
                if instance_generation is _ALL_GENERATIONS
                or lease.instance_generation == instance_generation
            ]
            for lease in released:
                state.leases.pop(lease.internal_id, None)
                self._leases.pop(lease.internal_id, None)
                if lease.external_req_id is not None:
                    self._external_to_internal.pop(lease.external_req_id, None)
            self._schedule_locked(time.time())

    def snapshot(self) -> Dict[str, Dict]:
        return {
            node_key: {
                "group_id": state.group_id,
                "tp": state.tp_size,
                "gpus": sorted(state.gpu_set),
                "leases": len(state.leases),
                "accepted_bundle": state.has_accepted_bundle,
                "accepted_bundle_count": state.accepted_bundle_count,
                "admitted_tokens": state.admitted_tokens,
                "worker_queued_requests": state.worker_queued_requests,
                "worker_queued_tokens": state.worker_queued_tokens,
                "worker_running_requests": state.worker_running_requests,
                "worker_running_tokens": state.worker_running_tokens,
                "worker_load": state.worker_load,
            }
            for node_key, state in self.instances.items()
        }

    def scheduler_snapshot(self) -> Dict:
        return {
            "version": 6,
            "policy": "rescue_only_mps_gate",
            "long_request_threshold": self.long_request_threshold,
            "bundle_credit_tokens": self.bundle_credit_tokens,
            "max_accepted_bundles_per_instance": self.max_accepted_bundles_per_instance,
            "preview_request_limit": self.preview_request_limit,
            "preview_pending_batches_per_instance": self.preview_pending_batches_per_instance,
            "request_utility_tokens": self.request_utility_tokens,
            "mps_overlap_slowdown": self.mps_overlap_slowdown,
            "mode_evaluations": self.mode_evaluations,
            "all_mode_decisions": self.all_mode_decisions,
            "short_only_decisions": self.short_only_decisions,
            "long_only_decisions": self.long_only_decisions,
            "opposite_drain_waits": self.opposite_drain_waits,
            "admitted_by_mode": dict(sorted(self.admitted_by_mode.items())),
        }
