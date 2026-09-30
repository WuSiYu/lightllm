"""Elastic TP scheduler built on V10's EEVDF/deadline admission policy.

V11 keeps V10's ordinary-PD lease lifecycle, virtual service clock, deadline
guard, and non-preemptive overlap reservation.  Its routing policy is elastic:
every pending request is evaluated against every available TP instance.  The
measured service curve, current virtual load, active MPS overlap, token credit,
and request deadline determine the selected TP.  There is deliberately no
request-length routing threshold.

The ``long_request_threshold`` constructor argument is accepted only so old
CLI/configuration files remain loadable.  V11 overrides the V9 class and level
methods, so that value is never consulted by admission or routing.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Tuple

from .flex_tp_selector_v10 import FlexTPSelectorV10
from .flex_tp_selector_v9 import RequestClass, V9Instance, V9Job


class FlexTPSelectorV11(FlexTPSelectorV10):
    """Deadline-aware elastic TP selection with no length partition."""

    is_flex_tp_v11 = True
    is_flex_tp_v10 = False
    pending_type = V9Job

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
        mode_slowdowns=None,
        mps_overlap_slowdown: float = 2.0,
        max_accepted_bundles_per_instance: int = 64,
        overload_policy: str = "best_effort",
        short_weight: float = 1.0,
        long_weight: float = 1.0,
        slack_weight: float = 1.0,
        overlap_slack_ratio: float = 0.10,
        routing_cost_weight: float = 0.50,
    ) -> None:
        # Keep the V10 signature for production callers.  V11 overrides every
        # method that could use the inherited length partition.
        super().__init__(
            pd_manager,
            slo_ttft=slo_ttft,
            long_request_threshold=long_request_threshold,
            batch_token_cap=batch_token_cap,
            batch_token_trigger=batch_token_trigger,
            max_inflight_per_instance=max_inflight_per_instance,
            max_admitted_tokens_per_instance=max_admitted_tokens_per_instance,
            batch_window_s=batch_window_s,
            prediction_margin_s=prediction_margin_s,
            replan_interval_s=replan_interval_s,
            mode_slowdowns=mode_slowdowns,
            mps_overlap_slowdown=mps_overlap_slowdown,
            max_accepted_bundles_per_instance=max_accepted_bundles_per_instance,
            overload_policy=overload_policy,
            short_weight=short_weight,
            long_weight=long_weight,
            slack_weight=slack_weight,
            overlap_slack_ratio=overlap_slack_ratio,
        )
        self.dynamic_route_evaluations = 0
        self.tp_selection_counts = collections.Counter()
        self.tp_selection_tokens = collections.Counter()
        self.routing_cost_weight = max(0.0, float(routing_cost_weight))

    def _class(self, seq_len: int) -> RequestClass:
        # A single queue is intentional: request length is an input to the
        # service model, not a policy-defined class boundary.
        return "short"

    def _initial_level(self, seq_len: int) -> int:
        # V9's feedback level is also length based.  V11 uses one EDF queue;
        # retaining level zero keeps the inherited job/lifecycle format while
        # preventing an implicit length threshold from affecting admission.
        return 0

    def _routing_instances(self) -> List[V9Instance]:
        return sorted(
            (instance for instance in self.instances.values() if instance.available),
            key=lambda instance: instance.node_key,
        )

    def _candidate(
        self,
        now: float,
        request_class: RequestClass = "short",
    ) -> Optional[Tuple[Tuple[float, ...], V9Job, V9Instance]]:
        """Return the best request/TP pair from one global elastic queue.

        A candidate that is predicted to meet its deadline is preferred to a
        miss.  EDF remains the first ordering key; effective lateness then
        compares the request/instance routes under the continuous resource
        price.  TP2 naturally wins when its fixed overhead is cheaper and TP4
        wins when its service curve or the current queue makes it finish sooner.
        """
        del request_class  # Compatibility with V10's two-class call shape.
        best: Optional[Tuple[Tuple[float, ...], V9Job, V9Instance]] = None
        jobs = sorted(self._pending.values(), key=lambda item: (item.deadline, item.enqueue_order))
        instances = self._routing_instances()
        smallest_tp = min((item.tp_size for item in instances), default=1)
        for job in jobs:
            for instance in instances:
                if not self._accepts(instance, job):
                    continue
                if not self._overlap_allowed(now, job, instance):
                    continue
                self.dynamic_route_evaluations += 1
                predicted = self._predicted_finish(now, instance, job)
                service = self._service(job, instance)
                active = self._active(instance)
                overlap_factor = max(0.0, self._slowdown(instance, active) - 1.0)
                # TP4 consumes twice the GPU footprint of TP2 in the common
                # topology.  Price that footprint and any shared-MPS
                # slowdown continuously by service time; no token boundary is
                # introduced.
                footprint_factor = max(0.0, instance.tp_size / smallest_tp - 1.0)
                route_price = self.routing_cost_weight * service * (
                    footprint_factor + overlap_factor
                )
                effective_predicted = predicted + route_price
                lateness = predicted - job.deadline
                miss = 1.0 if lateness > 0.0 else 0.0
                # For a fixed request/deadline, lower effective finish is
                # always the better TP route.  Using V10's positive-laxity
                # key directly would invert that comparison because V10 only
                # ever compared one pre-routed instance per class.
                urgency_key = (effective_predicted - job.deadline) / max(
                    self.slo_ttft, service, 1e-6
                )
                score = (
                    miss,
                    job.deadline,
                    urgency_key * self.slack_weight,
                    effective_predicted,
                    predicted,
                    service,
                    float(instance.admitted_tokens),
                    float(instance.tp_size),
                    float(job.enqueue_order),
                    instance.node_key,
                )
                candidate = (score, job, instance)
                if best is None or score < best[0]:
                    best = candidate
        return best

    def _admit(self, now: float, job: V9Job, instance: V9Instance) -> None:
        self.tp_selection_counts[instance.tp_size] += 1
        self.tp_selection_tokens[instance.tp_size] += job.seq_len
        super()._admit(now, job, instance)

    def _schedule_locked(self, now: float) -> None:
        if not self.decode_nodes or not any(instance.available for instance in self.instances.values()):
            self._fail_unserviceable_pending()
            return
        self.flow_decisions += 1
        while self._pending:
            selected = self._candidate(now)
            if selected is None:
                expired = [job for job in self._pending.values() if job.deadline <= now]
                if expired and self.overload_policy == "reject":
                    job = min(expired, key=lambda item: (item.deadline, item.enqueue_order))
                    self._pending.pop(job.internal_id, None)
                    if not job.future.done():
                        job.future.set_exception(RuntimeError("FlexTP V11 rejected request after deadline"))
                    continue
                return
            _, job, instance = selected
            self._admit(now, job, instance)

    def scheduler_snapshot(self) -> Dict:
        snapshot = super().scheduler_snapshot()
        snapshot.update(
            {
                "version": 11,
                "policy": "os_eevdf_elastic_tp",
                "routing": "dynamic_predicted_finish",
                "routing_threshold": None,
                "routing_cost_weight": self.routing_cost_weight,
                "dynamic_route_evaluations": self.dynamic_route_evaluations,
                "tp_selection_counts": dict(self.tp_selection_counts),
                "tp_selection_tokens": dict(self.tp_selection_tokens),
            }
        )
        return snapshot
