"""V12: continuous predicted-finish routing built on the V6 lease scheduler.

V12 keeps V6's bounded Prefill leases, bundle lifecycle, EDF admission and
MPS-aware prediction, but removes the request-length threshold from routing.
Every pending request is evaluated on every resident TP instance.  The
selector minimizes predicted completion time plus a transparent GPU-footprint
price.  The price discourages sending small work to TP4 merely because its
single-request kernel is a little faster, while the measured service curve and
current overlap/queue load can still move larger work to TP4.
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Tuple

from .flex_tp_selector_v6 import (
    FlexTPSelectorV6,
    RequestClass,
    RunMode,
    V6InstanceState,
    V6Pending,
)


class FlexTPSelectorV12(FlexTPSelectorV6):
    """Elastic V6 admission with continuous finish-time routing."""

    is_flex_tp_v12 = True
    is_flex_tp_v6 = False
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
        latency_constants=None,
        mode_slowdowns=None,
        mps_overlap_slowdown: float = 2.0,
        max_accepted_bundles_per_instance: int = 64,
        preview_request_limit: int = 128,
        preview_pending_batches_per_instance: int = 1,
        request_utility_tokens: float = 2000.0,
        routing_cost_weight: float = 2.0,
        tp4_service_ratio_limit: float = 0.60,
        overload_policy: str = "best_effort",
    ) -> None:
        # ``long_request_threshold`` remains accepted for old launchers, but
        # is intentionally not read anywhere in V12's routing path.
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
            latency_constants=latency_constants,
            mode_slowdowns=mode_slowdowns,
            mps_overlap_slowdown=mps_overlap_slowdown,
            max_accepted_bundles_per_instance=max_accepted_bundles_per_instance,
            preview_request_limit=preview_request_limit,
            preview_pending_batches_per_instance=preview_pending_batches_per_instance,
            request_utility_tokens=request_utility_tokens,
            overload_policy=overload_policy,
        )
        self.routing_cost_weight = max(0.0, float(routing_cost_weight))
        self.tp4_service_ratio_limit = min(1.0, max(0.0, float(tp4_service_ratio_limit)))
        self.dynamic_route_evaluations = 0
        self.tp_selection_counts = collections.Counter()
        self.tp_selection_tokens = collections.Counter()

    def _request_class(self, seq_len: int) -> RequestClass:
        """Return a model-derived hint, never a configured length bucket.

        The hint is used only for V6's overlap counters and diagnostics.  The
        actual candidate set is all TP instances in ``_targets`` below.
        """
        min_tp, max_tp = self._tp_bounds()
        if min_tp == max_tp:
            return "short"
        tp2 = self.latency_model.predict_exclusive(seq_len, min_tp)
        tp4 = self.latency_model.predict_exclusive(seq_len, max_tp)
        return "long" if tp4 < tp2 else "short"

    def _targets(self, request_class: RequestClass) -> List[V6InstanceState]:
        del request_class
        return sorted(
            (state for state in self.instances.values() if state.available),
            key=lambda state: state.node_key,
        )

    def _choose_mode(self, now: float) -> RunMode:
        # V6's rescue preview is meaningful when two disjoint request classes
        # own disjoint TP lanes.  V12 instead exposes one elastic queue and
        # prices overlap in each candidate's predicted finish.  ALL mode keeps
        # the admission path non-blocking and lets the finish-time objective
        # make the TP2/TP4 decision.
        # Reuse V6's counterfactual rescue decision.  Since V12's target set
        # is elastic, the preview is a workload-level signal rather than a
        # fixed short/long lane reservation; it can still serialize a class
        # when overlap would miss materially more deadlines.
        return super()._choose_mode(now)

    def _predict_finish(
        self,
        now: float,
        state: V6InstanceState,
        pending: V6Pending,
        mode: RunMode,
    ) -> float:
        del mode
        entries = [
            (lease.internal_id, lease.seq_len, lease.deadline)
            for lease in state.leases.values()
        ]
        entries.append((pending.internal_id, pending.seq_len, pending.deadline))
        elapsed = sum(
            self.latency_model.predict_batch([entry[1] for entry in batch], state.tp_size)
            for batch in self._pack_entries(entries)
        )
        # Include every currently busy overlapping instance, even when all
        # requests happen to share the same model-derived class hint.
        factor = self._slowdown(state, self._active_instances(include=state))
        return now + elapsed * factor + self.batch_window_s + self.prediction_margin_s

    def _route_cost(self, state: V6InstanceState, pending: V6Pending) -> float:
        available_tps = [
            instance.tp_size
            for instance in self.instances.values()
            if instance.available
        ]
        smallest_tp = max(1, min(available_tps, default=state.tp_size))
        # TP4 has twice the footprint of TP2 in the target topology.  This
        # continuous price is deliberately independent of a token boundary.
        footprint = max(1.0, state.tp_size / smallest_tp)
        service = self.latency_model.predict_exclusive(pending.seq_len, state.tp_size)
        return self.routing_cost_weight * service * footprint

    def _tp4_route_allowed(self, state: V6InstanceState, pending: V6Pending) -> bool:
        """Protect aggregate TP2 capacity using the measured service curve.

        A TP4 candidate is eligible when its solo service is at most the
        configured profile ratio of TP2 service.  This uses no request-length
        bucket; the ratio is evaluated per request.  Keeping the guard strict
        avoids turning temporary TP2 token-credit pressure into a second MPS
        workload, which would defeat V6's aggregate-throughput floor.
        """
        min_tp, max_tp = self._tp_bounds()
        if min_tp == max_tp or state.tp_size != max_tp:
            return True
        tp2_service = self.latency_model.predict_exclusive(pending.seq_len, min_tp)
        tp4_service = self.latency_model.predict_exclusive(pending.seq_len, max_tp)
        if tp4_service <= self.tp4_service_ratio_limit * tp2_service:
            return True
        return False

    def _schedule_locked(self, now: float) -> None:
        if not self.decode_nodes or not any(state.available for state in self.instances.values()):
            self._fail_unserviceable_pending()
            return

        while self._pending:
            mode = self._choose_mode(now)
            candidates = []
            for pending in sorted(
                self._pending.values(),
                key=lambda item: (item.deadline, item.enqueue_order),
            ):
                for state in self._targets("short"):
                    if not self._instance_accepts(state, pending):
                        continue
                    if not self._tp4_route_allowed(state, pending):
                        continue
                    self.dynamic_route_evaluations += 1
                    predicted = self._predict_finish(now, state, pending, mode)
                    exclusive_service = self.latency_model.predict_exclusive(
                        pending.seq_len, state.tp_size
                    )
                    effective = predicted + self._route_cost(state, pending)
                    candidates.append(
                        (
                            pending.deadline,
                            pending.enqueue_order,
                            effective,
                            # If the priced finish is tied, prefer the
                            # lower solo service curve before load/TP size.
                            exclusive_service,
                            predicted,
                            state.admitted_tokens,
                            len(state.leases),
                            state.tp_size,
                            state.node_key,
                            pending,
                            state,
                        )
                    )
            if not candidates:
                expired = [
                    item for item in self._pending.values() if item.deadline <= now
                ]
                if expired and self.overload_policy == "reject":
                    pending = min(expired, key=lambda item: (item.deadline, item.enqueue_order))
                    self._pending.pop(pending.internal_id, None)
                    if not pending.future.done():
                        pending.future.set_exception(
                            RuntimeError("FlexTP V12 rejected request after deadline")
                        )
                    continue
                return
            *_, pending, state = min(candidates)
            self.tp_selection_counts[state.tp_size] += 1
            self.tp_selection_tokens[state.tp_size] += pending.seq_len
            self._admit(now, pending, state, mode)

    def scheduler_snapshot(self) -> Dict:
        snapshot = super().scheduler_snapshot()
        snapshot.update(
            {
                "version": 12,
                "policy": "v6_continuous_predicted_finish",
                "routing": "all_instances_predicted_finish_plus_footprint_price",
                "routing_threshold": None,
                "routing_cost_weight": self.routing_cost_weight,
                "tp4_service_ratio_limit": self.tp4_service_ratio_limit,
                "dynamic_route_evaluations": self.dynamic_route_evaluations,
                "tp_selection_counts": dict(self.tp_selection_counts),
                "tp_selection_tokens": dict(self.tp_selection_tokens),
            }
        )
        return snapshot
