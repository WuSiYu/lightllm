"""V13: deadline-aware V12 routing with pressure-triggered TP4 spill.

V12's fixed TP4 service-ratio guard protects aggregate TP2 capacity, but it
can leave the TP4 lane idle while a TP2 queue is already building a tail.
V13 keeps the V12 lifecycle and objective, then relaxes that guard only when
the measured TP2 token credit is under pressure.  The latency scale is an
explicit calibration knob for a new hardware profile and is local to V13.
"""

from __future__ import annotations

import time
import logging
from typing import Optional

from .flex_tp_selector_v12 import FlexTPSelectorV12
from .flex_tp_selector_v6 import RunMode, V6InstanceState, V6Pending

logger = logging.getLogger(__name__)


class FlexTPSelectorV13(FlexTPSelectorV12):
    """V12 with deadline-sensitive TP4 spill and calibrated prediction scale."""

    is_flex_tp_v13 = True
    is_flex_tp_v12 = False

    def __init__(
        self,
        pd_manager,
        slo_ttft: Optional[float] = 5.0,
        *,
        latency_scale: float = 1.0,
        tp4_service_ratio_limit: float = 0.60,
        tp4_pressure_threshold: float = 1.0,
        routing_cost_weight: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(
            pd_manager,
            slo_ttft=slo_ttft,
            routing_cost_weight=routing_cost_weight,
            tp4_service_ratio_limit=tp4_service_ratio_limit,
            **kwargs,
        )
        if latency_scale <= 0:
            raise ValueError("latency_scale must be positive")
        self.latency_scale = float(latency_scale)
        self.tp4_pressure_threshold = min(1.0, max(0.0, float(tp4_pressure_threshold)))
        self.pressure_spill_count = 0
        self.long_request_tp4_count = 0

    def _instance_accepts(self, state: V6InstanceState, pending: V6Pending) -> bool:
        min_tp, max_tp = self._tp_bounds()
        if min_tp != max_tp and state.tp_size == min_tp and pending.seq_len > self.long_request_threshold:
            return False
        return super()._instance_accepts(state, pending)

    def _predict_finish(
        self,
        now: float,
        state: V6InstanceState,
        pending: V6Pending,
        mode: RunMode,
    ) -> float:
        return now + (super()._predict_finish(now, state, pending, mode) - now) * self.latency_scale

    def _tp2_pressure(self) -> float:
        tp2 = [
            state
            for state in self.instances.values()
            if state.available and state.tp_size == self._tp_bounds()[0]
        ]
        if not tp2:
            return 0.0
        return max(
            min(1.0, state.admitted_tokens / max(1, self.bundle_credit_tokens))
            for state in tp2
        )

    def _tp4_route_allowed(self, state: V6InstanceState, pending: V6Pending) -> bool:
        min_tp, max_tp = self._tp_bounds()
        if min_tp == max_tp or state.tp_size != max_tp:
            return True
        if pending.seq_len > self.long_request_threshold:
            self.long_request_tp4_count += 1
            return True
        tp2_service = self.latency_model.predict_exclusive(pending.seq_len, min_tp)
        tp4_service = self.latency_model.predict_exclusive(pending.seq_len, max_tp)
        ratio_ok = tp4_service <= self.tp4_service_ratio_limit * tp2_service
        if ratio_ok:
            return True

        pressure = self._tp2_pressure()
        # A pending request close to its deadline is allowed to use spare TP4
        # capacity even when its solo speedup is modest.
        tp2_states = [
            candidate
            for candidate in self.instances.values()
            if candidate.available and candidate.tp_size == min_tp
        ]
        urgent = pending.deadline <= time.time() and not any(
            self._instance_accepts(candidate, pending) for candidate in tp2_states
        )
        if pressure >= self.tp4_pressure_threshold or urgent:
            self.pressure_spill_count += 1
            return True
        return False

    def _route_cost(self, state: V6InstanceState, pending: V6Pending) -> float:
        base = super()._route_cost(state, pending)
        if state.tp_size == max(self._tp_bounds()):
            # Make TP4 progressively cheaper as TP2 credit saturates.  The
            # price never reaches zero, so a lightly loaded TP2 remains the
            # stable default for small work.
            pressure = self._tp2_pressure()
            if pressure <= self.tp4_pressure_threshold:
                return base
            return base * max(0.25, 1.0 - 0.75 * pressure)
        return base * self.latency_scale

    def scheduler_snapshot(self):
        snapshot = super().scheduler_snapshot()
        snapshot.update(
            {
                "version": 13,
                "policy": "v12_deadline_pressure_spill",
                "latency_scale": self.latency_scale,
                "tp4_pressure_threshold": self.tp4_pressure_threshold,
                "pressure_spill_count": self.pressure_spill_count,
                "long_request_tp4_count": self.long_request_tp4_count,
            }
        )
        return snapshot
