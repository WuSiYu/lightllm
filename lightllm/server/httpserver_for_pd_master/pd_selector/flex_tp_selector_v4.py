"""Class-isolated successor to :mod:`flex_tp_selector_v3`.

V3 answers "which TP is cheapest for this request?" before accounting for the
future work displaced from that TP.  A run of old long requests can therefore
occupy both TP2 workers and head-of-line block later short work.  V4 keeps the
V3 normal-PD lease/bundle lifecycle, but isolates request classes whenever the
topology exposes more than one TP size:

* requests above ``long_request_threshold`` use the largest resident TP;
* other requests use the smallest resident TP;
* a reservation for the long class never blocks an admissible short request.

This is deliberately a conservative repair.  V5 builds the MPS-first policy on
top of the same class isolation.
"""

from __future__ import annotations

from typing import Optional, Sequence

from .flex_tp_selector_v3 import (
    FlexTPSelectorV3,
    V3Candidate,
    V3InstanceState,
    V3Pending,
)


class FlexTPSelectorV4(FlexTPSelectorV3):
    """V3 lifecycle with TP-class isolation and short-queue protection."""

    is_flex_tp_v4 = True

    def __init__(
        self,
        pd_manager,
        slo_ttft: Optional[float] = 5.0,
        *,
        long_request_threshold: int = 4000,
        **kwargs,
    ) -> None:
        super().__init__(pd_manager, slo_ttft=slo_ttft, **kwargs)
        self.long_request_threshold = max(1, int(long_request_threshold))

    def _available_tp_bounds(self) -> tuple[int, int]:
        tp_sizes = [instance.tp_size for instance in self.instances.values() if instance.available]
        if not tp_sizes:
            return (1, 1)
        return (min(tp_sizes), max(tp_sizes))

    def _is_long(self, pending: V3Pending) -> bool:
        return pending.seq_len > self.long_request_threshold

    def _placement_allowed(self, pending: V3Pending, instance: V3InstanceState) -> bool:
        min_tp, max_tp = self._available_tp_bounds()
        if min_tp == max_tp:
            return instance.tp_size == min_tp
        target_tp = max_tp if self._is_long(pending) else min_tp
        return instance.tp_size == target_tp

    def _candidate(
        self,
        now: float,
        pending: V3Pending,
        instance: V3InstanceState,
    ) -> Optional[V3Candidate]:
        if not self._placement_allowed(pending, instance):
            return None
        return super()._candidate(now, pending, instance)

    def _reservation_allows(
        self,
        candidate: V3Candidate,
        reservations: Sequence[tuple[tuple[float, int], V3InstanceState]],
    ) -> bool:
        # Long-class reservations protect TP4 capacity, not the overlapping
        # TP2 short queues.  Existing leases are still protected by
        # candidate.protects_existing in the V3 feasibility check.
        if not self._is_long(candidate.pending):
            return True
        return super()._reservation_allows(candidate, reservations)

    def scheduler_snapshot(self):
        return {
            "version": 4,
            "long_request_threshold": self.long_request_threshold,
        }
