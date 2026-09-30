"""MPS-first FlexTP scheduler built from the naive policy's useful behavior.

The steady state is concurrent execution of the resident TP2 and TP4
instances.  V5 creates a temporary non-overlap window only when overlap makes
the new request miss its own deadline, or when admitting a long TP4 request
would invalidate already admitted TP2 work.  Short requests are allowed to
overlap older long work: protecting request goodput takes precedence over
preserving a long request that has already consumed its slack.
"""

from __future__ import annotations

from typing import Optional

from .flex_tp_selector_v3 import V3Candidate, V3InstanceState, V3Pending
from .flex_tp_selector_v4 import FlexTPSelectorV4


class FlexTPSelectorV5(FlexTPSelectorV4):
    """Threshold-class placement with concurrent MPS as the default mode."""

    is_flex_tp_v5 = True

    def __init__(
        self,
        *args,
        short_overlap_backlog_trigger: int = 3,
        urgent_slack_ratio: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.short_overlap_backlog_trigger = max(1, int(short_overlap_backlog_trigger))
        self.urgent_slack_ratio = max(0.0, min(1.0, float(urgent_slack_ratio)))
        self.overlap_candidate_evaluations = 0
        self.overlap_admissible_evaluations = 0
        self.serialization_candidate_evaluations = 0

    def _candidate(
        self,
        now: float,
        pending: V3Pending,
        instance: V3InstanceState,
    ) -> Optional[V3Candidate]:
        candidate = super()._candidate(now, pending, instance)
        if candidate is None:
            return None

        overlapping = [
            active
            for active in self._active_instances()
            if active is not instance and self._overlap(active, instance)
        ]
        if not overlapping:
            return candidate

        self.overlap_candidate_evaluations += 1
        if not candidate.candidate_meets_deadline:
            # Waiting for an overlapping lease to drain is the temporary
            # non-overlap state.  The inherited replanner retries on completion.
            self.serialization_candidate_evaluations += 1
            return candidate

        if not self._is_long(pending):
            # Naive MPS gets much of its request goodput from keeping the short
            # TP2 queues live while TP4 processes long work.  With sub-linear
            # slowdown the mode has aggregate gain, so overlap immediately.
            # At equal-share slowdown, preserve a threatened long lease for a
            # lone non-urgent short request; a real short backlog or urgent
            # deadline makes concurrency worthwhile again.
            candidate_active = self._active_instances(include=instance)
            mode_factor = max(
                self._slowdown(active, candidate_active)
                for active in candidate_active
                if active is instance or self._overlap(active, instance)
            )
            short_backlog = sum(
                not self._is_long(item) for item in self._pending.values()
            )
            remaining_slack = pending.deadline - candidate.predicted_finish
            urgent = remaining_slack <= self.slo_ttft * self.urgent_slack_ratio
            if (
                candidate.protects_existing
                or mode_factor < 2.0
                or short_backlog >= self.short_overlap_backlog_trigger
                or urgent
            ):
                candidate.protects_existing = True
                self.overlap_admissible_evaluations += 1
            else:
                self.serialization_candidate_evaluations += 1
            return candidate

        # A new long TP4 request may overlap only if it preserves existing TP2
        # leases.  Otherwise it waits, producing a bounded serial window rather
        # than turning non-overlap into the normal mode.
        if candidate.protects_existing:
            self.overlap_admissible_evaluations += 1
        else:
            self.serialization_candidate_evaluations += 1
        return candidate

    def scheduler_snapshot(self):
        return {
            "version": 5,
            "long_request_threshold": self.long_request_threshold,
            "mode_policy": "mps_first",
            "overlap_candidate_evaluations": self.overlap_candidate_evaluations,
            "overlap_admissible_evaluations": self.overlap_admissible_evaluations,
            "serialization_candidate_evaluations": self.serialization_candidate_evaluations,
            "short_overlap_backlog_trigger": self.short_overlap_backlog_trigger,
            "urgent_slack_ratio": self.urgent_slack_ratio,
        }
