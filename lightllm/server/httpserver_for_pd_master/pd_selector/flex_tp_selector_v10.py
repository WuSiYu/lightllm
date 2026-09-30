"""EEVDF-style TP-SMT scheduler for ordinary PD Prefill.

V10 borrows the operating-system idea of *Earliest Eligible Virtual Deadline
First* (EEVDF) and combines it with a real TTFT deadline guard.  Requests keep
the hard short/long TP partition used by V4-V9, but all pending requests share
one ordered run queue instead of a fixed epoch or a mandatory one-request-per-
class round.  A per-class virtual service clock gives a newly visible class a
fair turn while the real deadline/lateness key protects urgent work.

The implementation reuses V9's well-tested ordinary-PD lease lifecycle and
topology bookkeeping.  The scheduling policy itself is independent: it does
not use V9's MLFQ levels, aging promotions, or class round-robin rule.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from .flex_tp_selector_v9 import (
    FlexTPSelectorV9,
    RequestClass,
    V9Job,
    V9Instance,
)


class FlexTPSelectorV10(FlexTPSelectorV9):
    """Single EEVDF run queue with deadline-aware admission."""

    is_flex_tp_v10 = True
    is_flex_tp_v9 = False
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
    ) -> None:
        # Reuse only lifecycle/topology code.  V10 never calls the V9
        # _schedule_locked implementation or its MLFQ/CFS policy state.
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
            # These are accepted by the inherited job constructor for
            # compatibility, but V10's policy does not use MLFQ aging.
            aging_interval_s=0.150,
            interactive_token_limit=1024,
            short_class_weight=short_weight,
            long_class_weight=long_weight,
        )
        self.short_weight = max(0.01, float(short_weight))
        self.long_weight = max(0.01, float(long_weight))
        self.slack_weight = max(0.0, float(slack_weight))
        self.overlap_slack_ratio = max(0.0, float(overlap_slack_ratio))
        self._vtime: Dict[RequestClass, float] = {"short": 0.0, "long": 0.0}
        self.eevdf_admissions = 0
        self.deadline_guard_admissions = 0
        self.overdue_admissions = 0
        self.overlap_blocked = 0

    def _weight(self, request_class: RequestClass) -> float:
        return self.short_weight if request_class == "short" else self.long_weight

    def _service(self, job: V9Job, instance: V9Instance) -> float:
        return self.profile.predict_batch([job.seq_len], instance.tp_size)

    def _virtual_finish(self, request_class: RequestClass, job: V9Job, instance: V9Instance) -> float:
        # A class that was idle starts at the current minimum virtual time.  It
        # receives one fair turn immediately instead of carrying stale debt.
        baseline = min(self._vtime.values())
        virtual_start = max(self._vtime[request_class], baseline)
        return virtual_start + self._service(job, instance) / self._weight(request_class)

    def _overlap_allowed(self, now: float, job: V9Job, instance: V9Instance) -> bool:
        """Apply a non-preemptive shared-GPU reservation guard.

        Existing leases are treated like OS reservations: a new overlapping
        lease may consume spare slack, but cannot turn an on-time lease into a
        predicted miss unless the new request is already more urgent.  The
        guard only affects future admission; worker steps already in flight
        are never interrupted.
        """
        active = [
            other
            for other in self._active(instance)
            if other is not instance and self._overlap(other, instance)
        ]
        if not active:
            return True
        if job.deadline <= now:
            return True
        candidate_active = self._active(instance)
        candidate_finish = self._predicted_finish(now, instance, job)
        candidate_slack = job.deadline - candidate_finish
        reserve = self.slo_ttft * self.overlap_slack_ratio

        def projected_finish(target: V9Instance, active_states) -> float:
            entries = [(lease.internal_id, lease.seq_len) for lease in target.leases.values()]
            batches = self._pack([(internal_id, length, 0.0) for internal_id, length in entries])
            elapsed = sum(
                self.profile.predict_batch([item[1] for item in batch], target.tp_size)
                for batch in batches
            )
            return max(now, target.virtual_load) + elapsed * self._slowdown(target, active_states)

        before_active = self._active()
        before_slacks = [
            lease.deadline
            - (projected_finish(other, before_active) + self.batch_window_s + self.prediction_margin_s)
            for other in active
            for lease in other.leases.values()
        ]
        if not before_slacks:
            return True
        minimum_before = min(before_slacks)
        if minimum_before < 0.0:
            # Once a lease is already predicted late, serializing behind it
            # cannot recover its deadline and would only create head-of-line
            # blocking for the other TP lane.
            return True
        after_slacks = [
            lease.deadline
            - (projected_finish(other, candidate_active) + self.batch_window_s + self.prediction_margin_s)
            for other in active
            for lease in other.leases.values()
        ]
        minimum_after = min(after_slacks, default=minimum_before)
        # If adding the candidate would consume the existing reservation, hold
        # it back unless the candidate has strictly less slack than the work
        # it would delay. This is a non-preemptive slack-stealing decision.
        allowed = (
            minimum_after >= reserve
            or candidate_slack < minimum_before
        )
        if not allowed:
            self.overlap_blocked += 1
        return allowed

    def _candidate(
        self,
        now: float,
        request_class: RequestClass,
    ) -> Optional[Tuple[Tuple[float, ...], V9Job, V9Instance]]:
        best: Optional[Tuple[Tuple[float, ...], V9Job, V9Instance]] = None
        for job in sorted(
            (item for item in self._pending.values() if self._class(item.seq_len) == request_class),
            key=lambda item: (item.deadline, item.enqueue_order),
        ):
            instance = self._pick_instance(now, request_class, job)
            if instance is None:
                continue
            if not self._overlap_allowed(now, job, instance):
                continue
            predicted = self._predicted_finish(now, instance, job)
            virtual_finish = self._virtual_finish(request_class, job, instance)
            lateness = predicted - job.deadline
            # Feasible work is preferred.  EDF eligibility is the first-order
            # key, which bounds head-of-line delay for an older long request.
            # EEVDF's virtual deadline and normalized laxity then resolve
            # requests with comparable real deadlines.  Among misses, least
            # lateness remains the first-order rule.
            miss = 1.0 if lateness > 0 else 0.0
            normalized_laxity = (job.deadline - predicted) / max(
                self.slo_ttft, self._service(job, instance), 1e-6
            )
            urgency_key = lateness if miss else normalized_laxity
            score = (
                miss,
                job.deadline,
                urgency_key * self.slack_weight,
                virtual_finish,
                predicted,
                job.deadline,
                float(job.enqueue_order),
            )
            candidate = (score, job, instance)
            if best is None or score < best[0]:
                best = candidate
        return best

    def _admit(self, now: float, job: V9Job, instance: V9Instance) -> None:
        request_class = self._class(job.seq_len)
        virtual_finish = self._virtual_finish(request_class, job, instance)
        self._vtime[request_class] = virtual_finish
        predicted = self._predicted_finish(now, instance, job)
        if predicted <= job.deadline:
            self.deadline_guard_admissions += 1
        else:
            self.overdue_admissions += 1
        self.eevdf_admissions += 1
        super()._admit(now, job, instance)

    def _schedule_locked(self, now: float) -> None:
        if not self.decode_nodes or not any(instance.available for instance in self.instances.values()):
            self._fail_unserviceable_pending()
            return
        self.flow_decisions += 1
        while self._pending:
            candidates = {
                request_class: self._candidate(now, request_class)
                for request_class in ("short", "long")
            }
            candidates = {key: value for key, value in candidates.items() if value is not None}
            if not candidates:
                expired = [job for job in self._pending.values() if job.deadline <= now]
                if expired and self.overload_policy == "reject":
                    job = min(expired, key=lambda item: (item.deadline, item.enqueue_order))
                    self._pending.pop(job.internal_id, None)
                    if not job.future.done():
                        job.future.set_exception(RuntimeError("FlexTP V10 rejected request after deadline"))
                    continue
                return
            _, job, instance = min(candidates.values(), key=lambda item: item[0])
            self._admit(now, job, instance)

    def scheduler_snapshot(self) -> Dict:
        return {
            "version": 10,
            "policy": "os_eevdf_deadline_guard",
            "long_request_threshold": self.long_request_threshold,
            "weights": {"short": self.short_weight, "long": self.long_weight},
            "slack_weight": self.slack_weight,
            "virtual_time": dict(self._vtime),
            "token_credit": self.token_credit,
            "mps_overlap_slowdown": self.mps_overlap_slowdown,
            "flow_decisions": self.flow_decisions,
            "eevdf_admissions": self.eevdf_admissions,
            "deadline_guard_admissions": self.deadline_guard_admissions,
            "overdue_admissions": self.overdue_admissions,
            "overlap_slack_ratio": self.overlap_slack_ratio,
            "overlap_blocked": self.overlap_blocked,
            "credit_blocked": self.credit_blocked,
        }
