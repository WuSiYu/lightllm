#!/usr/bin/env python3
"""Discrete-event study for deadline-aware FlexTP prefill batching.

This is a design simulator, not a model of the complete LightLLM runtime.  The
exclusive TP2/TP4 latency curves are fitted from the repository's sequential
TTFT sweeps.  Batch amortization and MPS slowdown are explicit assumptions and
are swept by the report instead of being presented as measured facts.

The compared deployments all use eight physical GPUs:

* static_tp2: four disjoint TP2 instances;
* static_tp4: two TP4 instances;
* static_split: two TP2 instances on one four-GPU group and one TP4 instance
  on the other group, with a fixed 2K routing threshold;
* flex_no_mps: two four-GPU Flex groups, each exposing TP2+TP2+TP4, but never
  running overlapping placements concurrently;
* flex_mps: the same Flex groups with deadline-checked MPS overlap;
* flex_single: flex_mps with the current selector's one-request batches.

Run from the repository root:

    python test/benchmark/service/flex_tp_batching_sim.py
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


EPS = 1e-9


@dataclass(frozen=True)
class Job:
    job_id: int
    arrival: float
    input_tokens: int
    slo: float

    @property
    def deadline(self) -> float:
        return self.arrival + self.slo


@dataclass(frozen=True)
class Instance:
    name: str
    tp: int
    gpus: frozenset[int]
    group: int


@dataclass
class ActiveBatch:
    instance: Instance
    jobs: List[Job]
    remaining_work: float
    exclusive_work: float
    start: float

    @property
    def deadline(self) -> float:
        return min(job.deadline for job in self.jobs)


@dataclass
class Candidate:
    instance: Instance
    jobs: List[Job]
    exclusive_work: float
    finish: float
    active_finishes: Dict[str, float]
    protects_active: bool
    meets_own_deadlines: bool

    @property
    def max_lateness(self) -> float:
        return max(self.finish - job.deadline for job in self.jobs)


@dataclass
class Metrics:
    scenario: str
    policy: str
    offered_requests: int
    duration_s: float
    completed_requests: int
    throughput_rps: float
    throughput_tokens_s: float
    slo_attainment: float
    ttft_p50_s: float
    ttft_p95_s: float
    ttft_p99_s: float
    mean_batch_size: float
    mean_batch_tokens: float
    tp2_request_share: float
    exclusive_gpu_seconds: float
    process_gpu_seconds: float
    physical_gpu_seconds: float
    mps_overlap_gpu_seconds: float
    makespan_s: float


class LatencyModel:
    """Measured exclusive curves plus an explicit batch extrapolation.

    The model is ``max(A * sum(L) + B * sum(L^2), c) + d``.  For a
    one-request batch it is the fitted single-request curve.  For a real batch,
    paying ``d`` only once is an unvalidated amortization assumption that must
    be replaced by a measured batch profile matrix before production use.
    """

    # A, B, c, d; seconds.  TP2 comes from ttft_sweep_v7_tp2.csv.  TP4 is
    # refitted from the newer ttft_sweep_v8_tp4.csv, not the stale source value.
    FITS: Dict[int, Tuple[float, float, float, float]] = {
        2: (1.00917613e-4, 3.02421356e-9, 2.34109086e-2, 6.7949394e-2),
        4: (6.63949246e-5, 5.82568552e-10, 5.65174587e-2, 3.76787186e-2),
    }

    def batch_seconds(self, tp: int, jobs: Sequence[Job]) -> float:
        if not jobs:
            return 0.0
        a, b, c, d = self.FITS[tp]
        token_sum = sum(job.input_tokens for job in jobs)
        square_sum = sum(job.input_tokens * job.input_tokens for job in jobs)
        return max(a * token_sum + b * square_sum, c) + d


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def all_instances() -> Dict[str, Instance]:
    instances: Dict[str, Instance] = {}
    for group, offset in ((0, 0), (1, 4)):
        instances[f"g{group}-tp2a"] = Instance(
            f"g{group}-tp2a", 2, frozenset((offset, offset + 1)), group
        )
        instances[f"g{group}-tp2b"] = Instance(
            f"g{group}-tp2b", 2, frozenset((offset + 2, offset + 3)), group
        )
        instances[f"g{group}-tp4"] = Instance(
            f"g{group}-tp4", 4, frozenset(range(offset, offset + 4)), group
        )
    return instances


class Simulator:
    def __init__(
        self,
        scenario: str,
        jobs: Sequence[Job],
        policy: str,
        *,
        batch_window_s: float = 0.020,
        batch_token_limit: int = 8192,
        batch_token_trigger: int = 4096,
        mps_slowdown: float = 1.6,
        dispatch_guard_s: float = 0.010,
    ) -> None:
        self.scenario = scenario
        self.jobs = sorted(copy.copy(list(jobs)), key=lambda job: (job.arrival, job.job_id))
        self.policy = policy
        self.batch_window_s = 0.0 if policy == "flex_single" else batch_window_s
        self.batch_token_limit = batch_token_limit
        self.batch_token_trigger = 1 if policy == "flex_single" else batch_token_trigger
        self.mps_slowdown = max(1.0, mps_slowdown)
        self.dispatch_guard_s = dispatch_guard_s
        self.model = LatencyModel()

        topology = all_instances()
        if policy == "static_tp2":
            names = [name for name, instance in topology.items() if instance.tp == 2]
        elif policy == "static_tp4":
            names = [name for name, instance in topology.items() if instance.tp == 4]
        elif policy == "static_split":
            names = ["g0-tp2a", "g0-tp2b", "g1-tp4"]
        elif policy in ("flex_no_mps", "flex_mps", "flex_single"):
            names = list(topology)
        else:
            raise ValueError(f"unknown policy: {policy}")
        self.instances = [topology[name] for name in names]

        self.now = self.jobs[0].arrival if self.jobs else 0.0
        self.next_arrival_index = 0
        self.pending: List[Job] = []
        self.active: Dict[str, ActiveBatch] = {}
        self.finish_times: Dict[int, float] = {}
        self.job_tp: Dict[int, int] = {}
        self.started_batches: List[Tuple[int, int, float]] = []
        self.exclusive_gpu_seconds = 0.0
        self.process_gpu_seconds = 0.0
        self.physical_gpu_seconds = 0.0
        self.mps_overlap_gpu_seconds = 0.0

    @property
    def mps_enabled(self) -> bool:
        return self.policy in ("flex_mps", "flex_single")

    def _eligible(self, job: Job, instance: Instance) -> bool:
        if self.policy != "static_split":
            return True
        # A static split cannot borrow idle capacity across the class boundary.
        return (job.input_tokens <= 2048 and instance.tp == 2) or (
            job.input_tokens > 2048 and instance.tp == 4
        )

    def _overlaps_active(self, instance: Instance) -> bool:
        return any(instance.gpus & batch.instance.gpus for batch in self.active.values())

    def _slowdown(self, instance: Instance, active_instances: Iterable[Instance]) -> float:
        active_instances = list(active_instances)
        if not any(
            other.name != instance.name and instance.gpus & other.gpus
            for other in active_instances
        ):
            return 1.0
        return self.mps_slowdown

    def _predict_finishes(
        self, instance: Instance, exclusive_work: float
    ) -> Tuple[float, Dict[str, float]]:
        remaining = {name: batch.remaining_work for name, batch in self.active.items()}
        instances = {name: batch.instance for name, batch in self.active.items()}
        candidate_name = "__candidate__"
        remaining[candidate_name] = exclusive_work
        instances[candidate_name] = instance
        finishes: Dict[str, float] = {}
        simulated_now = self.now

        while remaining:
            active_instances = list(instances.values())
            wall_times = {
                name: work * self._slowdown(instances[name], active_instances)
                for name, work in remaining.items()
            }
            advance = min(wall_times.values())
            simulated_now += advance
            for name in list(remaining):
                slowdown = self._slowdown(instances[name], active_instances)
                remaining[name] = max(0.0, remaining[name] - advance / slowdown)
            for name in [name for name, work in remaining.items() if work <= EPS]:
                finishes[name] = simulated_now
                del remaining[name]
                del instances[name]
        return finishes[candidate_name], finishes

    def _predict_active_baseline(self) -> Dict[str, float]:
        """Finish times if no new batch is admitted at the current event."""
        remaining = {name: batch.remaining_work for name, batch in self.active.items()}
        instances = {name: batch.instance for name, batch in self.active.items()}
        finishes: Dict[str, float] = {}
        simulated_now = self.now
        while remaining:
            active_instances = list(instances.values())
            wall_times = {
                name: work * self._slowdown(instances[name], active_instances)
                for name, work in remaining.items()
            }
            advance = min(wall_times.values())
            simulated_now += advance
            for name in list(remaining):
                slowdown = self._slowdown(instances[name], active_instances)
                remaining[name] = max(0.0, remaining[name] - advance / slowdown)
            for name in [name for name, work in remaining.items() if work <= EPS]:
                finishes[name] = simulated_now
                del remaining[name]
                del instances[name]
        return finishes

    def _pack_jobs(self, instance: Instance) -> List[Job]:
        eligible = sorted(
            (job for job in self.pending if self._eligible(job, instance)),
            key=lambda job: (job.deadline, job.arrival, job.job_id),
        )
        if not eligible:
            return []
        if self.policy == "flex_single":
            return eligible[:1]

        packed: List[Job] = []
        token_sum = 0
        for job in eligible:
            if not packed and job.input_tokens >= self.batch_token_limit:
                return [job]
            if token_sum + job.input_tokens <= self.batch_token_limit:
                packed.append(job)
                token_sum += job.input_tokens
        return packed

    def _is_ready(self, jobs: Sequence[Job], exclusive_work: float) -> bool:
        if not jobs:
            return False
        if self.policy == "flex_single":
            return True
        token_sum = sum(job.input_tokens for job in jobs)
        oldest_arrival = min(job.arrival for job in jobs)
        earliest_deadline = min(job.deadline for job in jobs)
        conservative_slowdown = self.mps_slowdown if self._overlaps_active_for_jobs() else 1.0
        latest_start = earliest_deadline - exclusive_work * conservative_slowdown - self.dispatch_guard_s
        return (
            token_sum >= self.batch_token_trigger
            or self.now + EPS >= oldest_arrival + self.batch_window_s
            or self.now + EPS >= latest_start
        )

    def _overlaps_active_for_jobs(self) -> bool:
        # Used only for a conservative urgency bound before choosing a placement.
        return bool(self.active) and self.mps_enabled

    def _candidate(self, instance: Instance) -> Optional[Candidate]:
        if instance.name in self.active:
            return None
        if not self.mps_enabled and self._overlaps_active(instance):
            return None
        jobs = self._pack_jobs(instance)
        if not jobs:
            return None
        exclusive_work = self.model.batch_seconds(instance.tp, jobs)
        if not self._is_ready(jobs, exclusive_work):
            return None
        baseline_finishes = self._predict_active_baseline()
        finish, all_finishes = self._predict_finishes(instance, exclusive_work)
        protects_active = all(
            all_finishes[name]
            <= max(batch.deadline - self.dispatch_guard_s, baseline_finishes[name]) + EPS
            for name, batch in self.active.items()
        )
        meets_own = all(
            finish <= job.deadline - self.dispatch_guard_s + EPS for job in jobs
        )
        return Candidate(
            instance=instance,
            jobs=jobs,
            exclusive_work=exclusive_work,
            finish=finish,
            active_finishes=all_finishes,
            protects_active=protects_active,
            meets_own_deadlines=meets_own,
        )

    def _choose_candidate(self) -> Optional[Candidate]:
        candidates = [candidate for instance in self.instances if (candidate := self._candidate(instance))]
        if not candidates:
            return None

        protected = [candidate for candidate in candidates if candidate.protects_active]
        if not protected:
            # Waiting for an active batch is preferable to knowingly breaking an
            # already admitted deadline.
            return None
        feasible = [candidate for candidate in protected if candidate.meets_own_deadlines]
        pool = feasible or protected

        if self.policy.startswith("flex"):
            if feasible:
                # Lexicographic objective: deadline feasibility, smallest TP,
                # GPU cost, then completion time and batch fill.
                return min(
                    pool,
                    key=lambda candidate: (
                        candidate.instance.tp,
                        candidate.exclusive_work * candidate.instance.tp,
                        candidate.finish,
                        -len(candidate.jobs),
                    ),
                )
            non_overlapping = [
                candidate
                for candidate in pool
                if not any(
                    candidate.instance.gpus & batch.instance.gpus
                    for batch in self.active.values()
                )
            ]
            if non_overlapping:
                return min(
                    non_overlapping,
                    key=lambda candidate: (
                        candidate.max_lateness,
                        candidate.finish,
                        candidate.instance.tp,
                    ),
                )
            if self.active:
                # A request that is feasible on an empty placement should wait
                # for that placement instead of being downgraded to a currently
                # idle but too-slow TP.  Starting best effort here creates a
                # self-inflicted cascade and can starve the future TP4 window.
                return None
            return min(
                pool,
                key=lambda candidate: (
                    candidate.max_lateness,
                    candidate.finish,
                    candidate.instance.tp,
                ),
            )

        # Static policies have no TP choice.  Pick the least-late/earliest free
        # placement; fixed-partition eligibility was applied while packing.
        return min(
            pool,
            key=lambda candidate: (
                not candidate.meets_own_deadlines,
                candidate.max_lateness,
                candidate.finish,
            ),
        )

    def _start_candidate(self, candidate: Candidate) -> None:
        job_ids = {job.job_id for job in candidate.jobs}
        self.pending = [job for job in self.pending if job.job_id not in job_ids]
        self.active[candidate.instance.name] = ActiveBatch(
            instance=candidate.instance,
            jobs=candidate.jobs,
            remaining_work=candidate.exclusive_work,
            exclusive_work=candidate.exclusive_work,
            start=self.now,
        )
        self.started_batches.append(
            (candidate.instance.tp, len(candidate.jobs), sum(job.input_tokens for job in candidate.jobs))
        )
        self.exclusive_gpu_seconds += candidate.exclusive_work * candidate.instance.tp
        for job in candidate.jobs:
            self.job_tp[job.job_id] = candidate.instance.tp

    def _schedule(self) -> None:
        while self.pending:
            candidate = self._choose_candidate()
            if candidate is None:
                return
            self._start_candidate(candidate)

    def _complete_batches(self) -> None:
        completed_names = [
            name for name, batch in self.active.items() if batch.remaining_work <= EPS
        ]
        for name in completed_names:
            batch = self.active.pop(name)
            for job in batch.jobs:
                self.finish_times[job.job_id] = self.now

    def _next_wake(self) -> float:
        wake = math.inf
        for job in self.pending:
            wake = min(wake, job.arrival + self.batch_window_s)
            eligible_tps = [instance.tp for instance in self.instances if self._eligible(job, instance)]
            if eligible_tps:
                solo = min(
                    self.model.batch_seconds(tp, [job])
                    * (self.mps_slowdown if self.mps_enabled and self.active else 1.0)
                    for tp in eligible_tps
                )
                wake = min(wake, job.deadline - solo - self.dispatch_guard_s)
        if wake <= self.now + EPS:
            return math.inf
        return wake

    def _advance(self, next_time: float) -> None:
        delta = next_time - self.now
        if delta < -EPS:
            raise RuntimeError(f"time moved backwards: {self.now} -> {next_time}")
        if delta <= EPS:
            self.now = next_time
            return

        active_instances = [batch.instance for batch in self.active.values()]
        gpu_counts: Dict[int, int] = {}
        for instance in active_instances:
            for gpu in instance.gpus:
                gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
        self.process_gpu_seconds += delta * sum(instance.tp for instance in active_instances)
        self.physical_gpu_seconds += delta * len(gpu_counts)
        self.mps_overlap_gpu_seconds += delta * sum(1 for count in gpu_counts.values() if count > 1)

        for batch in self.active.values():
            slowdown = self._slowdown(batch.instance, active_instances)
            batch.remaining_work = max(0.0, batch.remaining_work - delta / slowdown)
        self.now = next_time

    def run(self) -> Metrics:
        if not self.jobs:
            return Metrics(self.scenario, self.policy, 0, 0.0, 0, *([0.0] * 14))

        while len(self.finish_times) < len(self.jobs):
            self._complete_batches()
            while (
                self.next_arrival_index < len(self.jobs)
                and self.jobs[self.next_arrival_index].arrival <= self.now + EPS
            ):
                self.pending.append(self.jobs[self.next_arrival_index])
                self.next_arrival_index += 1

            self._schedule()
            if len(self.finish_times) >= len(self.jobs):
                break

            events: List[float] = []
            if self.next_arrival_index < len(self.jobs):
                events.append(self.jobs[self.next_arrival_index].arrival)
            if self.active:
                active_instances = [batch.instance for batch in self.active.values()]
                events.extend(
                    self.now + batch.remaining_work * self._slowdown(batch.instance, active_instances)
                    for batch in self.active.values()
                )
            wake = self._next_wake()
            if wake < math.inf:
                events.append(wake)
            future_events = [event for event in events if event > self.now + EPS]
            if not future_events:
                if self.pending and not self.active:
                    raise RuntimeError("pending jobs have no eligible placement or wake event")
                # Numerical tie: advance a negligible amount so completion is observed.
                future_events = [self.now + 1e-8]
            self._advance(min(future_events))

        first_arrival = min(job.arrival for job in self.jobs)
        last_finish = max(self.finish_times.values())
        makespan = max(EPS, last_finish - first_arrival)
        offered_duration = max(EPS, max(job.arrival for job in self.jobs) - first_arrival)
        ttfts = [self.finish_times[job.job_id] - job.arrival for job in self.jobs]
        slo_hits = [ttft <= job.slo + EPS for ttft, job in zip(ttfts, self.jobs)]
        total_tokens = sum(job.input_tokens for job in self.jobs)
        batch_sizes = [size for _, size, _ in self.started_batches]
        batch_tokens = [tokens for _, _, tokens in self.started_batches]
        tp2_count = sum(1 for tp in self.job_tp.values() if tp == 2)

        return Metrics(
            scenario=self.scenario,
            policy=self.policy,
            offered_requests=len(self.jobs),
            duration_s=offered_duration,
            completed_requests=len(self.finish_times),
            throughput_rps=len(self.jobs) / makespan,
            throughput_tokens_s=total_tokens / makespan,
            slo_attainment=sum(slo_hits) / len(slo_hits),
            ttft_p50_s=percentile(ttfts, 0.50),
            ttft_p95_s=percentile(ttfts, 0.95),
            ttft_p99_s=percentile(ttfts, 0.99),
            mean_batch_size=statistics.fmean(batch_sizes),
            mean_batch_tokens=statistics.fmean(batch_tokens),
            tp2_request_share=tp2_count / len(self.jobs),
            exclusive_gpu_seconds=self.exclusive_gpu_seconds,
            process_gpu_seconds=self.process_gpu_seconds,
            physical_gpu_seconds=self.physical_gpu_seconds,
            mps_overlap_gpu_seconds=self.mps_overlap_gpu_seconds,
            makespan_s=makespan,
        )


def weighted_choice(rng: random.Random, choices: Sequence[Tuple[int, float]]) -> int:
    point = rng.random()
    cumulative = 0.0
    for value, weight in choices:
        cumulative += weight
        if point <= cumulative + EPS:
            return value
    return choices[-1][0]


def slo_for_length(length: int) -> float:
    if length <= 512:
        return 1.0
    if length <= 2048:
        return 2.0
    if length <= 4096:
        return 3.0
    return 4.0


def poisson_segment(
    rng: random.Random,
    start: float,
    duration: float,
    rate: float,
    lengths: Sequence[Tuple[int, float]],
    first_id: int,
    slo_scale: float = 1.0,
) -> List[Job]:
    jobs: List[Job] = []
    now = start
    job_id = first_id
    end = start + duration
    while True:
        now += rng.expovariate(rate)
        if now >= end:
            break
        length = weighted_choice(rng, lengths)
        jobs.append(Job(job_id, now, length, slo_for_length(length) * slo_scale))
        job_id += 1
    return jobs


def build_scenarios(seed: int) -> Dict[str, List[Job]]:
    scenarios: Dict[str, List[Job]] = {}

    rng = random.Random(seed + 1)
    scenarios["sparse_short"] = poisson_segment(
        rng, 0.0, 60.0, 4.0, ((128, 0.65), (512, 0.35)), 0
    )

    rng = random.Random(seed + 2)
    burst_jobs: List[Job] = []
    job_id = 0
    for burst in range(80):
        base = burst * 0.25
        for index in range(16):
            length = weighted_choice(rng, ((128, 0.45), (256, 0.35), (512, 0.20)))
            burst_jobs.append(Job(job_id, base + index * 0.0002, length, 1.0))
            job_id += 1
    scenarios["bursty_short"] = burst_jobs

    rng = random.Random(seed + 3)
    scenarios["mixed_poisson"] = poisson_segment(
        rng,
        0.0,
        60.0,
        14.0,
        ((256, 0.62), (1024, 0.20), (4096, 0.12), (8000, 0.06)),
        0,
    )

    rng = random.Random(seed + 4)
    phase_jobs: List[Job] = []
    phase_jobs += poisson_segment(rng, 0.0, 20.0, 52.0, ((128, 0.6), (512, 0.4)), 0)
    phase_jobs += poisson_segment(
        rng, 20.0, 20.0, 2.4, ((4096, 0.25), (8000, 0.75)), len(phase_jobs), 0.8
    )
    phase_jobs += poisson_segment(
        rng,
        40.0,
        20.0,
        13.0,
        ((256, 0.55), (1024, 0.20), (4096, 0.15), (8000, 0.10)),
        len(phase_jobs),
    )
    scenarios["phase_shift"] = sorted(phase_jobs, key=lambda job: (job.arrival, job.job_id))

    rng = random.Random(seed + 5)
    long_jobs = poisson_segment(rng, 0.0, 45.0, 2.6, ((8000, 1.0),), 0)
    scenarios["long_tight"] = [
        Job(job.job_id, job.arrival, job.input_tokens, 1.25) for job in long_jobs
    ]

    rng = random.Random(seed + 6)
    alternating: List[Job] = []
    job_id = 0
    for epoch in range(50):
        base = epoch * 0.8
        # TP2 exclusive is about 1.07 s while TP4 is about 0.61 s.  A 1.05 s
        # deadline therefore forces TP4, but can still permit useful MPS overlap
        # when the measured slowdown is sufficiently below equal sharing.
        alternating.append(Job(job_id, base, 8000, 1.05))
        job_id += 1
        alternating.append(Job(job_id, base + 0.0001, 8000, 1.05))
        job_id += 1
        for index in range(12):
            length = weighted_choice(rng, ((128, 0.7), (512, 0.3)))
            alternating.append(Job(job_id, base + 0.035 + index * 0.0003, length, 0.9))
            job_id += 1
    scenarios["long_short_overlap"] = alternating

    return scenarios


POLICIES = (
    "static_tp2",
    "static_tp4",
    "static_split",
    "flex_single",
    "flex_no_mps",
    "flex_mps",
)


def run_suite(
    seed: int, mps_slowdown: float
) -> Tuple[List[Metrics], List[Metrics], List[Metrics]]:
    scenarios = build_scenarios(seed)
    results: List[Metrics] = []
    for scenario, jobs in scenarios.items():
        for policy in POLICIES:
            results.append(
                Simulator(
                    scenario,
                    jobs,
                    policy,
                    mps_slowdown=mps_slowdown,
                ).run()
            )

    sensitivity: List[Metrics] = []
    for slowdown in (1.3, 1.6, 2.0):
        for scenario in ("mixed_poisson", "phase_shift", "long_short_overlap"):
            metrics = Simulator(
                f"{scenario}@mps={slowdown:.1f}",
                scenarios[scenario],
                "flex_mps",
                mps_slowdown=slowdown,
            ).run()
            sensitivity.append(metrics)

    batching_sensitivity: List[Metrics] = []
    batching_configs = (
        ("single_request", "flex_single", 8192, 1, 0.0),
        ("worker_6000_30ms", "flex_no_mps", 6000, 6000, 0.030),
        ("cooperative_8192_20ms", "flex_no_mps", 8192, 4096, 0.020),
        ("double_wait_8192_50ms", "flex_no_mps", 8192, 4096, 0.050),
    )
    for scenario in ("sparse_short", "bursty_short", "mixed_poisson"):
        for label, policy, token_limit, token_trigger, window in batching_configs:
            metrics = Simulator(
                scenario,
                scenarios[scenario],
                policy,
                batch_window_s=window,
                batch_token_limit=token_limit,
                batch_token_trigger=token_trigger,
                mps_slowdown=mps_slowdown,
            ).run()
            metrics.policy = label
            batching_sensitivity.append(metrics)
    return results, sensitivity, batching_sensitivity


def print_markdown(
    results: Sequence[Metrics],
    sensitivity: Sequence[Metrics],
    batching_sensitivity: Sequence[Metrics],
) -> None:
    print("| scenario | policy | SLO% | p95 TTFT(s) | req/s | tok/s | mean batch | TP2 share |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for item in results:
        print(
            f"| {item.scenario} | {item.policy} | {100 * item.slo_attainment:.1f} | "
            f"{item.ttft_p95_s:.3f} | {item.throughput_rps:.2f} | {item.throughput_tokens_s:.0f} | "
            f"{item.mean_batch_size:.2f} | {100 * item.tp2_request_share:.1f}% |"
        )
    print("\nMPS sensitivity:\n")
    print("| scenario | SLO% | p95 TTFT(s) | req/s | overlap GPU-s |")
    print("|---|---:|---:|---:|---:|")
    for item in sensitivity:
        print(
            f"| {item.scenario} | {100 * item.slo_attainment:.1f} | {item.ttft_p95_s:.3f} | "
            f"{item.throughput_rps:.2f} | {item.mps_overlap_gpu_seconds:.2f} |"
        )
    print("\nBatching sensitivity:\n")
    print("| scenario | batching | SLO% | p95 TTFT(s) | req/s | mean batch | mean tokens |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for item in batching_sensitivity:
        print(
            f"| {item.scenario} | {item.policy} | {100 * item.slo_attainment:.1f} | "
            f"{item.ttft_p95_s:.3f} | {item.throughput_rps:.2f} | "
            f"{item.mean_batch_size:.2f} | {item.mean_batch_tokens:.0f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--mps-slowdown", type=float, default=1.6)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    results, sensitivity, batching_sensitivity = run_suite(args.seed, args.mps_slowdown)
    if args.json:
        print(
            json.dumps(
                {
                    "seed": args.seed,
                    "default_mps_slowdown": args.mps_slowdown,
                    "results": [asdict(item) for item in results],
                    "mps_sensitivity": [asdict(item) for item in sensitivity],
                    "batching_sensitivity": [asdict(item) for item in batching_sensitivity],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print_markdown(results, sensitivity, batching_sensitivity)


if __name__ == "__main__":
    main()
