#!/usr/bin/env python3
"""Mechanism experiments for FlexTP V3 versus threshold-based MPS.

This driver deliberately includes a post-hoc oracle threshold baseline.  It
uses the production selectors and local ChunkedPrefillQueue through
``flex_tp_step_sim``; only workloads and experiment aggregation live here.
Decode is non-bottlenecking in every experiment.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from flex_tp_step_sim import (
    FlexTPStepSimulator,
    SimulationConfig,
    WorkloadRequest,
    _quiet_production_loggers,
)


REPORT_METRICS = (
    "completion_fraction",
    "offered_ttft_slo_attainment",
    "on_time_goodput_rps",
    "on_time_input_token_goodput_s",
    "offered_window_on_time_goodput_rps",
    "offered_window_on_time_input_token_goodput_s",
    "ttft_p95_s",
    "short_ttft_p95_s",
    "long_ttft_p95_s",
    "physical_gpu_s_per_on_time_request",
    "prefill_step_mean_batch",
    "tp2_request_share",
    "mps_overlap_wall_s",
)

REQUEST_GOODPUT_METRIC = "offered_window_on_time_goodput_rps"
TOKEN_GOODPUT_METRIC = "offered_window_on_time_input_token_goodput_s"


def policy_specs(workload: Sequence[WorkloadRequest]) -> List[Tuple[str, str, int]]:
    """Enumerate every distinct threshold mapping for a workload.

    A length-threshold policy only changes behavior when the threshold crosses
    an observed request length.  Including every such breakpoint makes the
    post-hoc oracle exact for this policy family, rather than a favorable sample
    of hand-picked thresholds.  The requested 4000-token baseline is retained
    even when it is not itself a workload breakpoint.
    """
    lengths = sorted({request.input_tokens for request in workload})
    thresholds = sorted({0, 4000, *lengths})
    max_length = lengths[-1]
    policies: List[Tuple[str, str, int]] = [
        ("v3", "v3", 4000),
        ("v4", "v4", 4000),
        ("v5", "v5", 4000),
    ]
    for threshold in thresholds:
        if threshold == 0:
            name = "naive-all-tp4"
        elif threshold == 4000:
            name = "naive-4k"
        elif threshold == max_length:
            name = "naive-all-tp2"
        else:
            name = f"naive-{threshold}"
        policies.append((name, "naive", threshold))
    return policies


def _append_fixed_rate(
    requests: List[WorkloadRequest],
    *,
    start_s: float,
    duration_s: float,
    rate: float,
    lengths: Sequence[int],
    source: str,
) -> None:
    count = int(duration_s * rate)
    for index in range(count):
        length = int(lengths[index % len(lengths)])
        requests.append(
            WorkloadRequest(
                request_id=len(requests) + 1,
                arrival_s=start_s + index / rate,
                input_tokens=length,
                output_tokens=1,
                source=source,
                is_long=length > 4000,
            )
        )


def threshold_cliff() -> List[WorkloadRequest]:
    """Nearly identical lengths land on opposite sides of a static threshold."""
    requests: List[WorkloadRequest] = []
    _append_fixed_rate(
        requests,
        start_s=0.0,
        duration_s=20.0,
        rate=4.0,
        lengths=(3900,),
        source="threshold-cliff-3900",
    )
    _append_fixed_rate(
        requests,
        start_s=20.0,
        duration_s=20.0,
        rate=4.0,
        lengths=(4100,),
        source="threshold-cliff-4100",
    )
    return requests


def phase_shift(seed: int) -> List[WorkloadRequest]:
    """Short-heavy, long-heavy, then mixed phases at one configured policy."""
    rng = random.Random(seed)
    requests: List[WorkloadRequest] = []
    _append_fixed_rate(
        requests,
        start_s=0.0,
        duration_s=20.0,
        rate=20.0,
        lengths=(256, 256, 512, 256, 512),
        source="phase-short",
    )
    _append_fixed_rate(
        requests,
        start_s=20.0,
        duration_s=20.0,
        rate=3.0,
        lengths=(8000,),
        source="phase-long",
    )
    mixed_lengths = tuple(
        rng.choices((256, 1024, 4100, 8000), weights=(55, 20, 15, 10), k=160)
    )
    _append_fixed_rate(
        requests,
        start_s=40.0,
        duration_s=20.0,
        rate=8.0,
        lengths=mixed_lengths,
        source="phase-mixed",
    )
    return sorted(requests, key=lambda item: (item.arrival_s, item.request_id))


def overlap_stress(epochs: int = 8) -> List[WorkloadRequest]:
    """A TP4 request followed by enough TP2 work to expose harmful overlap.

    At slowdown=2 this is approximately work-conserving.  Factors above 2
    represent super-linear NCCL/kernel contention, where serialization or
    deadline-gated overlap can improve both goodput and GPU cost.
    """
    requests: List[WorkloadRequest] = []
    for epoch in range(epochs):
        base = epoch * 4.0
        requests.append(
            WorkloadRequest(
                request_id=len(requests) + 1,
                arrival_s=base,
                input_tokens=20_000,
                output_tokens=1,
                source="overlap-long",
                is_long=True,
            )
        )
        for index in range(80):
            requests.append(
                WorkloadRequest(
                    request_id=len(requests) + 1,
                    arrival_s=base + 0.100 + index * 0.0001,
                    input_tokens=256,
                    output_tokens=1,
                    source="overlap-short",
                    is_long=False,
                )
            )
    return requests


def build_scenarios(seed: int) -> Dict[str, List[WorkloadRequest]]:
    return {
        "threshold-cliff": threshold_cliff(),
        "phase-shift": phase_shift(seed),
        "overlap-stress": overlap_stress(),
    }


def parse_slowdowns(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(not math.isfinite(item) or item < 1.0 for item in values):
        raise argparse.ArgumentTypeError("slowdowns must be finite values >= 1")
    return values


def _p95(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def run_experiments(args) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    scenarios = build_scenarios(args.seed)
    results: List[Dict] = []
    oracle_rows: List[Dict] = []
    source_rows: List[Dict] = []
    for slowdown in args.slowdowns:
        for scenario_name, workload in scenarios.items():
            scenario_results = []
            for policy_name, scheduler, threshold in policy_specs(workload):
                config = SimulationConfig(
                    scheduler=scheduler,
                    naive_length_threshold=threshold,
                    mps_overlap_slowdown=slowdown,
                    slo_ttft_s=args.slo_ttft,
                    simulate_decode=False,
                )
                with FlexTPStepSimulator(workload, config) as simulator:
                    summary = simulator.run()
                offered_by_source = collections.Counter(item.source for item in workload)
                completed_by_source: Dict[str, List] = collections.defaultdict(list)
                for result in simulator.results.values():
                    if result.prefill_finish_s is not None:
                        completed_by_source[result.workload.source].append(result)
                for source, offered in sorted(offered_by_source.items()):
                    completed = completed_by_source[source]
                    ttfts = [
                        result.prefill_finish_s - result.workload.arrival_s
                        for result in completed
                    ]
                    on_time = [ttft for ttft in ttfts if ttft <= args.slo_ttft]
                    source_rows.append(
                        {
                            "scenario": scenario_name,
                            "source": source,
                            "policy": policy_name,
                            "naive_length_threshold": threshold if scheduler == "naive" else None,
                            "mps_overlap_slowdown": slowdown,
                            "offered_requests": offered,
                            "completed_requests": len(completed),
                            "on_time_requests": len(on_time),
                            "offered_ttft_slo_attainment": len(on_time) / offered,
                            "completed_ttft_p95_s": _p95(ttfts),
                            "tp2_completed_share": (
                                sum(result.tp_size == 2 for result in completed) / len(completed)
                                if completed
                                else None
                            ),
                        }
                    )
                row = {
                    "scenario": scenario_name,
                    "policy": policy_name,
                    "naive_length_threshold": threshold if scheduler == "naive" else None,
                    "mps_overlap_slowdown": slowdown,
                    "requests": len(workload),
                    **{metric: summary[metric] for metric in REPORT_METRICS},
                }
                results.append(row)
                scenario_results.append(row)
                print(
                    f"{scenario_name} slowdown={slowdown:g} policy={policy_name} "
                    f"offered_slo={row['offered_ttft_slo_attainment']:.3f} "
                    f"offered_window_on_time_tok="
                    f"{row[TOKEN_GOODPUT_METRIC]:.1f}/s "
                    f"p95={row['ttft_p95_s']:.3f}s "
                    f"gpu/on_time={row['physical_gpu_s_per_on_time_request']:.3f}"
                )

            for dynamic_policy in ("v3", "v4", "v5"):
                ours = next(item for item in scenario_results if item["policy"] == dynamic_policy)
                naive_rows = [item for item in scenario_results if item["policy"].startswith("naive-")]
                naive_4k = next(item for item in naive_rows if item["policy"] == "naive-4k")
                oracle_token = max(
                    naive_rows,
                    key=lambda item: (
                        item[TOKEN_GOODPUT_METRIC],
                        item[REQUEST_GOODPUT_METRIC],
                    ),
                )
                oracle_request = max(
                    naive_rows,
                    key=lambda item: (
                        item[REQUEST_GOODPUT_METRIC],
                        item[TOKEN_GOODPUT_METRIC],
                    ),
                )
                oracle_rows.append(
                    {
                        "scenario": scenario_name,
                        "dynamic_policy": dynamic_policy,
                        "mps_overlap_slowdown": slowdown,
                        "dynamic_offered_window_on_time_input_token_goodput_s": ours[
                            TOKEN_GOODPUT_METRIC
                        ],
                        "naive_4k_offered_window_on_time_input_token_goodput_s": naive_4k[
                            TOKEN_GOODPUT_METRIC
                        ],
                        "token_goodput_gain_over_naive_4k": (
                            ours[TOKEN_GOODPUT_METRIC] / naive_4k[TOKEN_GOODPUT_METRIC] - 1.0
                            if naive_4k[TOKEN_GOODPUT_METRIC]
                            else None
                        ),
                        "oracle_token_policy": oracle_token["policy"],
                        "oracle_token_threshold": oracle_token["naive_length_threshold"],
                        "oracle_offered_window_on_time_input_token_goodput_s": oracle_token[
                            TOKEN_GOODPUT_METRIC
                        ],
                        "token_goodput_gain_over_oracle": (
                            ours[TOKEN_GOODPUT_METRIC] / oracle_token[TOKEN_GOODPUT_METRIC] - 1.0
                            if oracle_token[TOKEN_GOODPUT_METRIC]
                            else None
                        ),
                        "dynamic_offered_window_on_time_goodput_rps": ours[REQUEST_GOODPUT_METRIC],
                        "naive_4k_offered_window_on_time_goodput_rps": naive_4k[
                            REQUEST_GOODPUT_METRIC
                        ],
                        "request_goodput_gain_over_naive_4k": (
                            ours[REQUEST_GOODPUT_METRIC] / naive_4k[REQUEST_GOODPUT_METRIC] - 1.0
                            if naive_4k[REQUEST_GOODPUT_METRIC]
                            else None
                        ),
                        "oracle_request_policy": oracle_request["policy"],
                        "oracle_request_threshold": oracle_request["naive_length_threshold"],
                        "oracle_offered_window_on_time_goodput_rps": oracle_request[
                            REQUEST_GOODPUT_METRIC
                        ],
                        "request_goodput_gain_over_oracle": (
                            ours[REQUEST_GOODPUT_METRIC] / oracle_request[REQUEST_GOODPUT_METRIC] - 1.0
                            if oracle_request[REQUEST_GOODPUT_METRIC]
                            else None
                        ),
                        "ours_gpu_s_per_on_time_request": ours[
                            "physical_gpu_s_per_on_time_request"
                        ],
                        "oracle_token_gpu_s_per_on_time_request": oracle_token[
                            "physical_gpu_s_per_on_time_request"
                        ],
                    }
                )
    return results, oracle_rows, source_rows


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, oracle_rows: Sequence[Dict]) -> None:
    lines = [
        "# FlexTP mechanism experiment summary",
        "",
        "The oracle baseline selects the best fixed threshold after seeing the entire trace.",
        "Positive gain means ours exceeds that post-hoc threshold baseline.",
        "",
        "| scenario | policy | slowdown | token gain vs naive-4K | request gain vs naive-4K | oracle threshold | token gain vs oracle | request gain vs oracle | ours/oracle GPU-s per on-time request |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in oracle_rows:
        token_gain = row["token_goodput_gain_over_oracle"]
        request_gain = row["request_goodput_gain_over_oracle"]
        lines.append(
            f"| {row['scenario']} | {row['dynamic_policy']} | {row['mps_overlap_slowdown']:.1f} "
            f"| {row['token_goodput_gain_over_naive_4k'] * 100:+.1f}% "
            f"| {row['request_goodput_gain_over_naive_4k'] * 100:+.1f}% "
            f"| {row['oracle_token_policy']} "
            f"| {token_gain * 100:+.1f}% "
            f"| {request_gain * 100:+.1f}% "
            f"| {row['ours_gpu_s_per_on_time_request']:.3f}/"
            f"{row['oracle_token_gpu_s_per_on_time_request']:.3f} |"
        )
    lines.extend(
        [
            "",
            "These are model-based mechanism results, not GPU measurements. The slowdown and batch",
            "surfaces must be replaced by H200 profiles before using the numbers as a paper result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slowdowns", type=parse_slowdowns, default=[1.6, 2.0, 2.5])
    parser.add_argument("--slo-ttft", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("_/flex_tp_paper_analysis/mechanisms"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _quiet_production_loggers()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, oracle_rows, source_rows = run_experiments(args)
    (args.output_dir / "all_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "oracle_comparison.json").write_text(
        json.dumps(oracle_rows, indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.output_dir / "all_results.csv", results)
    write_csv(args.output_dir / "oracle_comparison.csv", oracle_rows)
    write_csv(args.output_dir / "source_results.csv", source_rows)
    write_report(args.output_dir / "report.md", oracle_rows)
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
