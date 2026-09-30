#!/usr/bin/env python3
"""Compare production FlexTP V3/V4/V5/V6 selectors on common workloads."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

from flex_tp_paper_experiments import phase_shift
from flex_tp_step_sim import (
    FlexTPStepSimulator,
    SimulationConfig,
    WorkloadRequest,
    _quiet_production_loggers,
    _repo_root,
    generate_servegen_mm_image,
    generate_synthetic_5pct,
)


POLICIES = ("v3", "v4", "v5", "v6", "naive")
TOKEN_METRIC = "offered_window_on_time_input_token_goodput_s"
REQUEST_METRIC = "offered_window_on_time_goodput_rps"


def workloads(seed: int, synthetic_prompts: int) -> Dict[str, Sequence[WorkloadRequest]]:
    result: Dict[str, Sequence[WorkloadRequest]] = {
        "phase-shift": phase_shift(seed),
        "synthetic-5pct-rate10": generate_synthetic_5pct(
            request_rate=10,
            num_prompts=synthetic_prompts,
            seed=seed,
        ),
    }
    for rate in (8, 9, 10):
        result[f"servegen-mm-image-rate{rate}"] = generate_servegen_mm_image(
            repo_root=_repo_root(),
            request_rate=rate,
            duration_s=180,
            seed=seed,
        )
    return result


def parse_slowdowns(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value < 1.0 for value in values):
        raise argparse.ArgumentTypeError("slowdowns must be comma-separated values >= 1")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slowdowns", type=parse_slowdowns, default=[1.6, 2.0])
    parser.add_argument("--synthetic-prompts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_/flex_tp_paper_analysis/v6-comparison"),
    )
    args = parser.parse_args()
    _quiet_production_loggers()

    rows: List[Dict] = []
    comparisons: List[Dict] = []
    for scenario, workload in workloads(args.seed, args.synthetic_prompts).items():
        for slowdown in args.slowdowns:
            summaries = {}
            for policy in POLICIES:
                config = SimulationConfig(
                    scheduler=policy,
                    naive_length_threshold=4000,
                    mps_overlap_slowdown=slowdown,
                    slo_ttft_s=3.0,
                    simulate_decode=False,
                )
                with FlexTPStepSimulator(workload, config) as simulator:
                    summary = simulator.run()
                summaries[policy] = summary
                row = {
                    "scenario": scenario,
                    "slowdown": slowdown,
                    "policy": policy,
                    "offered_slo": summary["offered_ttft_slo_attainment"],
                    "offered_window_request_goodput": summary[REQUEST_METRIC],
                    "offered_window_token_goodput": summary[TOKEN_METRIC],
                    "p95_ttft_s": summary["ttft_p95_s"],
                    "completion_fraction": summary["completion_fraction"],
                    "tp2_long_request_count": summary["tp2_long_request_count"],
                    "mps_overlap_wall_s": summary["mps_overlap_wall_s"],
                    "prefill_step_mean_batch": summary["prefill_step_mean_batch"],
                    "scheduler_stats": json.dumps(summary["scheduler_stats"], sort_keys=True),
                }
                rows.append(row)
                print(
                    f"{scenario} slowdown={slowdown:g} policy={policy} "
                    f"SLO={row['offered_slo']:.4f} "
                    f"token_goodput={row['offered_window_token_goodput']:.1f}/s "
                    f"p95={row['p95_ttft_s']:.3f}s "
                    f"overlap={row['mps_overlap_wall_s']:.1f}s"
                )

            v6 = summaries["v6"]
            if v6["tp2_long_request_count"] != 0:
                raise AssertionError(
                    f"V6 routed long work to TP2 in {scenario} slowdown={slowdown}"
                )
            for baseline in ("v3", "v4", "v5", "naive"):
                other = summaries[baseline]
                comparisons.append(
                    {
                        "scenario": scenario,
                        "slowdown": slowdown,
                        "baseline": baseline,
                        "v6_token_gain_pct": (
                            (v6[TOKEN_METRIC] / other[TOKEN_METRIC] - 1.0) * 100.0
                            if other[TOKEN_METRIC]
                            else None
                        ),
                        "v6_request_gain_pct": (
                            (v6[REQUEST_METRIC] / other[REQUEST_METRIC] - 1.0) * 100.0
                            if other[REQUEST_METRIC]
                            else None
                        ),
                        "v6_slo_delta_pp": (
                            v6["offered_ttft_slo_attainment"]
                            - other["offered_ttft_slo_attainment"]
                        )
                        * 100.0,
                        "v6_p95_delta_s": v6["ttft_p95_s"] - other["ttft_p95_s"],
                        "v6_overlap_ratio": (
                            v6["mps_overlap_wall_s"] / other["mps_overlap_wall_s"]
                            if other["mps_overlap_wall_s"]
                            else None
                        ),
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in (("results", rows), ("v6_comparison", comparisons)):
        (args.output_dir / f"{filename}.json").write_text(
            json.dumps(values, indent=2, sort_keys=True) + "\n"
        )
        with (args.output_dir / f"{filename}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    print(json.dumps({"status": "PASS", "runs": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
