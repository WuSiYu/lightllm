#!/usr/bin/env python3
"""Regression matrix for the V4/V5 scheduler objectives.

The gate compares production V4/V5 selector classes with the production
naive-4K selector through ``flex_tp_step_sim``.  Decode is intentionally
non-bottlenecking.  Model-based passes are necessary integration evidence, not
a substitute for H200 measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


POLICIES = ("v4", "v5", "naive")
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


def run_matrix(args) -> Tuple[List[Dict], List[Dict], int]:
    rows: List[Dict] = []
    failures: List[Dict] = []
    check_count = 0
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
                rows.append(
                    {
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
                        "scheduler_stats": json.dumps(summary["scheduler_stats"], sort_keys=True),
                    }
                )

            naive = summaries["naive"]
            for policy in ("v4", "v5"):
                current = summaries[policy]
                checks = {
                    "no_long_on_tp2": (
                        current["tp2_long_request_count"] == 0,
                        current["tp2_long_request_count"],
                        0,
                    ),
                    "slo_not_below_naive": (
                        current["offered_ttft_slo_attainment"] + 1e-12
                        >= naive["offered_ttft_slo_attainment"],
                        current["offered_ttft_slo_attainment"],
                        naive["offered_ttft_slo_attainment"],
                    ),
                    "token_goodput_above_naive": (
                        current[TOKEN_METRIC]
                        > naive[TOKEN_METRIC] * (1.0 + args.min_token_gain),
                        current[TOKEN_METRIC],
                        naive[TOKEN_METRIC],
                    ),
                }
                if policy == "v5" and naive["mps_overlap_wall_s"] > 0:
                    checks["mps_is_default_mode"] = (
                        current["mps_overlap_wall_s"]
                        >= naive["mps_overlap_wall_s"] * args.min_v5_overlap_ratio,
                        current["mps_overlap_wall_s"],
                        naive["mps_overlap_wall_s"] * args.min_v5_overlap_ratio,
                    )
                check_count += len(checks)
                for check, (passed, dynamic_value, reference_value) in checks.items():
                    if not passed:
                        failures.append(
                            {
                                "scenario": scenario,
                                "slowdown": slowdown,
                                "policy": policy,
                                "check": check,
                                "dynamic_value": dynamic_value,
                                "reference_value": reference_value,
                            }
                        )
    return rows, failures, check_count


def parse_slowdowns(value: str) -> List[float]:
    result = [float(item) for item in value.split(",")]
    if not result or any(item < 1.0 for item in result):
        raise argparse.ArgumentTypeError("slowdowns must be comma-separated values >= 1")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slowdowns", type=parse_slowdowns, default=[1.6, 2.0])
    parser.add_argument("--synthetic-prompts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-token-gain", type=float, default=0.001)
    parser.add_argument("--min-v5-overlap-ratio", type=float, default=0.75)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_/flex_tp_paper_analysis/v45-regression"),
    )
    args = parser.parse_args()
    _quiet_production_loggers()
    rows, failures, check_count = run_matrix(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['scenario']} slowdown={row['slowdown']:g} policy={row['policy']} "
            f"SLO={row['offered_slo']:.4f} "
            f"token_goodput={row['offered_window_token_goodput']:.1f}/s "
            f"overlap={row['mps_overlap_wall_s']:.1f}s"
        )
    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2))
        return 1
    print(json.dumps({"status": "PASS", "checks": check_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
