#!/usr/bin/env python3
"""Compare production FlexTP V3-V9 and naive on common TP-SMT workloads."""

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


POLICIES = (
    "fixed_tp2",
    "fixed_tp4",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
    "naive",
)
CANDIDATES = ("v6", "v7", "v8", "v9")
BASELINES = ("fixed_tp2", "fixed_tp4", "v3", "v4", "v5", "naive")
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
        "--fake-decode",
        action="store_true",
        help="retain the fixed/per-token KV transfer event while skipping Decode compute",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_/flex_tp_paper_analysis/v7-v9-comparison"),
    )
    args = parser.parse_args()
    _quiet_production_loggers()

    rows: List[Dict] = []
    comparisons: List[Dict] = []
    for scenario, workload in workloads(args.seed, args.synthetic_prompts).items():
        for slowdown in args.slowdowns:
            summaries: Dict[str, Dict] = {}
            for policy in POLICIES:
                config = SimulationConfig(
                    scheduler=policy,
                    naive_length_threshold=4000,
                    mps_overlap_slowdown=slowdown,
                    slo_ttft_s=3.0,
                    simulate_decode=False,
                    fake_decode=args.fake_decode,
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
                        "prefill_step_mean_batch": summary["prefill_step_mean_batch"],
                        "decode_mode": summary["timing_model"]["decode"],
                        "decode_step_count": summary["decode_step_count"],
                        "fake_decode_transfer_count": summary["fake_decode_transfer_count"],
                        "kv_transfer_wait_p95_s": summary["kv_transfer_wait_p95_s"],
                        "scheduler_stats": json.dumps(summary["scheduler_stats"], sort_keys=True),
                    }
                )
                print(
                    f"{scenario} slowdown={slowdown:g} policy={policy} "
                    f"SLO={summary['offered_ttft_slo_attainment']:.4f} "
                    f"token_goodput={summary[TOKEN_METRIC]:.1f}/s "
                    f"p95={summary['ttft_p95_s']:.3f}s "
                    f"overlap={summary['mps_overlap_wall_s']:.1f}s"
                )

            for candidate in CANDIDATES:
                if summaries[candidate]["tp2_long_request_count"] != 0:
                    raise AssertionError(
                        f"{candidate.upper()} routed long work to TP2 in {scenario} slowdown={slowdown}"
                    )
                for baseline in BASELINES:
                    left = summaries[candidate]
                    right = summaries[baseline]
                    comparisons.append(
                        {
                            "scenario": scenario,
                            "slowdown": slowdown,
                            "candidate": candidate,
                            "baseline": baseline,
                            "candidate_token_gain_pct": (
                                (left[TOKEN_METRIC] / right[TOKEN_METRIC] - 1.0) * 100.0
                                if right[TOKEN_METRIC]
                                else None
                            ),
                            "candidate_request_gain_pct": (
                                (left[REQUEST_METRIC] / right[REQUEST_METRIC] - 1.0) * 100.0
                                if right[REQUEST_METRIC]
                                else None
                            ),
                            "candidate_slo_delta_pp": (
                                left["offered_ttft_slo_attainment"]
                                - right["offered_ttft_slo_attainment"]
                            )
                            * 100.0,
                            "candidate_p95_delta_s": left["ttft_p95_s"] - right["ttft_p95_s"],
                            "candidate_completion_delta_pp": (
                                left["completion_fraction"] - right["completion_fraction"]
                            )
                            * 100.0,
                        }
                    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in (("results", rows), ("candidate_comparison", comparisons)):
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
