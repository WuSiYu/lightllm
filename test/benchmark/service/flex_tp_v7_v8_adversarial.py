#!/usr/bin/env python3
"""Adversarial and boundary traces for the FlexTP V7-V11 schedulers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from flex_tp_step_sim import FlexTPStepSimulator, SimulationConfig, WorkloadRequest, _quiet_production_loggers


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
    "v10",
    "v11",
    "naive",
)


def _request(request_id: int, arrival: float, length: int, source: str) -> WorkloadRequest:
    return WorkloadRequest(
        request_id=request_id,
        arrival_s=arrival,
        input_tokens=length,
        output_tokens=8,
        source=source,
        is_long=length > 4000,
    )


def traces() -> Dict[str, List[WorkloadRequest]]:
    boundary: List[WorkloadRequest] = []
    request_id = 1
    for round_id in range(12):
        for length in (3999, 4000, 4001, 8000):
            boundary.append(_request(request_id, round_id * 0.04, length, "threshold-edge"))
            request_id += 1

    crossfire = [_request(1, 0.0, 20000, "long-then-short")]
    crossfire.extend(
        _request(index + 2, 0.01 + index * 0.01, 256, "long-then-short")
        for index in range(40)
    )

    fair_mix: List[WorkloadRequest] = []
    request_id = 1
    for index in range(160):
        fair_mix.append(_request(request_id, index * 0.025, 512, "dual-lane-fair"))
        request_id += 1
    for index in range(40):
        fair_mix.append(_request(request_id, index * 0.1, 8000, "dual-lane-fair"))
        request_id += 1

    return {
        name: sorted(workload, key=lambda item: (item.arrival_s, item.request_id))
        for name, workload in (
            ("threshold-edge", boundary),
            ("long-then-short", crossfire),
            ("dual-lane-fair", fair_mix),
        )
    }


def parse_slowdowns(value: str) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(value < 1.0 for value in values):
        raise argparse.ArgumentTypeError("slowdowns must be comma-separated values >= 1")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slowdowns", type=parse_slowdowns, default=[1.6, 2.0, 2.5])
    parser.add_argument(
        "--fake-decode",
        action="store_true",
        help="retain the fixed/per-token KV transfer event while skipping Decode compute",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("_/flex_tp_paper_analysis/v7-v9-adversarial"))
    args = parser.parse_args()
    _quiet_production_loggers()
    rows: List[Dict] = []
    for scenario, workload in traces().items():
        for slowdown in args.slowdowns:
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
                if policy in {"v4", "v5", "v6", "v7", "v8", "v9", "v10"} and summary["tp2_long_request_count"]:
                    raise AssertionError(f"{policy} sent a long request to TP2 in {scenario}")
                if policy in {"v7", "v8", "v9", "v10", "v11"} and summary["completion_fraction"] < 1.0:
                    raise AssertionError(f"{policy} did not complete the best-effort trace in {scenario}")
                row = {
                    "scenario": scenario,
                    "slowdown": slowdown,
                    "policy": policy,
                    "offered_slo": summary["offered_ttft_slo_attainment"],
                    "token_goodput": summary["offered_window_on_time_input_token_goodput_s"],
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
                rows.append(row)
                print(
                    f"{scenario} slowdown={slowdown:g} policy={policy} "
                    f"SLO={row['offered_slo']:.4f} token_goodput={row['token_goodput']:.1f}/s "
                    f"p95={row['p95_ttft_s']:.3f}s"
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    with (args.output_dir / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS", "runs": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
