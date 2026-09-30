#!/usr/bin/env python3
"""Sweep request rate and target TTFT for every FlexTP scheduler.

The workload trace is generated once per ``(dataset, rate)`` and reused for
all target TTFT values and policies.  This keeps the sweep paired while still
executing the production selectors and the production ChunkedPrefillQueue
through ``flex_tp_step_sim``.  Independent dataset/rate/target shards can be
executed in parallel with ``--jobs``; the parent process owns final ordering
and artifact writing.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import hashlib
import io
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

plt = None
_PLOT_IMPORT_ERROR = None


def _load_matplotlib():
    """Load plotting support only in the parent after worker processes exit."""
    global plt, _PLOT_IMPORT_ERROR
    if plt is not None or _PLOT_IMPORT_ERROR is not None:
        return plt
    try:  # pragma: no cover - depends on the host image
        # Some environments ship a NumPy-incompatible matplotlib extension;
        # keep its optional import diagnostics out of benchmark logs.
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as pyplot

        plt = pyplot
    except Exception as exc:  # pragma: no cover - depends on the host image
        _PLOT_IMPORT_ERROR = repr(exc)
    return plt

from flex_tp_step_sim import (
    FlexTPStepSimulator,
    SimulationConfig,
    WorkloadRequest,
    _quiet_production_loggers,
    _repo_root,
    generate_servegen_mm_image,
    generate_servegen_dataset,
    generate_synthetic_5pct,
    generate_mooncake_workload,
)


SCHEDULERS = (
    "fixed_tp2",
    "fixed_tp4",
    "naive",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
    "v8",
    "v9",
    "v10",
    "v11",
    "v12",
    "v13",
    "naive_switch",
)
DATASETS = ("servegen-mm-image", "servegen-lang-large", "servegen-deepseek-r1", "mooncake-trace", "synthetic-5pct")
DEFAULT_TARGET_TTFT = (0.5, 1.0, 2.0, 4.0)
DEFAULT_RATES = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0)


def _parse_float_list(value: str, *, name: str, minimum: float = 0.0) -> List[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(not math.isfinite(item) or item < minimum for item in values):
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated finite values >= {minimum:g}")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} must not contain duplicates")
    return values


def _parse_datasets(value: str) -> List[str]:
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    if not datasets or any(item not in DATASETS for item in datasets):
        raise argparse.ArgumentTypeError(f"datasets must be a subset of {','.join(DATASETS)}")
    if len(set(datasets)) != len(datasets):
        raise argparse.ArgumentTypeError("datasets must not contain duplicates")
    return datasets


def _parse_schedulers(value: str) -> List[str]:
    schedulers = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not schedulers or any(item not in SCHEDULERS for item in schedulers):
        raise argparse.ArgumentTypeError(f"schedulers must be a subset of {','.join(SCHEDULERS)}")
    if len(set(schedulers)) != len(schedulers):
        raise argparse.ArgumentTypeError("schedulers must not contain duplicates")
    return schedulers


def _fingerprint(workload: Sequence[WorkloadRequest]) -> str:
    payload = json.dumps(
        [
            (round(float(item.arrival_s), 9), int(item.input_tokens), int(item.output_tokens))
            for item in workload
        ],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _make_workload(dataset: str, rate: float, args: argparse.Namespace) -> List[WorkloadRequest]:
    if dataset == "servegen-mm-image":
        workload = generate_servegen_mm_image(
            repo_root=_repo_root(),
            request_rate=rate,
            duration_s=args.duration,
            seed=args.seed,
            output_divisor=args.servegen_output_divisor,
        )
    elif dataset == "servegen-lang-large":
        workload = generate_servegen_dataset(
            repo_root=_repo_root(), model="m-large", category="language",
            request_rate=rate, duration_s=args.duration, seed=args.seed,
            output_divisor=args.servegen_output_divisor,
        )
    elif dataset == "servegen-deepseek-r1":
        workload = generate_servegen_dataset(
            repo_root=_repo_root(), model="deepseek-r1", category="reason",
            request_rate=rate, duration_s=args.duration, seed=args.seed,
            output_divisor=args.servegen_output_divisor,
        )
    elif dataset == "mooncake-trace":
        workload = generate_mooncake_workload(
            dataset_path=args.mooncake_dataset,
            request_rate=rate,
            num_prompts=args.synthetic_prompts,
            seed=args.seed,
            output_divisor=args.mooncake_output_divisor,
        )
    else:
        workload = generate_synthetic_5pct(
            request_rate=rate,
            num_prompts=args.synthetic_prompts,
            seed=args.seed,
        )
    if not workload:
        raise RuntimeError(f"{dataset} at rate={rate:g} generated no requests")
    return workload


def _config(args: argparse.Namespace, scheduler: str, target_ttft: float) -> SimulationConfig:
    return SimulationConfig(
        scheduler=scheduler,
        naive_length_threshold=args.long_threshold,
        mps_overlap_slowdown=args.mps_slowdown,
        slo_ttft_s=target_ttft,
        bundle_window_s=args.bundle_window_ms / 1000.0,
        bundle_token_cap=args.bundle_token_cap,
        bundle_token_trigger=args.bundle_token_trigger,
        max_inflight=args.max_inflight,
        instance_token_credit=args.instance_token_credit,
        prediction_margin_s=args.prediction_margin_ms / 1000.0,
        schedule_interval_s=args.schedule_interval_ms / 1000.0,
        chunked_prefill_size=args.chunked_prefill_size,
        prefill_batch_max_tokens=args.batch_max_tokens,
        max_total_tokens=args.max_total_tokens,
        fake_decode=True,
        simulate_decode=False,
        kv_transfer_fixed_s=args.fake_kv_fixed_ms / 1000.0,
        kv_transfer_us_per_token=args.fake_kv_us_per_token,
        v6_request_utility_tokens=args.v6_request_utility_tokens,
        v7_class_quantum_tokens=args.v7_class_quantum_tokens,
        v7_conflict_price_weight=args.v7_conflict_price_weight,
        v7_urgency_weight=args.v7_urgency_weight,
        v8_epoch_ms=args.v8_epoch_ms,
        v8_epoch_token_budget=args.v8_epoch_token_budget,
        v8_min_lane_quota=args.v8_min_lane_quota,
        v8_deadline_pressure_weight=args.v8_deadline_pressure_weight,
        v9_aging_interval_ms=args.v9_aging_interval_ms,
        v9_interactive_tokens=args.v9_interactive_tokens,
        v9_short_weight=args.v9_short_weight,
        v9_long_weight=args.v9_long_weight,
        v10_short_weight=args.v10_short_weight,
        v10_long_weight=args.v10_long_weight,
        v10_slack_weight=args.v10_slack_weight,
        v10_overlap_slack_ratio=args.v10_overlap_slack_ratio,
        v11_short_weight=args.v11_short_weight,
        v11_long_weight=args.v11_long_weight,
        v11_slack_weight=args.v11_slack_weight,
        v11_overlap_slack_ratio=args.v11_overlap_slack_ratio,
        v11_routing_cost_weight=args.v11_routing_cost_weight,
        v12_routing_cost_weight=args.v12_routing_cost_weight,
        v12_tp4_service_ratio_limit=args.v12_tp4_service_ratio_limit,
        v13_latency_scale=args.v13_latency_scale,
        v13_long_threshold=args.v13_long_threshold,
        v13_routing_cost_weight=args.v13_routing_cost_weight,
        v13_tp4_service_ratio_limit=args.v13_tp4_service_ratio_limit,
        v13_tp4_pressure_threshold=args.v13_tp4_pressure_threshold,
    )


def _row(
    dataset: str,
    rate: float,
    target_ttft: float,
    seed: int,
    fingerprint: str,
    summary: Dict,
) -> Dict:
    return {
        "dataset": dataset,
        "request_rate": rate,
        "target_ttft_s": target_ttft,
        "seed": seed,
        "trace_fingerprint": fingerprint,
        "scheduler": summary["scheduler"],
        "offered_requests": summary["offered_requests"],
        "completed_requests": summary["completed_requests"],
        "completion_fraction": summary["completion_fraction"],
        "rejected_requests": summary["rejected_requests"],
        "ttft_p50_s": summary["ttft_p50_s"],
        "ttft_p99_s": summary["ttft_p99_s"],
        "ttft_max_s": summary["ttft_max_s"],
        "offered_ttft_slo_attainment": summary["offered_ttft_slo_attainment"],
        "on_time_goodput_rps": summary["on_time_goodput_rps"],
        "on_time_input_token_goodput_s": summary["on_time_input_token_goodput_s"],
        "completion_throughput_rps": summary["completion_throughput_rps"],
        "prefill_token_throughput_s": summary["prefill_token_throughput_s"],
        "prefill_step_mean_batch": summary["prefill_step_mean_batch"],
        "bundle_mean_requests": summary["bundle_mean_requests"],
        "mps_overlap_wall_s": summary["mps_overlap_wall_s"],
        "tp2_long_request_count": summary["tp2_long_request_count"],
        "fake_decode_transfer_count": summary["fake_decode_transfer_count"],
    }


def _run_target_shard(
    dataset: str,
    rate: float,
    target_ttft: float,
    args: argparse.Namespace,
) -> tuple[str, float, float, str, Dict, List[Dict]]:
    """Run all schedulers for one paired dataset/rate/target shard."""
    _quiet_production_loggers()
    workload = _make_workload(dataset, rate, args)
    fingerprint = _fingerprint(workload)
    trace_details = {
        "fingerprint": fingerprint,
        "offered_requests": len(workload),
        "offered_duration_s": max(item.arrival_s for item in workload)
        - min(item.arrival_s for item in workload),
        "long_request_count": sum(item.input_tokens > args.long_threshold for item in workload),
    }
    rows: List[Dict] = []
    for scheduler in args.schedulers:
        config = _config(args, scheduler, target_ttft)
        with FlexTPStepSimulator(workload, config) as simulator:
            summary = simulator.run()
        rows.append(_row(dataset, rate, target_ttft, args.seed, fingerprint, summary))
    return dataset, rate, target_ttft, fingerprint, trace_details, rows


def _write_curves(
    rows: Sequence[Dict],
    output_dir: Path,
    datasets: Iterable[str],
    targets: Sequence[float],
    *,
    yscale: str = "linear",
    suffix: str = "",
) -> None:
    _load_matplotlib()
    if plt is None:
        return
    palette = tuple(plt.cm.tab20.colors)
    colors = {
        scheduler: palette[index % len(palette)]
        for index, scheduler in enumerate(SCHEDULERS)
    }
    for dataset in datasets:
        for metric, title, filename in (
            ("ttft_p50_s", "TTFT P50", "ttft_p50"),
            ("ttft_p99_s", "TTFT P99", "ttft_p99"),
        ):
            figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
            axes = axes.ravel()
            for axis, target in zip(axes, targets):
                selected = [
                    row
                    for row in rows
                    if row["dataset"] == dataset and row["target_ttft_s"] == target
                ]
                for scheduler in SCHEDULERS:
                    points = sorted(
                        (row for row in selected if row["scheduler"] == scheduler),
                        key=lambda row: row["request_rate"],
                    )
                    if not points:
                        continue
                    axis.plot(
                        [row["request_rate"] for row in points],
                        [row[metric] if row[metric] is not None else float("nan") for row in points],
                        marker="o",
                        linewidth=1.5,
                        markersize=3.5,
                        label=scheduler,
                        color=colors[scheduler],
                    )
                axis.axhline(target, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
                axis.set_title(f"target TTFT = {target:g}s")
                axis.set_xlabel("request rate (req/s)")
                axis.set_ylabel("TTFT (s)")
                axis.set_yscale(yscale)
                axis.grid(True, alpha=0.25)
            handles, labels = axes[0].get_legend_handles_labels()
            figure.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.955),
                ncol=5,
                frameon=False,
            )
            figure.suptitle(f"FlexTP {dataset}: {title} vs request rate", y=0.995)
            figure.tight_layout(rect=(0, 0, 1, 0.91))
            figure.savefig(output_dir / f"{dataset}.{filename}{suffix}.png", dpi=160)
            plt.close(figure)


def _write_report_json(
    output_dir: Path,
    args: argparse.Namespace,
    rows: Sequence[Dict],
    traces: Dict[str, Dict[float, Dict]],
) -> None:
    output = {
        "date_marker": "260905",
        "schedulers": list(args.schedulers),
        "datasets": list(args.datasets),
        "target_ttft_s": list(args.target_ttft),
        "request_rates": list(args.rates),
        "config": {
            "duration_s": args.duration,
            "synthetic_prompts": args.synthetic_prompts,
            "seed": args.seed,
            "mps_overlap_slowdown": args.mps_slowdown,
            "fake_kv_fixed_ms": args.fake_kv_fixed_ms,
            "fake_kv_us_per_token": args.fake_kv_us_per_token,
            "long_threshold": args.long_threshold,
            "bundle_window_ms": args.bundle_window_ms,
            "bundle_token_cap": args.bundle_token_cap,
            "bundle_token_trigger": args.bundle_token_trigger,
            "batch_max_tokens": args.batch_max_tokens,
            "chunked_prefill_size": args.chunked_prefill_size,
            "v10_short_weight": args.v10_short_weight,
            "v10_long_weight": args.v10_long_weight,
            "v10_slack_weight": args.v10_slack_weight,
            "v10_overlap_slack_ratio": args.v10_overlap_slack_ratio,
            "v11_short_weight": args.v11_short_weight,
            "v11_long_weight": args.v11_long_weight,
            "v11_slack_weight": args.v11_slack_weight,
            "v11_overlap_slack_ratio": args.v11_overlap_slack_ratio,
            "v11_routing_cost_weight": args.v11_routing_cost_weight,
            "v12_routing_cost_weight": args.v12_routing_cost_weight,
            "v12_tp4_service_ratio_limit": args.v12_tp4_service_ratio_limit,
            "v13_latency_scale": args.v13_latency_scale,
            "v13_long_threshold": args.v13_long_threshold,
            "v13_routing_cost_weight": args.v13_routing_cost_weight,
            "v13_tp4_service_ratio_limit": args.v13_tp4_service_ratio_limit,
            "v13_tp4_pressure_threshold": args.v13_tp4_pressure_threshold,
            "servegen_output_divisor": args.servegen_output_divisor,
            "mooncake_dataset": str(args.mooncake_dataset),
            "mooncake_output_divisor": args.mooncake_output_divisor,
        },
        "trace_fingerprints": {
            dataset: {str(rate): details for rate, details in rates.items()}
            for dataset, rates in traces.items()
        },
        "rows": list(rows),
    }
    (output_dir / "rate_ttft_sweep.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", type=_parse_datasets, default=list(DATASETS))
    parser.add_argument("--schedulers", type=_parse_schedulers, default=list(SCHEDULERS))
    parser.add_argument("--target-ttft", type=lambda value: _parse_float_list(value, name="target-ttft", minimum=0.001), default=list(DEFAULT_TARGET_TTFT))
    parser.add_argument("--rates", type=lambda value: _parse_float_list(value, name="rates", minimum=0.001), default=list(DEFAULT_RATES))
    parser.add_argument("--duration", type=int, default=60, help="ServeGen duration in seconds")
    parser.add_argument("--servegen-output-divisor", type=int, default=10)
    parser.add_argument("--mooncake-dataset", type=Path, default=_repo_root() / "mooncake_trace.jsonl")
    parser.add_argument("--mooncake-output-divisor", type=int, default=10)
    parser.add_argument("--synthetic-prompts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="number of independent dataset/rate/target worker processes (default capped at 8)",
    )
    parser.add_argument("--long-threshold", type=int, default=4000)
    parser.add_argument("--mps-slowdown", type=float, default=2.0)
    parser.add_argument("--fake-kv-fixed-ms", type=float, default=20.0)
    parser.add_argument("--fake-kv-us-per-token", type=float, default=0.0)
    parser.add_argument("--bundle-window-ms", type=float, default=20.0)
    parser.add_argument("--bundle-token-cap", type=int, default=8192)
    parser.add_argument("--bundle-token-trigger", type=int, default=4096)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--instance-token-credit", type=int, default=16384)
    parser.add_argument("--prediction-margin-ms", type=float, default=80.0)
    parser.add_argument("--schedule-interval-ms", type=float, default=5.0)
    parser.add_argument("--chunked-prefill-size", type=int, default=8192)
    parser.add_argument("--batch-max-tokens", type=int, default=16384)
    parser.add_argument("--max-total-tokens", type=int, default=70000)
    parser.add_argument("--v6-request-utility-tokens", type=float, default=2000.0)
    parser.add_argument("--v7-class-quantum-tokens", type=int, default=4096)
    parser.add_argument("--v7-conflict-price-weight", type=float, default=0.35)
    parser.add_argument("--v7-urgency-weight", type=float, default=2.0)
    parser.add_argument("--v8-epoch-ms", type=float, default=50.0)
    parser.add_argument("--v8-epoch-token-budget", type=int, default=16384)
    parser.add_argument("--v8-min-lane-quota", type=int, default=2048)
    parser.add_argument("--v8-deadline-pressure-weight", type=float, default=2.0)
    parser.add_argument("--v9-aging-interval-ms", type=float, default=150.0)
    parser.add_argument("--v9-interactive-tokens", type=int, default=1024)
    parser.add_argument("--v9-short-weight", type=float, default=1.0)
    parser.add_argument("--v9-long-weight", type=float, default=1.0)
    parser.add_argument("--v10-short-weight", type=float, default=1.0)
    parser.add_argument("--v10-long-weight", type=float, default=1.0)
    parser.add_argument("--v10-slack-weight", type=float, default=1.0)
    parser.add_argument("--v10-overlap-slack-ratio", type=float, default=0.10)
    parser.add_argument("--v11-short-weight", type=float, default=1.0)
    parser.add_argument("--v11-long-weight", type=float, default=1.0)
    parser.add_argument("--v11-slack-weight", type=float, default=1.0)
    parser.add_argument("--v11-overlap-slack-ratio", type=float, default=0.10)
    parser.add_argument("--v11-routing-cost-weight", type=float, default=0.50)
    parser.add_argument("--v12-routing-cost-weight", type=float, default=2.0)
    parser.add_argument("--v12-tp4-service-ratio-limit", type=float, default=0.60)
    parser.add_argument("--v13-latency-scale", type=float, default=1.0)
    parser.add_argument("--v13-long-threshold", type=int, default=12000)
    parser.add_argument("--v13-routing-cost-weight", type=float, default=2.0)
    parser.add_argument("--v13-tp4-service-ratio-limit", type=float, default=0.60)
    parser.add_argument("--v13-tp4-pressure-threshold", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("_/flex_tp_paper_analysis/rate-ttft-sweep-fixed-baselines-260905"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration <= 0 or args.synthetic_prompts <= 0:
        raise SystemExit("duration and synthetic-prompts must be positive")
    if args.jobs <= 0:
        raise SystemExit("jobs must be positive")
    if args.long_threshold <= 0 or args.mps_slowdown < 1.0:
        raise SystemExit("long-threshold must be positive and mps-slowdown must be >= 1")
    if args.servegen_output_divisor < 1 or args.mooncake_output_divisor < 1:
        raise SystemExit("output divisors must be >= 1")
    if not math.isfinite(args.v13_latency_scale) or args.v13_latency_scale <= 0:
        raise SystemExit("v13-latency-scale must be positive and finite")
    if not math.isfinite(args.v13_routing_cost_weight) or args.v13_routing_cost_weight < 0:
        raise SystemExit("v13-routing-cost-weight must be finite and non-negative")
    if not math.isfinite(args.v13_tp4_service_ratio_limit) or not 0 <= args.v13_tp4_service_ratio_limit <= 1:
        raise SystemExit("v13-tp4-service-ratio-limit must be in [0, 1]")
    if not math.isfinite(args.v13_tp4_pressure_threshold) or not 0 <= args.v13_tp4_pressure_threshold <= 1:
        raise SystemExit("v13-tp4-pressure-threshold must be in [0, 1]")
    if args.v13_long_threshold <= 0:
        raise SystemExit("v13-long-threshold must be positive")
    if "mooncake-trace" in args.datasets and not args.mooncake_dataset.is_file():
        raise SystemExit(f"mooncake dataset not found: {args.mooncake_dataset}")
    _quiet_production_loggers()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    traces: Dict[str, Dict[float, Dict]] = {}
    shard_count = len(args.datasets) * len(args.rates) * len(args.target_ttft)
    runs_per_shard = len(args.schedulers)
    total_runs = shard_count * runs_per_shard
    completed_runs = 0
    started = time.perf_counter()
    print(
        f"sweep: datasets={','.join(args.datasets)} schedulers={','.join(args.schedulers)} "
        f"targets={','.join(f'{value:g}' for value in args.target_ttft)} "
        f"rates={','.join(f'{value:g}' for value in args.rates)} runs={total_runs} jobs={args.jobs}",
        flush=True,
    )

    for dataset in args.datasets:
        traces[dataset] = {}
        for rate in args.rates:
            workload = _make_workload(dataset, rate, args)
            fingerprint = _fingerprint(workload)
            traces[dataset][rate] = {
                "fingerprint": fingerprint,
                "offered_requests": len(workload),
                "offered_duration_s": max(item.arrival_s for item in workload)
                - min(item.arrival_s for item in workload),
                "long_request_count": sum(item.input_tokens > args.long_threshold for item in workload),
            }
    shards = [
        (dataset, rate, target)
        for dataset in args.datasets
        for rate in args.rates
        for target in args.target_ttft
    ]
    # CUDA/NIXL imports may leave background threads alive.  Forking after
    # those imports can deadlock before the first shard is submitted; spawn
    # gives each worker a clean interpreter and keeps failed sweeps bounded by
    # the caller's outer timeout.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as executor:
        futures = {
            executor.submit(
                _run_target_shard,
                dataset,
                rate,
                target,
                args,
            ): (dataset, rate, target)
            for dataset, rate, target in shards
        }
        for future in as_completed(futures):
            dataset, rate, target = futures[future]
            shard_dataset, shard_rate, shard_target, fingerprint, trace_details, shard_rows = future.result()
            if (shard_dataset, shard_rate, shard_target) != (dataset, rate, target):
                raise RuntimeError("worker returned a mismatched sweep shard")
            rows.extend(shard_rows)
            completed_runs += len(shard_rows)
            print(
                f"[{completed_runs}/{total_runs}] shard={dataset} rate={rate:g} target={target:g} "
                f"rows={len(shard_rows)} workers={args.jobs}",
                flush=True,
            )

    scheduler_order = {name: index for index, name in enumerate(args.schedulers)}
    dataset_order = {name: index for index, name in enumerate(args.datasets)}
    rate_order = {value: index for index, value in enumerate(args.rates)}
    target_order = {value: index for index, value in enumerate(args.target_ttft)}
    rows.sort(
        key=lambda row: (
            dataset_order[row["dataset"]],
            rate_order[row["request_rate"]],
            target_order[row["target_ttft_s"]],
            scheduler_order[row["scheduler"]],
        )
    )

    fieldnames = list(rows[0])
    with (args.output_dir / "rate_ttft_sweep.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_report_json(args.output_dir, args, rows, traces)
    _write_curves(rows, args.output_dir, args.datasets, args.target_ttft)
    _write_curves(
        rows,
        args.output_dir,
        args.datasets,
        args.target_ttft,
        yscale="log",
        suffix=".log",
    )
    print(
        f"RATE_TTFT_SWEEP_OK runs={len(rows)} elapsed_s={time.perf_counter() - started:.1f} "
        f"output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
