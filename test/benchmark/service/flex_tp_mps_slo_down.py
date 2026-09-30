#!/usr/bin/env python3
"""Measure MPS slowdown sensitivity by prompt length for paper tables."""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
import math
from pathlib import Path
import sys

try:
    with contextlib.redirect_stderr(io.StringIO()):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
except Exception:
    plt = None

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flex_tp_step_sim import FlexTPStepSimulator, SimulationConfig, WorkloadRequest, _quiet_production_loggers


def _values(value: str, *, positive: bool = True):
    values = [float(item) for item in value.split(",") if item.strip()]
    if not values or any(not math.isfinite(item) or (item <= 0 if positive else item < 0) for item in values):
        raise argparse.ArgumentTypeError("values must be finite and positive")
    return values


def _workload(length: int, rate: float, requests: int, seed: int) -> list[WorkloadRequest]:
    # Identical paired traces across policies and slowdown factors.
    del seed
    return [WorkloadRequest(i + 1, i / rate, length, 40, "mps-slo-down", length >= 4096) for i in range(requests)]


def _run(length: int, rate: float, slowdown: float, scheduler: str, args) -> dict:
    workload = _workload(length, rate, args.requests, args.seed)
    config = SimulationConfig(
        scheduler=scheduler,
        slo_ttft_s=args.slo_ttft,
        mps_overlap_slowdown=slowdown,
        fake_decode=True,
        kv_transfer_fixed_s=args.fake_kv_ms / 1000.0,
        v12_routing_cost_weight=args.v12_routing_cost_weight,
        v12_tp4_service_ratio_limit=args.v12_ratio,
        v13_latency_scale=args.v13_latency_scale,
        v13_routing_cost_weight=args.v13_routing_cost_weight,
        v13_tp4_service_ratio_limit=args.v13_ratio,
        v13_tp4_pressure_threshold=args.v13_pressure,
    )
    with FlexTPStepSimulator(workload, config) as simulator:
        summary = simulator.run()
    return {
        "input_tokens": length,
        "request_rate": rate,
        "mps_slowdown": slowdown,
        "scheduler": scheduler,
        "slo_ttft_s": args.slo_ttft,
        "ttft_p50_s": summary["ttft_p50_s"],
        "ttft_p99_s": summary["ttft_p99_s"],
        "offered_slo": summary["offered_ttft_slo_attainment"],
        "completion_throughput_rps": summary["completion_throughput_rps"],
        "prefill_token_throughput_s": summary["prefill_token_throughput_s"],
        "tp4_request_share": summary.get("tp4_request_share", 0.0),
        "mps_overlap_wall_s": summary["mps_overlap_wall_s"],
        "completed_requests": summary["completed_requests"],
    }


def _write_svg(rows, lengths, slowdowns, schedulers, metric, ylabel, path: Path) -> None:
    """Small dependency-free fallback chart for hosts with broken Matplotlib."""
    width, height = 220 * len(lengths), 340
    colors = {name: color for name, color in zip(schedulers, ("#2563eb", "#dc2626", "#059669", "#9333ea"))}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             f'<text x="10" y="18" font-size="14">{ylabel} vs MPS slowdown</text>']
    for panel, length in enumerate(lengths):
        x0, y0, pw, ph = panel * 220 + 30, 35, 175, 220
        parts.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="white" stroke="#999"/>')
        parts.append(f'<text x="{x0 + 4}" y="{y0 + 15}" font-size="11">input={int(length)}</text>')
        panel_rows = [r for r in rows if r["input_tokens"] == int(length) and r["request_rate"] == 4.0]
        all_values = [float(r[metric]) for r in panel_rows]
        if not all_values:
            continue
        lo, hi = min(all_values), max(all_values)
        span = max(hi - lo, 1e-9)
        parts.append(f'<line x1="{x0 + 10}" y1="{y0 + 10}" x2="{x0 + 10}" y2="{y0 + ph - 10}" stroke="#666"/>')
        parts.append(f'<line x1="{x0 + 10}" y1="{y0 + ph - 10}" x2="{x0 + pw - 10}" y2="{y0 + ph - 10}" stroke="#666"/>')
        parts.append(f'<text x="{x0 + 12}" y="{y0 + 27}" font-size="8">max={hi:.3g}</text>')
        parts.append(f'<text x="{x0 + 12}" y="{y0 + ph - 13}" font-size="8">min={lo:.3g}</text>')
        for scheduler in schedulers:
            points = [r for r in panel_rows if r["scheduler"] == scheduler]
            points.sort(key=lambda r: r["mps_slowdown"])
            if not points:
                continue
            values = [float(r[metric]) for r in points]
            coords = []
            for row, value in zip(points, values):
                x = x0 + 10 + (float(row["mps_slowdown"]) - min(slowdowns)) / max(max(slowdowns) - min(slowdowns), 1e-9) * (pw - 20)
                y = y0 + ph - 10 - (value - lo) / span * (ph - 35)
                coords.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{colors.get(scheduler, "#111")}" stroke-width="1.8"/>')
        parts.append(f'<text x="{x0 + 4}" y="{y0 + ph + 15}" font-size="9">slowdown {min(slowdowns):g}..{max(slowdowns):g}</text>')
    legend_y = height - 12
    legend_x = 10
    for scheduler in schedulers:
        color = colors.get(scheduler, "#111")
        parts.append(f'<line x1="{legend_x}" y1="{legend_y - 3}" x2="{legend_x + 16}" y2="{legend_y - 3}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{legend_y}" font-size="10">{scheduler}</text>')
        legend_x += 90
    parts.append('</svg>')
    path.write_text("\n".join(parts))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=lambda x: _values(x), default=[256, 1024, 4096, 8192, 16384])
    parser.add_argument("--slowdowns", type=lambda x: _values(x), default=[1, 1.25, 1.5, 2, 3])
    parser.add_argument("--rates", type=lambda x: _values(x), default=[4, 8])
    parser.add_argument("--schedulers", default="fixed_tp2,fixed_tp4,v12,v13")
    parser.add_argument("--requests", type=int, default=80)
    parser.add_argument("--slo-ttft", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fake-kv-ms", type=float, default=20.0)
    parser.add_argument("--v12-ratio", type=float, default=0.60)
    parser.add_argument("--v12-routing-cost-weight", type=float, default=2.0)
    parser.add_argument("--v13-ratio", type=float, default=0.60)
    parser.add_argument("--v13-pressure", type=float, default=1.0)
    parser.add_argument("--v13-latency-scale", type=float, default=1.0)
    parser.add_argument("--v13-routing-cost-weight", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("_/flex_tp_paper_analysis/260905-mps-slo-down"))
    args = parser.parse_args()
    if args.requests <= 0 or args.slo_ttft <= 0:
        parser.error("requests and slo-ttft must be positive")
    if not math.isfinite(args.v13_pressure) or not 0 <= args.v13_pressure <= 1:
        parser.error("v13-pressure must be in [0, 1]")
    if not math.isfinite(args.v13_latency_scale) or args.v13_latency_scale <= 0:
        parser.error("v13-latency-scale must be positive and finite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _quiet_production_loggers()
    schedulers = [item.strip() for item in args.schedulers.split(",") if item.strip()]
    rows = []
    for length in args.lengths:
        for rate in args.rates:
            for slowdown in args.slowdowns:
                for scheduler in schedulers:
                    rows.append(_run(int(length), rate, slowdown, scheduler, args))
    fields = list(rows[0]) if rows else []
    csv_path = args.output_dir / "mps_slo_down.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if plt is not None:
        for metric, ylabel in (("ttft_p99_s", "TTFT p99 (s)"), ("offered_slo", "SLO attainment")):
            figure, axes = plt.subplots(1, len(args.lengths), figsize=(4 * len(args.lengths), 3.5), sharey=True)
            axes = [axes] if len(args.lengths) == 1 else axes
            for axis, length in zip(axes, args.lengths):
                for scheduler in schedulers:
                    points = [r for r in rows if r["input_tokens"] == int(length) and r["scheduler"] == scheduler and r["request_rate"] == args.rates[0]]
                    points.sort(key=lambda r: r["mps_slowdown"])
                    axis.plot([r["mps_slowdown"] for r in points], [r[metric] for r in points], marker="o", label=scheduler)
                axis.set_title(f"input={int(length)}")
                axis.set_xlabel("MPS slowdown")
                axis.grid(alpha=0.25)
            axes[0].set_ylabel(ylabel)
            axes[0].legend(frameon=False, fontsize=8)
            figure.tight_layout()
            figure.savefig(args.output_dir / f"mps_slo_down_{metric}.png", dpi=160)
            plt.close(figure)
    else:
        _write_svg(rows, args.lengths, args.slowdowns, schedulers, "ttft_p99_s", "TTFT p99 (s)", args.output_dir / "mps_slo_down_ttft_p99_s.svg")
        _write_svg(rows, args.lengths, args.slowdowns, schedulers, "offered_slo", "SLO attainment", args.output_dir / "mps_slo_down_offered_slo.svg")
    print(f"MPS_SLO_DOWN_OK rows={len(rows)} csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
