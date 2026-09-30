#!/usr/bin/env python3
"""汇总多个真实 GPU benchmark 目录，并绘制 TTFT p99 对照图。"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def percentile(values, p):
    values = sorted(values)
    if not values:
        return float("nan")
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def collect(case_name: str, root: Path):
    rows = []
    for path in sorted(root.rglob("bench_*.json")):
        payload = json.loads(path.read_text())
        records = payload.get("results", [])
        first = [float(item["token_latencys"][0]) for item in records if item.get("token_latencys")]
        lat = [float(item["latency"]) for item in records if item.get("latency") is not None]
        params = payload.get("args", {})
        mode = params.get("servegen_mode", path.parent.name)
        duration = max(float(params.get("servegen_duration", 120)), 1.0)
        rows.append(
            {
                "source": str(path),
                "case": case_name,
                "dataset": f"servegen-{mode}",
                "request_rate": float(params.get("request_rate", "nan")),
                "offered_requests": len(records),
                "completed_requests": len(records),
                "failed_requests": int(payload.get("failed_requests", 0) or 0),
                "ttft_p50_s": percentile(first, 0.50),
                "ttft_p99_s": percentile(first, 0.99),
                "latency_p99_s": percentile(lat, 0.99),
                "throughput_rps_nominal": len(records) / duration,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs=2, action="append", metavar=("NAME", "ROOT"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for name, root in args.case:
        rows.extend(collect(name, Path(root)))
    rows.sort(key=lambda r: (r["dataset"], r["request_rate"], r["case"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "source",
        "case",
        "dataset",
        "request_rate",
        "offered_requests",
        "completed_requests",
        "failed_requests",
        "ttft_p50_s",
        "ttft_p99_s",
        "latency_p99_s",
        "throughput_rps_nominal",
    ]
    with (args.output_dir / "gpu_rate_ttft_compare.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for dataset in sorted({row["dataset"] for row in rows}):
            plt.figure(figsize=(7, 4.5))
            for case in sorted({row["case"] for row in rows if row["dataset"] == dataset}):
                selected = sorted(
                    (row for row in rows if row["dataset"] == dataset and row["case"] == case),
                    key=lambda row: row["request_rate"],
                )
                if selected:
                    plt.plot(
                        [r["request_rate"] for r in selected],
                        [r["ttft_p99_s"] for r in selected],
                        marker="o",
                        label=case,
                    )
            plt.xlabel("offered request rate (req/s)")
            plt.ylabel("TTFT p99 (s)")
            plt.title(dataset + " (real GPU)")
            plt.grid(alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.output_dir / (dataset + ".ttft_p99.compare.png"), dpi=180)
            plt.close()
    except Exception as exc:
        (args.output_dir / "plot_error.txt").write_text(repr(exc) + "\n")

    print(json.dumps({"json_dumps": len(rows), "output": str(args.output_dir / "gpu_rate_ttft_compare.csv")}, indent=2))


if __name__ == "__main__":
    main()
