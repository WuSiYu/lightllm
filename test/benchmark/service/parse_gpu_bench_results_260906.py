#!/usr/bin/env python3
"""Convert real benchmark_serving_chat_req_rate JSON dumps to CSV/plots."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("servegen/*/bench_*.json")):
        payload = json.loads(path.read_text())
        records = payload.get("results", [])
        first = [float(item["token_latencys"][0]) for item in records if item.get("token_latencys")]
        lat = [float(item["latency"]) for item in records if item.get("latency") is not None]
        params = payload.get("args", {})
        mode = params.get("servegen_mode", path.parent.name)
        rows.append({
            "source": str(path),
            "dataset": f"servegen-{mode}",
            "request_rate": float(params.get("request_rate", "nan")),
            "scheduler": "v12-gpu",
            "offered_requests": len(records),
            "completed_requests": len(records),
            "ttft_p50_s": percentile(first, .50),
            "ttft_p99_s": percentile(first, .99),
            "latency_p99_s": percentile(lat, .99),
            "throughput_rps": len(records) / max(float(params.get("servegen_duration", 120)), 1.0),
        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["source", "dataset", "request_rate", "scheduler", "offered_requests", "completed_requests", "ttft_p50_s", "ttft_p99_s", "latency_p99_s", "throughput_rps"]
    with (args.output_dir / "gpu_rate_ttft.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for dataset in sorted({row["dataset"] for row in rows}):
            selected = sorted((row for row in rows if row["dataset"] == dataset), key=lambda row: row["request_rate"])
            if not selected: continue
            plt.figure(figsize=(7, 4.5))
            plt.plot([r["request_rate"] for r in selected], [r["ttft_p99_s"] for r in selected], marker="o", label="v12 GPU")
            plt.xlabel("offered request rate (req/s)"); plt.ylabel("TTFT p99 (s)")
            plt.title(dataset + " (real GPU)"); plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
            plt.savefig(args.output_dir / (dataset + ".ttft_p99.gpu.png"), dpi=180); plt.close()
    except Exception as exc:
        (args.output_dir / "plot_error.txt").write_text(repr(exc) + "\n")
    print(json.dumps({"json_dumps": len(rows), "output": str(args.output_dir / "gpu_rate_ttft.csv")}, indent=2))


if __name__ == "__main__":
    main()
