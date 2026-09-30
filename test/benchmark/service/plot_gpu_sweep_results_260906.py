#!/usr/bin/env python3
"""Aggregate prior real-GPU sweep result.json files into paper plots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("**/results.json")):
        payload = json.loads(path.read_text())
        cfg = payload.get("config", {})
        for policy, records in payload.get("results", {}).items():
            for record in records:
                if not record.get("successful_requests"):
                    continue
                scenario = record.get("scenario", "unknown")
                if "-r" in scenario:
                    base_scenario, rate = scenario.rsplit("-r", 1)
                else:
                    base_scenario = scenario
                    rate = cfg.get("synthetic_rate", cfg.get("servegen_rate", "nan"))
                rows.append({
                    "source": str(path),
                    "scenario": base_scenario,
                    "policy": policy,
                    "selector": record.get("selector", ""),
                    "request_rate": rate,
                    "ttft_p99_s": record.get("ttft_p99_s"),
                    "ttft_p95_s": record.get("ttft_p95_s"),
                    "successful_requests": record.get("successful_requests"),
                    "request_throughput_rps": record.get("request_throughput_rps"),
                    "input_token_throughput_s": record.get("input_token_throughput_s"),
                    "warmup_successful_requests": record.get("warmup", {}).get("successful_requests", 0),
                })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with (args.output_dir / "gpu_sweep_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        groups = {}
        for row in rows:
            groups.setdefault(row["scenario"], []).append(row)
        for scenario, selected in sorted(groups.items()):
            plt.figure(figsize=(7, 4.5))
            for policy in sorted({r["policy"] for r in selected}):
                part = sorted((r for r in selected if r["policy"] == policy), key=lambda r: float(r["request_rate"]))
                plt.plot([float(r["request_rate"]) for r in part], [float(r["ttft_p99_s"]) for r in part], marker="o", label=policy)
            plt.xlabel("offered request rate (req/s)"); plt.ylabel("TTFT p99 (s)")
            plt.title(scenario + " (real GPU historical sweep)"); plt.grid(alpha=.25); plt.legend(fontsize=8); plt.tight_layout()
            plt.savefig(args.output_dir / (scenario + ".ttft_p99.gpu.png"), dpi=180); plt.close()
    except Exception as exc:
        (args.output_dir / "plot_error.txt").write_text(repr(exc) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
