#!/usr/bin/env python3
"""从现有 PERF 日志中提取 chunked-prefill 的 batch/模型时间。"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path


PATTERN = re.compile(
    r"PERF - prefill - bs (?P<bs>\d+) token (?P<tokens>\d+) .*?"
    r"latency (?P<latency>[0-9.]+)ms \(model (?P<model>[0-9.]+)ms"
)


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * p
    low, high = int(index), min(len(values) - 1, int(index) + 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def extract(directory: Path):
    rows = []
    for filename in glob.glob(str(directory / "*.log")):
        with open(filename, errors="replace") as handle:
            for line in handle:
                match = PATTERN.search(line)
                if match:
                    rows.append({
                        "file": filename,
                        "batch_size": int(match["bs"]),
                        "batch_tokens": int(match["tokens"]),
                        "batch_latency_ms": float(match["latency"]),
                        "model_ms": float(match["model"]),
                    })
    return rows


def summarize(rows, max_tokens):
    rows = [row for row in rows if row["batch_tokens"] <= max_tokens]
    groups = {}
    for row in rows:
        groups.setdefault((row["batch_tokens"], row["batch_size"]), []).append(row)
    result = []
    for (tokens, batch_size), group in sorted(groups.items()):
        model = [item["model_ms"] for item in group]
        latency = [item["batch_latency_ms"] for item in group]
        result.append({
            "batch_tokens": tokens,
            "batch_size": batch_size,
            "count": len(group),
            "model_p50_ms": percentile(model, .5),
            "model_p90_ms": percentile(model, .9),
            "batch_p50_ms": percentile(latency, .5),
            "batch_p90_ms": percentile(latency, .9),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", nargs=2, action="append", required=True, metavar=("NAME", "DIR"))
    args = parser.parse_args()
    payload = {
        name: {
            "rows": len(extract(Path(directory))),
            "groups": summarize(extract(Path(directory)), args.max_tokens),
        }
        for name, directory in args.case
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
