#!/usr/bin/env python3
"""Analyze minute-scale nonstationarity in timestamped prefill traces."""
import argparse, json, statistics
from pathlib import Path

def analyze(path: Path, out: Path, bin_ms: int = 60_000):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    bins = {}
    for r in rows:
        b = int(r["timestamp"] // bin_ms)
        bins.setdefault(b, []).append(int(r["input_length"]))
    series = []
    for b in sorted(bins):
        x = bins[b]
        series.append({"minute": b, "n": len(x),
                       "p50": statistics.median(x),
                       "p90": statistics.quantiles(x, n=10)[8] if len(x) >= 10 else max(x),
                       "mean": statistics.mean(x),
                       "long_fraction": sum(v > 4000 for v in x) / len(x),
                       "short_fraction": sum(v <= 4000 for v in x) / len(x)})
    long_fracs = [x["long_fraction"] for x in series]
    p90s = [x["p90"] for x in series]
    majority = [x["long_fraction"] >= .5 for x in series]
    changes = sum(a != b for a, b in zip(majority, majority[1:]))
    result = {"source": str(path), "rows": len(rows), "minute_bins": len(series),
              "bin_ms": bin_ms, "overall_long_fraction": sum(v > 4000 for v in [r["input_length"] for r in rows]) / len(rows),
              "minute_long_fraction_min": min(long_fracs), "minute_long_fraction_max": max(long_fracs),
              "minute_long_fraction_mean": statistics.mean(long_fracs),
              "minute_long_fraction_stdev": statistics.pstdev(long_fracs),
              "minute_p90_input_min": min(p90s), "minute_p90_input_max": max(p90s),
              "majority_class_transitions": changes, "bins": series}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result

def main():
    p = argparse.ArgumentParser(); p.add_argument("inputs", nargs="+"); p.add_argument("--out-dir", required=True)
    args = p.parse_args(); out = Path(args.out_dir)
    for inp in args.inputs:
        r = analyze(Path(inp), out / (Path(inp).stem + ".minute.json"))
        print(Path(inp).name, json.dumps({k:r[k] for k in r if k != "bins"}, sort_keys=True))

if __name__ == "__main__": main()
