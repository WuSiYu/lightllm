#!/usr/bin/env python3
"""
Measure client-visible TTFT across a range of sequence lengths and fit
the latency model:  L(s) = max(a*s + b*s^2, c) + d

Usage:
    python test_fit_ttft.py --host 10.0.0.1 -p 60011 --tp 1 --repeat 3
    python test_fit_ttft.py --host 10.0.0.1 -p 60011 --tp 2 --repeat 3 --output result_tp2.csv
    python test_fit_ttft.py --host 10.0.0.1 --seq-start 10 --seq-end 5000 --repeat 5
"""

import argparse
import csv
import json
import random
import time
import uuid

import numpy as np
import requests
from scipy.optimize import curve_fit


# ── latency model ────────────────────────────────────────────────────
# L(s, tp) = max(a * s / tp + b * s^2 / tp,  c) + d
# When fitting for a single TP degree, we absorb 1/tp into a,b:
#   L(s) = max(A*s + B*s^2, c) + d
# After fitting we recover: a = A * tp,  b = B * tp

def latency_model(s, a, b, c, d):
    """Vectorised model: L = max(a*s + b*s^2, c) + d"""
    return np.maximum(a * s + b * s ** 2, c) + d


# ── sequence lengths to sweep ────────────────────────────────────────

def build_seq_lengths(start=10, end=10000, max_step=None):
    """
    Generate sequence lengths with half-decade stepping:
      10,15,20,25,...,95,100,150,200,...,950,1000,1500,...,9500,10000
    Each order-of-magnitude decade uses step = 5 * 10^(k-1), giving
    ~18 points per decade — dense enough for a good fit.
    Clipped to [start, end].
    """
    lengths = []
    decade = 10  # current decade floor (10, 100, 1000, ...)
    while decade <= end:
        step = decade // 2          # 5, 50, 500, ...
        if max_step is not None:
            step = min(step, max_step)
        hi = decade * 10            # upper bound of this decade
        for v in range(decade, min(hi, end + 1), step):
            if v >= start:
                lengths.append(v)
        decade = hi
    # make sure `end` is included
    if lengths and lengths[-1] != end and end >= start:
        lengths.append(end)
    return lengths


# ── single request ───────────────────────────────────────────────────

def measure_ttft(host, port, seq_len, max_new_tokens=1):
    """
    Send one request with `seq_len` input tokens (approx) and
    max_new_tokens=1, return wall-clock latency as client-visible TTFT.
    """
    # build prompt: pad to target length (rough char≈token mapping handled
    # by the model; we rely on the server's reported prompt_tokens later)
    nonce = uuid.uuid4().hex
    system_prompt = (
        f"{nonce} ( <-- cache_bypass, ignore it ), "
        "You are a helpful assistant."
    )
    # ~1 token per "token " word;  subtract overhead tokens
    pad_len = max(0, seq_len - 55)
    body = str(random.randint(0, 1_000_000)) + " token" * pad_len
    prompt = f"{system_prompt}\n\n{body}\n\nSummarise in one word:"

    url = f"http://{host}:{port}/generate"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "frequency_penalty": 1,
        },
    }

    start = time.perf_counter()
    resp = requests.post(url, json=payload)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()

    result = resp.json()
    prompt_tokens = result.get("prompt_tokens", None)
    return elapsed, prompt_tokens


# ── fit + plot (reusable) ─────────────────────────────────────────────

def fit_and_plot(xs, ys, tp, output_base):
    """
    Fit latency_model to (xs, ys), print results, save plot.
    xs: prompt token counts (1-D array)
    ys: TTFT in seconds (1-D array)
    tp: tensor parallelism degree
    output_base: path stem for .png (e.g. "ttft_sweep")
    """
    if len(xs) < 4:
        print("[skip] not enough data points for fitting")
        return

    p0 = [1e-5, 1e-9, 0.005, 0.002]
    bounds = ([0, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf])

    try:
        popt, pcov = curve_fit(latency_model, xs, ys, p0=p0, bounds=bounds,
                               maxfev=20000)
    except RuntimeError as e:
        print(f"[fit failed] {e}")
        return

    A, B, c, d = popt
    perr = np.sqrt(np.diag(pcov))
    a_per_gpu = A * tp
    b_per_gpu = B * tp

    residuals = ys - latency_model(xs, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    print("\n" + "=" * 50)
    print("  Fitted parameters  (absorbed 1/tp)")
    print("=" * 50)
    print(f"  A  = {A:.6e}  ± {perr[0]:.2e}   (a/tp)")
    print(f"  B  = {B:.6e}  ± {perr[1]:.2e}   (b/tp)")
    print(f"  c  = {c:.6e}  ± {perr[2]:.2e}   (overhead floor)")
    print(f"  d  = {d:.6e}  ± {perr[3]:.2e}   (non-overlappable)")
    print("-" * 50)
    print(f"  tp = {tp}")
    print(f"  a  = A*tp = {a_per_gpu:.6e}")
    print(f"  b  = B*tp = {b_per_gpu:.6e}")
    print(f"  R² = {r_squared:.6f}")
    print("=" * 50)

    # ── plot ──────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs_plot = np.linspace(xs.min(), xs.max(), 500)
        ys_plot = latency_model(xs_plot, *popt)

        fig, ax = plt.subplots(figsize=(4.5, 2.5))
        ax.scatter(xs, ys * 1000, s=20, label="measured", zorder=3)
        ax.plot(xs_plot, ys_plot * 1000, "r-", lw=1.5,
                label=f"fit  R²={r_squared:.4f}")
        ax.set_xlabel("Request seq_len")
        ax.set_ylabel("TTFT (ms)")
        ax.set_title(f"TTFT vs seq length  (TP={tp})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plot_path = output_base + ".png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        fig.savefig(plot_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {plot_path}")
    except ImportError:
        print("[info] matplotlib not installed, skipping plot")


# ── load existing CSV ────────────────────────────────────────────────

def load_csv(path):
    """
    Load a CSV produced by this script (or any CSV with columns
    actual_prompt_tokens / ttft_s).  Falls back to using the 1st and
    3rd columns if headers don't match.
    """
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        # try standard column names first
        tok_col = "actual_prompt_tokens" if "actual_prompt_tokens" in fields else None
        ttft_col = "ttft_s" if "ttft_s" in fields else None

        xs, ys = [], []
        for row in reader:
            if tok_col and ttft_col:
                xs.append(float(row[tok_col]))
                ys.append(float(row[ttft_col]))
            else:
                # fallback: 1st col = x, last col = y
                vals = list(row.values())
                xs.append(float(vals[0]))
                ys.append(float(vals[-1]))

    return np.array(xs), np.array(ys)


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sweep seq lengths, measure TTFT, fit a/b/c/d"
    )
    # ── mode: load existing CSV to fit only ──────────────────────────
    parser.add_argument("--load", default=None, metavar="CSV",
                        help="Skip measurement; load an existing CSV and "
                             "fit + plot only")
    # ── measurement args ─────────────────────────────────────────────
    parser.add_argument("--host", default=None)
    parser.add_argument("-p", "--port", type=int, default=60011)
    parser.add_argument("--seq-start", type=int, default=10,
                        help="Smallest seq length to test (default: 10)")
    parser.add_argument("--seq-end", type=int, default=10000,
                        help="Largest seq length to test (default: 10000)")
    parser.add_argument("--seq-max-step", type=int, default=None,
                        help="Maximum step size between seq lengths (default: None)")
    parser.add_argument("--tp", type=int, default=1,
                        help="TP degree (used to recover per-GPU a,b)")
    parser.add_argument("--repeat", type=int, default=3,
                        help="Repeats per seq length (median is kept)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup requests before sweep")
    parser.add_argument("--output", default="ttft_sweep.csv",
                        help="CSV output path")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    args = parser.parse_args()

    tp = args.tp

    # ── load mode: fit from existing CSV ─────────────────────────────
    if args.load:
        print(f"[load] reading {args.load} ...")
        xs, ys = load_csv(args.load)
        print(f"[load] {len(xs)} data points loaded")
        output_base = args.load.rsplit(".", 1)[0]   # strip .csv
        fit_and_plot(xs, ys, tp, output_base)
        return

    # ── measurement mode ─────────────────────────────────────────────
    if args.host is None:
        import subprocess
        args.host = subprocess.check_output(
            ["hostname", "-i"]).decode().strip()

    seq_lengths = build_seq_lengths(args.seq_start, args.seq_end, args.seq_max_step)

    # ── warmup ───────────────────────────────────────────────────────
    print(f"[warmup] sending {args.warmup} requests ...")
    for _ in range(args.warmup):
        measure_ttft(args.host, args.port, 128, args.max_new_tokens)

    # ── sweep ────────────────────────────────────────────────────────
    records = []  # (target_len, actual_prompt_tokens, ttft)
    print(f"\n{'seq_len':>8}  {'prompt_tok':>10}  {'ttft_ms':>10}")
    print("-" * 34)

    for slen in seq_lengths:
        latencies = []
        actual_tok = None
        for r in range(args.repeat):
            try:
                t, ptok = measure_ttft(
                    args.host, args.port, slen, args.max_new_tokens)
                latencies.append(t)
                if ptok is not None:
                    actual_tok = ptok
            except Exception as e:
                print(f"  [WARN] seq_len={slen} repeat={r} failed: {e}")

        if not latencies:
            print(f"{slen:>8}  {'FAIL':>10}  {'FAIL':>10}")
            continue

        median_t = float(np.median(latencies))
        tok_str = str(actual_tok) if actual_tok else "?"
        print(f"{slen:>8}  {tok_str:>10}  {median_t*1000:>10.2f}")
        records.append((slen, actual_tok or slen, median_t))

    # ── save csv ─────────────────────────────────────────────────────
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_seq_len", "actual_prompt_tokens", "ttft_s"])
        for row in records:
            w.writerow(row)
    print(f"\n[saved] {args.output}  ({len(records)} points)")

    # ── fit + plot ───────────────────────────────────────────────────
    xs = np.array([r[1] for r in records], dtype=np.float64)
    ys = np.array([r[2] for r in records], dtype=np.float64)
    output_base = args.output.rsplit(".", 1)[0]
    fit_and_plot(xs, ys, tp, output_base)


if __name__ == "__main__":
    main()
