"""
Benchmark: Strided (zero-copy slice) vs Contiguous GEMM — Transposed Weights
=============================================================================
Many inference frameworks pre-transpose weights for efficiency:
  Standard:    W shape (out, in),   GEMM = x @ W.T
  Transposed:  W shape (in, out),   GEMM = x @ W    (no transpose, cuBLAS-friendly)

With transposed weights, the strided case flips:
  - Column-parallel (gate_proj, up_proj): TP splits output dim → column slice → STRIDED
  - Row-parallel (down_proj): TP splits input dim → row slice → contiguous (not interesting)

This script benchmarks the strided column-parallel case with transposed weights.

Llama 3.3 70B MLP dimensions:
  hidden_size       = 8192
  intermediate_size = 28672

gate_proj / up_proj transposed weight: (hidden, intermediate) = (8192, 28672)
  TP2 shard: column slice → (8192, 14336)  — STRIDED
  TP4 from TP2: column slice → (8192, 7168)  — STRIDED
  TP8 from TP2: column slice → (8192, 3584)  — STRIDED

GEMM: output = x @ W   where x is (seq, 8192) and W is (8192, intermediate/tp)

Usage:
  python bench_strided_gemm_transposed.py [--device cuda:0] [--warmup 50] [--iters 200] [--dtype bf16]
"""

import argparse
import torch
import time
import sys

# ─── Llama 3.3 70B MLP dimensions ───
HIDDEN = 8192
INTERMEDIATE = 28672


def bench_matmul(x, w, warmup, iters):
    """
    Benchmark x @ w (no transpose).
    Returns: median time (ms), TFLOPS
    """
    for _ in range(warmup):
        torch.mm(x, w)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.mm(x, w)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    times.sort()
    median_ms = times[len(times) // 2]

    M, K = x.shape
    N = w.shape[1]
    flops = 2 * M * N * K
    tflops = flops / (median_ms / 1000) / 1e12

    return median_ms, tflops


def run_one(label, x, w_strided, warmup, iters):
    """Run benchmark for strided view and its contiguous copy."""
    w_contig = w_strided.contiguous()

    assert not w_strided.is_contiguous(), \
        f"[{label}] Expected strided tensor but got contiguous"
    assert w_contig.is_contiguous()

    ms_s, tf_s = bench_matmul(x, w_strided, warmup, iters)
    ms_c, tf_c = bench_matmul(x, w_contig, warmup, iters)

    diff_pct = (ms_s - ms_c) / ms_c * 100
    return ms_c, tf_c, ms_s, tf_s, diff_pct


def main():
    parser = argparse.ArgumentParser(
        description="Strided vs Contiguous GEMM benchmark (transposed weights)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Dtype:  {args.dtype}")
    print(f"Warmup: {args.warmup}  Iters: {args.iters}")
    print(f"Llama 3.3 70B MLP: hidden={HIDDEN}, intermediate={INTERMEDIATE}")
    print(f"NOTE: Using pre-transposed weights (in, out). GEMM = x @ W, no transpose.")
    print()

    # gate_proj / up_proj transposed: shape (hidden, intermediate) = (8192, 28672)
    # TP2 shard = column slice of full transposed weight
    # We simulate: full weight → TP2 shard → TP4/TP8 shard
    # Full transposed weight: (8192, 28672)
    # TP2 shard 0: W[:, 0:14336] = (8192, 14336) — strided
    full_weight = torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device=device)

    # TP2 shard (column slice from full weight)
    tp2_shard = full_weight[:, :INTERMEDIATE // 2]  # (8192, 14336) — strided
    print(f"Full transposed gate/up weight:  {list(full_weight.shape)}")
    print(f"TP2 shard (col slice):           {list(tp2_shard.shape)}, "
          f"contiguous={tp2_shard.is_contiguous()}, stride={tp2_shard.stride()}")
    print()

    seq_lens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # Slice TP2 shard → TP4 and TP8
    # TP4 from TP2: W_tp2[:, 0:7168] = (8192, 7168) — still strided
    # TP8 from TP2: W_tp2[:, 0:3584] = (8192, 3584) — still strided
    tp_configs = [
        ("gate/up TP2→TP4", tp2_shard[:, :INTERMEDIATE // 4]),   # (8192, 7168)
        ("gate/up TP2→TP8", tp2_shard[:, :INTERMEDIATE // 8]),   # (8192, 3584)
    ]

    print("=" * 90)
    print(f"{'Config':<30} {'SeqLen':>7} {'Contig ms':>10} {'Contig TF':>10} "
          f"{'Stride ms':>10} {'Stride TF':>10} {'Diff%':>7}")
    print("=" * 90)

    for label, w_slice in tp_configs:
        in_features = w_slice.shape[0]   # 8192
        out_features = w_slice.shape[1]  # 7168 or 3584
        print(f"# {label}: weight shape {list(w_slice.shape)}, "
              f"contiguous={w_slice.is_contiguous()}, stride={w_slice.stride()}")

        for seq_len in seq_lens:
            x = torch.randn(seq_len, in_features, dtype=dtype, device=device)
            ms_c, tf_c, ms_s, tf_s, diff = run_one(
                label, x, w_slice, args.warmup, args.iters
            )
            print(f"{label:<30} {seq_len:>7} {ms_c:>10.3f} {tf_c:>10.2f} "
                  f"{ms_s:>10.3f} {tf_s:>10.2f} {diff:>+7.2f}%")

        print("-" * 90)

    # Sanity: row-parallel (down_proj) transposed = (intermediate, hidden)
    # TP2 shard = row slice → contiguous
    down_tp2 = torch.randn(INTERMEDIATE // 2, HIDDEN, dtype=dtype, device=device)
    down_tp4 = down_tp2[:INTERMEDIATE // 4, :]  # row slice
    print()
    print(f"Sanity: down_proj transposed TP2 shard: {list(down_tp2.shape)}")
    print(f"  TP2→TP4 row slice: {list(down_tp4.shape)}, contiguous={down_tp4.is_contiguous()}")
    print("  (Row slice is always contiguous — no strided GEMM needed.)")


if __name__ == "__main__":
    main()
