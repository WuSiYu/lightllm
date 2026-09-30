"""
Benchmark: Strided (zero-copy slice) vs Contiguous GEMM
========================================================
Simulates Flex TP weight sharing for Llama 3.3 70B MLP layers.

TP2 weights are sliced to produce TP4 / TP8 shards.
  - Column-parallel layers (gate_proj, up_proj): TP2 shard shape (14336, 8192)
    → row slice → contiguous (not interesting)
  - Row-parallel layers (down_proj): TP2 shard shape (8192, 14336)
    → column slice → STRIDED (the interesting case)

We benchmark the GEMM: output = input @ weight.T  (i.e. F.linear)
for both the strided slice and a contiguous copy of the same data.

Llama 3.3 70B MLP dimensions:
  hidden_size      = 8192
  intermediate_size = 28672

Usage:
  python bench_strided_gemm.py [--device cuda:0] [--warmup 50] [--iters 200] [--dtype bf16]
"""

import argparse
import torch
import torch.nn.functional as F
import time
import sys

# ─── Llama 3.3 70B MLP dimensions ───
HIDDEN = 8192
INTERMEDIATE = 28672

def make_tp2_weights(dtype, device):
    """Create TP2 shard weights as they would exist in memory."""
    # Column-parallel: gate_proj / up_proj — TP2 shard = first half of rows
    # Full weight: (intermediate, hidden) = (28672, 8192)
    # TP2 shard: (14336, 8192)
    col_parallel = torch.randn(INTERMEDIATE // 2, HIDDEN, dtype=dtype, device=device)

    # Row-parallel: down_proj — TP2 shard = first half of columns
    # Full weight: (hidden, intermediate) = (8192, 28672)
    # TP2 shard: (8192, 14336)
    row_parallel = torch.randn(HIDDEN, INTERMEDIATE // 2, dtype=dtype, device=device)

    return col_parallel, row_parallel


def slice_to_tp(weight, target_tp, source_tp, parallel_mode):
    """
    Slice a source_tp shard to get a target_tp shard (first shard, shard_id=0).
    Returns a strided view (no copy).
    """
    ratio = target_tp // source_tp  # how many target shards per source shard
    if parallel_mode == "col":
        # row slice → always contiguous
        rows_per_shard = weight.shape[0] // ratio
        return weight[:rows_per_shard, :]
    else:
        # column slice → STRIDED
        cols_per_shard = weight.shape[1] // ratio
        return weight[:, :cols_per_shard]


def bench_gemm(x, weight, warmup, iters):
    """
    Benchmark F.linear(x, weight) = x @ weight.T
    Returns: median time (ms), TFLOPS
    """
    # warmup
    for _ in range(warmup):
        F.linear(x, weight)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        F.linear(x, weight)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms

    times.sort()
    median_ms = times[len(times) // 2]

    # FLOPS: 2 * M * N * K  (M=seq_len, N=out_features, K=in_features)
    M, K = x.shape
    N = weight.shape[0]
    flops = 2 * M * N * K
    tflops = flops / (median_ms / 1000) / 1e12

    return median_ms, tflops


def run_one(label, x, weight_strided, warmup, iters):
    """Run benchmark for strided view and its contiguous copy."""
    weight_contig = weight_strided.contiguous()

    assert not weight_strided.is_contiguous(), \
        f"[{label}] Expected strided tensor but got contiguous"
    assert weight_contig.is_contiguous()

    ms_s, tf_s = bench_gemm(x, weight_strided, warmup, iters)
    ms_c, tf_c = bench_gemm(x, weight_contig, warmup, iters)

    diff_pct = (ms_s - ms_c) / ms_c * 100
    return ms_c, tf_c, ms_s, tf_s, diff_pct


def main():
    parser = argparse.ArgumentParser(description="Strided vs Contiguous GEMM benchmark")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Dtype:  {args.dtype}")
    print(f"Warmup: {args.warmup}  Iters: {args.iters}")
    print(f"Llama 3.3 70B MLP: hidden={HIDDEN}, intermediate={INTERMEDIATE}")
    print()

    # Create TP2 base weights
    col_parallel_tp2, row_parallel_tp2 = make_tp2_weights(dtype, device)
    print(f"TP2 col-parallel (gate/up) shape: {list(col_parallel_tp2.shape)}")
    print(f"TP2 row-parallel (down)    shape: {list(row_parallel_tp2.shape)}")
    print()

    seq_lens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    tp_targets = [4, 8]  # slice TP2 → TP4, TP2 → TP8

    # We only benchmark row-parallel (down_proj) because that's the strided case.
    # Column-parallel slicing is always contiguous.
    print("=" * 90)
    print(f"{'Config':<30} {'SeqLen':>7} {'Contig ms':>10} {'Contig TF':>10} "
          f"{'Stride ms':>10} {'Stride TF':>10} {'Diff%':>7}")
    print("=" * 90)

    results = []

    for target_tp in tp_targets:
        # Row-parallel: down_proj — column slice is strided
        w_slice = slice_to_tp(row_parallel_tp2, target_tp, 2, "row")
        # w_slice shape: (8192, 14336 // (target_tp // 2))
        # For F.linear: weight is (out_features, in_features)
        # down_proj: input (seq, intermediate/tp) → output (seq, hidden)
        # So weight for F.linear = (hidden, intermediate/tp) — but that's w_slice already
        # Actually F.linear expects weight (out, in), computes x @ W.T
        # For down_proj: x is (seq, intermediate/tp), W is (hidden, intermediate/tp)
        # output = x @ W.T = (seq, intermediate/tp) @ (intermediate/tp, hidden) = (seq, hidden)

        in_features = w_slice.shape[1]  # intermediate / tp
        out_features = w_slice.shape[0]  # hidden

        label = f"down_proj TP2→TP{target_tp}"

        for seq_len in seq_lens:
            x = torch.randn(seq_len, in_features, dtype=dtype, device=device)

            ms_c, tf_c, ms_s, tf_s, diff = run_one(
                label, x, w_slice, args.warmup, args.iters
            )
            print(f"{label:<30} {seq_len:>7} {ms_c:>10.3f} {tf_c:>10.2f} "
                  f"{ms_s:>10.3f} {tf_s:>10.2f} {diff:>+7.2f}%")
            results.append({
                "config": label, "seq_len": seq_len,
                "contig_ms": ms_c, "contig_tflops": tf_c,
                "strided_ms": ms_s, "strided_tflops": tf_s,
                "diff_pct": diff,
            })

        print("-" * 90)

    # Also benchmark col-parallel to show it's contiguous (sanity check)
    print()
    print("Sanity check: column-parallel (gate/up_proj) slicing is contiguous")
    for target_tp in tp_targets:
        w_slice = slice_to_tp(col_parallel_tp2, target_tp, 2, "col")
        print(f"  TP2→TP{target_tp} gate_proj slice shape: {list(w_slice.shape)}, "
              f"contiguous: {w_slice.is_contiguous()}")

    print()
    print("Done. Row-parallel (down_proj) is the only strided case;")
    print("column-parallel (gate/up_proj) slicing always produces contiguous tensors.")


if __name__ == "__main__":
    main()
