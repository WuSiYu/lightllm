import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def collect_points(jsonl_path: Path, omit_first: int) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Return scatter x/y pairs for GEMM_OP, ATTN_OP, and COMM_OP."""
    gemm_x: List[float] = []
    gemm_y: List[float] = []
    attn_x: List[float] = []
    attn_y: List[float] = []
    comm_x: List[float] = []
    comm_y: List[float] = []

    with jsonl_path.open("r") as f:
        for idx, line in enumerate(f):
            if idx < omit_first or not line or line[0] == "#":
                continue
            data = json.loads(line)
            op_type = data.get("type")
            shapes = data.get("shapes", {}) or {}
            latency = data.get("t_elapsed_ms")
            if latency is None:
                continue
            latency_s = float(latency) / 1000.0
            if latency_s <= 0:
                continue

            if op_type == "GEMM_OP":
                m = shapes.get("m")
                k = shapes.get("k")
                n = shapes.get("n")
                if m is not None and k is not None and n is not None:
                    gemm_x.append(float(m))
                    tflops = 2.0 * float(m) * float(k) * float(n) / 1e12
                    gemm_y.append(tflops / latency_s)
            elif op_type == "ATTN_OP":
                seqlen = shapes.get("seqlen")
                if seqlen is not None:
                    attn_x.append(float(seqlen))
                    attn_y.append(float(seqlen) / latency_s)
            elif op_type == "COMM_OP":
                size = shapes.get("size")
                if size is not None:
                    comm_x.append(float(size) / 1e6)
                    # size assumed to be bytes; convert to GB/s.
                    comm_y.append((float(size) / latency_s) / 1e9)

    return gemm_x, gemm_y, attn_x, attn_y, comm_x, comm_y


def plot_scatter(gemm_xy, attn_xy, comm_xy, output: Path | None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(5, 9), sharex=False)
    fig.suptitle("Operator Throughput")

    (gemm_x, gemm_y), (attn_x, attn_y), (comm_x, comm_y) = gemm_xy, attn_xy, comm_xy

    axes[0].scatter(gemm_x, gemm_y, s=12, alpha=0.7, color="#1f77b4")
    axes[0].set_title("GEMM_OP")
    axes[0].set_xlabel("m")
    axes[0].set_ylabel("tflops/s")
    axes[0].set_ylim(bottom=0)

    axes[1].scatter(attn_x, attn_y, s=12, alpha=0.7, color="#d62728")
    axes[1].set_title("ATTN_OP")
    axes[1].set_xlabel("seqlen")
    axes[1].set_ylabel("tokens/s")
    axes[1].set_ylim(bottom=0)

    axes[2].scatter(comm_x, comm_y, s=12, alpha=0.7, color="#2ca02c")
    axes[2].set_title("COMM_OP")
    axes[2].set_xlabel("size (MB)")
    axes[2].set_ylabel("GB/s")
    axes[2].set_ylim(bottom=0)

    fig.tight_layout()

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GEMM_OP, ATTN_OP, COMM_OP latency scatter charts from JSONL perf logs.")
    parser.add_argument("jsonl", type=Path, help="Path to perf JSONL file")
    parser.add_argument("--omit-first", type=int, default=0, help="Skip first N lines (default: 0)")
    parser.add_argument("--output", type=Path, help="Output image path; if omitted, show the plot interactively")
    args = parser.parse_args()

    gemm_x, gemm_y, attn_x, attn_y, comm_x, comm_y = collect_points(args.jsonl, args.omit_first)
    plot_scatter((gemm_x, gemm_y), (attn_x, attn_y), (comm_x, comm_y), args.output)


if __name__ == "__main__":
    main()
