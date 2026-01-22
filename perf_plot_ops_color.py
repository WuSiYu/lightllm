import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import matplotlib.pyplot as plt


def collect_points(
    jsonl_path: Path, omit_first: int, attn_scale: float = 1, comm_scale: float = 1
) -> Tuple[List[float], List[float], List[str], List[int], List[float], List[float], List[str], List[int], List[float], List[float], List[str], List[int]]:
    """Return scatter data and labels for GEMM_OP, ATTN_OP, and COMM_OP."""

    gemm_x: List[float] = []
    gemm_y: List[float] = []
    gemm_labels: List[str] = []
    gemm_pos_idx: List[int] = []

    attn_x: List[float] = []
    attn_y: List[float] = []
    attn_labels: List[str] = []
    attn_pos_idx: List[int] = []

    comm_x: List[float] = []
    comm_y: List[float] = []
    comm_labels: List[str] = []
    comm_pos_idx: List[int] = []

    layer_global_pos: int = 0
    layer_op_color_idx_pos: Dict[str, int] = {"GEMM_OP": 0, "ATTN_OP": 0, "COMM_OP": 0}
    layer_aligned = False

    with jsonl_path.open("r") as f:
        for idx, line in enumerate(f):
            if not line or line[0] == "#":
                continue
            if idx < omit_first:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line ({idx}):", line)
                continue
            op_type = data.get("type")
            shapes = data.get("shapes", {}) or {}
            latency = data.get("t_elapsed_ms")
            if not layer_aligned:
                if op_type == "LAYER":
                    layer_aligned = True
                else:
                    continue
            layer_global_pos += 1
            if latency is None:
                continue
            latency_s = float(latency) / 1000.0
            if latency_s <= 0:
                continue

            if op_type == "LAYER":
                layer_global_pos = 0
                layer_op_color_idx_pos = {"GEMM_OP": 0, "ATTN_OP": 0, "COMM_OP": 0}
                continue

            if op_type in layer_op_color_idx_pos:
                layer_op_color_idx_pos[op_type] += 1
            if op_type == "GEMM_OP":
                m = shapes.get("m")
                k = shapes.get("k")
                n = shapes.get("n")
                if m is not None and k is not None and n is not None:
                    pos = layer_global_pos
                    layer_global_pos += 1
                    gemm_x.append(float(m))
                    tflops = 2.0 * float(m) * float(k) * float(n) / 1e12
                    gemm_y.append(tflops / latency_s)
                    gemm_labels.append(f"{data.get('name', 'gemm')}.{pos + 1}")
                    gemm_pos_idx.append(layer_op_color_idx_pos[op_type])
                else:
                    flops = shapes.get("flops")
                    total_m = shapes.get("total_m")
                    if flops is not None and total_m is not None:
                        pos = layer_global_pos
                        layer_global_pos += 1
                        gemm_x.append(float(total_m))
                        tflops = float(flops) / 1e12
                        gemm_y.append(tflops / latency_s)
                        gemm_labels.append(f"{data.get('name', 'gemm')}.{pos + 1}")
                        gemm_pos_idx.append(layer_op_color_idx_pos[op_type])
            elif op_type == "ATTN_OP":
                seqlen = shapes.get("seqlen")
                if seqlen is not None:
                    pos = layer_global_pos
                    layer_global_pos += 1
                    attn_x.append(float(seqlen))
                    attn_y.append(float(seqlen) / latency_s / 1e6 * attn_scale)
                    attn_labels.append(f"{data.get('name', 'attn')}.{pos + 1}")
                    attn_pos_idx.append(layer_op_color_idx_pos[op_type])
            elif op_type == "COMM_OP":
                size = shapes.get("size")
                if size is not None:
                    pos = layer_global_pos
                    layer_global_pos += 1
                    comm_x.append(float(size) / 1e6)
                    # size assumed to be bytes; convert to GB/s.
                    comm_y.append((float(size) / latency_s) / 1e9 * comm_scale)
                    comm_labels.append(f"{data.get('name', 'comm')}.{pos + 1}")
                    comm_pos_idx.append(layer_op_color_idx_pos[op_type])

    return (
        gemm_x,
        gemm_y,
        gemm_labels,
        gemm_pos_idx,
        attn_x,
        attn_y,
        attn_labels,
        attn_pos_idx,
        comm_x,
        comm_y,
        comm_labels,
        comm_pos_idx,
    )


def plot_scatter(gemm_xy, attn_xy, comm_xy, name, output: Path | None) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(5, 9), sharex=False)
    fig.suptitle(f"Operator Throughput \n{name}")

    (
        gemm_x,
        gemm_y,
        gemm_labels,
        gemm_pos_idx,
    ) = gemm_xy
    (
        attn_x,
        attn_y,
        attn_labels,
        attn_pos_idx,
    ) = attn_xy
    (
        comm_x,
        comm_y,
        comm_labels,
        comm_pos_idx,
    ) = comm_xy

    cmap = plt.get_cmap("Set1")

    def scatter_with_legend(ax, xs, ys, labels, pos_idx, title, xlabel, ylabel):
        colors = [cmap(i % cmap.N) for i in pos_idx]
        ax.scatter(xs, ys, s=14, alpha=0.03, c=colors)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)

        handles = []
        seen = {}
        for lbl, color in zip(labels, colors):
            safe_lbl = ' ' + lbl if lbl.startswith('_') else lbl
            if lbl in seen:
                continue
            seen[lbl] = True
            handles.append(plt.Line2D([0], [0], marker="o", color="w", label=safe_lbl, markerfacecolor=color, markersize=6))
        if handles:
            ax.legend(handles=handles, title="op_name.pos", loc="best", fontsize="x-small")

        # map each label to its first-seen color
        color_map = {}
        for lbl, color in zip(labels, colors):
            if lbl not in color_map:
                color_map[lbl] = color

        # compute average y per label
        sums = defaultdict(float)
        counts = defaultdict(int)
        for lbl, y in zip(labels, ys):
            try:
                val = float(y)
            except Exception:
                continue
            sums[lbl] += val
            counts[lbl] += 1

        # draw a horizontal dashed line for each label's average and annotate it
        if sums:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            for lbl, total in sums.items():
                avg = total / counts[lbl]
                color = color_map.get(lbl, "gray")
                ax.axhline(avg, color=color, linestyle="--", linewidth=0.8, alpha=0.9)
                xpos = xlim[1]
                ax.text(xpos, avg + ylim[1] * 0.02, f" {lbl} avg={avg:.3g}", color=color, va="center", ha="right", fontsize="x-small", alpha=0.9)
            ax.set_xlim(xlim)


    scatter_with_legend(axes[0], gemm_x, gemm_y, gemm_labels, gemm_pos_idx, "GEMM_OP", "m", "tflops/s")
    scatter_with_legend(axes[1], attn_x, attn_y, attn_labels, attn_pos_idx, "ATTN_OP", "seqlen", "M tokens/s")
    scatter_with_legend(axes[2], comm_x, comm_y, comm_labels, comm_pos_idx, "COMM_OP", "size (MB)", "GB/s")

    fig.tight_layout()

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=300)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GEMM_OP, ATTN_OP, COMM_OP throughput scatter charts from JSONL perf logs.")
    parser.add_argument("jsonl", type=Path, help="Path to perf JSONL file")
    parser.add_argument("--omit-first", type=int, default=0, help="Skip first N lines (default: 0)")
    parser.add_argument("--output", type=Path, help="Output image path; if omitted, show the plot interactively")
    parser.add_argument("--attn-scale", type=float, default=1, help="")
    parser.add_argument("--comm-scale", type=float, default=1, help="")
    args = parser.parse_args()

    gemm_x, gemm_y, gemm_labels, gemm_pos_idx, attn_x, attn_y, attn_labels, attn_pos_idx, comm_x, comm_y, comm_labels, comm_pos_idx = collect_points(args.jsonl, args.omit_first, args.attn_scale, args.comm_scale)
    plot_scatter((gemm_x, gemm_y, gemm_labels, gemm_pos_idx), (attn_x, attn_y, attn_labels, attn_pos_idx), (comm_x, comm_y, comm_labels, comm_pos_idx), args.jsonl, args.output)


if __name__ == "__main__":
    main()
