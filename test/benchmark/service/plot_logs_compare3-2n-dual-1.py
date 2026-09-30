import os
from pathlib import Path
import re
import sys
import matplotlib.pyplot as plt

# ============== CONFIG ==============

SYSTEM_NAMES = ["TP-2", "TP-4", "Static Partition", "TP Switch", "TP-SMT (ours)"]
# SYSTEM_NAMES = ["TP-2", "TP-4", "static", "static_plus", "TP Switch", "TP-SMT (ours)"]
DATASET_NAMES = ["Synthetic mix (5%)"]
# DATASET_NAMES = ["ServeGen mm-image", "ServeGen deepseek-r1", "Synthetic mix (5%)"]

# 9 directories: rows = datasets, cols = systems
# DIRS[i][j] = directory for dataset i, system j
DIRS = [
    # # Dataset 1
    # [
    #     "_/_SAVE/260908-2N-70b_fixed_tp2_sg2/servegen/mm-image",
    #     "_/_SAVE/260908-2N-70b_fixed_tp4_sg2/servegen/mm-image",
    #     # "_/_SAVE/260908-2N-70b_2node_static_sg2/servegen/mm-image",
    #     "_/_SAVE/260908-2N-70b_2node_static_plus_sg2/servegen/mm-image",
    #     "_/_SAVE/260908-2N-70b_naive_switch_sg2/servegen/mm-image",
    #     "_/_SAVE/260908-2N-70b_v14_sg2/servegen/mm-image",
    # ],
    # [
    #     "_/_SAVE/260908-2N-70b_fixed_tp2_sg-dsr1/servegen/deepseek-r1",
    #     "_/_SAVE/260908-2N-70b_fixed_tp4_sg-dsr1/servegen/deepseek-r1",
    #     # "_/_SAVE/260908-2N-70b_2node_static_sg-dsr1/servegen/deepseek-r1",
    #     "_/_SAVE/260908-2N-70b_2node_static_plus_sg-dsr1/servegen/deepseek-r1",
    #     "_/_SAVE/260908-2N-70b_naive_switch_sg-dsr1/servegen/deepseek-r1",
    #     "_/_SAVE/260908-2N-70b_v14_sg-dsr1/servegen/deepseek-r1",
    # ],
    [
        "_/_SAVE/260908-2N-70b_fixed_tp2_simple.1-5/simple.1-5",
        "_/_SAVE/260908-2N-70b_fixed_tp4_simple.1-5/simple.1-5",
        # "_/_SAVE/260908-2N-70b_2node_static_simple.1-5/simple.1-5",
        "_/_SAVE/260908-2N-70b_2node_static_plus_simple.1-5/simple.1-5",
        "_/_SAVE/260908-2N-70b_naive_switch_simple.1-5/simple.1-5",
        "_/_SAVE/260908-2N-70b_v14_simple.1-5/simple.1-5",
    ],
]

OUTPUT = "compare-2n.png"

# ====================================

SYSTEM_COLORS = ["#BA4E4E", "#D29834", "#3D79B6", "#8C4D99", "#4E9A4E"]
# SYSTEM_COLORS = ["#BA4E4E", "#D29834", "#3D79B6", "#8C4D99", "#5B5B5B", "#4E9A4E"]
# SYSTEM_COLORS = ["tab:blue", "tab:orange", "tab:green"]
P50_STYLE = {"linestyle": "--", "marker": "o"}
P90_STYLE = {"linestyle": ":", "marker": "^"}
P99_STYLE = {"linestyle": "-", "marker": "s"}
AVG_STYLE = {"linestyle": "-.", "marker": "D"}
THROUGHPUT_STYLE = {"linestyle": "-", "marker": "^"}

# Configure the y-axis independently for each dataset and column.  The left
# column contains the tail percentiles; the right column contains average and
# median TTFT, which generally need a different scale.
Y_LIMS_P99_P90 = [(0, 8000), (0, 8000), (0, 8000)]
Y_LIMS_AVG_P50 = [(0, 8000), (0, 8000), (0, 8000)]
RATES_WHITELIST = [list(range(1, 100)), list(range(1, 100)), list(range(1, 100))]
RATES_BLACKLIST = [[8, 10, 14], [8, 10, 14], [8, 10, 14]]

def parse_log(filepath):
    text = open(filepath).read()

    m = re.search(r"Throughput:\s+([\d.]+)\s+requests/s", text)
    throughput = float(m.group(1)) if m else None

    # Keep the percentile values tied to the TTFT section.  Other sections in
    # the same log also contain p50/p99 lines (request and decode latency).
    m = re.search(
        r"Average first token latency:\s*([\d.]+)\s*ms\s*\n"
        r"\s*p50:\s*([\d.]+)\s*ms,\s*p90:\s*([\d.]+)\s*ms.*?"
        r"p99:\s*([\d.]+)\s*ms",
        text,
    )
    ftl_avg = float(m.group(1)) if m else None
    ftl_p50 = float(m.group(2)) if m else None
    ftl_p90 = float(m.group(3)) if m else None
    ftl_p99 = float(m.group(4)) if m else None

    return throughput, ftl_avg, ftl_p50, ftl_p90, ftl_p99


def load_dir(log_dir):
    data = {}
    for fname in os.listdir(log_dir):
        m = re.match(r"^(\d+(?:\.\d+)?)\.log$", fname)
        if not m:
            continue
        rate = float(m.group(1))
        result = parse_log(os.path.join(log_dir, fname))
        if result[0] is not None:
            data[rate] = result
    rates = sorted(data.keys())
    print(f"Loaded {len(rates)} rates from {log_dir}: {rates}")
    return (
        rates,
        [data[r][0] for r in rates],
        [data[r][1] for r in rates],  # average TTFT
        [data[r][2] for r in rates],  # p50
        [data[r][3] for r in rates],  # p90
        [data[r][4] for r in rates],  # p99
    )


def main():
    fig, axes = plt.subplots(len(DATASET_NAMES), 2, figsize=(12, 4), squeeze=False)

    for i, (ax_left, ax_right) in enumerate(axes):
        row_rates = set()

        for j, sys_name in enumerate(SYSTEM_NAMES):
            log_dir = DIRS[i][j]
            if not os.path.isdir(log_dir):
                print(f"WARNING: {log_dir} not found, skipping")
                continue

            rates, throughputs, avgs, p50s, p90s, p99s = load_dir(log_dir)
            filtered = [
                (r, avg, p50, p90, p99, throughput)
                for r, avg, p50, p90, p99, throughput in zip(
                    rates, avgs, p50s, p90s, p99s, throughputs
                )
                if r in RATES_WHITELIST[i]
            ]
            filtered = [
                (r, avg, p50, p90, p99, throughput)
                for r, avg, p50, p90, p99, throughput in zip(
                    rates, avgs, p50s, p90s, p99s, throughputs
                )
                if r not in RATES_BLACKLIST[i]
            ]
            if not filtered:
                continue
            rates, avgs, p50s, p90s, p99s, throughputs = map(list, zip(*filtered))
            row_rates.update(rates)
            color = SYSTEM_COLORS[j]

            # Only add label on the first subplot to avoid duplicates
            label_avg = f"{sys_name} TTFT avg" if i == 0 else None
            label_p50 = f"{sys_name} TTFT P50" if i == 0 else None
            label_p90 = f"{sys_name} TTFT P90" if i == 0 else None
            label_p99 = f"{sys_name} TTFT P99" if i == 0 else None
            ax_left.plot(rates, p90s, color=color, label=label_p90, markersize=5, **P90_STYLE)
            ax_left.plot(rates, p99s, color=color, label=label_p99, markersize=5, **P99_STYLE)
            ax_right.plot(rates, avgs, color=color, label=label_avg, markersize=5, **AVG_STYLE)
            ax_right.plot(rates, p50s, color=color, label=label_p50, markersize=5, **P50_STYLE)

        for ax in (ax_left, ax_right):
            ax.set_xlabel("Request Rate (req/s)")
            ax.set_xticks(sorted(row_rates))
            ax.grid(True, alpha=0.3)
        ax_left.set_ylabel("First Token Latency (ms)")
        ax_right.set_ylabel("First Token Latency (ms)")
        ax_left.set_ylim(*Y_LIMS_P99_P90[i])
        ax_right.set_ylim(*Y_LIMS_AVG_P50[i])
        ax_left.set_title(f"{DATASET_NAMES[i]}: TTFT P99 / P90")
        ax_right.set_title(f"{DATASET_NAMES[i]}: TTFT avg / P50")

    # Shared legend at the top
    handles_left, labels_left = axes[0, 0].get_legend_handles_labels()
    handles_right, labels_right = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="lower center",
        fontsize=8,
        ncol=len(SYSTEM_NAMES),
    )
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    plt.savefig(OUTPUT, dpi=150)
    plt.savefig(OUTPUT.replace(".png", ".pdf"))
    print(f"Saved to {Path(OUTPUT).resolve()}")


if __name__ == "__main__":
    main()
