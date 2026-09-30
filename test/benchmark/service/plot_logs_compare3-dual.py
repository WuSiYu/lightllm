import os
from pathlib import Path
import re
import sys
import matplotlib.pyplot as plt

# ============== CONFIG ==============

SYSTEM_NAMES = ["TP2", "TP4", "TP Switch", "TP-SMT (ours)"]
DATASET_NAMES = ["ServeGen mm-image", "ServeGen deepseek-r1", "Synthetic mix (5%)"]

# Directory matrix: rows = datasets, columns = systems.
# DIRS[i][j] = directory for dataset i, system j
DIRS = [
    # Dataset 1
    [
        "_/_SAVE/260907-70b_p22d4_fixed_tp2_sg2/servegen/mm-image",
        "_/_SAVE/260909-70b_fixed_tp4_sg2/servegen/mm-image",
        "_/_SAVE/260910-70b_naive_switch_sg2/servegen/mm-image",
        "_/_SAVE/260907-70b_p22.4d4_v13_sg2/servegen/mm-image",
    ],
    [
        "_/_SAVE/260907-70b_p22d4_fixed_tp2_sg-dsr1/servegen/deepseek-r1",
        "_/_SAVE/260909-70b_fixed_tp4_sg-dsr1/servegen/deepseek-r1",
        "_/_SAVE/260910-70b_naive_switch_sg-dsr1/servegen/deepseek-r1",
        "_/_SAVE/260907-70b_p22.4d4_v13_sg-dsr1/servegen/deepseek-r1",
    ],
    [
        "_/_SAVE/260907-70b_p22d4_fixed_tp2_simple.1/simple.1-5",
        "_/_SAVE/260909-70b_fixed_tp4_simple.1-5/simple.1-5",
        "_/_SAVE/260910-70b_naive_switch_simple.1-5/simple.1-5",
        "_/_SAVE/260907-70b_p22.4d4_v13_simple.1/simple.1-5",
    ],
]

OUTPUT = "compare.png"

# ====================================

SYSTEM_COLORS = ["#BA4E4E", "#D29834", "#3D79B6", "#4E9A4E"]
# SYSTEM_COLORS = ["tab:blue", "tab:orange", "tab:green"]
P50_STYLE = {"linestyle": ":", "marker": "o"}
AVG_STYLE = {"linestyle": "-", "marker": "D"}
P90_STYLE = {"linestyle": ":", "marker": "^"}
P99_STYLE = {"linestyle": "-", "marker": "s"}
THROUGHPUT_STYLE = {"linestyle": "-", "marker": "^"}

# Each dataset can use a different y range in either column.
Y_LIMS_P99_P90 = [(0, 8), (0, 8), (0, 8)]
Y_LIMS_AVG_P50 = [(0, 4), (0, 4), (0, 4)]
RATES_WHITELIST = [list(range(1, 100)), list(range(1, 8)), list(range(1, 100))]

def parse_log(filepath):
    text = open(filepath).read()

    m = re.search(r"Throughput:\s+([\d.]+)\s+requests/s", text)
    throughput = float(m.group(1)) if m else None

    # The TTFT summary is emitted as an average line followed by percentile values.
    m = re.search(
        r"Average first token latency:\s*([\d.]+)\s*ms\s*\n"
        r"\s*p50:\s*([\d.]+)\s*ms,\s*p90:\s*([\d.]+)\s*ms,.*?"
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
    print(f"Loaded {len(rates)} rates from {log_dir}: {rates} \n {data}")
    return (
        rates,
        [data[r][0] for r in rates],
        [data[r][1] for r in rates],
        [data[r][2] for r in rates],
        [data[r][3] for r in rates],
        [data[r][4] for r in rates],
    )


def main():
    # Rows are datasets; left column is P99/P90 and right column is avg/P50.
    fig, axes = plt.subplots(3, 2, figsize=(6.5, 8), squeeze=False)

    for i in range(len(DATASET_NAMES)):
        ax_left, ax_right = axes[i]
        row_rates = set()

        for j, sys_name in enumerate(SYSTEM_NAMES):
            log_dir = DIRS[i][j]
            if not os.path.isdir(log_dir):
                print(f"WARNING: {log_dir} not found, skipping")
                continue

            rates, throughputs, avgs, p50s, p90s, p99s = load_dir(log_dir)
            allowed = set(RATES_WHITELIST[i])
            filtered = [
                (r, avg, p50, p90, p99)
                for r, avg, p50, p90, p99 in zip(rates, avgs, p50s, p90s, p99s)
                if r in allowed
            ]
            rates = [item[0] for item in filtered]
            avgs = [item[1] for item in filtered]
            p50s = [item[2] for item in filtered]
            p90s = [item[3] for item in filtered]
            p99s = [item[4] for item in filtered]
            row_rates.update(rates)
            color = SYSTEM_COLORS[j]

            # Keep the legend compact by adding labels only on the first row.
            label_p99 = f"{sys_name} TTFT P99" if i == 0 else None
            label_p90 = f"{sys_name} TTFT P90" if i == 0 else None
            label_avg = f"{sys_name} TTFT avg" if i == 0 else None
            label_p50 = f"{sys_name} TTFT P50" if i == 0 else None
            ax_left.plot(rates, [x/1000 for x in p99s], color=color, label=label_p99, markersize=5, **P99_STYLE)
            ax_left.plot(rates, [x/1000 for x in p90s], color=color, label=label_p90, markersize=5, **P90_STYLE)
            ax_right.plot(rates, [x/1000 for x in avgs], color=color, label=label_avg, markersize=5, **AVG_STYLE)
            ax_right.plot(rates, [x/1000 for x in p50s], color=color, label=label_p50, markersize=5, **P50_STYLE)

        for ax in (ax_left, ax_right):
            ax.set_xlabel("Request Rate (req/s)")
            ax.set_xticks(sorted(row_rates))
            ax.set_ylabel("TTFT (s)")
            ax.grid(True, alpha=0.3)
        ax_left.set_ylim(*Y_LIMS_P99_P90[i])
        ax_right.set_ylim(*Y_LIMS_AVG_P50[i])
        ax_left.set_title(f"{DATASET_NAMES[i]} TTFT P99/P90", fontsize=11)
        ax_right.set_title(f"{DATASET_NAMES[i]} TTFT avg/P50", fontsize=11)

    # Shared legend at the top
    handles = []
    labels = []
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=len(SYSTEM_NAMES))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(OUTPUT, dpi=150)
    plt.savefig(OUTPUT.replace(".png", ".pdf"))
    print(f"Saved to {Path(OUTPUT).resolve()}")


if __name__ == "__main__":
    main()
