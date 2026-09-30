import os
from pathlib import Path
import re
import sys
import matplotlib.pyplot as plt

# ============== CONFIG ==============

SYSTEM_NAMES = ["1x TP-4", "2x TP-2", "Flex TP (ours)"]
DATASET_NAMES = ["ServeGen mm-image", "Dataset 2", "Synthetic mix (5%)"]

# 9 directories: rows = datasets, cols = systems
# DIRS[i][j] = directory for dataset i, system j
DIRS = [
    # Dataset 1
    [
        "test/benchmark/service/_/260409-70b_p4d4_v8_mps_sg2/servegen/mm-image",
        "test/benchmark/service/_/260409-70b_p22d4_v8_mps_sg2/servegen/mm-image",
        "test/benchmark/service/_/260409-70b_p22.4d4_v8_mps_flex_naive_mix_sg2/servegen/mm-image",
    ],
    # Dataset 2
    # [
    #     "test/benchmark/service/_/260409-70b_p4d4_v8_mps_sg1/servegen/m-large",
    #     "test/benchmark/service/_/260409-70b_p22d4_v8_mps_sg1/servegen/m-large",
    #     "test/benchmark/service/_/260409-70b_p22.4d4_v8_mps_flex_naive_8k_sg1/servegen/m-large",
    # ],
    [
        "path/to/dataset3/systemA",
        "path/to/dataset3/systemB",
        "path/to/dataset3/systemC",
    ],
    # Dataset 3
    [
        "test/benchmark/service/_/260409-70b_p4d4_v8_mps_simple.1-5/simple.1-5",
        "test/benchmark/service/_/260409-70b_p22d4_v8_mps_simple.1-5/simple.1-5",
        "test/benchmark/service/_/260409-70b_p22.4d4_v8_mps_naive_8k_simple.1-5/simple.1-5",
    ],
]

OUTPUT = "compare.png"

# ====================================

SYSTEM_COLORS = ["#BA4E4E", "#D29834", "#3D79B6"]
# SYSTEM_COLORS = ["tab:blue", "tab:orange", "tab:green"]
P50_STYLE = {"linestyle": "--", "marker": "o"}
P99_STYLE = {"linestyle": "-", "marker": "s"}
THROUGHPUT_STYLE = {"linestyle": "-", "marker": "^"}

MAX_TTFT = [4000, 8000, 8000]
RATES_WHITELIST = [list(range(1, 9)), list(range(1, 21)), list(range(1, 21))]

def parse_log(filepath):
    text = open(filepath).read()

    m = re.search(r"Throughput:\s+([\d.]+)\s+requests/s", text)
    throughput = float(m.group(1)) if m else None

    m = re.search(r"Average first token latency:.*\n\s+p50:\s+([\d.]+)\s+ms.*p99:\s+([\d.]+)\s+ms", text)
    ftl_p50 = float(m.group(1)) if m else None
    ftl_p99 = float(m.group(2)) if m else None

    return throughput, ftl_p50, ftl_p99


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
    return (
        rates,
        [data[r][0] for r in rates],
        [data[r][1] for r in rates],
        [data[r][2] for r in rates],
    )


def main():
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 8))

    for i, ax1 in enumerate(axes):
        # ax2 = ax1.twinx()

        for j, sys_name in enumerate(SYSTEM_NAMES):
            log_dir = DIRS[i][j]
            if not os.path.isdir(log_dir):
                print(f"WARNING: {log_dir} not found, skipping")
                continue

            rates, throughputs, p50s, p99s = load_dir(log_dir)
            rates = [r for r in rates if r in RATES_WHITELIST[i]]
            p50s = [p for r, p in zip(rates, p50s) if r in RATES_WHITELIST[i]]
            p99s = [p for r, p in zip(rates, p99s) if r in RATES_WHITELIST[i]]
            throughputs = [t for r, t in zip(rates, throughputs) if r in RATES_WHITELIST[i]]
            color = SYSTEM_COLORS[j]

            # Only add label on the first subplot to avoid duplicates
            label_p50 = f"{sys_name} TTFT P50" if i == 0 else None
            label_p99 = f"{sys_name} TTFT P99" if i == 0 else None
            ax1.plot(rates, p50s, color=color, label=label_p50, markersize=5, **P50_STYLE)
            ax1.plot(rates, p99s, color=color, label=label_p99, markersize=5, **P99_STYLE)
            # ax2.plot(rates, throughputs, color=color, label=f"{sys_name} Throughput", markersize=5, **THROUGHPUT_STYLE)

        ax1.set_xlabel("Request Rate (req/s)")
        ax1.set_xticks(rates)
        ax1.set_ylabel("First Token Latency (ms)")
        ax1.set_ylim(0, MAX_TTFT[i])
        # ax2.set_ylabel("Throughput (req/s)")
        ax1.set_title(DATASET_NAMES[i])
        ax1.grid(True, alpha=0.3)

    # Shared legend at the top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=8, ncol=len(SYSTEM_NAMES))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUTPUT, dpi=150)
    plt.savefig(OUTPUT.replace(".png", ".pdf"))
    print(f"Saved to {Path(OUTPUT).resolve()}")


if __name__ == "__main__":
    main()
