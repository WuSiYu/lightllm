import os
import re
import sys
import matplotlib.pyplot as plt


def parse_log(filepath):
    text = open(filepath).read()

    m = re.search(r"Throughput:\s+([\d.]+)\s+requests/s", text)
    throughput = float(m.group(1)) if m else None

    m = re.search(r"Average first token latency:.*\n\s+p50:\s+([\d.]+)\s+ms.*p99:\s+([\d.]+)\s+ms", text)
    ftl_p50 = float(m.group(1)) if m else None
    ftl_p99 = float(m.group(2)) if m else None

    return throughput, ftl_p50, ftl_p99


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <log_dir> [output.png]")
        sys.exit(1)

    log_dir = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else os.path.join(log_dir, "plot.png")

    data = {}
    for fname in os.listdir(log_dir):
        m = re.match(r"^(\d+(?:\.\d+)?)\.log$", fname)
        if not m:
            continue
        rate = float(m.group(1))
        throughput, ftl_p50, ftl_p99 = parse_log(os.path.join(log_dir, fname))
        if throughput is not None:
            data[rate] = (throughput, ftl_p50, ftl_p99)

    if not data:
        print("No valid log files found.")
        sys.exit(1)

    rates = sorted(data.keys())
    throughputs = [data[r][0] for r in rates]
    p50s = [data[r][1] for r in rates]
    p99s = [data[r][2] for r in rates]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left y-axis: TTFT
    ax1.plot(rates, p50s, "o-", color="tab:blue", label="TTFT P50")
    ax1.plot(rates, p99s, "s-", color="tab:orange", label="TTFT P99")
    ax1.set_ylim(0, 8000)
    ax1.set_xlabel("Request Rate (req/s)")
    ax1.set_ylabel("First Token Latency (ms)")
    ax1.grid(True, alpha=0.3)

    # Right y-axis: Throughput
    ax2 = ax1.twinx()
    ax2.plot(rates, throughputs, "^-", color="tab:green", label="Throughput")
    ax2.set_ylabel("Throughput (req/s)")

    # Merge legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("TTFT & Throughput vs Request Rate")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
