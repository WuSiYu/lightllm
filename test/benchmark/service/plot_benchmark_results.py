import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_cdf(ax, data, label, xlabel, title):
    if not data:
        print(f"Warning: No data for {label}")
        return

    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

    ax.plot(sorted_data, yvals, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True)

    # Add some statistics to the plot
    p50 = np.percentile(sorted_data, 50)
    p90 = np.percentile(sorted_data, 90)
    p99 = np.percentile(sorted_data, 99)
    max_val = np.max(sorted_data)
    stats_text = f"P50: {p50:.2f}\nP90: {p90:.2f}\nP99: {p99:.2f}\nMax: {max_val:.2f}"
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results from JSON dump.")
    parser.add_argument("filename", type=str, help="Path to the JSON dump file.")
    parser.add_argument("--output", type=str, default=None, help="Output filename for the plot.")
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(f"Error: File {args.filename} not found.")
        return

    print(f"Loading data from {args.filename}...")
    with open(args.filename, 'r') as f:
        data = json.load(f)

    results = data.get('results', [])
    if not results:
        print("Error: No results found in the JSON file.")
        return

    ttft_list = []
    req_max_tpot_list = []
    req_avg_tpot_list = []
    all_tpot_list = []

    for r in results:
        token_latencys = r.get('token_latencys', [])
        if not token_latencys:
            continue

        # TTFT is the first token latency
        ttft_list.append(token_latencys[0])

        # TPOTs are the subsequent latencies
        if len(token_latencys) > 1:
            tpots = token_latencys[1:]
            req_max_tpot_list.append(max(tpots))
            req_avg_tpot_list.append(np.mean(tpots))
            all_tpot_list.extend(tpots)
    # Convert to ms for better readability
    ttft_ms = [x * 1000 for x in ttft_list]
    req_max_tpot_ms = [x * 1000 for x in req_max_tpot_list]
    req_avg_tpot_ms = [x * 1000 for x in req_avg_tpot_list]
    all_tpot_ms = [x * 1000 for x in all_tpot_list]

    fig, axes = plt.subplots(4, 1, figsize=(10, 20))

    plot_cdf(axes[0], ttft_ms, "TTFT", "Latency (ms)", "Time To First Token (TTFT) CDF")
    plot_cdf(axes[1], req_max_tpot_ms, "Req-Max TPOT", "Latency (ms)", "Request Max Time Per Output Token (TPOT) CDF")
    plot_cdf(axes[2], req_avg_tpot_ms, "Req-Avg TPOT", "Latency (ms)", "Request Average Time Per Output Token (TPOT) CDF")
    plot_cdf(axes[3], all_tpot_ms, "All TPOT", "Latency (ms)", "All Time Per Output Token (TPOT) CDF")
    output_filename = args.output
    if not output_filename:
        base_name = os.path.splitext(args.filename)[0]
        output_filename = base_name + "_plots.png"

    plt.savefig(output_filename)
    print(f"Plots saved to {output_filename}")

if __name__ == "__main__":
    main()
