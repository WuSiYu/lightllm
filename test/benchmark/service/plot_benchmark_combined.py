import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re

def plot_cdf(ax, data_dict, xlabel, title):
    if not data_dict:
        print(f"Warning: No data for {title}")
        return

    # Sort keys by rate if possible, otherwise by filename
    def sort_key(k):
        rate = data_dict[k]['rate']
        if rate is not None:
            return (0, rate)
        return (1, k)

    sorted_keys = sorted(data_dict.keys(), key=sort_key)

    for label in sorted_keys:
        data = data_dict[label]['data']
        rate = data_dict[label]['rate']
        display_label = f"Rate: {rate}" if rate is not None else label

        if not data:
            continue

        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)
        ax.plot(sorted_data, yvals, label=display_label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results from multiple JSON dumps.")
    parser.add_argument("--pattern", type=str, default="bench_260108_*__n3000_rate*_trim.json", help="Glob pattern for JSON dump files.")
    parser.add_argument("--output", type=str, default="combined_benchmark_plots.png", help="Output filename for the plot.")
    args = parser.parse_args()

    # Find files
    # Check if the pattern contains a directory part, if not, assume current directory
    if os.path.dirname(args.pattern):
         search_pattern = args.pattern
    else:
         search_pattern = os.path.join(os.getcwd(), args.pattern)

    files = glob.glob(search_pattern)
    if not files:
        print(f"Error: No files found matching pattern: {search_pattern}")
        # Try relative to the script location if running from a different working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_pattern = os.path.join(script_dir, args.pattern)
        print(f"Trying pattern: {search_pattern}")
        files = glob.glob(search_pattern)
        if not files:
            print("Still no files found.")
            return

    print(f"Found {len(files)} files: {files}")

    ttft_data = {}
    req_max_tpot_data = {}
    req_avg_tpot_data = {}
    all_tpot_data = {}

    rate_pattern = re.compile(r"rate([0-9.]+)")

    for filename in files:
        print(f"Loading data from {filename}...")
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        base_name = os.path.basename(filename)
        match = rate_pattern.search(base_name)
        rate = float(match.group(1)) if match else None

        # fallback label if rate not found
        label = base_name

        results = data.get('results', [])
        if not results:
            print(f"Warning: No results in {filename}")
            continue

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

        # Convert to ms
        ttft_data[label] = {
            'data': [x * 1000 for x in ttft_list],
            'rate': rate
        }
        req_max_tpot_data[label] = {
            'data': [x * 1000 for x in req_max_tpot_list],
            'rate': rate
        }
        req_avg_tpot_data[label] = {
            'data': [x * 1000 for x in req_avg_tpot_list],
            'rate': rate
        }
        all_tpot_data[label] = {
            'data': [x * 1000 for x in all_tpot_list],
            'rate': rate
        }

    fig, axes = plt.subplots(4, 1, figsize=(12, 24))

    plot_cdf(axes[0], ttft_data, "Latency (ms)", "Time To First Token (TTFT) CDF")
    plot_cdf(axes[1], req_max_tpot_data, "Latency (ms)", "Request Max Time Per Output Token (TPOT) CDF")
    plot_cdf(axes[2], req_avg_tpot_data, "Latency (ms)", "Request Average Time Per Output Token (TPOT) CDF")
    plot_cdf(axes[3], all_tpot_data, "Latency (ms)", "All Time Per Output Token (TPOT) CDF")

    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Combined plots saved to {args.output}")

if __name__ == "__main__":
    main()
