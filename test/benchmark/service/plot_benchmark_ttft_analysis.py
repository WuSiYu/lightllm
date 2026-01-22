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
        display_label = f"Rate: {rate}" if rate is not None else 'inf'

        if not data:
            continue

        sorted_data = np.sort(data)
        yvals = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        ax.plot(sorted_data, yvals, label=display_label)

    ax.set_xlim(left=0, right=5000)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("CDF")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

def plot_timeline(ax, data_dict, xlabel, ylabel, title):
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
        display_label = f"Rate: {rate}" if rate is not None else 'inf'

        if rate == 8:
            # Skip rate 8 for clarity in timeline plot
            continue

        if not data:
            continue

        x = np.arange(len(data))
        # Use a thinner line and alpha to handle many overlaps
        ax.plot(x, data, label=display_label, alpha=0.6, linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    # ylim
    ax.set_ylim(bottom=0, top=2000)
    # Move legend outside to avoid obscuring data
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def main():
    parser = argparse.ArgumentParser(description="Plot TTFT benchmark results from multiple JSON dumps.")
    parser.add_argument("--pattern", type=str, default="bench_260108_*__n3000_rate*_trim.json", help="Glob pattern for JSON dump files.")
    parser.add_argument("--output", type=str, default="combined_ttft_analysis.png", help="Output filename for the plot.")
    args = parser.parse_args()

    # Find files logic matching plot_benchmark_combined.py
    if os.path.dirname(args.pattern):
         search_pattern = args.pattern
    else:
         search_pattern = os.path.join(os.getcwd(), args.pattern)

    files = glob.glob(search_pattern)
    if not files:
        # Try relative to the script location if running from a different working directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_pattern = os.path.join(script_dir, args.pattern)
        print(f"Trying pattern: {search_pattern}")
        files = glob.glob(search_pattern)
        if not files:
            print(f"Error: No files found matching pattern: {args.pattern}")
            return

    print(f"Found {len(files)} files: {files}")

    ttft_data = {}
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
        for r in results:
            token_latencys = r.get('token_latencys', [])
            if not token_latencys:
                # Handle cases with no tokens? Skip or 0?
                # Usually implies error, skipping.
                continue

            # TTFT is the first token latency
            ttft_list.append(token_latencys[0])

        # Convert to ms
        ttft_data[label] = {
            'data': [x * 1000 for x in ttft_list],
            'rate': rate
        }

    # Use constrained_layout or tight_layout with adjustment for the side legend
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), constrained_layout=True)

    plot_cdf(axes[0], ttft_data, "Latency (ms)", "Time To First Token (TTFT) CDF")
    plot_timeline(axes[1], ttft_data, "Request Index (Completion Order)", "Latency (ms)", "TTFT over Request Index")

    plt.savefig(args.output, dpi=300)
    print(f"TTFT Analysis plots saved to {args.output}")

if __name__ == "__main__":
    main()
