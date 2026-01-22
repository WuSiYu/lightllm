import re
import matplotlib.pyplot as plt
import argparse
import os

def parse_log_file(filename):
    """Parse log file and extract PERF data"""
    vit_data = []  # Store (vit first dimension, latency)
    prefill_data = []  # Store (prefill first dimension, latency)

    # Regular expression to match both PERF line formats
    pattern = r'PERF - (\w+) torch\.Size\(\[(\d+)(?:,.*)?\]\) - ([\d.]+) ms'

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if 'PERF' in line:
                    match = re.search(pattern, line)
                    if match:
                        perf_type = match.group(1)  # vit or prefill
                        first_dim = int(match.group(2))  # First dimension of torch.Size
                        time_ms = float(match.group(3))  # Latency in ms

                        if perf_type == 'vit':
                            vit_data.append((first_dim, time_ms))
                        elif perf_type == 'prefill':
                            prefill_data.append((first_dim, time_ms))
                    else:
                        print(f"Warning: Line {line_num} format mismatch: {line.strip()}")

        print(f"Parsing completed: Found {len(vit_data)} vit records, {len(prefill_data)} prefill records")
        return vit_data, prefill_data

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return [], []
    except Exception as e:
        print(f"Error: Exception occurred while reading file - {e}")
        return [], []

def calculate_throughput_trend(x_data, y_data, perf_type):
    """Calculate throughput trend and return annotation text"""
    if len(x_data) < 2:
        return ""

    # Calculate throughput for each data point
    throughputs = []
    for x, y_ms in zip(x_data, y_data):
        if y_ms > 0:  # Avoid division by zero
            if perf_type == 'vit':
                # For vit: images per second = (batch_size * 1000) / latency_ms
                throughput = (x * 1000) / y_ms
                throughputs.append(throughput)
            elif perf_type == 'prefill':
                # For prefill: tokens per second = (sequence_length * 1000) / latency_ms
                throughput = (x * 1000) / y_ms
                throughputs.append(throughput)

    if not throughputs:
        return ""

    # Calculate Median throughput
    median_throughput = np.median(throughputs)

    # Calculate throughput trend (linear regression on throughput vs batch size)
    if len(x_data) > 1:
        try:
            z = np.polyfit(x_data, throughputs, 1)
            slope = z[0]
            if perf_type == 'vit':
                unit = "images/s"
            else:
                unit = "tokens/s"

            if slope > 0:
                trend = "increasing"
            else:
                trend = "decreasing"

            return f"Median: {median_throughput:.1f} {unit}"
            # return f"Avg: {avg_throughput:.1f} {unit}\nTrend: {trend} ({abs(slope):.2f} {unit}/unit)"
        except:
            return f"Median: {median_throughput:.1f} {unit}"


    return ""

def create_scatter_plots(vit_data, prefill_data, title, output_filename=None):
    """Create scatter plots with throughput annotations"""
    if not vit_data and not prefill_data:
        print("No data available for plotting")
        return

    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16)

    # Plot vit scatter plot
    if vit_data:
        vit_dims, vit_times = zip(*vit_data)
        scatter1 = ax1.scatter(vit_dims, vit_times, alpha=0.7, color='blue', label='vit', s=50)
        ax1.set_xlabel('Batch Size (First Dimension)')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Vit Performance Analysis')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add trend line
        if len(vit_data) > 1:
            z = np.polyfit(vit_dims, vit_times, 1)
            p = np.poly1d(z)
            trend_line1 = ax1.plot(vit_dims, p(vit_dims), "r--", alpha=0.8, linewidth=2, label='Trend')

            # Add throughput annotation
            throughput_text = calculate_throughput_trend(vit_dims, vit_times, 'vit')
            if throughput_text:
                ax1.annotate(throughput_text,
                           xy=(0.03, 0.85), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                           fontsize=9, ha='left', va='top')

    # Plot prefill scatter plot
    if prefill_data:
        prefill_dims, prefill_times = zip(*prefill_data)
        scatter2 = ax2.scatter(prefill_dims, prefill_times, alpha=0.7, color='green', label='prefill', s=50)
        ax2.set_xlabel('Sequence Length (First Dimension)')
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Prefill Performance Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add trend line
        if len(prefill_data) > 1:
            z = np.polyfit(prefill_dims, prefill_times, 1)
            p = np.poly1d(z)
            trend_line2 = ax2.plot(prefill_dims, p(prefill_dims), "r--", alpha=0.8, linewidth=2, label='Trend')

            # Add throughput annotation
            throughput_text = calculate_throughput_trend(prefill_dims, prefill_times, 'prefill')
            if throughput_text:
                ax2.annotate(throughput_text,
                           xy=(0.03, 0.85), xycoords='axes fraction',
                           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                           fontsize=9, ha='left', va='top')

    plt.tight_layout()

    # Save or display the figure
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {output_filename}")
    else:
        plt.show()

def print_statistics(vit_data, prefill_data):
    """Print statistical information"""
    if vit_data:
        vit_dims, vit_times = zip(*vit_data)

        # Calculate throughput statistics for vit
        vit_throughputs = [(dim * 1000) / time for dim, time in vit_data]
        avg_vit_throughput = sum(vit_throughputs) / len(vit_throughputs)

        print(f"\nVit Statistics:")
        print(f"  Data points: {len(vit_data)}")
        print(f"  Batch size range: {min(vit_dims)} - {max(vit_dims)}")
        print(f"  Latency range: {min(vit_times):.2f} - {max(vit_times):.2f} ms")
        print(f"  Average latency: {sum(vit_times)/len(vit_times):.2f} ms")
        print(f"  Average throughput: {avg_vit_throughput:.2f} images/s")

    if prefill_data:
        prefill_dims, prefill_times = zip(*prefill_data)

        # Calculate throughput statistics for prefill
        prefill_throughputs = [(dim * 1000) / time for dim, time in prefill_data]
        avg_prefill_throughput = sum(prefill_throughputs) / len(prefill_throughputs)

        print(f"\nPrefill Statistics:")
        print(f"  Data points: {len(prefill_data)}")
        print(f"  Sequence length range: {min(prefill_dims)} - {max(prefill_dims)}")
        print(f"  Latency range: {min(prefill_times):.2f} - {max(prefill_times):.2f} ms")
        print(f"  Average latency: {sum(prefill_times)/len(prefill_times):.2f} ms")
        print(f"  Average throughput: {avg_prefill_throughput:.2f} tokens/s")

def main():
    parser = argparse.ArgumentParser(description='Analyze PERF logs and generate performance scatter plots')
    parser.add_argument('logfile', help='Log filename')
    parser.add_argument('-o', '--output', help='Output image filename (optional)')

    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.logfile):
        print(f"Error: File '{args.logfile}' does not exist")
        return

    # Parse log file
    vit_data, prefill_data = parse_log_file(args.logfile)

    # omit first 5 data points for better trend analysis
    SKIP = 5
    vit_data = vit_data[SKIP:] if len(vit_data) > SKIP else vit_data
    prefill_data = prefill_data[SKIP:] if len(prefill_data) > SKIP else prefill_data

    if not vit_data and not prefill_data:
        print("No valid PERF data found")
        return

    # Print statistics
    print_statistics(vit_data, prefill_data)

    # Create scatter plots
    create_scatter_plots(vit_data, prefill_data, args.logfile, args.output)

if __name__ == "__main__":
    # Import numpy for trend lines, skip if not available
    try:
        import numpy as np
    except ImportError:
        print("Warning: numpy not installed, trend lines will be skipped")
        np = None

    main()