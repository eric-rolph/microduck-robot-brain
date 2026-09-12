"""
Plot cyclictest latency histogram to evaluate real-time jitter.
Generates log-scale frequency distribution and cumulative distribution function (CDF).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_cyclictest_histogram(filepath: str = "latency_hist.txt", save_path: str = "latency_plot.png") -> None:
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    # Format: [Latency_us, Thread_0_count, (Thread_1_count, ...)]
    data = np.loadtxt(filepath, comments="#")
    if data.ndim == 1 or data.shape[1] < 2:
        print(f"Error: Unexpected format in {filepath}")
        return

    latencies = data[:, 0]
    counts = data[:, 1:].sum(axis=1)

    total_samples = counts.sum()
    if total_samples == 0:
        print("Error: No samples recorded in histogram.")
        return

    # Calculate statistics from binned data
    cdf = np.cumsum(counts) / total_samples
    valid_mask = counts > 0

    min_lat = latencies[valid_mask][0]
    max_lat = latencies[valid_mask][-1]
    mean_lat = np.sum(latencies * counts) / total_samples
    p50 = latencies[np.searchsorted(cdf, 0.50)]
    p99 = latencies[np.searchsorted(cdf, 0.99)]
    p99_9 = latencies[np.searchsorted(cdf, 0.999)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 1. Log-scale frequency distribution
    ax1.step(latencies, counts, where="mid", color="#1f77b4", linewidth=1.5, label="Samples / Bin")
    ax1.set_yscale("log")
    ax1.set_ylabel("Count (Log Scale)")
    ax1.set_title("Cyclictest Latency Distribution and Jitter Profile")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    ax1.axvline(p99, color="orange", linestyle="--", linewidth=1.2, label=f"99th Percentile: {p99:.1f} us")
    ax1.axvline(p99_9, color="red", linestyle="--", linewidth=1.2, label=f"99.9th Percentile: {p99_9:.1f} us")
    ax1.axvline(50.0, color="darkred", linestyle=":", linewidth=1.5, label="50 us RT Jitter Budget")
    ax1.legend(loc="upper right")

    # 2. Cumulative Distribution Function (CDF)
    ax2.plot(latencies, cdf * 100, color="#2ca02c", linewidth=1.5, label="Cumulative Probability")
    ax2.set_xlabel("Latency (microseconds)")
    ax2.set_ylabel("CDF (%)")
    ax2.set_ylim(0, 102)
    ax2.grid(True, linestyle="--", alpha=0.5)

    stats_text = (
        f"Samples: {int(total_samples):,}\n"
        f"Min: {min_lat:.1f} us\n"
        f"Avg: {mean_lat:.1f} us\n"
        f"Median (P50): {p50:.1f} us\n"
        f"P99: {p99:.1f} us\n"
        f"P99.9: {p99_9:.1f} us\n"
        f"Max: {max_lat:.1f} us"
    )
    ax2.text(
        0.97, 0.25, stats_text,
        transform=ax2.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.85),
        fontfamily="monospace",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    print(stats_text)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "latency_hist.txt"
    plot_cyclictest_histogram(target)
