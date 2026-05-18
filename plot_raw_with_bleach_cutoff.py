"""
Plot raw calcium traces: full timeseries and after initial bleaching dip.

Automatically determines a bleaching cutoff by finding when the rate of
fluorescence decay (averaged across all traces) drops below a threshold,
indicating the fast photobleaching phase has ended.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("/home/lauren/quick_wholebrain_overview/outputs")
TRACES_CSV = OUTPUT_DIR / "traces" / "raw_traces.csv"

# Condition mapping (after exclusions)
BLAC_CAT_DEAD = ["2026-03-03-01", "2026-03-03-02", "2026-03-03-03"]
BLAC = ["2026-03-03-05", "2026-03-04-02", "2026-03-04-03", "2026-03-04-04", "2026-03-04-05"]

COLOR_DEAD = "#4477AA"
COLOR_ALIVE = "#CC3311"


def find_bleach_cutoff(time_s, traces_dict, smooth_window=15, deriv_threshold=0.002):
    """
    Determine the time index where the initial bleaching dip ends.

    Normalizes each trace to [0, 1] relative to its own range, averages them,
    smooths the result, computes the derivative, and finds the first index
    where the magnitude of the derivative drops below the threshold.
    """
    # Normalize each trace to its own baseline (first frame) and stack
    normed = []
    for trace in traces_dict.values():
        baseline = trace[0]
        if baseline > 0:
            normed.append(trace / baseline)
    normed = np.array(normed)

    # Average across all animals
    avg = normed.mean(axis=0)

    # Smooth to reduce noise
    avg_smooth = uniform_filter1d(avg, size=smooth_window)

    # Compute derivative (rate of change per frame)
    deriv = np.diff(avg_smooth)

    # Find first index where derivative is no longer strongly negative
    # (i.e., the steep bleaching decay has ended)
    for i in range(len(deriv)):
        if deriv[i] > -deriv_threshold:
            return i

    # Fallback: use frame 50
    return 50


if __name__ == "__main__":
    # Load raw traces
    df = pd.read_csv(TRACES_CSV)
    time_s = df["time_s"].values

    # Extract mean and sum traces per animal
    all_names = BLAC_CAT_DEAD + BLAC
    traces_mean = {}
    traces_sum = {}
    for name in all_names:
        traces_mean[name] = df[f"{name}_mean"].values
        traces_sum[name] = df[f"{name}_sum"].values

    # Determine bleaching cutoff
    cutoff_idx = find_bleach_cutoff(time_s, traces_mean)
    cutoff_time = time_s[cutoff_idx]
    print(f"Bleaching cutoff: frame {cutoff_idx}, time {cutoff_time:.1f} s")

    # ── Plot 1: Full raw traces ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in BLAC_CAT_DEAD:
        ax.plot(time_s, traces_mean[name], color=COLOR_DEAD, alpha=0.4, linewidth=0.8)
    for name in BLAC:
        ax.plot(time_s, traces_mean[name], color=COLOR_ALIVE, alpha=0.4, linewidth=0.8)

    # Group means
    dead_group = np.array([traces_mean[n] for n in BLAC_CAT_DEAD])
    alive_group = np.array([traces_mean[n] for n in BLAC])
    ax.plot(time_s, dead_group.mean(axis=0), color=COLOR_DEAD, linewidth=2.5,
            label="BlaC-Cat.Dead")
    ax.plot(time_s, alive_group.mean(axis=0), color=COLOR_ALIVE, linewidth=2.5,
            label="BlaC")

    # Mark cutoff
    ax.axvline(cutoff_time, color="gray", linestyle="--", linewidth=1.5,
               label=f"Bleach cutoff ({cutoff_time:.1f} s)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean fluorescence intensity")
    ax.set_title("Raw Calcium Traces — Full Timeseries")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "raw_traces_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved raw_traces_full.png")

    # ── Plot 2: Raw traces after bleaching cutoff ─────────────────────────
    time_trimmed = time_s[cutoff_idx:]
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in BLAC_CAT_DEAD:
        ax.plot(time_trimmed, traces_mean[name][cutoff_idx:],
                color=COLOR_DEAD, alpha=0.4, linewidth=0.8)
    for name in BLAC:
        ax.plot(time_trimmed, traces_mean[name][cutoff_idx:],
                color=COLOR_ALIVE, alpha=0.4, linewidth=0.8)

    # Group means
    ax.plot(time_trimmed, dead_group.mean(axis=0)[cutoff_idx:],
            color=COLOR_DEAD, linewidth=2.5, label="BlaC-Cat.Dead")
    ax.plot(time_trimmed, alive_group.mean(axis=0)[cutoff_idx:],
            color=COLOR_ALIVE, linewidth=2.5, label="BlaC")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean fluorescence intensity")
    ax.set_title(f"Raw Calcium Traces — After Bleaching Cutoff ({cutoff_time:.1f} s, frame {cutoff_idx})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "raw_traces_post_bleach.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved raw_traces_post_bleach.png")

    # ── Plot 3: Full sum traces ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in BLAC_CAT_DEAD:
        ax.plot(time_s, traces_sum[name], color=COLOR_DEAD, alpha=0.4, linewidth=0.8)
    for name in BLAC:
        ax.plot(time_s, traces_sum[name], color=COLOR_ALIVE, alpha=0.4, linewidth=0.8)

    dead_group_sum = np.array([traces_sum[n] for n in BLAC_CAT_DEAD])
    alive_group_sum = np.array([traces_sum[n] for n in BLAC])
    ax.plot(time_s, dead_group_sum.mean(axis=0), color=COLOR_DEAD, linewidth=2.5,
            label="BlaC-Cat.Dead")
    ax.plot(time_s, alive_group_sum.mean(axis=0), color=COLOR_ALIVE, linewidth=2.5,
            label="BlaC")

    ax.axvline(cutoff_time, color="gray", linestyle="--", linewidth=1.5,
               label=f"Bleach cutoff ({cutoff_time:.1f} s)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sum fluorescence intensity")
    ax.set_title("Raw Calcium Traces (Sum) — Full Timeseries")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "raw_traces_sum_full.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved raw_traces_sum_full.png")

    # ── Plot 4: Sum traces after bleaching cutoff ─────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in BLAC_CAT_DEAD:
        ax.plot(time_trimmed, traces_sum[name][cutoff_idx:],
                color=COLOR_DEAD, alpha=0.4, linewidth=0.8)
    for name in BLAC:
        ax.plot(time_trimmed, traces_sum[name][cutoff_idx:],
                color=COLOR_ALIVE, alpha=0.4, linewidth=0.8)

    ax.plot(time_trimmed, dead_group_sum.mean(axis=0)[cutoff_idx:],
            color=COLOR_DEAD, linewidth=2.5, label="BlaC-Cat.Dead")
    ax.plot(time_trimmed, alive_group_sum.mean(axis=0)[cutoff_idx:],
            color=COLOR_ALIVE, linewidth=2.5, label="BlaC")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Sum fluorescence intensity")
    ax.set_title(f"Raw Calcium Traces (Sum) — After Bleaching Cutoff ({cutoff_time:.1f} s, frame {cutoff_idx})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "raw_traces_sum_post_bleach.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved raw_traces_sum_post_bleach.png")
