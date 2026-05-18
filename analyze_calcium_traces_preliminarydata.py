"""
Quick whole-brain calcium trace analysis.

Segments calcium signal from Z-MIPs of unregistered zarr data, extracts
fluorescence traces, corrects for photobleaching using BlaC-Catalytically Dead
controls, and compares BlaC vs BlaC-Catalytically Dead conditions.
"""

import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.optimize import curve_fit
from scipy.stats import median_abs_deviation, ttest_ind, mannwhitneyu, sem

# ── Configuration ──────────────────────────────────────────────────────────────

PREPROCESSED_DIR = Path("/home/lauren/wholistic_preprocessing/preprocessed")
OUTPUT_DIR = Path("/home/lauren/quick_wholebrain_overview/outputs")
FRAME_RATE_HZ = 1.7

# Condition mapping
BLAC_CAT_DEAD = ["2026-03-03-01", "2026-03-03-02", "2026-03-03-03", "2026-03-03-04"]
BLAC = ["2026-03-03-05", "2026-03-04-02", "2026-03-04-03", "2026-03-04-04", "2026-03-04-05"]

# Animals to exclude from analysis (e.g. bad data)
EXCLUDE = ["2026-03-03-04"]

CALCIUM_CHANNEL = 0
THRESHOLD_FRAMES = 10   # number of early frames to average for threshold mask
THRESHOLD_K = 4          # threshold = median + k * MAD

# Colors
COLOR_DEAD = "#4477AA"
COLOR_ALIVE = "#CC3311"


# ── Helper functions ───────────────────────────────────────────────────────────

def open_zarr(name: str):
    """Open a preprocessed zarr array in read mode."""
    path = str(PREPROCESSED_DIR / f"{name}.zarr")
    return zarr.open(path, mode="r")


def compute_mip(arr, t: int, channel: int) -> np.ndarray:
    """Compute Z-axis max intensity projection for one timepoint/channel."""
    vol = np.asarray(arr[t, channel])  # (Z, Y, X)
    return vol.max(axis=0).astype(np.float32)  # (Y, X)


def compute_threshold_mask(arr, channel: int, n_frames: int = 10, k: float = 4.0) -> np.ndarray:
    """
    Compute a binary mask of signal pixels from the average of early MIPs.

    Uses median + k * MAD thresholding. The median is a robust estimate of
    background since the worm head occupies a minority of the FOV.
    """
    # Average MIPs from first n_frames for a cleaner reference
    mip_sum = np.zeros((arr.shape[3], arr.shape[4]), dtype=np.float64)
    for t in range(min(n_frames, arr.shape[0])):
        mip_sum += compute_mip(arr, t, channel)
    avg_mip = (mip_sum / min(n_frames, arr.shape[0])).astype(np.float32)

    # Threshold using median + k * MAD
    med = np.median(avg_mip)
    mad = median_abs_deviation(avg_mip, axis=None)
    threshold = med + k * mad
    mask = avg_mip > threshold

    signal_frac = mask.sum() / mask.size
    print(f"    Threshold: {threshold:.1f} (median={med:.1f}, MAD={mad:.1f})")
    print(f"    Signal pixels: {mask.sum()} / {mask.size} ({signal_frac:.1%})")

    return mask, avg_mip


def extract_trace(arr, mask: np.ndarray, channel: int):
    """Extract mean and sum intensity traces over all timepoints."""
    n_t = arr.shape[0]
    mean_trace = np.zeros(n_t, dtype=np.float64)
    sum_trace = np.zeros(n_t, dtype=np.float64)

    n_signal = mask.sum()
    if n_signal == 0:
        print("    WARNING: mask has no signal pixels!")
        return mean_trace, sum_trace

    for t in range(n_t):
        mip = compute_mip(arr, t, channel)
        vals = mip[mask]
        mean_trace[t] = vals.mean()
        sum_trace[t] = vals.sum()

        if (t + 1) % 200 == 0:
            print(f"    Frame {t + 1}/{n_t}")

    return mean_trace, sum_trace


def double_exp(t, a, b, c, d, e):
    """Double exponential decay: a*exp(-b*t) + c*exp(-d*t) + e"""
    return a * np.exp(-b * t) + c * np.exp(-d * t) + e


def single_exp(t, a, b, c):
    """Single exponential decay: a*exp(-b*t) + c"""
    return a * np.exp(-b * t) + c


def fit_bleach_curve(time_s: np.ndarray, trace: np.ndarray):
    """
    Fit a double exponential to a normalized bleach trace.
    Falls back to single exponential if double doesn't converge.
    Returns (fitted_values, fit_func, params, model_name).
    """
    # Try double exponential first
    try:
        p0 = [0.3, 0.05, 0.5, 0.005, 0.2]
        bounds = ([0, 0, 0, 0, 0], [np.inf, 1.0, np.inf, 0.1, np.inf])
        popt, _ = curve_fit(double_exp, time_s, trace, p0=p0, bounds=bounds, maxfev=10000)
        fitted = double_exp(time_s, *popt)
        print(f"    Double exp fit: a={popt[0]:.3f}, b={popt[1]:.4f}, "
              f"c={popt[2]:.3f}, d={popt[3]:.5f}, e={popt[4]:.3f}")
        return fitted, double_exp, popt, "double_exp"
    except RuntimeError:
        print("    Double exp failed, trying single exp...")

    # Fallback to single exponential
    p0 = [0.5, 0.01, 0.5]
    bounds = ([0, 0, 0], [np.inf, 1.0, np.inf])
    popt, _ = curve_fit(single_exp, time_s, trace, p0=p0, bounds=bounds, maxfev=10000)
    fitted = single_exp(time_s, *popt)
    print(f"    Single exp fit: a={popt[0]:.3f}, b={popt[1]:.4f}, c={popt[2]:.3f}")
    return fitted, single_exp, popt, "single_exp"


def make_qc_video(arr, mask: np.ndarray, name: str, channel: int):
    """
    Generate MP4 of channel MIP over time with ROI outline overlaid.
    """
    from skimage.segmentation import find_boundaries

    n_t = arr.shape[0]
    outline = find_boundaries(mask, mode="thick")

    # Compute contrast from sampled MIPs
    sample_indices = [0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]
    all_vals = []
    for si in sample_indices:
        mip = compute_mip(arr, si, channel)
        all_vals.append(mip.ravel())
    all_vals = np.concatenate(all_vals)
    p1, p99 = np.percentile(all_vals, [0.5, 99.5])

    # Video dimensions (pad to even for h264)
    n_y, n_x = arr.shape[3], arr.shape[4]
    h_pad = n_y + (n_y % 2)
    w_pad = n_x + (n_x % 2)

    out_path = str(OUTPUT_DIR / "videos" / f"qc_roi_{name}.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w_pad * 3}x{h_pad}",  # 3 channels (RGB)
        "-pix_fmt", "rgb24",
        "-r", "7",
        "-i", "-",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for t in range(n_t):
        mip = compute_mip(arr, t, channel)

        # Normalize with gamma correction
        norm = np.clip((mip - p1) / max(p99 - p1, 1e-8), 0, 1)
        gray = (np.power(norm, 0.5) * 255).astype(np.uint8)

        # Create RGB frame: grayscale with green ROI outline
        frame = np.zeros((h_pad, w_pad, 3), dtype=np.uint8)
        frame[:n_y, :n_x, 0] = gray
        frame[:n_y, :n_x, 1] = gray
        frame[:n_y, :n_x, 2] = gray

        # Draw ROI outline in green
        frame[:n_y, :n_x, 0][outline] = 0
        frame[:n_y, :n_x, 1][outline] = 255
        frame[:n_y, :n_x, 2][outline] = 0

        proc.stdin.write(frame.tobytes())

        if (t + 1) % 200 == 0:
            print(f"    Video frame {t + 1}/{n_t}")

    proc.stdin.close()
    proc.wait()
    print(f"    Saved: {out_path}")


# ── Main execution ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Create output directories
    for subdir in ["traces", "plots", "videos"]:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Apply exclusions
    blac_cat_dead = [n for n in BLAC_CAT_DEAD if n not in EXCLUDE]
    blac = [n for n in BLAC if n not in EXCLUDE]
    all_names = blac_cat_dead + blac
    all_conditions = (["BlaC-Cat.Dead"] * len(blac_cat_dead) +
                      ["BlaC"] * len(blac))
    time_s = np.arange(800) / FRAME_RATE_HZ

    if EXCLUDE:
        print(f"  Excluding: {EXCLUDE}")

    # ── Step 1: Extract traces ─────────────────────────────────────────────
    print("=" * 60)
    print("  STEP 1: Extract calcium traces")
    print("=" * 60)

    traces_mean = {}
    traces_sum = {}
    masks = {}
    avg_mips = {}

    for name, cond in zip(all_names, all_conditions):
        print(f"\n  {name} ({cond})")
        arr = open_zarr(name)
        print(f"    Shape: {arr.shape}")

        mask, avg_mip = compute_threshold_mask(arr, CALCIUM_CHANNEL,
                                                THRESHOLD_FRAMES, THRESHOLD_K)
        masks[name] = mask
        avg_mips[name] = avg_mip

        mean_trace, sum_trace = extract_trace(arr, mask, CALCIUM_CHANNEL)
        traces_mean[name] = mean_trace
        traces_sum[name] = sum_trace
        print(f"    Mean intensity: [{mean_trace.min():.1f}, {mean_trace.max():.1f}]")

    # ── Step 2: QC - Threshold mask visualization ──────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 2: QC - Threshold masks")
    print("=" * 60)

    n_animals = len(all_names)
    ncols = min(5, n_animals)
    nrows = (n_animals + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (name, cond) in enumerate(zip(all_names, all_conditions)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        avg_mip = avg_mips[name]
        mask = masks[name]

        # Show MIP with mask overlay
        ax.imshow(avg_mip, cmap="gray", vmin=np.percentile(avg_mip, 1),
                  vmax=np.percentile(avg_mip, 99.5))
        # Overlay mask boundary
        from skimage.segmentation import find_boundaries
        boundary = find_boundaries(mask, mode="thick")
        overlay = np.zeros((*avg_mip.shape, 4))
        overlay[boundary] = [0, 1, 0, 0.8]
        ax.imshow(overlay)
        ax.set_title(f"{name}\n({cond})", fontsize=9)
        ax.axis("off")

    # Hide empty subplots
    for idx in range(n_animals, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    plt.suptitle("Threshold Masks (green outline) on Averaged Early MIPs", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "threshold_masks.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved threshold_masks.png")

    # ── Step 3: QC videos ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: QC videos with ROI overlay")
    print("=" * 60)

    for name in all_names:
        print(f"\n  {name}")
        arr = open_zarr(name)
        make_qc_video(arr, masks[name], name, CALCIUM_CHANNEL)

    # ── Step 4: Bleach curve fitting ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4: Bleach curve fitting (BlaC-Cat.Dead only)")
    print("=" * 60)

    # Average BlaC-Cat.Dead mean traces
    dead_traces = np.array([traces_mean[n] for n in blac_cat_dead])
    dead_mean = dead_traces.mean(axis=0)

    # Normalize to start at 1.0 (use first 5 frames for robustness)
    baseline = np.mean(dead_mean[:5])
    dead_mean_norm = dead_mean / baseline
    print(f"  Baseline (first 5 frames): {baseline:.1f}")

    # Fit
    fitted_curve, fit_func, fit_params, model_name = fit_bleach_curve(time_s, dead_mean_norm)
    print(f"  Model: {model_name}")

    # Plot bleach fit
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot individual dead traces (normalized)
    for name in blac_cat_dead:
        norm_trace = traces_mean[name] / np.mean(traces_mean[name][:5])
        ax.plot(time_s, norm_trace, color=COLOR_DEAD, alpha=0.3, linewidth=0.8)
    ax.plot(time_s, dead_mean_norm, color=COLOR_DEAD, linewidth=2, label="BlaC-Cat.Dead mean")
    ax.plot(time_s, fitted_curve, color="black", linewidth=2, linestyle="--",
            label=f"Fit ({model_name})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized fluorescence")
    ax.set_title("Bleach Curve Fit (BlaC-Catalytically Dead)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "bleach_fit.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved bleach_fit.png")

    # Save bleach fit data
    bleach_df = pd.DataFrame({
        "time_s": time_s,
        "dead_mean_norm": dead_mean_norm,
        "fitted_curve": fitted_curve,
    })
    bleach_df.to_csv(OUTPUT_DIR / "traces" / "bleach_fit.csv", index=False)

    # ── Step 5: Bleach correction ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5: Bleach correction")
    print("=" * 60)

    # Floor to avoid division by near-zero
    fitted_curve_safe = np.maximum(fitted_curve, 0.01)

    corrected_mean = {}
    corrected_sum = {}
    for name in all_names:
        # Normalize each trace to its own baseline, then correct
        trace_baseline = np.mean(traces_mean[name][:5])
        norm_trace = traces_mean[name] / trace_baseline
        corrected_mean[name] = norm_trace / fitted_curve_safe

        sum_baseline = np.mean(traces_sum[name][:5])
        norm_sum = traces_sum[name] / sum_baseline
        corrected_sum[name] = norm_sum / fitted_curve_safe

    print("  Done.")

    # ── Step 6: Save trace CSVs ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 6: Save trace data")
    print("=" * 60)

    # Raw traces
    raw_data = {"time_s": time_s}
    for name in all_names:
        raw_data[f"{name}_mean"] = traces_mean[name]
        raw_data[f"{name}_sum"] = traces_sum[name]
    pd.DataFrame(raw_data).to_csv(OUTPUT_DIR / "traces" / "raw_traces.csv", index=False)

    # Corrected traces
    corr_data = {"time_s": time_s}
    for name in all_names:
        corr_data[f"{name}_mean"] = corrected_mean[name]
        corr_data[f"{name}_sum"] = corrected_sum[name]
    pd.DataFrame(corr_data).to_csv(OUTPUT_DIR / "traces" / "corrected_traces.csv", index=False)
    print("  Saved raw_traces.csv and corrected_traces.csv")

    # ── Step 7: Plotting ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 7: Plotting")
    print("=" * 60)

    for metric_name, raw_traces, corr_traces in [
        ("mean", traces_mean, corrected_mean),
        ("sum", traces_sum, corrected_sum),
    ]:
        # --- Raw traces ---
        fig, ax = plt.subplots(figsize=(12, 5))
        for name in blac_cat_dead:
            ax.plot(time_s, raw_traces[name], color=COLOR_DEAD, alpha=0.3, linewidth=0.8)
        for name in blac:
            ax.plot(time_s, raw_traces[name], color=COLOR_ALIVE, alpha=0.3, linewidth=0.8)

        dead_group = np.array([raw_traces[n] for n in blac_cat_dead])
        alive_group = np.array([raw_traces[n] for n in blac])
        ax.plot(time_s, dead_group.mean(axis=0), color=COLOR_DEAD, linewidth=2.5,
                label="BlaC-Cat.Dead")
        ax.plot(time_s, alive_group.mean(axis=0), color=COLOR_ALIVE, linewidth=2.5,
                label="BlaC")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{metric_name.capitalize()} intensity")
        ax.set_title(f"Raw {metric_name.capitalize()} Intensity Traces")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plots" / f"raw_traces_{metric_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # --- Corrected traces ---
        fig, ax = plt.subplots(figsize=(12, 5))
        for name in blac_cat_dead:
            ax.plot(time_s, corr_traces[name], color=COLOR_DEAD, alpha=0.3, linewidth=0.8)
        for name in blac:
            ax.plot(time_s, corr_traces[name], color=COLOR_ALIVE, alpha=0.3, linewidth=0.8)

        dead_group = np.array([corr_traces[n] for n in blac_cat_dead])
        alive_group = np.array([corr_traces[n] for n in blac])
        ax.plot(time_s, dead_group.mean(axis=0), color=COLOR_DEAD, linewidth=2.5,
                label="BlaC-Cat.Dead")
        ax.plot(time_s, alive_group.mean(axis=0), color=COLOR_ALIVE, linewidth=2.5,
                label="BlaC")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Corrected {metric_name} intensity")
        ax.set_title(f"Bleach-Corrected {metric_name.capitalize()} Intensity Traces")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plots" / f"corrected_traces_{metric_name}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # --- Comparison: mean ± SEM ---
        fig, ax = plt.subplots(figsize=(12, 5))
        dead_group = np.array([corr_traces[n] for n in blac_cat_dead])
        alive_group = np.array([corr_traces[n] for n in blac])

        dead_m = dead_group.mean(axis=0)
        dead_se = sem(dead_group, axis=0)
        alive_m = alive_group.mean(axis=0)
        alive_se = sem(alive_group, axis=0)

        ax.plot(time_s, dead_m, color=COLOR_DEAD, linewidth=2, label="BlaC-Cat.Dead")
        ax.fill_between(time_s, dead_m - dead_se, dead_m + dead_se,
                        color=COLOR_DEAD, alpha=0.2)
        ax.plot(time_s, alive_m, color=COLOR_ALIVE, linewidth=2, label="BlaC")
        ax.fill_between(time_s, alive_m - alive_se, alive_m + alive_se,
                        color=COLOR_ALIVE, alpha=0.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Corrected {metric_name} intensity")
        ax.set_title(f"BlaC vs BlaC-Cat.Dead — {metric_name.capitalize()} ± SEM (bleach-corrected)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plots" / f"comparison_{metric_name}_sem.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    print("  Saved all trace and comparison plots")

    # ── Step 8: Basic statistics ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 8: Basic statistics")
    print("=" * 60)

    # Use frames 50-750 to avoid edge artifacts
    window = slice(50, 750)

    stats_rows = []
    for metric_name, corr_traces in [("mean", corrected_mean), ("sum", corrected_sum)]:
        alive_vals = [corr_traces[n][window].mean() for n in blac]
        dead_vals = [corr_traces[n][window].mean() for n in blac_cat_dead]

        t_stat, p_ttest = ttest_ind(alive_vals, dead_vals)
        u_stat, p_mann = mannwhitneyu(alive_vals, dead_vals, alternative="two-sided")

        row = {
            "metric": f"{metric_name}_intensity",
            "blac_mean": np.mean(alive_vals),
            "blac_sem": sem(alive_vals),
            "blac_cat_dead_mean": np.mean(dead_vals),
            "blac_cat_dead_sem": sem(dead_vals),
            "t_statistic": t_stat,
            "p_ttest": p_ttest,
            "u_statistic": u_stat,
            "p_mannwhitney": p_mann,
            "n_blac": len(alive_vals),
            "n_blac_cat_dead": len(dead_vals),
        }
        stats_rows.append(row)
        print(f"  {metric_name}: BlaC={np.mean(alive_vals):.3f}±{sem(alive_vals):.3f}, "
              f"Dead={np.mean(dead_vals):.3f}±{sem(dead_vals):.3f}, "
              f"t-test p={p_ttest:.4f}, Mann-Whitney p={p_mann:.4f}")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUTPUT_DIR / "stats" / "summary_statistics.csv", index=False)
    print("  Saved summary_statistics.csv")

    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)
    print(f"  Outputs in: {OUTPUT_DIR}")
