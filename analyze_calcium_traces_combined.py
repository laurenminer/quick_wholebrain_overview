"""
Combined whole-brain calcium trace analysis.

Combines preliminary (March) and new (April) datasets, extracts fluorescence
traces, corrects for photobleaching using ALL animals, and compares BlaC vs
BlaC-Catalytically Dead conditions with post-bleach raw statistics.
"""

import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.stats import median_abs_deviation, mannwhitneyu, sem

# ── Configuration ──────────────────────────────────────────────────────────────

PREPROCESSED_DIR_PRELIMINARY = Path("/home/lauren/wholistic_preprocessing/preprocessed")
PREPROCESSED_DIR_NEW = Path(
    "/store1/lauren/Tetramisole_Immobilized_Imaging/"
    "2026_cAMP_wholebrain_with_pdfr1_BlaC_Immobilized/ImagedImmediately/pre-processed"
)

OUTPUT_DIR = Path("/home/lauren/quick_wholebrain_overview/outputs")
FRAME_RATE_HZ = 1.7
N_FRAMES = 800
BLEACH_CUTOFF_FRAME = 50

ANIMALS = [
    # Preliminary dataset — BlaC-Cat.Dead
    {"name": "2026-03-03-01", "condition": "BlaC-Cat.Dead", "dataset": "preliminary", "dir": PREPROCESSED_DIR_PRELIMINARY},
    {"name": "2026-03-03-02", "condition": "BlaC-Cat.Dead", "dataset": "preliminary", "dir": PREPROCESSED_DIR_PRELIMINARY},
    {"name": "2026-03-03-03", "condition": "BlaC-Cat.Dead", "dataset": "preliminary", "dir": PREPROCESSED_DIR_PRELIMINARY},
    # Preliminary dataset — BlaC
    {"name": "2026-03-03-05", "condition": "BlaC", "dataset": "preliminary", "dir": PREPROCESSED_DIR_PRELIMINARY},
    {"name": "2026-03-04-02", "condition": "BlaC", "dataset": "preliminary", "dir": PREPROCESSED_DIR_PRELIMINARY},
    {"name": "2026-03-04-03", "condition": "BlaC", "dataset": "preliminary", "dir": PREPROCESSED_DIR_PRELIMINARY},
    {"name": "2026-03-04-04", "condition": "BlaC", "dataset": "preliminary", "dir": PREPROCESSED_DIR_PRELIMINARY},
    {"name": "2026-03-04-05", "condition": "BlaC", "dataset": "preliminary", "dir": PREPROCESSED_DIR_PRELIMINARY},
    # New dataset — BlaC-Cat.Dead
    {"name": "20260409_SWF1578_1", "condition": "BlaC-Cat.Dead", "dataset": "new", "dir": PREPROCESSED_DIR_NEW},
    {"name": "20260409_SWF1578_2", "condition": "BlaC-Cat.Dead", "dataset": "new", "dir": PREPROCESSED_DIR_NEW},
    {"name": "20260409_SWF1578_3", "condition": "BlaC-Cat.Dead", "dataset": "new", "dir": PREPROCESSED_DIR_NEW},
    {"name": "20260409_SWF1578_4", "condition": "BlaC-Cat.Dead", "dataset": "new", "dir": PREPROCESSED_DIR_NEW},
    {"name": "20260409_SWF1578_5", "condition": "BlaC-Cat.Dead", "dataset": "new", "dir": PREPROCESSED_DIR_NEW},
    # New dataset — BlaC
    {"name": "20260409_SWF1555_1", "condition": "BlaC", "dataset": "new", "dir": PREPROCESSED_DIR_NEW},
    {"name": "20260409_SWF1555_2", "condition": "BlaC", "dataset": "new", "dir": PREPROCESSED_DIR_NEW},
    {"name": "20260409_SWF1555_3", "condition": "BlaC", "dataset": "new", "dir": PREPROCESSED_DIR_NEW},
    {"name": "20260409_SWF1555_4", "condition": "BlaC", "dataset": "new", "dir": PREPROCESSED_DIR_NEW},
    {"name": "20260409_SWF1555_5", "condition": "BlaC", "dataset": "new", "dir": PREPROCESSED_DIR_NEW},
    {"name": "20260409_SWF1555_6", "condition": "BlaC", "dataset": "new", "dir": PREPROCESSED_DIR_NEW},
]

all_names = [a["name"] for a in ANIMALS]
blac_cat_dead = [a["name"] for a in ANIMALS if a["condition"] == "BlaC-Cat.Dead"]
blac = [a["name"] for a in ANIMALS if a["condition"] == "BlaC"]
animal_dir = {a["name"]: a["dir"] for a in ANIMALS}
animal_condition = {a["name"]: a["condition"] for a in ANIMALS}
animal_dataset = {a["name"]: a["dataset"] for a in ANIMALS}

CALCIUM_CHANNEL = 0
THRESHOLD_FRAMES = 10
THRESHOLD_K = 4

COLOR_DEAD = "#4477AA"
COLOR_ALIVE = "#CC3311"


# ── Helper functions ───────────────────────────────────────────────────────────

def open_zarr(name: str, preprocessed_dir: Path):
    path = str(preprocessed_dir / f"{name}.zarr")
    return zarr.open(path, mode="r")


def compute_mip(arr, t: int, channel: int) -> np.ndarray:
    vol = np.asarray(arr[t, channel])
    return vol.max(axis=0).astype(np.float32)


def compute_threshold_mask(arr, channel: int, n_frames: int = 10, k: float = 4.0) -> np.ndarray:
    mip_sum = np.zeros((arr.shape[3], arr.shape[4]), dtype=np.float64)
    for t in range(min(n_frames, arr.shape[0])):
        mip_sum += compute_mip(arr, t, channel)
    avg_mip = (mip_sum / min(n_frames, arr.shape[0])).astype(np.float32)

    med = np.median(avg_mip)
    mad = median_abs_deviation(avg_mip, axis=None)
    threshold = med + k * mad
    mask = avg_mip > threshold

    signal_frac = mask.sum() / mask.size
    print(f"    Threshold: {threshold:.1f} (median={med:.1f}, MAD={mad:.1f})")
    print(f"    Signal pixels: {mask.sum()} / {mask.size} ({signal_frac:.1%})")

    return mask, avg_mip


def extract_trace(arr, mask: np.ndarray, channel: int):
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



def make_qc_video(arr, mask: np.ndarray, name: str, channel: int):
    from skimage.segmentation import find_boundaries

    n_t = arr.shape[0]
    outline = find_boundaries(mask, mode="thick")

    sample_indices = [0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]
    all_vals = []
    for si in sample_indices:
        mip = compute_mip(arr, si, channel)
        all_vals.append(mip.ravel())
    all_vals = np.concatenate(all_vals)
    p1, p99 = np.percentile(all_vals, [0.5, 99.5])

    n_y, n_x = arr.shape[3], arr.shape[4]
    h_pad = n_y + (n_y % 2)
    w_pad = n_x + (n_x % 2)

    out_path = str(OUTPUT_DIR / "videos" / f"qc_roi_{name}.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w_pad * 3}x{h_pad}",
        "-pix_fmt", "rgb24",
        "-r", "7",
        "-i", "-",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for t in range(n_t):
        mip = compute_mip(arr, t, channel)
        norm = np.clip((mip - p1) / max(p99 - p1, 1e-8), 0, 1)
        gray = (np.power(norm, 0.5) * 255).astype(np.uint8)

        frame = np.zeros((h_pad, w_pad, 3), dtype=np.uint8)
        frame[:n_y, :n_x, 0] = gray
        frame[:n_y, :n_x, 1] = gray
        frame[:n_y, :n_x, 2] = gray

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
    for subdir in ["traces", "plots", "videos", "stats"]:
        (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

    time_s = np.arange(N_FRAMES) / FRAME_RATE_HZ
    print(f"  Total animals: {len(all_names)} ({len(blac_cat_dead)} Dead, {len(blac)} BlaC)")
    print(f"  Frames: {N_FRAMES}, Bleach cutoff: frame {BLEACH_CUTOFF_FRAME}")

    # ── Step 1: Extract traces ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1: Extract calcium traces")
    print("=" * 60)

    traces_mean = {}
    traces_sum = {}
    masks = {}
    avg_mips = {}

    for name in all_names:
        cond = animal_condition[name]
        ds = animal_dataset[name]
        print(f"\n  {name} ({cond}, {ds})")
        arr = open_zarr(name, animal_dir[name])
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

    for idx, name in enumerate(all_names):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        avg_mip = avg_mips[name]
        mask = masks[name]

        ax.imshow(avg_mip, cmap="gray", vmin=np.percentile(avg_mip, 1),
                  vmax=np.percentile(avg_mip, 99.5))
        from skimage.segmentation import find_boundaries
        boundary = find_boundaries(mask, mode="thick")
        overlay = np.zeros((*avg_mip.shape, 4))
        overlay[boundary] = [0, 1, 0, 0.8]
        ax.imshow(overlay)
        cond = animal_condition[name]
        ds = animal_dataset[name]
        ax.set_title(f"{name}\n({cond}, {ds})", fontsize=8)
        ax.axis("off")

    for idx in range(n_animals, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    plt.suptitle("Threshold Masks (green outline) on Averaged Early MIPs — Combined", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "threshold_masks_combined.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved threshold_masks_combined.png")

    # ── Step 3: QC videos ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 3: QC videos with ROI overlay")
    print("=" * 60)

    for name in all_names:
        print(f"\n  {name}")
        arr = open_zarr(name, animal_dir[name])
        make_qc_video(arr, masks[name], name, CALCIUM_CHANNEL)

    # ── Step 4: Save raw trace CSVs ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 4: Save trace data")
    print("=" * 60)

    raw_data = {"time_s": time_s}
    for name in all_names:
        raw_data[f"{name}_mean"] = traces_mean[name]
        raw_data[f"{name}_sum"] = traces_sum[name]
    pd.DataFrame(raw_data).to_csv(OUTPUT_DIR / "traces" / "raw_traces_combined.csv", index=False)
    print("  Saved raw_traces_combined.csv")

    # ── Step 5: Plot raw traces (full + post-bleach) ──────────────────────
    print("\n" + "=" * 60)
    print("  STEP 5: Plotting raw traces")
    print("=" * 60)

    time_post = time_s[BLEACH_CUTOFF_FRAME:]

    for metric_name, raw_traces in [("mean", traces_mean), ("sum", traces_sum)]:
        dead_group = np.array([raw_traces[n] for n in blac_cat_dead])
        alive_group = np.array([raw_traces[n] for n in blac])

        # Full raw traces with cutoff line
        fig, ax = plt.subplots(figsize=(12, 5))
        for name in blac_cat_dead:
            ax.plot(time_s, raw_traces[name], color=COLOR_DEAD, alpha=0.3, linewidth=0.8)
        for name in blac:
            ax.plot(time_s, raw_traces[name], color=COLOR_ALIVE, alpha=0.3, linewidth=0.8)
        ax.plot(time_s, dead_group.mean(axis=0), color=COLOR_DEAD, linewidth=2.5,
                label="BlaC-Cat.Dead")
        ax.plot(time_s, alive_group.mean(axis=0), color=COLOR_ALIVE, linewidth=2.5,
                label="BlaC")
        ax.axvline(time_s[BLEACH_CUTOFF_FRAME], color="orange", linestyle=":", linewidth=1.5,
                   label=f"Bleach cutoff (frame {BLEACH_CUTOFF_FRAME})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{metric_name.capitalize()} intensity")
        ax.set_title(f"Raw {metric_name.capitalize()} Intensity Traces — Combined")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plots" / f"raw_traces_{metric_name}_combined.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # Post-bleach raw traces only
        fig, ax = plt.subplots(figsize=(12, 5))
        for name in blac_cat_dead:
            ax.plot(time_post, raw_traces[name][BLEACH_CUTOFF_FRAME:],
                    color=COLOR_DEAD, alpha=0.3, linewidth=0.8)
        for name in blac:
            ax.plot(time_post, raw_traces[name][BLEACH_CUTOFF_FRAME:],
                    color=COLOR_ALIVE, alpha=0.3, linewidth=0.8)
        ax.plot(time_post, dead_group.mean(axis=0)[BLEACH_CUTOFF_FRAME:],
                color=COLOR_DEAD, linewidth=2.5, label="BlaC-Cat.Dead")
        ax.plot(time_post, alive_group.mean(axis=0)[BLEACH_CUTOFF_FRAME:],
                color=COLOR_ALIVE, linewidth=2.5, label="BlaC")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{metric_name.capitalize()} intensity")
        ax.set_title(f"Raw {metric_name.capitalize()} — Post-Bleach (frame {BLEACH_CUTOFF_FRAME}+) — Combined")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plots" / f"raw_traces_{metric_name}_post_bleach_combined.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        # Comparison: mean ± SEM (raw, post-bleach)
        fig, ax = plt.subplots(figsize=(12, 5))
        dead_m = dead_group.mean(axis=0)[BLEACH_CUTOFF_FRAME:]
        dead_se = sem(dead_group, axis=0)[BLEACH_CUTOFF_FRAME:]
        alive_m = alive_group.mean(axis=0)[BLEACH_CUTOFF_FRAME:]
        alive_se = sem(alive_group, axis=0)[BLEACH_CUTOFF_FRAME:]

        ax.plot(time_post, dead_m, color=COLOR_DEAD, linewidth=2, label="BlaC-Cat.Dead")
        ax.fill_between(time_post, dead_m - dead_se, dead_m + dead_se,
                        color=COLOR_DEAD, alpha=0.2)
        ax.plot(time_post, alive_m, color=COLOR_ALIVE, linewidth=2, label="BlaC")
        ax.fill_between(time_post, alive_m - alive_se, alive_m + alive_se,
                        color=COLOR_ALIVE, alpha=0.2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"Raw {metric_name} intensity")
        ax.set_title(f"BlaC vs BlaC-Cat.Dead — Raw {metric_name.capitalize()} ± SEM Post-Bleach (Combined)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plots" / f"comparison_{metric_name}_sem_combined.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

    print("  Saved all trace plots")

    # ── Step 8: Post-bleach raw statistics ─────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 8: Post-bleach raw statistics")
    print("=" * 60)

    post_bleach_averages = {}
    for name in all_names:
        post_bleach_averages[name] = traces_mean[name][BLEACH_CUTOFF_FRAME:].mean()

    blac_vals = [post_bleach_averages[n] for n in blac]
    dead_vals = [post_bleach_averages[n] for n in blac_cat_dead]

    u_stat, p_mann = mannwhitneyu(blac_vals, dead_vals, alternative="two-sided")
    print(f"  BlaC (n={len(blac)}): {np.mean(blac_vals):.2f} ± {sem(blac_vals):.2f}")
    print(f"  Dead (n={len(blac_cat_dead)}): {np.mean(dead_vals):.2f} ± {sem(dead_vals):.2f}")
    print(f"  Mann-Whitney U={u_stat:.1f}, p={p_mann:.4f}")

    # CSV
    stats_rows = []
    for a in ANIMALS:
        stats_rows.append({
            "animal": a["name"],
            "condition": a["condition"],
            "dataset": a["dataset"],
            "post_bleach_mean_raw": post_bleach_averages[a["name"]],
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df["mannwhitney_U"] = u_stat
    stats_df["mannwhitney_p"] = p_mann
    stats_df["n_blac"] = len(blac)
    stats_df["n_blac_cat_dead"] = len(blac_cat_dead)
    stats_df.to_csv(OUTPUT_DIR / "stats" / "post_bleach_raw_stats_combined.csv", index=False)
    print("  Saved post_bleach_raw_stats_combined.csv")

    # Dot plot
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(6, 6))

    for name in blac_cat_dead:
        marker = "o" if animal_dataset[name] == "preliminary" else "^"
        jitter = np.random.uniform(-0.12, 0.12)
        ax.plot(0 + jitter, post_bleach_averages[name],
                marker, color=COLOR_DEAD, markersize=9, alpha=0.8,
                markeredgecolor="white", markeredgewidth=0.5)

    for name in blac:
        marker = "o" if animal_dataset[name] == "preliminary" else "^"
        jitter = np.random.uniform(-0.12, 0.12)
        ax.plot(1 + jitter, post_bleach_averages[name],
                marker, color=COLOR_ALIVE, markersize=9, alpha=0.8,
                markeredgecolor="white", markeredgewidth=0.5)

    ax.hlines(np.mean(dead_vals), -0.25, 0.25, color=COLOR_DEAD, linewidth=2.5)
    ax.hlines(np.mean(blac_vals), 0.75, 1.25, color=COLOR_ALIVE, linewidth=2.5)

    # Legend for dataset markers
    ax.plot([], [], "o", color="gray", markersize=8, label="Preliminary")
    ax.plot([], [], "^", color="gray", markersize=8, label="New")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BlaC-Cat.Dead", "BlaC"])
    ax.set_ylabel("Post-bleach avg raw mean intensity")
    ax.set_title(f"Post-Bleach Raw Intensity (Mann-Whitney p={p_mann:.4f})")
    ax.set_xlim(-0.5, 1.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "plots" / "dotplot_post_bleach_raw_combined.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved dotplot_post_bleach_raw_combined.png")

    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)
    print(f"  Outputs in: {OUTPUT_DIR}")
