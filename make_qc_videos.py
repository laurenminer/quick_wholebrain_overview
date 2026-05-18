"""
Standalone QC video generator.

For each animal in the requested dataset, reads the zarr and writes
outputs/videos/qc_roi_{animal}.mp4 — a grayscale MIP movie with the
per-frame fluid top-N mask outlined in green. Uses the same per-frame
top-N selection as the analysis scripts, so the videos match exactly what
the analysis used.

Usage:
    python make_qc_videos.py finaldata
    python make_qc_videos.py preliminarydata

Skips animals whose video already exists. Delete an MP4 to force
regeneration for that animal.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import zarr
from skimage.segmentation import find_boundaries


def compute_mip(arr, t: int, channel: int) -> np.ndarray:
    vol = np.asarray(arr[t, channel])
    return vol.max(axis=0).astype(np.float32)


def make_video(arr, name: str, channel: int, n_top: int, out_path: Path):
    n_t = arr.shape[0]
    n_y, n_x = arr.shape[3], arr.shape[4]

    sample_indices = [0, n_t // 4, n_t // 2, 3 * n_t // 4, n_t - 1]
    all_vals = []
    for si in sample_indices:
        all_vals.append(compute_mip(arr, si, channel).ravel())
    all_vals = np.concatenate(all_vals)
    p1, p99 = np.percentile(all_vals, [0.5, 99.5])

    h_pad = n_y + (n_y % 2)
    w_pad = n_x + (n_x % 2)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w_pad * 3}x{h_pad}",
        "-pix_fmt", "rgb24",
        "-r", "7",
        "-i", "-",
        "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        for t in range(n_t):
            mip = compute_mip(arr, t, channel)
            flat = mip.ravel()
            idx = np.argpartition(flat, -n_top)[-n_top:]
            mask = np.zeros((n_y, n_x), dtype=bool)
            mask.ravel()[idx] = True
            outline = find_boundaries(mask, mode="thick")

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
                print(f"    Frame {t + 1}/{n_t}")
    finally:
        proc.stdin.close()
        proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["finaldata", "preliminarydata"],
                        help="Which analysis script's output to use.")
    args = parser.parse_args()

    if args.dataset == "finaldata":
        from analyze_calcium_traces_finaldata import (
            OUTPUT_DIR, all_names, animal_dir, cache_path,
        )
    else:
        from analyze_calcium_traces_preliminarydata import (
            OUTPUT_DIR, all_names, animal_dir, cache_path,
        )

    videos_dir = OUTPUT_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Dataset: {args.dataset}")
    print(f"  Animals: {len(all_names)}")
    print(f"  Output:  {videos_dir}")

    for name in all_names:
        out_path = videos_dir / f"qc_roi_{name}.mp4"
        if out_path.exists():
            print(f"  {name}: video exists, skipping")
            continue

        cp = cache_path(name)
        if not cp.exists():
            print(f"  {name}: no cache at {cp}; run the analysis script first. Skipping.")
            continue

        cached = np.load(cp, allow_pickle=True)
        import json
        params = json.loads(str(cached["params_json"]))
        n_top = int(params["n_top_pixels"])
        channel = int(params["channel"])

        zarr_path = animal_dir[name] / f"{name}.zarr"
        if not zarr_path.exists():
            print(f"  {name}: zarr missing at {zarr_path}; skipping.")
            continue

        print(f"\n  {name}: writing video (n_top={n_top}, channel={channel})")
        arr = zarr.open(str(zarr_path), mode="r")
        make_video(arr, name, channel, n_top, out_path)
        print(f"  {name}: saved {out_path}")


if __name__ == "__main__":
    main()
