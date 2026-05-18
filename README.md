# Quick Whole-Brain Overview

Quick analysis of whole-brain calcium imaging data comparing BlaC vs BlaC-Catalytically Dead conditions in *C. elegans*.

## Scripts

Two analysis scripts run the same pipeline on two different datasets:

| Script | Dataset |
|---|---|
| `analyze_calcium_traces_preliminarydata.py` | Preliminary dataset (March imaging) — `/home/lauren/wholistic_preprocessing/preprocessed/` |
| `analyze_calcium_traces_finaldata.py` | Final dataset — `/store1/lauren/Tetramisole_Immobilized_Imaging/2026_cAMP_wholebrain_with_pdfr1_BlaC_Immobilized/ImagedImmediately/pre-processed/` |

The analysis logic is byte-identical between the two — only the data-source path and the animal-to-condition mapping at the top differ. Outputs from the two scripts coexist in the same `outputs/` directory; final-dataset outputs are suffixed `_combined`.

Plus one standalone helper:

| Script | Purpose |
|---|---|
| `make_qc_videos.py finaldata` / `preliminarydata` | Generates per-animal MP4 with the per-frame ROI outlined. Reads cache files from the chosen analysis to use matching ROI parameters. |

## What the analysis does

1. **Validates image dimensions** — raises if not every animal in the dataset shares the same `(Y, X)` shape.
2. **Per-frame fluid ROI** — for each frame's Z-MIP, selects the top `TOP_PERCENT` (default 5%) brightest pixels. Same pixel count in every frame, every animal.
3. **Extracts mean and summed fluorescence traces** — averaged over the per-frame ROI.
4. **Caches per-animal results** — `outputs/cache/{animal}.npz` stores traces + QC arrays keyed by a params hash. Re-runs only recompute animals whose cache is missing or stale.
5. **Drops first `BLEACH_CUTOFF_FRAME` frames** (default 50) for stats.
6. **Compares BlaC vs BlaC-Cat.Dead** with Mann-Whitney on post-bleach raw mean intensity.

There is **no** bleach-curve fitting. Bleach is handled by simply discarding the bleach-transient frames.

## Data

- **Shape:** `(800, 2, 80, 210, 322)` = (T, C, Z, Y, X), ~1.7 Hz
- **Channel 0:** Nuclear calcium sensor (GCaMP) — bleaches significantly, hence the first-50-frames cutoff
- **Channel 1:** mCherry — expressed in subset of neurons + muscles

### Condition mapping

| Condition | Preliminary (file list) | Final (filename strain match) |
|---|---|---|
| BlaC-Cat.Dead (control) | `2026-03-03-01..03` (-04 excluded) | filenames containing `SWF1578` |
| BlaC (experimental) | `2026-03-03-05`, `2026-03-04-02..05` | filenames containing `SWF1555` |

## Running

```bash
cd /home/lauren/quick_wholebrain_overview

# Run analysis (cold run computes all animals; warm run uses cache)
uv run python analyze_calcium_traces_finaldata.py
uv run python analyze_calcium_traces_preliminarydata.py

# Generate QC videos separately (slow; can run any time after analysis)
uv run python make_qc_videos.py finaldata
uv run python make_qc_videos.py preliminarydata
```

### Adding a new animal

1. Drop the new `.zarr` into the appropriate `preprocessed/` directory.
2. Re-run the relevant analysis script. Existing animals print `cache hit` and skip; only the new animal computes.

### Invalidating the cache

Change any value in `cache_params()` (e.g. `TOP_PERCENT`) or bump `CACHE_VERSION`. The next run treats every animal as stale and recomputes.

## Outputs

```
outputs/
  cache/                                # per-animal results, keyed by params hash
    {animal}.npz                        # mean_trace, sum_trace, avg_mip,
                                        # mask_freq, snapshot MIPs+masks at
                                        # frames 50/400/799, params_hash

  traces/
    raw_traces.csv                      # preliminary
    raw_traces_combined.csv             # final

  plots/
    mask_frequency.png                  # preliminary: avg-MIP + heatmap of
    mask_frequency_combined.png         # final:        fraction of frames each
                                        #               pixel was in the top-N
    raw_traces_{mean,sum}[_combined].png
    raw_traces_{mean,sum}_post_bleach[_combined].png
    comparison_{mean,sum}_sem[_combined].png
    dotplot_post_bleach_raw[_combined].png
    snapshots/
      snapshots_{animal}.png            # MIP + mask outline at frames 50/400/799

  stats/
    post_bleach_raw_stats.csv           # preliminary (per-animal + group stat)
    post_bleach_raw_stats_combined.csv  # final

  videos/                               # populated only by make_qc_videos.py
    qc_roi_{animal}.mp4                 # per-frame top-N ROI outlined in green
```

## Key parameters (top of each analysis script)

| Constant | Default | Meaning |
|---|---|---|
| `TOP_PERCENT` | `5` | Fluid ROI = top N% of pixels in each frame's MIP |
| `BLEACH_CUTOFF_FRAME` | `50` | Frames < this are dropped for stats; full traces still plotted with a cutoff line |
| `CALCIUM_CHANNEL` | `0` | Which channel is the calcium sensor |
| `MASK_SNAPSHOT_FRAMES` | `[50, 400, 799]` | Frames for which per-animal mask snapshot plots are saved |
| `CACHE_VERSION` | `1` | Bump to invalidate every cached `.npz` |
