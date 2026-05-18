# Quick Whole-Brain Overview

Quick analysis of whole-brain calcium imaging data comparing BlaC vs BlaC-Catalytically Dead conditions in C. elegans.

## What it does

1. Loads preprocessed zarr files (z-stack timeseries, 2-channel)
2. Computes Z-axis max intensity projections (MIPs) for each timepoint
3. Segments signal pixels using median + MAD thresholding on early frames
4. Extracts mean and summed fluorescence traces over time
5. Fits a bleach curve from BlaC-Catalytically Dead controls (double exponential)
6. Corrects all traces for photobleaching
7. Generates comparison plots and basic statistics (t-test, Mann-Whitney)

## Data

- **Source:** `/home/lauren/wholistic_preprocessing/preprocessed/*.zarr`
- **Shape:** `(800, 2, 80, 210, 322)` = (T, C, Z, Y, X), uint16, ~1.7 Hz
- **Channel 0:** Nuclear calcium sensor (GCaMP) — bleaches significantly
- **Channel 1:** mCherry — expressed in subset of neurons + muscles

### Condition mapping

| Condition | Animals |
|-----------|---------|
| BlaC-Cat.Dead (control) | 2026-03-03-01, -02, -03, -04 |
| BlaC (experimental) | 2026-03-03-05, 2026-03-04-02, -03, -04, -05 |

## Running

```bash
cd /home/lauren/quick_wholebrain_overview
uv run python analyze_calcium_traces.py
```

## Outputs

```
outputs/
  traces/
    raw_traces.csv           # Raw mean & sum intensity per animal
    corrected_traces.csv     # Bleach-corrected traces
    bleach_fit.csv           # Fitted bleach curve
  plots/
    threshold_masks.png      # QC: masks overlaid on averaged MIPs
    bleach_fit.png            # Bleach curve fit visualization
    raw_traces_mean.png      # Raw mean intensity traces by condition
    raw_traces_sum.png       # Raw sum intensity traces by condition
    corrected_traces_mean.png
    corrected_traces_sum.png
    comparison_mean_sem.png  # Mean ± SEM comparison (bleach-corrected)
    comparison_sum_sem.png
  videos/
    qc_roi_*.mp4             # Per-animal MIP video with ROI outline
  stats/
    summary_statistics.csv   # t-test and Mann-Whitney results
```

## Key parameters

- `THRESHOLD_K = 4` — threshold = median + 4 * MAD (adjustable)
- `THRESHOLD_FRAMES = 10` — number of early frames averaged for mask computation
- Bleach correction uses double exponential fit on BlaC-Cat.Dead traces only
