"""
Power analysis for BlaC vs BlaC-Cat.Dead post-bleach raw mean intensity.

Reads precomputed per-animal traces from the cache produced by
analyze_calcium_traces_preliminarydata.py (the canonical analysis), then:
  1. Computes the observed effect size (Cohen's d, Hedges' g).
  2. Estimates n per group required to detect that effect at various powers
     using analytic t-test power and simulation-based Mann-Whitney U power
     (the test actually used in the analysis).
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, t

# ── Configuration ──────────────────────────────────────────────────────────────

# Per-animal trace cache written by analyze_calcium_traces_preliminarydata.py.
# Each {animal}.npz contains `mean_trace` covering all frames; we slice
# post-bleach here so this script stays decoupled from BLEACH_CUTOFF_FRAME
# choices baked into the analyze run.
CACHE_DIR = Path(
    "/store1/lauren/Tetramisole_Immobilized_Imaging/"
    "2026_cAMP_wholebrain_with_pdfr1_BlaC_Immobilized/PreliminaryData/cache"
)
OUTPUT_DIR = Path("/home/lauren/quick_wholebrain_overview/outputs/power_analysis")

BLEACH_CUTOFF_FRAME = 50

# Condition mapping — must match analyze_calcium_traces_preliminarydata.py
BLAC_CAT_DEAD_ALL = ["2026-03-03-01", "2026-03-03-02", "2026-03-03-03", "2026-03-03-04"]
BLAC_ALL = ["2026-03-03-05", "2026-03-04-02", "2026-03-04-03", "2026-03-04-04", "2026-03-04-05"]
EXCLUDE = ["2026-03-03-04"]   # wrong length timeseries

BLAC_CAT_DEAD = [n for n in BLAC_CAT_DEAD_ALL if n not in EXCLUDE]
BLAC = [n for n in BLAC_ALL if n not in EXCLUDE]

# Power analysis settings
ALPHA = 0.05
TARGET_POWERS = [0.7, 0.8, 0.9, 0.95]
SAMPLE_SIZES = list(range(3, 31))   # n per group to scan
N_SIMULATIONS = 5000                # bootstrap reps per (n, model)
RNG_SEED = 42

# ── Cache loader ──────────────────────────────────────────────────────────────

def load_post_bleach_mean(animal: str) -> float:
    path = CACHE_DIR / f"{animal}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"No cache for {animal} at {path}. "
            f"Run analyze_calcium_traces_preliminarydata.py first."
        )
    data = np.load(path)
    return float(data["mean_trace"][BLEACH_CUTOFF_FRAME:].mean())


# ── Effect-size helpers ───────────────────────────────────────────────────────

def cohens_d(x, y):
    """Cohen's d with pooled SD (Hedges-style pooling)."""
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled, pooled


def hedges_g(d, n1, n2):
    """Small-sample-corrected effect size."""
    df = n1 + n2 - 2
    J = 1 - 3 / (4 * df - 1)
    return d * J


# ── Analytic t-test power ─────────────────────────────────────────────────────

def ttest_power(n_per_group, d, alpha=0.05):
    """Two-sided independent t-test power (equal n)."""
    df = 2 * n_per_group - 2
    nc = d * np.sqrt(n_per_group / 2)              # noncentrality
    crit = t.ppf(1 - alpha / 2, df)
    # Approx with normal for the noncentral upper tail (good enough for planning)
    return 1 - norm.cdf(crit - nc) + norm.cdf(-crit - nc)


def n_for_power(d, target, alpha=0.05, n_max=500):
    for n in range(2, n_max + 1):
        if ttest_power(n, d, alpha) >= target:
            return n
    return None


# ── Simulation-based Mann-Whitney power ───────────────────────────────────────

def simulate_mw_power(n_per_group, mu1, mu2, sd, n_sims, alpha, rng):
    """
    Draw two normal samples with the given means/SD, run Mann-Whitney U,
    return fraction of sims with p < alpha.
    """
    rejects = 0
    for _ in range(n_sims):
        x = rng.normal(mu1, sd, n_per_group)
        y = rng.normal(mu2, sd, n_per_group)
        _, p = mannwhitneyu(x, y, alternative="two-sided")
        if p < alpha:
            rejects += 1
    return rejects / n_sims


def simulate_mw_power_bootstrap(n_per_group, blac_obs, dead_obs, n_sims, alpha, rng):
    """
    Non-parametric: resample with replacement from the observed preliminary
    distributions. Captures the actual shape/spread of the data, not just mean+SD.
    """
    rejects = 0
    for _ in range(n_sims):
        x = rng.choice(blac_obs, size=n_per_group, replace=True)
        y = rng.choice(dead_obs, size=n_per_group, replace=True)
        _, p = mannwhitneyu(x, y, alternative="two-sided")
        if p < alpha:
            rejects += 1
    return rejects / n_sims


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    # Step 1: load post-bleach means from cache
    print("=" * 60)
    print("  Loading post-bleach means from analyze cache")
    print("=" * 60)
    print(f"  Cache dir: {CACHE_DIR}")
    print(f"  Bleach cutoff frame: {BLEACH_CUTOFF_FRAME}")

    rows = []
    for name in BLAC_CAT_DEAD:
        val = load_post_bleach_mean(name)
        rows.append({"animal": name, "condition": "BlaC-Cat.Dead",
                     "post_bleach_mean": val})
        print(f"  {name} (BlaC-Cat.Dead): post-bleach mean = {val:.2f}")
    for name in BLAC:
        val = load_post_bleach_mean(name)
        rows.append({"animal": name, "condition": "BlaC",
                     "post_bleach_mean": val})
        print(f"  {name} (BlaC):          post-bleach mean = {val:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "preliminary_post_bleach_means.csv", index=False)

    blac_vals = df.loc[df.condition == "BlaC", "post_bleach_mean"].values
    dead_vals = df.loc[df.condition == "BlaC-Cat.Dead", "post_bleach_mean"].values

    # Step 2: effect size
    print("\n" + "=" * 60)
    print("  Effect size from preliminary data")
    print("=" * 60)

    n_blac, n_dead = len(blac_vals), len(dead_vals)
    mean_blac, mean_dead = blac_vals.mean(), dead_vals.mean()
    sd_blac, sd_dead = blac_vals.std(ddof=1), dead_vals.std(ddof=1)

    d, sd_pooled = cohens_d(blac_vals, dead_vals)
    g = hedges_g(d, n_blac, n_dead)

    print(f"  BlaC          (n={n_blac}): {mean_blac:.2f} ± {sd_blac:.2f}")
    print(f"  BlaC-Cat.Dead (n={n_dead}): {mean_dead:.2f} ± {sd_dead:.2f}")
    print(f"  Pooled SD:        {sd_pooled:.2f}")
    print(f"  Cohen's d:        {d:+.3f}")
    print(f"  Hedges' g:        {g:+.3f}")

    # Run Mann-Whitney on the preliminary data itself for reference
    u, p_obs = mannwhitneyu(blac_vals, dead_vals, alternative="two-sided")
    print(f"  Preliminary Mann-Whitney: U={u:.1f}, p={p_obs:.4f}")

    # Step 3: analytic power table
    print("\n" + "=" * 60)
    print("  Analytic t-test power (Cohen's d = {:.3f})".format(d))
    print("=" * 60)

    print(f"  {'target power':>14} | {'n per group':>11}")
    print(f"  {'-'*14}-+-{'-'*11}")
    target_table = []
    for tp in TARGET_POWERS:
        n_req = n_for_power(abs(d), tp, ALPHA)
        target_table.append({"target_power": tp, "n_per_group_ttest": n_req})
        print(f"  {tp:>14.2f} | {str(n_req):>11}")

    # Step 4: simulation-based power across sample sizes
    print("\n" + "=" * 60)
    print("  Simulation-based power (Mann-Whitney U)")
    print("=" * 60)
    print(f"  N simulations per point: {N_SIMULATIONS}")
    print(f"  Effect: BlaC mean={mean_blac:.2f}, Dead mean={mean_dead:.2f}, "
          f"pooled SD={sd_pooled:.2f}")

    power_rows = []
    for n in SAMPLE_SIZES:
        p_param = simulate_mw_power(
            n, mean_blac, mean_dead, sd_pooled,
            N_SIMULATIONS, ALPHA, rng,
        )
        p_boot = simulate_mw_power_bootstrap(
            n, blac_vals, dead_vals,
            N_SIMULATIONS, ALPHA, rng,
        )
        p_tt = ttest_power(n, abs(d), ALPHA)
        power_rows.append({
            "n_per_group": n,
            "power_ttest_analytic": p_tt,
            "power_mw_parametric_sim": p_param,
            "power_mw_bootstrap_sim": p_boot,
        })
        print(f"  n={n:2d} | t-test={p_tt:.3f} | "
              f"MW(normal)={p_param:.3f} | MW(bootstrap)={p_boot:.3f}")

    power_df = pd.DataFrame(power_rows)
    power_df.to_csv(OUTPUT_DIR / "power_curve.csv", index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'power_curve.csv'}")

    # Step 5: smallest n reaching each target power (per method)
    print("\n" + "=" * 60)
    print("  Sample size to reach each target power")
    print("=" * 60)
    print(f"  {'target':>6} | {'t-test':>7} | {'MW(norm)':>9} | {'MW(boot)':>9}")
    print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*9}-+-{'-'*9}")

    summary_rows = []
    for tp in TARGET_POWERS:
        def first_n(col):
            hits = power_df.loc[power_df[col] >= tp, "n_per_group"]
            return int(hits.iloc[0]) if len(hits) else None

        n_tt = first_n("power_ttest_analytic")
        n_par = first_n("power_mw_parametric_sim")
        n_boot = first_n("power_mw_bootstrap_sim")
        summary_rows.append({
            "target_power": tp,
            "n_ttest": n_tt,
            "n_mw_parametric": n_par,
            "n_mw_bootstrap": n_boot,
        })
        print(f"  {tp:>6.2f} | {str(n_tt):>7} | {str(n_par):>9} | {str(n_boot):>9}")

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "sample_size_summary.csv", index=False,
    )

    # Step 6: power curve plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(power_df["n_per_group"], power_df["power_ttest_analytic"],
            "-",  color="#999999", linewidth=2, label="t-test (analytic)")
    ax.plot(power_df["n_per_group"], power_df["power_mw_parametric_sim"],
            "o-", color="#4477AA", linewidth=2,
            label="Mann-Whitney (normal sim)")
    ax.plot(power_df["n_per_group"], power_df["power_mw_bootstrap_sim"],
            "s-", color="#CC3311", linewidth=2,
            label="Mann-Whitney (bootstrap sim)")
    for tp in TARGET_POWERS:
        ax.axhline(tp, color="k", linestyle=":", linewidth=0.7, alpha=0.4)
    ax.axhline(0.8, color="k", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("n per group")
    ax.set_ylabel("Power")
    ax.set_ylim(0, 1.02)
    ax.set_title(
        f"Power vs sample size  |  Cohen's d = {d:+.2f}, Hedges' g = {g:+.2f}  "
        f"|  α = {ALPHA}"
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "power_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Step 7: write a short text report
    report = OUTPUT_DIR / "power_analysis_report.txt"
    with open(report, "w") as f:
        f.write("Power analysis based on preliminary post-bleach means\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"BlaC          (n={n_blac}): {mean_blac:.2f} ± {sd_blac:.2f}\n")
        f.write(f"BlaC-Cat.Dead (n={n_dead}): {mean_dead:.2f} ± {sd_dead:.2f}\n")
        f.write(f"Pooled SD: {sd_pooled:.2f}\n")
        f.write(f"Cohen's d: {d:+.3f}\n")
        f.write(f"Hedges' g: {g:+.3f}\n")
        f.write(f"Preliminary Mann-Whitney: U={u:.1f}, p={p_obs:.4f}\n\n")
        f.write(f"Alpha = {ALPHA}, simulations = {N_SIMULATIONS}\n\n")
        f.write("Sample size required (per group):\n")
        for row in summary_rows:
            f.write(
                f"  power={row['target_power']:.2f}  "
                f"t-test={row['n_ttest']}  "
                f"MW(norm)={row['n_mw_parametric']}  "
                f"MW(boot)={row['n_mw_bootstrap']}\n"
            )
    print(f"\n  Saved: {report}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Outputs in: {OUTPUT_DIR}")
