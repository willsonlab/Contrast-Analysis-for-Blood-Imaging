#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PBS-normalised CSF with user-editable data-processing block.
Modify ONLY the function `load_and_preprocess()` to change lp/mm
filtering or how intensity data is prepared.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from scipy.signal import medfilt


# ─────────────────────────────────────────────────────────────────────────────
# USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
file_path  = r"filename.csv"
# frequency window (lp/mm) used to define PBS baseline
baseline_lo = 0.03
baseline_hi = 0.05

max_freq   = 2        # analysis cutoff lp/mm (still respected)
low_bins   = 5          # PBS baseline averaging window
smooth_win = 50        # envelope smoothing
median_k   = 1          # spike suppression kernel
noise_frac = 0.03       # noise floor fraction (of PBS baseline)

# output folders
base_dir   = os.path.dirname(file_path)
fname_root = os.path.splitext(os.path.basename(file_path))[0]
plot_dir   = os.path.join(base_dir, "plots-csf-PBSnorm")
os.makedirs(plot_dir, exist_ok=True)
out_table  = os.path.join(base_dir, f"{fname_root}_CSF_table_PBSnorm.csv")
out_sum    = os.path.join(base_dir, f"{fname_root}_CSF_summary_PBSnorm.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 1) USER-EDITABLE DATA-LOADING + PREPROCESSING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def load_and_preprocess(path, max_freq):
    """
    Load raw CSV, extract lp/mm, optionally transform/crop/smooth data.

    MODIFY ONLY THIS FUNCTION if you want to change how data is filtered.

    Must return:
        lpmm_ : 1D array of frequencies used
        data  : DataFrame of intensities only (no lp/mm col)
        df_   : filtered dataframe including lp/mm + intensities
    """
    df = pd.read_csv(path)

    # full lp/mm axis from first column
    lpmm = df.iloc[:, 0].values
    n    = len(lpmm)
    idx  = np.arange(n)

    # base frequency limit
    mask = lpmm <= max_freq

    # EXTRA cropping you wanted:
    #   - keep only lp/mm <= 2
    #   - drop first 5 rows
    # if you want to change this behaviour, edit these two lines
    mask &= (lpmm <= 1.9)
    mask &= (idx >= 0)

    # apply mask and reset index so everything stays aligned
    df_   = df[mask].reset_index(drop=True)
    lpmm_ = df_.iloc[:, 0].values          # filtered lp/mm
    data  = df_.iloc[:, 1:]                # intensity columns only

    return lpmm_, data, df_


# run the loading function
lpmm_, data, df_ = load_and_preprocess(file_path, max_freq)


# ─────────────────────────────────────────────────────────────────────────────
# 2) Identify PBS reference signal
# ─────────────────────────────────────────────────────────────────────────────
pbs_cols = [c for c in data.columns if "PBS" in c.upper()]
if not pbs_cols:
    raise ValueError("No column containing 'PBS' found in the filtered data.")

ref_col = pbs_cols[0]
ref_sig = df_[ref_col].values  # same length as lpmm_ now


# ─────────────────────────────────────────────────────────────────────────────
# 3) Envelope & contrast helper
# ─────────────────────────────────────────────────────────────────────────────
def envelope_contrast(signal, freqs, win=smooth_win):
    """
    Compute smoothed upper/lower envelopes and local contrast.
    signal and freqs MUST have the same length.
    """
    signal = np.asarray(signal)
    freqs  = np.asarray(freqs)
    n      = len(signal)

    if len(freqs) != n:
        raise ValueError(
            f"Length mismatch: signal={n}, freqs={len(freqs)}. "
            "Check your masking/cropping in load_and_preprocess()."
        )

    upper  = np.zeros_like(signal)
    lower  = np.zeros_like(signal)
    w_sizes = np.clip((10.0 / freqs).astype(int), 7, 151)

    for i in range(n):
        hw = w_sizes[i] // 2
        s  = slice(max(0, i - hw), min(n, i + hw + 1))
        upper[i] = signal[s].max()
        lower[i] = signal[s].min()

    up_s = uniform_filter1d(upper, win)
    lo_s = uniform_filter1d(lower, win)
    con  = (up_s - lo_s) / (up_s + lo_s)
    return up_s, lo_s, con


# PBS baseline
_, _, ref_con = envelope_contrast(ref_sig, lpmm_)

# indices for 0.03–0.05 lp/mm
baseline_mask = (lpmm_ >= baseline_lo) & (lpmm_ <= baseline_hi)
if not np.any(baseline_mask):
    raise ValueError("No lp/mm points in the baseline range 0.03–0.05.")

baseline = np.mean(ref_con[baseline_mask])

if baseline <= 0 or np.isnan(baseline):
    raise ValueError("PBS baseline is zero or NaN in 0.03–0.05 lp/mm.")





noise_floor = noise_frac * baseline


# ─────────────────────────────────────────────────────────────────────────────
# 4) Output containers
# ─────────────────────────────────────────────────────────────────────────────
csf_table = pd.DataFrame({"lp/mm": lpmm_})
summary   = []

def find_csf_freq(freqs, csf, level):
    """
    Find the FIRST frequency (from low to high) where CSF falls through `level`.
    Uses local linear interpolation between the two neighbouring samples
    that straddle the level. If no such crossing exists, returns NaN.
    """
    freqs = np.asarray(freqs)
    csf   = np.asarray(csf)

    # keep only finite points
    valid = np.isfinite(freqs) & np.isfinite(csf)
    if valid.sum() < 2:
        return np.nan

    f = freqs[valid]
    v = csf[valid]

    # ensure sorted by frequency (just in case)
    order = np.argsort(f)
    f = f[order]
    v = v[order]

    # must be above the level somewhere, or there's no meaningful crossing
    if v.max() < level:
        return np.nan

    # find first index where CSF >= level
    above = np.where(v >= level)[0]
    if len(above) == 0:
        return np.nan
    start = above[0]

    # scan forward to find first drop through the level
    for k in range(start, len(v) - 1):
        v1, v2 = v[k], v[k+1]
        f1, f2 = f[k], f[k+1]

        # looking for a crossing from >= level to <= level
        if (v1 >= level and v2 <= level) or (v1 <= level and v2 >= level):
            if v2 == v1:
                # flat segment at exactly the level
                return 0.5 * (f1 + f2)
            # linear interpolation in this segment
            t = (level - v1) / (v2 - v1)
            f_cross = f1 + t * (f2 - f1)
            return float(f_cross)

    # if we never straddle the level, no crossing
    return np.nan

# ─────────────────────────────────────────────────────────────────────────────
# 5) Main loop over samples
# ─────────────────────────────────────────────────────────────────────────────
for col in data.columns:
    sig = data[col].values              # use FILTERED data, not full df
    up, lo, con = envelope_contrast(sig, lpmm_)

    # PBS-normalised CSF
    csf = 100.0 * con / baseline

    # kill regions where contrast is below noise floor
    csf[np.abs(con) < noise_floor] = 0.0
    csf = np.clip(csf, 0, 100)
    csf = medfilt(csf, median_k)

    csf_table[col] = csf

    csf50 = find_csf_freq(lpmm_, csf, 50)
    csf10 = find_csf_freq(lpmm_, csf, 10)
    csf05 = find_csf_freq(lpmm_, csf, 5)

    summary.append({
        "Sample": col,
        "CSF50_lpmm": csf50,
        "CSF10_lpmm": csf10,
        "CSF5_lpmm": csf05,
        "PBS_baseline": baseline,
    })

    # plotting
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()

    ax1.plot(lpmm_, sig, color="orange", label="Raw intensity")
    ax1.plot(lpmm_, up, "--g", label="Upper envelope")
    ax1.plot(lpmm_, lo, "--b", label="Lower envelope")
    ax1.set_ylabel("Intensity")
    ax1.set_ylim(0, 255)

    ax2.plot(lpmm_, csf, "-r", label="CSF (%)")
    ax2.set_ylabel("CSF (%)", color="red")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="y", labelcolor="red")

    # annotate CSF thresholds
    def mark(freq, level, color):
        # Only draw if frequency is valid and within current x-limits
        if not np.isfinite(freq):
            return
        xmin, xmax = ax1.get_xlim()
        if freq <= xmin or freq >= xmax:
            return
    
        ax2.axvline(freq, color=color, ls="--", alpha=0.7)
        ax2.text(freq * 1.05, level + 2,
                 f"{level}%→{freq:.2f}",
                 color=color, fontsize=12, ha="left")

    mark(csf50, 50, "purple")
    mark(csf10, 10, "teal")
    mark(csf05, 5, "brown")

    ax1.set_xscale("log")
    ax1.set_xlim(lpmm_[lpmm_>0].min(), lpmm_.max())

     # custom tick locations (log spaced)
    xticks = [0.03, 0.05, 0.1, 0.2, 0.5, 1, 1.5]
    
    # filter ticks to range actually present
    xticks = [x for x in xticks if (lpmm_[0] <= x <= lpmm_[-1])]
    
    ax1.set_xticks(xticks)
    ax1.set_xticklabels([f"{x:g}" for x in xticks])

    
    
    ax1.set_xlabel("Spatial frequency (lp/mm)")
    ax1.set_title(f"{col} (PBS-normalised CSF)")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{col.replace('/','_')}_CSF.png"), dpi=300)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6) Save outputs
# ─────────────────────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary)
csf_full   = pd.concat([csf_table, summary_df.set_index("Sample").T])
csf_full.to_csv(out_table, index=False)
summary_df.to_csv(out_sum, index=False)

print("✓ CSF curves →", plot_dir)
print("✓ CSF table  →", out_table)
print("✓ CSF summary→", out_sum)
