"""Feature extraction from ZTF light curves.

Simplified, undergrad-scope version. Computes:
  - 8 global features (amplitude, std, mad, skew, linear trend, linear chi2,
    max slope, periodogram amplitude)
  - 1 extra global feature (periodogram std)
  - 8 g-band features
  - 8 r-band features
  - 4 cross-band features (color, peak color, amplitude ratio, abs mag)
  - 9 data-quality (DQ) features — kept OUT of the ML input, used only
    to sanity-check / flag results after the fact
"""

import pandas as pd
import pickle
import numpy as np
from scipy import stats
from scipy.signal import lombscargle
import os


def safe_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_periodogram(mjds, mags, n_freq=100):
    """
    Lomb-Scargle periodogram, reduced to two numbers:
      - amplitude: how strong the strongest periodic signal is
      - std: how much power is spread across the periodogram

    Both are just descriptive summaries of "does this light curve
    look periodic at all" — we do NOT try to extract or trust an
    actual period value, since that needs far more care.
    """
    if len(mags) < 10:
        return {"periodogram_amp": np.nan, "periodogram_std": np.nan}

    t = mjds - mjds[0]
    if t[-1] <= 0:
        return {"periodogram_amp": np.nan, "periodogram_std": np.nan}

    fmin = 1.0 / (t[-1] + 1.0)
    median_dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    fmax = 0.5 / max(median_dt, 0.1)

    if fmax <= fmin:
        return {"periodogram_amp": np.nan, "periodogram_std": np.nan}

    freqs = np.linspace(fmin, fmax, n_freq)
    mags_norm = mags - np.mean(mags)

    try:
        pgram = lombscargle(t, mags_norm, freqs * 2 * np.pi, normalize=True)
    except Exception:
        return {"periodogram_amp": np.nan, "periodogram_std": np.nan}

    amp = (np.max(pgram) - np.min(pgram)) / 2.0
    pstd = np.std(pgram)

    return {"periodogram_amp": amp, "periodogram_std": pstd}

def compute_stetson_k(mags, magerrs):
    """Stetson K — ratio of mean absolute deviation to RMS deviation,
    normalized by photometric error. ~0.8 for Gaussian noise;
    lower values mean a few large deviations dominate (bursty/spiky),
    higher values mean deviations are more uniform."""
    if len(mags) < 3:
        return np.nan

    weights = 1.0 / (magerrs + 1e-3)
    weighted_mean = np.average(mags, weights=weights)
    normalized = (mags - weighted_mean) / (magerrs + 1e-3)

    sum_abs = np.sum(np.abs(normalized))
    sum_sq = np.sum(normalized ** 2)

    if sum_sq <= 0:
        return np.nan

    return sum_abs / np.sqrt(len(mags) * sum_sq)


def band_features(mags, mjds, magerrs, prefix, res):
    """Compute the 8 per-band features and write them into `res`."""
    keys = ["nobs", "amplitude", "std", "mad", "skew",
            "linear_trend", "linear_chi2", "max_slope", "stetson_k"]

    if len(mags) < 3:
        for k in keys:
            res[prefix + k] = np.nan
        return

    res[prefix + "nobs"] = len(mags)
    res[prefix + "amplitude"] = (np.max(mags) - np.min(mags)) / 2.0
    res[prefix + "std"] = np.std(mags)
    res[prefix + "mad"] = np.median(np.abs(mags - np.median(mags)))
    res[prefix + "skew"] = stats.skew(mags)
    res[prefix + "stetson_k"] = compute_stetson_k(mags, magerrs)

    if len(mags) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(mjds, mags)
        predicted = slope * mjds + intercept
        res[prefix + "linear_trend"] = slope
        # res[prefix + "linear_chi2"] = np.sum(((mags - predicted) / magerrs) ** 2) / max(len(mags) - 2, 1)
        adjusted_errors = np.sqrt(magerrs**2 + 0.02**2)
        res[prefix + "linear_chi2"] = np.sum(((mags - predicted) / adjusted_errors) ** 2) / max(len(mags) - 2, 1)
    else:
        res[prefix + "linear_trend"] = np.nan
        res[prefix + "linear_chi2"] = np.nan

    if len(mags) > 1:
        dt = np.maximum(np.diff(mjds), 0.1)
        dm = np.diff(mags)
        res[prefix + "max_slope"] = np.percentile(np.abs(dm / dt), 95)
    else:
        res[prefix + "max_slope"] = np.nan


def extract_features(ztf_id, df, label, redshift, strict_mode=True):
    """Extract the simplified feature set from a single light curve."""

    if df is None or df.empty:
        return None
    if "mjd" not in df.columns or "fid" not in df.columns:
        return None

    mag_col = safe_column(df, ["magpsf", "magpsf_corr"])
    if mag_col is None:
        return None

    if "magerr" not in df.columns:
        df["magerr"] = 0.1

    df = df.copy()

    frac_imputed = df["imputed"].sum() / len(df) if "imputed" in df.columns else 0.0

    # Basic quality filter
    df = df[df["magerr"].between(0.001, 0.5)]
    df = df.drop_duplicates(subset=["mjd", "fid"])
    df = df.sort_values("mjd")
    if len(df) < 10:
        return None

    df = df[df[mag_col].between(12, 24)]
    if len(df) < 10:
        return None

    mjd_sorted = df["mjd"].values
    total_span = mjd_sorted[-1] - mjd_sorted[0]
    if total_span < 1:
        return None

    gaps = np.diff(mjd_sorted)
    max_gap = np.max(gaps) if len(gaps) > 0 else 0
    median_gap = np.median(gaps) if len(gaps) > 0 else 0

    n_total = len(df)
    n_g = len(df[df["fid"] == 1])
    n_r = len(df[df["fid"] == 2])

    if strict_mode:
        if max_gap > 150:
            return None
        if n_total < 15:
            return None
        if frac_imputed > 0.15:
            return None
        if n_g < 5 or n_r < 5:
            return None

    # === DQ features (excluded from ML input, kept for post-hoc filtering) ===
    res = {
        "ztf_id": ztf_id,
        "label": label,
        "redshift": redshift,

        "dq_total_points": n_total,
        "dq_g_frac": n_g / n_total if n_total > 0 else 0,
        "dq_r_frac": n_r / n_total if n_total > 0 else 0,
        "dq_span": total_span,
        "dq_max_gap": max_gap,
        "dq_median_gap": median_gap,
        "dq_frac_imputed": frac_imputed,
        "dq_log_n": np.log10(n_total),
        "dq_sampling_density": n_total / max(total_span, 1),
    }

    # === Global features (8 + 1 periodogram_std) ===
    all_mags = df[mag_col].values
    all_magerrs = df["magerr"].values
    all_mjds = df["mjd"].values

    res["global_amplitude"] = (np.max(all_mags) - np.min(all_mags)) / 2.0
    res["global_std"] = np.std(all_mags)
    res["global_mad"] = np.median(np.abs(all_mags - np.median(all_mags)))
    res["global_skew"] = stats.skew(all_mags)

    if len(all_mags) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_mjds, all_mags)
        predicted = slope * all_mjds + intercept
        res["global_linear_trend"] = slope
        # res["global_linear_chi2"] = np.sum(((all_mags - predicted) / all_magerrs) ** 2) / max(len(all_mags) - 2, 1)
        adjusted_errors = np.sqrt(all_magerrs**2 + 0.02**2)
        res["global_linear_chi2"] = np.sum(((all_mags - predicted) / adjusted_errors) ** 2) / max(len(all_mags) - 2, 1)
    else:
        res["global_linear_trend"] = np.nan
        res["global_linear_chi2"] = np.nan

    if len(all_mags) > 1:
        dt = np.maximum(np.diff(all_mjds), 0.1)
        dm = np.diff(all_mags)
        # res["global_max_slope"] = np.max(np.abs(dm / dt))
        res["global_max_slope"] = np.percentile(np.abs(dm / dt), 95)
    else:
        res["global_max_slope"] = np.nan

    pgram = compute_periodogram(all_mjds, all_mags)
    res["global_periodogram_amp"] = pgram["periodogram_amp"]
    res["global_periodogram_std"] = pgram["periodogram_std"]

    # === Per-band features ===
    all_peaks = []
    all_medians = []

    for fid, band in [(1, "g"), (2, "r")]:
        b = df[df["fid"] == fid].sort_values("mjd")
        prefix = f"{band}_"

        if len(b) >= 3:
            all_peaks.append(np.min(b[mag_col].values))
            all_medians.append(np.median(b[mag_col].values))
        else:
            all_peaks.append(np.nan)
            all_medians.append(np.nan)

        band_features(
            b[mag_col].values, b["mjd"].values, b["magerr"].values, prefix, res
        )

    # === Cross-band features (4) ===
    if len(all_medians) == 2 and not any(np.isnan(all_medians)):
        res["color_gr"] = all_medians[0] - all_medians[1]
    else:
        res["color_gr"] = np.nan

    if len(all_peaks) == 2 and not any(np.isnan(all_peaks)):
        res["peak_color"] = all_peaks[0] - all_peaks[1]
    else:
        res["peak_color"] = np.nan

    g_amp = res.get("g_amplitude", np.nan)
    r_amp = res.get("r_amplitude", np.nan)
    if not np.isnan(g_amp) and not np.isnan(r_amp) and r_amp > 0:
        res["amplitude_ratio_gr"] = g_amp / r_amp
    else:
        res["amplitude_ratio_gr"] = np.nan

    try:
        z = float(redshift)
        if z > 0.001 and not np.isnan(all_peaks[0]):
            dist_pc = (z * 3e5) / 70 * 1e6
            res["abs_mag_g"] = all_peaks[0] - 5 * np.log10(dist_pc) + 5
        else:
            res["abs_mag_g"] = np.nan
    except Exception:
        res["abs_mag_g"] = np.nan

    return res


def run_features(data_dir="./data", strict_mode=True):
    """Extract features from all downloaded light curves."""

    lcs_path = os.path.join(data_dir, "ztf_lcs_all.pkl")
    with open(lcs_path, "rb") as f:
        all_lcs = pickle.load(f)

    bts_path = os.path.join(data_dir, "bts_all_labeled.csv")
    bts = pd.read_csv(bts_path).set_index("ZTFID")

    final_data = []
    failed = 0

    for ztf_id, df in all_lcs.items():
        if ztf_id in bts.index:
            label = bts.loc[ztf_id, "type"]
            redshift = bts.loc[ztf_id, "redshift"]
        else:
            label = "Unknown"
            redshift = 0

        feat = extract_features(ztf_id, df, label, redshift, strict_mode=strict_mode)
        if feat is not None:
            final_data.append(feat)
        else:
            failed += 1

    df_final = pd.DataFrame(final_data)

    if df_final.empty:
        print("No valid samples produced. Check filters.")
        return

    df_final = df_final.dropna(axis=1, how="all")

    mode_str = "strict" if strict_mode else "loose"
    out_file = os.path.join(data_dir, f"ztf_features_{mode_str}.csv")
    df_final.to_csv(out_file, index=False)

    dq_cols = [c for c in df_final.columns if c.startswith("dq_")]
    global_cols = [c for c in df_final.columns if c.startswith("global_")]
    g_cols = [c for c in df_final.columns if c.startswith("g_")]
    r_cols = [c for c in df_final.columns if c.startswith("r_")]
    cross_cols = ["color_gr", "peak_color", "amplitude_ratio_gr", "abs_mag_g"]
    cross_cols = [c for c in cross_cols if c in df_final.columns]

    print(f"\nDONE ({mode_str})")
    print(f"  Valid objects: {len(df_final)}")
    print(f"  Rejected: {failed}")
    print(f"  Total columns: {len(df_final.columns)}")
    print(f"\n  Feature breakdown:")
    print(f"    DQ features (excluded from ML): {len(dq_cols)}")
    print(f"    Global features: {len(global_cols)}")
    print(f"    g-band features: {len(g_cols)}")
    print(f"    r-band features: {len(r_cols)}")
    print(f"    Cross-band features: {len(cross_cols)}")
    print(f"    ML feature total: {len(global_cols) + len(g_cols) + len(r_cols) + len(cross_cols)}")
    print(f"  Saved to {out_file}")


if __name__ == "__main__":
    run_features(strict_mode=True)