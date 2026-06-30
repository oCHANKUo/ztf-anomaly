"""Feature extraction from ZTF light curves — ZTF DR3 inspired."""

import pandas as pd
import pickle
import numpy as np
from scipy import stats
from scipy.signal import lombscargle
import os


def mag_to_flux(mag):
    return 10 ** (-0.4 * (mag - 25.0))


def safe_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_periodogram_features(mjds, mags, n_freq=100):
    """
    Compute Lomb-Scargle periodogram features.
    Returns dict of features or NaNs if insufficient data.
    """
    if len(mags) < 10:
        return {
            'periodogram_amp': np.nan,
            'periodogram_std': np.nan,
            'periodogram_eta': np.nan,
            'period_peak': np.nan,
            'period_snr': np.nan,
        }
    
    t = mjds - mjds[0]
    if t[-1] <= 0:
        return {
            'periodogram_amp': np.nan,
            'periodogram_std': np.nan,
            'periodogram_eta': np.nan,
            'period_peak': np.nan,
            'period_snr': np.nan,
        }
    
    # Frequency grid: from 1/span to Nyquist (0.5 / median_dt)
    fmin = 1.0 / (t[-1] + 1.0)
    median_dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    fmax = 0.5 / max(median_dt, 0.1)
    
    if fmax <= fmin:
        return {
            'periodogram_amp': np.nan,
            'periodogram_std': np.nan,
            'periodogram_eta': np.nan,
            'period_peak': np.nan,
            'period_snr': np.nan,
        }
    
    freqs = np.linspace(fmin, fmax, n_freq)
    
    # Normalize and compute
    mags_norm = mags - np.mean(mags)
    try:
        pgram = lombscargle(t, mags_norm, freqs * 2 * np.pi, normalize=True)
    except:
        return {
            'periodogram_amp': np.nan,
            'periodogram_std': np.nan,
            'periodogram_eta': np.nan,
            'period_peak': np.nan,
            'period_snr': np.nan,
        }
    
    # Features
    amp = (np.max(pgram) - np.min(pgram)) / 2.0
    pstd = np.std(pgram)
    
    # Von Neumann eta for periodogram
    if len(pgram) > 1 and np.var(pgram) > 0:
        peta = np.sum(np.diff(pgram)**2) / ((len(pgram) - 1) * np.var(pgram))
    else:
        peta = np.nan
    
    # Peak period and SNR
    peak_idx = np.argmax(pgram)
    peak_freq = freqs[peak_idx]
    period = 1.0 / peak_freq if peak_freq > 0 else np.nan
    snr = pgram[peak_idx] / pstd if pstd > 0 else np.nan
    
    return {
        'periodogram_amp': amp,
        'periodogram_std': pstd,
        'periodogram_eta': peta,
        'period_peak': period,
        'period_snr': snr,
    }


def compute_stetson_k(mags, magerrs):
    """Compute Stetson K index — robust variability measure."""
    if len(mags) < 3:
        return np.nan
    
    weights = 1.0 / (magerrs + 1e-3)
    weighted_mean = np.average(mags, weights=weights)
    normalized = (mags - weighted_mean) / (magerrs + 1e-3)
    
    sum_abs = np.sum(np.abs(normalized))
    sum_sq = np.sum(normalized**2)
    
    if sum_sq <= 0:
        return np.nan
    
    return sum_abs / np.sqrt(len(mags) * sum_sq)


def extract_features(ztf_id, df, label, redshift, strict_mode=False):
    """
    Extract features from a single light curve.
    ZTF DR3 inspired: gap-robust statistical features, no sigma-clipping.
    """
    
    if df is None or df.empty:
        return None
    
    if 'mjd' not in df.columns or 'fid' not in df.columns:
        return None
    
    mag_col = safe_column(df, ['magpsf', 'magpsf_corr'])
    if mag_col is None:
        return None
    
    if 'magerr' not in df.columns:
        df['magerr'] = 0.1
    
    df = df.copy()
    
    # Track imputation
    frac_imputed = df["imputed"].sum() / len(df) if "imputed" in df.columns else 0.0
    
    # Basic quality filter: reasonable errors
    df = df[df['magerr'].between(0.001, 0.5)]
    df = df.drop_duplicates(subset=['mjd', 'fid'])
    df = df.sort_values('mjd')
    
    if len(df) < 10:
        return None
    
    # Magnitude range filter (catches obvious garbage)
    df = df[df[mag_col].between(12, 24)]
    if len(df) < 10:
        return None
    
    mjd_sorted = df['mjd'].values
    total_span = mjd_sorted[-1] - mjd_sorted[0]
    if total_span < 1:
        return None
    
    gaps = np.diff(mjd_sorted)
    max_gap = np.max(gaps) if len(gaps) > 0 else 0
    median_gap = np.median(gaps) if len(gaps) > 0 else 0
    
    n_total = len(df)
    n_g = len(df[df['fid'] == 1])
    n_r = len(df[df['fid'] == 2])
    
    # === STRICT MODE: tighter cuts but still gap-tolerant ===
    if strict_mode:
        if max_gap > 150:  # relaxed from 100
            return None
        if n_total < 15:  # relaxed from 20
            return None
        if frac_imputed > 0.15:  # relaxed from 0.10
            return None
        if n_g < 5 or n_r < 5:  # relaxed from 8
            return None
    
    # === DATA QUALITY FEATURES (for post-hoc filtering, NOT fed to ML) ===
    res = {
        "ztf_id": ztf_id,
        "label": label,
        "redshift": redshift,
        
        # DQ features — prefix with dq_ so training script can exclude them
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
    
    # === GLOBAL FEATURES (all bands combined) ===
    all_mags = df[mag_col].values
    all_magerrs = df['magerr'].values
    all_mjds = df['mjd'].values
    
    # Amplitude (half-range)
    res['global_amplitude'] = (np.max(all_mags) - np.min(all_mags)) / 2.0
    
    # Standard deviation
    res['global_std'] = np.std(all_mags)
    
    # Median absolute deviation (robust to outliers — no clipping needed!)
    res['global_mad'] = np.median(np.abs(all_mags - np.median(all_mags)))
    
    # Skewness and kurtosis
    res['global_skew'] = stats.skew(all_mags)
    res['global_kurt'] = stats.kurtosis(all_mags)
    
    # Mean and median
    res['global_mean'] = np.mean(all_mags)
    res['global_median'] = np.median(all_mags)
    
    # Percentiles
    res['global_p5'] = np.percentile(all_mags, 5)
    res['global_p95'] = np.percentile(all_mags, 95)
    res['global_p25'] = np.percentile(all_mags, 25)
    res['global_p75'] = np.percentile(all_mags, 75)
    
    # Inter-percentile ranges
    res['global_ipr_90'] = res['global_p95'] - res['global_p5']
    res['global_ipr_50'] = res['global_p75'] - res['global_p25']
    
    # Von Neumann eta (serial correlation)
    if len(all_mags) > 1 and np.var(all_mags) > 0:
        res['global_eta'] = np.sum(np.diff(all_mags)**2) / ((len(all_mags) - 1) * np.var(all_mags))
    else:
        res['global_eta'] = np.nan
    
    # Stetson K (robust variability index)
    res['global_stetson_k'] = compute_stetson_k(all_mags, all_magerrs)
    
    # Beyond N standard deviations (outlier fraction — KEY for anomalies)
    mean_mag = np.mean(all_mags)
    std_mag = np.std(all_mags)
    if std_mag > 0:
        res['global_beyond1std'] = np.mean(np.abs(all_mags - mean_mag) > std_mag)
        res['global_beyond2std'] = np.mean(np.abs(all_mags - mean_mag) > 2 * std_mag)
    else:
        res['global_beyond1std'] = 0.0
        res['global_beyond2std'] = 0.0
    
    # Cusum (cumulative sum range)
    if len(all_mags) > 1 and np.std(all_mags) > 0:
        cusum = np.cumsum(all_mags - mean_mag) / (len(all_mags) * std_mag)
        res['global_cusum'] = np.max(cusum) - np.min(cusum)
    else:
        res['global_cusum'] = np.nan
    
    # Linear trend
    if len(all_mags) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(all_mjds, all_mags)
        res['global_linear_trend'] = slope
        # Reduced chi2 of linear fit
        predicted = slope * all_mjds + intercept
        res['global_linear_chi2'] = np.sum(((all_mags - predicted) / all_magerrs)**2) / max(len(all_mags) - 2, 1)
        res['global_linear_r2'] = r_value**2
    else:
        res['global_linear_trend'] = np.nan
        res['global_linear_chi2'] = np.nan
        res['global_linear_r2'] = np.nan
    
    # Maximum slope between consecutive observations
    if len(all_mags) > 1:
        dt = np.diff(all_mjds)
        dm = np.diff(all_mags)
        # Avoid division by zero or tiny gaps
        safe_dt = np.maximum(dt, 0.1)
        slopes = np.abs(dm / safe_dt)
        res['global_max_slope'] = np.max(slopes)
    else:
        res['global_max_slope'] = np.nan
    
    # Global periodogram features
    global_pgram = compute_periodogram_features(all_mjds, all_mags)
    for k, v in global_pgram.items():
        res[f'global_{k}'] = v
    
    # === PER-BAND FEATURES ===
    all_peaks = []
    all_medians = []
    
    for fid, band in [(1, 'g'), (2, 'r')]:
        b = df[df['fid'] == fid].sort_values('mjd')
        prefix = f"{band}_"
        
        if len(b) < 3:
            # Fill NaNs for all band features
            band_keys = [
                'nobs', 'amplitude', 'std', 'mad', 'skew', 'kurt',
                'mean', 'median', 'p5', 'p95', 'p25', 'p75',
                'ipr_90', 'ipr_50', 'eta', 'stetson_k',
                'beyond1std', 'beyond2std', 'cusum',
                'max_slope', 'linear_trend', 'linear_chi2', 'linear_r2',
                'periodogram_amp', 'periodogram_std', 'periodogram_eta',
                'period_peak', 'period_snr'
            ]
            for k in band_keys:
                res[prefix + k] = np.nan
            continue
        
        mags = b[mag_col].values
        mjds = b['mjd'].values
        magerrs = b['magerr'].values
        
        # Basic stats
        res[prefix + 'nobs'] = len(mags)
        res[prefix + 'amplitude'] = (np.max(mags) - np.min(mags)) / 2.0
        res[prefix + 'std'] = np.std(mags)
        res[prefix + 'mad'] = np.median(np.abs(mags - np.median(mags)))
        res[prefix + 'skew'] = stats.skew(mags)
        res[prefix + 'kurt'] = stats.kurtosis(mags)
        res[prefix + 'mean'] = np.mean(mags)
        res[prefix + 'median'] = np.median(mags)
        res[prefix + 'p5'] = np.percentile(mags, 5)
        res[prefix + 'p95'] = np.percentile(mags, 95)
        res[prefix + 'p25'] = np.percentile(mags, 25)
        res[prefix + 'p75'] = np.percentile(mags, 75)
        res[prefix + 'ipr_90'] = res[prefix + 'p95'] - res[prefix + 'p5']
        res[prefix + 'ipr_50'] = res[prefix + 'p75'] - res[prefix + 'p25']
        
        # Store for cross-band features
        all_peaks.append(np.min(mags))  # peak = minimum magnitude (brightest)
        all_medians.append(res[prefix + 'median'])
        
        # Von Neumann eta
        if len(mags) > 1 and np.var(mags) > 0:
            res[prefix + 'eta'] = np.sum(np.diff(mags)**2) / ((len(mags) - 1) * np.var(mags))
        else:
            res[prefix + 'eta'] = np.nan
        
        # Stetson K
        res[prefix + 'stetson_k'] = compute_stetson_k(mags, magerrs)
        
        # Beyond N std
        bmean = np.mean(mags)
        bstd = np.std(mags)
        if bstd > 0:
            res[prefix + 'beyond1std'] = np.mean(np.abs(mags - bmean) > bstd)
            res[prefix + 'beyond2std'] = np.mean(np.abs(mags - bmean) > 2 * bstd)
        else:
            res[prefix + 'beyond1std'] = 0.0
            res[prefix + 'beyond2std'] = 0.0
        
        # Cusum
        if len(mags) > 1 and bstd > 0:
            cusum = np.cumsum(mags - bmean) / (len(mags) * bstd)
            res[prefix + 'cusum'] = np.max(cusum) - np.min(cusum)
        else:
            res[prefix + 'cusum'] = np.nan
        
        # Linear trend
        if len(mags) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(mjds, mags)
            res[prefix + 'linear_trend'] = slope
            predicted = slope * mjds + intercept
            res[prefix + 'linear_chi2'] = np.sum(((mags - predicted) / magerrs)**2) / max(len(mags) - 2, 1)
            res[prefix + 'linear_r2'] = r_value**2
        else:
            res[prefix + 'linear_trend'] = np.nan
            res[prefix + 'linear_chi2'] = np.nan
            res[prefix + 'linear_r2'] = np.nan
        
        # Max slope
        if len(mags) > 1:
            dt = np.diff(mjds)
            dm = np.diff(mags)
            safe_dt = np.maximum(dt, 0.1)
            slopes = np.abs(dm / safe_dt)
            res[prefix + 'max_slope'] = np.max(slopes)
        else:
            res[prefix + 'max_slope'] = np.nan
        
        # Periodogram features
        pgram = compute_periodogram_features(mjds, mags)
        for k, v in pgram.items():
            res[prefix + k] = v
    
    # === CROSS-BAND FEATURES ===
    # Color
    if len(all_medians) == 2 and not any(np.isnan(all_medians)):
        res['color_gr'] = all_medians[0] - all_medians[1]  # g - r
    else:
        res['color_gr'] = np.nan
    
    # Peak color
    if len(all_peaks) == 2:
        res['peak_color'] = all_peaks[0] - all_peaks[1]
    else:
        res['peak_color'] = np.nan
    
    # Amplitude ratio
    g_amp = res.get('g_amplitude', np.nan)
    r_amp = res.get('r_amplitude', np.nan)
    if not np.isnan(g_amp) and not np.isnan(r_amp) and r_amp > 0:
        res['amplitude_ratio_gr'] = g_amp / r_amp
    else:
        res['amplitude_ratio_gr'] = np.nan
    
    # Band eta ratio (catches differential sampling artifacts)
    g_eta = res.get('g_eta', np.nan)
    r_eta = res.get('r_eta', np.nan)
    if not np.isnan(g_eta) and not np.isnan(r_eta) and r_eta > 0:
        res['eta_ratio_gr'] = g_eta / r_eta
    else:
        res['eta_ratio_gr'] = np.nan
    
    # Absolute magnitude
    try:
        z = float(redshift)
        if z > 0.001 and not np.isnan(all_peaks[0]):
            dist_pc = (z * 3e5) / 70 * 1e6
            res["abs_mag_g"] = all_peaks[0] - 5 * np.log10(dist_pc) + 5
        else:
            res["abs_mag_g"] = np.nan
    except:
        res["abs_mag_g"] = np.nan
    
    return res


def run_features(data_dir="./data", strict_mode=True):
    """Extract features from all light curves."""
    
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
    
    # Drop columns that are ALL NaN
    df_final = df_final.dropna(axis=1, how='all')
    
    mode_str = "strict" if strict_mode else "loose"
    out_file = os.path.join(data_dir, f"ztf_features_{mode_str}.csv")
    df_final.to_csv(out_file, index=False)
    
    print(f"\nDONE ({mode_str})")
    print(f"  Valid objects: {len(df_final)}")
    print(f"  Rejected: {failed}")
    print(f"  Feature count: {len(df_final.columns)}")
    print(f"  Saved to {out_file}")
    
    # Print feature breakdown
    dq_cols = [c for c in df_final.columns if c.startswith('dq_')]
    global_cols = [c for c in df_final.columns if c.startswith('global_')]
    g_cols = [c for c in df_final.columns if c.startswith('g_')]
    r_cols = [c for c in df_final.columns if c.startswith('r_')]
    other_cols = [c for c in df_final.columns if c not in dq_cols + global_cols + g_cols + r_cols + ['ztf_id', 'label', 'redshift']]
    
    print(f"\n  Feature breakdown:")
    print(f"    DQ features: {len(dq_cols)}")
    print(f"    Global features: {len(global_cols)}")
    print(f"    g-band features: {len(g_cols)}")
    print(f"    r-band features: {len(r_cols)}")
    print(f"    Other features: {len(other_cols)}")


if __name__ == "__main__":
    run_features(strict_mode=True)