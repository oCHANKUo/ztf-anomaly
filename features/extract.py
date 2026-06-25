"""Feature extraction from ZTF light curves."""

import pandas as pd
import pickle
import numpy as np
from scipy import stats
import os


def mag_to_flux(mag):
    return 10 ** (-0.4 * (mag - 25.0))


def sigma_clip(mags, mjds, magerrs, sigma=3.0):
    if len(mags) < 5:
        return mags, mjds, magerrs
    z_scores = np.abs(stats.zscore(mags))
    mask = z_scores < sigma
    peak_idx = np.argmin(mags)
    mask[peak_idx] = True
    return mags[mask], mjds[mask], magerrs[mask]


def safe_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_features(ztf_id, df, label, redshift, strict_mode=False):
    """Extract features from a single light curve."""
    
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
    frac_imputed = df["imputed"].sum() / len(df) if "imputed" in df.columns else 0.0
    
    df = df[df['magerr'].between(0.0, 0.3)]
    df = df.drop_duplicates(subset=['mjd', 'fid'])
    df = df.sort_values('mjd')
    
    if len(df) < 10:
        return None
    
    df = df[df[mag_col].between(12, 24)]
    if len(df) < 10:
        return None
    
    mjd_sorted = df['mjd'].values
    total_span = mjd_sorted[-1] - mjd_sorted[0]
    if total_span < 1:
        return None
    
    gaps = np.diff(mjd_sorted)
    max_gap = np.max(gaps) if len(gaps) > 0 else 0
    
    if len(gaps) > 0 and max_gap > 200:
        return None
    
    n_total = len(df)
    n_g = len(df[df['fid'] == 1])
    n_r = len(df[df['fid'] == 2])
    
    # Strict quality cuts
    if strict_mode:
        if max_gap > 100:
            return None
        if n_total < 20:
            return None
        if frac_imputed > 0.10:
            return None
        if n_g < 8 or n_r < 8:
            return None
        
        mags = df[mag_col].values
        peak_idx = np.argmin(mags)
        if peak_idx < 3 or peak_idx > len(mags) - 4:
            return None
    
    # Data quality features
    res = {
        "ztf_id": ztf_id,
        "label": label,
        "redshift": redshift,
        "dq_total_points": n_total,
        "dq_g_frac": n_g / n_total if n_total > 0 else 0,
        "dq_r_frac": n_r / n_total if n_total > 0 else 0,
        "dq_span": total_span,
        "dq_max_gap": max_gap,
        "dq_frac_imputed": frac_imputed,
        "dq_log_n": np.log10(n_total),
        "dq_sampling_density": n_total / max(total_span, 1),
    }
    
    all_peaks = []
    all_durations = []
    
    for fid, band in [(1, 'g'), (2, 'r')]:
        b = df[df['fid'] == fid].sort_values('mjd')
        
        base_keys = [
            'peak', 'rise', 'fall', 'duration', 'stability',
            'skew', 'kurt', 'nobs', 'slope_pre', 'slope_post',
            'time_to_peak', 'asymmetry', 'wmean'
        ]
        
        if len(b) < 3:
            for k in base_keys:
                res[f"{band}_{k}"] = np.nan
            continue
        
        mags = b[mag_col].values
        mjds = b['mjd'].values
        magerrs = b['magerr'].values
        
        mags, mjds, magerrs = sigma_clip(mags, mjds, magerrs)
        
        if len(mags) < 3:
            for k in base_keys:
                res[f"{band}_{k}"] = np.nan
            continue
        
        flux = mag_to_flux(mags)
        
        peak_idx = np.argmin(mags)
        peak_mag = mags[peak_idx]
        peak_mjd = mjds[peak_idx]
        
        res[f"{band}_peak"] = peak_mag
        res[f"{band}_duration"] = mjds[-1] - mjds[0]
        res[f"{band}_nobs"] = len(mags)
        res[f"{band}_wmean"] = np.average(mags, weights=1 / (magerrs + 1e-3))
        
        all_peaks.append(peak_mag)
        all_durations.append(res[f"{band}_duration"])
        
        if peak_idx > 0:
            dt = mjds[peak_idx] - mjds[0]
            dflux = flux[peak_idx] - flux[0]
            res[f"{band}_rise"] = dflux / max(dt, 0.1)
            res[f"{band}_time_to_peak"] = dt
        else:
            res[f"{band}_rise"] = 0
            res[f"{band}_time_to_peak"] = 0
        
        if peak_idx < len(mags) - 1:
            dt = mjds[-1] - mjds[peak_idx]
            dflux = flux[-1] - flux[peak_idx]
            res[f"{band}_fall"] = dflux / max(dt, 0.1)
        else:
            res[f"{band}_fall"] = 0
        
        res[f"{band}_stability"] = np.mean(np.abs(np.diff(mags)))
        res[f"{band}_skew"] = stats.skew(mags)
        res[f"{band}_kurt"] = stats.kurtosis(mags)
        
        rise_t = peak_mjd - mjds[0]
        fall_t = mjds[-1] - peak_mjd
        res[f"{band}_asymmetry"] = rise_t / max(fall_t, 0.1)
    
    res["peak_color"] = (all_peaks[0] - all_peaks[1]) if len(all_peaks) == 2 else np.nan
    res["duration_ratio"] = (all_durations[0] / all_durations[1]) if len(all_durations) == 2 and all_durations[1] > 0 else np.nan
    
    try:
        z = float(redshift)
    except:
        z = 0.0
    
    if z > 0.001 and "g_peak" in res:
        dist_pc = (z * 3e5) / 70 * 1e6
        res["abs_mag_g"] = res["g_peak"] - 5 * np.log10(dist_pc) + 5
    else:
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
    
    df_final = df_final.dropna(axis=1, how='all')
    
    mode_str = "strict" if strict_mode else "loose"
    out_file = os.path.join(data_dir, f"ztf_features_{mode_str}.csv")
    df_final.to_csv(out_file, index=False)
    
    print(f"\nDONE ({mode_str})")
    print(f"  Valid objects: {len(df_final)}")
    print(f"  Rejected: {failed}")
    print(f"  Feature count: {len(df_final.columns)}")
    print(f"  Saved to {out_file}")


if __name__ == "__main__":
    run_features(strict_mode=True)