"""Quality checks and diagnostics for ZTF light curves."""

import pandas as pd
import numpy as np
import pickle
import os


def run_quality_check(data_dir="./data"):
    """Run comprehensive quality check on downloaded light curves."""
    
    lcs_path = os.path.join(data_dir, "ztf_lcs_all.pkl")
    with open(lcs_path, "rb") as f:
        all_lcs = pickle.load(f)
    
    print(f"\nTotal light curves: {len(all_lcs)}")
    
    metrics = []
    for ztf_id, df in all_lcs.items():
        mag_col = "magpsf" if "magpsf" in df.columns else "magpsf_corr"
        
        n_total = len(df)
        n_g = len(df[df["fid"] == 1])
        n_r = len(df[df["fid"] == 2])
        
        mjds = df["mjd"].sort_values().values
        span = mjds[-1] - mjds[0] if len(mjds) > 1 else 0
        gaps = np.diff(mjds) if len(mjds) > 1 else []
        max_gap = np.max(gaps) if len(gaps) > 0 else 0
        median_gap = np.median(gaps) if len(gaps) > 0 else 0
        
        mags = df[mag_col].values
        mag_range = np.max(mags) - np.min(mags) if len(mags) > 0 else 0
        
        # Track imputation if present
        frac_imputed = df["imputed"].sum() / len(df) if "imputed" in df.columns else 0.0
        
        metrics.append({
            "ztf_id": ztf_id,
            "n_total": n_total,
            "n_g": n_g,
            "n_r": n_r,
            "span": span,
            "max_gap": max_gap,
            "median_gap": median_gap,
            "mag_range": mag_range,
            "frac_imputed": frac_imputed,
        })
    
    df_metrics = pd.DataFrame(metrics)
    
    print(f"\n--- Quality Statistics ---")
    print(f"  Mean points per object: {df_metrics['n_total'].mean():.1f}")
    print(f"  Mean span: {df_metrics['span'].mean():.1f} days")
    print(f"  Mean max gap: {df_metrics['max_gap'].mean():.1f} days")
    print(f"  Mean median gap: {df_metrics['median_gap'].mean():.1f} days")
    print(f"  Mean mag range: {df_metrics['mag_range'].mean():.2f}")
    print(f"  Mean frac imputed: {df_metrics['frac_imputed'].mean():.3f}")
    
    print(f"\n--- Potential Issues ---")
    print(f"  Objects with <10 points: {(df_metrics['n_total'] < 10).sum()}")
    print(f"  Objects with >100 day gap: {(df_metrics['max_gap'] > 100).sum()}")
    print(f"  Objects with >200 day gap: {(df_metrics['max_gap'] > 200).sum()}")
    print(f"  Objects with <0.5 mag range: {(df_metrics['mag_range'] < 0.5).sum()}")
    print(f"  Single-band objects: {((df_metrics['n_g'] == 0) | (df_metrics['n_r'] == 0)).sum()}")
    print(f"  Objects with >10% imputed: {(df_metrics['frac_imputed'] > 0.10).sum()}")
    
    report_path = os.path.join(data_dir, "quality_report.csv")
    df_metrics.to_csv(report_path, index=False)
    print(f"\nSaved quality report to {report_path}")


if __name__ == "__main__":
    run_quality_check()