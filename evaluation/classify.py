"""Classify anomalies as real astrophysical vs data artifacts."""

import json
import pandas as pd
import numpy as np
import pickle
import os


def classify_anomalies(data_dir="./data", output_dir="./output"):
    """Classify each anomaly as real, borderline, or data artifact."""
    
    json_path = os.path.join(output_dir, "top_anomalies_if.json")
    with open(json_path, "r") as f:
        anomalies = json.load(f)
    
    feature_file = os.path.join(data_dir, "ztf_features_strict.csv")
    df_features = pd.read_csv(feature_file)
    df_features = df_features.set_index("ztf_id")
    
    pkl_path = os.path.join(data_dir, "ztf_lcs_all.pkl")
    with open(pkl_path, "rb") as f:
        all_lcs = pickle.load(f)
    
    print("=" * 70)
    print("ANOMALY CLASSIFICATION")
    print("=" * 70)
    
    real_count = 0
    borderline_count = 0
    artifact_count = 0
    
    for anom in anomalies[:10]:
        ztf_id = anom["ztf_id"]
        score = anom.get("raw_score", 0)
        primary = anom.get("primary_feature", "N/A")
        
        if ztf_id not in df_features.index:
            print(f"\n{ztf_id}: NOT IN FEATURES")
            continue
        
        row = df_features.loc[ztf_id]
        
        n_total = row.get("dq_total_points", 0)
        n_g = row.get("dq_g_frac", 0) * n_total
        n_r = row.get("dq_r_frac", 0) * n_total
        max_gap = row.get("dq_max_gap", 0)
        span = row.get("dq_span", 0)
        frac_imp = row.get("dq_frac_imputed", 0)
        
        if ztf_id in all_lcs:
            df_lc = all_lcs[ztf_id]
            mag_col = "magpsf" if "magpsf" in df_lc.columns else "magpsf_corr"
            mags = df_lc[mag_col].values
            mag_range = np.max(mags) - np.min(mags)
            peak_idx = np.argmin(mags)
            n_pre = peak_idx
            n_post = len(mags) - peak_idx - 1
        else:
            mag_range = 0
            n_pre = n_post = 0
        
        issues = []
        if max_gap > 80: issues.append("large_gap")
        if n_g < 10 or n_r < 10: issues.append("sparse_band")
        if frac_imp > 0.05: issues.append("heavily_imputed")
        if n_pre < 5 or n_post < 5: issues.append("incomplete_peak")
        if mag_range < 1.0: issues.append("flat_curve")
        
        if len(issues) >= 3:
            verdict = "DATA ARTIFACT"
            confidence = "HIGH"
            artifact_count += 1
        elif len(issues) >= 1:
            verdict = "BORDERLINE"
            confidence = "MEDIUM"
            borderline_count += 1
        else:
            verdict = "REAL ASTROPHYSICAL"
            confidence = "HIGH"
            real_count += 1
        
        print(f"\n{ztf_id} | Score:{score:.3f} | Key:{primary}")
        print(f"  n={n_total:.0f} (g={n_g:.0f}, r={n_r:.0f}) | span={span:.0f}d | max_gap={max_gap:.0f}d | Δmag={mag_range:.2f}")
        print(f"  Pre-peak:{n_pre} | Post-peak:{n_post} | Imputed:{frac_imp:.1%}")
        print(f"  Issues: {', '.join(issues) if issues else 'NONE'}")
        print(f"  VERDICT: {verdict} (confidence: {confidence})")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Real astrophysical: {real_count}")
    print(f"  Borderline: {borderline_count}")
    print(f"  Data artifacts: {artifact_count}")
    print(f"\nGUIDELINES:")
    print(f"  Real astrophysical → Keep, these are your true anomalies")
    print(f"  Borderline → Keep but flag, verify with visual inspection")
    print(f"  Data artifacts → Remove from results")
    print("=" * 70)
    
    report_path = os.path.join(output_dir, "anomaly_classification.txt")
    with open(report_path, "w") as f:
        f.write("Anomaly Classification Report\n")
        f.write("=" * 70 + "\n")
        f.write(f"Real astrophysical: {real_count}\n")
        f.write(f"Borderline: {borderline_count}\n")
        f.write(f"Data artifacts: {artifact_count}\n")
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    classify_anomalies()