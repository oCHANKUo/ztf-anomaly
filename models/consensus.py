"""Ensemble anomaly detection with DQ filtering — ZTF DR3 style."""

import pandas as pd
import numpy as np
import json
import os


def compute_dq_risk(df, dq_cols):
    """
    Compute data quality risk score (0 = good, 1 = terrible).
    Higher = more likely to be bogus due to data quality.
    """
    risk = np.zeros(len(df))
    
    if 'dq_max_gap' in dq_cols:
        gap_risk = np.clip(df['dq_max_gap'] / 100.0, 0, 1)
        risk += gap_risk * 0.25
    
    if 'dq_total_points' in dq_cols:
        n_risk = np.clip(1.0 - (df['dq_total_points'] / 20.0), 0, 1)
        risk += n_risk * 0.25
    
    if 'dq_frac_imputed' in dq_cols:
        imp_risk = np.clip(df['dq_frac_imputed'] / 0.15, 0, 1)
        risk += imp_risk * 0.20
    
    if 'dq_sampling_density' in dq_cols:
        density_risk = np.clip(1.0 - (df['dq_sampling_density'] / 0.1), 0, 1)
        risk += density_risk * 0.15
    
    if 'dq_span' in dq_cols:
        span_risk = np.clip(1.0 - (df['dq_span'] / 10.0), 0, 1)
        risk += span_risk * 0.15
    
    return np.clip(risk, 0, 1)


def bogus_filter(df):
    """
    ZTF DR3 inspired: high linear chi2 + low periodogram amp → bogus.
    Returns boolean series: True = likely bogus.
    """
    required = ['global_linear_chi2', 'g_periodogram_amp', 'r_periodogram_amp']
    if not all(r in df.columns for r in required):
        print("Warning: Missing features for bogus filter, skipping")
        return pd.Series(False, index=df.index)
    
    # High chi2 = erratic data that doesn't fit a line
    chi2_thresh = df['global_linear_chi2'].quantile(0.85)
    chi2_high = df['global_linear_chi2'] > chi2_thresh
    
    # Low periodogram amplitude = no coherent structure
    g_amp_thresh = df['g_periodogram_amp'].quantile(0.25)
    r_amp_thresh = df['r_periodogram_amp'].quantile(0.25)
    amp_low = (df['g_periodogram_amp'] < g_amp_thresh) & (df['r_periodogram_amp'] < r_amp_thresh)
    
    return chi2_high & amp_low


def ensemble(data_dir="./data", output_dir="./output", top_n=50):
    """Combine IF + AE + OC-SVM with DQ filtering."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original features (for DQ info)
    feature_file = os.path.join(data_dir, "ztf_features_strict.csv")
    df_full = pd.read_csv(feature_file)
    
    # Load scores from each model
    scores_if = pd.read_csv(os.path.join(output_dir, "scores_if.csv"))
    scores_ae = pd.read_csv(os.path.join(output_dir, "scores_ae.csv"))
    scores_ocsvm = pd.read_csv(os.path.join(output_dir, "scores_ocsvm.csv"))
    
    # Merge
    df = df_full.merge(scores_if, on="ztf_id", how="left")
    df = df.merge(scores_ae, on="ztf_id", how="left")
    df = df.merge(scores_ocsvm, on="ztf_id", how="left")
    
    # Normalize scores to [0, 1] where 1 = most anomalous
    # IF: decision_function, higher = more normal
    if_min, if_max = df["if_score"].min(), df["if_score"].max()
    df["if_anomaly"] = 1.0 - (df["if_score"] - if_min) / (if_max - if_min + 1e-10)
    
    # AE: MSE, higher = more anomalous
    ae_min, ae_max = df["ae_mse"].min(), df["ae_mse"].max()
    df["ae_anomaly"] = (df["ae_mse"] - ae_min) / (ae_max - ae_min + 1e-10)
    
    # OC-SVM: decision_function, higher = more normal
    svm_min, svm_max = df["ocsvm_score"].min(), df["ocsvm_score"].max()
    df["ocsvm_anomaly"] = 1.0 - (df["ocsvm_score"] - svm_min) / (svm_max - svm_min + 1e-10)
    
    # Simple average ensemble
    df["ensemble_score"] = (df["if_anomaly"] + df["ae_anomaly"] + df["ocsvm_anomaly"]) / 3.0
    
    # === DQ FILTERING ===
    dq_cols = [c for c in df.columns if c.startswith("dq_")]
    df["dq_risk"] = compute_dq_risk(df, dq_cols)
    df["likely_bogus"] = bogus_filter(df)
    
    # Final score: ensemble downweighted by DQ risk and bogus flag
    # High DQ risk → lower final score (less likely to be real anomaly)
    # Likely bogus → penalized
    bogus_penalty = df["likely_bogus"].astype(float) * 0.5
    df["final_score"] = df["ensemble_score"] * (1.0 - df["dq_risk"] * 0.5) * (1.0 - bogus_penalty)
    
    # Rank by final score (higher = more anomalous, after DQ filtering)
    # Actually we want to RANK anomalies, so sort descending by final_score
    df["final_rank"] = df["final_score"].rank(ascending=False, method="min")
    
    # Save full results
    out_cols = [
        "ztf_id", "label", "redshift",
        "if_anomaly", "ae_anomaly", "ocsvm_anomaly", "ensemble_score",
        "dq_risk", "likely_bogus", "final_score", "final_rank"
    ] + dq_cols
    df[out_cols].to_csv(os.path.join(output_dir, "ensemble_results.csv"), index=False)
    
    # Top candidates
    top_candidates = df.nsmallest(top_n, "final_rank")
    
    print(f"\n{'='*60}")
    print(f"ENSEMBLE RESULTS")
    print(f"{'='*60}")
    print(f"Total objects: {len(df)}")
    print(f"Likely bogus flagged: {df['likely_bogus'].sum()} ({df['likely_bogus'].mean()*100:.1f}%)")
    print(f"High DQ risk (>0.5): {(df['dq_risk'] > 0.5).sum()}")
    print(f"\n--- TOP {top_n} ANOMALY CANDIDATES ---")
    
    results = []
    for idx, row in top_candidates.iterrows():
        print(f"\nRank {int(row['final_rank'])}: {row['ztf_id']} | {row['label']}")
        print(f"  Final score: {row['final_score']:.4f}")
        print(f"  Ensemble: {row['ensemble_score']:.4f} | "
              f"IF: {row['if_anomaly']:.4f} | AE: {row['ae_anomaly']:.4f} | "
              f"SVM: {row['ocsvm_anomaly']:.4f}")
        print(f"  DQ risk: {row['dq_risk']:.3f} | Likely bogus: {'YES' if row['likely_bogus'] else 'NO'}")
        
        results.append({
            "rank": int(row["final_rank"]),
            "ztf_id": row["ztf_id"],
            "label": row["label"],
            "final_score": float(row["final_score"]),
            "ensemble_score": float(row["ensemble_score"]),
            "if_anomaly": float(row["if_anomaly"]),
            "ae_anomaly": float(row["ae_anomaly"]),
            "ocsvm_anomaly": float(row["ocsvm_anomaly"]),
            "dq_risk": float(row["dq_risk"]),
            "likely_bogus": bool(row["likely_bogus"]),
        })
    
    # Save top results
    out_path = os.path.join(output_dir, "top_anomalies_ensemble.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved to {out_path}")
    print(f"Full results: {os.path.join(output_dir, 'ensemble_results.csv')}")
    
    # Summary by label
    print(f"\n--- TOP {top_n} BY LABEL ---")
    print(top_candidates["label"].value_counts().to_string())
    
    # DQ analysis of top candidates
    print(f"\n--- DQ ANALYSIS OF TOP {top_n} ---")
    print(f"Mean DQ risk: {top_candidates['dq_risk'].mean():.3f}")
    print(f"Bogus flagged: {top_candidates['likely_bogus'].sum()}")
    
    return df


if __name__ == "__main__":
    ensemble()