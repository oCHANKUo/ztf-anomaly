"""Isolation Forest anomaly detection — ZTF DR3 style."""

import pandas as pd
import numpy as np
import json
import os
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def detect(data_dir="./data", output_dir="./output", top_n=50):
    """Run Isolation Forest anomaly detection."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    feature_file = os.path.join(data_dir, "ztf_features_strict.csv")
    df = pd.read_csv(feature_file)
    
    # === CRITICAL: Exclude DQ features from ML input ===
    meta_cols = ["ztf_id", "label", "redshift"]
    dq_cols = [c for c in df.columns if c.startswith("dq_")]
    ml_cols = [c for c in df.columns if c not in meta_cols + dq_cols]
    
    print(f"Features used: {len(ml_cols)} (excluded {len(dq_cols)} DQ features)")
    
    X = df[ml_cols].select_dtypes(include=[np.number])
    numeric_cols = X.columns.tolist()
    
    # Impute and scale
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Fit on ALL data (unsupervised — no train/val split needed for contamination tuning)
    # ZTF paper: 1000 trees, subsample 1000
    model = IsolationForest(
        n_estimators=1000,
        max_samples=min(1000, len(X_scaled)),
        contamination='auto',  # or 0.05 for ~5% outliers
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)
    
    # Scores: higher = more normal, lower = more anomalous
    df["if_score"] = model.decision_function(X_scaled)
    df["if_outlier"] = model.predict(X_scaled)  # -1 = outlier, 1 = inlier
    
    # Feature deviations from median (for interpretation)
    medians = np.median(X_scaled, axis=0)
    
    # Save model
    models_dir = os.path.join(output_dir, "models", "isolation_forest")
    os.makedirs(models_dir, exist_ok=True)
    dump(model, os.path.join(models_dir, "model.joblib"))
    dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    dump(imputer, os.path.join(models_dir, "imputer.joblib"))
    with open(os.path.join(models_dir, "features.json"), "w") as f:
        json.dump(numeric_cols, f)
    
    # Get top anomalies
    anomalies = df.nsmallest(top_n, "if_score")
    
    print(f"\n--- Isolation Forest ---")
    print(f"Dataset: {len(df)} objects | Features: {len(numeric_cols)}")
    print(f"Outliers (auto contamination): {(df['if_outlier'] == -1).sum()}")
    
    results = []
    for idx, row in anomalies.iterrows():
        x_scaled = scaler.transform(imputer.transform([X.loc[idx].values]))[0]
        deviations = np.abs(x_scaled - medians)
        top3_idx = np.argsort(deviations)[-3:][::-1]
        top3_features = [(numeric_cols[i], float(deviations[i])) for i in top3_idx]
        
        is_dq = any(f.startswith("dq_") for f, _ in top3_features[:1])
        
        print(f"\nRank {len(results)+1}: {row['ztf_id']} | {row['label']}")
        print(f"  IF score: {row['if_score']:.4f}")
        print(f"  Top features: {', '.join([f'{f} ({d:.2f}σ)' for f, d in top3_features])}")
        
        results.append({
            "rank": len(results) + 1,
            "ztf_id": row["ztf_id"],
            "label": row["label"],
            "if_score": float(row["if_score"]),
            "top3_features": top3_features,
        })
    
    # Save results
    out_path = os.path.join(output_dir, "anomalies_if.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save full scores for ensemble
    df[["ztf_id", "if_score", "if_outlier"]].to_csv(
        os.path.join(output_dir, "scores_if.csv"), index=False
    )
    
    print(f"\nSaved top {top_n} to {out_path}")
    print(f"Saved all scores to {os.path.join(output_dir, 'scores_if.csv')}")
    
    return df[["ztf_id", "if_score", "if_outlier"]]


if __name__ == "__main__":
    detect()