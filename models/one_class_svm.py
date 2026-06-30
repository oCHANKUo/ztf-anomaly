"""One-Class SVM anomaly detection — ZTF DR3 style."""

import pandas as pd
import numpy as np
import json
import os
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump


def detect_ocsvm(data_dir="./data", output_dir="./output", top_n=50):
    """Run One-Class SVM anomaly detection."""
    
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
    
    # OC-SVM is slow — subsample for training if large dataset
    n_train = min(5000, len(X_scaled))
    if len(X_scaled) > n_train:
        print(f"Subsampling to {n_train} for OC-SVM training")
        np.random.seed(42)
        train_idx = np.random.choice(len(X_scaled), n_train, replace=False)
        X_train = X_scaled[train_idx]
    else:
        X_train = X_scaled
    
    # Fit model
    # ZTF paper: gamma=0.01, RBF kernel
    model = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.05, verbose=True)
    model.fit(X_train)
    
    # Score all data
    df["ocsvm_score"] = model.decision_function(X_scaled)
    df["ocsvm_outlier"] = model.predict(X_scaled)
    
    # Save
    models_dir = os.path.join(output_dir, "models", "one_class_svm")
    os.makedirs(models_dir, exist_ok=True)
    dump(model, os.path.join(models_dir, "model.joblib"))
    dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    dump(imputer, os.path.join(models_dir, "imputer.joblib"))
    with open(os.path.join(models_dir, "features.json"), "w") as f:
        json.dump(numeric_cols, f)
    
    # Top anomalies (most negative score = most anomalous)
    anomalies = df.nsmallest(top_n, "ocsvm_score")
    
    print(f"\n--- One-Class SVM ---")
    print(f"Dataset: {len(df)} objects | Train subsample: {len(X_train)}")
    print(f"Outliers: {(df['ocsvm_outlier'] == -1).sum()}")
    
    results = []
    for idx, row in anomalies.iterrows():
        print(f"\nRank {len(results)+1}: {row['ztf_id']} | {row['label']}")
        print(f"  OC-SVM score: {row['ocsvm_score']:.4f}")
        
        results.append({
            "rank": len(results) + 1,
            "ztf_id": row["ztf_id"],
            "label": row["label"],
            "ocsvm_score": float(row["ocsvm_score"]),
        })
    
    out_path = os.path.join(output_dir, "anomalies_ocsvm.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    df[["ztf_id", "ocsvm_score", "ocsvm_outlier"]].to_csv(
        os.path.join(output_dir, "scores_ocsvm.csv"), index=False
    )
    
    print(f"\nSaved top {top_n} to {out_path}")
    
    return df[["ztf_id", "ocsvm_score", "ocsvm_outlier"]]


if __name__ == "__main__":
    detect_ocsvm()