"""Isolation Forest anomaly detection."""

import pandas as pd
import numpy as np
import json
import os
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

top_anomalies_count = 15


def detect(data_dir="./data", output_dir="./output"):
    """Run Isolation Forest anomaly detection."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    feature_file = os.path.join(data_dir, "ztf_features_strict.csv")
    df = pd.read_csv(feature_file)
    
    feature_cols = [c for c in df.columns if c not in ["ztf_id", "label"]]
    
    X = df[feature_cols].select_dtypes(include=[np.number])
    numeric_cols = X.columns.tolist()
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
    
    best_model = None
    best_contamination = 0.05
    
    for cont in [0.02, 0.05, 0.08, 0.10]:
        model = IsolationForest(contamination=cont, random_state=42, n_estimators=200)
        model.fit(X_train)
        val_scores = model.score_samples(X_val)
        n_anom = np.sum(val_scores < np.percentile(val_scores, cont * 100))
        if abs(n_anom / len(val_scores) - cont) < 0.02:
            best_contamination = cont
            best_model = model
            break
    
    if best_model is None:
        best_model = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
        best_model.fit(X_scaled)
    
    df["anomaly_label"] = best_model.predict(X_scaled)
    df["raw_score"] = best_model.score_samples(X_scaled)
    
    medians = np.median(X_scaled, axis=0)
    
    anomalies = df.sort_values("raw_score").head(top_anomalies_count)
    
    print(f"--- Isolation Forest (contamination={best_contamination}) ---")
    print(f"Dataset: {len(df)} objects")
    
    results = []
    for idx, row in anomalies.iterrows():
        x_scaled = scaler.transform(imputer.transform([row[numeric_cols].values]))[0]
        deviations = np.abs(x_scaled - medians)
        top_idx = np.argmax(deviations)
        top_feature = numeric_cols[top_idx]
        deviation = deviations[top_idx]
        
        top3_idx = np.argsort(deviations)[-3:][::-1]
        top3_features = [(numeric_cols[i], deviations[i]) for i in top3_idx]
        
        is_dq = top_feature.startswith("dq_")
        
        print(f"ID: {row['ztf_id']} | Type: {row['label']}")
        print(f"  Score: {row['raw_score']:.3f} | Primary: {top_feature} ({deviation:.1f}σ)")
        print(f"  DQ-driven: {'YES' if is_dq else 'NO'}")
        
        results.append({
            "ztf_id": row["ztf_id"],
            "label": row["label"],
            "raw_score": float(row["raw_score"]),
            "primary_feature": top_feature,
            "deviation": float(deviation),
            "is_dq_driven": bool(is_dq),
            "top3_features": top3_features
        })
    
    out_path = os.path.join(output_dir, "top_anomalies_if.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    n_dq = sum(1 for r in results if r["is_dq_driven"])
    print(f"\nData-quality driven: {n_dq}/{len(results)}")
    print(f"Astro-physics driven: {len(results) - n_dq}/{len(results)}")
    print(f"Saved to {out_path}")
    # Save model and preprocessors for reuse
    models_dir = os.path.join(output_dir, "models", "isolation_forest")
    os.makedirs(models_dir, exist_ok=True)
    dump(best_model, os.path.join(models_dir, "isolation_forest_model.joblib"))
    dump(scaler, os.path.join(models_dir, "isolation_forest_scaler.joblib"))
    dump(imputer, os.path.join(models_dir, "isolation_forest_imputer.joblib"))
    with open(os.path.join(models_dir, "isolation_forest_features.json"), "w") as f:
        json.dump(numeric_cols, f)
    print(f"Saved model and preprocessors to {models_dir}")


if __name__ == "__main__":
    detect()