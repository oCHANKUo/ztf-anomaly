"""One-Class SVM anomaly detection."""

import pandas as pd
import numpy as np
import json
import os
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

top_anomalies_count = 15

def detect_ocsvm(data_dir="./data", output_dir="./output"):
    """Run One-Class SVM anomaly detection."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    feature_file = os.path.join(data_dir, "ztf_features_strict.csv")
    df = pd.read_csv(feature_file)
    
    feature_cols = [c for c in df.columns if c not in ["ztf_id", "label"]]
    
    X = df[feature_cols].select_dtypes(include=[np.number])
    numeric_cols = X.columns.tolist()
    
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)
    
    best_nu = 0.05
    best_gamma = 'scale'
    best_score = -np.inf
    
    for nu in [0.03, 0.05, 0.08]:
        for gamma in ['scale', 'auto', 0.01]:
            model = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
            model.fit(X_train)
            val_scores = model.decision_function(X_val)
            median_score = np.median(val_scores)
            if median_score > best_score:
                best_score = median_score
                best_nu = nu
                best_gamma = gamma
    
    model = OneClassSVM(kernel='rbf', nu=best_nu, gamma=best_gamma)
    model.fit(X_scaled)
    
    df["anomaly_label"] = model.predict(X_scaled)
    df["ocsvm_score"] = model.decision_function(X_scaled)
    
    medians = np.median(X_scaled, axis=0)
    
    anomalies = df.sort_values("ocsvm_score").head(top_anomalies_count)
    
    print(f"--- One-Class SVM (nu={best_nu}, gamma={best_gamma}) ---")
    print(f"Dataset: {len(df)} objects")
    
    results = []
    for idx, row in anomalies.iterrows():
        x_scaled = scaler.transform(imputer.transform([row[numeric_cols].values]))[0]
        distances = np.abs(x_scaled - medians)
        top_idx = np.argmax(distances)
        top_feature = numeric_cols[top_idx]
        
        top3_idx = np.argsort(distances)[-3:][::-1]
        top3_features = [(numeric_cols[i], distances[i]) for i in top3_idx]
        
        print(f"ID: {row['ztf_id']} | Type: {row['label']}")
        print(f"  Distance: {row['ocsvm_score']:.4f} | Extreme: {top_feature}")
        
        results.append({
            "ztf_id": row["ztf_id"],
            "label": row["label"],
            "ocsvm_score": float(row["ocsvm_score"]),
            "primary_feature": top_feature,
            "top3_features": top3_features
        })
    
    out_path = os.path.join(output_dir, "top_anomalies_ocsvm.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    detect_ocsvm()