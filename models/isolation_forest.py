"""Isolation Forest anomaly detection.

This is the primary/simplest detector — trees partition the feature
space randomly, and points that get isolated in very few splits
(short average path length) are scored as anomalous.
"""

import pandas as pd
import numpy as np
import json
import os
from joblib import dump
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def detect(data_dir="./data", output_dir="./output", top_n=50):

    os.makedirs(output_dir, exist_ok=True)

    feature_file = os.path.join(data_dir, "ztf_features_strict.csv")
    df = pd.read_csv(feature_file)

    # DQ features are excluded from the ML input — they're for
    # post-hoc sanity checks only, not for training the model.
    meta_cols = ["ztf_id", "label", "redshift"]
    dq_cols = [c for c in df.columns if c.startswith("dq_")]
    ml_cols = [c for c in df.columns if c not in meta_cols + dq_cols]

    print(f"Features used: {len(ml_cols)} (excluded {len(dq_cols)} DQ features)")

    X = df[ml_cols].select_dtypes(include=[np.number])
    numeric_cols = X.columns.tolist()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    model = IsolationForest(
        n_estimators=1000,
        max_samples=min(1000, len(X_scaled)),
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # decision_function: higher = more normal, lower = more anomalous
    df["if_score"] = model.decision_function(X_scaled)
    df["if_outlier"] = model.predict(X_scaled)  # -1 = outlier, 1 = inlier

    medians = np.median(X_scaled, axis=0)

    # === Save a single portable bundle (model + preprocessing + feature names) ===
    models_dir = os.path.join(output_dir, "models", "isolation_forest")
    os.makedirs(models_dir, exist_ok=True)

    bundle = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "feature_names": numeric_cols,
    }
    dump(bundle, os.path.join(models_dir, "if_bundle.joblib"))

    # Keep individual files too, in case you want them separately
    dump(model, os.path.join(models_dir, "model.joblib"))
    dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    dump(imputer, os.path.join(models_dir, "imputer.joblib"))
    with open(os.path.join(models_dir, "features.json"), "w") as f:
        json.dump(numeric_cols, f)

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

        print(f"\nRank {len(results)+1}: {row['ztf_id']} | {row['label']}")
        print(f"  IF score: {row['if_score']:.4f}")
        print(f"  Top features: {', '.join([f'{f} ({d:.2f}sigma)' for f, d in top3_features])}")

        results.append({
            "rank": len(results) + 1,
            "ztf_id": row["ztf_id"],
            "label": row["label"],
            "if_score": float(row["if_score"]),
            "top3_features": top3_features,
        })

    with open(os.path.join(output_dir, "anomalies_if.json"), "w") as f:
        json.dump(results, f, indent=2)

    df[["ztf_id", "if_score", "if_outlier"]].to_csv(
        os.path.join(output_dir, "scores_if.csv"), index=False
    )

    print(f"\nSaved top {top_n} to {os.path.join(output_dir, 'anomalies_if.json')}")
    print(f"Saved portable model bundle to {os.path.join(models_dir, 'if_bundle.joblib')}")

    return df[["ztf_id", "if_score", "if_outlier"]]


def load_bundle(bundle_path):
    """Load a saved bundle elsewhere and get back a ready-to-use predict function."""
    from joblib import load
    bundle = load(bundle_path)

    def predict(feature_df):
        X = feature_df[bundle["feature_names"]]
        X_imp = bundle["imputer"].transform(X)
        X_scaled = bundle["scaler"].transform(X_imp)
        return bundle["model"].decision_function(X_scaled)

    return predict


if __name__ == "__main__":
    detect()