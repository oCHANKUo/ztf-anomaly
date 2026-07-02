"""One-Class SVM anomaly detection.

Fits a boundary around the "normal" region of feature space; points
that fall outside it (negative decision function) are anomalies.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from joblib import dump


def detect_ocsvm(data_dir="./data", output_dir="./output", top_n=50):

    os.makedirs(output_dir, exist_ok=True)

    feature_file = os.path.join(data_dir, "ztf_features_strict.csv")
    df = pd.read_csv(feature_file)

    meta_cols = ["ztf_id", "label", "redshift"]
    dq_cols = [c for c in df.columns if c.startswith("dq_")]
    ml_cols = [c for c in df.columns if c not in meta_cols + dq_cols]

    volume_metrics = ["g_nobs", "r_nobs", "global_amplitude", "g_amplitude", "r_amplitude"]
    ml_cols = [c for c in ml_cols if c not in volume_metrics]

    print(f"Features used: {len(ml_cols)} (excluded {len(dq_cols)} DQ features)")

    X = df[ml_cols].select_dtypes(include=[np.number])
    numeric_cols = X.columns.tolist()

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    n_train = min(5000, len(X_scaled))
    if len(X_scaled) > n_train:
        print(f"Subsampling to {n_train} for OC-SVM training")
        np.random.seed(42)
        train_idx = np.random.choice(len(X_scaled), n_train, replace=False)
        X_train = X_scaled[train_idx]
    else:
        X_train = X_scaled

    model = OneClassSVM(kernel="rbf", gamma=0.01, nu=0.05, verbose=True)
    model.fit(X_train)

    df["ocsvm_score"] = model.decision_function(X_scaled)
    df["ocsvm_outlier"] = model.predict(X_scaled)

    # === Save portable bundle ===
    models_dir = os.path.join(output_dir, "models", "one_class_svm")
    os.makedirs(models_dir, exist_ok=True)

    bundle = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "feature_names": numeric_cols,
    }
    dump(bundle, os.path.join(models_dir, "ocsvm_bundle.joblib"))

    dump(model, os.path.join(models_dir, "model.joblib"))
    dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    dump(imputer, os.path.join(models_dir, "imputer.joblib"))
    with open(os.path.join(models_dir, "features.json"), "w") as f:
        json.dump(numeric_cols, f)

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

    with open(os.path.join(output_dir, "anomalies_ocsvm.json"), "w") as f:
        json.dump(results, f, indent=2)

    df[["ztf_id", "ocsvm_score", "ocsvm_outlier"]].to_csv(
        os.path.join(output_dir, "scores_ocsvm.csv"), index=False
    )

    print(f"\nSaved top {top_n} to {os.path.join(output_dir, 'anomalies_ocsvm.json')}")
    print(f"Saved portable model bundle to {os.path.join(models_dir, 'ocsvm_bundle.joblib')}")

    return df[["ztf_id", "ocsvm_score", "ocsvm_outlier"]]


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
    detect_ocsvm()