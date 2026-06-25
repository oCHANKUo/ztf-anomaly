"""Consensus merging of anomaly detection results."""

import json
import pandas as pd
import numpy as np
from scipy import stats
import os


def merge_results(output_dir="./output"):
    """Merge results from all three models and find consensus anomalies."""
    
    files = {
        "IF": os.path.join(output_dir, "top_anomalies_if.json"),
        "AE": os.path.join(output_dir, "top_anomalies_ae.json"),
        "SVM": os.path.join(output_dir, "top_anomalies_ocsvm.json"),
    }
    
    data = {}
    for name, path in files.items():
        if os.path.exists(path):
            with open(path, "r") as f:
                data[name] = json.load(f)
        else:
            print(f"Warning: {path} not found")
            data[name] = []
    
    dfs = {}
    
    if data["IF"]:
        df_if = pd.DataFrame(data["IF"])
        df_if["if_rank"] = stats.rankdata(df_if["raw_score"], method="min")
        df_if["if_norm"] = (df_if["if_rank"] - 1) / (len(df_if) - 1) if len(df_if) > 1 else 0
        dfs["IF"] = df_if[["ztf_id", "label", "raw_score", "if_norm"]]
    
    if data["AE"]:
        df_ae = pd.DataFrame(data["AE"])
        df_ae["ae_rank"] = stats.rankdata(-df_ae["ae_score"], method="min")
        df_ae["ae_norm"] = (df_ae["ae_rank"] - 1) / (len(df_ae) - 1) if len(df_ae) > 1 else 0
        dfs["AE"] = df_ae[["ztf_id", "ae_score", "ae_norm"]]
    
    if data["SVM"]:
        df_svm = pd.DataFrame(data["SVM"])
        df_svm["svm_rank"] = stats.rankdata(df_svm["ocsvm_score"], method="min")
        df_svm["svm_norm"] = (df_svm["svm_rank"] - 1) / (len(df_svm) - 1) if len(df_svm) > 1 else 0
        dfs["SVM"] = df_svm[["ztf_id", "ocsvm_score", "svm_norm"]]
    
    if not dfs:
        print("No model results found.")
        return
    
    merged = None
    for name, df in dfs.items():
        if merged is None:
            merged = df.copy()
        else:
            merged = merged.merge(df, on="ztf_id", how="outer")
    
    for col in ["if_norm", "ae_norm", "svm_norm"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(1.0)
    
    norm_cols = [c for c in ["if_norm", "ae_norm", "svm_norm"] if c in merged.columns]
    merged["consensus_score"] = merged[norm_cols].mean(axis=1)
    
    merged["vote_count"] = (
        merged.get("if_norm", pd.Series([1.0]*len(merged))).lt(1.0).astype(int) +
        merged.get("ae_norm", pd.Series([1.0]*len(merged))).lt(1.0).astype(int) +
        merged.get("svm_norm", pd.Series([1.0]*len(merged))).lt(1.0).astype(int)
    )
    
    merged = merged.sort_values(["vote_count", "consensus_score"], ascending=[False, True])
    
    print("\n--- Consensus Anomalies ---")
    print("Rank | ID | Votes | Consensus | IF | AE | SVM")
    print("-" * 70)
    
    for i, (_, row) in enumerate(merged.head(15).iterrows(), 1):
        if_score = f"{row['raw_score']:.3f}" if pd.notna(row.get('raw_score')) else "N/A"
        ae_score = f"{row['ae_score']:.3f}" if pd.notna(row.get('ae_score')) else "N/A"
        svm_score = f"{row['ocsvm_score']:.3f}" if pd.notna(row.get('ocsvm_score')) else "N/A"
        
        print(f"{i:2d}   | {row['ztf_id']} | {int(row['vote_count'])}     | "
              f"{row['consensus_score']:.3f}     | {if_score:>7s} | "
              f"{ae_score:>7s} | {svm_score:>8s}")
    
    high_conf = merged[(merged["vote_count"] >= 2) & (merged["consensus_score"] < 0.3)]
    print(f"\nHigh-confidence anomalies (2+ votes, score < 0.3): {len(high_conf)}")
    
    out_path = os.path.join(output_dir, "consensus_anomalies.csv")
    merged.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    merge_results()