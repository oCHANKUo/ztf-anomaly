"""Autoencoder anomaly detection — ZTF DR3 style."""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import dump


class AnomalyAE(nn.Module):
    def __init__(self, input_dim, encoding_dim=None):
        super(AnomalyAE, self).__init__()
        if encoding_dim is None:
            encoding_dim = max(8, input_dim // 4)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def detect_ae(data_dir="./data", output_dir="./output", top_n=50, epochs=200):
    """Run Autoencoder anomaly detection."""
    
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
    
    # Train/val split
    np.random.seed(42)
    n = len(X_scaled)
    perm = np.random.permutation(n)
    train_idx = perm[:int(0.8 * n)]
    val_idx = perm[int(0.8 * n):]
    
    X_train = torch.FloatTensor(X_scaled[train_idx])
    X_val = torch.FloatTensor(X_scaled[val_idx])
    X_full = torch.FloatTensor(X_scaled)
    
    # Model
    input_dim = len(numeric_cols)
    model = AnomalyAE(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    best_val_loss = np.inf
    patience = 30
    patience_counter = 0
    best_state = None
    
    print(f"\nTraining Autoencoder (input_dim={input_dim})...")
    
    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, X_train)
        loss.backward()
        optimizer.step()
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, X_val).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d} | Train: {loss.item():.6f} | Val: {val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    # Load best
    model.load_state_dict(best_state)
    model.eval()
    
    # Reconstruction error on full dataset
    with torch.no_grad():
        predictions = model(X_full)
        per_feature_error = torch.abs(X_full - predictions).numpy()
        mse = np.mean((X_full.numpy() - predictions.numpy())**2, axis=1)
    
    df["ae_mse"] = mse
    
    # Save
    models_dir = os.path.join(output_dir, "models", "autoencoder")
    os.makedirs(models_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(models_dir, "model.pth"))
    dump(scaler, os.path.join(models_dir, "scaler.joblib"))
    dump(imputer, os.path.join(models_dir, "imputer.joblib"))
    with open(os.path.join(models_dir, "features.json"), "w") as f:
        json.dump(numeric_cols, f)
    
    # Top anomalies
    anomalies = df.nlargest(top_n, "ae_mse")
    
    print(f"\n--- Autoencoder ---")
    print(f"Dataset: {len(df)} objects | Best val loss: {best_val_loss:.6f}")
    
    results = []
    for idx, row in anomalies.iterrows():
        x_idx = df.index.get_loc(idx)
        feature_errors = per_feature_error[x_idx]
        top3_idx = np.argsort(feature_errors)[-3:][::-1]
        top3_features = [(numeric_cols[i], float(feature_errors[i])) for i in top3_idx]
        
        print(f"\nRank {len(results)+1}: {row['ztf_id']} | {row['label']}")
        print(f"  AE MSE: {row['ae_mse']:.6f}")
        print(f"  Top features: {', '.join([f'{f} ({e:.4f})' for f, e in top3_features])}")
        
        results.append({
            "rank": len(results) + 1,
            "ztf_id": row["ztf_id"],
            "label": row["label"],
            "ae_mse": float(row["ae_mse"]),
            "top3_features": top3_features,
        })
    
    out_path = os.path.join(output_dir, "anomalies_ae.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    df[["ztf_id", "ae_mse"]].to_csv(
        os.path.join(output_dir, "scores_ae.csv"), index=False
    )
    
    print(f"\nSaved top {top_n} to {out_path}")
    
    return df[["ztf_id", "ae_mse"]]


if __name__ == "__main__":
    detect_ae()