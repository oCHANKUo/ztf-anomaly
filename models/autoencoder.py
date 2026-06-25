"""Autoencoder anomaly detection."""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


class AnomalyAE(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def detect_ae(data_dir="./data", output_dir="./output"):
    """Run Autoencoder anomaly detection."""
    
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
    
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_full_tensor = torch.FloatTensor(X_scaled)
    
    input_dim = len(numeric_cols)
    model = AnomalyAE(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    best_val_loss = np.inf
    patience = 20
    patience_counter = 0
    
    print("Training Autoencoder...")
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, X_train_tensor)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, X_val_tensor).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}, Loss: {loss.item():.6f}, Val: {val_loss:.6f}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    
    model.load_state_dict(best_state)
    
    model.eval()
    with torch.no_grad():
        predictions = model(X_full_tensor)
        per_feature_error = torch.abs(X_full_tensor - predictions).numpy()
        
        feature_stds = np.std(X_scaled, axis=0) + 1e-6
        weights = 1.0 / feature_stds
        weights = weights / np.sum(weights)
        
        reconstruction_errors = np.sum(per_feature_error * weights, axis=1)
    
    df["anomaly_score"] = reconstruction_errors
    
    anomalies = df.sort_values("anomaly_score", ascending=False).head(15)
    
    print(f"--- Autoencoder ---")
    print(f"Dataset: {len(df)} objects")
    
    results = []
    for idx, row in anomalies.iterrows():
        x_idx = df.index.get_loc(idx)
        feature_errors = per_feature_error[x_idx]
        
        top3_idx = np.argsort(feature_errors)[-3:][::-1]
        top3_features = [(numeric_cols[i], float(feature_errors[i])) for i in top3_idx]
        
        print(f"ID: {row['ztf_id']} | Type: {row['label']}")
        print(f"  Reconstruction Error: {row['anomaly_score']:.6f}")
        
        results.append({
            "ztf_id": row["ztf_id"],
            "label": row["label"],
            "ae_score": float(row["anomaly_score"]),
            "top3_features": top3_features
        })
    
    out_path = os.path.join(output_dir, "top_anomalies_ae.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    detect_ae()