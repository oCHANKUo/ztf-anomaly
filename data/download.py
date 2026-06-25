"""Download ZTF BTS catalog and light curves."""

import pandas as pd
import pickle
import time
import os

try:
    from alerce.core import Alerce
    ALERCE_AVAILABLE = True
except ImportError:
    ALERCE_AVAILABLE = False


def run_download(limit=1000, output_dir="./data"):
    """Download ZTF BTS catalog and light curves."""
    
    if not ALERCE_AVAILABLE:
        print("Warning: alerce not installed. Install: pip install alerce")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    client = Alerce()
    
    print("Fetching BTS Catalog...")
    bts = pd.read_csv(
        "https://sites.astro.caltech.edu/ztf/bts/explorer.php?format=csv"
    )
    
    bts_labeled = bts[bts["type"] != "-"].copy()
    bts_path = os.path.join(output_dir, "bts_all_labeled.csv")
    bts_labeled.to_csv(bts_path, index=False)
    
    print(f"Total labeled objects: {len(bts_labeled)}")
    print("\nTop Classes:")
    print(bts_labeled["type"].value_counts().head(10))
    
    ztf_ids = bts_labeled["ZTFID"].dropna().unique()[:limit]
    all_lcs = {}
    
    rejection_stats = {
        "empty": 0, "rb": 0, "isdiffpos": 0, "magerr": 0,
        "magnitude": 0, "detections": 0, "position": 0
    }
    
    print(f"\nDownloading detections for {len(ztf_ids)} objects...")
    
    for i, ztf_id in enumerate(ztf_ids):
        try:
            df_det = client.query_detections(ztf_id, survey="ztf", format="pandas")
            
            if df_det is None or df_det.empty:
                rejection_stats["empty"] += 1
                continue
            
            if "rb" in df_det.columns:
                df_det = df_det[df_det["rb"] >= 0.55]
                if len(df_det) == 0:
                    rejection_stats["rb"] += 1
                    continue
            
            if "isdiffpos" in df_det.columns:
                positive = ["t", "1", True]
                df_det = df_det[df_det["isdiffpos"].isin(positive)]
                if len(df_det) == 0:
                    rejection_stats["isdiffpos"] += 1
                    continue
            
            if "magerr" in df_det.columns:
                df_det = df_det[(df_det["magerr"] > 0) & (df_det["magerr"] < 0.5)]
                if len(df_det) == 0:
                    rejection_stats["magerr"] += 1
                    continue
            
            mag_col = "magpsf" if "magpsf" in df_det.columns else "magpsf_corr"
            if mag_col in df_det.columns:
                df_det = df_det[(df_det[mag_col] > 12) & (df_det[mag_col] < 25)]
                if len(df_det) == 0:
                    rejection_stats["magnitude"] += 1
                    continue
            
            g_count = len(df_det[df_det["fid"] == 1])
            r_count = len(df_det[df_det["fid"] == 2])
            if len(df_det) < 8 or g_count < 3 or r_count < 3:
                rejection_stats["detections"] += 1
                continue
            
            if "ra" in df_det.columns and "dec" in df_det.columns:
                if df_det["ra"].std() > 0.01 or df_det["dec"].std() > 0.01:
                    rejection_stats["position"] += 1
                    continue
            
            all_lcs[ztf_id] = df_det
            
        except Exception as e:
            print(f"Failed {ztf_id}: {e}")
        
        if i % 10 == 0:
            print(f"Progress: {i}/{len(ztf_ids)} | Collected: {len(all_lcs)}")
        time.sleep(0.05)
    
    lcs_path = os.path.join(output_dir, "ztf_lcs_all.pkl")
    with open(lcs_path, "wb") as f:
        pickle.dump(all_lcs, f)
    
    print("\n========================")
    print("DOWNLOAD COMPLETE")
    print("========================")
    print(f"Accepted objects: {len(all_lcs)}")
    print("\nRejection Summary:")
    for key, value in rejection_stats.items():
        print(f"  {key:12s}: {value}")
    print(f"\nSaved to {lcs_path}")


if __name__ == "__main__":
    run_download(5000)