"""Visualization of anomaly light curves."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import pickle
import os


def plot_anomaly_lightcurves(data_dir="./data", output_dir="./output",
                              json_path="top_anomalies_if.json",
                              output_file="anomaly_lightcurves.png",
                              top_n=10):
    """Plot top anomaly light curves in a scientific format."""
    
    json_full = os.path.join(output_dir, json_path)
    pkl_path = os.path.join(data_dir, "ztf_lcs_all.pkl")
    
    with open(json_full, "r") as f:
        anomalies = json.load(f)
    
    with open(pkl_path, "rb") as f:
        all_lcs = pickle.load(f)
    
    anomalies = anomalies[:top_n]
    n = len(anomalies)
    
    if n == 0:
        print("No anomalies found.")
        return
    
    ncols = 2
    nrows = (n + 1) // 2
    
    fig = plt.figure(figsize=(14, 3.5 * nrows))
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.4, wspace=0.25)
    
    colors = {'g': '#1a9850', 'r': '#d73027'}
    band_names = {1: 'g', 2: 'r'}
    
    for i, anom in enumerate(anomalies):
        ztf_id = anom["ztf_id"]
        label = anom.get("label", "Unknown")
        score = anom.get("raw_score", anom.get("ae_score", anom.get("ocsvm_score", 0)))
        primary = anom.get("primary_feature", "N/A")
        
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        
        if ztf_id not in all_lcs:
            ax.text(0.5, 0.5, f"{ztf_id}\n(no data)", 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, style='italic', color='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        df = all_lcs[ztf_id].copy()
        t0 = df["mjd"].min()
        df["days"] = df["mjd"] - t0
        
        for fid, band_name in band_names.items():
            band_data = df[df["fid"] == fid].sort_values("days")
            if len(band_data) == 0:
                continue
            
            mag_col = "magpsf" if "magpsf" in band_data.columns else "magpsf_corr"
            x = band_data["days"].values
            y = band_data[mag_col].values
            yerr = band_data["magerr"].values if "magerr" in band_data.columns else None
            
            ax.errorbar(x, y, yerr=yerr,
                       fmt='o', color=colors[band_name], 
                       markersize=3.5, capsize=1.5, elinewidth=0.8,
                       alpha=0.85, label=f'{band_name}')
            ax.plot(x, y, '-', color=colors[band_name], alpha=0.3, linewidth=0.8)
        
        ax.invert_yaxis()
        ax.set_xlabel("Days since first detection", fontsize=8)
        ax.set_ylabel("Magnitude", fontsize=8)
        
        title = f"{ztf_id}  |  {label}"
        ax.set_title(title, fontsize=9, fontweight='bold', pad=2)
        ax.text(0.02, 0.98, f"Score: {score:.3f} | {primary}", 
               transform=ax.transAxes, fontsize=7, va='top', ha='left',
               color='#444444', style='italic')
        
        ax.legend(loc='lower right', fontsize=6.5, framealpha=0.9, 
                 edgecolor='gray', fancybox=False)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
        ax.tick_params(labelsize=7)
        ax.set_facecolor('#fff5f5')
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#cc0000')
    
    fig.suptitle("Top Anomalies — ZTF Supernova Light Curves", 
                fontsize=13, fontweight='bold', y=0.99)
    
    out_path = os.path.join(output_dir, output_file)
    plt.savefig(out_path, dpi=250, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    plot_anomaly_lightcurves()