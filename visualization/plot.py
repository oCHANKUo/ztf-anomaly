"""Visualization for ZTF anomaly detection pipeline."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import json
import os
from pathlib import Path


# Ensure Chinese fonts work (pre-configured in your environment)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_data(data_dir="./data", output_dir="./output"):
    """Load all necessary data files."""
    data = {}
    
    # Features
    feat_path = os.path.join(data_dir, "ztf_features_strict.csv")
    if os.path.exists(feat_path):
        data["features"] = pd.read_csv(feat_path)
    
    # Light curves
    lcs_path = os.path.join(data_dir, "ztf_lcs_all.pkl")
    if os.path.exists(lcs_path):
        with open(lcs_path, "rb") as f:
            data["lightcurves"] = pickle.load(f)
    
    # Ensemble results
    ens_path = os.path.join(output_dir, "ensemble_results.csv")
    if os.path.exists(ens_path):
        data["ensemble"] = pd.read_csv(ens_path)
    
    # Individual model scores
    for model in ["if", "ae", "ocsvm"]:
        score_path = os.path.join(output_dir, f"scores_{model}.csv")
        if os.path.exists(score_path):
            data[f"scores_{model}"] = pd.read_csv(score_path)
    
    # Quality report
    qual_path = os.path.join(data_dir, "quality_report.csv")
    if os.path.exists(qual_path):
        data["quality"] = pd.read_csv(qual_path)
    
    print(f"Loaded: {list(data.keys())}")
    return data


# =============================================================================
# 1. LIGHT CURVE PLOTS
# =============================================================================

def plot_lightcurve(ztf_id, lc_dict, features_df=None, save_dir=None):
    """Plot a single light curve with bands and metadata."""
    if ztf_id not in lc_dict:
        print(f"Light curve not found: {ztf_id}")
        return None
    
    df = lc_dict[ztf_id].copy().sort_values("mjd")
    mag_col = "magpsf" if "magpsf" in df.columns else "magpsf_corr"

    if "magerr" not in df.columns:
        df["magerr"] = 0.1  # default error
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = {1: '#2ca02c', 2: '#d62728'}  # g=green, r=red
    labels = {1: 'g', 2: 'r'}
    
    for fid in [1, 2]:
        band = df[df["fid"] == fid]
        if len(band) == 0:
            continue
        ax.errorbar(
            band["mjd"], band[mag_col], yerr=band["magerr"],
            fmt='o', color=colors[fid], alpha=0.7, markersize=4,
            label=f'{labels[fid]}-band ({len(band)} pts)'
        )
    
    ax.invert_yaxis()
    ax.set_xlabel("MJD")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"ZTF {ztf_id}")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add feature annotation if available
    if features_df is not None:
        row = features_df[features_df["ztf_id"] == ztf_id]
        if len(row) > 0:
            info = f"N={int(row['dq_total_points'].values[0])}, " \
                   f"Span={row['dq_span'].values[0]:.1f}d, " \
                   f"MaxGap={row['dq_max_gap'].values[0]:.1f}d"
            ax.text(0.02, 0.02, info, transform=ax.transAxes,
                   fontsize=8, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(save_dir, f"lc_{ztf_id}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path
    return fig


def plot_top_lightcurves(data, top_n=20, save_dir=None):
    """Plot light curves for top ensemble anomalies."""
    if save_dir:
        save_dir = ensure_dir(save_dir)
    
    if "ensemble" not in data or "lightcurves" not in data:
        print("Missing ensemble or lightcurve data")
        return
    
    ens = data["ensemble"].nsmallest(top_n, "final_rank")
    features = data.get("features")
    
    saved = []
    for _, row in ens.iterrows():
        ztf_id = row["ztf_id"]
        path = plot_lightcurve(ztf_id, data["lightcurves"], features, save_dir)
        if path:
            saved.append(path)
    
    print(f"Saved {len(saved)} light curve plots")
    return saved


def plot_lightcurve_grid(data, ztf_ids, ncols=3, title=None, save_path=None):
    """Plot multiple light curves in a grid."""
    n = len(ztf_ids)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    colors = {1: '#2ca02c', 2: '#d62728'}
    
    for idx, ztf_id in enumerate(ztf_ids):
        ax = axes[idx]
        if ztf_id not in data.get("lightcurves", {}):
            ax.set_visible(False)
            continue
        
        df = data["lightcurves"][ztf_id].sort_values("mjd")
        mag_col = "magpsf" if "magpsf" in df.columns else "magpsf_corr"

        # FIX: Add magerr if missing
        if "magerr" not in df.columns:
            df["magerr"] = 0.1
        
        for fid in [1, 2]:
            band = df[df["fid"] == fid]
            if len(band) > 0:
                ax.errorbar(band["mjd"], band[mag_col], yerr=band["magerr"],
                          fmt='o', color=colors[fid], alpha=0.6, markersize=3)
        
        ax.invert_yaxis()
        ax.set_title(ztf_id, fontsize=9)
        ax.tick_params(labelsize=7)
    
    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return fig


# =============================================================================
# 2. FEATURE DISTRIBUTION PLOTS
# =============================================================================

def plot_feature_histograms(data, features=None, n_cols=4, save_dir=None):
    """Plot histograms of key features, split by label if available."""
    if "features" not in data:
        print("No features data")
        return
    
    df = data["features"]
    
    # Select features to plot
    if features is None:
        # Pick interesting ones: global + some band features
        features = [
            'global_amplitude', 'global_std', 'global_skew', 'global_eta',
            'global_stetson_k', 'global_beyond2std', 'global_linear_chi2',
            'g_periodogram_amp', 'r_periodogram_amp', 'g_eta', 'r_eta',
            'color_gr', 'peak_color'
        ]
        # Filter to existing columns
        features = [f for f in features if f in df.columns]
    
    n = len(features)
    n_rows = (n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 2.5*n_rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for idx, feat in enumerate(features):
        ax = axes[idx]
        
        # Split by label if available
        if 'label' in df.columns and df['label'].notna().sum() > 10:
            # Top 5 labels
            top_labels = df['label'].value_counts().head(5).index
            for lab in top_labels:
                subset = df[df['label'] == lab][feat].dropna()
                if len(subset) > 5:
                    ax.hist(subset, bins=30, alpha=0.5, label=lab, density=True)
            ax.legend(fontsize=7)
        else:
            ax.hist(df[feat].dropna(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        
        ax.set_title(feat, fontsize=9)
        ax.tick_params(labelsize=7)
    
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(ensure_dir(save_dir), "feature_histograms.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path
    plt.show()
    return fig


def plot_bogus_filter_space(data, save_dir=None):
    """Recreate ZTF DR3 Figure 18: Periodogram Amplitude vs Reduced Chi2."""
    if "features" not in data:
        print("No features data")
        return
    
    df = data["features"]
    
    req = ['global_linear_chi2', 'g_periodogram_amp', 'r_periodogram_amp']
    if not all(r in df.columns for r in req):
        print(f"Missing features for bogus filter plot. Need: {req}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Use g-band periodogram amplitude (or average both)
    x = df['global_linear_chi2'].values
    y = ((df['g_periodogram_amp'].fillna(0) + df['r_periodogram_amp'].fillna(0)) / 2).values
    
    # Scatter all points
    mask_finite = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
    ax.scatter(x[mask_finite], y[mask_finite], c='lightgray', alpha=0.3, s=5, label='All objects')
    
    # Mark likely bogus region
    chi2_thresh = np.percentile(x[mask_finite], 85)
    amp_thresh = np.percentile(y[mask_finite], 25)
    
    # Highlight bogus candidates
    bogus_mask = (x > chi2_thresh) & (y < amp_thresh) & mask_finite
    ax.scatter(x[bogus_mask], y[bogus_mask], c='red', alpha=0.7, s=15, label='Likely bogus')
    
    # Threshold lines
    ax.axvline(chi2_thresh, color='red', linestyle='--', alpha=0.5, label=f'χ² 85th %ile = {chi2_thresh:.1f}')
    ax.axhline(amp_thresh, color='red', linestyle='--', alpha=0.5, label=f'Amp 25th %ile = {amp_thresh:.1f}')
    
    ax.set_xlabel("Global Linear Fit χ²", fontsize=10)
    ax.set_ylabel("Mean Periodogram Amplitude", fontsize=10)
    ax.set_title("Bogus Filter Space (ZTF DR3 Style)", fontsize=11)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(ensure_dir(save_dir), "bogus_filter_space.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"Saved bogus filter plot to {path}")
        return path
    plt.show()
    return fig


# =============================================================================
# 3. ANOMALY RANKING PLOTS
# =============================================================================

def plot_model_comparison(data, save_dir=None):
    """Scatter matrix comparing model anomaly scores."""
    if "ensemble" not in data:
        print("No ensemble data")
        return
    
    df = data["ensemble"]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # IF vs AE
    ax = axes[0, 0]
    ax.scatter(df["if_anomaly"], df["ae_anomaly"], c='steelblue', alpha=0.4, s=8)
    ax.set_xlabel("Isolation Forest")
    ax.set_ylabel("Autoencoder")
    ax.set_title("IF vs AE")
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.3)
    
    # IF vs OC-SVM
    ax = axes[0, 1]
    ax.scatter(df["if_anomaly"], df["ocsvm_anomaly"], c='steelblue', alpha=0.4, s=8)
    ax.set_xlabel("Isolation Forest")
    ax.set_ylabel("One-Class SVM")
    ax.set_title("IF vs OC-SVM")
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.3)
    
    # AE vs OC-SVM
    ax = axes[1, 0]
    ax.scatter(df["ae_anomaly"], df["ocsvm_anomaly"], c='steelblue', alpha=0.4, s=8)
    ax.set_xlabel("Autoencoder")
    ax.set_ylabel("One-Class SVM")
    ax.set_title("AE vs OC-SVM")
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.3)
    
    # Ensemble vs DQ risk
    ax = axes[1, 1]
    scatter = ax.scatter(df["ensemble_score"], df["dq_risk"], 
                        c=df["final_score"], cmap='RdYlGn_r', alpha=0.6, s=10)
    ax.set_xlabel("Ensemble Score (raw)")
    ax.set_ylabel("DQ Risk")
    ax.set_title("Ensemble vs DQ Risk (color = final score)")
    plt.colorbar(scatter, ax=ax, label='Final Score')
    
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(ensure_dir(save_dir), "model_comparison.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"Saved model comparison to {path}")
        return path
    plt.show()
    return fig


def plot_top_candidates_summary(data, top_n=30, save_dir=None):
    """Horizontal bar chart of top candidates with scores and DQ flags."""
    if "ensemble" not in data:
        print("No ensemble data")
        return
    
    df = data["ensemble"].nsmallest(top_n, "final_rank").copy()
    
    fig, ax = plt.subplots(figsize=(10, 0.4*top_n + 2))
    
    y_pos = np.arange(len(df))
    
    # Bar colors: red if bogus, orange if high DQ, green otherwise
    colors = []
    for _, row in df.iterrows():
        if row["likely_bogus"]:
            colors.append("#d62728")  # red
        elif row["dq_risk"] > 0.5:
            colors.append("#ff7f0e")  # orange
        else:
            colors.append("#2ca02c")  # green
    
    bars = ax.barh(y_pos, df["final_score"], color=colors, alpha=0.8, edgecolor='black')
    
    # Labels
    labels = [f"{row.ztf_id} | {row.label[:15] if pd.notna(row.label) else 'Unknown'}" 
              for _, row in df.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Final Anomaly Score (DQ-penalized)", fontsize=10)
    ax.set_title(f"Top {top_n} Anomaly Candidates", fontsize=12)
    
    # Add score annotations
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row.final_score + 0.01, i, 
               f"{row.final_score:.3f} (E:{row.ensemble_score:.3f}, DQ:{row.dq_risk:.2f})",
               va='center', fontsize=7)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ca02c', label='Clean'),
        Patch(facecolor='#ff7f0e', label='High DQ Risk'),
        Patch(facecolor='#d62728', label='Likely Bogus')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(ensure_dir(save_dir), "top_candidates.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"Saved top candidates to {path}")
        return path
    plt.show()
    return fig


# =============================================================================
# 4. DQ ANALYSIS PLOTS
# =============================================================================

def plot_dq_feature_correlation(data, save_dir=None):
    """Heatmap of how DQ features correlate with anomaly scores."""
    if "ensemble" not in data or "features" not in data:
        print("Missing data")
        return
    
    ens = data["ensemble"]
    feat = data["features"]
    
    dq_cols = [c for c in feat.columns if c.startswith("dq_")]
    if len(dq_cols) == 0:
        print("No DQ features found")
        return
    
    # Merge scores with DQ features
    df = ens[["ztf_id", "ensemble_score", "final_score"]].merge(
        feat[["ztf_id"] + dq_cols], on="ztf_id"
    )
    
    corr_cols = dq_cols + ["ensemble_score", "final_score"]
    corr = df[corr_cols].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(corr_cols)))
    ax.set_yticks(range(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(corr_cols, fontsize=8)
    
    # Add text annotations
    for i in range(len(corr_cols)):
        for j in range(len(corr_cols)):
            text = ax.text(j, i, f'{corr.values[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)
    
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title("DQ Features vs Anomaly Scores", fontsize=11)
    
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(ensure_dir(save_dir), "dq_correlation.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"Saved DQ correlation to {path}")
        return path
    plt.show()
    return fig


def plot_feature_importance(data, model="if", top_n=15, save_dir=None):
    """Plot which features drive anomalies for a given model."""
    # This requires the model object; load from saved file
    model_dir = os.path.join("./output", "models")
    model_paths = {
        "if": os.path.join(model_dir, "isolation_forest", "model.joblib"),
        "ae": os.path.join(model_dir, "autoencoder", "model.pth"),
        "ocsvm": os.path.join(model_dir, "one_class_svm", "model.joblib")
    }
    
    # For IF, we can use feature deviations from median
    if "ensemble" not in data or "features" not in data:
        print("Missing data")
        return
    
    ens = data["ensemble"].nsmallest(top_n, "final_rank")
    feat = data["features"]
    
    # Get feature names (excluding meta and DQ)
    meta_cols = ["ztf_id", "label", "redshift"]
    dq_cols = [c for c in feat.columns if c.startswith("dq_")]
    ml_cols = [c for c in feat.columns if c not in meta_cols + dq_cols 
               and feat[c].dtype in [np.float64, np.float32, np.int64]]
    
    # For top anomalies, compute median absolute deviation from population
    pop_median = feat[ml_cols].median()
    
    deviations = []
    for _, row in ens.iterrows():
        ztf_id = row["ztf_id"]
        frow = feat[feat["ztf_id"] == ztf_id][ml_cols]
        if len(frow) == 0:
            continue
        dev = np.abs(frow.iloc[0] - pop_median) / (feat[ml_cols].std() + 1e-10)
        deviations.append(dev)
    
    if not deviations:
        print("No deviation data")
        return
    
    avg_dev = pd.Series(np.mean(deviations, axis=0)).sort_values(ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(avg_dev))
    ax.barh(y_pos, avg_dev.values, color='steelblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(avg_dev.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |z-score| from population median", fontsize=10)
    ax.set_title(f"Top Features Driving Anomalies (Top {top_n} candidates)", fontsize=11)
    
    plt.tight_layout()
    
    if save_dir:
        path = os.path.join(ensure_dir(save_dir), f"feature_importance_{model}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"Saved feature importance to {path}")
        return path
    plt.show()
    return fig


# =============================================================================
# 5. MASTER PLOT FUNCTION
# =============================================================================

def generate_all_plots(data_dir="./data", output_dir="./output", viz_dir="./visualization/output"):
    """Generate all standard plots for the pipeline."""
    
    viz_dir = ensure_dir(viz_dir)
    print(f"Output directory: {viz_dir}")
    
    data = load_data(data_dir, output_dir)
    
    plots_generated = []
    
    # 1. Feature distributions
    print("\n--- Feature Histograms ---")
    path = plot_feature_histograms(data, save_dir=viz_dir)
    if path: plots_generated.append(path)
    
    # 2. Bogus filter space
    print("\n--- Bogus Filter Space ---")
    path = plot_bogus_filter_space(data, save_dir=viz_dir)
    if path: plots_generated.append(path)
    
    # 3. Model comparison (only if ensemble exists)
    if "ensemble" in data:
        print("\n--- Model Comparison ---")
        path = plot_model_comparison(data, save_dir=viz_dir)
        if path: plots_generated.append(path)
        
        print("\n--- Top Candidates Summary ---")
        path = plot_top_candidates_summary(data, top_n=30, save_dir=viz_dir)
        if path: plots_generated.append(path)
        
        print("\n--- DQ Correlation ---")
        path = plot_dq_feature_correlation(data, save_dir=viz_dir)
        if path: plots_generated.append(path)
        
        print("\n--- Feature Importance ---")
        path = plot_feature_importance(data, model="if", top_n=30, save_dir=viz_dir)
        if path: plots_generated.append(path)
    
    # 4. Light curves of top candidates
    if "lightcurves" in data and "ensemble" in data:
        print("\n--- Top Light Curves ---")
        lc_dir = ensure_dir(os.path.join(viz_dir, "lightcurves"))
        paths = plot_top_lightcurves(data, top_n=20, save_dir=lc_dir)
        plots_generated.extend(paths or [])
        
        # Grid plot of top 9
        print("\n--- Light Curve Grid ---")
        top_ids = data["ensemble"].nsmallest(9, "final_rank")["ztf_id"].tolist()
        grid_path = os.path.join(viz_dir, "lc_grid_top9.png")
        plot_lightcurve_grid(data, top_ids, ncols=3, 
                            title="Top 9 Anomaly Candidates",
                            save_path=grid_path)
        plots_generated.append(grid_path)
    
    print(f"\n{'='*60}")
    print(f"Generated {len(plots_generated)} plots in {viz_dir}")
    print(f"{'='*60}")
    for p in plots_generated:
        print(f"  {os.path.basename(p)}")
    
    return plots_generated


# =============================================================================
# 6. QUICK PLOTS (for interactive use)
# =============================================================================

def quick_lc(ztf_id, data_dir="./data"):
    """Quickly plot a single light curve by ID."""
    data = load_data(data_dir)
    if "lightcurves" not in data:
        print("No light curves loaded")
        return
    fig = plot_lightcurve(ztf_id, data["lightcurves"], data.get("features"))
    plt.show()
    return fig


def quick_top(n=10, data_dir="./data", output_dir="./output"):
    """Quickly show top N anomalies."""
    data = load_data(data_dir, output_dir)
    if "ensemble" not in data:
        print("Run ensemble first")
        return
    top = data["ensemble"].nsmallest(n, "final_rank")
    print(top[["ztf_id", "label", "ensemble_score", "dq_risk", "likely_bogus", "final_score"]])
    return top


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ZTF Anomaly Visualization")
    parser.add_argument("--data-dir", default="./data", help="Data directory")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    parser.add_argument("--viz-dir", default="./visualization/output", help="Visualization output")
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument("--lc", type=str, help="Plot specific light curve ID")
    parser.add_argument("--top", type=int, help="Show top N anomalies")
    
    args = parser.parse_args()
    
    if args.lc:
        quick_lc(args.lc, args.data_dir)
    elif args.top:
        quick_top(args.top, args.data_dir, args.output_dir)
    else:
        generate_all_plots(args.data_dir, args.output_dir, args.viz_dir)