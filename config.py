"""Configuration parameters for the ZTF Anomaly Detection Pipeline."""

# Download parameters
DOWNLOAD_LIMIT = 1000
SLEEP_TIME = 0.05

# Quality cuts
QUALITY_CUTS = {
    "rb_threshold": 0.55,
    "magerr_max": 0.3,
    "magerr_min": 0.0,
    "mag_min": 12.0,
    "mag_max": 24.0,
    "min_detections_total": 8,
    "min_detections_per_band": 3,
    "position_tolerance": 0.01,
}

# STRICT mode cuts
STRICT_CUTS = {
    "min_total_points": 20,
    "min_per_band": 8,
    "max_gap": 100,
    "max_frac_imputed": 0.10,
    "min_pre_peak": 3,
    "min_post_peak": 3,
}

# LOOSE mode cuts ( for comparison )
LOOSE_CUTS = {
    "min_total_points": 10,
    "min_per_band": 3,
    "max_gap": 200,
    "max_frac_imputed": 0.50,
    "min_pre_peak": 0,
    "min_post_peak": 0,
}

# Imputation parameters (optional)
IMPUTATION = {
    "gap_threshold": 30,
    "max_imputed_per_object": 5,
    "length_scale": 20,
}

# Model parameters
MODELS = {
    "isolation_forest": {
        "contamination": 0.05,
        "n_estimators": 200,
        "random_state": 42,
    },
    "one_class_svm": {
        "kernel": "rbf",
        "nu": 0.05,
        "gamma": "scale",
    },
    "autoencoder": {
        "hidden_dims": [16, 8, 4],
        "lr": 0.005,
        "weight_decay": 1e-5,
        "epochs": 500,
        "patience": 20,
    },
}

# Feature selection
ROBUST_FEATURES = [
    "g_peak", "r_peak", "peak_color", "abs_mag_g",
    "total_span", "n_detections",
    "dq_total_points", "dq_span", "dq_sampling_density", "dq_log_n",
]

SHAPE_FEATURES = [
    "g_skew", "r_skew", "g_kurt", "r_kurt",
    "g_asymmetry", "r_asymmetry",
]