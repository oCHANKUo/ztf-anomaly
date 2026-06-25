"""
ZTF Anomaly Detection Pipeline
==============================

A complete system for detecting anomalous supernova light curves in ZTF data
using three machine learning models: Isolation Forest, One-Class SVM, and Autoencoder.

Usage:
    from ztf_anomaly import Pipeline
    pipeline = Pipeline(data_dir="./data", output_dir="./output")
    pipeline.run_all()
"""

__version__ = "1.0.0"