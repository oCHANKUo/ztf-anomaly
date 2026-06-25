"""Main pipeline orchestrator for ZTF Anomaly Detection."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztf_anomaly.config import QUALITY_CUTS, STRICT_CUTS, MODELS
from ztf_anomaly.data.download import run_download
from ztf_anomaly.data.quality import run_quality_check
from ztf_anomaly.features.extract import run_features
from ztf_anomaly.models.isolation_forest import detect as if_detect
from ztf_anomaly.models.one_class_svm import detect_ocsvm
from ztf_anomaly.models.autoencoder import detect_ae
from ztf_anomaly.models.consensus import merge_results
from ztf_anomaly.visualization.plot import plot_anomaly_lightcurves
from ztf_anomaly.evaluation.classify import classify_anomalies


class Pipeline:
    """End-to-end ZTF anomaly detection pipeline."""
    
    def __init__(self, data_dir="./data", output_dir="./output"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    
    def step1_download(self, limit=1000):
        """Download ZTF BTS catalog and light curves."""
        print("\n" + "="*60)
        print("STEP 1: Download")
        print("="*60)
        run_download(limit=limit, output_dir=self.data_dir)
    
    def step2_quality(self):
        """Apply quality cuts and diagnostics."""
        print("\n" + "="*60)
        print("STEP 2: Quality Check")
        print("="*60)
        run_quality_check(data_dir=self.data_dir)
    
    def step3_features(self, strict_mode=True):
        """Extract features from light curves."""
        print("\n" + "="*60)
        print("STEP 3: Feature Extraction")
        print("="*60)
        run_features(data_dir=self.data_dir, strict_mode=strict_mode)
    
    def step4_models(self):
        """Run all 3 anomaly detection models."""
        print("\n" + "="*60)
        print("STEP 4: Anomaly Detection")
        print("="*60)
        
        print("\n--- Isolation Forest ---")
        if_detect(data_dir=self.data_dir, output_dir=self.output_dir)
        
        print("\n--- One-Class SVM ---")
        detect_ocsvm(data_dir=self.data_dir, output_dir=self.output_dir)
        
        print("\n--- Autoencoder ---")
        detect_ae(data_dir=self.data_dir, output_dir=self.output_dir)
    
    def step5_consensus(self):
        """Merge results from all 3 models."""
        print("\n" + "="*60)
        print("STEP 5: Consensus")
        print("="*60)
        merge_results(output_dir=self.output_dir)
    
    def step6_visualize(self, top_n=10):
        """Visualize top anomalies."""
        print("\n" + "="*60)
        print("STEP 6: Visualization")
        print("="*60)
        plot_anomaly_lightcurves(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            top_n=top_n
        )
    
    def step7_evaluate(self):
        """Classify and evaluate anomalies."""
        print("\n" + "="*60)
        print("STEP 7: Evaluation")
        print("="*60)
        classify_anomalies(
            data_dir=self.data_dir,
            output_dir=self.output_dir
        )
    
    def run_all(self, download_limit=1000, strict_mode=True, top_n=10):
        """Run complete pipeline."""
        print("\n" + "="*60)
        print("ZTF ANOMALY DETECTION PIPELINE")
        print("="*60)
        
        self.step1_download(limit=download_limit)
        self.step2_quality()
        self.step3_features(strict_mode=strict_mode)
        self.step4_models()
        self.step5_consensus()
        self.step6_visualize(top_n=top_n)
        self.step7_evaluate()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Results saved to: {self.output_dir}")


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run_all()