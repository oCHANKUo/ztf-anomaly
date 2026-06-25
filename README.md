# ZTF Supernova Anomaly Detection System
## Overview

This system identifies unusual supernovae in data from the Zwicky Transient Facility (ZTF).
The three methods employed are:

- **Isolation Forest** — A tree-based algorithm that identifies outliers by measuring how quickly an object can be separated from the rest of the dataset.
- **One-Class Support Vector Machine** — A boundary-based method that learns the region containing "normal" supernovae and flags anything falling outside it.
- **Autoencoder** — A neural network trained to reconstruct typical supernova light curves; objects that are difficult to reconstruct are flagged as unusual.

By combining the results from all three methods, the system produces a ranked list of the most promising candidates for follow-up observation.

---

## Requirements

### Software

- Python 3.8 or later
- pip (the standard Python package installer, usually included with Python)

### Required Packages

Install all dependencies with the following command, entered into your terminal or command prompt:

```bash
pip install alerce pandas numpy scipy scikit-learn torch matplotlib
```

---

## Installation and Setup

### Step 1: Obtain the Code

file strcuture:

```
ztf_anomaly/
├── __init__.py
├── config.py
├── pipeline.py
├── data/
│   ├── __init__.py
│   ├── download.py
│   └── quality.py
├── features/
│   ├── __init__.py
│   └── extract.py
├── models/
│   ├── __init__.py
│   ├── isolation_forest.py
│   ├── one_class_svm.py
│   ├── autoencoder.py
│   └── consensus.py
├── visualization/
│   ├── __init__.py
│   └── plot.py
└── evaluation/
    ├── __init__.py
    └── classify.py
```

### Step 2: Navigate to the Working Directory

Use the `cd` command to move to the folder that **contains** the `ztf_anomaly/` directory. For example:

```bash
cd /Users/yourname/Documents/research
```
Replace the path above with the actual location on your computer.

### Step 3: Run the Complete Pipeline

Execute the following single command:

```bash
python -c "from ztf_anomaly.pipeline import Pipeline; p = Pipeline(); p.run_all()"
```

This command will perform the following actions in sequence:

1. **Download** approximately 1,000 labeled supernovae from the ZTF Bright Transient Survey catalog.
2. **Inspect** the downloaded data for quality issues and remove unreliable measurements.
3. **Extract** numerical features from each light curve, such as peak brightness, color, and duration.
4. **Analyze** the data using all three machine learning models.
5. **Combine** the individual model outputs into a single consensus ranking.
6. **Generate** plots of the most anomalous light curves.
7. **Evaluate** each candidate and classify it as likely physical, uncertain, or likely a data artifact.

### Step 5: Review the Results

After the pipeline completes, the following files will be available in the `output/` folder:

| File | Contents |
|------|----------|
| `top_anomalies_if.json` | Ranked list from the Isolation Forest |
| `top_anomalies_ocsvm.json` | Ranked list from the One-Class SVM |
| `top_anomalies_ae.json` | Ranked list from the Autoencoder |
| `consensus_anomalies.csv` | Combined ranking with vote counts and consensus scores |
| `anomaly_lightcurves.png` | Light curve plots of the top candidates |
| `anomaly_classification.txt` | Evaluation report classifying each candidate |

Open `anomaly_lightcurves.png` to inspect the light curves visually. Open `consensus_anomalies.csv` to examine the numerical rankings. Open `anomaly_classification.txt` to see which candidates are considered reliable and which may be affected by data quality issues.

---

## Running the Pipeline in Stages

If the full pipeline is interrupted or if you wish to run only certain steps, each stage can be executed independently. The commands below correspond to the seven stages described above.

```bash
# Stage 1: Data acquisition
python -c "from ztf_anomaly.pipeline import Pipeline; p = Pipeline(); p.step1_download(1000)"

# Stage 2: Quality assurance
python -c "from ztf_anomaly.pipeline import Pipeline; p = Pipeline(); p.step2_quality()"

# Stage 3: Feature extraction
python -c "from ztf_anomaly.pipeline import Pipeline; p = Pipeline(); p.step3_features(strict_mode=True)"

# Stage 4: Model execution
python -c "from ztf_anomaly.pipeline import Pipeline; p = Pipeline(); p.step4_models()"

# Stage 5: Consensus building
python -c "from ztf_anomaly.pipeline import Pipeline; p = Pipeline(); p.step5_consensus()"

# Stage 6: Visualization
python -c "from ztf_anomaly.pipeline import Pipeline; p = Pipeline(); p.step6_visualize(top_n=10)"

# Stage 7: Evaluation
python -c "from ztf_anomaly.pipeline import Pipeline; p = Pipeline(); p.step7_evaluate()"
```
---

## Understanding the Output

### Consensus Ranking (`consensus_anomalies.csv`)

The most important columns are:

| Column | Description |
|--------|-------------|
| `ztf_id` | The unique identifier assigned by the ZTF survey |
| `vote_count` | Number of models (out of 3) that flagged this object. Values range from 0 to 3. Higher values indicate stronger agreement. |
| `consensus_score` | A normalized measure of anomaly strength, ranging from 0 (highly anomalous) to 1 (typical). Lower values are more unusual. |
| `if_norm`, `ae_norm`, `svm_norm` | Individual model rankings, each from 0 (most anomalous) to 1 (least anomalous). |

**High-confidence candidates** are those with `vote_count >= 2` and `consensus_score < 0.3`. These objects were flagged by at least two independent algorithms and are ranked among the most extreme in the dataset.

### Classification Report (`anomaly_classification.txt`)

Each candidate is assigned one of three categories:

| Category | Meaning | Recommended Action |
|----------|---------|-------------------|
| **REAL ASTROPHYSICAL** | The object exhibits genuinely unusual physical properties. | Prioritize for follow-up observation. |
| **BORDERLINE** | The object may be physically unusual, but data quality issues cannot be ruled out. | Examine the light curve visually before deciding. |
| **DATA ARTIFACT** | The anomaly is likely caused by sparse sampling, large gaps, or other measurement issues rather than astrophysics. | Exclude from scientific analysis. |

### Light Curve Plots (`anomaly_lightcurves.png`)

Each panel displays one candidate anomaly:

- **Green circles** — measurements in the g-band (green/blue optical filter)
- **Red circles** — measurements in the r-band (red optical filter)
- **Vertical bars** — photometric uncertainty (error bars)
- **Light red background** — indicates the object was flagged as anomalous

---

## Troubleshooting

### Error: `No valid samples produced`

**Cause:** The quality criteria are too restrictive for the downloaded data.

**Solution:** Open `ztf_anomaly/config.py` and adjust the strictness parameters. For example, lowering `min_total_points` from 20 to 15 or increasing `max_gap` from 100 to 150 will allow more objects to pass.

### Issue: Plots appear empty or malformed

**Cause:** The model outputs may be missing or empty, often because an earlier stage failed.

**Solution:** Verify that Stage 4 (model execution) completed without errors. Check that the files `top_anomalies_if.json`, `top_anomalies_ocsvm.json`, and `top_anomalies_ae.json` exist in the `output/` folder.

---

## Configuration

System behavior can be adjusted by editing the file `ztf_anomaly/config.py`. The most commonly modified parameters are listed below.

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `DOWNLOAD_LIMIT` | Number of supernovae to retrieve from the archive | `1000` |
| `STRICT_CUTS["min_total_points"]` | Minimum total detections required per object | `20` |
| `STRICT_CUTS["min_per_band"]` | Minimum detections required in each optical band | `8` |
| `STRICT_CUTS["max_gap"]` | Largest allowable gap between observations, in days | `100` |
| `STRICT_CUTS["max_frac_imputed"]` | Maximum fraction of synthetic (interpolated) data points | `0.10` |
| `MODELS["isolation_forest"]["contamination"]` | Expected proportion of anomalies in the dataset | `0.05` |
| `MODELS["one_class_svm"]["nu"]` | Upper bound on the fraction of training errors | `0.05` |

---

## Scientific Background

This system is designed for comparative evaluation of unsupervised anomaly detection methods in time-domain astronomy. Unlike supervised classification, which requires labeled examples of every class, unsupervised anomaly detection can identify rare or novel phenomena that have never been seen before. This makes it particularly valuable for surveys like ZTF, where the most scientifically interesting objects are often the least common.

The pipeline incorporates design principles from several published studies:

- **LAISS** (Aleo et al. 2024): Emphasizes strict quality control and feature-based detection. The authors achieve high purity by aggressively filtering data artifacts before analysis.
- **SNAD** (Pruzhinskaya et al. 2021): Demonstrates the value of active learning and human-in-the-loop vetting. Their study found that a significant fraction of automatically flagged outliers were instrumental artifacts rather than physical sources.
- **ZTF AD** (2025): Employs multi-modal deep learning, combining photometric features, image cutouts, and light curve sequences. Their results show that different data representations capture different kinds of anomalies.

This system focuses on the feature-based approach for interpretability and computational efficiency, while acknowledging that image-based and sequence-based methods may achieve higher accuracy for specific anomaly types.

---

## Citation

If this system is used in published research, please cite the following foundational works:

- Aleo, P. D., et al. (2024). "LAISS: A Generalizable and Scalable Machine Learning Pipeline for Anomaly Detection in the Zwicky Transient Facility." *The Astrophysical Journal*, 969, 2, 74.
- Pruzhinskaya, M. V., et al. (2021). "SNAD: Anomaly Detection in the Fourth Data Release of the Zwicky Transient Facility." *Astronomy & Astrophysics*, 648, A118.

---

## License

This software is released under the MIT License. You are free to use, modify, and distribute it for any purpose, provided that appropriate credit is given to the original authors.
