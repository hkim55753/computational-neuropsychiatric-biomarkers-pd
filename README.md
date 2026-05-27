# Computational Neuropsychiatric Biomarkers in Parkinson’s Disease
### Computational Analysis of Parkinson’s Disease Motor Biomarkers Using Wearable Sensor Data

## Overview

This repository contains code for the computational analysis of motor function in Parkinson’s disease (PD) using smartwatch-based wearable sensor data from the PhysioNet Parkinson’s Disease Smartwatch Dataset.

The objective of this project is to develop interpretable computational biomarkers that distinguish Parkinson’s disease from healthy controls using task-based motion data.

The analysis focuses on:
- Signal preprocessing of wrist-worn accelerometer and gyroscope data
- Extraction of time-series motor features
- Statistical comparison between clinical groups
- Machine learning classification of Parkinson’s disease
- Interpretation of task-specific motor signatures

---

## Dataset

This project uses the publicly available PhysioNet Parkinson’s Disease Smartwatch Dataset:

https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/

The dataset includes:
- Multi-task wrist sensor recordings
- Accelerometer and gyroscope signals
- Healthy and Parkinson’s disease participants

Local dataset path used in this repository:


---

## Methods

### 1. Data Preprocessing
Raw smartwatch signals are processed to:
- Segment recordings by motor task
- Align multi-channel sensor streams
- Remove low-frequency trends using L1 trend filtering

### 2. Feature Extraction
For each task and sensor channel, the following features are computed:
- Signal variance (motor variability)
- Spectral entropy (signal complexity)
- Jerk (movement smoothness)

### 3. Statistical Analysis
Group-level comparisons between Parkinson’s disease and healthy controls are performed using:
- Non-parametric statistical tests
- Effect size estimation (Cohen’s d)

### 4. Machine Learning
A logistic regression model is used to classify Parkinson’s disease versus healthy controls using task-aware feature aggregation.

Model performance is evaluated using cross-validated ROC-AUC.

---

## Results

- The task-aware classification model achieves an average ROC-AUC of approximately 0.75.
- Parkinson’s disease participants show increased motor variability and reduced signal entropy compared to healthy controls.
- Task-specific differences are observed, indicating that motor impairment is not uniform across activities.
- Certain motor tasks (e.g., object handling and postural control tasks) contribute more strongly to classification performance.

---

## Reproducibility

To reproduce the main results:

bash
python src/plotting/figure3_motor_phenotype_nature_noseaborn.py
python src/plotting/figure5_task_aware_roc.py
python src/plotting/figure6_task_feature_importance.py

