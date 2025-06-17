# Anomaly Detection in ICU Patient Monitoring

> A two-part data science project that explores interpretable and real-time approaches to anomaly detection in ICU vital signs.

---

## Overview

This repository documents the development of two machine learning systems for ICU patient monitoring — one using **classic time-series forecasting** and the other built for **real-time anomaly detection** with Kafka and unsupervised ML models.

Created across two graduate-level courses at Harvard University, these projects highlight how healthcare ML systems can evolve from transparent statistical modeling to scalable streaming applications.

---

##  Project Modules

### 1. [`legacy_stats_model/`]
**Title**: _Interpretable Time-Series Modeling with ARIMA & ETS_

- Detects anomalies in systolic blood pressure (SysBP) and pulse
- Uses ARIMA and Exponential Smoothing to define confidence-based anomaly bounds
- Focuses on transparency and explainability for clinical decision-making


---

### 2. [`kafka_streaming_model/`]
**Title**: _Real-Time ICU Anomaly Detection with Kafka + Unsupervised ML_

- Simulates live data streams using Kafka
- Detects anomalies using Isolation Forest and One-Class SVM
- Uses SHAP for explainability and clustering for patient similarity insights

---

## Dataset

- **Source**: [Kaggle - ICU Patient Dataset](https://www.kaggle.com/datasets/ukveteran/icu-patients)
- **Size**: 200 rows × 9 features
- **Key Features**:
  - `SysBP`: Systolic Blood Pressure
  - `Pulse`: Heart rate
  - `Survive`: Survival outcome (binary)
  - `Infection`: Infection presence (e.g., MRSA/sepsis)
  - `Emergency`: Emergency admission flag

> Note: The dataset is fully clean and ideal for small-scale clinical prototyping.

---

## Use Case

These models could be deployed as clinical decision support tools to:
- Alert ICU staff to early signs of instability
- Enable remote patient monitoring with interpretable alerts
- Serve as educational tools for explainable healthcare AI

---



  
