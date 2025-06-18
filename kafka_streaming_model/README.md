# Real-Time ICU Anomaly Detection with Kafka and Unsupervised Machine Learning

**Project Type**: Real-Time Anomaly Detection System for Critical Care Monitoring with Apache Kafka + ML

---

## Overview

This project extends earlier ICU anomaly detection research by developing a **real-time, streaming pipeline** that uses physiological waveform signals and unsupervised machine learning to identify anomalies in critically ill patients.

Using **Apache Kafka** for simulation, this system processes real-time ICU data streams, applies models like **Isolation Forest** and **One-Class SVM**, and uses **SHAP** for interpretability. Patient similarity is explored via clustering, offering a path toward **dynamic triage support**.

Developed for Harvard’s graduate course in Data Mining, the project demonstrates how scalable ML systems can support timely, interpretable clinical interventions.

---

## Objectives

- Simulate real-time ICU monitoring via **Kafka-based streaming**
- Detect anomalies using **unsupervised ML models**
- Provide **explainable insights** via SHAP values
- Cluster patients by risk patterns using **K-Means** and **Agglomerative Clustering**
- Advance scalable decision-support systems for healthcare

---

## Dataset

* **Source**: MIMIC-III Waveform Database Matched Subset (via PhysioNet)
* **Format**: High-frequency, multi-channel waveform time-series
* **Key Channels**:
  - `PLETH`: Photoplethysmogram — measures blood volume changes
  - `RESP`: Respiratory waveform — monitors breathing patterns
  - `ABP`: Arterial Blood Pressure waveform — tracks cardiovascular status

> These waveforms are sampled at high resolution, making them ideal for real-time ingestion and anomaly detection simulations.

---

## Methodology

### 1. Kafka Streaming Simulation
- Converted waveform data into JSON-format chunks
- Used Kafka producers to emit records and consumers to process batches
- Enabled near-real-time anomaly scoring with low latency

### 2. Unsupervised Anomaly Detection
- Applied **Isolation Forest** and **One-Class SVM** to live batches
- Scored each time window to flag outliers in waveform dynamics

### 3. Explainability with SHAP
- Used SHAP values to explain contributions of input signals
- Identified which waveform shifts (e.g., PLETH spikes, ABP drops) triggered model alerts

### 4. Patient Similarity via Clustering
- Generated anomaly vectors and SHAP profiles for each patient
- Applied **K-Means** and **Agglomerative Clustering** to group similar trajectories
- Explored correlation between clusters and ICU outcomes

---

## Key Findings

- **Isolation Forest outperformed One-Class SVM** in both recall and speed for streaming data
- **PLETH and ABP waveforms were most predictive** of anomalies based on SHAP analysis
- Clustering revealed **distinct subgroups** with shared risk signals
- Demonstrated potential to move from static monitoring to **interpretable, continuous ICU analytics**

---

## Future Work

- Build a live dashboard with **Streamlit or Dash** for real-time monitoring and alert visualization
- Expand to include additional ICU parameters (SpO2, EtCO₂, ECG)
- Compare SHAP results across models to validate stability of feature importance
- Explore deployment on edge/embedded systems for bedside analytics

---

## Tools Used

- **Python**
- **Apache Kafka** — real-time streaming infrastructure
- **Scikit-learn** — ML algorithms (IF, OCSVM), clustering
- **SHAP** — explainability layer
- **Pandas, NumPy** — data transformation and batching
- **Matplotlib, Seaborn** — visual insights
- **Jupyter Notebook** — development environment
