# Anomaly Detection in ICU Patient Monitoring

> A three part data science project demonstraing interpretable and real-time anomaly detection in ICU patient vitals, enhanced by adaptive alert prioritization. 
---

## Overview

This repository contains three end-to-end systems for detecting anomalies in ICU vital signs:

- A classic time-series model (ARIMA/ETS)

- A real-time streaming anomaly detector with Kafka + ML

- A dynamic alert prioritizer that adapts based on clinician feedback (simulated)

Originally created across two graduate-level courses at Harvard University, this project builds from traditional time-series models to real-time streaming systems that incorporate feedback-driven alert priorization. 

---

##  Project Modules

### 1. [`legacy_stats_model/`](./legacy_stats_model/)
- Uses ARIMA & Exponential Smoothing for anomaly detection in SysBP & Pulse
- Defines confidence-based thresholds
- Emphasizes explainability for clinicians
---

### 2. [`kafka_streaming_model/`](./kafka_streaming_model/)

- Streams patient vitals using a Kafka producer-consumer pipeline 
- Detects anomalies using Isolation Forest and One-Class SVM
- Includes SHAP for explainability and clustering for patient similarity insights

### 3. [`realtime_alert_system/`](./realtime_alert_system/)
- Recieves anomaly alerts from Kafka
- Ranks alerts based on severity(SysBP, Pulse, Emergency)
- Incorporates simulated clinician feedback to adapt alert importance over time
- Near real-time performance(~1 second latency per record)

![Real-Time Kafka Alerts](./realtime_alert_system/kafka_alerts.png)
<sub> The Kafka consumer console below shows the alert system actively processing patient vitals:
- Records are streamed one by one from the producer.
- Anomalies are flagged based on:
   - Abnormal Pulse
   - Out-of-range SysBP
   - Emergency admissions
- Alerts are assigned a severity score based on the number of triggered conditions.
- The system then prioritizes alerts dynamically based on severity and feedback bias.
- This validates that the pipeline is functioning as intended — even with mostly normal data, the mixed dataset (mixed_focus.csv) helps surface and test real-time alert logic effectively. </sub>



---

## Dataset

- **Source**: [Kaggle - Predict Mortality of ICU Patients](https://www.kaggle.com/datasets/msafi04/predict-mortality-of-icu-patients-physionet)
- **Size**: 12,000 ICU time series segments from 4,000 unique patients
- **Format**: 48-hour windows of vital signs, lab results, and static features per patient stay
- **Key Features**:
  - `SysBP`: Systolic Blood Pressure
  - `Pulse`: Heart rate
  - `Survive`: Survival outcome
  - `Infection`: Infection presence (e.g., MRSA/sepsis)
  - `Emergency`: Binary indicator for emergency admission

**Note**: Data was preprocessed to filter valid vitals (SysBP, Pulse), select relevant features, and standardize time windows to simulate real-time streaming.

---

## Use Case

These models simulate components of clinical decision support tool, enabling: 
- Early detection of life-threatening instability
- Real-time alerting and triage
- Feedback- informed alert ranking for ICU providers
- Rapid prototyping of healthcare ML systems 

---
## Installation 
Clone the repo and install dependencies 
```bash
git clone https://github.com/roshijay/Anomaly-Detection-in-ICU.git
cd Anomaly-Detection-in-ICU
pip install -r requirements.txt
```
---
# How to Run 
- 1. Run the legacy Staistical Model( ARIMA/ETS)
     ```bash
     cd legacy_stats_model
     jupyter notebook legacy_model.ipynb  # or run legacy_model.py
     ```

- 2. Run the Real-time streaming + Alert Priorization Pipeline 
     Terminal 1: Kafka producer(Patient records)
     ```bash 
     cd realtime_alert_system/kafka_bridge 
     python producer.py
     ```

     Terminal 2: Kafka consumer(Anomaly Detection + Alert Ranking) 
     ```bash 
     cd realtime_alert_system/kafka_bridge 
     python consumer.py
     ```

---
# Requirements 
- This project is split into two main components, each with its own set of core dependencies:

  1. Legacy Statistical Model (ARIMA/ETS)
     *Interpretable time-series forecasting using classic statistical methods.*
     Libraries:
     - pandas, numpy: Data manipulation
     - matplotlib, seaborn: Visualization
     - statsmodel-ARIMA, Exponential Smoothing modeling
     - jupyter: Running notebooks interactively
       
  2. Real-Time Kafka Model + Recommender
     *Anomaly detection + alert prioritization pipeline with streaming support.*
     Libraries:
     - pandas, numpy, scikit-learn – Data handling & ML
     - kafka-python – Kafka producer/consumer
     - shap – Model explainability
     - time, json, os – Python standard libs for IO and formatting

---
# Project Structure 
```
Anomaly-Detection-in-ICU/
│
├── legacy_stats_model/                  # ARIMA/ETS time-series model
│   ├── legacy_model.ipynb
│   └── README.md
│
├── kafka_streaming_model/               # Kafka stream + unsupervised ML
│   ├── producer/
│   ├── consumer/
│   └── README.md
│
├── realtime_alert_system/               # Real-time alert pipeline (modular)
│   ├── kafka_bridge/                    # Kafka producer and consumer scripts
│   ├── streamlit_ui/                    # (Optional) Streamlit feedback interface
│   ├── alert_prioritizer.py             # Rule-based alert scoring and ranking
│   ├── recommender.py                   # Feedback-aware bias adjustment
│   ├── feedback_loop.py                 # Placeholder for clinician feedback tracking
│   ├── integrator.py                    # (Planned) Orchestrates alert + feedback
│   └── __pycache__/                     # Compiled bytecode (auto-generated)
│
├── data/                                # Processed dataset (Kaggle ICU subset)
│   └── processed_kaggle_icu.csv
│
├── requirements.txt                     # Dependencies for all components
└── README.md    
```
---


  
