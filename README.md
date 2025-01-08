# Anomaly-Detection-in-ICU
Leveraging statistical time-series analysis to identify anomalies in ICU patient vital signs. This project aims to enhance early warning systems for critical care monitoring through data-driven insights. 
## Table of Contents
- [Introduction](#introduction)
- [Project Goals](#project-goals)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
---
## Introduction
Early detection of anomalies in ICU vital signs can significantly improve patient outcomes. This project applies statistical methods to analyze time-series data from ICU patients and identify patterns that signal potential health risks.

---
## Project Goals
- Detect anomalies in ICU patient vital signs, focusing on systolic blood pressure (SysBP).
- Enhance early warning systems for critical care monitoring.
---
## Dataset
- **Source:** ICU patient vital signs from Kaggle.
- **Variables:** SysBP, heart rate, oxygen saturation, etc.
- **Data Description:**
  - Raw data includes timestamps, vital sign measurements, and patient identifiers.
  - Processed data involves cleaning, normalization, and feature extraction.

---

## Methodology
1. **Data Preprocessing:**
   - Handling missing values using interpolation.
   - Normalizing vital sign measurements.
2. **Statistical Analysis:**
   - Stationarity tests (e.g., ADF Test).
   - Time-series decomposition (e.g., STL).
3. **Anomaly Detection:**
   - Applying ARIMA to forecast expected values.
   - Identifying anomalies based on confidence intervals.
4. **Visualization:**
   - Time-series plots with anomaly markers.

---

## Key Findings
- Systolic blood pressure (SysBP) data was stationary with no significant autocorrelation.
- Statistical thresholds effectively identified anomalies.
- STL decomposition highlighted potential patterns but revealed no significant seasonality.
---

  
