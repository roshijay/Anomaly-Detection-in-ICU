# Anomaly Detection in Critical Care: A Data-Driven Approach to ICU Patient Monitoring

**Project Type**: Time-Series Anomaly Detection for Proactive ICU Monitoring and Early Warning Systems

---

## Overview

This project focuses on building an interpretable, data-driven framework to enhance early-warning systems in Intensive Care Units (ICUs). By applying advanced time-series modeling techniques, we detect critical anomalies in patient vital signs—specifically systolic blood pressure (SysBP) and pulse rate. The primary goal is to flag early signs of patient deterioration, such as those preceding sepsis or cardiac failure, thereby supporting timely clinical interventions and improving patient outcomes in critical care environments.

---

## Objectives

* **Early Anomaly Identification**: Precisely identify deviations in SysBP and pulse rate that serve as precursors to life-threatening complications like sepsis or cardiac failure.
* **Transparent Model Development**: Develop a highly interpretable alternative to traditional "black-box" machine learning models using statistical methods like ARIMA and Exponential Smoothing, ensuring clinical explainability.
* **Clinical Reasoning Integration**: Systematically apply clinical reasoning to statistical modeling through rigorous ACF/PACF analysis and the robust application of confidence intervals for anomaly threshold definition.
* **Enhanced Decision Support**: Provide early, actionable alerts to support ICU decision-making and facilitate effective remote patient monitoring.

---

## Dataset

* **Source**: [Kaggle - ICU Patient Dataset](https://www.kaggle.com/datasets/ukveteran/icu-patients?resource=download)
* **Size**: 200 records, 9 variables
* **Key Variables**:
    * `SysBP`: Systolic blood pressure (primary target for anomaly detection)
    * `Pulse`: Heart rate (secondary target for anomaly detection)
    * `Survive`: Survival status (0 = deceased, 1 = survived)
    * `Infection`: Binary infection flag (e.g., MRSA, sepsis)
    * `Emergency`: Emergency admission status
* **Note**: This dataset is fully cleaned and pre-processed, with no missing values, making it ideal for immediate exploratory modeling and time-series analysis.

---

## Methodology

### 1. Exploratory Data Analysis (EDA)

* Visualized key vital signs (SysBP, Pulse) segmented by patient outcomes (survived vs. deceased) to understand baseline patterns and deviations.
* Identified non-linear and non-stationary patterns within the time-series data, informing subsequent modeling choices.

### 2. Data Preparation

* Performed necessary normalization on SysBP and pulse rates to ensure consistent scaling for time-series models.
* Cleaned and removed irrelevant fields (e.g., unnamed index columns) to streamline the dataset.

### 3. Time-Series Modeling

* Implemented **ARIMA (AutoRegressive Integrated Moving Average)** and **Exponential Smoothing (ETS)** models, chosen for their interpretability and effectiveness in capturing time-dependent patterns.
* Conducted thorough **ACF (Autocorrelation Function)** and **PACF (Partial Autocorrelation Function)** analysis to accurately determine model orders and parameters.
* Incorporated **dynamic confidence intervals** around model predictions to statistically define robust anomaly thresholds.

### 4. Anomaly Detection

* Flagged individual data points that fell outside the established prediction intervals, indicating potential anomalies.
* Cross-referenced detected anomalies with survival outcomes and infection status to validate their clinical significance.

---

## Key Findings & Impact

* **Effective Model Performance**: ARIMA and ETS models successfully captured serial dependencies and underlying patterns in both SysBP and Pulse rate time-series data.
* **Clinically Relevant Anomaly Correlation**: Statistically defined anomalies were strongly correlated with negative patient outcomes. For instance, critically low SysBP readings identified as anomalies were predominantly observed in deceased patient cohorts, highlighting the model's ability to identify high-risk indicators.
* **Pattern Analysis**: STL (Seasonal-Trend decomposition using Loess) decomposition confirmed that the vital sign data was predominantly noise-dominant with minimal inherent seasonality, which guided the choice of appropriate time-series models.
* **Potential for Early Intervention**: The identified anomalies provide a foundation for developing early warning systems that could significantly reduce the time to intervention in critical care settings, potentially improving patient survival rates and reducing long-term complications.

---

## Future Work
* Multivariate Anomaly Detection: Extend the current framework to include additional vital signs such as SpO2 (blood oxygen saturation) and temperature for a more holistic view of patient status.
* Real-time Integration: Develop and integrate anomaly alerts into a dynamic, real-time dashboard using tools like Streamlit, enabling continuous patient monitoring for healthcare professionals.
* Model Comparison: Conduct a comparative study between traditional statistical methods (like those used here) and advanced deep learning techniques for time-series anomaly detection.

---

## Tools Used
* Python: Core programming language
* Pandas, NumPy: For efficient data manipulation and numerical operations.
* Statsmodels: Comprehensive library for statistical modeling, including ARIMA, Exponential Smoothing, and time-series diagnostics.
* Matplotlib, Seaborn: For creating high-quality static and statistical visualizations.
* Jupyter Notebook: For interactive development, experimentation, and analysis presentation.

--- 
