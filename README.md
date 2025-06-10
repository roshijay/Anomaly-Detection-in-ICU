##Anomaly Detection in Critical Care

**Project Type**: Time-Series Anomaly Detection for ICU Monitoring

---

## Overview
This project applies time-series modeling techniques to detect anomalies in ICU patient vital signs — specifically systolic blood pressure (SysBP) and pulse rate. The goal is to improve early-warning systems in critical care environments by building an interpretable, data-driven framework that flags early signs of patient deterioration.

---

## Objectives
- Identify anomalies in SysBP and pulse that precede life-threatening complications such as sepsis or cardiac failure.
- Build a transparent alternative to black-box ML models using ARIMA and Exponential Smoothing.
- Apply clinical reasoning to statistical modeling using ACF/PACF analysis and confidence intervals.
- Support ICU decision-making and remote patient monitoring through early alerts.

---

## Dataset
- **Source**: [Kaggle - ICU Patient Dataset](https://www.kaggle.com/datasets/ukveteran/icu-patients?resource=download)
- **Size**: 200 records, 9 variables
- **Key Variables**:
  - `SysBP`: Systolic blood pressure (target for anomaly detection)
  - `Pulse`: Heart rate
  - `Survive`: Survival status (0 = deceased, 1 = survived)
  - `Infection`: Binary infection flag (e.g., MRSA, sepsis)
  - `Emergency`: Emergency admission status
- **Note**: Dataset is fully clean with no missing values — ideal for exploratory modeling

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Visualized vital signs by outcome (survived vs. deceased)
- Identified non-linear and non-stationary patterns

### 2. Data Preparation
- Normalized SysBP and pulse rates
- Removed irrelevant fields (e.g., unnamed index)

### 3. Time-Series Modeling
- Applied **ARIMA** and **Exponential Smoothing (ETS)**
- Conducted **ACF/PACF** analysis for model tuning
- Incorporated **confidence intervals** to define anomaly thresholds

### 4. Anomaly Detection
- Flagged data points falling outside prediction intervals
- Compared anomalies across survival outcomes and infection status

---

## Key Findings
- ARIMA and ETS models effectively modeled serial dependencies in SysBP and Pulse.
- Statistically defined anomalies correlated with negative outcomes (e.g., low SysBP in deceased patients).
- STL decomposition confirmed noise-dominant behavior with minimal seasonality.

---

## Future Work 
- Extend anomaly detection to multivariate vital signs(SpO2, temperature)
- Integrate alerts into a real-time dashboard using Streamlit
- Compare traditional staistical methods with deep learning
  
---

## Tools Used
- **Python**: Core language
- **Pandas, NumPy**: Data handling
- **Statsmodels**: ARIMA, smoothing, diagnostics
- **Matplotlib, Seaborn**: Visualization
- **Jupyter Notebook**: Interactive development

---

## How to Run
- Open the Jupyter notebook and run cells top to bottom.  
- No custom packages required beyond standard Python libraries.

  
