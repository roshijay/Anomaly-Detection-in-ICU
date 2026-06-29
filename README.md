# ICU Sepsis Early Warning System

> A production-grade clinical decision support system for early sepsis detection, built on 1.55 million ICU patient-hours from the PhysioNet 2019 Challenge dataset.

[![CI](https://github.com/roshijay/Anomaly-Detection-in-ICU/actions/workflows/ci.yml/badge.svg)](https://github.com/roshijay/Anomaly-Detection-in-ICU/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![XGBoost AUC](https://img.shields.io/badge/XGBoost%20AUC-0.81-green)

---

## Project Story

This repository documents a three-phase progression from exploratory research to a production-ready ML system:

| Phase | Notebook | Description |
|-------|----------|-------------|
| Phase 1 (Dec 2024) | `notebooks/exploratory_analysis.ipynb` | Statistical EDA on ICU cohort — IQR analysis, chi-square, ANOVA, ARIMA time-series modeling |
| Phase 2 (May 2025) | `notebooks/waveform_prototype.ipynb` | MIMIC-III waveform anomaly detection — Kafka streaming, Isolation Forest, One-Class SVM, SHAP, Streamlit dashboard |
| Phase 3 (Present) | Production system (this repo) | PhysioNet 2019, 1.55M patient-hours, XGBoost + FastAPI + Docker + CI/CD |

---

## What This System Does

The system accepts real-time ICU patient vitals and lab values via a REST API and returns a calibrated sepsis risk score with a clinical flag. It is designed as a **decision support tool** — surfacing high-risk patients for clinician review, not replacing clinical judgment.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"HR": 110, "Temp": 38.9, "SBP": 88, "Lactate": 4.2, "WBC": 18.5}'

# Response:
# {"sepsis_risk_score": 0.1869, "flagged": false, "threshold_used": 0.2635}
```

---

## Dataset

- **Source:** PhysioNet Computing in Cardiology Challenge 2019
- **Size:** 1,552,210 patient-hours across 40,336 ICU patients
- **Label:** SepsisLabel — defined using Sepsis-3 criteria (SOFA score increase ≥ 2)
- **Sepsis rate:** 7.27% of patients, 1.8% of patient-hours
- **Variables:** 40 clinical features — 7 vitals, 26 labs, 7 demographics/administrative

---

## Model Performance

| Model | AUC-ROC | Notes |
|-------|---------|-------|
| XGBoost | 0.8133 | Patient-level split, no time artifacts |
| Isolation Forest | 0.6290 | Unsupervised baseline |

**Split strategy:** Patient-level train/test split — entire patients assigned to either train or test, preventing data leakage from the same patient appearing in both sets.

**Threshold:** 0.2635 — selected to achieve recall ≥ 0.85, prioritizing sensitivity over specificity given the asymmetric cost of missed sepsis cases.

**SHAP top features:** `Lactate_hours_since_measured`, `Bilirubin_total_last_known`, `WBC_last_known`, `Creatinine_last_known` — all aligned with Sepsis-3 SOFA score components, validating that the model learns genuine physiological signal.

---

## Feature Engineering

**Vitals (7 features × 5 statistics = 35 features):**
6-hour rolling mean, std, min, max, and trend for HR, O2Sat, Temp, SBP, MAP, DBP, Resp.

**Labs (7 labs × 3 features = 21 features):**
Forward-filled last known value, binary "was measured" flag, and hours since last measurement — MNAR-aware design that treats lab ordering as a clinical signal.

**Total: 85 engineered features**

---

## Production Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| Model training | XGBoost + Scikit-learn | Patient-level split, class weighting |
| Experiment tracking | MLflow | Parameters, metrics, model registry |
| REST API | FastAPI + Pydantic | Auto-validated endpoints, /docs UI |
| Testing | Pytest + httpx | 4 tests covering structure, validation, behavior |
| CI/CD | GitHub Actions | Auto-runs on every push to main |
| Containerization | Docker | One-command deployment |
| Explainability | SHAP TreeExplainer | Per-prediction feature attribution |

---

## Quick Start

### Run with Docker (recommended)
```bash
git clone https://github.com/roshijay/Anomaly-Detection-in-ICU.git
cd Anomaly-Detection-in-ICU
docker build -t icudetect .
docker run -p 8000:8000 icudetect
```

### Run locally
```bash
git clone https://github.com/roshijay/Anomaly-Detection-in-ICU.git
cd Anomaly-Detection-in-ICU
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd src && uvicorn api:app --reload
```

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

### Run tests
```bash
python -m pytest tests/test_api.py -v
```

---

## Project Structure
icudetect/

├── src/

│   ├── api.py                    # FastAPI endpoints

│   ├── train_model.py            # Model training with MLflow tracking

│   └── feature_engineering.py   # Rolling window + lab recency features

├── notebooks/

│   └── exploratory_analysis.ipynb  # EDA: missingness, distributions, SHAP

├── tests/

│   ├── test_api.py               # Pytest suite (4 tests)

│   ├── test_clinical_behavior.py # Clinical behavioral tests (local only)

│   └── fixtures/                 # Synthetic data + model for CI

├── models/                       # Saved model artifacts (gitignored)

├── data/                         # Dataset files (gitignored)

├── .github/workflows/ci.yml      # GitHub Actions CI pipeline

├── Dockerfile                    # Container definition

├── MODEL_CARD.md                 # Intended use, metrics, limitations, ethics

└── requirements.txt              # Locked dependencies

---

## Ethical Considerations

This system is a **clinical decision support tool**, not a diagnostic system. All predictions must be reviewed by a qualified clinician before any clinical action is taken. See [MODEL_CARD.md](./MODEL_CARD.md) for full details on intended use, limitations, fairness considerations, and known gaps.

---

## Author

**Roshini Jayasankar**
Harvard Data Science | June 2026
[GitHub](https://github.com/roshijay)
