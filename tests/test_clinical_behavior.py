"""
Clinical behavior tests - run locally with the production model only.
These tests encode clinical expectations that depend on a properly
trained model and are NOT suitable for CI with a toy fixture model.

Run with:
    python -m pytest tests/test_clinical_behavior.py -v
"""
from api import app
import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

client = TestClient(app)


def test_labs_increase_risk_score():
    """
    Adding concerning lab values to an abnormal vitals profile
    should increase the risk score. Validates that lab features
    (identified as most important in SHAP analysis) are correctly
    handled by build_feature_row.
    """
    abnormal_vitals = {
        "HR": 110, "O2Sat": 92, "Temp": 38.9,
        "SBP": 88, "MAP": 60, "DBP": 50,
        "Resp": 26, "Age": 72, "Gender": 1
    }
    abnormal_vitals_with_labs = {
        **abnormal_vitals,
        "Lactate": 4.2, "Creatinine": 2.1, "WBC": 18.5
    }

    score_no_labs = client.post(
        "/predict", json=abnormal_vitals).json()["sepsis_risk_score"]
    score_with_labs = client.post(
        "/predict", json=abnormal_vitals_with_labs).json()["sepsis_risk_score"]

    assert score_with_labs > score_no_labs, (
        f"Expected labs to increase risk score: "
        f"no_labs={score_no_labs}, with_labs={score_with_labs}"
    )


def test_normal_patient_low_risk():
    """
    A patient with completely normal vitals should score below threshold.
    """
    normal_vitals = {
        "HR": 72, "O2Sat": 98, "Temp": 37.0,
        "SBP": 120, "MAP": 85, "DBP": 75,
        "Resp": 14, "Age": 45, "Gender": 0
    }
    response = client.post("/predict", json=normal_vitals)
    score = response.json()["sepsis_risk_score"]
    assert score < 0.5, f"Normal patient should have low risk, got {score}"
