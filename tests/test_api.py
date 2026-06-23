import sys
import os
import pytest
from fastapi.testclient import TestClient

# Add src/ to path so we can import the app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from api import app

client = TestClient(app)


def test_health_check():
    """
    Health endpoint should always return 200 with status ok.
    If this fails, the API isn't running correctly at all.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_structure():
    """
    A valid request should return 200 with a risk score between 0 and 1,
    a boolean flagged field, and the threshold used.
    """
    payload = {
        "HR": 85,
        "O2Sat": 97,
        "Temp": 37.0,
        "SBP": 120,
        "MAP": 80,
        "DBP": 70,
        "Resp": 16,
        "Age": 55,
        "Gender": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "sepsis_risk_score" in data
    assert "flagged" in data
    assert "threshold_used" in data
    assert 0.0 <= data["sepsis_risk_score"] <= 1.0
    assert isinstance(data["flagged"], bool)


def test_predict_rejects_invalid_input():
    """
    Sending a string where a float is expected should return 422.
    This confirms Pydantic validation is actually running.
    """
    payload = {
        "HR": "not a number",
        "Age": 65
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_labs_increase_risk_score():
    """
    Behavioral test: adding concerning lab values to an already
    abnormal vitals profile should increase the risk score.
    This encodes a clinical expectation directly into the test suite —
    if feature engineering or build_feature_row breaks lab handling,
    this test catches it automatically.
    """
    abnormal_vitals = {
        "HR": 110,
        "O2Sat": 92,
        "Temp": 38.9,
        "SBP": 88,
        "MAP": 60,
        "DBP": 50,
        "Resp": 26,
        "Age": 72,
        "Gender": 1
    }

    abnormal_vitals_with_labs = {
        **abnormal_vitals,
        "Lactate": 4.2,
        "Creatinine": 2.1,
        "WBC": 18.5
    }

    response_no_labs = client.post("/predict", json=abnormal_vitals)
    response_with_labs = client.post("/predict", json=abnormal_vitals_with_labs)

    assert response_no_labs.status_code == 200
    assert response_with_labs.status_code == 200

    score_no_labs = response_no_labs.json()["sepsis_risk_score"]
    score_with_labs = response_with_labs.json()["sepsis_risk_score"]

    assert score_with_labs > score_no_labs, (
        f"Expected labs to increase risk score: "
        f"no_labs={score_no_labs}, with_labs={score_with_labs}"
    )


def test_predict_works_with_partial_input():
    """
    All fields are optional - the API should accept a request
    with only some vitals provided and still return a valid response.
    """
    payload = {"HR": 95, "Age": 60}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0.0 <= response.json()["sepsis_risk_score"] <= 1.0