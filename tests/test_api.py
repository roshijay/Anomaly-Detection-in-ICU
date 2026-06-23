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
        "HR": 85, "O2Sat": 97, "Temp": 37.0,
        "SBP": 120, "MAP": 80, "DBP": 70,
        "Resp": 16, "Age": 55, "Gender": 0
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
    payload = {"HR": "not a number", "Age": 65}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_works_with_partial_input():
    """
    All fields are optional - the API should accept a request
    with only some vitals provided and still return a valid response.
    """
    payload = {"HR": 95, "Age": 60}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0.0 <= response.json()["sepsis_risk_score"] <= 1.0
