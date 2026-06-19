import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional


class PatientVitals(BaseModel):
    """
    Defines the exact shape of a valid prediction request.
    FastAPI uses this to automatically validate incoming data
    and reject malformed requests before our code even runs.
    """
    HR: Optional[float] = Field(None, description="Heart rate")
    O2Sat: Optional[float] = Field(None, description="Oxygen saturation %")
    Temp: Optional[float] = Field(None, description="Temperature in Celsius")
    SBP: Optional[float] = Field(None, description="Systolic blood pressure")
    MAP: Optional[float] = Field(None, description="Mean arterial pressure")
    DBP: Optional[float] = Field(None, description="Diastolic blood pressure")
    Resp: Optional[float] = Field(None, description="Respiratory rate")
    Age: Optional[float] = Field(None, description="Patient age")
    Gender: Optional[float] = Field(None, description="0 = female, 1 = male")
    
    # Optional lab values - if not provided, treated as not-yet-measured
    Lactate: Optional[float] = Field(None, description="Lactate (mmol/L)")
    Creatinine: Optional[float] = Field(None, description="Creatinine (mg/dL)")
    WBC: Optional[float] = Field(None, description="White blood cell count")
    BUN: Optional[float] = Field(None, description="Blood urea nitrogen")
    Platelets: Optional[float] = Field(None, description="Platelet count")
    Bilirubin_total: Optional[float] = Field(None, description="Total bilirubin")
    Glucose: Optional[float] = Field(None, description="Glucose (mg/dL)")

# Load the trained model once, at startup - not on every request
xgb_model = joblib.load('../models/xgb_model.joblib')

# The exact feature columns the model was trained on, in order
MODEL_FEATURES = xgb_model.get_booster().feature_names

app = FastAPI(title="ICU Sepsis Risk API", version="0.1.0")


@app.get("/health")
def health_check():
    """
    Simple endpoint to confirm the API is running.
    Standard practice - load balancers and monitoring tools
    ping this to check if the service is alive.
    """
    return {"status": "ok"}

def build_feature_row(vitals: PatientVitals) -> pd.DataFrame:
    """
    Convert a single sparse API request into the full feature row
    the model expects. Since a single API call has no patient history,
    rolling/trend features default to the raw value (mean) or 0 (std, trend) -
    effectively treating this reading as the patient's only known data point.
    This is a documented simplification: real deployment would maintain
    a rolling buffer of recent readings per patient.
    """
    vitals_dict = vitals.dict()
    row = {}

    for feature in MODEL_FEATURES:
        if feature in vitals_dict:
            # Direct match - raw vital or demographic
            row[feature] = vitals_dict[feature]
        elif feature.endswith('_rolling_mean'):
            base = feature.replace('_rolling_mean', '')
            row[feature] = vitals_dict.get(base, 0)
        elif feature.endswith('_rolling_std'):
            row[feature] = 0  # no variance with a single reading
        elif feature.endswith('_rolling_min') or feature.endswith('_rolling_max'):
            base = feature.split('_rolling_')[0]
            row[feature] = vitals_dict.get(base, 0)
        elif feature.endswith('_trend'):
            row[feature] = 0  # no trend with a single reading
        elif feature.endswith('_was_measured'):
            base = feature.replace('_was_measured', '')
            lab_value = vitals_dict.get(base)
            row[feature] = 1 if lab_value is not None else 0
        elif feature.endswith('_last_known'):
            base = feature.replace('_last_known', '')
            lab_value = vitals_dict.get(base)
            row[feature] = lab_value if lab_value is not None else 0
        elif feature.endswith('_hours_since_measured'):
            base = feature.replace('_hours_since_measured', '')
            lab_value = vitals_dict.get(base)
            # If a value was provided, treat it as just measured (0 hours ago).
            # If not provided, treat as never measured (large value).
            row[feature] = 0 if lab_value is not None else 999
        else:
            row[feature] = 0  # fallback default

    df_row = pd.DataFrame([row])
    # Ensure exact column order matches training
    df_row = df_row[MODEL_FEATURES]
    return df_row

@app.post("/predict")
def predict(vitals: PatientVitals):
    """
    Accept patient vitals, build the full feature row, and return
    a sepsis risk score from the trained XGBoost model.
    """
    feature_row = build_feature_row(vitals)

    # predict_proba returns [P(no sepsis), P(sepsis)] - we want index 1
    risk_score = float(xgb_model.predict_proba(feature_row)[0][1])

    # Using our clinically-tuned threshold from training (recall >= 0.85)
    THRESHOLD = 0.2635
    flagged = risk_score >= THRESHOLD

    return {
        "sepsis_risk_score": round(risk_score, 4),
        "flagged": flagged,
        "threshold_used": THRESHOLD
    }
