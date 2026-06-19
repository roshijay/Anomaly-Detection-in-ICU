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

app = FastAPI(title="ICU Sepsis Risk API", version="0.1.0")


@app.get("/health")
def health_check():
    """
    Simple endpoint to confirm the API is running.
    Standard practice - load balancers and monitoring tools
    ping this to check if the service is alive.
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(vitals: PatientVitals):
    """
    Accept patient vitals and return a placeholder response.
    Model loading and real prediction logic comes next.
    """
    return {
        "received": vitals.dict(),
        "message": "Model not loaded yet - this is a placeholder"
    }