
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tools.insurance_prediction_tool import predict_insurance_cost

app = FastAPI(
    title="Diabetes Insurance Cost Prediction API",
    version="1.0.0",
    description="Educational ML API. Not for diagnosis or final insurance decisions.",
)


class PatientInput(BaseModel):
    age: int = Field(..., ge=18, le=100)
    sex: Literal["male", "female"]
    bmi: float = Field(..., ge=10, le=70)
    smoker: Literal["yes", "no"]
    children: int = Field(..., ge=0, le=20)
    region: Literal["northeast", "northwest", "southeast", "southwest"]
    diabetes: Literal["yes", "no"]
    hypertension: Literal["yes", "no"]
    hba1c: float = Field(..., ge=3.5, le=15)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Diabetes insurance prediction API is running."}


@app.post("/predict")
def predict(payload: PatientInput):
    try:
        return predict_insurance_cost(**payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
