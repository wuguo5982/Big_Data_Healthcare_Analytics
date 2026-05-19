"""FastAPI service for the NIH ClinicalBERT multimodal AI demo."""

from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from utils import build_patient_summary


app = FastAPI(
    title="NIH ClinicalBERT Multimodal Clinical AI API",
    version="1.0",
)


class PatientRequest(BaseModel):
    """
    Request body for prediction endpoint.
    """

    patient_id: str


@app.get("/")
def health_check() -> dict:
    """
    Basic API health check.
    """

    return {
        "status": "ok",
        "message": "API is running.",
        "disclaimer": "Research demo only. Not for clinical use.",
    }


@app.get("/patient/{patient_id}")
def get_patient_summary(patient_id: str) -> dict:
    """
    Return explainable patient summary by patient ID.
    """

    try:
        patient_summary = build_patient_summary(patient_id)

        return patient_summary

    except Exception as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )


@app.post("/predict")
def predict_patient_risk(request: PatientRequest) -> dict:
    """
    Generate explainable patient prediction.

    Example request:
    {
        "patient_id": "P00001"
    }
    """

    try:
        prediction_result = build_patient_summary(
            request.patient_id
        )

        return prediction_result

    except Exception as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        )