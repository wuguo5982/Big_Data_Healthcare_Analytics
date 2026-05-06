"""FastAPI service for Agentic FWA Detection.

Run from the project root:
    uvicorn app.api:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Keep imports stable if this module is launched from a different working dir.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic_fwa_workflow import AgenticFWAWorkflow
from src.config import ANOMALY_MODEL_PATH, RAG_INDEX_PATH, RISK_MODEL_PATH
from src.rag_engine import FWARAGEngine


class ClaimRequest(BaseModel):
    claim_id: str
    provider_id: str
    patient_id: str
    procedure_code: str
    diagnosis_code: str
    claim_amount: float
    patient_age: int
    num_prior_claims: int
    days_since_last_claim: int
    is_high_risk_provider: int


app = FastAPI(
    title="Agentic Healthcare FWA Detection API",
    description="Agentic RAG + ML + anomaly detection + MLOps-ready API for healthcare FWA detection.",
    version="1.1",
)


def artifacts_ready() -> bool:
    """Check whether required local inference artifacts exist."""
    return all(path.exists() for path in [RISK_MODEL_PATH, ANOMALY_MODEL_PATH, RAG_INDEX_PATH])


@lru_cache(maxsize=1)
def get_agent() -> AgenticFWAWorkflow:
    """
    Load the agent once and reuse it across API requests.

    Annotation:
    In production, these artifacts would usually be loaded from S3, SageMaker
    Model Registry, MLflow Model Registry, or an internal model artifact store.
    """
    if not artifacts_ready():
        raise RuntimeError("Model/RAG artifacts are missing. Run `python -m src.train` first.")
    rag = FWARAGEngine.load()
    return AgenticFWAWorkflow(rag)


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if artifacts_ready() else "missing_artifacts",
        "service": "agentic-fwa-detection",
        "required_artifacts": {
            "risk_model": str(RISK_MODEL_PATH),
            "anomaly_model": str(ANOMALY_MODEL_PATH),
            "rag_index": str(RAG_INDEX_PATH),
        },
    }


@app.post("/analyze-claim")
def analyze_claim(claim: ClaimRequest):
    """
    Main inference endpoint.

    Annotation:
    A real production API should include authentication, request logging, rate
    limits, PHI/PII controls, audit trails, and role-based access control.
    """
    try:
        agent = get_agent()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return agent.run(claim.model_dump())
