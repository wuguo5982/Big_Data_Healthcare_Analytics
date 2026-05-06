
"""
Document and claims analytics helpers for the Streamlit dashboard.

These utilities are lightweight and deterministic so the app can run locally
without requiring GPU or external services.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json
import pandas as pd

from src.config import CLAIMS_PATH, CLINICAL_NOTES_PATH, POLICY_PATH, AUDIT_RULES_PATH


def load_claims() -> pd.DataFrame:
    if CLAIMS_PATH.exists():
        return pd.read_csv(CLAIMS_PATH)

    # Demo fallback data when the CSV is missing.
    return pd.DataFrame(
        [
            {
                "claim_id": "C0001",
                "provider_id": "P101",
                "procedure_code": "99213",
                "diagnosis_code": "E11.9",
                "amount_billed": 850.0,
                "amount_paid": 420.0,
                "risk_label": "medium",
                "risk_score": 0.62,
            },
            {
                "claim_id": "C0002",
                "provider_id": "P208",
                "procedure_code": "97110",
                "diagnosis_code": "M54.5",
                "amount_billed": 2300.0,
                "amount_paid": 900.0,
                "risk_label": "high",
                "risk_score": 0.86,
            },
            {
                "claim_id": "C0003",
                "provider_id": "P101",
                "procedure_code": "93000",
                "diagnosis_code": "R07.9",
                "amount_billed": 450.0,
                "amount_paid": 210.0,
                "risk_label": "low",
                "risk_score": 0.24,
            },
        ]
    )


def load_clinical_notes() -> pd.DataFrame:
    if CLINICAL_NOTES_PATH.exists():
        return pd.read_csv(CLINICAL_NOTES_PATH)

    return pd.DataFrame(
        [
            {"note_id": "N001", "claim_id": "C0001", "note_text": "Patient follow-up visit with diabetes management documentation."},
            {"note_id": "N002", "claim_id": "C0002", "note_text": "Physical therapy documentation appears limited for high billed amount."},
            {"note_id": "N003", "claim_id": "C0003", "note_text": "Chest pain evaluation with ECG performed and documented."},
        ]
    )


def load_policy_text() -> str:
    if POLICY_PATH.exists():
        return POLICY_PATH.read_text(errors="ignore")
    return (
        "CMS FWA policy demo text: abnormal billing patterns, duplicate claims, "
        "excessive utilization, diagnosis-procedure inconsistency, and provider outlier "
        "behavior may require manual audit review."
    )


def load_audit_rules() -> Any:
    if AUDIT_RULES_PATH.exists():
        try:
            return json.loads(AUDIT_RULES_PATH.read_text(errors="ignore"))
        except Exception:
            return AUDIT_RULES_PATH.read_text(errors="ignore")
    return [
        {"rule_id": "R001", "name": "High billed amount outlier", "risk": "high"},
        {"rule_id": "R002", "name": "Diagnosis-procedure mismatch", "risk": "high"},
        {"rule_id": "R003", "name": "Duplicate claim pattern", "risk": "medium"},
    ]


def summarize_claims(df: pd.DataFrame) -> Dict[str, Any]:
    summary = {
        "num_claims": int(len(df)),
        "num_providers": int(df["provider_id"].nunique()) if "provider_id" in df else 0,
        "total_billed": float(df["amount_billed"].sum()) if "amount_billed" in df else 0.0,
        "avg_billed": float(df["amount_billed"].mean()) if "amount_billed" in df and len(df) else 0.0,
    }

    if "risk_label" in df:
        summary["risk_distribution"] = df["risk_label"].value_counts().to_dict()
    elif "risk_score" in df:
        summary["high_risk_count"] = int((df["risk_score"] >= 0.75).sum())

    return summary


def provider_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    if not {"provider_id", "amount_billed"}.issubset(df.columns):
        return pd.DataFrame()

    agg = df.groupby("provider_id").agg(
        claims_count=("provider_id", "count"),
        total_billed=("amount_billed", "sum"),
        avg_billed=("amount_billed", "mean"),
    ).reset_index()

    if "risk_score" in df:
        risk = df.groupby("provider_id")["risk_score"].mean().reset_index(name="avg_risk_score")
        agg = agg.merge(risk, on="provider_id", how="left")

    return agg.sort_values(["avg_risk_score" if "avg_risk_score" in agg else "total_billed"], ascending=False)
